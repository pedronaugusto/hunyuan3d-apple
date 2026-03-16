"""
MLX model worker for Hunyuan3D API server.

Shape generation runs through MLX on Apple Silicon.
Texture generation uses the MLX thin-adapter pipeline layered on top of
upstream orchestration.
"""
import os
import time
import traceback
import uuid
import base64
import numpy as np
from io import BytesIO
from PIL import Image

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision compatibility fix before other imports
try:
    from hy3dpaint.utils.torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from hy3dpaint.convert_utils import create_glb_with_pbr_materials
from model_paths import resolve_hunyuan_paths


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
    }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def load_image_from_base64(image):
    if ',' in image:
        image = image.split(',', 1)[1]
    return Image.open(BytesIO(base64.b64decode(image)))


class MlxModelWorker:
    """MLX-accelerated worker for 3D model generation."""

    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache',
                 **kwargs):
        self.resolved_model_paths = resolve_hunyuan_paths(model_path)
        self.model_path = self.resolved_model_paths.root
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir

        logger.info(f"Loading MLX models on worker {self.worker_id} ...")

        # Background remover (CPU, lightweight)
        self.rembg = BackgroundRemover()

        # Shape pipeline — pure MLX
        from mlx_backend.shape_pipeline import MlxShapePipeline
        self.pipeline = MlxShapePipeline(
            weights_dir=self.model_path,
            lazy_load=True,
        )

        # Texture pipeline — MLX thin adapter over upstream orchestration.
        from mlx_backend.texture_pipeline import MlxTexturePipeline
        self.paint_pipeline = MlxTexturePipeline(
            model_dir=self.model_path,
        )

        # Clean cache
        os.makedirs(self.save_dir, exist_ok=True)
        for file in os.listdir(self.save_dir):
            filepath = os.path.join(self.save_dir, file)
            if os.path.isfile(filepath):
                os.remove(filepath)

        logger.info("MLX model worker ready.")

    def get_queue_length(self):
        if self.model_semaphore is None:
            return 0
        return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
               (len(self.model_semaphore._waiters) if hasattr(self.model_semaphore, '_waiters') and self.model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {"speed": 1, "queue_length": self.get_queue_length()}

    def get_diagnostics(self):
        return {
            "backend": "mlx",
            "model_root": self.resolved_model_paths.root,
            "shape_dir": self.resolved_model_paths.shape_dir,
            "texture_dir": self.resolved_model_paths.texture_dir,
        }

    def generate(self, uid, params):
        start_time = time.time()
        logger.info(f"[MLX] Generating 3D model for uid: {uid}")

        # Handle input image
        if 'image' in params:
            image = load_image_from_base64(params["image"])
        else:
            raise ValueError("No input image provided")

        image = image.convert("RGBA")
        if not self._has_alpha(image):
            # No meaningful alpha channel (all pixels >= 250) → remove background
            image = self.rembg(image)

        # Shape generation via MLX
        try:
            num_steps = params.get('num_inference_steps', 20)
            guidance = params.get('guidance_scale', 5.5)
            seed = params.get('seed', 42)
            resolution = params.get('octree_resolution', 384)
            num_chunks = params.get('num_chunks', 8000)
            max_shape_attempts = params.get('shape_retry_attempts', 3)
            last_error = None
            trace_dir = os.path.join(self.save_dir, f"{str(uid)}_shape_trace")
            prev_trace_dir = os.environ.get("HY3D_MLX_SHAPE_TRACE_DIR")
            os.environ["HY3D_MLX_SHAPE_TRACE_DIR"] = trace_dir

            try:
                for attempt in range(1, max_shape_attempts + 1):
                    try:
                        mesh = self.pipeline(
                            image=image,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance,
                            seed=seed,
                            octree_resolution=resolution,
                            num_chunks=num_chunks,
                        )[0]
                        break
                    except ValueError as e:
                        last_error = e
                        if "Surface level must be within volume data range" not in str(e) or attempt == max_shape_attempts:
                            raise
                        logger.warning(
                            "Shape sample %s/%s produced no surface; retrying",
                            attempt, max_shape_attempts,
                        )
                else:
                    raise last_error
            finally:
                if prev_trace_dir is None:
                    os.environ.pop("HY3D_MLX_SHAPE_TRACE_DIR", None)
                else:
                    os.environ["HY3D_MLX_SHAPE_TRACE_DIR"] = prev_trace_dir
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            logger.error(f"Shape generation failed: {e}")
            raise ValueError(f"Failed to generate 3D mesh: {str(e)}")

        # Export initial mesh
        initial_save_path = os.path.join(self.save_dir, f'{str(uid)}_initial.glb')
        mesh.export(initial_save_path)

        # Texture generation (MLX paint pipeline)
        if params.get('texture', False):
            try:
                output_mesh_path_obj = os.path.join(self.save_dir, f'{str(uid)}_texturing.obj')
                texture_trace_dir = os.path.join(self.save_dir, f"{str(uid)}_texture_trace")
                texture_internal_trace_dir = os.path.join(self.save_dir, f"{str(uid)}_texture_internal_trace")
                prev_trace_dir = os.environ.get("HY3D_TEXTURE_TRACE_DIR")
                prev_internal_trace_dir = os.environ.get("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR")
                os.environ["HY3D_TEXTURE_TRACE_DIR"] = texture_trace_dir
                os.environ["HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR"] = texture_internal_trace_dir
                try:
                    textured_path_obj = self.paint_pipeline(
                        mesh_path=initial_save_path,
                        image_path=image,
                        output_mesh_path=output_mesh_path_obj,
                        save_glb=False,
                        seed=params.get('seed', 42),
                        texture_steps=params.get('texture_steps'),
                        texture_guidance=params.get('texture_guidance'),
                    )
                finally:
                    if prev_trace_dir is None:
                        os.environ.pop("HY3D_TEXTURE_TRACE_DIR", None)
                    else:
                        os.environ["HY3D_TEXTURE_TRACE_DIR"] = prev_trace_dir
                    if prev_internal_trace_dir is None:
                        os.environ.pop("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR", None)
                    else:
                        os.environ["HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR"] = prev_internal_trace_dir
                logger.info("---Texture generation takes %s seconds ---" % (time.time() - start_time))

                glb_path_textured = os.path.join(self.save_dir, f'{str(uid)}_texturing.glb')
                quick_convert_with_obj2gltf(textured_path_obj, glb_path_textured)
                final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.glb')
                os.rename(glb_path_textured, final_save_path)

            except Exception as e:
                logger.error(f"Texture generation failed: {e}")
                traceback.print_exc()
                final_save_path = initial_save_path
                logger.warning(f"Using untextured mesh as fallback: {final_save_path}")
        else:
            final_save_path = initial_save_path

        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        return final_save_path, uid

    @staticmethod
    def _has_alpha(image):
        """Check if image has meaningful alpha channel."""
        if image.mode != 'RGBA':
            return False
        alpha = np.array(image.getchannel('A'))
        return alpha.min() < 250
