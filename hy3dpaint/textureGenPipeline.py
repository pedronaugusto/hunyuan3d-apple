# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import copy
import trimesh
import numpy as np
import json
import shutil
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.image_super_utils import imageSuperNet
from utils.uvwrap_utils import mesh_uv_wrap
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        # Force CPU for rasterizer — custom rasterizer already runs on CPU;
        # MPS tensor indexing has bugs with large texture buffers.
        render_device = "cpu"
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
            device=render_device,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def load_models(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        print("Models Loaded.")

    def _trace_dir(self):
        trace_dir = os.environ.get("HY3D_TEXTURE_TRACE_DIR")
        if not trace_dir:
            return None
        os.makedirs(trace_dir, exist_ok=True)
        return trace_dir

    def _trace_copy(self, src_path, name):
        trace_dir = self._trace_dir()
        if trace_dir is None or not src_path or not os.path.exists(src_path):
            return
        dst = os.path.join(trace_dir, name)
        if os.path.abspath(src_path) == os.path.abspath(dst):
            return
        shutil.copy2(src_path, dst)

    def _trace_image(self, name, image):
        trace_dir = self._trace_dir()
        if trace_dir is None or image is None:
            return
        if isinstance(image, Image.Image):
            image.save(os.path.join(trace_dir, name))
            return
        if hasattr(image, "detach") and hasattr(image, "cpu"):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(trace_dir, name))

    def _trace_json(self, name, payload):
        trace_dir = self._trace_dir()
        if trace_dir is None:
            return
        with open(os.path.join(trace_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _trace_mesh_state(self, name, mesh):
        trace_dir = self._trace_dir()
        if trace_dir is None or mesh is None:
            return
        payload = {
            "vertices": int(len(mesh.vertices)) if getattr(mesh, "vertices", None) is not None else None,
            "faces": int(len(mesh.faces)) if getattr(mesh, "faces", None) is not None else None,
        }
        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        if uv is None:
            payload["uv_present"] = False
            payload["uv_shape"] = None
        else:
            uv_np = np.asarray(uv)
            payload["uv_present"] = True
            payload["uv_shape"] = list(uv_np.shape)
            payload["uv_min"] = uv_np.min(axis=0).tolist() if uv_np.size else None
            payload["uv_max"] = uv_np.max(axis=0).tolist() if uv_np.size else None
        self._trace_json(name, payload)

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=True, save_glb=True,
                 seed: int = 42, texture_steps=None, texture_guidance=None):
        """Generate texture for 3D mesh using multiview diffusion"""
        # Ensure image_prompt is a list
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        if not isinstance(image_prompt, List):
            image_prompt = [image_prompt]
        else:
            image_prompt = image_path

        # Process mesh
        path = os.path.dirname(mesh_path)
        if use_remesh:
            print("Texture: remeshing mesh...")
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path
        self._trace_copy(processed_mesh_path, "remeshed_mesh" + os.path.splitext(processed_mesh_path)[1])

        # Output path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        print("Texture: loading mesh and generating UVs...")
        mesh = trimesh.load(processed_mesh_path)
        self._trace_mesh_state("mesh_loaded_state.json", mesh)
        mesh = mesh_uv_wrap(mesh)
        self._trace_mesh_state("mesh_after_uv_wrap_state.json", mesh)
        trace_dir = self._trace_dir()
        if trace_dir is not None:
            mesh.export(os.path.join(trace_dir, "uvwrapped_mesh.obj"))
        reloaded_uvwrapped = None
        if trace_dir is not None:
            try:
                reloaded_uvwrapped = trimesh.load(os.path.join(trace_dir, "uvwrapped_mesh.obj"))
            except Exception:
                reloaded_uvwrapped = None
        self._trace_mesh_state("mesh_reloaded_uvwrapped_state.json", reloaded_uvwrapped)
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        print("Texture: selecting views and rendering control maps...")
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )
        self._trace_json(
            "selected_views.json",
            {
                "elevs": selected_camera_elevs,
                "azims": selected_camera_azims,
                "weights": selected_view_weights,
            },
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
        for idx, image in enumerate(normal_maps):
            self._trace_image(f"control_normal_{idx:02d}.png", image)
        for idx, image in enumerate(position_maps):
            self._trace_image(f"control_position_{idx:02d}.png", image)

        ##########  Style  ###########
        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        ###########  Multiview  ##########
        print("Texture: running multiview diffusion...")
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
            seed=seed,
            num_inference_steps=texture_steps,
            guidance_scale=texture_guidance,
        )
        for token, images in multiviews_pbr.items():
            for idx, image in enumerate(images):
                self._trace_image(f"multiview_{token}_{idx:02d}.png", image)
        ###########  Enhance  ##########
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

        for i in tqdm(range(len(enhance_images["albedo"])), desc="Texture: upscaling views"):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])

        ###########  Bake  ##########
        print("Texture: baking texture maps...")
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        self._trace_image("bake_albedo_pre_inpaint.png", texture * 255.0)
        self._trace_image("bake_albedo_mask.png", mask_np)
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        self._trace_image("bake_mr_pre_inpaint.png", texture_mr * 255.0)
        self._trace_image("bake_mr_mask.png", mask_mr_np)

        ##########  inpaint  ###########
        print("Texture: inpainting and exporting mesh...")
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self._trace_image("bake_albedo_final.png", texture * 255.0)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self._trace_image("bake_mr_final.png", texture_mr * 255.0)
            self.render.set_texture_mr(texture_mr)

        self.render.save_mesh(output_mesh_path, downsample=True)
        self._trace_copy(output_mesh_path, "textured_mesh.obj")
        self._trace_copy(output_mesh_path.replace(".obj", ".mtl"), "textured_mesh.mtl")
        self._trace_copy(output_mesh_path.replace(".obj", ".jpg"), "textured_mesh.jpg")
        self._trace_copy(output_mesh_path.replace(".obj", "_metallic.jpg"), "textured_mesh_metallic.jpg")
        self._trace_copy(output_mesh_path.replace(".obj", "_roughness.jpg"), "textured_mesh_roughness.jpg")

        if save_glb:
            from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
            output_glb_path = output_mesh_path.replace(".obj", ".glb")
            self._trace_copy(output_glb_path, "textured_mesh.glb")

        return output_mesh_path
