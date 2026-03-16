"""
MLX Texture Generation Pipeline for Hunyuan3D.

This is a hybrid pipeline: keep upstream orchestration wherever possible and
swap in MLX-backed model execution only for the heavy neural components.
"""
import gc
import json
import logging
import os
import re
import sys
import numpy as np
import torch
from PIL import Image
from typing import List, Optional
from types import SimpleNamespace

import mlx.core as mx

hy3dpaint_dir = os.path.join(os.path.dirname(__file__), '..', 'hy3dpaint')
if hy3dpaint_dir not in sys.path:
    sys.path.insert(0, hy3dpaint_dir)
from textureGenPipeline import Hunyuan3DPaintPipeline, ViewProcessor, Hunyuan3DPaintConfig
from utils.multiview_utils import multiviewDiffusionNet
from utils.image_super_utils import imageSuperNet
from hunyuanpaintpbr.pipeline import HunyuanPaintPipeline

from . import load_safetensors

log = logging.getLogger("mlx_texture.pipeline")


def _clear_cache():
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _internal_trace_dir() -> str | None:
    trace_dir = os.environ.get("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR")
    if not trace_dir:
        return None
    os.makedirs(trace_dir, exist_ok=True)
    return trace_dir


def _tensor_stats(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _tensor_stats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensor_stats(v) for v in value]
    if torch.is_tensor(value):
        arr = value.detach().cpu().float().numpy()
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.array(value, dtype=np.float32)
    else:
        return value
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def _write_internal_trace(name: str, payload) -> None:
    trace_dir = _internal_trace_dir()
    if trace_dir is None:
        return
    with open(os.path.join(trace_dir, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_internal_array(name: str, value) -> None:
    trace_dir = _internal_trace_dir()
    if trace_dir is None or value is None:
        return
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.array(value)
    else:
        return
    np.save(os.path.join(trace_dir, name), arr)


class MlxImageSuperNet(imageSuperNet):
    """Upstream imageSuperNet shell with MLX-backed execution."""

    def __init__(self, pipeline: "MlxTexturePipeline"):
        self.pipeline = pipeline

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.pipeline._esrgan_upscale(image)


class _LatentDist:
    def __init__(self, sample: torch.Tensor):
        self._sample = sample

    def sample(self):
        return self._sample


class _EncodeOutput:
    def __init__(self, sample: torch.Tensor):
        self.latent_dist = _LatentDist(sample)


class _MlxAutoencoderKLAdapter(torch.nn.Module):
    def __init__(self, owner: "MlxTexturePipeline"):
        super().__init__()
        self.owner = owner
        self._dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float16), requires_grad=False)
        self.config = SimpleNamespace(
            block_out_channels=(128, 256, 512, 512),
            scaling_factor=0.18215,
        )

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    _encode_call_count = 0

    def encode(self, x: torch.Tensor):
        self.owner._load_vae()
        _MlxAutoencoderKLAdapter._encode_call_count += 1
        call_id = _MlxAutoencoderKLAdapter._encode_call_count
        _tracing = os.environ.get("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR")
        if _tracing:
            _write_internal_trace(f"vae_encode_{call_id}_input.json", _tensor_stats(x))
            _write_internal_array(f"vae_encode_{call_id}_input.npy", x)
        # The caller (encode_images in pipeline.py:165) already normalizes
        # images from [0,1] to [-1,1] via (x - 0.5) * 2.0 before calling
        # vae.encode(). We must NOT normalize again — doing so maps [-1,1]
        # to [-3,1] and corrupts all encoded latents (ref, normal, position).
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # Always compute in float32 inside MLX, regardless of input dtype
        mx_x = mx.array(x_nhwc.detach().cpu().float().numpy())
        mx_mean, mx_logvar = self.owner._vae.encode(mx_x)
        mx.eval(mx_mean, mx_logvar)
        if _tracing:
            _write_internal_array(f"vae_encode_{call_id}_mean.npy", np.array(mx_mean))
            _write_internal_array(f"vae_encode_{call_id}_logvar.npy", np.array(mx_logvar))
        # Use mean (mode) instead of sampling — posterior std is ~0.0004,
        # negligible vs mean magnitude of ~4.5. Deterministic = more stable.
        mx_lat = mx_mean * self.owner._vae.scaling_factor
        mx.eval(mx_lat)
        lat = torch.from_numpy(np.array(mx_lat)).permute(0, 3, 1, 2).contiguous()
        lat = lat / self.config.scaling_factor
        result = lat.to(dtype=x.dtype)
        if _tracing:
            _write_internal_trace(f"vae_encode_{call_id}_output.json", _tensor_stats(result))
            _write_internal_array(f"vae_encode_{call_id}_output.npy", result)
        return _EncodeOutput(result)

    def decode(self, z: torch.Tensor, return_dict: bool = False, generator=None):
        del generator
        self.owner._load_vae()
        z_nhwc = z.permute(0, 2, 3, 1).contiguous() * self.config.scaling_factor
        # Always compute in float32 inside MLX, regardless of input dtype
        mx_z = mx.array(z_nhwc.detach().cpu().float().numpy())
        mx_out = self.owner._vae.decode(mx_z)
        mx.eval(mx_out)
        out = torch.from_numpy(np.array(mx_out)).permute(0, 3, 1, 2).contiguous()
        if return_dict:
            return SimpleNamespace(sample=out.to(dtype=z.dtype))
        return (out.to(dtype=z.dtype),)


class _MlxUNet2p5DAdapter(torch.nn.Module):
    def __init__(self, owner: "MlxTexturePipeline"):
        super().__init__()
        self.owner = owner
        self._dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float16), requires_grad=False)
        self.config = SimpleNamespace(
            in_channels=4,
            sample_size=64,
            time_cond_proj_dim=None,
        )
        self.pbr_setting = ["albedo", "mr"]
        self.use_dino = True
        self.use_ra = True
        self.use_learned_text_clip = True
        for token in self.pbr_setting:
            self.register_buffer(f"learned_text_clip_{token}", torch.zeros((77, 1024), dtype=torch.float16))
        self.register_buffer("learned_text_clip_ref", torch.zeros((77, 1024), dtype=torch.float16))

    def sync_learned_conditioning(self):
        self.owner._load_unet()
        for token in self.pbr_setting:
            value = getattr(self.owner._unet, f"learned_text_clip_{token}", None)
            if value is None:
                continue
            self._buffers[f"learned_text_clip_{token}"] = torch.from_numpy(
                np.array(value, dtype=np.float32)
            ).half()
        value = getattr(self.owner._unet, "learned_text_clip_ref", None)
        if value is not None:
            self._buffers["learned_text_clip_ref"] = torch.from_numpy(
                np.array(value, dtype=np.float32)
            ).half()

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def forward(self, sample, timestep, encoder_hidden_states=None, timestep_cond=None,
                cross_attention_kwargs=None, added_cond_kwargs=None, return_dict=False, **kwargs):
        del timestep_cond, cross_attention_kwargs, added_cond_kwargs
        self.owner._load_unet()

        def _to_mx(value, key=None):
            if value is None:
                return None
            if isinstance(value, dict):
                return {k: _to_mx(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_mx(v, key) for v in value]
            if torch.is_tensor(value):
                if key == "position_maps" and value.ndim == 5:
                    # Keep position maps out of MLX entirely. They are only used
                    # by the upstream voxel-index helper, and bouncing them
                    # through MLX introduces an unnecessary conversion boundary
                    # in the live texture path.
                    return value.permute(0, 1, 3, 4, 2).contiguous().detach().cpu().numpy()
                # Always compute in float32 inside MLX, regardless of torch dtype
                return mx.array(value.detach().cpu().float().numpy())
            return value

        sample_mx = _to_mx(sample, "sample")
        timestep_mx = _to_mx(timestep, "timestep")
        encoder_hidden_states_mx = _to_mx(encoder_hidden_states, "encoder_hidden_states")

        _tracing = os.environ.get("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR")
        if _tracing:
            _write_internal_trace(
                "adapter_forward_in.json",
                {
                    "sample": _tensor_stats(sample),
                    "timestep": _tensor_stats(timestep),
                    "encoder_hidden_states": _tensor_stats(encoder_hidden_states),
                    "kwargs_keys": sorted(k for k in kwargs if k != "cache"),
                },
            )
            _write_internal_array("adapter_sample.npy", sample)
            _write_internal_array("adapter_timestep.npy", timestep)
            _write_internal_array("adapter_encoder_hidden_states.npy", encoder_hidden_states)

        cache = kwargs.get("cache")
        if cache is None:
            cache = {}
            kwargs["cache"] = cache

        # Build the MLX kwargs dict, but preserve the `cache` dict by
        # reference so that MlxUNet2p5D's modifications (condition_embed_dict,
        # dino_hidden_states_proj, position_voxel_indices) persist across
        # denoising steps — matching upstream behaviour where the cache is
        # populated on step 1 and reused on subsequent steps.
        mx_kwargs = {k: _to_mx(v, k) for k, v in kwargs.items() if k != "cache"}
        mx_kwargs["cache"] = cache

        out = self.owner._unet(
            sample_mx,
            timestep_mx,
            encoder_hidden_states_mx,
            **mx_kwargs,
        )
        mx.eval(out)
        out_np = np.array(out)
        if _tracing:
            _write_internal_trace(
                "adapter_forward_out.json",
                {
                    "raw_out": _tensor_stats(out_np),
                },
            )
            _write_internal_array("adapter_raw_out.npy", out_np)
        # MLX UNet always returns NHWC; adapter must return NCHW to shared pipeline
        if out_np.ndim == 4:
            out_np = out_np.transpose(0, 3, 1, 2)  # NHWC → NCHW
        out_torch = torch.from_numpy(out_np).to(dtype=sample.dtype)
        if return_dict:
            return SimpleNamespace(sample=out_torch)
        return (out_torch,)


class _MlxInnerPaintPipeline(HunyuanPaintPipeline):
    def __init__(self, owner: "MlxTexturePipeline"):
        import diffusers
        from diffusers import DiffusionPipeline
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.schedulers import UniPCMultistepScheduler

        DiffusionPipeline.__init__(self)
        self.owner = owner
        vae = _MlxAutoencoderKLAdapter(owner)
        unet = _MlxUNet2p5DAdapter(owner)
        # Must match the upstream scheduler config exactly — wrong beta
        # values / schedule type produce a completely different noise schedule,
        # causing the UNet to operate at wrong noise levels and generate garbage.
        scheduler = UniPCMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            steps_offset=1,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )
        self.register_modules(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def prepare(self):
        self.owner._load_unet()
        self.unet.sync_learned_conditioning()
        return

    def eval(self):
        return self

    def run_safety_checker(self, image, device, dtype):
        del device, dtype
        return image, None

    def maybe_free_model_hooks(self):
        # This MLX-backed wrapper does not use diffusers offload hooks.
        return


class _MlxDinoCondAdapter:
    def __init__(self, pipeline: "MlxTexturePipeline"):
        self.pipeline = pipeline

    def __call__(self, image: Image.Image):
        self.pipeline._load_dino()
        # DINO weights are already fp16 from _load_dino().
        out = self.pipeline._dino([image])
        mx.eval(out)
        result = torch.from_numpy(np.array(out, dtype=np.float32)).to(torch.float16)
        if os.environ.get("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR"):
            _write_internal_trace("dino_output.json", _tensor_stats(result))
            _write_internal_array("dino_output.npy", result)
        return result


class MlxMultiviewDiffusionNet(multiviewDiffusionNet):
    """Upstream multiviewDiffusionNet shell with MLX-backed execution."""

    def __init__(self, pipeline: "MlxTexturePipeline"):
        self.pipeline = pipeline
        self.device = "cpu"
        self.cfg = SimpleNamespace(model=SimpleNamespace(params=SimpleNamespace(view_size=pipeline.view_size)))
        self.mode = "pbr"
        self.pipeline = _MlxInnerPaintPipeline(pipeline)
        self.dino_v2 = _MlxDinoCondAdapter(pipeline)


class MlxTexturePipeline(Hunyuan3DPaintPipeline):
    """Full MLX texture generation pipeline."""

    def __init__(self, model_dir: str, weights_dir: str = None):
        """
        Args:
            model_dir: path to Hunyuan3D-2.1 model directory
            weights_dir: path to converted MLX weights (default: model_dir/mlx_weights)
        """
        # Keep MLX texture logs quiet by default; upstream progress bars remain visible.
        logging.getLogger("mlx_texture").setLevel(logging.WARNING)
        logging.getLogger("mlx_texture.attention").setLevel(logging.WARNING)
        logging.getLogger("mlx_texture.unet").setLevel(logging.WARNING)
        logging.getLogger("mlx_texture.unet2p5d").setLevel(logging.WARNING)

        self.model_dir = model_dir
        self.weights_dir = weights_dir or os.path.join(model_dir, "mlx_weights")

        # Lazy-loaded models
        self._dino = None
        self._unet = None
        self._vae = None
        self._esrgan = None
        self._render = None
        self._current_camera_azims = []
        self.view_size = 512
        self.render_size = 2048
        self.texture_size = 4096
        self.bake_exp = 4
        self.negative_prompt = "watermark, ugly, deformed, noisy, blurry, low contrast"

        config = Hunyuan3DPaintConfig(max_num_view=6, resolution=self.view_size)
        super().__init__(config=config)
        self.render = self._get_render()
        self.view_processor = ViewProcessor(self.config, self.render)

    def load_models(self):
        """Mirror upstream Hunyuan3DPaintPipeline model registry."""
        self.models["super_model"] = MlxImageSuperNet(self)
        self.models["multiview_model"] = MlxMultiviewDiffusionNet(self)

    # -------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------

    def _load_dino(self):
        if self._dino is not None:
            return
        from .dinov2 import MlxDINOv2
        from . import remap_dinov2_weights

        self._dino = MlxDINOv2(
            dim=1536, num_heads=24, num_layers=40,
            patch_size=14, image_size=224, weights_image_size=518,
        )

        # Search for weights in order of preference
        candidates = [
            os.path.join(self.weights_dir, "dinov2_giant.safetensors"),
            # HuggingFace cache
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--facebook--dinov2-giant/"
                "snapshots/611a9d42f2335e0f921f1e313ad3c1b7178d206d/model.safetensors"
            ),
        ]
        # Also try glob for any snapshot
        import glob
        candidates += sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--dinov2-giant/snapshots/*/model.safetensors"
        )))

        for weights_path in candidates:
            if os.path.exists(weights_path):
                raw = load_safetensors(weights_path)
                weights = remap_dinov2_weights(raw)
                self._dino.load_weights(list(weights.items()))
                # Convert to fp16 once at load time to match upstream
                # (multiview_utils.py:54 loads DINO in fp16).
                import mlx.utils
                fp16_weights = [(k, v.astype(mx.float16))
                                for k, v in mlx.utils.tree_flatten(self._dino.parameters())]
                self._dino.load_weights(fp16_weights)
                return

        log.warning("DINOv2-giant weights not found")

    def _load_vae(self):
        if self._vae is not None:
            return
        from .vae_kl import MlxAutoencoderKL

        # Try loading from diffusers directory
        vae_candidates = [
            os.path.join(self.model_dir, "hunyuan3d-paintpbr-v2-1", "vae"),
            os.path.join(self.model_dir, "hunyuan-texture-v2-1", "vae"),
        ]
        for vae_dir in vae_candidates:
            if os.path.isdir(vae_dir):
                self._vae = MlxAutoencoderKL.from_diffusers(vae_dir)
                return

        self._vae = MlxAutoencoderKL()
        weights_path = os.path.join(self.weights_dir, "vae.safetensors")
        if os.path.exists(weights_path):
            raw = load_safetensors(weights_path)
            self._vae.load_weights(list(raw.items()))

    def _load_unet(self):
        if self._unet is not None:
            return
        from .unet2p5d import MlxUNet2p5D
        from .unet_blocks import UNet2DConditionModel

        base_unet = UNet2DConditionModel(
            in_channels=12,
            out_channels=4,
            block_out_channels=[320, 640, 1280, 1280],
            layers_per_block=2,
            cross_attention_dim=1024,
            attention_head_dim=[5, 10, 20, 20],
        )
        self._unet = MlxUNet2p5D(base_unet, use_dino=True)

        # Prefer the upstream PyTorch checkpoint path for the live texture UNet.
        # The legacy converted MLX safetensors can be stale or incorrectly
        # converted even when the key set matches, which makes texture failures
        # look like model-path bugs instead of load-path bugs.
        unet_dir = os.path.join(self.model_dir, "hunyuan3d-paintpbr-v2-1", "unet")
        bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
        if os.path.exists(bin_path):
            self._load_unet_from_pytorch(bin_path)
            return

        # Fall back to a pre-converted MLX snapshot only if the upstream
        # checkpoint is unavailable.
        weights_path = os.path.join(self.weights_dir, "unet.safetensors")
        if os.path.exists(weights_path):
            raw = load_safetensors(weights_path)
            self._unet.load_weights(list(raw.items()), strict=False)
            return

        log.warning("UNet weights not found")

    @staticmethod
    def _remap_unet_key(k: str) -> str:
        """Remap PyTorch UNet2p5D weight keys to MLX naming convention."""
        # Preserve top-level learned conditioning weights outside the wrapped UNet.
        for prefix in ("unet.image_proj_model_dino.", "unet.learned_text_clip_"):
            if k.startswith(prefix):
                return k[len("unet."):]

        # Keep the main wrapped UNet under the `unet.` prefix.
        k = re.sub(r"\.to_out\.0\.", ".to_out.", k)
        k = re.sub(r"\.to_out_mr\.0\.", ".to_out_mr.", k)
        k = k.replace(".ff.net.0.", ".ff.net_0.")
        k = k.replace(".ff.net.2.", ".ff.net_2.")

        # 2.5D wrapper remaps. Order matters: rewrite processor branches before
        # inserting `.base.` for the original transformer block.
        k = k.replace(".transformer.attn1.processor.", ".mda_processor.")
        k = k.replace(".transformer.", ".base.")
        k = k.replace(".attn_dino.to_q.", ".dino_to_q.")
        k = k.replace(".attn_dino.to_k.", ".dino_to_k.")
        k = k.replace(".attn_dino.to_v.", ".dino_to_v.")
        k = k.replace(".attn_dino.to_out.", ".dino_to_out.")
        k = k.replace(".attn_multiview.to_q.", ".ma_to_q.")
        k = k.replace(".attn_multiview.to_k.", ".ma_to_k.")
        k = k.replace(".attn_multiview.to_v.", ".ma_to_v.")
        k = k.replace(".attn_multiview.to_out.", ".ma_to_out.")
        k = k.replace(".attn_refview.processor.", ".ra_processor.")
        k = k.replace(".attn_refview.to_q.", ".ra_to_q.")
        k = k.replace(".attn_refview.to_k.", ".ra_to_k.")
        k = k.replace(".attn_refview.to_v.", ".ra_to_v.")
        k = k.replace(".attn_refview.to_out.", ".ra_to_out.")

        return k

    def _load_unet_from_pytorch(self, bin_path: str):
        """Load UNet weights from PyTorch .bin file with key remapping."""
        import torch
        state = torch.load(bin_path, map_location="cpu", weights_only=False)

        weights = {}
        for k, v in state.items():
            arr = v.float().numpy()
            # Conv2d: PyTorch OIHW → MLX OHWI
            if arr.ndim == 4 and "weight" in k:
                arr = np.transpose(arr, (0, 2, 3, 1))

            new_k = self._remap_unet_key(k)
            weights[new_k] = mx.array(arr)

        self._unet.load_weights(list(weights.items()), strict=False)

        # Comprehensive weight loading verification
        import mlx.utils
        model_params = dict(mlx.utils.tree_flatten(self._unet.parameters()))
        model_keys = set(model_params.keys())
        weight_keys = set(weights.keys())
        loaded = model_keys & weight_keys
        missing = model_keys - weight_keys
        extra = weight_keys - model_keys

        # Per-layer shape comparison for loaded weights
        shape_mismatches = []
        for k in loaded:
            model_shape = tuple(model_params[k].shape)
            weight_shape = tuple(weights[k].shape)
            if model_shape != weight_shape:
                shape_mismatches.append((k, model_shape, weight_shape))

        total_model = len(model_keys)
        total_loaded = len(loaded)
        log.info(
            "UNet weight verification: %d/%d params loaded, %d missing, %d extra, %d shape mismatches",
            total_loaded, total_model, len(missing), len(extra), len(shape_mismatches),
        )
        if missing:
            log.warning("UNet missing %d params: %s", len(missing), sorted(missing)[:10])
        if extra:
            log.warning("UNet extra %d keys (not in model): %s", len(extra), sorted(extra)[:10])
        if shape_mismatches:
            for k, m_shape, w_shape in shape_mismatches:
                log.error("UNet shape mismatch: %s model=%s weight=%s", k, m_shape, w_shape)

    def _load_esrgan(self):
        if self._esrgan is not None:
            return
        from .esrgan import MlxESRGAN

        # Try loading from .pth directly
        pth_path = os.path.join(os.path.dirname(__file__), '..', 'hy3dpaint',
                                'ckpt', 'RealESRGAN_x4plus.pth')
        if os.path.exists(pth_path):
            self._esrgan = MlxESRGAN.from_pth(pth_path)
            return

        self._esrgan = MlxESRGAN()
        weights_path = os.path.join(self.weights_dir, "esrgan.safetensors")
        if os.path.exists(weights_path):
            raw = load_safetensors(weights_path)
            self._esrgan.load_weights(list(raw.items()))

    def _get_render(self):
        if self._render is not None:
            return self._render
        from DifferentiableRenderer.MeshRender import MeshRender

        self._render = MeshRender(
            default_resolution=self.render_size,
            texture_size=self.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
            device="cpu",
        )
        return self._render

    def _unload(self, *names):
        for name in names:
            attr = f'_{name}'
            if hasattr(self, attr):
                setattr(self, attr, None)
        _clear_cache()

    # -------------------------------------------------------------------
    # Image processing helpers
    # -------------------------------------------------------------------

    def _pil_to_latent(self, img: Image.Image) -> mx.array:
        """Encode a PIL image to VAE latent. Returns (1, H/8, W/8, 4) NHWC."""
        arr = np.array(img.resize((self.view_size, self.view_size))).astype(np.float32) / 255.0
        arr = (arr - 0.5) * 2.0  # [-1, 1]
        x = mx.array(arr)[None]  # (1, H, W, 3) NHWC
        mean, logvar = self._vae.encode(x)
        # Sample from posterior to match upstream
        std = mx.exp(0.5 * logvar)
        noise = mx.random.normal(mean.shape)
        sample = mean + std * noise
        return sample * self._vae.scaling_factor  # (1, H/8, W/8, 4)

    def _latent_to_pil(self, latent: mx.array) -> Image.Image:
        """Decode a single latent to PIL image. latent: (1, H/8, W/8, 4) NHWC."""
        decoded = self._vae.decode(latent)  # (1, H, W, 3) NHWC
        mx.eval(decoded)
        img = np.array(decoded[0])
        img = np.clip((img + 1.0) * 0.5 * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _esrgan_upscale(self, img: Image.Image) -> Image.Image:
        """4x upscale a PIL image using ESRGAN."""
        self._load_esrgan()
        arr = np.array(img).astype(np.float32) / 255.0
        x = mx.array(arr)[None]  # (1, H, W, 3) NHWC
        out = self._esrgan(x)
        mx.eval(out)
        out_np = np.clip(np.array(out[0]) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

    def __call__(self, mesh_path: str, image_path, output_mesh_path: str = None,
                 use_remesh: bool = True, save_glb: bool = True,
                 seed: int = 42,
                 texture_steps: int = None, texture_guidance: float = None) -> str:
        return super().__call__(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_mesh_path,
            use_remesh=use_remesh,
            save_glb=save_glb,
            seed=seed,
            texture_steps=texture_steps,
            texture_guidance=texture_guidance,
        )
