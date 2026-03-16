"""
MLX Shape Generation Pipeline for Hunyuan3D.

Orchestrates: DINO encode → DiT denoise → VAE decode → Geo decode → marching cubes.
Sequential model loading with memory cleanup between stages.
"""
import gc
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import torch

hy3d_path = os.path.join(os.path.dirname(__file__), '..', 'hy3dshape')
if hy3d_path not in sys.path:
    sys.path.insert(0, hy3d_path)
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
from hy3dshape.models.autoencoders import (
    MCSurfaceExtractor,
    SurfaceExtractors,
    VanillaVolumeDecoder,
    VectsetVAE,
)
from hy3dshape.utils import synchronize_timer

from . import (
    load_safetensors,
    remap_dit_weights, remap_vae_weights,
    remap_geo_decoder_weights, remap_dinov2_weights,
)
from .dinov2 import MlxDINOv2
from .dit import HunYuanDiTPlain
from .vae_transformer import VAETransformer
from .geo_decoder import CrossAttentionDecoder

MLX_ARRAY_TYPE = type(mx.array([0], dtype=mx.float32))


class _MlxConditionerAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, image=None, **kwargs):
        del kwargs
        self.pipeline._load_dino()
        if image is None:
            raise ValueError("image must be provided to the MLX conditioner adapter")

        image_np = image.detach().cpu().numpy()
        image_list = [
            ((sample.transpose(1, 2, 0) + 1.0) / 2.0).astype(np.float32)
            for sample in image_np
        ]
        cond_features = self.pipeline._dino(image_list)
        mx.eval(cond_features)
        return {
            "main": torch.from_numpy(np.array(cond_features)).to(
                device=self.pipeline.device,
                dtype=self.pipeline.dtype,
            )
        }

    def unconditional_embedding(self, bsz, **kwargs):
        del kwargs
        return {
            "main": torch.zeros(
                (bsz, 1370, 1024),
                device=self.pipeline.device,
                dtype=self.pipeline.dtype,
            )
        }


class _MlxDiTAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, x, t, cond, **kwargs):
        def _to_mx(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                return mx.array(value.detach().cpu().numpy())
            if isinstance(value, dict):
                return {k: _to_mx(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_mx(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_mx(v) for v in value)
            return value

        self.pipeline._load_dit()
        x_mx = _to_mx(x)
        t_mx = _to_mx(t)
        cond_mx = _to_mx(cond)
        kwargs_mx = {k: _to_mx(v) for k, v in kwargs.items()}
        out = self.pipeline._dit(x_mx, t_mx, cond_mx, **kwargs_mx)
        mx.eval(out)
        return torch.from_numpy(np.array(out)).to(dtype=x.dtype)


class _MlxGeoDecoderAdapter(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, queries, latents, **kwargs):
        del kwargs
        self.pipeline._load_geo()
        queries_mx = mx.array(queries.detach().cpu().numpy())
        latents_mx = mx.array(latents.detach().cpu().numpy())
        out = self.pipeline._geo(queries_mx, latents_mx)
        mx.eval(out)
        return torch.from_numpy(np.array(out)).to(device=queries.device, dtype=queries.dtype)


class _MlxShapeVAEFacade(VectsetVAE):
    latent_shape = (4096, 64)
    scale_factor = 1.0039506158752403

    def __init__(self, pipeline):
        super().__init__(
            volume_decoder=VanillaVolumeDecoder(),
            surface_extractor=MCSurfaceExtractor(),
        )
        self.pipeline = pipeline
        self.geo_decoder = _MlxGeoDecoderAdapter(pipeline)

    def forward(self, latents):
        self.pipeline._load_vae()
        latents_mx = mx.array(latents.detach().cpu().numpy())
        out = self.pipeline._vae(latents_mx)
        mx.eval(out)
        out_torch = torch.from_numpy(np.array(out)).to(device=latents.device, dtype=latents.dtype)
        self.pipeline._trace_save("decoded_latents_mlx", out_torch)
        return out_torch

    def decode(self, latents):
        return self.forward(latents)

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        mc_algo = kwargs.get("mc_algo")
        if mc_algo is not None:
            if mc_algo not in SurfaceExtractors:
                raise ValueError(f"Unknown mc_algo {mc_algo}")
            self.surface_extractor = SurfaceExtractors[mc_algo]()
        with synchronize_timer('Volume decoding'):
            grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        self.pipeline._trace_save("grid_logits_mlx", grid_logits)
        with synchronize_timer('Surface extraction'):
            outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs

    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='dmc',
    ):
        del adaptive_kv_selection, topk_mode, mc_algo
        if enabled:
            raise NotImplementedError(
                "FlashVDM decoder is not supported in the MLX shape pipeline. "
                "The MLX geo decoder does not implement the upstream cross-attention processor hooks."
            )
        self.volume_decoder = VanillaVolumeDecoder()
        self.surface_extractor = MCSurfaceExtractor()


def _clear_gpu():
    """Clear MLX Metal cache."""
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


class MlxHunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTFlowMatchingPipeline):
    """
    Shape generation pipeline using pure MLX for all neural network inference.
    Marching cubes stays in PyTorch/CPU.

    Weight files expected (safetensors format):
        {weights_dir}/hunyuan3d-dit-v2-1/model.safetensors
        {weights_dir}/hunyuan3d-vae-v2-1/model.safetensors
        {weights_dir}/hunyuan3d-dit-v2-1/conditioner.safetensors
    Or the original .ckpt files (will use torch.load as fallback).
    """

    def __init__(self, weights_dir: str, lazy_load: bool = True):
        """
        Args:
            weights_dir: path to weights/tencent/Hunyuan3D-2.1/
            lazy_load: if True, models are loaded on first use and freed after
        """
        self.weights_dir = weights_dir
        self.dit_dir = os.path.join(weights_dir, 'hunyuan3d-dit-v2-1')
        self.vae_dir = os.path.join(weights_dir, 'hunyuan3d-vae-v2-1')
        self.lazy_load = lazy_load

        # Image preprocessor (CPU, lightweight)
        from hy3dshape.preprocessors import ImageProcessorV2
        self.image_processor = ImageProcessorV2(size=512, border_ratio=0.15)

        # Models loaded lazily
        self._dino = None
        self._dit = None
        self._vae = None
        self._geo = None
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.conditioner = _MlxConditionerAdapter(self)
        self.model = _MlxDiTAdapter(self)
        self.vae = _MlxShapeVAEFacade(self)
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.kwargs = {}

        super().__init__(
            vae=self.vae,
            model=self.model,
            scheduler=self.scheduler,
            conditioner=self.conditioner,
            image_processor=self.image_processor,
            device="cpu",
            dtype=torch.float32,
        )

        # Preload if not lazy
        if not lazy_load:
            self._load_all()

    def _trace_dir(self):
        trace_dir = os.environ.get("HY3D_MLX_SHAPE_TRACE_DIR")
        if not trace_dir:
            return None
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _trace_save(self, name: str, value):
        trace_dir = self._trace_dir()
        if trace_dir is None:
            return
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, MLX_ARRAY_TYPE):
            arr = np.array(value)
        else:
            arr = np.array(value)
        np.save(trace_dir / f"{name}.npy", arr)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device) if not isinstance(device, torch.device) else device
        if dtype is not None:
            self.dtype = dtype

    def _load_all(self):
        self._load_dino()
        self._load_dit()
        self._load_vae()
        self._load_geo()

    def _load_dino(self):
        if self._dino is not None:
            return
        print("[MLX] Loading DINOv2...")
        self._dino = MlxDINOv2(
            dim=1024, num_heads=16, num_layers=24,
            patch_size=14, image_size=518,
        )

        # Try safetensors first, fall back to loading via HF transformers
        st_path = os.path.join(self.dit_dir, 'conditioner.safetensors')
        if os.path.exists(st_path):
            raw = load_safetensors(st_path)
            weights = remap_dinov2_weights(raw)
        else:
            # Load from the conditioner inside the combined .ckpt
            print("[MLX] No conditioner.safetensors found, loading from .ckpt via torch...")
            import torch
            ckpt_path = os.path.join(self.dit_dir, 'model.fp16.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if 'conditioner' in ckpt:
                cond_state = ckpt['conditioner']
            else:
                # conditioner weights are embedded in main_image_encoder.model.*
                cond_state = {k: v for k, v in ckpt.items()
                              if 'main_image_encoder' in k}
            raw_mx = {k: mx.array(v.float().numpy()) for k, v in cond_state.items()}
            weights = remap_dinov2_weights(raw_mx)
            del ckpt

        self._dino.load_weights(list(weights.items()))
        print(f"[MLX] DINOv2 loaded.")

    def _load_dit(self):
        if self._dit is not None:
            return
        print("[MLX] Loading DiT...")
        self._dit = HunYuanDiTPlain(
            input_size=4096, in_channels=64, hidden_size=2048,
            context_dim=1024, depth=21, num_heads=16,
            qk_norm=True, text_len=1370,
            qkv_bias=False, num_moe_layers=6,
            num_experts=8, moe_top_k=2,
        )

        st_path = os.path.join(self.dit_dir, 'model.safetensors')
        if os.path.exists(st_path):
            raw = load_safetensors(st_path)
            # Filter to 'model.' prefix keys
            model_weights = {}
            for k, v in raw.items():
                if k.startswith('model.'):
                    model_weights[k[len('model.'):]] = v
                else:
                    model_weights[k] = v
            weights = remap_dit_weights(model_weights)
        else:
            print("[MLX] No model.safetensors found, loading from .ckpt via torch...")
            import torch
            ckpt_path = os.path.join(self.dit_dir, 'model.fp16.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if 'model' in ckpt:
                state = ckpt['model']
            else:
                state = ckpt
            raw_mx = {k: mx.array(v.float().numpy()) for k, v in state.items()}
            weights = remap_dit_weights(raw_mx)
            del ckpt

        # Check for unloaded weights
        import mlx.utils
        model_keys = set(k for k, _ in mlx.utils.tree_flatten(self._dit.parameters()))
        weight_keys = set(weights.keys())
        missing = model_keys - weight_keys
        extra = weight_keys - model_keys
        if missing:
            print(f"[MLX] WARNING: {len(missing)} DiT params NOT in weights: {sorted(missing)[:10]}...")
        if extra:
            print(f"[MLX] WARNING: {len(extra)} weight keys NOT in DiT model: {sorted(extra)[:10]}...")

        self._dit.load_weights(list(weights.items()))
        print(f"[MLX] DiT loaded ({sum(p.size for _, p in mlx.utils.tree_flatten(self._dit.parameters()))} params). "
              f"Matched: {len(model_keys & weight_keys)}/{len(model_keys)} params")

    def _load_vae(self):
        if self._vae is not None:
            return
        print("[MLX] Loading VAE transformer...")
        self._vae = VAETransformer(
            num_latents=4096, embed_dim=64,
            width=1024, heads=16,
            num_decoder_layers=16,
            qkv_bias=False, qk_norm=True,
        )

        st_path = os.path.join(self.vae_dir, 'model.safetensors')
        if os.path.exists(st_path):
            raw = load_safetensors(st_path)
            vae_weights = {}
            for k, v in raw.items():
                # Only keep post_kl and transformer keys
                if k.startswith('post_kl.') or k.startswith('transformer.'):
                    vae_weights[k] = v
            weights = remap_vae_weights(vae_weights)
        else:
            print("[MLX] No VAE safetensors found, loading from .ckpt via torch...")
            import torch
            ckpt_path = os.path.join(self.vae_dir, 'model.fp16.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            vae_state = {}
            for k, v in ckpt.items():
                if k.startswith('post_kl.') or k.startswith('transformer.'):
                    vae_state[k] = v
            raw_mx = {k: mx.array(v.float().numpy()) for k, v in vae_state.items()}
            weights = remap_vae_weights(raw_mx)
            del ckpt

        self._vae.load_weights(list(weights.items()))
        print("[MLX] VAE loaded.")

    def _load_geo(self):
        if self._geo is not None:
            return
        print("[MLX] Loading geo decoder...")
        self._geo = CrossAttentionDecoder(
            num_latents=4096, out_channels=1,
            width=1024, heads=16,
            mlp_expand_ratio=4,
            num_freqs=8, include_pi=False,
            qkv_bias=False, qk_norm=True,
            enable_ln_post=True,
            downsample_ratio=1,
        )

        st_path = os.path.join(self.vae_dir, 'model.safetensors')
        if os.path.exists(st_path):
            raw = load_safetensors(st_path)
            geo_weights = {k: v for k, v in raw.items() if k.startswith('geo_decoder.')}
            weights = remap_geo_decoder_weights(geo_weights)
        else:
            import torch
            ckpt_path = os.path.join(self.vae_dir, 'model.fp16.ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            geo_state = {k: v for k, v in ckpt.items() if k.startswith('geo_decoder.')}
            raw_mx = {k: mx.array(v.float().numpy()) for k, v in geo_state.items()}
            weights = remap_geo_decoder_weights(raw_mx)
            del ckpt

        self._geo.load_weights(list(weights.items()))
        print("[MLX] Geo decoder loaded.")

    def _unload(self, *names):
        """Unload models to free memory."""
        for name in names:
            attr = f'_{name}'
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)
        _clear_gpu()

    def _volume_decode(
        self,
        decoded_latents: mx.array,
        *,
        bounds=1.01,
        num_chunks: int = 8000,
        octree_resolution: int = 384,
        enable_pbar: bool = True,
    ) -> mx.array:
        """Compatibility wrapper around the upstream-owned volume decoder."""
        decoded_latents_torch = torch.from_numpy(np.array(decoded_latents)).to(dtype=torch.float32)
        grid_logits_torch = self.vae.volume_decoder(
            decoded_latents_torch,
            self.vae.geo_decoder,
            bounds=bounds,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            enable_pbar=enable_pbar,
        )
        grid_logits = mx.array(grid_logits_torch.detach().cpu().numpy())
        mx.eval(grid_logits)
        return grid_logits.astype(mx.float32)

    def _latents2mesh(
        self,
        decoded_latents: mx.array,
        *,
        bounds=1.01,
        mc_level: float = 0.0,
        num_chunks: int = 8000,
        octree_resolution: int = 384,
        mc_algo: str = "mc",
        enable_pbar: bool = True,
    ):
        """Compatibility wrapper around the upstream-owned ShapeVAE facade."""
        if mc_algo is None:
            mc_algo = "mc"

        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()
        decoded_latents_torch = torch.from_numpy(np.array(decoded_latents)).to(dtype=torch.float32)
        return self.vae.latents2mesh(
            decoded_latents_torch,
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )

    def _export(
        self,
        latents,
        output_type: str = "trimesh",
        box_v: float = 1.01,
        mc_level: float = 0.0,
        num_chunks: int = 8000,
        octree_resolution: int = 384,
        mc_algo: str = "mc",
        enable_pbar: bool = True,
    ):
        """Thin trace/unload wrapper around the upstream export path."""
        if output_type != "latent":
            self._trace_save("sampled_latents_mlx", latents)
        outputs = super()._export(
            latents,
            output_type,
            box_v,
            mc_level,
            num_chunks,
            octree_resolution,
            mc_algo,
            enable_pbar,
        )
        if self.lazy_load and output_type != "latent":
            self._unload('vae', 'geo')
            _clear_gpu()
        return outputs

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        del replace_vae
        self.vae.enable_flashvdm_decoder(
            enabled=enabled,
            adaptive_kv_selection=adaptive_kv_selection,
            topk_mode=topk_mode,
            mc_algo=mc_algo,
        )

    def __call__(self, image, num_inference_steps=20, guidance_scale=5.0,
                 seed=42, octree_resolution=384, num_chunks=8000, **kwargs):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return super().__call__(
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            **kwargs,
        )


MlxShapePipeline = MlxHunyuan3DDiTFlowMatchingPipeline
