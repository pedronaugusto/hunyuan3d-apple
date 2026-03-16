"""
UNet2p5DConditionModel — MLX port of the multiview PBR texture UNet.
Ported from hy3dpaint/hunyuanpaintpbr/unet/modules.py

This wraps a standard UNet2DConditionModel with:
- Material-Dimensional Attention (MDA) via SelfAttnProcessor2_0
- Multiview Attention (MA) via PoseRoPEAttnProcessor2_0
- Reference Attention (RA) via RefAttnProcessor2_0
- DINO feature conditioning
- Dual-stream reference feature extraction
"""
import json
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .norm import PreciseLayerNorm

from .unet_blocks import (
    UNet2DConditionModel,
    Timesteps,
    TimestepEmbedding,
)
from .unet_attention import (
    SelfAttnProcessor2_0,
    RefAttnProcessor2_0,
    PoseRoPEAttnProcessor2_0,
    get_3d_rotary_pos_embed,
)


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
    if hasattr(value, "shape") and hasattr(value, "dtype"):
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
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.array(value)
    else:
        return
    np.save(os.path.join(trace_dir, name), arr)


class ImageProjModel(nn.Module):
    """Projects DINO image embeddings into cross-attention space.
    Linear(clip_dim → extra_tokens * cross_attn_dim) + LayerNorm."""

    def __init__(self, cross_attention_dim: int = 1024,
                 clip_embeddings_dim: int = 1536,
                 clip_extra_context_tokens: int = 4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim,
                              clip_extra_context_tokens * cross_attention_dim)
        self.norm = PreciseLayerNorm(cross_attention_dim)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        """
        image_embeds: (B, N_tokens, clip_dim) or (B, clip_dim)
        Returns: (B, N_tokens * extra_tokens, cross_attn_dim)
        """
        if image_embeds.ndim == 2:
            embeds = image_embeds[:, None, :]  # (B, 1, D)
        else:
            embeds = image_embeds  # (B, N, D)

        B, N, D = embeds.shape
        embeds_flat = embeds.reshape(B * N, D)
        tokens = self.proj(embeds_flat).reshape(
            B * N, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        tokens = self.norm(tokens)
        tokens = tokens.reshape(B, N * self.clip_extra_context_tokens,
                                self.cross_attention_dim)
        return tokens


def compute_discrete_voxel_indices(position_maps: mx.array,
                                    grid_resolution: int = 8,
                                    voxel_resolution: int = 128) -> mx.array:
    """Quantize position maps to discrete voxel indices.

    Args:
        position_maps: (B, N, H, W, 3) position maps in [0,1] — NHWC format
        grid_resolution: spatial downsampling factor
        voxel_resolution: quantization resolution

    Returns:
        (B, N*grid_res*grid_res, 3) integer voxel indices
    """
    position_np = np.array(position_maps, dtype=np.float32)
    B, N, H, W, C = position_np.shape
    assert C == 3
    assert H % grid_resolution == 0 and W % grid_resolution == 0

    # Direct numpy translation of the upstream PyTorch implementation.
    position = np.transpose(position_np, (0, 1, 4, 2, 3)).astype(np.float16)  # (B, N, C, H, W)
    valid_mask = np.all(position != 1.0, axis=2, keepdims=True)
    valid_mask = np.broadcast_to(valid_mask, position.shape)
    position = np.where(valid_mask, position, 0.0)

    grid_h = H // grid_resolution
    grid_w = W // grid_resolution
    position = position.reshape(B, N, C, grid_resolution, grid_h, grid_resolution, grid_w)
    position = np.transpose(position, (0, 1, 3, 5, 2, 4, 6))

    valid_mask = valid_mask.reshape(B, N, C, grid_resolution, grid_h, grid_resolution, grid_w)
    valid_mask = np.transpose(valid_mask, (0, 1, 3, 5, 2, 4, 6))

    grid_position = position.sum(axis=(-2, -1))
    count_masked = valid_mask.sum(axis=(-2, -1))
    grid_position = grid_position / np.maximum(count_masked, 1.0)

    voxel_mask_thres = grid_h * grid_w // (4 * 4)
    grid_position = np.where(count_masked < voxel_mask_thres, 0.0, grid_position)

    grid_position = np.clip(np.transpose(grid_position, (0, 1, 4, 2, 3)), 0.0, 1.0)
    voxel_indices = np.round(grid_position * (voxel_resolution - 1)).astype(np.int32)
    voxel_indices = np.transpose(voxel_indices, (0, 1, 3, 4, 2)).reshape(B, N * grid_resolution * grid_resolution, 3)
    return mx.array(voxel_indices)


def calc_multires_voxel_indices(position_maps: mx.array,
                                 grid_resolutions: list,
                                 voxel_resolutions: list) -> dict:
    """Generate multi-resolution voxel indices for position encoding.

    Returns dict keyed by flattened sequence length, values are
    {'voxel_indices': (B, L, 3), 'voxel_resolution': int}.
    """
    import torch
    from hy3dpaint.hunyuanpaintpbr.unet.modules import calc_multires_voxel_idxs

    # The upstream compute_discrete_voxel_indice calls position.half(), and
    # when position is already float16 this is a no-op — so subsequent
    # in-place modifications (position[mask] = 0) corrupt the data for
    # later grid resolutions.  This corruption is *intentional* upstream
    # behaviour that the model was trained with: background pixels set to 0
    # in the first iteration bleed into later (coarser) voxel grids.
    # Passing float32 "fixes" the corruption but gives voxel indices the
    # model never saw during training, breaking spatial texture mapping.
    # We must replicate the upstream float16 path exactly.
    pos_t = torch.from_numpy(np.array(position_maps, dtype=np.float16)).permute(0, 1, 4, 2, 3)
    upstream = calc_multires_voxel_idxs(
        pos_t,
        grid_resolutions=grid_resolutions,
        voxel_resolutions=voxel_resolutions,
    )
    return {
        key: {
            "voxel_indices": mx.array(value["voxel_indices"].cpu().numpy().astype(np.int32)),
            "voxel_resolution": value["voxel_resolution"],
        }
        for key, value in upstream.items()
    }


class Basic2p5DTransformerBlock(nn.Module):
    """Enhanced transformer block with MDA, MA, RA, and DINO attention.

    Wraps a standard BasicTransformerBlock and adds specialized attention.
    """

    def __init__(self, base_block, layer_name: str,
                 use_ma: bool = True, use_ra: bool = True,
                 use_mda: bool = True, use_dino: bool = True,
                 pbr_setting: list = None):
        super().__init__()
        self.base = base_block
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.use_mda = use_mda
        self.use_dino = use_dino
        self.pbr_setting = pbr_setting or ["albedo", "mr"]

        dim = base_block.dim
        num_heads = base_block.num_heads
        dim_head = dim // num_heads
        cross_attn_dim = base_block.cross_attention_dim

        if use_mda:
            self.mda_processor = SelfAttnProcessor2_0(
                query_dim=dim, heads=num_heads, dim_head=dim_head,
                pbr_setting=self.pbr_setting,
            )

        if use_ma:
            # Multiview self-attention with RoPE
            self.ma_to_q = nn.Linear(dim, dim, bias=False)
            self.ma_to_k = nn.Linear(dim, dim, bias=False)
            self.ma_to_v = nn.Linear(dim, dim, bias=False)
            self.ma_to_out = nn.Linear(dim, dim)
            self.ma_norm_q = None
            self.ma_norm_k = None
            self.ma_processor = PoseRoPEAttnProcessor2_0(num_heads, dim_head)

        if use_ra:
            self.ra_to_q = nn.Linear(dim, dim, bias=False)
            self.ra_to_k = nn.Linear(dim, dim, bias=False)
            self.ra_to_v = nn.Linear(dim, dim, bias=False)
            self.ra_to_out = nn.Linear(dim, dim)
            self.ra_processor = RefAttnProcessor2_0(
                query_dim=dim, heads=num_heads, dim_head=dim_head,
                pbr_setting=self.pbr_setting,
            )

        if use_dino:
            self.dino_to_q = nn.Linear(dim, dim, bias=False)
            self.dino_to_k = nn.Linear(cross_attn_dim, dim, bias=False)
            self.dino_to_v = nn.Linear(cross_attn_dim, dim, bias=False)
            self.dino_to_out = nn.Linear(dim, dim)

    def __call__(self, hidden_states: mx.array,
                 context: mx.array = None,
                 cross_attention_kwargs: dict = None) -> mx.array:
        """Called from Transformer2DModel with standard interface.

        cross_attention_kwargs dict should contain:
            mode, num_in_batch, mva_scale, ref_scale,
            condition_embed_dict, dino_hidden_states, position_voxel_indices
        """
        encoder_hidden_states = context
        kwargs = cross_attention_kwargs or {}
        mode = kwargs.get("mode", "r")
        num_in_batch = kwargs.get("num_in_batch", 1)
        mva_scale = kwargs.get("mva_scale", 1.0)
        ref_scale = kwargs.get("ref_scale", 1.0)
        condition_embed_dict = kwargs.get("condition_embed_dict")
        dino_hidden_states = kwargs.get("dino_hidden_states")
        position_voxel_indices = kwargs.get("position_voxel_indices")

        B_total, L, C = hidden_states.shape
        N_pbr = len(self.pbr_setting)
        N = num_in_batch

        # Norm 1
        norm_hs = self.base.norm1(hidden_states)

        # 1. Material-Dimensional Self-Attention (MDA)
        # Skip MDA in write mode (ref stream doesn't have N_pbr batches)
        if self.use_mda and "w" not in kwargs.get("mode", "r"):
            B = B_total // (N_pbr * N)
            mda_input = norm_hs.reshape(B, N_pbr, N, L, C)
            attn_out = self.mda_processor(
                mda_input,
                self.base.attn1.to_q, self.base.attn1.to_k,
                self.base.attn1.to_v, self.base.attn1.to_out,
            )
            attn_out = attn_out.reshape(B_total, L, C)
        else:
            q = self.base.attn1.to_q(norm_hs)
            k = self.base.attn1.to_k(norm_hs)
            v = self.base.attn1.to_v(norm_hs)
            H = self.base.num_heads
            D = C // H
            q = q.reshape(B_total, L, H, D).transpose(0, 2, 1, 3)
            k = k.reshape(B_total, L, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(B_total, L, H, D).transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
            attn_out = out.transpose(0, 2, 1, 3).reshape(B_total, L, C)
            attn_out = self.base.attn1.to_out(attn_out)

        hidden_states = hidden_states + attn_out

        # 1.2 Reference Attention (RA)
        if "w" in mode and condition_embed_dict is not None:
            # Write mode: store features for reference
            B = B_total // N
            condition_embed_dict[self.layer_name] = norm_hs.reshape(
                B, N * L, C
            )

        if "r" in mode and self.use_ra and condition_embed_dict is not None:
            condition_embed = condition_embed_dict.get(self.layer_name)
            if condition_embed is not None:
                B = B_total // (N_pbr * N)
                # Use only albedo features for ref attention
                ref_hs = norm_hs.reshape(B, N_pbr, N, L, C)[:, 0]  # (B, N, L, C)
                ref_hs = ref_hs.reshape(B, N * L, C)
                ra_out = self.ra_processor(
                    ref_hs, condition_embed,
                    self.ra_to_q, self.ra_to_k,
                    self.ra_to_v, self.ra_to_out,
                )  # (B, N_pbr, N*L, C)
                ra_out = ra_out.reshape(B, N_pbr, N, L, C)
                if hasattr(ref_scale, "shape"):
                    ref_scale_arr = mx.array(ref_scale, dtype=ra_out.dtype)
                    if ref_scale_arr.ndim == 0:
                        hidden_states = hidden_states + ref_scale_arr * ra_out.reshape(B_total, L, C)
                    else:
                        ref_scale_arr = ref_scale_arr.reshape(B, 1, 1, 1, 1)
                        hidden_states = hidden_states + (ref_scale_arr * ra_out).reshape(B_total, L, C)
                else:
                    hidden_states = hidden_states + ref_scale * ra_out.reshape(B_total, L, C)

        # 1.3 Multiview Attention (MA)
        if N > 1 and self.use_ma:
            B_pbr = B_total // N  # B * N_pbr
            mv_hs = norm_hs.reshape(B_pbr, N * L, C)

            position_indices = None
            if position_voxel_indices is not None:
                seq_len = N * L
                if seq_len in position_voxel_indices:
                    position_indices = position_voxel_indices[seq_len]

            ma_out = self.ma_processor(
                mv_hs,
                self.ma_to_q, self.ma_to_k,
                self.ma_to_v, self.ma_to_out,
                position_indices=position_indices,
                n_pbrs=N_pbr,
            )
            ma_out = ma_out.reshape(B_total, L, C)
            hidden_states = hidden_states + mva_scale * ma_out

        # 2. Cross-Attention to text
        # Compute norm2 once before cross-attn; reused by DINO (matches upstream)
        norm_hs2 = self.base.norm2(hidden_states)
        if self.base.attn2 is not None and encoder_hidden_states is not None:
            H = self.base.num_heads
            D = C // H

            q = self.base.attn2.to_q(norm_hs2)
            k = self.base.attn2.to_k(encoder_hidden_states)
            v = self.base.attn2.to_v(encoder_hidden_states)

            _, S, _ = encoder_hidden_states.shape
            q = q.reshape(B_total, L, H, D).transpose(0, 2, 1, 3)
            k = k.reshape(B_total, S, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(B_total, S, H, D).transpose(0, 2, 1, 3)

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
            attn_out = out.transpose(0, 2, 1, 3).reshape(B_total, L, C)
            attn_out = self.base.attn2.to_out(attn_out)
            hidden_states = hidden_states + attn_out

        # 2.5 DINO attention (reuse norm_hs2 from before cross-attn, matching upstream)
        if self.use_dino and dino_hidden_states is not None:
            # Broadcast DINO features to all views and materials
            B = B_total // (N_pbr * N)
            dino_hs = mx.broadcast_to(
                dino_hidden_states[:, None, :, :],
                (B, N_pbr * N, dino_hidden_states.shape[1], dino_hidden_states.shape[2])
            ).reshape(B_total, -1, dino_hidden_states.shape[-1])

            H = self.base.num_heads
            D = C // H
            q = self.dino_to_q(norm_hs2)
            k = self.dino_to_k(dino_hs)
            v = self.dino_to_v(dino_hs)

            S_dino = dino_hs.shape[1]
            q = q.reshape(B_total, L, H, D).transpose(0, 2, 1, 3)
            k = k.reshape(B_total, S_dino, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(B_total, S_dino, H, D).transpose(0, 2, 1, 3)

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
            attn_out = out.transpose(0, 2, 1, 3).reshape(B_total, L, C)
            attn_out = self.dino_to_out(attn_out)
            hidden_states = hidden_states + attn_out

        # 3. Feed-forward
        norm_hs3 = self.base.norm3(hidden_states)
        ff_out = self.base.ff(norm_hs3)
        hidden_states = hidden_states + ff_out

        return hidden_states


class MlxUNet2p5D(nn.Module):
    """Full UNet2.5D for multiview PBR texture generation.

    Wraps a standard UNet2DConditionModel and enhances transformer blocks
    with MDA, MA, RA, and DINO attention mechanisms.
    """

    def __init__(self, unet: UNet2DConditionModel,
                 pbr_setting: list = None,
                 use_dino: bool = True):
        super().__init__()
        self.unet = unet
        self.pbr_setting = pbr_setting or ["albedo", "mr"]
        self.use_dino = use_dino
        self.use_ma = True
        self.use_ra = True
        self.use_mda = True
        self.use_position_rope = True

        # Learned text embeddings per material
        for token in self.pbr_setting:
            setattr(self, f"learned_text_clip_{token}",
                    mx.zeros((77, 1024)))
        self.learned_text_clip_ref = mx.zeros((77, 1024))

        # DINO projector
        if use_dino:
            self.image_proj_model_dino = ImageProjModel(
                cross_attention_dim=1024,
                clip_embeddings_dim=1536,
                clip_extra_context_tokens=4,
            )

        # Upstream keeps a separate 4-channel reference UNet and only widens the
        # main stream to 12 channels. Reusing the widened main UNet here breaks
        # reference-attention parity.
        self.unet_dual = self._make_dual_stream_unet()

        # Replace standard transformer blocks with 2.5D blocks. The dual
        # reference stream matches upstream: wrapped, but with all added 2.5D
        # branches disabled.
        self._wrap_transformer_blocks(
            self.unet,
            use_ma=self.use_ma,
            use_ra=self.use_ra,
            use_mda=self.use_mda,
            use_dino=self.use_dino,
            pbr_setting=self.pbr_setting,
        )
        self._wrap_transformer_blocks(
            self.unet_dual,
            use_ma=False,
            use_ra=False,
            use_mda=False,
            use_dino=False,
            pbr_setting=None,
        )

    def _make_dual_stream_unet(self) -> UNet2DConditionModel:
        """Build the upstream-style 4-channel dual reference UNet."""
        return UNet2DConditionModel(
            in_channels=4,
            out_channels=self.unet.out_channels,
            block_out_channels=[320, 640, 1280, 1280],
            layers_per_block=2,
            cross_attention_dim=1024,
            attention_head_dim=[5, 10, 20, 20],
        )

    def _wrap_transformer_blocks(self, model, use_ma: bool, use_ra: bool,
                                 use_mda: bool, use_dino: bool,
                                 pbr_setting: list | None):
        """Replace standard BasicTransformerBlock instances with Basic2p5DTransformerBlock."""
        from .unet_blocks import (
            CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn,
            Transformer2DModel,
        )

        block_idx = 0
        for block in model.down_blocks:
            if isinstance(block, CrossAttnDownBlock2D):
                for t2d in block.attentions:
                    for i, tb in enumerate(t2d.transformer_blocks):
                        name = f"down_{block_idx}"
                        t2d.transformer_blocks[i] = Basic2p5DTransformerBlock(
                            tb, layer_name=name,
                            use_ma=use_ma, use_ra=use_ra,
                            use_mda=use_mda, use_dino=use_dino,
                            pbr_setting=pbr_setting,
                        )
                        block_idx += 1

        for t2d in model.mid_block.attentions:
            for i, tb in enumerate(t2d.transformer_blocks):
                name = f"mid_{block_idx}"
                t2d.transformer_blocks[i] = Basic2p5DTransformerBlock(
                    tb, layer_name=name,
                    use_ma=use_ma, use_ra=use_ra,
                    use_mda=use_mda, use_dino=use_dino,
                    pbr_setting=pbr_setting,
                )
                block_idx += 1

        for block in model.up_blocks:
            if isinstance(block, CrossAttnUpBlock2D):
                for t2d in block.attentions:
                    for i, tb in enumerate(t2d.transformer_blocks):
                        name = f"up_{block_idx}"
                        t2d.transformer_blocks[i] = Basic2p5DTransformerBlock(
                            tb, layer_name=name,
                            use_ma=use_ma, use_ra=use_ra,
                            use_mda=use_mda, use_dino=use_dino,
                            pbr_setting=pbr_setting,
                        )
                        block_idx += 1

    def __call__(self, sample: mx.array, timestep: mx.array,
                 encoder_hidden_states: mx.array,
                 **cached_condition) -> mx.array:
        """
        Args:
            sample: (B, N_pbr, N_gen, C, H, W) noisy latents — NCHW per-view
            timestep: (B,) or scalar timestep
            encoder_hidden_states: (B, N_pbr, 77, 1024) text embeddings
            cached_condition: dict with embeds_normal, embeds_position,
                ref_latents, dino_hidden_states, position_maps, etc.

        Returns:
            (B*N_pbr*N_gen, C, H, W) noise prediction
        """
        if "cache" not in cached_condition:
            cached_condition["cache"] = {}

        def _as_mx_array(name: str, value) -> mx.array:
            if isinstance(value, np.ndarray):
                value = mx.array(value)
            elif not hasattr(value, "shape") or not hasattr(value, "dtype"):
                value = mx.array(value)
            return value

        B, N_pbr, N_gen, C, H, W = sample.shape
        _tracing = os.environ.get("HY3D_MLX_TEXTURE_INTERNAL_TRACE_DIR")
        if _tracing:
            _write_internal_trace(
                "unet2p5d_in.json",
                {
                    "sample": _tensor_stats(sample),
                    "timestep": _tensor_stats(timestep),
                    "encoder_hidden_states": _tensor_stats(encoder_hidden_states),
                    "cached_condition_keys": sorted(cached_condition.keys()),
                    "embeds_normal": _tensor_stats(cached_condition.get("embeds_normal")),
                    "embeds_position": _tensor_stats(cached_condition.get("embeds_position")),
                    "ref_latents": _tensor_stats(cached_condition.get("ref_latents")),
                    "dino_hidden_states": _tensor_stats(cached_condition.get("dino_hidden_states")),
                    "position_maps": _tensor_stats(cached_condition.get("position_maps")),
                },
            )
            _write_internal_array("unet_sample.npy", sample)
            _write_internal_array("unet_timestep.npy", timestep)
            _write_internal_array("unet_encoder_hidden_states.npy", encoder_hidden_states)
            _write_internal_array("unet_embeds_normal.npy", cached_condition.get("embeds_normal"))
            _write_internal_array("unet_embeds_position.npy", cached_condition.get("embeds_position"))
            _write_internal_array("unet_ref_latents.npy", cached_condition.get("ref_latents"))
            _write_internal_array("unet_dino_hidden_states.npy", cached_condition.get("dino_hidden_states"))
            _write_internal_array("unet_position_maps.npy", cached_condition.get("position_maps"))

        # Concat normal/position embeddings to sample channels
        sample = _as_mx_array("sample", sample)
        inputs = [sample]
        if "embeds_normal" in cached_condition:
            cached_condition["embeds_normal"] = _as_mx_array(
                "embeds_normal", cached_condition["embeds_normal"]
            )
            normal = mx.broadcast_to(
                cached_condition["embeds_normal"][:, None, :, :, :, :],
                (B, N_pbr, *cached_condition["embeds_normal"].shape[1:])
            )
            inputs.append(normal)
        if "embeds_position" in cached_condition:
            cached_condition["embeds_position"] = _as_mx_array(
                "embeds_position", cached_condition["embeds_position"]
            )
            pos = mx.broadcast_to(
                cached_condition["embeds_position"][:, None, :, :, :, :],
                (B, N_pbr, *cached_condition["embeds_position"].shape[1:])
            )
            inputs.append(pos)

        expected_shape = (B, N_pbr, N_gen, H, W)
        for idx, inp in enumerate(inputs):
            inp = _as_mx_array(f"inputs[{idx}]", inp)
            if inp.shape[:3] != expected_shape[:3] or inp.shape[-2:] != expected_shape[-2:]:
                raise ValueError(
                    f"UNet2p5D concat shape mismatch for inputs[{idx}]: "
                    f"got {inp.shape}, expected (B, N_pbr, N_gen, C, H, W) with "
                    f"B={B}, N_pbr={N_pbr}, N_gen={N_gen}, H={H}, W={W}"
                )
            inputs[idx] = inp

        sample = mx.concatenate(tuple(inputs), axis=3)  # concat on C dim
        if _tracing:
            _write_internal_trace(
                "unet2p5d_concat.json",
                {
                    "concat_sample": _tensor_stats(sample),
                },
            )
            _write_internal_array("unet_concat_sample.npy", sample)

        # Flatten to (B*N_pbr*N_gen, C_total, H, W)
        sample = sample.reshape(B * N_pbr * N_gen, -1, H, W)

        # Prepare text conditioning
        enc_hs = mx.broadcast_to(
            encoder_hidden_states[:, :, None, :, :],
            (B, N_pbr, N_gen, encoder_hidden_states.shape[2], encoder_hidden_states.shape[3])
        )
        S_txt, D_txt = encoder_hidden_states.shape[2], encoder_hidden_states.shape[3]
        enc_hs = enc_hs.reshape(B * N_pbr * N_gen, S_txt, D_txt)

        # Compute position voxel indices for RoPE
        position_voxel_indices = None
        cache = cached_condition["cache"]
        if self.use_position_rope and "position_maps" in cached_condition:
            if "position_voxel_indices" not in cache:
                pos_maps = cached_condition["position_maps"]  # (B, N, H, W, 3), keep in host memory
                position_voxel_indices = calc_multires_voxel_indices(
                    pos_maps,
                    grid_resolutions=[H, H // 2, H // 4, H // 8],
                    voxel_resolutions=[H * 8, H * 4, H * 2, H],
                )
                cache["position_voxel_indices"] = position_voxel_indices
            else:
                position_voxel_indices = cache["position_voxel_indices"]

        # Project DINO features
        dino_hidden_states = None
        if self.use_dino and "dino_hidden_states" in cached_condition:
            if "dino_hidden_states_proj" not in cache:
                dino_hs = _as_mx_array("dino_hidden_states", cached_condition["dino_hidden_states"])
                cached_condition["dino_hidden_states"] = dino_hs
                dino_hidden_states = self.image_proj_model_dino(dino_hs)
                cache["dino_hidden_states_proj"] = dino_hidden_states
            else:
                dino_hidden_states = cache["dino_hidden_states_proj"]

        # Reference feature extraction (dual-stream)
        condition_embed_dict = None
        if self.use_ra:
            if "condition_embed_dict" not in cache:
                condition_embed_dict = {}
                ref_latents = _as_mx_array("ref_latents", cached_condition["ref_latents"])
                cached_condition["ref_latents"] = ref_latents  # (B_ref, N_ref, C, H, W) — B_ref=1 always
                B_ref = ref_latents.shape[0]
                N_ref = ref_latents.shape[1]
                ref_flat = ref_latents.reshape(B_ref * N_ref, *ref_latents.shape[2:])

                ref_clip = self.learned_text_clip_ref
                ref_text = mx.broadcast_to(
                    ref_clip[None, :, :],
                    (B_ref * N_ref, ref_clip.shape[0], ref_clip.shape[1])
                )

                # Run dual UNet in write mode to populate condition_embed_dict
                unet_ref = self.unet_dual if self.unet_dual is not None else self.unet
                ref_cross_kwargs = {
                    "mode": "w",
                    "num_in_batch": N_ref,
                    "condition_embed_dict": condition_embed_dict,
                }
                # Forward the reference latents through the UNet.
                # The "w" mode Basic2p5DTransformerBlocks will write their
                # hidden states into condition_embed_dict keyed by layer name.
                ref_timestep = mx.zeros((B_ref * N_ref,), dtype=mx.int32)
                _ = unet_ref(ref_flat, ref_timestep, ref_text,
                             cross_attention_kwargs=ref_cross_kwargs)
                # Force eval to free ref UNet graph before main pass
                mx.eval(*condition_embed_dict.values())
                cache["condition_embed_dict"] = condition_embed_dict
            else:
                condition_embed_dict = cache["condition_embed_dict"]
        if _tracing:
            _write_internal_trace(
                "unet2p5d_conditioning.json",
                {
                    "enc_hs": _tensor_stats(enc_hs),
                    "dino_hidden_states": _tensor_stats(dino_hidden_states),
                    "condition_embed_dict": _tensor_stats(condition_embed_dict),
                    "position_voxel_indices_keys": (
                        sorted(str(k) for k in position_voxel_indices.keys())
                        if position_voxel_indices is not None else None
                    ),
                },
            )

        # Cross-attention kwargs for main forward
        cross_kwargs = {
            "mode": "r",
            "num_in_batch": N_gen,
            "dino_hidden_states": dino_hidden_states,
            "condition_embed_dict": condition_embed_dict,
            "mva_scale": cached_condition.get("mva_scale", 1.0),
            "ref_scale": cached_condition.get("ref_scale", 1.0),
            "position_voxel_indices": position_voxel_indices,
        }

        # Run main UNet
        out = self.unet(sample, timestep, enc_hs,
                        cross_attention_kwargs=cross_kwargs)
        if _tracing:
            _write_internal_trace(
                "unet2p5d_out.json",
                {
                    "out": _tensor_stats(out),
                },
            )
            _write_internal_array("unet_out.npy", out)
        return out
