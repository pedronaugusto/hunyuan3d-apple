"""
Attention processors for Hunyuan3D texture UNet in MLX.
Ported from hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py

Three processor classes:
  - SelfAttnProcessor2_0: Material-aware self-attention (separate Q/K/V per PBR material)
  - RefAttnProcessor2_0: Reference attention (shared Q/K, separate V per material)
  - PoseRoPEAttnProcessor2_0: Multiview attention with 3D rotary position embeddings

Plus RoPE utilities (get_1d_rotary_pos_embed, get_3d_rotary_pos_embed, apply_rotary_emb).

Design:
  - Callers pass the base attention projections (attn_q, attn_k, attn_v, attn_out)
    as arguments so the processors can reuse them for the albedo branch.
  - All computation in float32 (no half-precision management needed on MLX).
  - Uses mx.fast.scaled_dot_product_attention for the attention kernel.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Rotary Position Embedding utilities
# ---------------------------------------------------------------------------

def get_1d_rotary_pos_embed(
    dim: int, pos: mx.array, theta: float = 10000.0,
    linear_factor: float = 1.0, ntk_factor: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Compute 1D rotary position embeddings.

    Args:
        dim: Embedding dimension (must be even).
        pos: Position indices, shape (L,).
        theta: Base frequency.
        linear_factor: Scales frequency linearly (upstream default 1.0).
        ntk_factor: Scales base theta for NTK-aware interpolation (upstream default 1.0).

    Returns:
        (cos_emb, sin_emb) each of shape (L, dim).
    """
    assert dim % 2 == 0
    half = dim // 2
    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[:half] / dim)) / linear_factor
    # (L, half)
    freqs = pos[:, None] * freqs[None, :]
    cos_emb = mx.cos(freqs)
    sin_emb = mx.sin(freqs)
    # repeat_interleave(2, dim=1): [c0,c0,c1,c1,...]
    cos_emb = mx.repeat(cos_emb, 2, axis=1)
    sin_emb = mx.repeat(sin_emb, 2, axis=1)
    return cos_emb, sin_emb


def get_3d_rotary_pos_embed(
    position: mx.array, embed_dim: int, voxel_resolution: int, theta: int = 10000
) -> Tuple[mx.array, mx.array]:
    """Compute 3D rotary position embeddings for spatial coordinates.

    Args:
        position: Integer voxel indices, shape (..., 3).
        embed_dim: Total embedding dimension.
        voxel_resolution: Voxel grid size (positions index into 0..voxel_resolution-1).
        theta: Base frequency.

    Returns:
        (cos, sin) each of shape (..., embed_dim).
    """
    assert position.shape[-1] == 3
    dim_xy = embed_dim // 8 * 3
    dim_z = embed_dim // 8 * 2

    grid = mx.arange(voxel_resolution, dtype=mx.float32)
    freqs_xy = get_1d_rotary_pos_embed(dim_xy, grid, theta)  # (voxel_res, dim_xy)
    freqs_z = get_1d_rotary_pos_embed(dim_z, grid, theta)    # (voxel_res, dim_z)

    xy_cos, xy_sin = freqs_xy
    z_cos, z_sin = freqs_z

    flat = position.reshape(-1, 3).astype(mx.int32)  # (N, 3)
    x_cos = xy_cos[flat[:, 0]]
    x_sin = xy_sin[flat[:, 0]]
    y_cos = xy_cos[flat[:, 1]]
    y_sin = xy_sin[flat[:, 1]]
    zc = z_cos[flat[:, 2]]
    zs = z_sin[flat[:, 2]]

    cos = mx.concatenate([x_cos, y_cos, zc], axis=-1)  # (N, embed_dim)
    sin = mx.concatenate([x_sin, y_sin, zs], axis=-1)

    cos = cos.reshape(*position.shape[:-1], embed_dim)
    sin = sin.reshape(*position.shape[:-1], embed_dim)
    return cos, sin


def apply_rotary_emb(
    x: mx.array, freqs_cis: Tuple[mx.array, mx.array]
) -> mx.array:
    """Apply rotary position embeddings.

    The upstream PyTorch implementation does:
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = stack([-x_imag, x_real], dim=-1).flatten(3)
        out = x * cos + x_rotated * sin

    Args:
        x: Input tensor, shape (B, H, L, D).
        freqs_cis: Tuple of (cos, sin).
            - (L, D) broadcasts to all batches/heads.
            - (B, L, D) broadcasts to all heads.

    Returns:
        Tensor with RoPE applied, same shape as x.
    """
    cos, sin = freqs_cis
    # Expand cos/sin for broadcasting with (B, H, L, D)
    if cos.ndim == 2:  # (L, D) -> (1, 1, L, D)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.ndim == 3:  # (B, L, D) -> (B, 1, L, D)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    # Rotate pairs: for consecutive (x0, x1), compute (-x1, x0) * sin + (x0, x1) * cos
    x_r = x.reshape(*x.shape[:-1], -1, 2)  # (..., D//2, 2)
    x_real = x_r[..., 0]  # (..., D//2)
    x_imag = x_r[..., 1]
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)

    return x * cos + x_rotated * sin


# ---------------------------------------------------------------------------
# Attention helper: multi-head reshape + SDPA
# ---------------------------------------------------------------------------

def _reshape_for_heads(t: mx.array, batch: int, seq: int, heads: int, dim_head: int) -> mx.array:
    """Reshape (batch, seq, inner_dim) -> (batch, heads, seq, dim_head)."""
    return t.reshape(batch, seq, heads, dim_head).transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# SelfAttnProcessor2_0 -- material-aware self-attention
# ---------------------------------------------------------------------------

class SelfAttnProcessor2_0(nn.Module):
    """Material-aware self-attention with separate Q/K/V per PBR material.

    The albedo branch reuses the caller-provided base attention projections
    (attn_q, attn_k, attn_v, attn_out). Additional materials (e.g. "mr")
    get their own projection weights stored in this module.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        pbr_setting: Optional[List[str]] = None,
    ):
        super().__init__()
        self.pbr_setting = pbr_setting or ["albedo", "mr"]
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        # Albedo uses the base attention's projections (passed at call time).
        # Additional materials get dedicated projections.
        for token in self.pbr_setting:
            if token == "albedo":
                continue
            setattr(self, f"to_q_{token}", nn.Linear(query_dim, inner_dim, bias=bias))
            setattr(self, f"to_k_{token}", nn.Linear(query_dim, inner_dim, bias=bias))
            setattr(self, f"to_v_{token}", nn.Linear(query_dim, inner_dim, bias=bias))
            setattr(self, f"to_out_{token}", nn.Linear(inner_dim, query_dim))

    def __call__(
        self,
        hidden_states: mx.array,
        attn_q: nn.Linear,
        attn_k: nn.Linear,
        attn_v: nn.Linear,
        attn_out: nn.Linear,
        attn_norm_q: Optional[nn.Module] = None,
        attn_norm_k: Optional[nn.Module] = None,
    ) -> mx.array:
        """Process material-aware self-attention.

        Args:
            hidden_states: (B, N_pbr, N, L, C) -- pre-shaped by the caller,
                where N_pbr = number of PBR materials, N = number of views,
                L = spatial tokens, C = channels.
            attn_q/k/v/out: Base attention projections (used for the albedo branch).
            attn_norm_q/k: Optional QK normalization layers.

        Returns:
            (B, N_pbr, N, L, C) with attention applied per-material.
        """
        B, N_pbr, N, L, C = hidden_states.shape
        H, D = self.heads, self.dim_head
        scale = D ** -0.5
        results = []

        for i, token in enumerate(self.pbr_setting):
            hs = hidden_states[:, i]           # (B, N, L, C)
            hs = hs.reshape(B * N, L, C)       # merge batch and view dims

            if token == "albedo":
                q = attn_q(hs)
                k = attn_k(hs)
                v = attn_v(hs)
            else:
                q = getattr(self, f"to_q_{token}")(hs)
                k = getattr(self, f"to_k_{token}")(hs)
                v = getattr(self, f"to_v_{token}")(hs)

            # Multi-head reshape: (B*N, L, H*D) -> (B*N, H, L, D)
            q = _reshape_for_heads(q, B * N, L, H, D)
            k = _reshape_for_heads(k, B * N, L, H, D)
            v = _reshape_for_heads(v, B * N, L, H, D)

            if attn_norm_q is not None:
                q = attn_norm_q(q)
                k = attn_norm_k(k)

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            # (B*N, H, L, D) -> (B*N, L, H*D)
            out = out.transpose(0, 2, 1, 3).reshape(B * N, L, H * D)

            if token == "albedo":
                out = attn_out(out)
            else:
                out = getattr(self, f"to_out_{token}")(out)

            results.append(out.reshape(B, N, L, C))

        return mx.stack(results, axis=1)  # (B, N_pbr, N, L, C)


# ---------------------------------------------------------------------------
# RefAttnProcessor2_0 -- reference attention with shared Q/K
# ---------------------------------------------------------------------------

class RefAttnProcessor2_0(nn.Module):
    """Reference attention: shared Q/K, separate V per PBR material.

    Query and key are computed once (from the base projections), while each
    material branch has its own value projection + output linear. This is
    efficient for cross-attending to shared reference features while
    producing material-specific outputs.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        pbr_setting: Optional[List[str]] = None,
    ):
        super().__init__()
        self.pbr_setting = pbr_setting or ["albedo", "mr"]
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        # Only V and out projections for non-albedo materials
        for token in self.pbr_setting:
            if token == "albedo":
                continue
            setattr(self, f"to_v_{token}", nn.Linear(query_dim, inner_dim, bias=bias))
            setattr(self, f"to_out_{token}", nn.Linear(inner_dim, query_dim))

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attn_q: nn.Linear,
        attn_k: nn.Linear,
        attn_v: nn.Linear,
        attn_out: nn.Linear,
        attn_norm_q: Optional[nn.Module] = None,
        attn_norm_k: Optional[nn.Module] = None,
    ) -> mx.array:
        """Process reference attention.

        Args:
            hidden_states: (B, N*L, C) -- flattened across views.
            encoder_hidden_states: (B, N_ref*L, C) -- reference features.
            attn_q/k/v/out: Base attention projections (albedo branch).
            attn_norm_q/k: Optional QK normalization layers.

        Returns:
            (B, N_pbr, N*L, C) -- one output per PBR material, stacked.
        """
        B, NL, C = hidden_states.shape
        H, D = self.heads, self.dim_head
        scale = D ** -0.5
        ref_len = encoder_hidden_states.shape[1]

        # Shared Q and K (computed once)
        q = attn_q(hidden_states)                          # (B, NL, H*D)
        k = attn_k(encoder_hidden_states)                  # (B, ref_len, H*D)

        q = _reshape_for_heads(q, B, NL, H, D)            # (B, H, NL, D)
        k = _reshape_for_heads(k, B, ref_len, H, D)       # (B, H, ref_len, D)

        if attn_norm_q is not None:
            q = attn_norm_q(q)
            k = attn_norm_k(k)

        # Per-material SDPA: mx.fast.scaled_dot_product_attention gives
        # wrong results when V has a different last dim than Q/K (same bug
        # as MPS PyTorch). Run one SDPA per material with shared Q/K.
        results = []
        for token in self.pbr_setting:
            if token == "albedo":
                v_proj = attn_v(encoder_hidden_states)
            else:
                v_proj = getattr(self, f"to_v_{token}")(encoder_hidden_states)
            v = _reshape_for_heads(v_proj, B, ref_len, H, D)

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            out_proj = out.transpose(0, 2, 1, 3).reshape(B, NL, H * D)
            if token == "albedo":
                out_proj = attn_out(out_proj)
            else:
                out_proj = getattr(self, f"to_out_{token}")(out_proj)
            results.append(out_proj)

        return mx.stack(results, axis=1)  # (B, N_pbr, NL, C)



# ---------------------------------------------------------------------------
# PoseRoPEAttnProcessor2_0 -- multiview attention with 3D RoPE
# ---------------------------------------------------------------------------

class PoseRoPEAttnProcessor2_0:
    """Multiview attention with 3D rotary position embeddings.

    This is a plain callable (not nn.Module) -- it has no learnable parameters.
    RoPE is applied to Q and K before the attention computation. The 3D RoPE
    embeddings are derived from voxel position indices and cached in the
    position_indices dict (keyed by head_dim) for reuse across layers.
    """

    def __init__(self, num_heads: int, dim_head: int):
        self.num_heads = num_heads
        self.dim_head = dim_head

    def __call__(
        self,
        hidden_states: mx.array,
        attn_q: nn.Linear,
        attn_k: nn.Linear,
        attn_v: nn.Linear,
        attn_out: nn.Linear,
        attn_norm_q: Optional[nn.Module] = None,
        attn_norm_k: Optional[nn.Module] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        position_indices: Optional[Dict] = None,
        n_pbrs: int = 1,
    ) -> mx.array:
        """Process multiview attention with 3D RoPE.

        Args:
            hidden_states: (B_eff, N*L, C) where B_eff = B * n_pbrs.
            attn_q/k/v/out: Base attention projections.
            attn_norm_q/k: Optional QK normalization layers.
            encoder_hidden_states: Optional cross-attention context.
                If None, self-attention is performed.
            position_indices: Dict with keys:
                - "voxel_indices": (B, L, 3) integer voxel coordinates
                - "voxel_resolution": int, grid size
                - <head_dim>: cached (cos, sin) tuple (filled on first call)
            n_pbrs: Number of PBR materials (used to tile voxel indices).

        Returns:
            (B_eff, N*L, C) attention output.
        """
        B, NL, C = hidden_states.shape
        H, D = self.num_heads, self.dim_head
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        kv_len = kv_input.shape[1]

        q = attn_q(hidden_states)        # (B, NL, H*D)
        k = attn_k(kv_input)             # (B, kv_len, H*D)
        v = attn_v(kv_input)             # (B, kv_len, H*D)

        q = _reshape_for_heads(q, B, NL, H, D)
        k = _reshape_for_heads(k, B, kv_len, H, D)
        v = _reshape_for_heads(v, B, kv_len, H, D)

        if attn_norm_q is not None:
            q = attn_norm_q(q)
            k = attn_norm_k(k)

        # Apply 3D RoPE if position indices are provided
        # Skip when head_dim not divisible by 8 — the 3D RoPE formula requires it
        if position_indices is not None and D % 8 == 0:
            if D in position_indices:
                rope_emb = position_indices[D]
            else:
                # Tile voxel indices across PBR materials:
                # (B, L, 3) -> (B * n_pbrs, L, 3)
                voxel_idx = position_indices["voxel_indices"]
                if n_pbrs > 1:
                    voxel_idx = mx.tile(voxel_idx[:, None], (1, n_pbrs, 1, 1))
                    voxel_idx = voxel_idx.reshape(-1, *voxel_idx.shape[2:])
                rope_emb = get_3d_rotary_pos_embed(
                    voxel_idx, D, voxel_resolution=position_indices["voxel_resolution"]
                )
                position_indices[D] = rope_emb

            q = apply_rotary_emb(q, rope_emb)
            k = apply_rotary_emb(k, rope_emb)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        # (B, H, NL, D) -> (B, NL, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, NL, H * D)
        return attn_out(out)
