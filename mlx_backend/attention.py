"""
Self-attention and cross-attention for Hunyuan DiT in MLX.
Ported from hy3dshape/hy3dshape/models/denoisers/hunyuandit.py

IMPORTANT: The upstream uses a non-standard cat-view-split pattern that
interleaves Q/K/V features across attention heads. The model weights were
trained with this pattern, so we must replicate it exactly.
"""
import mlx.core as mx
import mlx.nn as nn
from .norm import QKRMSNorm


class SelfAttention(nn.Module):
    """Self-attention with separate Q/K/V projections, QK-norm, and
    upstream-matching head interleaving."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 qk_norm: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        if qk_norm:
            self.q_norm = QKRMSNorm(self.head_dim)
            self.k_norm = QKRMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Upstream does: cat -> view(1, -1, H, 3D) -> split -> reshape(B, N, H, D).
        # Flattening batch and sequence together matters when CFG runs with B > 1.
        qkv = mx.concatenate([q, k, v], axis=-1)  # (B, N, 3*dim)
        qkv = qkv.reshape(1, B * N, H, 3 * D)
        q = qkv[:, :, :, :D].reshape(B, N, H, D)
        k = qkv[:, :, :, D:2 * D].reshape(B, N, H, D)
        v = qkv[:, :, :, 2 * D:].reshape(B, N, H, D)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention with upstream-matching K/V interleaving.
    Q uses straightforward reshape, K/V use cat-view-split interleaving."""

    def __init__(self, qdim: int, kdim: int, num_heads: int,
                 qkv_bias: bool = True, qk_norm: bool = False):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        self.head_dim = qdim // num_heads

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        if qk_norm:
            self.q_norm = QKRMSNorm(self.head_dim)
            self.k_norm = QKRMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        B, S1, _ = x.shape
        _, S2, _ = y.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x).reshape(B, S1, H, D)  # Q: simple reshape (upstream line 208)
        k = self.to_k(y)
        v = self.to_v(y)

        # Upstream does: cat -> view(1, -1, H, 2D) -> split -> reshape(B, S2, H, D).
        # This is batch-sensitive under batched CFG and must match exactly.
        kv = mx.concatenate([k, v], axis=-1)  # (B, S2, 2*qdim)
        kv = kv.reshape(1, B * S2, H, 2 * D)
        k = kv[:, :, :, :D].reshape(B, S2, H, D)
        v = kv[:, :, :, D:].reshape(B, S2, H, D)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, S1, self.qdim)
        return self.out_proj(out)
