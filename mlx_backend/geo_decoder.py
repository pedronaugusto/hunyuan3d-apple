"""
CrossAttentionDecoder + FourierEmbedder for occupancy prediction in MLX.
Ported from hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py

Processes query points in chunks against decoded latents to produce
occupancy values for marching cubes.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .norm import QKLayerNorm


class FourierEmbedder(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates."""

    def __init__(self, num_freqs: int = 8, input_dim: int = 3,
                 include_input: bool = True, include_pi: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Frequencies: 2^0, 2^1, ..., 2^(num_freqs-1)
        # Stored as plain list to avoid being treated as a model parameter
        freqs = (2.0 ** mx.arange(num_freqs, dtype=mx.float32))
        if include_pi:
            freqs = freqs * math.pi
        self._frequencies = freqs  # underscore prefix hides from parameters()

        temp = 1 if include_input or num_freqs == 0 else 0
        self.out_dim = input_dim * (num_freqs * 2 + temp)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (..., dim) -> (..., out_dim)"""
        if self.num_freqs > 0:
            # x[..., None] * freqs -> (..., dim, num_freqs)
            embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
            if self.include_input:
                return mx.concatenate([x, mx.sin(embed), mx.cos(embed)], axis=-1)
            else:
                return mx.concatenate([mx.sin(embed), mx.cos(embed)], axis=-1)
        return x


class GeoQKVCrossAttention(nn.Module):
    """QKV cross-attention for geo decoder."""

    def __init__(self, width: int, heads: int, data_width: int = None,
                 qkv_bias: bool = False, qk_norm: bool = True):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.data_width = data_width or width

        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        if qk_norm:
            self.q_norm = QKLayerNorm(self.head_dim)
            self.k_norm = QKLayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array, data: mx.array) -> mx.array:
        B, N_q, _ = x.shape
        _, N_kv, _ = data.shape
        H, D = self.heads, self.head_dim

        q = self.c_q(x).reshape(B, N_q, H, D)
        kv = self.c_kv(data).reshape(B, N_kv, H, 2 * D)
        k, v = mx.split(kv, 2, axis=-1)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N_q, -1)
        return self.c_proj(out)


class GeoMLP(nn.Module):
    """MLP for geo decoder blocks."""

    def __init__(self, width: int, expand_ratio: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention block for geo decoder."""

    def __init__(self, width: int, heads: int, data_width: int = None,
                 mlp_expand_ratio: int = 4,
                 qkv_bias: bool = False, qk_norm: bool = True):
        super().__init__()
        data_width = data_width or width
        self.attn = GeoQKVCrossAttention(width, heads, data_width,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.ln_2 = nn.LayerNorm(data_width, eps=1e-6)
        self.ln_3 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = GeoMLP(width, expand_ratio=mlp_expand_ratio)

    def __call__(self, x: mx.array, data: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class CrossAttentionDecoder(nn.Module):
    """Geo decoder: FourierEmbedder → query_proj → cross-attn → output_proj → occupancy."""

    def __init__(self, num_latents: int = 4096, out_channels: int = 1,
                 width: int = 1024, heads: int = 16,
                 mlp_expand_ratio: int = 4,
                 num_freqs: int = 8, include_pi: bool = False,
                 qkv_bias: bool = False, qk_norm: bool = True,
                 enable_ln_post: bool = True,
                 downsample_ratio: int = 1):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)

        if downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        self.downsample_ratio = downsample_ratio

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width, heads=heads, mlp_expand_ratio=mlp_expand_ratio,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
        )

        self.enable_ln_post = enable_ln_post
        if enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def __call__(self, queries: mx.array, latents: mx.array) -> mx.array:
        """
        Args:
            queries: (B, P, 3) 3D query points
            latents: (B, num_latents, width) decoded VAE features
        Returns:
            (B, P, 1) occupancy logits
        """
        query_embeddings = self.query_proj(
            self.fourier_embedder(queries).astype(latents.dtype)
        )
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        return self.output_proj(x)
