"""
ShapeVAE decoder (post_kl + transformer) in MLX.
Ported from hy3dshape/hy3dshape/models/autoencoders/model.py + attention_blocks.py

Config from hunyuan3d-vae-v2-1/config.yaml:
  num_latents=4096, embed_dim=64, width=1024, heads=16,
  num_decoder_layers=16, qkv_bias=false, qk_norm=true
"""
import mlx.core as mx
import mlx.nn as nn
from .norm import QKLayerNorm


class VAESelfAttention(nn.Module):
    """Fused QKV self-attention with optional QK-norm."""

    def __init__(self, width: int, heads: int, qkv_bias: bool = False,
                 qk_norm: bool = True):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        if qk_norm:
            self.q_norm = QKLayerNorm(self.head_dim)
            self.k_norm = QKLayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.heads, self.head_dim

        # Upstream interleaving: view(B,N,H,3D) → split per head
        qkv = self.c_qkv(x).reshape(B, N, H, 3 * D)
        q = qkv[:, :, :, :D]
        k = qkv[:, :, :, D:2*D]
        v = qkv[:, :, :, 2*D:]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.c_proj(out)


class VAEMLP(nn.Module):
    """MLP: Linear(w, 4w) → GELU → Linear(4w, w)."""

    def __init__(self, width: int):
        super().__init__()
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    """LN → self-attn + residual → LN → MLP + residual."""

    def __init__(self, width: int, heads: int, qkv_bias: bool = False,
                 qk_norm: bool = True):
        super().__init__()
        self.attn = VAESelfAttention(width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = VAEMLP(width)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VAETransformer(nn.Module):
    """post_kl linear + 16-layer transformer for ShapeVAE decoding."""

    def __init__(self, num_latents: int = 4096, embed_dim: int = 64,
                 width: int = 1024, heads: int = 16,
                 num_decoder_layers: int = 16,
                 qkv_bias: bool = False, qk_norm: bool = True):
        super().__init__()
        self.post_kl = nn.Linear(embed_dim, width)
        self.resblocks = [
            ResidualAttentionBlock(width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
            for _ in range(num_decoder_layers)
        ]

    def __call__(self, latents: mx.array) -> mx.array:
        """
        Args:
            latents: (B, num_latents, embed_dim) — raw latent vectors
        Returns:
            (B, num_latents, width) decoded features
        """
        x = self.post_kl(latents)
        for block in self.resblocks:
            x = block(x)
        return x
