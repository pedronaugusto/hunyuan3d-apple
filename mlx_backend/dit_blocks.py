"""
HunYuanDiTBlock for MLX.
Ported from hy3dshape/hy3dshape/models/denoisers/hunyuandit.py
"""
import mlx.core as mx
import mlx.nn as nn
from .attention import SelfAttention, CrossAttention
from .moe import MoEBlock
from .norm import PreciseLayerNorm


class DiTMLP(nn.Module):
    """Simple MLP: Linear(w, 4w) → GELU → Linear(4w, w)."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class HunYuanDiTBlock(nn.Module):
    """Single DiT block: self-attn → cross-attn → MoE/MLP, with optional U-skip."""

    def __init__(self, hidden_size: int, num_heads: int,
                 text_states_dim: int = 1024,
                 qk_norm: bool = False, qkv_bias: bool = False,
                 skip_connection: bool = False,
                 use_moe: bool = False,
                 num_experts: int = 8, moe_top_k: int = 2):
        super().__init__()

        self.norm1 = PreciseLayerNorm(hidden_size, eps=1e-6)
        self.attn1 = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)

        self.norm2 = PreciseLayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads,
                                    qkv_bias=qkv_bias, qk_norm=qk_norm)

        self.norm3 = PreciseLayerNorm(hidden_size, eps=1e-6)

        self.use_moe = use_moe
        if use_moe:
            self.moe = MoEBlock(hidden_size, num_experts=num_experts,
                                moe_top_k=moe_top_k,
                                ff_inner_dim=int(hidden_size * 4.0),
                                ff_bias=True)
        else:
            self.mlp = DiTMLP(hidden_size)

        if skip_connection:
            self.skip_norm = PreciseLayerNorm(hidden_size, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

    def __call__(self, x: mx.array, c: mx.array = None,
                 text_states: mx.array = None,
                 skip_value: mx.array = None) -> mx.array:
        if self.skip_linear is not None:
            cat = mx.concatenate([skip_value, x], axis=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-attention
        x = x + self.attn1(self.norm1(x))

        # Cross-attention
        x = x + self.attn2(self.norm2(x), text_states)

        # FFN
        mlp_in = self.norm3(x)
        if self.use_moe:
            x = x + self.moe(mlp_in)
        else:
            x = x + self.mlp(mlp_in)

        return x
