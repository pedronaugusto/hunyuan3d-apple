"""
Timestep embedding for Hunyuan DiT in MLX.
Ported from hy3dshape/hy3dshape/models/denoisers/hunyuandit.py
"""
import math
import mlx.core as mx
import mlx.nn as nn


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, num_channels: int, downscale_freq_shift: float = 0.0,
                 scale: int = 1, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.scale = scale
        # Pre-compute frequency basis (never changes)
        half_dim = num_channels // 2
        exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - downscale_freq_shift)
        self._freqs = mx.exp(exponent)  # (half_dim,)

    def __call__(self, timesteps: mx.array) -> mx.array:
        """timesteps: (B,) 1D array."""
        emb = timesteps[:, None].astype(mx.float32) * self._freqs[None, :]
        emb = self.scale * emb
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        if self.num_channels % 2 == 1:
            emb = mx.pad(emb, [(0, 0), (0, 1)])
        return emb


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256,
                 cond_proj_dim: int = None, out_size: int = None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = MLP2(hidden_size, frequency_embedding_size, out_size)
        self.frequency_embedding_size = frequency_embedding_size
        self.time_embed = Timesteps(hidden_size)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

    def __call__(self, t: mx.array, condition: mx.array = None) -> mx.array:
        t_freq = self.time_embed(t)
        t_freq = t_freq.astype(self.mlp.fc1.weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)
        t_emb = self.mlp(t_freq)
        return t_emb[:, None, :]  # (B, 1, D)


class MLP2(nn.Module):
    """Two-layer MLP with GELU: Linear → GELU → Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))
