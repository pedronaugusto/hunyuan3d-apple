"""
Normalization layers for MLX backend.
"""
import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm that casts to float32 internally (matches PyTorch LayerNorm32)."""

    def __init__(self, dims: int, elementwise_affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x.astype(x_dtype)


class PreciseLayerNorm(nn.Module):
    """Two-pass LayerNorm avoiding mx.fast.layer_norm's fused kernel.

    The fused kernel's single-pass variance uses a different parallel
    reduction order than PyTorch CPU, causing ~1.4e-6 max error per call.
    This two-pass version (mean → center → var → normalize) matches
    PyTorch to ~9.5e-7 (1.5x better) and stays within MLX's lazy
    evaluation graph, which preserves operator fusion across layers.
    Over 63 LayerNorms in the DiT, this halves the compounded error.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        var = mx.mean(centered * centered, axis=-1, keepdims=True)
        return centered * mx.rsqrt(var + self.eps) * self.weight + self.bias


class QKRMSNorm(nn.Module):
    """Per-head RMSNorm for Q/K normalization (matches PyTorch nn.RMSNorm on head_dim)."""

    def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight if hasattr(self, "weight") else None
        return mx.fast.rms_norm(x, weight, self.eps)


class QKLayerNorm(nn.Module):
    """Per-head LayerNorm for Q/K normalization (matches upstream nn.LayerNorm on head_dim)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return x * self.weight + self.bias
