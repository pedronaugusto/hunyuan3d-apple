"""
DINOv2 ViT-L/14 feature extractor in MLX.
Adapted from trellis2 dinov3.py — removed register tokens and RoPE,
uses absolute position embeddings, patch_size=14, image_size=518.

Config from config.yaml:
  hidden_size=1024, num_attention_heads=16, num_hidden_layers=24,
  patch_size=14, image_size=518, qkv_bias=true
"""
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


class DINOv2PatchEmbed(nn.Module):
    """Patch embedding: manual conv via reshape + matmul, + CLS token.

    Supports position embedding interpolation for input sizes that differ
    from the stored embedding grid (matching HuggingFace Dinov2Model behavior).
    """

    def __init__(self, dim: int = 1024, patch_size: int = 14, image_size: int = 518):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.weight = mx.zeros((dim, patch_size, patch_size, 3))
        self.bias = mx.zeros((dim,))
        self.cls_token = mx.zeros((1, 1, dim))
        # Position embeddings: 1 CLS + num_patches (may be interpolated at runtime)
        num_patches = (image_size // patch_size) ** 2
        self.position_embeddings = mx.zeros((1, 1 + num_patches, dim))

    def _interpolate_pos_encoding(self, num_patches_h: int, num_patches_w: int) -> mx.array:
        """Interpolate stored position embeddings to match actual input grid size.

        Matches HuggingFace Dinov2Embeddings.interpolate_pos_encoding().
        """
        num_positions = self.position_embeddings.shape[1] - 1  # exclude CLS
        stored_grid = int(num_positions ** 0.5)

        if num_patches_h == stored_grid and num_patches_w == stored_grid:
            return self.position_embeddings

        # Split CLS and patch position embeddings
        cls_pos = self.position_embeddings[:, :1, :]  # (1, 1, dim)
        patch_pos = self.position_embeddings[:, 1:, :]  # (1, stored_grid^2, dim)

        dim = patch_pos.shape[-1]
        # Reshape to spatial grid: (1, stored_grid, stored_grid, dim)
        patch_pos = patch_pos.reshape(1, stored_grid, stored_grid, dim)

        # Use torch for bicubic interpolation (matching HF implementation).
        # HF upcasts to float32 for interpolation then casts back to the
        # model dtype (float16). We must do the same cast-back or the pos
        # embed stays float32 while the rest of the model is float16,
        # introducing a 0.19 max_diff that compounds to ~18 over 40 layers.
        import torch
        import torch.nn.functional as F
        target_dtype = patch_pos.dtype
        patch_pos_t = torch.from_numpy(np.array(patch_pos)).permute(0, 3, 1, 2).float()
        patch_pos_t = F.interpolate(
            patch_pos_t,
            size=(num_patches_h, num_patches_w),
            mode="bicubic",
            align_corners=False,
        )
        # Cast back to model dtype to match HF precision path
        patch_pos_interp = mx.array(patch_pos_t.permute(0, 2, 3, 1).numpy()).astype(target_dtype)
        patch_pos_interp = patch_pos_interp.reshape(1, num_patches_h * num_patches_w, dim)

        return mx.concatenate([cls_pos, patch_pos_interp], axis=1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, 3) -> (B, 1+num_patches, dim)"""
        B, H, W, C = x.shape
        P = self.patch_size
        nH, nW = H // P, W // P

        # Reshape into patches
        x = x.reshape(B, nH, P, nW, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, nH * nW, P * P * C)

        # Linear projection
        w = self.weight.reshape(self.weight.shape[0], -1).T  # (P*P*3, dim)
        patches = x @ w + self.bias

        # Prepend CLS
        cls = mx.broadcast_to(self.cls_token, (B, 1, patches.shape[-1]))
        tokens = mx.concatenate([cls, patches], axis=1)

        # Add position embeddings (interpolated if input size differs from stored)
        pos_embed = self._interpolate_pos_encoding(nH, nW)
        tokens = tokens + pos_embed
        return tokens


class DINOv2TransformerBlock(nn.Module):
    """ViT block: LN → MHSA → LN → MLP."""

    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0,
                 use_swiglu: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DINOv2Attention(dim, num_heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = DINOv2MLP(dim, hidden, use_swiglu=use_swiglu)
        # Layer scale (DINOv2 uses per-channel learnable scale on residuals)
        self.layer_scale1 = mx.ones((dim,))
        self.layer_scale2 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale1 * self.attn(self.norm1(x))
        x = x + self.layer_scale2 * self.mlp(self.norm2(x))
        return x


class DINOv2Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = D ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class DINOv2MLP(nn.Module):
    """MLP with GELU (standard) or SwiGLU."""

    def __init__(self, dim: int, hidden: int, use_swiglu: bool = False):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # SwiGLU: fused gate+up projection
            swiglu_hidden = int(hidden * 2 / 3)
            self.fc1 = nn.Linear(dim, swiglu_hidden * 2, bias=True)
            self.fc2 = nn.Linear(swiglu_hidden, dim, bias=True)
        else:
            self.fc1 = nn.Linear(dim, hidden, bias=True)
            self.fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_swiglu:
            h = self.fc1(x)
            gate, up = mx.split(h, 2, axis=-1)
            return self.fc2(nn.silu(gate) * up)
        return self.fc2(nn.gelu(self.fc1(x)))


class MlxDINOv2(nn.Module):
    """DINOv2 ViT-L/14, 24 layers, dim=1024, 16 heads."""

    def __init__(self, dim: int = 1024, num_heads: int = 16, num_layers: int = 24,
                 patch_size: int = 14, mlp_ratio: float = 4.0, image_size: int = 518,
                 use_swiglu: bool = None, weights_image_size: int = None):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.image_size = image_size
        # weights_image_size: size the stored weights were trained at.
        # If different from image_size, position embeddings are allocated at
        # weights_image_size for loading, then interpolated at runtime.
        self._weights_image_size = weights_image_size or image_size

        # Auto-detect SwiGLU for giant model (dim=1536 uses SwiGLU per HF config)
        if use_swiglu is None:
            use_swiglu = dim >= 1536

        # Allocate position embeddings at weights size so load_weights succeeds
        self.embeddings = DINOv2PatchEmbed(dim, patch_size, self._weights_image_size)
        self.layers = [DINOv2TransformerBlock(dim, num_heads, mlp_ratio, use_swiglu=use_swiglu)
                       for _ in range(num_layers)]
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def __call__(self, images: list) -> mx.array:
        """
        Extract features from PIL images.
        Returns (B, 1+num_patches, dim) including CLS token.
        """
        x = self._preprocess(images)
        # Cast input to match weight dtype so all compute runs in the same
        # precision (upstream runs DINO entirely in fp16).
        weight_dtype = self.embeddings.weight.dtype
        if x.dtype != weight_dtype:
            x = x.astype(weight_dtype)
        h = self.embeddings(x)
        for layer in self.layers:
            h = layer(h)
        h = self.layernorm(h)
        return h

    def _preprocess(self, images: list) -> mx.array:
        """Match upstream BitImageProcessor: resize shortest_edge=256, center crop to image_size, bicubic, normalize."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                img = img.convert('RGB')
                # Step 1: Resize so shortest edge = 256 (matching upstream BitImageProcessor)
                w, h = img.size
                short = min(w, h)
                scale = 256.0 / short
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
                # Step 2: Center crop to image_size x image_size
                left = (new_w - self.image_size) // 2
                top = (new_h - self.image_size) // 2
                img = img.crop((left, top, left + self.image_size, top + self.image_size))
                arr = np.array(img).astype(np.float32) / 255.0
            else:
                arr = np.array(img, dtype=np.float32)
                if arr.ndim != 3 or arr.shape[-1] != 3:
                    raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
                if arr.max() > 1.5:
                    arr = arr / 255.0
                if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
                    import torch
                    import torch.nn.functional as F

                    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    tensor = F.interpolate(
                        tensor,
                        size=(self.image_size, self.image_size),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )
                    arr = tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            arr = (arr - mean) / std
            processed.append(arr)

        return mx.array(np.stack(processed))
