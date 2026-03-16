"""
AutoencoderKL (diffusers-style VAE) ported to MLX for Hunyuan3D texture pipeline.

Standard Stable Diffusion VAE: latent_channels=4, spatial downscale 8x.
All spatial tensors are NHWC. Conv weights converted from PyTorch OIHW → MLX OHWI.
"""
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Tuple


class GroupNormSiLU(nn.Module):
    """GroupNorm followed by SiLU activation."""

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.norm(x))


class ResnetBlock2D(nn.Module):
    """Standard VAE ResNet block: norm1→silu→conv1→norm2→silu→conv2 (+skip)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head attention over spatial dims (mid-block attention).

    Flattens H*W into a sequence, applies Q/K/V with 1x1 convs.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.group_norm(x)

        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)

        B, H, W, C = q.shape
        # Flatten spatial dims → (B, H*W, C)
        q = q.reshape(B, H * W, C)
        k = k.reshape(B, H * W, C)
        v = v.reshape(B, H * W, C)

        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(0, 2, 1)  # (B, HW, HW)
        attn = mx.softmax(attn, axis=-1)
        h = attn @ v  # (B, HW, C)

        h = h.reshape(B, H, W, C)
        h = self.to_out(h)
        return residual + h


class Downsample2D(nn.Module):
    """Stride-2 downsample with asymmetric padding (pad right+bottom by 1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        # Asymmetric pad: 0 left/top, 1 right/bottom → NHWC
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest-neighbor 2x upsample followed by 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        # Nearest-neighbor 2x: repeat each pixel along H and W
        x = mx.repeat(x, repeats=2, axis=1)
        x = mx.repeat(x, repeats=2, axis=2)
        return self.conv(x)


class DownBlock2D(nn.Module):
    """Encoder block: N resnet blocks + optional downsample."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: int = 2, add_downsample: bool = True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(in_ch, out_channels))
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = [Downsample2D(out_channels)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x


class UpBlock2D(nn.Module):
    """Decoder block: N resnet blocks + optional upsample."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: int = 3, add_upsample: bool = True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(in_ch, out_channels))
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = [Upsample2D(out_channels)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for us in self.upsamplers:
                x = us(x)
        return x


class MidBlock2D(nn.Module):
    """Mid-block: resnet → attention → resnet."""

    def __init__(self, channels: int):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels),
            ResnetBlock2D(channels, channels),
        ]
        self.attentions = [AttentionBlock(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Encoder(nn.Module):
    """VAE Encoder: (B, H, W, 3) → (B, H/8, W/8, 8)."""

    def __init__(self, in_channels: int = 3,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 layers_per_block: int = 2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
                                 kernel_size=3, padding=1)
        self.down_blocks = []
        out_ch = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            in_ch = out_ch
            out_ch = ch
            is_last = (i == len(block_out_channels) - 1)
            self.down_blocks.append(DownBlock2D(
                in_ch, out_ch,
                num_layers=layers_per_block,
                add_downsample=not is_last,
            ))

        self.mid_block = MidBlock2D(block_out_channels[-1])
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1],
                                         pytorch_compatible=True)
        # 8 output channels: 4 mean + 4 logvar
        self.conv_out = nn.Conv2d(block_out_channels[-1], 8,
                                  kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = nn.silu(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    """VAE Decoder: (B, H/8, W/8, 4) → (B, H, W, 3)."""

    def __init__(self, out_channels: int = 3, latent_channels: int = 4,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 layers_per_block: int = 2):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0],
                                 kernel_size=3, padding=1)
        self.mid_block = MidBlock2D(reversed_channels[0])

        self.up_blocks = []
        out_ch = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            in_ch = out_ch
            out_ch = ch
            is_last = (i == len(reversed_channels) - 1)
            self.up_blocks.append(UpBlock2D(
                in_ch, out_ch,
                num_layers=layers_per_block + 1,  # decoder has 3 resnets per block
                add_upsample=not is_last,
            ))

        self.conv_norm_out = nn.GroupNorm(32, reversed_channels[-1],
                                         pytorch_compatible=True)
        self.conv_out = nn.Conv2d(reversed_channels[-1], out_channels,
                                  kernel_size=3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = nn.silu(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x


class MlxAutoencoderKL(nn.Module):
    """AutoencoderKL ported to MLX. Standard SD VAE (latent_channels=4, 8x spatial)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 latent_channels: int = 4,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 layers_per_block: int = 2):
        super().__init__()
        self.encoder = Encoder(in_channels, block_out_channels, layers_per_block)
        self.decoder = Decoder(out_channels, latent_channels,
                               block_out_channels, layers_per_block)
        # Post-quant conv (encoder side) and quant conv (decoder side)
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels,
                                         kernel_size=1)

    # Standard SD VAE scaling factor (latent magnitude normalization)
    scaling_factor = 0.18215

    def encode(self, x: mx.array) -> tuple:
        """Encode image to latent posterior. (B, H, W, 3) → (mean, logvar) each (B, H/8, W/8, 4)."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = mx.split(h, 2, axis=-1)
        return mean, logvar

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to image. (B, H/8, W/8, 4) → (B, H, W, 3).

        Unscales by scaling_factor before decoding to match upstream convention.
        """
        z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)

    @staticmethod
    def from_diffusers(model_dir: str) -> "MlxAutoencoderKL":
        """Load from a diffusers model directory (safetensors weights).

        Expects either:
          - model_dir/diffusion_pytorch_model.safetensors
          - model_dir/diffusion_pytorch_model.fp16.safetensors
        Conv weights are transposed from PyTorch OIHW → MLX OHWI.
        """
        model_path = Path(model_dir)
        candidates = [
            model_path / "diffusion_pytorch_model.safetensors",
            model_path / "diffusion_pytorch_model.fp16.safetensors",
            model_path / "diffusion_pytorch_model.bin",
        ]
        weights_file = None
        for c in candidates:
            if c.exists():
                weights_file = c
                break
        if weights_file is None:
            raise FileNotFoundError(
                f"No weights found in {model_dir}. "
                f"Looked for: {[c.name for c in candidates]}"
            )

        if str(weights_file).endswith('.bin'):
            import torch
            state = torch.load(str(weights_file), map_location="cpu", weights_only=False)
            raw = {k: mx.array(v.float().numpy()) for k, v in state.items()}
        else:
            raw = mx.load(str(weights_file))
        converted = _convert_diffusers_weights(raw)

        model = MlxAutoencoderKL()
        model.load_weights(list(converted.items()))
        return model


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------

def _is_conv_weight(key: str) -> bool:
    """Check if a weight key corresponds to a Conv2d weight (4D tensor).

    Checks both original key names and remapped names.
    """
    return key.endswith(".weight") and any(
        part in key for part in ("conv_in", "conv_out", "conv1", "conv2",
                                 "conv_shortcut", "to_q", "to_k", "to_v",
                                 "to_out", "query", "key", "value",
                                 "proj_attn", "conv", "quant_conv",
                                 "post_quant_conv")
    ) and "norm" not in key


def _remap_key(key: str) -> str:
    """Remap diffusers weight key to our module path.

    Diffusers uses `downsamplers.0.conv` / `upsamplers.0.conv` which matches
    our structure directly. Attention keys use `query`/`key`/`value`/`proj_attn`
    in some diffusers versions — we remap those to our to_q/to_k/to_v/to_out.
    """
    # Diffusers attention key variants
    key = key.replace(".query.", ".to_q.")
    key = key.replace(".key.", ".to_k.")
    key = key.replace(".value.", ".to_v.")
    key = key.replace(".proj_attn.", ".to_out.")
    return key


def _convert_diffusers_weights(raw: dict) -> dict:
    """Convert diffusers VAE weights to MLX format.

    - Transpose conv weights: PyTorch (O, I, H, W) → MLX (O, H, W, I)
    - Reshape 2D attention weights to 4D for Conv2d(kernel_size=1)
    - Remap key names where needed
    """
    converted = {}
    for key, value in raw.items():
        new_key = _remap_key(key)

        if _is_conv_weight(key):
            if value.ndim == 4:
                # PyTorch: (O, I, H, W) → MLX Conv2d: (O, H, W, I)
                value = value.transpose(0, 2, 3, 1)
            elif value.ndim == 2:
                # 2D Linear weight (O, I) → Conv2d 1x1: (O, 1, 1, I)
                value = value.reshape(value.shape[0], 1, 1, value.shape[1])

        converted[new_key] = value
    return converted
