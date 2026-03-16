"""
RRDBNet (Real-ESRGAN 4x upscaler) in MLX.
Ported from basicsr.archs.rrdbnet_arch.RRDBNet.

Architecture: 23 RRDB blocks (each 3x ResidualDenseBlock), nearest-neighbor
upsample 2x twice for 4x total. Fully convolutional, no attention.
"""
import mlx.core as mx
import mlx.nn as nn


def _nearest_upsample_2x(x: mx.array) -> mx.array:
    """Nearest-neighbor 2x upsample. Input/output: (B, H, W, C) NHWC."""
    B, H, W, C = x.shape
    # Repeat along H and W
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


class ResidualDenseBlock(nn.Module):
    """5-conv dense block with dense (concat) connections.

    Channel progression (num_feat=64, num_grow_ch=32):
        conv1: 64  -> 32
        conv2: 96  -> 32
        conv3: 128 -> 32
        conv4: 160 -> 32
        conv5: 192 -> 64
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = nn.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = nn.leaky_relu(self.conv2(mx.concatenate([x, x1], axis=-1)), negative_slope=0.2)
        x3 = nn.leaky_relu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)), negative_slope=0.2)
        x4 = nn.leaky_relu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)), negative_slope=0.2)
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 sequential RDBs with residual scaling."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class MlxESRGAN(nn.Module):
    """RRDBNet 4x upscaler in MLX.

    All convolutions are 3x3, stride=1, padding=1.
    MLX uses NHWC layout; weight conversion handles NCHW -> NHWC transpose.
    """

    def __init__(self, num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, padding=1)
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, padding=1)

    def __call__(self, image: mx.array) -> mx.array:
        """Upscale image 4x.

        Args:
            image: (B, H, W, 3) float32 in [0, 1]
        Returns:
            (B, H*4, W*4, 3) float32 clamped to [0, 1]
        """
        feat = self.conv_first(image)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # Upsample 2x twice = 4x total
        feat = nn.leaky_relu(self.conv_up1(_nearest_upsample_2x(feat)), negative_slope=0.2)
        feat = nn.leaky_relu(self.conv_up2(_nearest_upsample_2x(feat)), negative_slope=0.2)
        out = self.conv_last(nn.leaky_relu(self.conv_hr(feat), negative_slope=0.2))
        return mx.clip(out, 0.0, 1.0)

    @staticmethod
    def from_pth(path: str) -> "MlxESRGAN":
        """Load from a RealESRGAN .pth file (PyTorch weights)."""
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        # Some .pth files wrap weights under 'params_ema' or 'params'
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]

        mlx_weights = convert_esrgan_weights(state)
        model = MlxESRGAN()
        model.load_weights(list(mlx_weights.items()))
        return model


def convert_esrgan_weights(pytorch_state_dict: dict) -> dict:
    """Convert RealESRGAN .pth weights to MLX format.

    Handles:
    - Key remapping from PyTorch names to MLX module paths
    - Conv2d weight transposition: (O, I, H, W) -> (O, H, W, I)
    """
    import numpy as np

    mapping = {}

    for key, value in pytorch_state_dict.items():
        if hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.array(value)

        mlx_key = _remap_key(key)

        # Transpose conv weights: (O, I, kH, kW) -> (O, kH, kW, I)
        if "weight" in key and arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)

        mapping[mlx_key] = mx.array(arr)

    return mapping


def _remap_key(key: str) -> str:
    """Remap a single PyTorch state_dict key to the MLX module path.

    PyTorch keys look like:
        conv_first.weight
        body.0.rdb1.conv1.weight
        body.0.rdb1.conv1.bias
        conv_body.weight
        conv_up1.weight
        ...

    MLX module paths are identical except `body.N` -> `body.N` (list indexing
    works the same way in MLX weight loading).
    """
    # Keys already match the MLX structure exactly
    return key
