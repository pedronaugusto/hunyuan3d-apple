"""
UNet2DConditionModel building blocks for the Hunyuan3D texture pipeline in MLX.

Implements standard Stable Diffusion UNet blocks (ResNet, attention, down/up/mid)
with NHWC layout throughout. Designed to compose into a full UNet via a separate
unet.py module.

All convolutions and norms use MLX's channels-last convention.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .norm import PreciseLayerNorm


# ---------------------------------------------------------------------------
# Timestep embeddings (SD-style: sinusoidal → two-layer MLP)
# ---------------------------------------------------------------------------

class Timesteps:
    """Sinusoidal timestep embeddings (not a nn.Module — no learnable params)."""

    def __init__(self, num_channels: int = 320, max_period: int = 10000):
        self.num_channels = num_channels
        half = num_channels // 2
        exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
        self._freqs = mx.exp(exponent)

    def __call__(self, timesteps: mx.array) -> mx.array:
        """timesteps: (B,) → (B, num_channels)."""
        args = timesteps[:, None].astype(mx.float32) * self._freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.num_channels % 2 == 1:
            emb = mx.pad(emb, [(0, 0), (0, 1)])
        return emb


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal embeddings into model space."""

    def __init__(self, channel: int = 320, time_embed_dim: int = 1280):
        super().__init__()
        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        """sample: (B, channel) → (B, time_embed_dim)."""
        return self.linear_2(nn.silu(self.linear_1(sample)))


# ---------------------------------------------------------------------------
# Core conv blocks
# ---------------------------------------------------------------------------

class ResnetBlock2D(nn.Module):
    """Standard ResNet block with optional time-embedding injection.

    Layout: NHWC throughout. GroupNorm acts on the last (channel) dimension.
    """

    def __init__(self, in_channels: int, out_channels: int = None,
                 temb_channels: int = 1280):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(32, pytorch_compatible=True, dims= in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = (
            nn.Linear(temb_channels, out_channels) if temb_channels else None
        )

        self.norm2 = nn.GroupNorm(32, pytorch_compatible=True, dims= out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

    def __call__(self, x: mx.array, temb: mx.array = None) -> mx.array:
        """x: (B, H, W, C), temb: (B, temb_channels) or None."""
        residual = x

        x = nn.silu(self.norm1(x))
        x = self.conv1(x)

        if temb is not None and self.time_emb_proj is not None:
            # Project temb and broadcast over spatial dims
            x = x + self.time_emb_proj(nn.silu(temb))[:, None, None, :]

        x = nn.silu(self.norm2(x))
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual


class Downsample2D(nn.Module):
    """Spatial 2x downsample via stride-2 convolution with symmetric padding."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) → (B, H//2, W//2, C)."""
        return self.conv(x)


class Upsample2D(nn.Module):
    """Spatial 2x upsample via nearest-neighbor interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) → (B, 2H, 2W, C)."""
        B, H, W, C = x.shape
        # Nearest-neighbor 2x: repeat along H and W
        x = mx.repeat(x[:, :, None, :, None, :], repeats=2, axis=2)
        x = mx.repeat(x, repeats=2, axis=4)
        x = x.reshape(B, H * 2, W * 2, C)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    """GELU-gated linear unit: splits input in half, gates with GELU."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    """Position-wise feed-forward with GEGLU activation."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net_0 = GEGLU(dim, inner_dim)
        self.net_2 = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net_2(self.net_0(x))


class CrossAttentionMHA(nn.Module):
    """Multi-head attention supporting both self- and cross-attention.

    Used inside BasicTransformerBlock. Standard Q/K/V projections with
    multi-head scaled dot-product attention.
    """

    def __init__(self, query_dim: int, num_heads: int, dim_head: int,
                 kv_dim: int = None):
        super().__init__()
        kv_dim = kv_dim or query_dim
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def __call__(self, x: mx.array, context: mx.array = None) -> mx.array:
        """x: (B, N, D), context: (B, S, kv_dim) or None for self-attn."""
        context = context if context is not None else x
        B, N, _ = x.shape
        H, D = self.num_heads, self.dim_head

        q = self.to_q(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, -1, H, D).transpose(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, H, D).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        return self.to_out(out)


class SpatialAttention(nn.Module):
    """Single-head attention over spatial dimensions (used in UNet mid-block)."""

    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, pytorch_compatible=True, dims= channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) → (B, H, W, C)."""
        B, H, W, C = x.shape
        residual = x

        x = self.group_norm(x).reshape(B, H * W, C)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)
        out = self.to_out(attn @ v)

        return out.reshape(B, H, W, C) + residual


# ---------------------------------------------------------------------------
# Transformer blocks (for cross-attention in UNet)
# ---------------------------------------------------------------------------

class BasicTransformerBlock(nn.Module):
    """Standard transformer block: self-attn -> cross-attn -> FFN.

    Pre-norm with LayerNorm on each sub-block.
    """

    def __init__(self, dim: int, num_heads: int, dim_head: int,
                 cross_attention_dim: int = 1024):
        super().__init__()
        self._dim = dim
        self._num_heads = num_heads
        self._dim_head = dim_head
        self._cross_attention_dim = cross_attention_dim

        self.norm1 = PreciseLayerNorm(dim)
        self.attn1 = CrossAttentionMHA(dim, num_heads, dim_head)

        self.norm2 = PreciseLayerNorm(dim)
        self.attn2 = CrossAttentionMHA(dim, num_heads, dim_head,
                                       kv_dim=cross_attention_dim)

        self.norm3 = PreciseLayerNorm(dim)
        self.ff = FeedForward(dim)

    @property
    def dim(self):
        return self._dim

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def cross_attention_dim(self):
        return self._cross_attention_dim

    def __call__(self, x: mx.array, context: mx.array = None,
                 cross_attention_kwargs: dict = None) -> mx.array:
        """x: (B, N, D), context: (B, S, cross_dim)."""
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer2DModel(nn.Module):
    """Spatial transformer that wraps BasicTransformerBlocks.

    Flattens spatial dims before attention, restores afterwards.
    """

    def __init__(self, num_attention_heads: int, attention_head_dim: int,
                 in_channels: int, num_layers: int = 1,
                 cross_attention_dim: int = 1024):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(32, pytorch_compatible=True, dims= in_channels)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = [
            BasicTransformerBlock(inner_dim, num_attention_heads,
                                 attention_head_dim, cross_attention_dim)
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(self, x: mx.array, context: mx.array = None,
                 cross_attention_kwargs: dict = None) -> mx.array:
        """x: (B, H, W, C), context: (B, S, cross_dim)."""
        B, H, W, C = x.shape
        residual = x

        x = self.norm(x).reshape(B, H * W, C)
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, context=context,
                       cross_attention_kwargs=cross_attention_kwargs)

        x = self.proj_out(x).reshape(B, H, W, C)
        return x + residual


# ---------------------------------------------------------------------------
# Down blocks
# ---------------------------------------------------------------------------

class DownBlock2D(nn.Module):
    """UNet down block with ResNet layers and optional downsampler (no attention)."""

    def __init__(self, in_channels: int, out_channels: int,
                 temb_channels: int = 1280, num_layers: int = 2,
                 add_downsample: bool = True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(res_in, out_channels, temb_channels))

        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else []

    def __call__(self, x: mx.array, temb: mx.array = None,
                 context: mx.array = None) -> tuple:
        """Returns (hidden_state, list_of_residuals)."""
        output_states = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            output_states.append(x)

        for down in self.downsamplers:
            x = down(x)
            output_states.append(x)

        return x, output_states


class CrossAttnDownBlock2D(nn.Module):
    """UNet down block with ResNet + Transformer2D layers and optional downsampler."""

    def __init__(self, in_channels: int, out_channels: int,
                 temb_channels: int = 1280, num_layers: int = 2,
                 num_attention_heads: int = 10,
                 cross_attention_dim: int = 1024,
                 add_downsample: bool = True):
        super().__init__()
        head_dim = out_channels // num_attention_heads

        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(res_in, out_channels, temb_channels))
            self.attentions.append(
                Transformer2DModel(num_attention_heads, head_dim,
                                   out_channels, num_layers=1,
                                   cross_attention_dim=cross_attention_dim)
            )

        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else []

    def __call__(self, x: mx.array, temb: mx.array = None,
                 context: mx.array = None,
                 cross_attention_kwargs: dict = None) -> tuple:
        """Returns (hidden_state, list_of_residuals)."""
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, temb)
            x = attn(x, context=context,
                      cross_attention_kwargs=cross_attention_kwargs)
            output_states.append(x)

        for down in self.downsamplers:
            x = down(x)
            output_states.append(x)

        return x, output_states


# ---------------------------------------------------------------------------
# Up blocks
# ---------------------------------------------------------------------------

class UpBlock2D(nn.Module):
    """UNet up block with ResNet layers and optional upsampler (no attention).

    Each resnet receives the concatenation of the current hidden state and
    the corresponding skip connection from the encoder.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 prev_output_channel: int, temb_channels: int = 1280,
                 num_layers: int = 3, add_upsample: bool = True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            # Main stream channels
            main_ch = prev_output_channel if i == 0 else out_channels
            # Skip connection channels (last layer gets from deeper block)
            skip_ch = in_channels if (i == num_layers - 1) else out_channels
            self.resnets.append(ResnetBlock2D(main_ch + skip_ch, out_channels, temb_channels))

        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(self, x: mx.array, res_hidden_states: list,
                 temb: mx.array = None, context: mx.array = None) -> mx.array:
        """x: current hidden state, res_hidden_states: skip connections (popped)."""
        for resnet in self.resnets:
            skip = res_hidden_states.pop()
            x = mx.concatenate([x, skip], axis=-1)
            x = resnet(x, temb)

        for up in self.upsamplers:
            x = up(x)

        return x


class CrossAttnUpBlock2D(nn.Module):
    """UNet up block with ResNet + Transformer2D layers and optional upsampler.

    Like UpBlock2D but with cross-attention after each resnet.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 prev_output_channel: int, temb_channels: int = 1280,
                 num_layers: int = 3, num_attention_heads: int = 10,
                 cross_attention_dim: int = 1024,
                 add_upsample: bool = True):
        super().__init__()
        head_dim = out_channels // num_attention_heads

        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            # Main stream channels
            main_ch = prev_output_channel if i == 0 else out_channels
            # Skip connection channels (last layer gets from deeper block)
            skip_ch = in_channels if (i == num_layers - 1) else out_channels
            self.resnets.append(ResnetBlock2D(main_ch + skip_ch, out_channels, temb_channels))
            self.attentions.append(
                Transformer2DModel(num_attention_heads, head_dim,
                                   out_channels, num_layers=1,
                                   cross_attention_dim=cross_attention_dim)
            )

        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(self, x: mx.array, res_hidden_states: list,
                 temb: mx.array = None, context: mx.array = None,
                 cross_attention_kwargs: dict = None) -> mx.array:
        """x: current hidden state, res_hidden_states: skip connections (popped)."""
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = res_hidden_states.pop()
            x = mx.concatenate([x, skip], axis=-1)
            x = resnet(x, temb)
            x = attn(x, context=context,
                      cross_attention_kwargs=cross_attention_kwargs)

        for up in self.upsamplers:
            x = up(x)

        return x


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------

class UNetMidBlock2DCrossAttn(nn.Module):
    """UNet mid block: ResNet -> Transformer2D -> ResNet."""

    def __init__(self, in_channels: int, temb_channels: int = 1280,
                 num_attention_heads: int = 20,
                 cross_attention_dim: int = 1024):
        super().__init__()
        head_dim = in_channels // num_attention_heads

        self.resnets = [
            ResnetBlock2D(in_channels, in_channels, temb_channels),
            ResnetBlock2D(in_channels, in_channels, temb_channels),
        ]
        self.attentions = [
            Transformer2DModel(num_attention_heads, head_dim,
                               in_channels, num_layers=1,
                               cross_attention_dim=cross_attention_dim)
        ]

    def __call__(self, x: mx.array, temb: mx.array = None,
                 context: mx.array = None,
                 cross_attention_kwargs: dict = None) -> mx.array:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context=context,
                                cross_attention_kwargs=cross_attention_kwargs)
        x = self.resnets[1](x, temb)
        return x


# ---------------------------------------------------------------------------
# Full UNet2DConditionModel
# ---------------------------------------------------------------------------

class UNet2DConditionModel(nn.Module):
    """Standard UNet2D with cross-attention, matching diffusers architecture.

    Config for Hunyuan3D texture:
        in_channels=12, out_channels=4,
        block_out_channels=[320, 640, 1280, 1280],
        layers_per_block=2, cross_attention_dim=1024,
        attention_head_dim=[5, 10, 20, 20]
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 block_out_channels: list = None,
                 layers_per_block: int = 2,
                 cross_attention_dim: int = 1024,
                 attention_head_dim: list = None,
                 down_block_types: list = None,
                 up_block_types: list = None):
        super().__init__()
        block_out_channels = block_out_channels or [320, 640, 1280, 1280]
        attention_head_dim = attention_head_dim or [5, 10, 20, 20]
        n_blocks = len(block_out_channels)

        # Default block types matching Hunyuan3D config:
        # down: [CrossAttn, CrossAttn, CrossAttn, Down]
        # up: [Up, CrossAttn, CrossAttn, CrossAttn]
        if down_block_types is None:
            down_block_types = ["CrossAttnDownBlock2D"] * (n_blocks - 1) + ["DownBlock2D"]
        if up_block_types is None:
            up_block_types = ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * (n_blocks - 1)

        # Store config as plain Python values (not mx arrays) so MLX Module
        # doesn't intercept them. Use _private names to avoid nn.Module tracking.
        self._cfg_in_channels = in_channels
        self._cfg_out_channels = out_channels

        # Time embedding
        time_embed_dim = block_out_channels[0] * 4  # 1280
        self.time_proj = Timesteps(block_out_channels[0])
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)

        # Down blocks
        self.down_blocks = []
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            is_last = (i == n_blocks - 1)
            # attention_head_dim[i] IS the number of heads (diffusers convention)
            num_heads = attention_head_dim[i]

            if down_block_types[i] == "DownBlock2D":
                self.down_blocks.append(
                    DownBlock2D(in_ch, out_ch, time_embed_dim,
                                num_layers=layers_per_block,
                                add_downsample=not is_last)
                )
            else:
                self.down_blocks.append(
                    CrossAttnDownBlock2D(in_ch, out_ch, time_embed_dim,
                                         num_layers=layers_per_block,
                                         num_attention_heads=num_heads,
                                         cross_attention_dim=cross_attention_dim,
                                         add_downsample=not is_last)
                )
            in_ch = out_ch

        # Mid block
        mid_heads = attention_head_dim[-1]
        self.mid_block = UNetMidBlock2DCrossAttn(
            block_out_channels[-1], time_embed_dim,
            num_attention_heads=mid_heads,
            cross_attention_dim=cross_attention_dim,
        )

        # Up blocks (reversed channels + skip connections)
        # Diffusers convention:
        #   output_channel = reversed[i]   (target output of this up block)
        #   input_channel  = reversed[i+1] (from deeper down block, used for last resnet's skip)
        #   prev_output_channel = output of previous up block (or mid block)
        reversed_channels = list(reversed(block_out_channels))
        reversed_head_dims = list(reversed(attention_head_dim))

        self.up_blocks = []
        output_channel = reversed_channels[0]
        for i in range(len(reversed_channels)):
            is_last = (i == len(reversed_channels) - 1)
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            input_channel = reversed_channels[min(i + 1, len(reversed_channels) - 1)]
            num_heads = reversed_head_dims[i]

            if up_block_types[i] == "UpBlock2D":
                self.up_blocks.append(
                    UpBlock2D(input_channel, output_channel, prev_output_channel,
                              time_embed_dim,
                              num_layers=layers_per_block + 1,
                              add_upsample=not is_last)
                )
            else:
                self.up_blocks.append(
                    CrossAttnUpBlock2D(input_channel, output_channel,
                                       prev_output_channel, time_embed_dim,
                                       num_layers=layers_per_block + 1,
                                       num_attention_heads=num_heads,
                                       cross_attention_dim=cross_attention_dim,
                                       add_upsample=not is_last)
                )

        # Output
        self.conv_norm_out = nn.GroupNorm(32, pytorch_compatible=True, dims= block_out_channels[0])
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    @property
    def in_channels(self):
        return self._cfg_in_channels

    @property
    def out_channels(self):
        return self._cfg_out_channels

    def __call__(self, sample: mx.array, timestep: mx.array,
                 encoder_hidden_states: mx.array = None,
                 cross_attention_kwargs: dict = None) -> mx.array:
        """
        sample: (B, H, W, C) NHWC format — or (B, C, H, W) NCHW to be transposed
        timestep: scalar or (B,) timestep
        encoder_hidden_states: (B, S, D) cross-attention context
        Returns: (B, H, W, out_channels) noise prediction
        """
        # UNet2DConditionModel always receives NCHW from the 2.5D wrapper
        # and internally operates in NHWC.  The old heuristic
        # (shape[1] < shape[2]) happens to work for typical dims but is
        # fragile — replace with unconditional transpose.
        if sample.ndim == 4:
            sample = sample.transpose(0, 2, 3, 1)  # NCHW → NHWC

        # Time embedding
        if timestep.ndim == 0:
            timestep = mx.broadcast_to(timestep, (sample.shape[0],))
        t_emb = self.time_proj(timestep.astype(mx.float32))
        t_emb = self.time_embedding(t_emb)

        # Input conv
        x = self.conv_in(sample)

        # Down — eval after each block to bound Metal memory
        down_block_res = [x]
        for i, block in enumerate(self.down_blocks):
            block_name = type(block).__name__
            if isinstance(block, CrossAttnDownBlock2D):
                x, res = block(x, t_emb, encoder_hidden_states,
                               cross_attention_kwargs=cross_attention_kwargs)
            else:
                x, res = block(x, t_emb)
            down_block_res.extend(res)
            mx.eval(x)

        # Mid
        x = self.mid_block(x, t_emb, encoder_hidden_states,
                           cross_attention_kwargs=cross_attention_kwargs)
        mx.eval(x)

        # Up — eval after each block to bound Metal memory
        for i, block in enumerate(self.up_blocks):
            n_resnets = len(block.resnets)
            skips = down_block_res[-n_resnets:]
            down_block_res = down_block_res[:-n_resnets]

            block_name = type(block).__name__
            if isinstance(block, CrossAttnUpBlock2D):
                x = block(x, skips, t_emb, encoder_hidden_states,
                          cross_attention_kwargs=cross_attention_kwargs)
            else:
                x = block(x, skips, t_emb)

        # Output
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x
