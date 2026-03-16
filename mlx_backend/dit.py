"""
HunYuanDiTPlain — the full DiT denoiser in MLX.
Ported from hy3dshape/hy3dshape/models/denoisers/hunyuandit.py.
"""
import mlx.core as mx
import mlx.nn as nn
from .timestep_embed import TimestepEmbedder
from .dit_blocks import HunYuanDiTBlock, PreciseLayerNorm


class _SiLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class _SequentialProjector(nn.Module):
    """Small Sequential-compatible MLP so weight keys match upstream remapping."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = [
            nn.Linear(in_dim, hidden_dim, bias=True),
            _SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionPool(nn.Module):
    """Batch-first port of the upstream attention pool."""

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        out_dim = embed_dim if output_dim is None else output_dim

        self.positional_embedding = mx.zeros((spacial_dim + 1, embed_dim))
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.c_proj = nn.Linear(embed_dim, out_dim, bias=True)

    def __call__(self, x: mx.array, attention_mask: mx.array = None) -> mx.array:
        if attention_mask is not None:
            weights = attention_mask[..., None]
            pooled = (x * weights).sum(axis=1, keepdims=True) / weights.sum(axis=1, keepdims=True)
        else:
            pooled = x.mean(axis=1, keepdims=True)

        x = mx.concatenate([pooled, x], axis=1)
        x = x + self.positional_embedding[None, :, :].astype(x.dtype)

        q = self.q_proj(x[:, :1, :])
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, _, C = q.shape
        H, D = self.num_heads, self.head_dim
        L = k.shape[1]

        q = q.reshape(B, 1, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, D).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, C)
        return self.c_proj(out)[:, 0, :]


class FinalLayer(nn.Module):
    """Final norm + linear, drops the first token (timestep embedding)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = PreciseLayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm_final(x)
        x = x[:, 1:]  # drop the prepended timestep token
        return self.linear(x)


class HunYuanDiTPlain(nn.Module):
    """Full HunYuanDiT with U-skip connections and MoE in last N blocks."""

    def __init__(
        self,
        input_size: int = 4096,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        qk_norm: bool = True,
        text_len: int = 1370,
        qkv_bias: bool = False,
        with_decoupled_ca: bool = False,
        additional_cond_hidden_state: int = 768,
        use_pos_emb: bool = False,
        use_attention_pooling: bool = False,
        guidance_cond_proj_dim: int = None,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.text_len = text_len
        self.with_decoupled_ca = with_decoupled_ca
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(
            hidden_size,
            hidden_size * 4,
            cond_proj_dim=guidance_cond_proj_dim,
        )

        if self.use_pos_emb:
            self.pos_embed = mx.zeros((1, input_size, hidden_size))

        if use_attention_pooling:
            self.pooler = AttentionPool(text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = _SequentialProjector(1024, hidden_size * 4, hidden_size)

        if with_decoupled_ca:
            self.additional_cond_proj = _SequentialProjector(
                additional_cond_hidden_state,
                hidden_size * 4,
                1024,
            )

        self.blocks = [
            HunYuanDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                text_states_dim=context_dim,
                qk_norm=qk_norm,
                qkv_bias=qkv_bias,
                skip_connection=(layer > depth // 2),
                use_moe=(depth - layer <= num_moe_layers),
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for layer in range(depth)
        ]

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def __call__(self, x: mx.array, t: mx.array, contexts: dict, **kwargs) -> mx.array:
        """
        Args:
            x: (B, num_latents, in_channels) noisy latents
            t: (B,) timesteps in [0, 1000] range (sigma * num_train_timesteps)
            contexts: dict with 'main' key → (B, text_len, context_dim)
        Returns:
            (B, num_latents, in_channels) noise prediction
        """
        cond = contexts['main']

        t_emb = self.t_embedder(t, condition=kwargs.get("guidance_cond"))  # (B, 1, hidden_size)
        x = self.x_embedder(x)  # (B, num_latents, hidden_size)

        if self.use_pos_emb:
            x = x + self.pos_embed.astype(x.dtype)

        if self.use_attention_pooling:
            c = t_emb + self.extra_embedder(self.pooler(cond, None))[:, None, :]
        else:
            c = t_emb

        if self.with_decoupled_ca and "additional" in contexts:
            additional_cond = self.additional_cond_proj(contexts["additional"])
            cond = mx.concatenate([cond, additional_cond], axis=1)

        # Prepend conditioning token
        x = mx.concatenate([c, x], axis=1)  # (B, 1+num_latents, hidden_size)

        # U-skip: first half saves, second half consumes
        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x
