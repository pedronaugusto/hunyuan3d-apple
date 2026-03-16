"""
Mixture of Experts for Hunyuan DiT in MLX.
Ported from hy3dshape/hy3dshape/models/denoisers/moe_layers.py

Gate routing uses PyTorch CPU to guarantee identical expert selection
with the upstream model (float32 Metal matmul can flip borderline
top-k decisions, causing ~0.007 hidden-state spikes per MoE block).
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class MoEGate(nn.Module):
    """Softmax gating with top-k expert selection.

    Gate logits are computed via PyTorch CPU to match upstream expert
    routing exactly.  The gate tensor is tiny (N x 8), so the
    MLX->numpy->torch->numpy->MLX round-trip adds negligible overhead.
    """

    def __init__(self, embed_dim: int, num_experts: int = 8,
                 num_experts_per_tok: int = 2):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.weight = mx.zeros((num_experts, embed_dim))

    def __call__(self, hidden_states: mx.array):
        """
        Args:
            hidden_states: (B, S, D)
        Returns:
            topk_idx: (B*S, top_k) expert indices
            topk_weight: (B*S, top_k) expert weights
        """
        bsz, seq_len, h = hidden_states.shape
        mx.eval(hidden_states)

        # Move to PyTorch CPU for gate decision (matches upstream exactly)
        x_np = np.array(hidden_states.reshape(-1, h))
        w_np = np.array(self.weight)
        with torch.no_grad():
            x_pt = torch.from_numpy(x_np)
            w_pt = torch.from_numpy(w_np)
            logits = F.linear(x_pt, w_pt, None)
            scores = logits.softmax(dim=-1)
            topk_weight_pt, topk_idx_pt = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )

        topk_idx = mx.array(topk_idx_pt.numpy())
        topk_weight = mx.array(topk_weight_pt.numpy())
        return topk_idx, topk_weight


class FeedForward(nn.Module):
    """GELU feedforward matching diffusers FeedForward."""

    def __init__(self, dim: int, inner_dim: int, bias: bool = True):
        super().__init__()
        # diffusers FeedForward with activation_fn='gelu': Linear → GELU → Linear
        self.net_0_proj = nn.Linear(dim, inner_dim, bias=bias)
        self.net_2 = nn.Linear(inner_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net_2(nn.gelu(self.net_0_proj(x)))


class MoEBlock(nn.Module):
    """MoE block: gated expert routing + shared expert."""

    def __init__(self, dim: int, num_experts: int = 8, moe_top_k: int = 2,
                 ff_inner_dim: int = None, ff_bias: bool = True, **kwargs):
        super().__init__()
        if ff_inner_dim is None:
            ff_inner_dim = dim * 4
        self.moe_top_k = moe_top_k
        self.num_experts = num_experts
        self.experts = [FeedForward(dim, ff_inner_dim, bias=ff_bias)
                        for _ in range(num_experts)]
        self.gate = MoEGate(embed_dim=dim, num_experts=num_experts,
                            num_experts_per_tok=moe_top_k)
        self.shared_experts = FeedForward(dim, ff_inner_dim, bias=ff_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)

        x = hidden_states.reshape(-1, hidden_states.shape[-1])  # (N, D)
        N, D = x.shape

        # Match upstream inference structure:
        # 1. evaluate every expert on every token
        # 2. build a flat weighted one-hot routing tensor from the top-k indices
        # 3. reduce with the same per-token [N, E] weight layout
        expert_outputs = mx.stack([expert(x) for expert in self.experts], axis=0)
        E = self.num_experts
        flat_topk_idx = topk_idx.reshape(-1)                     # (N * top_k,)
        flat_topk_weight = topk_weight.reshape(-1, 1)           # (N * top_k, 1)
        expert_range = mx.arange(E)[None, :]                    # (1, E)
        one_hot = (flat_topk_idx[:, None] == expert_range).astype(flat_topk_weight.dtype)
        weighted = one_hot * flat_topk_weight                   # (N * top_k, E)
        per_token_weights = weighted.reshape(N, self.moe_top_k, E).sum(axis=1)  # (N, E)

        # Upstream uses an einsum-style weighted reduction over [E, N, D].
        result = mx.einsum(
            "ne,end->nd",
            per_token_weights.astype(expert_outputs.dtype),
            expert_outputs,
        )

        y = result.reshape(*orig_shape)
        y = y + self.shared_experts(identity)
        return y
