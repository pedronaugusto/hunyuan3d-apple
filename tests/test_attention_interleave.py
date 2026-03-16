"""
Numerical verification that MLX attention interleaving matches upstream PyTorch.

Tests all 3 attention variants:
1. DiT SelfAttention (Q/K/V interleaved)
2. DiT CrossAttention (K/V interleaved, Q simple)
3. VAE SelfAttention (fused QKV interleaved)
"""
import sys
import numpy as np

import torch
import torch.nn as torch_nn

import mlx.core as mx
import mlx.nn as mlx_nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from mlx_backend.attention import SelfAttention as MLXSelfAttention
from mlx_backend.attention import CrossAttention as MLXCrossAttention
from mlx_backend.vae_transformer import VAESelfAttention as MLXVAESelfAttention


# ── Upstream PyTorch reference implementations ──────────────────────────


class PTSelfAttention(torch_nn.Module):
    """Upstream DiT self-attention with cat-view-split interleaving."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = torch_nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch_nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch_nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = torch_nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # Upstream: cat → view(B,N,H,3D) → chunk
        qkv = torch.cat([q, k, v], dim=-1).view(B, N, H, 3 * D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class PTCrossAttention(torch_nn.Module):
    """Upstream DiT cross-attention: Q simple reshape, K/V interleaved."""

    def __init__(self, qdim, kdim, num_heads, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = qdim // num_heads
        self.to_q = torch_nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = torch_nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = torch_nn.Linear(kdim, qdim, bias=qkv_bias)
        self.out_proj = torch_nn.Linear(qdim, qdim, bias=True)

    def forward(self, x, y):
        B, S1, _ = x.shape
        _, S2, _ = y.shape
        H, D = self.num_heads, self.head_dim
        q = self.to_q(x).view(B, S1, H, D)  # simple reshape
        k = self.to_k(y)
        v = self.to_v(y)
        kv = torch.cat([k, v], dim=-1).view(B, S2, H, 2 * D)
        k, v = kv.split(D, dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S1, -1)
        return self.out_proj(out)


class PTVAESelfAttention(torch_nn.Module):
    """Upstream VAE QKVMultiheadAttention with interleaving."""

    def __init__(self, width, heads, qkv_bias=False, qk_norm=False):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.c_qkv = torch_nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = torch_nn.Linear(width, width)

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.heads, self.head_dim
        qkv = self.c_qkv(x).view(B, N, H, 3 * D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.c_proj(out)


# ── Helpers ─────────────────────────────────────────────────────────────


def copy_linear_pt_to_mlx(pt_linear, mlx_linear):
    """Copy weights from PyTorch Linear to MLX Linear."""
    mlx_linear.weight = mx.array(pt_linear.weight.detach().numpy())
    if pt_linear.bias is not None:
        mlx_linear.bias = mx.array(pt_linear.bias.detach().numpy())


def compare(name, pt_out, mlx_out, atol=1e-4):
    pt_np = pt_out.detach().numpy()
    mlx_np = np.array(mlx_out)
    max_diff = np.max(np.abs(pt_np - mlx_np))
    match = np.allclose(pt_np, mlx_np, atol=atol)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6e} (atol={atol})")
    return match


# ── Tests ───────────────────────────────────────────────────────────────


def test_self_attention():
    print("Test 1: DiT SelfAttention interleaving")
    dim, heads = 64, 4  # small for fast test
    B, N = 1, 8

    torch.manual_seed(42)
    pt = PTSelfAttention(dim, heads, qkv_bias=True, qk_norm=False)
    mlx_mod = MLXSelfAttention(dim, heads, qkv_bias=True, qk_norm=False)

    # Copy weights
    copy_linear_pt_to_mlx(pt.to_q, mlx_mod.to_q)
    copy_linear_pt_to_mlx(pt.to_k, mlx_mod.to_k)
    copy_linear_pt_to_mlx(pt.to_v, mlx_mod.to_v)
    copy_linear_pt_to_mlx(pt.out_proj, mlx_mod.out_proj)

    x_np = np.random.randn(B, N, dim).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_mlx = mx.array(x_np)

    with torch.no_grad():
        pt_out = pt(x_pt)
    mlx_out = mlx_mod(x_mlx)
    mx.eval(mlx_out)

    return compare("SelfAttention", pt_out, mlx_out)


def test_cross_attention():
    print("Test 2: DiT CrossAttention interleaving")
    qdim, kdim, heads = 64, 32, 4
    B, S1, S2 = 1, 8, 6

    torch.manual_seed(42)
    pt = PTCrossAttention(qdim, kdim, heads, qkv_bias=True, qk_norm=False)
    mlx_mod = MLXCrossAttention(qdim, kdim, heads, qkv_bias=True, qk_norm=False)

    copy_linear_pt_to_mlx(pt.to_q, mlx_mod.to_q)
    copy_linear_pt_to_mlx(pt.to_k, mlx_mod.to_k)
    copy_linear_pt_to_mlx(pt.to_v, mlx_mod.to_v)
    copy_linear_pt_to_mlx(pt.out_proj, mlx_mod.out_proj)

    x_np = np.random.randn(B, S1, qdim).astype(np.float32)
    y_np = np.random.randn(B, S2, kdim).astype(np.float32)
    x_pt, y_pt = torch.from_numpy(x_np), torch.from_numpy(y_np)
    x_mlx, y_mlx = mx.array(x_np), mx.array(y_np)

    with torch.no_grad():
        pt_out = pt(x_pt, y_pt)
    mlx_out = mlx_mod(x_mlx, y_mlx)
    mx.eval(mlx_out)

    return compare("CrossAttention", pt_out, mlx_out)


def test_vae_self_attention():
    print("Test 3: VAE SelfAttention interleaving")
    width, heads = 64, 4
    B, N = 1, 8

    torch.manual_seed(42)
    pt = PTVAESelfAttention(width, heads, qkv_bias=False, qk_norm=False)
    mlx_mod = MLXVAESelfAttention(width, heads, qkv_bias=False, qk_norm=False)

    copy_linear_pt_to_mlx(pt.c_qkv, mlx_mod.c_qkv)
    copy_linear_pt_to_mlx(pt.c_proj, mlx_mod.c_proj)

    x_np = np.random.randn(B, N, width).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_mlx = mx.array(x_np)

    with torch.no_grad():
        pt_out = pt(x_pt)
    mlx_out = mlx_mod(x_mlx)
    mx.eval(mlx_out)

    return compare("VAE SelfAttention", pt_out, mlx_out)


def test_interleave_matters():
    """Verify that the old (wrong) simple reshape gives DIFFERENT results,
    confirming the interleaving fix is not a no-op."""
    print("Test 4: Verify interleaving != simple reshape (sanity check)")
    dim, heads = 64, 4
    B, N = 1, 8
    H, D = heads, dim // heads

    np.random.seed(123)
    x = np.random.randn(B, N, dim).astype(np.float32)
    q_w = np.random.randn(dim, dim).astype(np.float32)
    k_w = np.random.randn(dim, dim).astype(np.float32)
    v_w = np.random.randn(dim, dim).astype(np.float32)

    q_proj = x @ q_w.T
    k_proj = x @ k_w.T
    v_proj = x @ v_w.T

    # Old way: simple reshape
    q_old = q_proj.reshape(B, N, H, D)
    k_old = k_proj.reshape(B, N, H, D)
    v_old = v_proj.reshape(B, N, H, D)

    # New way: interleaved
    qkv = np.concatenate([q_proj, k_proj, v_proj], axis=-1)
    qkv = qkv.reshape(B, N, H, 3 * D)
    q_new = qkv[:, :, :, :D]
    k_new = qkv[:, :, :, D:2*D]
    v_new = qkv[:, :, :, 2*D:]

    q_diff = not np.allclose(q_old, q_new)
    k_diff = not np.allclose(k_old, k_new)
    v_diff = not np.allclose(v_old, v_new)
    differs = q_diff and k_diff and v_diff

    status = "PASS" if differs else "FAIL"
    print(f"  [{status}] Old != New: q={q_diff}, k={k_diff}, v={v_diff}")
    return differs


if __name__ == "__main__":
    results = [
        test_self_attention(),
        test_cross_attention(),
        test_vae_self_attention(),
        test_interleave_matters(),
    ]
    print()
    if all(results):
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
