"""
MLX backend for Hunyuan3D on Apple Silicon.

Weight loading uses mx.load() for zero-copy safetensors reads.
Weights must first be converted from .ckpt to .safetensors using
scripts/convert_weights.py.
"""
import re
import mlx.core as mx


def load_safetensors(path: str) -> dict:
    """Load safetensors weights via MLX (zero-copy on unified memory)."""
    return mx.load(path)


def _remap_sequential(key: str) -> str:
    """
    Convert PyTorch nn.Sequential index notation to MLX.
    e.g. 'mlp.0.weight' → 'mlp.layers.0.weight'
    """
    sequential_containers = [
        'extra_embedder', 'additional_cond_proj', 'default_modulation',
        'mlp.mlp',
    ]
    for container in sequential_containers:
        pattern = re.escape(container) + r'\.(\d+)\.'
        replacement = container + r'.layers.\1.'
        key = re.sub(pattern, replacement, key)
        pattern_end = re.escape(container) + r'\.(\d+)$'
        replacement_end = container + r'.layers.\1'
        key = re.sub(pattern_end, replacement_end, key)
    return key


def remap_dit_weights(weights: dict) -> dict:
    """
    Remap DiT .ckpt/.safetensors keys to MLX module paths.

    PyTorch keys:
        blocks.N.attn1.to_q.weight → blocks.N.attn1.to_q.weight (same)
        blocks.N.moe.experts.M.net.0.proj.weight → blocks.N.moe.experts.M.net_0_proj.weight
        blocks.N.moe.shared_experts.net.0.proj.weight → blocks.N.moe.shared_experts.net_0_proj.weight
        t_embedder.mlp.0.weight → t_embedder.mlp.layers.0.weight
    """
    remapped = {}
    for k, v in weights.items():
        new_k = k

        # nn.Sequential → .layers.N.
        new_k = _remap_sequential(new_k)

        # diffusers FeedForward: net.0.proj → net_0_proj, net.2 → net_2
        new_k = re.sub(r'\.net\.0\.proj\.', '.net_0_proj.', new_k)
        new_k = re.sub(r'\.net\.2\.', '.net_2.', new_k)

        # t_embedder.mlp Sequential: 0 → fc1, 2 → fc2 (1 is GELU activation)
        new_k = re.sub(r't_embedder\.mlp\.0\.', 't_embedder.mlp.fc1.', new_k)
        new_k = re.sub(r't_embedder\.mlp\.2\.', 't_embedder.mlp.fc2.', new_k)

        # MoEGate uses raw Parameter 'weight', keep as-is

        remapped[new_k] = v
    return remapped


def remap_vae_weights(weights: dict) -> dict:
    """
    Remap VAE .ckpt/.safetensors keys to MLX module paths.

    PyTorch VAE structure:
        post_kl.weight → post_kl.weight
        transformer.resblocks.N.attn.c_qkv.weight → resblocks.N.attn.c_qkv.weight
        geo_decoder.* → kept for geo_decoder loading
    """
    remapped = {}
    for k, v in weights.items():
        new_k = k
        # The VAETransformer has post_kl and resblocks from transformer
        # In the .ckpt, keys are: post_kl.*, transformer.resblocks.N.*
        # Our MLX VAETransformer has: post_kl.*, resblocks.N.*
        if new_k.startswith('transformer.'):
            new_k = new_k[len('transformer.'):]
        # Strip extra .attention. level: attn.attention.q_norm → attn.q_norm
        new_k = new_k.replace('.attn.attention.', '.attn.')
        remapped[new_k] = v
    return remapped


def remap_geo_decoder_weights(weights: dict) -> dict:
    """Remap geo decoder weights."""
    remapped = {}
    for k, v in weights.items():
        new_k = k
        if new_k.startswith('geo_decoder.'):
            new_k = new_k[len('geo_decoder.'):]
        # Strip extra .attention. level: attn.attention.q_norm → attn.q_norm
        new_k = new_k.replace('.attn.attention.', '.attn.')
        remapped[new_k] = v
    return remapped


def remap_dinov2_weights(weights: dict) -> dict:
    """
    Remap HuggingFace DINOv2 weight keys to our flat MLX structure.

    HF keys like:
        dinov2.embeddings.patch_embeddings.projection.weight → embeddings.weight
        dinov2.encoder.layer.N.attention.attention.query.weight → layers.N.attn.qkv.weight (fused)
    """
    remapped = {}

    # Collect Q/K/V per layer for fusion
    qkv_parts = {}  # layer_idx -> {'q_weight':, 'k_weight':, ...}

    for k, v in weights.items():
        # Strip prefix: HF uses 'main_image_encoder.model.' in ckpt
        k2 = k
        for prefix in ['main_image_encoder.model.', 'dinov2.']:
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
                break

        # Patch embeddings
        if 'patch_embeddings.projection.weight' in k2:
            # HF: (dim, 3, P, P) -> our: (dim, P, P, 3)
            v = v.transpose(0, 2, 3, 1)
            remapped['embeddings.weight'] = v
            continue
        if 'patch_embeddings.projection.bias' in k2:
            remapped['embeddings.bias'] = v
            continue
        if k2 == 'embeddings.cls_token':
            remapped['embeddings.cls_token'] = v
            continue
        if k2 == 'embeddings.position_embeddings':
            remapped['embeddings.position_embeddings'] = v
            continue
        if k2 == 'embeddings.mask_token':
            continue  # Not used in inference

        # Layer norm (final)
        if k2.startswith('layernorm.'):
            suffix = k2[len('layernorm.'):]
            remapped[f'layernorm.{suffix}'] = v
            continue

        # Encoder layers
        m = re.match(r'encoder\.layer\.(\d+)\.(.*)', k2)
        if m:
            idx = int(m.group(1))
            rest = m.group(2)
            _remap_dinov2_layer(idx, rest, v, remapped, qkv_parts)
            continue

    # Fuse Q/K/V into QKV
    for idx, parts in qkv_parts.items():
        for param in ['weight', 'bias']:
            q = parts.get(f'q_{param}')
            k = parts.get(f'k_{param}')
            v_val = parts.get(f'v_{param}')
            if q is not None and k is not None and v_val is not None:
                fused = mx.concatenate([q, k, v_val], axis=0)
                remapped[f'layers.{idx}.attn.qkv.{param}'] = fused

    return remapped


def _remap_dinov2_layer(idx: int, rest: str, v, remapped: dict, qkv_parts: dict):
    prefix = f'layers.{idx}'

    # Self-attention Q/K/V (need to fuse)
    if rest.startswith('attention.attention.'):
        attn_rest = rest[len('attention.attention.'):]
        if attn_rest.startswith('query.'):
            param = attn_rest[len('query.'):]
            qkv_parts.setdefault(idx, {})[f'q_{param}'] = v
        elif attn_rest.startswith('key.'):
            param = attn_rest[len('key.'):]
            qkv_parts.setdefault(idx, {})[f'k_{param}'] = v
        elif attn_rest.startswith('value.'):
            param = attn_rest[len('value.'):]
            qkv_parts.setdefault(idx, {})[f'v_{param}'] = v
        return

    if rest.startswith('attention.output.dense.'):
        param = rest[len('attention.output.dense.'):]
        remapped[f'{prefix}.attn.proj.{param}'] = v
        return

    # Layer norms
    if rest.startswith('norm1.') or rest.startswith('layer_norm1.'):
        suffix = rest.split('.', 1)[1]
        remapped[f'{prefix}.norm1.{suffix}'] = v
        return
    if rest.startswith('norm2.') or rest.startswith('layer_norm2.'):
        suffix = rest.split('.', 1)[1]
        remapped[f'{prefix}.norm2.{suffix}'] = v
        return

    # Layer scale
    if rest.startswith('layer_scale1.'):
        param = rest[len('layer_scale1.'):]
        remapped[f'{prefix}.layer_scale1'] = v
        return
    if rest.startswith('layer_scale2.'):
        param = rest[len('layer_scale2.'):]
        remapped[f'{prefix}.layer_scale2'] = v
        return

    # MLP (handles fc1/fc2, weights_in/weights_out, intermediate/output naming)
    if rest.startswith('mlp.fc1.') or rest.startswith('mlp.weights_in.') or rest.startswith('intermediate.dense.'):
        if rest.startswith('mlp.fc1.'):
            suffix = rest[len('mlp.fc1.'):]
        elif rest.startswith('mlp.weights_in.'):
            suffix = rest[len('mlp.weights_in.'):]
        else:
            suffix = rest[len('intermediate.dense.'):]
        remapped[f'{prefix}.mlp.fc1.{suffix}'] = v
        return
    if rest.startswith('mlp.fc2.') or rest.startswith('mlp.weights_out.') or rest.startswith('output.dense.'):
        if rest.startswith('mlp.fc2.'):
            suffix = rest[len('mlp.fc2.'):]
        elif rest.startswith('mlp.weights_out.'):
            suffix = rest[len('mlp.weights_out.'):]
        else:
            suffix = rest[len('output.dense.'):]
        remapped[f'{prefix}.mlp.fc2.{suffix}'] = v
        return
