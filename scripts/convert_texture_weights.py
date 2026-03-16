#!/usr/bin/env python3
"""
Convert Hunyuan3D texture pipeline weights to MLX safetensors format.

Converts:
- UNet2p5D: diffusion_pytorch_model.bin → safetensors
- VAE: diffusion_pytorch_model.safetensors (key remapping)
- ESRGAN: RealESRGAN_x4plus.pth → safetensors
- DINOv2-giant: HuggingFace → safetensors
"""
import argparse
import os
import sys
import numpy as np

def convert_conv_weight(w):
    """PyTorch Conv2d (O,I,H,W) → MLX Conv2d (O,H,W,I)."""
    return np.transpose(w, (0, 2, 3, 1))


def convert_unet_weights(src_path: str, dst_path: str):
    """Convert UNet2p5D diffusion_pytorch_model.bin to safetensors."""
    import torch
    import mlx.core as mx

    print(f"Loading UNet from {src_path}...")
    state = torch.load(src_path, map_location='cpu', weights_only=True)

    remapped = {}
    for k, v in state.items():
        v_np = v.float().numpy()

        # Conv weights need transposition
        if 'conv' in k and v_np.ndim == 4 and v_np.shape[2:] != (1, 1):
            v_np = convert_conv_weight(v_np)
        elif v_np.ndim == 4 and v_np.shape[2] > 1:
            v_np = convert_conv_weight(v_np)

        # Key remapping: diffusers-style → our MLX module paths
        new_k = k
        # GroupNorm in MLX uses same key names
        # Linear/Conv use same key names

        remapped[new_k] = mx.array(v_np)

    mx.savez(dst_path, **remapped)
    print(f"Saved {len(remapped)} UNet weights to {dst_path}")
    print(f"Sample keys: {sorted(remapped.keys())[:5]}")


def convert_esrgan_weights(src_path: str, dst_path: str):
    """Convert RealESRGAN .pth weights to safetensors."""
    import torch
    import mlx.core as mx

    print(f"Loading ESRGAN from {src_path}...")
    state = torch.load(src_path, map_location='cpu', weights_only=True)

    # RealESRGAN .pth has 'params_ema' or 'params' key
    if 'params_ema' in state:
        state = state['params_ema']
    elif 'params' in state:
        state = state['params']

    remapped = {}
    for k, v in state.items():
        v_np = v.float().numpy()

        # Conv2d weights: (O,I,H,W) → (O,H,W,I)
        if v_np.ndim == 4:
            v_np = convert_conv_weight(v_np)

        # Remap keys: PyTorch RRDBNet → our MLX structure
        # body.N.rdb1.conv1.weight → body.layers.N.rdb1.conv1.weight
        new_k = k

        remapped[new_k] = mx.array(v_np)

    mx.savez(dst_path, **remapped)
    print(f"Saved {len(remapped)} ESRGAN weights to {dst_path}")


def convert_vae_weights(src_dir: str, dst_path: str):
    """Convert diffusers AutoencoderKL weights to safetensors."""
    import torch
    import mlx.core as mx
    from safetensors.torch import load_file

    st_path = os.path.join(src_dir, 'diffusion_pytorch_model.safetensors')
    if os.path.exists(st_path):
        print(f"Loading VAE from {st_path}...")
        state = load_file(st_path)
    else:
        bin_path = os.path.join(src_dir, 'diffusion_pytorch_model.bin')
        print(f"Loading VAE from {bin_path}...")
        state = torch.load(bin_path, map_location='cpu', weights_only=True)

    remapped = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v_np = v.float().numpy()
        else:
            v_np = np.array(v)

        if v_np.ndim == 4:
            v_np = convert_conv_weight(v_np)

        remapped[k] = mx.array(v_np)

    mx.savez(dst_path, **remapped)
    print(f"Saved {len(remapped)} VAE weights to {dst_path}")


def convert_dinov2_giant_weights(model_name: str, dst_path: str):
    """Convert HuggingFace DINOv2-giant to safetensors."""
    import torch
    import mlx.core as mx
    from transformers import AutoModel

    print(f"Loading DINOv2 from {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    state = model.state_dict()

    remapped = {}
    qkv_parts = {}

    for k, v in state.items():
        v_np = v.float().numpy()

        # Strip 'dinov2.' prefix if present
        k2 = k
        if k2.startswith('dinov2.'):
            k2 = k2[len('dinov2.'):]

        # Patch embeddings
        if 'patch_embeddings.projection.weight' in k2:
            v_np = np.transpose(v_np, (0, 2, 3, 1))
            remapped['embeddings.weight'] = mx.array(v_np)
            continue
        if 'patch_embeddings.projection.bias' in k2:
            remapped['embeddings.bias'] = mx.array(v_np)
            continue
        if k2 == 'embeddings.cls_token':
            remapped['embeddings.cls_token'] = mx.array(v_np)
            continue
        if k2 == 'embeddings.position_embeddings':
            remapped['embeddings.position_embeddings'] = mx.array(v_np)
            continue
        if k2 == 'embeddings.mask_token':
            continue

        # Layernorm
        if k2.startswith('layernorm.'):
            remapped[k2] = mx.array(v_np)
            continue

        # Encoder layers
        import re
        m = re.match(r'encoder\.layer\.(\d+)\.(.*)', k2)
        if m:
            idx = int(m.group(1))
            rest = m.group(2)
            prefix = f'layers.{idx}'

            # Q/K/V for fusion
            if 'attention.attention.query.' in rest:
                param = rest.split('.')[-1]
                qkv_parts.setdefault(idx, {})[f'q_{param}'] = v_np
            elif 'attention.attention.key.' in rest:
                param = rest.split('.')[-1]
                qkv_parts.setdefault(idx, {})[f'k_{param}'] = v_np
            elif 'attention.attention.value.' in rest:
                param = rest.split('.')[-1]
                qkv_parts.setdefault(idx, {})[f'v_{param}'] = v_np
            elif 'attention.output.dense.' in rest:
                param = rest.split('.')[-1]
                remapped[f'{prefix}.attn.proj.{param}'] = mx.array(v_np)
            elif rest.startswith('norm1.') or rest.startswith('layer_norm1.'):
                suffix = rest.split('.', 1)[1]
                remapped[f'{prefix}.norm1.{suffix}'] = mx.array(v_np)
            elif rest.startswith('norm2.') or rest.startswith('layer_norm2.'):
                suffix = rest.split('.', 1)[1]
                remapped[f'{prefix}.norm2.{suffix}'] = mx.array(v_np)
            elif 'layer_scale1' in rest:
                remapped[f'{prefix}.layer_scale1'] = mx.array(v_np)
            elif 'layer_scale2' in rest:
                remapped[f'{prefix}.layer_scale2'] = mx.array(v_np)
            elif rest.startswith('mlp.fc1.') or rest.startswith('intermediate.dense.'):
                param = rest.split('.')[-1]
                remapped[f'{prefix}.mlp.fc1.{param}'] = mx.array(v_np)
            elif rest.startswith('mlp.fc2.') or rest.startswith('output.dense.'):
                param = rest.split('.')[-1]
                remapped[f'{prefix}.mlp.fc2.{param}'] = mx.array(v_np)
            continue

    # Fuse QKV
    for idx, parts in qkv_parts.items():
        for param in ['weight', 'bias']:
            q = parts.get(f'q_{param}')
            k = parts.get(f'k_{param}')
            v_val = parts.get(f'v_{param}')
            if q is not None and k is not None and v_val is not None:
                fused = np.concatenate([q, k, v_val], axis=0)
                remapped[f'layers.{idx}.attn.qkv.{param}'] = mx.array(fused)

    mx.savez(dst_path, **remapped)
    print(f"Saved {len(remapped)} DINOv2-giant weights to {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert texture pipeline weights to MLX")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Root model directory (e.g. tencent/Hunyuan3D-2.1 snapshot)")
    parser.add_argument("--esrgan-path", type=str, default=None,
                        help="Path to RealESRGAN_x4plus.pth")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for safetensors (default: model-dir/mlx_weights)")
    parser.add_argument("--skip-dino", action="store_true",
                        help="Skip DINOv2-giant conversion")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.model_dir, "mlx_weights")
    os.makedirs(out_dir, exist_ok=True)

    # UNet
    unet_src = os.path.join(args.model_dir, "hunyuan3d-paintpbr-v2-1",
                            "diffusion_pytorch_model.bin")
    if os.path.exists(unet_src):
        convert_unet_weights(unet_src, os.path.join(out_dir, "unet.safetensors"))
    else:
        print(f"UNet not found at {unet_src}, skipping")

    # VAE
    vae_dir = os.path.join(args.model_dir, "hunyuan3d-paintpbr-v2-1", "vae")
    if os.path.isdir(vae_dir):
        convert_vae_weights(vae_dir, os.path.join(out_dir, "vae.safetensors"))
    else:
        print(f"VAE not found at {vae_dir}, skipping")

    # ESRGAN
    esrgan_path = args.esrgan_path
    if esrgan_path is None:
        esrgan_path = os.path.join(os.path.dirname(__file__), "..",
                                    "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
    if os.path.exists(esrgan_path):
        convert_esrgan_weights(esrgan_path, os.path.join(out_dir, "esrgan.safetensors"))
    else:
        print(f"ESRGAN not found at {esrgan_path}, skipping")

    # DINOv2-giant
    if not args.skip_dino:
        convert_dinov2_giant_weights("facebook/dinov2-giant",
                                      os.path.join(out_dir, "dinov2_giant.safetensors"))

    print(f"\nAll weights saved to {out_dir}")


if __name__ == "__main__":
    main()
