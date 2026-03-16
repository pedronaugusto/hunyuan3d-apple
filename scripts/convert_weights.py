#!/usr/bin/env python3
"""
Convert Hunyuan3D .ckpt weights to .safetensors for MLX zero-copy loading.

Usage:
    python scripts/convert_weights.py --weights-dir weights/tencent/Hunyuan3D-2.1

This creates:
    hunyuan3d-dit-v2-1/model.safetensors   (DiT model weights)
    hunyuan3d-dit-v2-1/conditioner.safetensors (DINOv2 conditioner weights)
    hunyuan3d-vae-v2-1/model.safetensors   (VAE + geo decoder weights)
"""
import argparse
import os
import torch
from safetensors.torch import save_file


def convert_dit(weights_dir: str):
    dit_dir = os.path.join(weights_dir, 'hunyuan3d-dit-v2-1')
    ckpt_path = os.path.join(dit_dir, 'model.fp16.ckpt')

    if not os.path.exists(ckpt_path):
        print(f"  Skipping DiT: {ckpt_path} not found")
        return

    st_model = os.path.join(dit_dir, 'model.safetensors')
    st_cond = os.path.join(dit_dir, 'conditioner.safetensors')

    if os.path.exists(st_model) and os.path.exists(st_cond):
        print(f"  DiT safetensors already exist, skipping")
        return

    print(f"  Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # The DiT ckpt has nested dicts: 'model', 'vae', 'conditioner'
    # Each value is an OrderedDict of tensors
    model_state = {}
    cond_state = {}
    vae_from_dit = {}

    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        # Nested format: top-level keys are section names
        model_state = {k: v.float().contiguous() for k, v in ckpt['model'].items()}
        if 'conditioner' in ckpt and isinstance(ckpt['conditioner'], dict):
            cond_state = {k: v.float().contiguous() for k, v in ckpt['conditioner'].items()}
        if 'vae' in ckpt and isinstance(ckpt['vae'], dict):
            vae_from_dit = {k: v.float().contiguous() for k, v in ckpt['vae'].items()}
    else:
        # Flat format: keys are prefixed with section
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor):
                v = v.float().contiguous()
                if k.startswith('conditioner.') or k.startswith('main_image_encoder.'):
                    cond_state[k] = v
                else:
                    model_state[k] = v

    if model_state and not os.path.exists(st_model):
        print(f"  Saving {st_model} ({len(model_state)} tensors)...")
        save_file(model_state, st_model)

    if cond_state and not os.path.exists(st_cond):
        print(f"  Saving {st_cond} ({len(cond_state)} tensors)...")
        save_file(cond_state, st_cond)

    # Also save VAE weights from DiT ckpt if the VAE dir doesn't have safetensors yet
    if vae_from_dit:
        vae_dir = os.path.join(weights_dir, 'hunyuan3d-vae-v2-1')
        os.makedirs(vae_dir, exist_ok=True)
        vae_st = os.path.join(vae_dir, 'model.safetensors')
        if not os.path.exists(vae_st):
            print(f"  Saving VAE from DiT ckpt: {vae_st} ({len(vae_from_dit)} tensors)...")
            save_file(vae_from_dit, vae_st)

    del ckpt
    print("  DiT conversion done.")


def convert_vae(weights_dir: str):
    vae_dir = os.path.join(weights_dir, 'hunyuan3d-vae-v2-1')
    ckpt_path = os.path.join(vae_dir, 'model.fp16.ckpt')

    if not os.path.exists(ckpt_path):
        print(f"  Skipping VAE: {ckpt_path} not found")
        return

    st_path = os.path.join(vae_dir, 'model.safetensors')
    if os.path.exists(st_path):
        print(f"  VAE safetensors already exist, skipping")
        return

    print(f"  Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    state = {k: v.float().contiguous() for k, v in ckpt.items()}

    print(f"  Saving {st_path} ({len(state)} tensors)...")
    save_file(state, st_path)

    del ckpt
    print("  VAE conversion done.")


def main():
    parser = argparse.ArgumentParser(description="Convert Hunyuan3D weights to safetensors")
    parser.add_argument('--weights-dir', type=str,
                        default='weights/tencent/Hunyuan3D-2.1',
                        help='Path to Hunyuan3D weights directory')
    args = parser.parse_args()

    print(f"Converting weights in {args.weights_dir}...")
    convert_dit(args.weights_dir)
    convert_vae(args.weights_dir)
    print("All done.")


if __name__ == '__main__':
    main()
