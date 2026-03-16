"""
Test actual weight loading for MLX texture pipeline components.
Verifies that model weights load correctly from disk.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mlx.core as mx


MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights', 'tencent', 'Hunyuan3D-2.1')


def test_esrgan_load():
    """Test ESRGAN loading from .pth."""
    from mlx_backend.esrgan import MlxESRGAN

    pth_path = os.path.join(os.path.dirname(__file__), '..', 'hy3dpaint',
                            'ckpt', 'RealESRGAN_x4plus.pth')
    if not os.path.exists(pth_path):
        print("  ESRGAN: SKIP (no weights)")
        return

    model = MlxESRGAN.from_pth(pth_path)

    # Test inference
    x = mx.random.normal((1, 64, 64, 3)) * 0.5 + 0.5
    out = model(x)
    mx.eval(out)
    assert out.shape == (1, 256, 256, 3)
    print(f"  ESRGAN: OK ({out.shape}, range [{float(out.min()):.2f}, {float(out.max()):.2f}])")


def test_vae_load():
    """Test VAE loading from diffusers format."""
    from mlx_backend.vae_kl import MlxAutoencoderKL

    vae_dir = os.path.join(MODEL_DIR, "hunyuan3d-paintpbr-v2-1", "vae")
    if not os.path.isdir(vae_dir):
        print(f"  VAE: SKIP (no dir: {vae_dir})")
        return

    vae = MlxAutoencoderKL.from_diffusers(vae_dir)

    # Test encode/decode
    x = mx.random.normal((1, 256, 256, 3))
    z = vae.encode(x)
    mx.eval(z)
    print(f"  VAE encode: {x.shape} -> {z.shape}")

    recon = vae.decode(z)
    mx.eval(recon)
    print(f"  VAE decode: {z.shape} -> {recon.shape}")
    print(f"  VAE: OK")


def test_dino_load():
    """Test DINOv2-giant loading from HuggingFace cache."""
    from mlx_backend.dinov2 import MlxDINOv2
    from mlx_backend import remap_dinov2_weights, load_safetensors
    from PIL import Image
    import glob

    model = MlxDINOv2(
        dim=1536, num_heads=24, num_layers=40,
        patch_size=14, image_size=518,
    )

    # Find weights
    candidates = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--dinov2-giant/snapshots/*/model.safetensors"
    )))
    if not candidates:
        print("  DINOv2-giant: SKIP (not in HF cache)")
        return

    raw = load_safetensors(candidates[0])
    weights = remap_dinov2_weights(raw)
    model.load_weights(list(weights.items()))

    # Test
    img = Image.fromarray(np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8))
    out = model([img])
    mx.eval(out)

    num_patches = (518 // 14) ** 2
    assert out.shape == (1, 1 + num_patches, 1536), f"Got {out.shape}"
    print(f"  DINOv2-giant: OK shape={out.shape}")


def count_params(model):
    """Count total parameters in a model."""
    total = 0
    for k, v in model.parameters().items() if hasattr(model, 'parameters') else []:
        if isinstance(v, mx.array):
            total += v.size
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, mx.array):
                    total += vv.size
    # Simpler: just sum all leaf arrays
    import mlx.nn as nn
    leaves = nn.utils.tree_flatten(model.parameters())
    return sum(v.size for _, v in leaves)


if __name__ == "__main__":
    tests = [
        ("ESRGAN", test_esrgan_load),
        ("VAE", test_vae_load),
        ("DINOv2-giant", test_dino_load),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
