"""
Convert Hunyuan3D texture UNet weights from PyTorch .bin to MLX format.

Maps upstream key structure to our MLX Basic2p5DTransformerBlock hierarchy.
"""
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mlx.core as mx


def remap_unet_key(key: str) -> str:
    """Remap a single PyTorch UNet weight key to MLX key format.

    Key transformations inside transformer_blocks:
    - transformer.X → base.X  (base BasicTransformerBlock)
    - attn_dino.to_X → dino_to_X
    - attn_multiview.to_X → ma_to_X
    - attn_refview.to_X → ra_to_X
    - transformer.attn1.processor.to_X_mr → mda_processor.to_X_mr
    - attn_refview.processor.to_X_mr → ra_processor.to_X_mr
    - .to_out.0. → .to_out.  (remove Sequential index)
    - ff.net.0. → ff.net_0.  (MLX uses attribute names)
    - ff.net.2. → ff.net_2.
    """
    # Remove .0. from Sequential wrappers like to_out.0.weight
    key = re.sub(r'\.to_out\.0\.', '.to_out.', key)
    key = re.sub(r'\.to_out_mr\.0\.', '.to_out_mr.', key)

    # FF net: PyTorch net.0. / net.2. → MLX net_0. / net_2.
    key = key.replace('.ff.net.0.', '.ff.net_0.')
    key = key.replace('.ff.net.2.', '.ff.net_2.')

    # 2.5D block remapping (order matters — do processor before general)
    key = key.replace('.transformer.attn1.processor.', '.mda_processor.')
    key = key.replace('.transformer.', '.base.')
    key = key.replace('.attn_dino.to_q.', '.dino_to_q.')
    key = key.replace('.attn_dino.to_k.', '.dino_to_k.')
    key = key.replace('.attn_dino.to_v.', '.dino_to_v.')
    key = key.replace('.attn_dino.to_out.', '.dino_to_out.')
    key = key.replace('.attn_multiview.to_q.', '.ma_to_q.')
    key = key.replace('.attn_multiview.to_k.', '.ma_to_k.')
    key = key.replace('.attn_multiview.to_v.', '.ma_to_v.')
    key = key.replace('.attn_multiview.to_out.', '.ma_to_out.')
    key = key.replace('.attn_refview.processor.', '.ra_processor.')
    key = key.replace('.attn_refview.to_q.', '.ra_to_q.')
    key = key.replace('.attn_refview.to_k.', '.ra_to_k.')
    key = key.replace('.attn_refview.to_v.', '.ra_to_v.')
    key = key.replace('.attn_refview.to_out.', '.ra_to_out.')

    return key


def remap_top_level_key(key: str) -> str:
    """Remap top-level MlxUNet2p5D keys.

    unet.image_proj_model_dino.X → image_proj_model_dino.X
    unet.learned_text_clip_X → learned_text_clip_X
    """
    for prefix in ['unet.image_proj_model_dino.', 'unet.learned_text_clip_']:
        if key.startswith(prefix):
            return key[len('unet.'):]
    return key


def convert_unet_weights(bin_path: str) -> dict:
    """Convert PyTorch UNet .bin to MLX weight dict."""
    import torch
    state = torch.load(bin_path, map_location="cpu", weights_only=False)

    converted = {}
    for key, value in state.items():
        arr = value.float().numpy()
        new_key = remap_unet_key(key)
        new_key = remap_top_level_key(new_key)

        # Conv2d: PyTorch OIHW → MLX OHWI
        if arr.ndim == 4 and 'weight' in key:
            arr = np.transpose(arr, (0, 2, 3, 1))

        converted[new_key] = mx.array(arr)

    return converted


def verify_weights(converted: dict, verbose: bool = False):
    """Verify weights match MLX model structure."""
    from mlx_backend.unet_blocks import UNet2DConditionModel
    from mlx_backend.unet2p5d import MlxUNet2p5D
    import mlx.nn as nn

    base = UNet2DConditionModel(
        in_channels=12, out_channels=4,
        block_out_channels=[320, 640, 1280, 1280],
        layers_per_block=2, cross_attention_dim=1024,
        attention_head_dim=[5, 10, 20, 20],
    )
    model = MlxUNet2p5D(base, use_dino=True)

    model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))

    # Split by prefix
    main_keys = {k for k in converted if k.startswith('unet.')}
    dual_keys = {k for k in converted if k.startswith('unet_dual.')}
    other_keys = {k for k in converted if not k.startswith('unet.') and not k.startswith('unet_dual.')}

    print(f"Checkpoint: {len(main_keys)} main + {len(dual_keys)} dual + {len(other_keys)} other")
    print(f"Model expects: {len(model_keys)} parameters")

    # Check main UNet + top-level keys match
    all_ckpt_keys = main_keys | other_keys
    matched = all_ckpt_keys & model_keys
    in_ckpt_not_model = all_ckpt_keys - model_keys
    in_model_not_ckpt = model_keys - all_ckpt_keys

    print(f"Matched: {len(matched)}")
    print(f"In ckpt but not model: {len(in_ckpt_not_model)}")
    print(f"In model but not ckpt: {len(in_model_not_ckpt)}")

    if verbose or in_ckpt_not_model:
        if in_ckpt_not_model:
            print("\nMissing in model (first 20):")
            for k in sorted(in_ckpt_not_model)[:20]:
                print(f"  {k}")
    if verbose or in_model_not_ckpt:
        if in_model_not_ckpt:
            print("\nMissing in ckpt (first 20):")
            for k in sorted(in_model_not_ckpt)[:20]:
                print(f"  {k}")

    return matched, in_ckpt_not_model, in_model_not_ckpt


if __name__ == "__main__":
    bin_path = os.path.join(os.path.dirname(__file__), '..',
                            'weights', 'tencent', 'Hunyuan3D-2.1',
                            'hunyuan3d-paintpbr-v2-1', 'unet',
                            'diffusion_pytorch_model.bin')

    if not os.path.exists(bin_path):
        print(f"Weights not found: {bin_path}")
        sys.exit(1)

    print(f"Converting: {bin_path}")
    converted = convert_unet_weights(bin_path)
    matched, extra, missing = verify_weights(converted, verbose=True)

    if len(extra) == 0 and len(missing) == 0:
        # Save
        out_dir = os.path.join(os.path.dirname(bin_path), '..', '..', 'mlx_weights')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "unet.safetensors")
        mx.save_safetensors(out_path, converted)
        print(f"\nSaved: {out_path}")
    else:
        print("\nKey mismatch — fix mapping before saving.")
