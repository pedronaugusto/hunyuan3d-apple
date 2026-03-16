"""Test texture pipeline only, skipping shape generation.

Usage:
    python scripts/test_texture_only.py \
        --mesh gradio_cache/*_initial.glb \
        --image references/coraline.webp \
        --output output_textured.glb
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
from mlx_backend.texture_pipeline import MlxTexturePipeline
from hy3dpaint.convert_utils import create_glb_with_pbr_materials
from model_paths import resolve_hunyuan_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to untextured GLB mesh")
    parser.add_argument("--image", required=True, help="Reference image for texturing")
    parser.add_argument("--output", required=True, help="Output textured GLB path")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--texture-steps", type=int, default=15)
    parser.add_argument("--texture-guidance", type=float, default=3.0)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(__file__))
    model_root = resolve_hunyuan_paths(args.model_path).root

    print("Loading texture pipeline...")
    pipeline = MlxTexturePipeline(model_root)

    image = Image.open(args.image).convert("RGBA")
    output_obj = args.output.replace(".glb", ".obj")

    print(f"Texturing {args.mesh} ...")
    textured_obj = pipeline(
        mesh_path=args.mesh,
        image_path=image,
        output_mesh_path=output_obj,
        save_glb=False,
        texture_steps=args.texture_steps,
        texture_guidance=args.texture_guidance,
    )

    # Convert to GLB with PBR
    create_glb_with_pbr_materials(textured_obj, {
        'albedo': textured_obj.replace('.obj', '.jpg'),
        'metallic': textured_obj.replace('.obj', '_metallic.jpg'),
        'roughness': textured_obj.replace('.obj', '_roughness.jpg'),
    }, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
