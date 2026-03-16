#!/usr/bin/env python3
"""CLI for texturing existing 3D meshes using Hunyuan3D paint pipeline."""

import argparse
import os
import sys
import time

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from model_paths import resolve_hunyuan_paths


def main():
    parser = argparse.ArgumentParser(description="Texture a 3D mesh using Hunyuan3D")
    parser.add_argument("mesh", help="Path to input mesh (.glb or .obj)")
    parser.add_argument("image", help="Path to reference image")
    parser.add_argument("-o", "--output", help="Output path (default: <mesh>_textured.glb)")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--face-count", type=int, default=40000, help="Target face count (default: 40000)")
    parser.add_argument("--render-size", type=int, default=2048, help="Render resolution (default: 2048)")
    parser.add_argument("--texture-size", type=int, default=2048, help="Texture resolution: 1024/2048/4096 (default: 2048)")
    parser.add_argument("--views", type=int, default=6, choices=range(6, 10), help="Number of views: 6-9 (default: 6)")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 768], help="Diffusion resolution: 512 or 768 (default: 512)")
    parser.add_argument("--view-size", type=int, default=None, help="Multiview view size (default: auto, 512)")
    parser.add_argument("--infer-steps", type=int, default=None, help="Diffusion steps (default: 15 UniPC). Lower = faster")
    parser.add_argument("--attention-slicing", action="store_true", help="Enable attention slicing (may help MPS)")
    parser.add_argument("--no-remesh", action="store_true", help="Skip mesh simplification")
    args = parser.parse_args()

    if not os.path.exists(args.mesh):
        print(f"Error: mesh not found: {args.mesh}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    if args.output is None:
        base, _ = os.path.splitext(args.mesh)
        args.output = f"{base}_textured.glb"

    resolved = resolve_hunyuan_paths(args.model_path)

    print(f"Mesh:         {args.mesh}")
    print(f"Reference:    {args.image}")
    print(f"Output:       {args.output}")
    print(f"Face count:   {args.face_count}")
    print(f"Render size:  {args.render_size}")
    print(f"Texture size: {args.texture_size}")
    print(f"Views:        {args.views}")
    print(f"Resolution:   {args.resolution}")
    print()

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(args.views, args.resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    conf.multiview_pretrained_path = resolved.texture_dir
    conf.render_size = args.render_size
    conf.texture_size = args.texture_size
    conf.face_count = args.face_count
    conf.infer_steps = args.infer_steps
    conf.attention_slicing = args.attention_slicing

    print(f"Device: {conf.device}")
    print("Loading models...")
    pipeline = Hunyuan3DPaintPipeline(conf)

    from PIL import Image
    image = Image.open(args.image).convert("RGBA")

    output_obj = os.path.splitext(args.output)[0] + ".obj"

    print("Generating texture...")
    start = time.time()
    result = pipeline(
        mesh_path=args.mesh,
        image_path=image,
        output_mesh_path=output_obj,
        use_remesh=not args.no_remesh,
        save_glb=False,
    )
    elapsed = time.time() - start
    print(f"Texture generation took {elapsed:.1f}s")

    # Convert to GLB
    if args.output.endswith(".glb"):
        from hy3dpaint.convert_utils import create_glb_with_pbr_materials
        textures = {
            'albedo': result.replace('.obj', '.jpg'),
            'metallic': result.replace('.obj', '_metallic.jpg'),
            'roughness': result.replace('.obj', '_roughness.jpg'),
        }
        create_glb_with_pbr_materials(result, textures, args.output)
        print(f"Saved: {args.output}")
    else:
        print(f"Saved: {result}")


if __name__ == "__main__":
    main()
