import argparse
import base64
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8081/generate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--shape-retry-attempts", type=int, default=3)
    parser.add_argument("--texture-steps", type=int, default=None, help="Texture denoise steps (default: 15)")
    parser.add_argument("--texture-guidance", type=float, default=None, help="Texture CFG scale (default: 3.0)")
    args = parser.parse_args()

    image_bytes = Path(args.image).read_bytes()
    payload = {
        "image": base64.b64encode(image_bytes).decode(),
        "texture": args.texture,
        "seed": args.seed,
        "octree_resolution": args.octree_resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "shape_retry_attempts": args.shape_retry_attempts,
    }
    if args.texture_steps is not None:
        payload["texture_steps"] = args.texture_steps
    if args.texture_guidance is not None:
        payload["texture_guidance"] = args.texture_guidance

    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(req, timeout=3600) as response:
        data = response.read()
        output_path.write_bytes(data)
        print(f"saved {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
