"""Download Hunyuan3D-2.1 weights into a local directory.

Downloads the HuggingFace snapshot into weights/tencent/Hunyuan3D-2.1/ so
the rest of the codebase can resolve it automatically without setting
HUNYUAN_LOCAL_ROOT.

Example:
    python scripts/download_weights.py
    python scripts/download_weights.py --output-dir weights/tencent/Hunyuan3D-2.1
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


DEFAULT_REPO = "tencent/Hunyuan3D-2.1"
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "tencent", "Hunyuan3D-2.1",
)

DEFAULT_PATTERNS = [
    "hunyuan3d-dit-v2-1/*",
    "hunyuan3d-paintpbr-v2-1/*",
    "hunyuan3d-vae-v2-1/*",
    "hy3dpaint/**/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hunyuan3D-2.1 weights locally")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--full-snapshot",
        action="store_true",
        help="Download the entire repo instead of just model weights",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    download_kwargs = {
        "repo_id": args.repo_id,
        "local_dir": output_dir,
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if not args.full_snapshot:
        download_kwargs["allow_patterns"] = DEFAULT_PATTERNS

    snapshot_path = snapshot_download(**download_kwargs)
    print(f"Downloaded {args.repo_id} to {snapshot_path}")


if __name__ == "__main__":
    main()
