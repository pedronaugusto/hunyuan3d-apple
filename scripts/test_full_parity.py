#!/usr/bin/env python3
"""
Real live-path parity acceptance runner.

This is the only "full parity" script that should matter:
1. Shape: real-image PT vs MLX parity through final mesh.
2. Texture: real-image real-mesh PT vs MLX parity through traced texture stages.

It runs the heavy jobs serially and writes a single report with subprocess
statuses plus paths to the underlying shape/texture reports.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def _run(cmd: list[str], cwd: Path) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
        help="Path to reference image (required)",
    )
    parser.add_argument("--shape-mesh")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape-steps", type=int, default=20)
    parser.add_argument("--shape-guidance", type=float, default=5.5)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--texture-steps", type=int, default=15)
    parser.add_argument("--texture-guidance", type=float, default=3.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shape_dir = output_dir / "shape"
    texture_dir = output_dir / "texture"
    shape_dir.mkdir(exist_ok=True)
    texture_dir.mkdir(exist_ok=True)

    shape_report_path = shape_dir / "report.json"
    texture_report_path = texture_dir / "report.json"

    shape_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test_shape_real_input_parity.py"),
        "--image", os.path.abspath(args.image),
        "--seed", str(args.seed),
        "--steps", str(args.shape_steps),
        "--guidance", str(args.shape_guidance),
        "--octree-resolution", str(args.octree_resolution),
        "--max-stage", "final_mesh",
        "--output-dir", str(shape_dir),
    ]
    shape_run = _run(shape_cmd, REPO_DIR)

    shape_mesh = args.shape_mesh
    if not shape_mesh:
        # Prefer the actual mesh exported by the live shape run if present.
        for candidate in (
            shape_dir / "mlx_final_mesh.glb",
            shape_dir / "mlx_final_mesh.obj",
        ):
            if candidate.exists():
                shape_mesh = str(candidate)
                break

    texture_run = None
    if shape_mesh:
        texture_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "test_texture_real_input_parity.py"),
            "--mesh", os.path.abspath(shape_mesh),
            "--image", os.path.abspath(args.image),
            "--backend", "both",
            "--texture-steps", str(args.texture_steps),
            "--texture-guidance", str(args.texture_guidance),
            "--output-dir", str(texture_dir),
        ]
        texture_run = _run(texture_cmd, REPO_DIR)
    else:
        texture_run = {
            "cmd": None,
            "returncode": None,
            "stdout": "",
            "stderr": "No shape mesh path available for texture parity run.",
        }

    report = {
        "image": os.path.abspath(args.image),
        "shape_mesh": os.path.abspath(shape_mesh) if shape_mesh else None,
        "shape_report": str(shape_report_path),
        "texture_report": str(texture_report_path),
        "shape": shape_run,
        "texture": texture_run,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    failed = False
    if shape_run["returncode"] != 0:
        failed = True
    if texture_run and texture_run["returncode"] not in (0, None):
        failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
