#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
mkdir -p "${TMPDIR:-$ROOT/../tmp_runtime}"
export TMPDIR="${TMPDIR:-$ROOT/../tmp_runtime}"

# Auto-build C++ extensions if missing
INPAINT_SO="$ROOT/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor$(python3-config --extension-suffix)"
if [ ! -f "$INPAINT_SO" ]; then
  echo "Building mesh_inpaint_processor C++ extension..."
  bash "$ROOT/hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh"
  echo "  Built $INPAINT_SO"
fi

# Convert weights to safetensors if needed (first run only)
MODEL_PATH="${MODEL_PATH:-$ROOT/weights/tencent/Hunyuan3D-2.1}"
BACKEND="${BACKEND:-mlx}"

if [ "$BACKEND" = "mlx" ]; then
  DIT_ST="$MODEL_PATH/hunyuan3d-dit-v2-1/model.safetensors"
  if [ ! -f "$DIT_ST" ]; then
    echo "Converting weights to safetensors for MLX..."
    python scripts/convert_weights.py --weights-dir "$MODEL_PATH"
  fi
fi

mkdir -p "${CACHE_PATH:-$ROOT/gradio_cache_test}"

python api_server.py \
  --backend "$BACKEND" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8081}" \
  --model_path "$MODEL_PATH" \
  --cache-path "${CACHE_PATH:-gradio_cache_test}"
