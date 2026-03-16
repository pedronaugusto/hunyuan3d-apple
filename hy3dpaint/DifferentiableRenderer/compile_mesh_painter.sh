#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

EXTRA_FLAGS=""
if [ "$(uname)" = "Darwin" ]; then
  EXTRA_FLAGS="-undefined dynamic_lookup -std=c++17"
else
  EXTRA_FLAGS="-std=c++11"
fi

c++ -O3 -Wall -shared -fPIC $EXTRA_FLAGS \
  $(python3 -m pybind11 --includes) \
  mesh_inpaint_processor.cpp \
  -o mesh_inpaint_processor$(python3-config --extension-suffix)
