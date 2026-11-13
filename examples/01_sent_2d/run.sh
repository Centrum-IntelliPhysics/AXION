#!/usr/bin/env bash
# Two-dimensional single-edge-notched tension (paper Section 4.1.1):
# uniform mesh, then the adaptive computation.
#
#   bash run.sh                 # AT2, both configurations
#   PFF_MODEL=CZM bash run.sh   # uniform mesh with another model (AT1 | AT2 | CZM)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo ">>> JAX device check:"
"${PYTHON}" -c "import jax; print('  backend =', jax.default_backend(), '| devices =', jax.devices())"

echo ">>> uniform mesh (100 x 100, 700 increments)"
"${PYTHON}" -u sent_2d_uniform.py

echo ">>> adaptive mesh (48 x 48 refined around the crack, 700 increments)"
"${PYTHON}" -u sent_2d_adaptive.py
