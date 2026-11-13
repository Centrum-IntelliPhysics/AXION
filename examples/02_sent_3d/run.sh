#!/usr/bin/env bash
# Three-dimensional single-edge-notched tension on one GPU (paper Section 4.1.2):
# 150 x 150 x 15 HEX8, 160 increments of 5e-5 mm.
#
#   bash run.sh
#   PFF_NX=40 PFF_NY=40 PFF_NZ=4 PFF_NSTEPS=10 bash run.sh    # small test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo ">>> JAX device check:"
"${PYTHON}" -c "import jax; print('  backend =', jax.default_backend(), '| devices =', jax.devices())"
"${PYTHON}" -u sent_3d.py
