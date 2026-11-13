#!/usr/bin/env bash
# The Koyna gravity dam on one GPU (paper Section 4.2.3, Fig. 8): uniform element
# size h = 0.015 m over the whole cross-section, l = 0.03 m, 9.6e7 degrees of
# freedom. Needs a GPU with about 96 GB of memory.
#
#   bash run_koyna.sh
#   PFF_COARSE=1 PFF_UNIFORM_H=0 PFF_L=0.3 bash run_koyna.sh   # small graded-mesh test
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE

export PFF_MODEL="${PFF_MODEL:-AT2}"
export PFF_MESH="${PFF_MESH:-TRI3}"
export PFF_L="${PFF_L:-0.03}"                    # regularization length [m]
export PFF_UNIFORM_H="${PFF_UNIFORM_H:-0.015}"   # uniform element size [m]
export PFF_N_FILL="${PFF_N_FILL:-12}"            # reservoir filling increments (elastic)
export PFF_N_OVF="${PFF_N_OVF:-50}"              # overflow increments
export PFF_OVF_MAX="${PFF_OVF_MAX:-10.0}"        # final overflow height [m]
export PFF_LIGHT_OUTPUT="${PFF_LIGHT_OUTPUT:-1}" # memory-light field output
export PFF_OUTPUT_DIR="${PFF_OUTPUT_DIR:-output_koyna}"

echo ">>> JAX device check:"
"${PYTHON}" -c "import jax; print('  backend =', jax.default_backend(), '| devices =', jax.devices())"

START_TS=$(date +%s)
"${PYTHON}" -u koyna_dam.py
echo ">>> wall-clock seconds: $(( $(date +%s) - START_TS ))"
