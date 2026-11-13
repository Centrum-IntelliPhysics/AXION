#!/usr/bin/env bash
# Three-dimensional premolar under occlusal loading (paper Section 4.2.2, Fig. 7).
#
#   bash run_premolar.sh
#   PFF_MESH_ONLY=1 bash run_premolar.sh        # build and write the mesh only
#   PFF_COARSE=1 PFF_MESH_ONLY=1 bash run_premolar.sh   # coarse mesh preview
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE

# Model and discretization
export PFF_MODEL="${PFF_MODEL:-CZM}"
export PFF_L="${PFF_L:-0.25}"                    # regularization length [mm]
export PFF_H="${PFF_H:-0.125}"                   # element size in the crown [mm]
export PFF_H_BONE="${PFF_H_BONE:-0.8}"           # element size in the root and bone [mm]
export PFF_DRIVING="${PFF_DRIVING:-rankine}"

# Loading: increments of 0.004 mm, refined to 0.002 mm where the crack grows
export PFF_DELTA_STEP="${PFF_DELTA_STEP:-0.004}"
export PFF_DELTA_MAX="${PFF_DELTA_MAX:-0.184}"
export PFF_REFINE_START="${PFF_REFINE_START:-0.164}"
export PFF_REFINE_END="${PFF_REFINE_END:-0.176}"
export PFF_REFINE_STEP="${PFF_REFINE_STEP:-0.002}"
export PFF_SAVE_EVERY="${PFF_SAVE_EVERY:-5}"

# Solver controls
export PFF_MAX_STAGGER="${PFF_MAX_STAGGER:-60}"
export PFF_NEWTON_MAX="${PFF_NEWTON_MAX:-20}"
export PFF_CG_MAXITER="${PFF_CG_MAXITER:-8000}"
export PFF_CG_TOL="${PFF_CG_TOL:-1e-4}"
export PFF_PG_MAXIT="${PFF_PG_MAXIT:-2000}"      # cap on projected Jacobi sweeps
export PFF_H_CAP="${PFF_H_CAP:-100}"
export PFF_D_FREEZE="${PFF_D_FREEZE:-0.99}"
export PFF_STOP_AFTER_FAILURE="${PFF_STOP_AFTER_FAILURE:-1}"
export PFF_STOP_LOAD_FRAC="${PFF_STOP_LOAD_FRAC:-0.05}"
export PFF_VERBOSE="${PFF_VERBOSE:-1}"

export PFF_OUTPUT_DIR="${PFF_OUTPUT_DIR:-output_premolar}"

echo ">>> JAX device check:"
"${PYTHON}" -c "import jax; print('  backend =', jax.default_backend(), '| devices =', jax.devices())"

START_TS=$(date +%s)
"${PYTHON}" -u premolar.py
echo ">>> wall-clock seconds: $(( $(date +%s) - START_TS ))"
