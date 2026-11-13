#!/usr/bin/env bash
# Runs every case of the derivative-verification study (verify_ad.py) on one GPU.
#   bash run_verify.sh                       # GPU 0
#   CUDA_VISIBLE_DEVICES=2 bash run_verify.sh
# Overridable: PFF_NX, PFF_NY, VER_U_BAR, VER_DU, VER_MS_TOL, LIN_TOL_TIGHT, LIN_TOL_LOOSE
# The three cases are independent and can also be launched on separate GPUs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHON="${PYTHON:-python}"
LIN_TOL_TIGHT="${LIN_TOL_TIGHT:-1e-10}"
LIN_TOL_LOOSE="${LIN_TOL_LOOSE:-1e-6}"
OUT="${VER_OUTPUT_DIR:-${HERE}/output}"
mkdir -p "${OUT}"

# AT2: Taylor test, Newton history, dF/dGc and dF/dE0 (forward mode vs central differences)
PFF_MODEL=AT2 VER_LIN_TOL="${LIN_TOL_TIGHT}" VER_TAG="lin${LIN_TOL_TIGHT}" \
    "${PYTHON}" -u verify_ad.py 2>&1 | tee "${OUT}/log_AT2_lin${LIN_TOL_TIGHT}.txt"

# AT2 again with the looser CG tolerance (sensitivities only)
PFF_MODEL=AT2 VER_LIN_TOL="${LIN_TOL_LOOSE}" VER_TAG="lin${LIN_TOL_LOOSE}" VER_TESTS=sens \
    "${PYTHON}" -u verify_ad.py 2>&1 | tee "${OUT}/log_AT2_lin${LIN_TOL_LOOSE}.txt"

# PF-CZM: Taylor test of the damage residual, Newton history, elastic check dF/dE0 = F/E0
PFF_MODEL=CZM VER_LIN_TOL="${LIN_TOL_TIGHT}" VER_TAG="lin${LIN_TOL_TIGHT}" \
    "${PYTHON}" -u verify_ad.py 2>&1 | tee "${OUT}/log_CZM_lin${LIN_TOL_TIGHT}.txt"

echo "All done. Logs, JSON and figures in ${OUT}/"
