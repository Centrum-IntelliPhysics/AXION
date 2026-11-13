#!/usr/bin/env bash
# The Koyna gravity dam on several GPUs of one node (paper Section 4.2.3).
#
#   CUDA_VISIBLE_DEVICES=0,1 bash run_koyna_multigpu.sh
#   PFF_SELFTEST=1 bash run_koyna_multigpu.sh     # wiring check on 4 emulated CPU devices
#
# On a cluster with one process per GPU, launch koyna_dam_multigpu.py with srun.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

if [[ "${PFF_SELFTEST:-0}" != "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE

export PFF_MODEL="${PFF_MODEL:-AT2}"
export PFF_MESH="${PFF_MESH:-TRI3}"
export PFF_L="${PFF_L:-0.03}"
export PFF_UNIFORM_H="${PFF_UNIFORM_H:-0.015}"
export PFF_N_FILL="${PFF_N_FILL:-12}"
export PFF_N_OVF="${PFF_N_OVF:-50}"
export PFF_OVF_MAX="${PFF_OVF_MAX:-10.0}"
export PFF_OUTPUT_DIR="${PFF_OUTPUT_DIR:-output_koyna_multigpu}"

START_TS=$(date +%s)
"${PYTHON}" -u koyna_dam_multigpu.py
echo ">>> wall-clock seconds: $(( $(date +%s) - START_TS ))"
