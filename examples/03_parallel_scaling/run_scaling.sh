#!/usr/bin/env bash
# Strong- and weak-scaling sweeps of the 3D benchmark on the GPUs of one node
# (paper Section 4.1.3, Fig. 4). One Python process per run; the number of
# visible GPUs is the number of subdomains D.
#
#   GPUS=0,1,2,3 bash run_scaling.sh strong      # fixed mesh 150 x 150 x 15
#   GPUS=0,1,2,3 bash run_scaling.sh weak        # NZ = 15 * D (constant work per GPU)
#   GPUS=0,1 DEVSET="1 2" bash run_scaling.sh weak
#
# Every run appends one row to scaling_<mode>.csv; plot with
#   python plot_scaling.py strong      (or weak)
set -euo pipefail

MODE="${1:-strong}"
if [[ "${MODE}" != "strong" && "${MODE}" != "weak" ]]; then
  echo "usage: run_scaling.sh strong|weak" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

GPUS="${GPUS:-0,1,2,3}"                 # physical GPU ids, in the order to use them
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

MODEL="${MODEL:-AT2}"
NX="${NX:-150}"; NY="${NY:-150}"
NZ0="${NZ0:-15}"                        # weak: layers per GPU; strong: total layers
NSTEPS="${NSTEPS:-160}"; DU="${DU:-5e-5}"
DEVSET="${DEVSET:-1 2 4}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PFF_CSV="scaling_${MODE}.csv"

echo "${MODE}-scaling sweep: MODEL=${MODEL}  NX=${NX} NY=${NY} NZ0=${NZ0}  GPUs=${GPUS}  D in {${DEVSET}}"
for D in ${DEVSET}; do
  if (( D > ${#GPU_ARR[@]} )); then
    echo "!! skip D=${D}: only ${#GPU_ARR[@]} GPU(s) listed in GPUS=${GPUS}"
    continue
  fi
  VIS="$(IFS=,; echo "${GPU_ARR[*]:0:${D}}")"
  if [[ "${MODE}" == "weak" ]]; then NZ=$(( NZ0 * D )); else NZ="${NZ0}"; fi
  echo "==================================================================="
  echo ">>> D=${D}  CUDA_VISIBLE_DEVICES=${VIS}  mesh ${NX}x${NY}x${NZ}"
  echo "==================================================================="
  CUDA_VISIBLE_DEVICES="${VIS}" \
  PFF_MODEL="${MODEL}" PFF_NX="${NX}" PFF_NY="${NY}" PFF_NZ="${NZ}" \
  PFF_NSTEPS="${NSTEPS}" PFF_DU="${DU}" \
  PFF_TAG="${MODE}_gpu${D}" PFF_WARMUP=1 \
      "${PYTHON}" -u sent_3d_multigpu.py
done
echo "Sweep done. Plot with:  ${PYTHON} plot_scaling.py ${MODE}"
