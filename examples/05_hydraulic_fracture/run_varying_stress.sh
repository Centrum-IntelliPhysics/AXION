#!/usr/bin/env bash
# Single vertical notch under depth-varying in-situ stress and gravity (paper
# Fig. 6 (d)-(f); Zhou et al. 2023, Figs. 19-21): the horizontal stress grows
# from 1 MPa at the top to 5 MPa at the base, the vertical tractions are 2 MPa
# and 2.25 MPa; 280 time steps of 0.05 s.
#
#   bash run_varying_stress.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Same material, flow, mesh and time-step settings as the two-notch case.
export PFF_NT="${PFF_NT:-280}"                   # 280 * 0.05 s = 14 s
export PFF_DT="${PFF_DT:-0.05}"
export PFF_QF="${PFF_QF:-10.0}"
export PFF_L0="${PFF_L0:-0.1}"
export PFF_ELE_TYPE="${PFF_ELE_TYPE:-TRI3}"
export PFF_QUADRATURE_ORDER="${PFF_QUADRATURE_ORDER:-2}"
export PFF_MESH_KIND="${PFF_MESH_KIND:-unstructured_gmsh}"
export PFF_MESH_H="${PFF_MESH_H:-0.035}"
export PFF_SAVE_EVERY="${PFF_SAVE_EVERY:-20}"    # one snapshot per second
export PFF_ENFORCE_REFLECTION_SYMMETRY=0

export PFF_MAX_STAGGER="${PFF_MAX_STAGGER:-1000}"
export PFF_ANDERSON_DEPTH="${PFF_ANDERSON_DEPTH:-5}"
export PFF_ANDERSON_START="${PFF_ANDERSON_START:-2}"
export PFF_ANDERSON_BETA="${PFF_ANDERSON_BETA:-1.0}"
export PFF_ANDERSON_REG="${PFF_ANDERSON_REG:-1e-8}"
export PFF_UP_MAX_ITER="${PFF_UP_MAX_ITER:-50}"
export PFF_UP_TOL="${PFF_UP_TOL:-1e-6}"
export PFF_STAGGER_RELAX="${PFF_STAGGER_RELAX:-0.5}"
export PFF_RELAX_AFTER="${PFF_RELAX_AFTER:-20}"
export PFF_RESUME="${PFF_RESUME:-1}"

# Vertical central notch. The solver builds the linear stress field and the
# mechanical body force of -25 kN/m^3 that equilibrates it.
export PFF_NOTCH_LAYOUT=single
export PFF_THETA=90
export PFF_STRESS_PROFILE=fig19_linear
export PFF_SIG0_SIGN=-1
export PFF_SIG0_MODE=both
export PFF_OUTDIR="${PFF_OUTDIR:-output_varying_stress}"

echo ">>> JAX device check:"
"${PYTHON}" -c "import jax; print('  backend =', jax.default_backend(), '| devices =', jax.devices())"

START_TS=$(date +%s)
"${PYTHON}" -u hydraulic_fracture.py
echo ">>> wall-clock seconds: $(( $(date +%s) - START_TS ))"

if [[ "${PFF_SKIP_PLOT:-0}" != "1" ]]; then
  "${PYTHON}" plot_varying_stress_case.py --output-dir "${PFF_OUTDIR}" \
      --geometry "${PFF_OUTDIR}/varying_stress_case_geometry.png" \
      --results "${PFF_OUTDIR}/varying_stress_case_results.png"
fi
echo ">>> Done. Fields and history in ${PFF_OUTDIR}/"
