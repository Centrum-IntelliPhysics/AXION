#!/usr/bin/env bash
# Two parallel horizontal notches under uniform in-situ stress (paper Fig. 6 (a)-(c);
# Zhou et al. 2023, Fig. 18): two 0.8 m notches separated by 1 m in a 10 m square
# under 0.5 MPa biaxial compression, 50 time steps of 0.05 s.
#
#   bash run_two_notch.sh                                   # with the symmetry projection
#   PFF_ENFORCE_REFLECTION_SYMMETRY=0 PFF_OUTDIR=output_two_notch_no_projection \
#       bash run_two_notch.sh                               # without it (paper Fig. 9)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Discretization and time stepping
export PFF_NT="${PFF_NT:-50}"                    # 50 * 0.05 s = 2.5 s
export PFF_DT="${PFF_DT:-0.05}"
export PFF_QF="${PFF_QF:-10.0}"                  # injection rate [kg/(m^3 s)]
export PFF_L0="${PFF_L0:-0.1}"
export PFF_ELE_TYPE="${PFF_ELE_TYPE:-TRI3}"
export PFF_QUADRATURE_ORDER="${PFF_QUADRATURE_ORDER:-2}"
# Unstructured linear triangles: Gmsh meshes one quadrant, which is reflected
# about both axes. A target size of 0.035 m gives a longest edge of 0.0477 m.
export PFF_MESH_KIND="${PFF_MESH_KIND:-unstructured_gmsh}"
export PFF_MESH_H="${PFF_MESH_H:-0.035}"
export PFF_SAVE_EVERY="${PFF_SAVE_EVERY:-5}"

# Solver controls
export PFF_ENFORCE_REFLECTION_SYMMETRY="${PFF_ENFORCE_REFLECTION_SYMMETRY:-1}"
export PFF_MAX_STAGGER="${PFF_MAX_STAGGER:-1000}"
export PFF_ANDERSON_DEPTH="${PFF_ANDERSON_DEPTH:-5}"
export PFF_ANDERSON_START="${PFF_ANDERSON_START:-2}"
export PFF_ANDERSON_BETA="${PFF_ANDERSON_BETA:-1.0}"
export PFF_ANDERSON_REG="${PFF_ANDERSON_REG:-1e-8}"
export PFF_STAGGER_RELAX="${PFF_STAGGER_RELAX:-0.5}"
export PFF_RELAX_AFTER="${PFF_RELAX_AFTER:-20}"
export PFF_RESUME="${PFF_RESUME:-1}"

# Geometry, source and stress boundary condition. The injection is distributed
# over a band of half-width 0.075 m about each notch.
export PFF_NOTCH_LAYOUT=double_horizontal
export PFF_NOTCH_SPACING=1.0
export PFF_THETA=0
export PFF_SOURCE_HALF_WIDTH="${PFF_SOURCE_HALF_WIDTH:-0.075}"
export PFF_SIGX0=0.5
export PFF_SIGY0=0.5
export PFF_SIG0_SIGN=-1
export PFF_SIG0_MODE=both
export PFF_OUTDIR="${PFF_OUTDIR:-output_two_notch}"

echo ">>> JAX device check:"
"${PYTHON}" -c "import jax; print('  backend =', jax.default_backend(), '| devices =', jax.devices())"

START_TS=$(date +%s)
"${PYTHON}" -u hydraulic_fracture.py
echo ">>> wall-clock seconds: $(( $(date +%s) - START_TS ))"

if [[ "${PFF_SKIP_PLOT:-0}" != "1" && "${PFF_ENFORCE_REFLECTION_SYMMETRY}" == "1" ]]; then
  "${PYTHON}" plot_two_notch_case.py --output-dir "${PFF_OUTDIR}" \
      --figure "${PFF_OUTDIR}/two_notch_case.png"
fi
echo ">>> Done. Fields and history in ${PFF_OUTDIR}/"
