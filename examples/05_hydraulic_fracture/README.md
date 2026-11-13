# 05 - Hydraulic fracture in a poroelastic medium (Sections 3.3 and 4.2.1)

Three-field hydro-poroelastic phase-field formulation of Zhou, Zhuang & Rabczuk
(arXiv:2309.08584): displacement, phase field and pore pressure, with fluid
properties that follow the phase field. Square domain of side 10 m, notches of
length 0.8 m prescribed through an initial history field, injection at
10 kg/(m^3 s), p = 0 on the outer boundary, time increment 0.05 s. Unstructured
Gmsh triangulation generated on one quadrant and reflected about both axes:
188 968 linear triangles, 95 057 nodes, 3.80e5 DoFs.

| Script | Configuration | Paper |
|---|---|---|
| `run_two_notch.sh` | two parallel horizontal notches, uniform remote stress of 0.5 MPa, 50 time steps | Fig. 6 (a)-(c), 1.62 h |
| `run_varying_stress.sh` | single vertical notch, gravity and depth-varying in-situ stress, 280 time steps | Fig. 6 (d)-(f), 5.32 h |

```bash
bash run_two_notch.sh
bash run_varying_stress.sh
# the two-notch computation without the reflection-symmetry projection (Fig. 9)
PFF_ENFORCE_REFLECTION_SYMMETRY=0 PFF_OUTDIR=output_two_notch_no_projection bash run_two_notch.sh
```

All three subproblems are written as weak forms in `hydraulic_fracture.py`; the
tangent blocks, including the derivatives of the interpolated fluid properties,
follow from automatic differentiation. Each time step converges the coupled
(u, p) block, renews the history field and solves the phase field, with Anderson
acceleration of the staggered iteration. For the two-notch case the fields are
projected onto the reflection symmetry of the configuration.

Runs checkpoint after every time step and resume with `PFF_RESUME=1` (the
default of the run scripts). Output folder: `mesh.npz`, `state_checkpoint.npz`,
`notch_history.npz` (time, crack half-length, notch pressure), `vtk/step_*.vtu`
and `phi_*.png`. `plot_two_notch_case.py` and `plot_varying_stress_case.py`
assemble the result figures from an output folder.

Requires `gmsh`.
