# 06 - Three-dimensional premolar (Section 4.2.2)

Fracture of a human premolar under occlusal loading: six constituents whose
stiffnesses span more than four orders of magnitude, a thin periodontal
ligament, contact loading by a rigid sphere of 4 mm diameter, and a crack that
nucleates without any notch or initial defect. The configuration follows the
intact-tooth study of Zhu et al. (Eng. Fract. Mech. 343, 2026) on an idealized
geometry with the dimensions of an average maxillary second premolar.

* PF-CZM with l = 0.25 mm; crack driven by the Rankine-type measure
  <sigma_1>^2 / 2E of the largest principal effective stress.
* Contact: frictionless unilateral spring within d_s = 0.3 mm of the sphere,
  stiffness 5e3 N/mm^3; its tangent comes from automatic differentiation.
* Conforming tetrahedral mesh of tooth, ligament and bone (Gmsh): h = 0.125 mm
  in the crown, 0.8 mm in the root and bone; 6.87e5 DoFs.
* Sphere lowered in increments of 0.004 mm, refined to 0.002 mm for
  delta in [0.164, 0.176] mm.

```bash
bash run_premolar.sh
PFF_COARSE=1 PFF_MESH_ONLY=1 bash run_premolar.sh    # coarse mesh preview, no solve
```

Paper: 5.59 h on one GPU; the crack nucleates at the floor of the central
fissure (delta = 0.140 mm) and runs through the enamel into the dentin.

Material parameters (Table 6 of the paper) and all controls are environment
variables documented in `premolar.py` and set in `run_premolar.sh`.

Output (`output_premolar/`): `mesh_start_CZM.vtu` (mesh with tissue labels),
`vtk/CZM/premolar_delta_*.vtu` (damage `sol`, displacement `u`, history ratio),
`history_CZM.npz` (load-displacement record), `timing_CZM.txt`.

Requires `gmsh`.
