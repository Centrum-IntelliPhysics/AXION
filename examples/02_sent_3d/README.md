# 02 - Three-dimensional single-edge-notched tension (Section 4.1.2)

Slab of side 1 mm with a through-thickness pre-crack at mid-height, clamped at
the bottom face and pulled at the top face in 160 increments of 5e-5 mm.
Uniform grid of 150 x 150 x 15 trilinear hexahedra, 366 016 nodes and 1.46e6
DoFs. AT2 model with the hybrid formulation (isotropically degraded stress,
history from the tensile energy of the spectral split).

```bash
bash run.sh
PFF_NX=40 PFF_NY=40 PFF_NZ=4 PFF_NSTEPS=10 python sent_3d.py    # small test
```

Paper: 57.8 min on one GPU, peak load 126.6 N.

Output (`output/`): `force_disp_AT2.h5` (displacement, reaction force),
`final_AT2.h5` and `final_AT2.vtu` (mesh, damage), `summary_AT2.txt`.

Settings (`PFF_*` environment variables) are listed in the header of
`sent_3d.py`. `PFF_MODEL=AT1|CZM` solves the damage subproblem with the
active-set Newton iteration.
