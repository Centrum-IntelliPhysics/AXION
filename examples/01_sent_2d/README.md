# 01 - Two-dimensional single-edge-notched tension (Section 4.1.1)

Unit square (1 mm) with a horizontal pre-crack from the left edge to the centre,
clamped at the bottom and pulled at the top in 700 increments of 1e-5 mm under
plane strain. AT2 model, E = 210 kN/mm^2, nu = 0.3, G_f = 2.7e-3 kN/mm,
l = 0.03 mm. The pre-crack is a geometric slit (duplicated nodes).

| Script | Configuration | Paper |
|---|---|---|
| `sent_2d_uniform.py` | fixed 100 x 100 QUAD4 mesh (h = l/3), 3.08e4 DoFs | 4.55 min on one GPU |
| `sent_2d_adaptive.py` | 48 x 48 mesh refined around the crack (two levels, h_min = 1/192 mm), 1.13e4 DoFs at the end | 2.41 min |

```bash
bash run.sh                        # both, AT2
python sent_2d_uniform.py          # uniform only
PFF_MODEL=CZM python sent_2d_uniform.py     # AT1 | AT2 | CZM share the element definition
python sent_2d_adaptive.py
```

The adaptive computation uses `amr_mesh.py` (fixed-capacity quadtree: refine,
2:1 balance, hanging-node constraints), `amr_fem.py` (QUAD4 operators and
constrained CG) and `amr_driver.py` (refinement inside the compiled load loop).

Output (`output_uniform/`, `output_adaptive/`): load-displacement record
(`sol_<MODEL>.npz`, `force_disp_AMR.h5`), final state (`final_*.npz/.h5/.vtu`),
`summary_*.txt` with DoFs, wall-clock time and peak load, and figures.

All settings are environment variables; see the header of each script.
