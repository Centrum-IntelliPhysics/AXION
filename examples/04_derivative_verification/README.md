# 04 - Verification of the automatically differentiated derivatives (Section 4.1.4)

Runs on top of the two-dimensional benchmark (`../01_sent_2d/sent_2d_uniform.py`)
with 150 x 150 elements and l = 0.02 mm, loaded to 5e-3 mm in 50 increments of
1e-4 mm (largest nodal damage 0.27).

| Test | What is checked | Paper |
|---|---|---|
| `taylor` | Taylor remainder of the element tangents: r1 decreases at second order where the residual is smooth | Table 4 |
| `newton` | residual history of Newton's method on the displacement subproblem | Fig. 5 (a) |
| `sens` | dF/dE_0 and dF/dG_f by forward-mode differentiation of the loading history vs. central differences, at two CG tolerances | Fig. 5 (b) |
| `elastic` | PF-CZM below the damage threshold: dF/dE_0 = F/E_0 | text |

```bash
bash run_verify.sh                       # AT2 (tight and loose CG tolerance) and PF-CZM
PFF_MODEL=CZM python verify_ad.py        # one case
VER_TESTS=taylor,newton python verify_ad.py
```

The sensitivity runs differentiate through the load-step scan and the staggered
while-loop in forward mode; each nonlinear solve is differentiated implicitly
at its converged state. Expect about 2 min for the PF-CZM case and a few tens
of minutes or more for each AT2 sensitivity case on one GPU.

Output (`output/`): `log_*.txt`, `verify_<MODEL>_<TAG>.json` with every number,
and `verify_<MODEL>_<TAG>.png` with the two panels of Fig. 5.
