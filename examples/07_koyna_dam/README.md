# 07 - The Koyna gravity dam (Section 4.2.3)

Standard benchmark for the fracture analysis of concrete gravity dams, with the
geometry and data of Santillan et al. (Eng. Fract. Mech. 178, 2017): cross-section
103 m high and 70 m wide at the base, horizontal notch of 1.93 m on the upstream
face at 66.5 m, E = 25 GPa, nu = 0.2, G_f = 100 N/m, density 2450 kg/m^3, clamped
base, plane strain, AT2 model. Loads in three stages: self-weight, hydrostatic
pressure of the reservoir, then an overflow raised to 10 m.

The paper resolves the damage band uniformly across the whole structure:
l = 0.03 m and h = l/2 = 0.015 m everywhere, three-node triangles, 9.6e7 DoFs.

```bash
bash run_koyna.sh                                   # one GPU (about 96 GB), paper: 6.50 h
CUDA_VISIBLE_DEVICES=0,1 bash run_koyna_multigpu.sh # two GPUs, paper: 3.17 h
```

Small test on a graded mesh (refined around the expected crack path only):

```bash
PFF_COARSE=1 PFF_UNIFORM_H=0 PFF_L=0.3 bash run_koyna.sh
```

Files
* `koyna_dam.py` - geometry, mesh, loads, fields and the single-GPU solver.
  Self-weight and reservoir pressure enter the displacement residual as
  consistent nodal load vectors.
* `koyna_dam_multigpu.py` - the same computation with the linear solves
  distributed over the GPUs by `distributed_cg.py` (row partition of the
  operator; the element evaluation is replicated).
  `PFF_SELFTEST=1 python koyna_dam_multigpu.py` checks the distributed solve
  against a dense solve on 4 emulated CPU devices.

Output: `koyna_AT2.xdmf` (+ `.h5`) time series of damage and displacement,
`timing_AT2.txt`, diagnostics (energies, base reaction, crest displacement) and
figures; the multi-GPU driver writes `history_koyna_mgpu_AT2.npz`.

Settings are environment variables documented in `koyna_dam.py`.
