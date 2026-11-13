# 03 - Parallel scaling on several GPUs (Sections 2.7 and 4.1.3)

The 3D benchmark of `02_sent_3d` solved with the cell decomposition of
`axion.distributed`: the mesh is partitioned into one subdomain per GPU, every
device evaluates only its own cells, operator actions are preceded by a halo
exchange and the CG inner products are global reductions.

```bash
GPUS=0,1,2,3 bash run_scaling.sh strong    # fixed mesh 150 x 150 x 15 on 1, 2, 4 GPUs
GPUS=0,1,2,3 bash run_scaling.sh weak      # 15 layers per GPU: NZ = 15, 30, 60
python plot_scaling.py strong              # table + scaling_strong_AT2.png
python plot_scaling.py weak
```

A single run uses all visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python sent_3d_multigpu.py
```

Paper (Fig. 4): strong scaling 57.8 / 29.7 / 15.7 min on 1 / 2 / 4 GPUs
(efficiency 92% on four); weak scaling 57.8 / 58.6 / 60.1 min (96%).

The timed run is preceded by a run on a zero-load schedule that compiles all
kernels (`PFF_WARMUP=1`), so the reported time excludes compilation. The
partition uses METIS (`pymetis`); `PFF_PARTITION=sfc` or `slab` need no extra
package.

Check without a GPU (4 emulated CPU devices, tiny mesh, compares the
distributed staggered solution with a dense serial one):

```bash
python -m axion.distributed                 # kernels only (from the repository root)
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    PFF_VERIFY=1 PFF_PARTITION=sfc python sent_3d_multigpu.py
```

Output: one folder `output_<tag>_<MODEL>/` per run and one row per run in
`scaling_<mode>.csv`.
