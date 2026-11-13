"""Two-dimensional single-edge-notched tension with adaptive refinement
(paper Sections 2.6 and 4.1.1).

Same specimen, material and loading as ``sent_2d_uniform.py``, but the
computation starts from a coarse 48 x 48 QUAD4 mesh and refines around the
evolving crack with the quadtree scheme of Section 2.6:

  * a cell is marked when its nodal damage exceeds PFF_PHI_TH (0.5) and its
    size still exceeds PFF_TARGET_H (l/3); marked cells are split 1 -> 4 and
    the 2:1 balance is restored,
  * the mesh lives in fixed-capacity buffers ("arenas"), so refinement changes
    array values but not array shapes and the load loop stays compiled,
  * if the number of live cells or nodes approaches the capacity, the buffers
    are enlarged by the factor PFF_GROW and the kernels are compiled again
    (one "capacity bucket" per growth).

Modules
    amr_mesh.py     fixed-capacity quadtree: refine, 2:1 balance, hanging nodes
    amr_fem.py      QUAD4 operators (elasticity, AT2 damage), constrained CG
    amr_driver.py   boundary conditions and the refine-inside-the-load-loop driver

Environment variables
    PFF_NX0          cells per direction of the initial mesh (default 48)
    PFF_LMAX         maximum refinement level (default 3)
    PFF_TARGET_H     element size below which a cell is not refined (default l/3)
    PFF_PHI_TH       damage threshold of the refinement indicator (default 0.5)
    PFF_NSTEPS       number of load increments (default 700)
    PFF_DU           displacement increment in mm (default 1e-5)
    PFF_C0_FAC, PFF_N0_FAC, PFF_H0_FAC   initial capacities as multiples of the
                     initial mesh (defaults 2, 2, 0.5)
    PFF_GROW, PFF_HI growth factor and fill fraction that triggers growth
                     (defaults 1.1 and 0.9)
    PFF_OUTPUT_DIR   output folder (default ./output_adaptive)

Output
    force_disp_AMR.h5   disps, forces and run attributes
    final_AMR.h5        final mesh (points, cells), damage, displacement
    summary_AMR.txt, final_damage_AMR.png, load_displacement_AMR.png
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)   # int64 node keys are load-bearing
import numpy as onp
import time, os, math, h5py

import amr_driver
from amr_fem import Material
from amr_mesh import Lattice

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PFF_OUTPUT_DIR", os.path.join(HERE, "output_adaptive"))
os.makedirs(OUT, exist_ok=True)
MODEL = "AMR"


def _memory_report(tag):
    """Print live and peak allocator bytes when the GPU backend exposes them."""
    if os.environ.get("PFF_MEM_DIAG", "1") == "0":
        return
    mib = 1024.0 ** 2
    for dev in jax.local_devices():
        stats = dev.memory_stats() or {}
        values = []
        for key in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "bytes_limit"):
            if key in stats:
                values.append(f"{key}={stats[key] / mib:.1f} MiB")
        print(f"[gpu-mem] {tag} {dev}: " + (", ".join(values) or "unavailable"),
              flush=True)


def _require_grid_alignment(value, spacing, name):
    index = round(value / spacing)
    aligned = index * spacing
    if not math.isclose(value, aligned, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"{name}={value:.12g} is not on the initial grid (spacing={spacing:.12g}); "
            f"nearest grid coordinate is {aligned:.12g}.")


def main():
    # region parameters
    # =============================================================================
    # 1. Material and model parameters (same as sent_2d_uniform.py)
    # =============================================================================
    E0     = 210e3        # Young's modulus            [N/mm^2]
    nu    = 0.3          # Poisson's ratio            [-]
    Gc   = 2.7          # critical energy release    [N/mm]
    l     = 0.03          # phase-field length scale   [mm]
    mat = Material(E=E0, nu=nu, Gc=Gc, l=l, k_res=1e-8)
    Lx = Ly = 1.0

    # -- mesh, refinement and loading controls --
    NX0 = int(os.environ.get("PFF_NX0", "48"))
    LMAX = int(os.environ.get("PFF_LMAX", "3"))
    TARGET_HMIN = float(os.environ.get(
        "PFF_TARGET_H", os.environ.get("PFF_TARGET_HMIN", f"{l / 3.0:.6g}")))  # finest h = l/3
    NSTEPS = int(os.environ.get("PFF_NSTEPS", "700"))
    DU = float(os.environ.get("PFF_DU", "1e-5"))   # 700 * 1e-5 -> final displacement 0.007 mm
    MAX_STAGGER = int(os.environ.get("PFF_MAX_STAGGER", "1000"))
    PHI_TH = float(os.environ.get("PFF_PHI_TH", "0.5"))
    Y_CRACK = float(os.environ.get("PFF_Y_CRACK", "0.5"))
    X_TIP = float(os.environ.get("PFF_X_TIP", "0.5"))

    if NX0 <= 0:
        raise ValueError(f"PFF_NX0 must be positive, got {NX0}")
    if LMAX < 0:
        raise ValueError(f"PFF_LMAX must be non-negative, got {LMAX}")
    if TARGET_HMIN <= 0.0:
        raise ValueError(f"PFF_TARGET_H must be positive, got {TARGET_HMIN}")
    if NSTEPS <= 0 or DU <= 0.0:
        raise ValueError(f"PFF_NSTEPS and PFF_DU must be positive, got {NSTEPS}, {DU}")
    if MAX_STAGGER <= 0:
        raise ValueError(f"PFF_MAX_STAGGER must be positive, got {MAX_STAGGER}")
    if not 0.0 <= PHI_TH <= 1.0:
        raise ValueError(f"PFF_PHI_TH must lie in [0, 1], got {PHI_TH}")
    if not 0.0 < Y_CRACK < Ly or not 0.0 < X_TIP < Lx:
        raise ValueError(
            f"Require 0<Y_CRACK<{Ly} and 0<X_TIP<{Lx}; got {Y_CRACK}, {X_TIP}")

    h0 = Lx / NX0
    _require_grid_alignment(Y_CRACK, Ly / NX0, "PFF_Y_CRACK")
    _require_grid_alignment(X_TIP, h0, "PFF_X_TIP")
    finest_h = h0 / (2 ** LMAX)
    required_lmax = max(0, int(math.ceil(math.log2(h0 / TARGET_HMIN))))
    if TARGET_HMIN < finest_h * (1.0 - 1e-12):
        raise ValueError(
            f"PFF_TARGET_H={TARGET_HMIN:.6g} is unreachable with NX0={NX0}, "
            f"LMAX={LMAX}; finest_h={finest_h:.6g}. Set PFF_LMAX>={required_lmax} "
            "or use a larger target h.")
    if TARGET_HMIN >= h0:
        print(f"[config-warning] target_h={TARGET_HMIN:.6g} >= h0={h0:.6g}; "
              "size-based refinement is disabled at level 0.", flush=True)

    amr_driver.MAX_STAGGER = MAX_STAGGER
    amr_driver.PHI_TH = PHI_TH
    lat = Lattice(NX0=NX0, NY0=NX0, LMAX=LMAX, Lx=Lx, Ly=Ly)

    # 0, DU, 2*DU, ..., NSTEPS*DU (NSTEPS load increments).
    disps = jnp.asarray(onp.arange(NSTEPS + 1, dtype=onp.float64) * DU)

    # -- bucketing parameters (start modest; capacity self-grows) --
    #   seg_len small enough that one segment cannot add > (1-hi)*C_cap cells;
    #   the run RAISES a clear error if a segment overflows (then lower seg_len).
    # The defaults allocate twice the initial mesh, which exceeds the peak of the
    # default run, so no capacity growth (and no recompilation) is triggered.
    # Lower them (via PFF_*_FAC) to exercise the growth path.
    C0_FAC = float(os.environ.get("PFF_C0_FAC", "2.0"))
    N0_FAC = float(os.environ.get("PFF_N0_FAC", "2.0"))
    H0_FAC = float(os.environ.get("PFF_H0_FAC", "0.5"))
    GROW = float(os.environ.get("PFF_GROW", "1.1"))
    HI = float(os.environ.get("PFF_HI", "0.9"))
    SEG_LEN = int(os.environ.get("PFF_SEG_LEN", "1"))
    if min(C0_FAC, N0_FAC, H0_FAC) <= 0.0:
        raise ValueError("PFF_C0_FAC, PFF_N0_FAC, and PFF_H0_FAC must be positive")
    if GROW <= 1.0 or not 0.0 < HI < 1.0 or SEG_LEN <= 0:
        raise ValueError(f"Require GROW>1, 0<HI<1, SEG_LEN>0; got {GROW}, {HI}, {SEG_LEN}")

    n_cells0 = NX0 * NX0
    n_nodes0 = (NX0 + 1) * (NX0 + 1)
    slit_nodes0 = int(onp.count_nonzero(
        onp.arange(NX0 + 1, dtype=onp.float64) * h0 < X_TIP - 1e-9))
    C0 = max(n_cells0, int(math.ceil(n_cells0 * C0_FAC)))
    N0 = max(n_nodes0 + slit_nodes0,
             int(math.ceil((n_nodes0 + slit_nodes0) * N0_FAC)))
    H0 = max(1, int(math.ceil(n_cells0 * H0_FAC)))
    #----------------------------------------------------------------------------
    #  (C0,N0,H0) are the initial capacities; GROW is the growth factor; HI is the
    #  high-water mark for growth; SEG_LEN is the number of load steps per segment.
    #----------------------------------------------------------------------------

    print(f"mode-I AMR-in-scan (bucketed): NX0={NX0}, LMAX={LMAX}, h0={h0:.6g}, "
          f"finest_h={finest_h:.6g}, target_h={TARGET_HMIN:.6g}, l={l}, "
          f"slit=(y={Y_CRACK}, x<{X_TIP}), "
          f"nsteps={NSTEPS}, du={DU:.4e}, final_disp={float(disps[-1]):.4e}, "
          f"phi_th={PHI_TH}, max_stagger={MAX_STAGGER}, seg_len={SEG_LEN}, "
          f"grow={GROW}, hi={HI}, start caps=(C{C0},N{N0},H{H0})", flush=True)
    _memory_report("before AMR run")
    wall0 = time.time(); t0 = time.time()
    carry, forces, n_live = amr_driver.run_bucketed(
        mat, lat, disps, C0=C0, N0=N0, H0=H0, Ly=Ly,
        y_crack=Y_CRACK, x_tip=X_TIP, target_hmin=TARGET_HMIN,
        seg_len=SEG_LEN, grow=GROW, hi=HI)
    onp.asarray(forces)
    _memory_report("after solve")
    solve_time = time.time() - t0
    wall_time = time.time() - wall0
    state = carry[0]; sol_u = carry[1]; sol_d = carry[2]

    # -- 4. extract the final adapted mesh (LIVE cells; node ids already 0..n-1) --
    n_nodes = int(state.n_nodes)
    points = onp.asarray(state.node_xy)[:n_nodes]
    damage = onp.asarray(sol_d)[:n_nodes]
    disp = onp.asarray(sol_u).reshape(-1, 2)[:n_nodes]   # (n_nodes, 2) nodal (ux,uy)
    live = onp.asarray(state.cell_valid)
    cells = onp.asarray(state.cell_nodes)[live]
    n_cells = int(live.sum())

    disps_np = onp.asarray(disps); forces_np = onp.asarray(forces)
    peak_load = float(onp.max(onp.abs(forces_np)))
    print(f"done in {solve_time:.1f}s   live cells {int(n_live[0])} -> {n_cells}  nodes {n_nodes}")
    print(f"peak |F| = {peak_load:.2f} at step {int(onp.abs(forces_np).argmax())}/{len(forces_np)}")

    # -- 5. save the load-displacement record and the final state --
    with h5py.File(os.path.join(OUT, f"force_disp_{MODEL}.h5"), "w") as f:
        f.create_dataset("disps", data=disps_np)
        f.create_dataset("forces", data=forces_np)
        f.attrs["n_cells"] = n_cells; f.attrs["n_nodes"] = n_nodes
        f.attrs["solve_time"] = solve_time; f.attrs["wall_time"] = wall_time
        f.attrs["peak_load"] = peak_load
        f.attrs["NX0"] = NX0; f.attrs["LMAX"] = LMAX
        f.attrs["target_h"] = TARGET_HMIN; f.attrs["du"] = DU
        f.attrs["nsteps"] = NSTEPS; f.attrs["phi_th"] = PHI_TH
        f.attrs["y_crack"] = Y_CRACK; f.attrs["x_tip"] = X_TIP
        f.attrs["crack_type"] = "geometric slit: y=Y_CRACK, x<X_TIP"
    with h5py.File(os.path.join(OUT, f"final_{MODEL}.h5"), "w") as f:
        f.create_dataset("points", data=points)
        f.create_dataset("cells", data=cells)
        f.create_dataset("sol_d", data=damage.reshape(-1))
        f.create_dataset("sol_u", data=disp)
        f.create_dataset("disps", data=disps_np)
        f.create_dataset("forces", data=forces_np)
        f.attrs["n_cells"] = n_cells; f.attrs["n_nodes"] = n_nodes
        f.attrs["solve_time"] = solve_time; f.attrs["wall_time"] = wall_time
        f.attrs["peak_load"] = peak_load
        f.attrs["Lx"] = float(Lx); f.attrs["Ly"] = float(Ly)
        f.attrs["NX0"] = NX0; f.attrs["LMAX"] = LMAX
        f.attrs["target_h"] = TARGET_HMIN; f.attrs["du"] = DU
        f.attrs["nsteps"] = NSTEPS; f.attrs["phi_th"] = PHI_TH
        f.attrs["y_crack"] = Y_CRACK; f.attrs["x_tip"] = X_TIP
        f.attrs["crack_type"] = "geometric slit: y=Y_CRACK, x<X_TIP"
    print(f"saved force_disp_{MODEL}.h5 and final_{MODEL}.h5 in {OUT}")

    # Compact study summary: DOF, time, peak load, peak displacement.
    # DOF on the final adapted mesh: 2 displacement + 1 damage per node.
    n_dofs = 3 * n_nodes
    _abs_f = onp.abs(forces_np)
    _peak_i = int(_abs_f.argmax())
    peak_disp = float(disps_np[_peak_i])
    study_summary = (
        f"=============== SUMMARY ({MODEL}, adaptive) ===============\n"
        f"  DOF        : {n_dofs}   (final mesh: {n_nodes} nodes, {n_cells} cells)\n"
        f"  time [s]   : {solve_time:.3f}\n"
        f"  peak load  : {peak_load:.6f}   [N/mm]\n"
        f"  peak disp  : {peak_disp:.6e}   [mm]\n"
        f"==========================================================")
    print("\n" + study_summary)
    with open(os.path.join(OUT, f"summary_{MODEL}.txt"), "w") as fh:
        fh.write(study_summary + "\n")

    # -- 6. figures --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # Final damage / crack path on the adapted mesh.
    d_flat = damage.reshape(-1)
    polys = points[cells]
    d_cell = d_flat[cells].mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    pc = PolyCollection(polys, array=d_cell, cmap="inferno",
                        edgecolors="white", linewidths=0.3)
    pc.set_clim(0., 1.); ax.add_collection(pc)
    ax.set_xlim(polys[:, :, 0].min(), polys[:, :, 0].max())
    ax.set_ylim(polys[:, :, 1].min(), polys[:, :, 1].max())
    ax.set_aspect("equal")
    ax.set_title(f"Final crack path + adapted mesh ({n_dofs} DOFs, max d = {float(d_flat.max()):.2f})")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02, label="damage  d")
    out_d = os.path.join(OUT, f"final_damage_{MODEL}.png")
    fig.tight_layout(); fig.savefig(out_d, dpi=200); plt.close(fig)

    # Load-displacement response.
    n = min(len(disps_np), len(forces_np))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(disps_np[:n], onp.abs(forces_np[:n]), "-", color="tab:red", lw=1.8,
            marker="o", ms=3, markevery=max(1, n // 40), label=f"{MODEL} (adaptive)")
    ax.axvline(peak_disp, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Top-surface displacement [mm]")
    ax.set_ylabel("Top-surface reaction force [N/mm]")
    ax.set_title("Load-displacement response")
    ax.grid(alpha=0.4); ax.set_xlim(left=0.); ax.set_ylim(bottom=0.)
    ax.legend(frameon=True)
    out_f = os.path.join(OUT, f"load_displacement_{MODEL}.png")
    fig.tight_layout(); fig.savefig(out_f, dpi=200); plt.close(fig)
    print(f"saved {os.path.basename(out_d)} and {os.path.basename(out_f)}")


if __name__ == "__main__":
    main()
