import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
from mpi4py import MPI

from dolfinx import io

import gmsh_io
from amr import AMRState, mark_cells
from problem import PFFProblem, STAGGER_TOL, SOLVER_SUMMARY

comm = MPI.COMM_WORLD
rank = comm.rank

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = {"uniform": os.path.join(HERE, "mesh", "paper_uniform", "plate.msh"),
        "adaptive": os.path.join(HERE, "mesh", "adaptive", "plate.msh")}
CSV_HEADER = ("step,disp,force,force_kN,max_d,cells,dofs,"
              "stagger_its,ms_err,n_remesh,t_wall")


class Params:
    E0 = 210_000.0
    nu = 0.3
    Gc = 2.7
    l = 0.03
    Lx = 1.0
    Ly = 1.0
    x_tip = 0.5
    k_res = 1e-8
    quad_deg = 2
    max_stagger = 1000
    ms_tol = STAGGER_TOL
    amr_alpha = 0.8
    amr_d_threshold = 0.5

    def __init__(self):
        self.mu = self.E0 / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.h = self.l / 3.0


class XDMFWriter:
    def __init__(self, base):
        self.base = base
        self.f = None
        self.msh = None
        self.seg = 0
        if rank == 0:
            import glob
            for fp in glob.glob(f"{base}*.xdmf") + glob.glob(f"{base}*.h5"):
                os.remove(fp)
        comm.Barrier()

    def write(self, prob, t):
        if prob.msh is not self.msh:
            if self.f is not None:
                self.f.close()
            name = (f"{self.base}.xdmf" if self.seg == 0
                    else f"{self.base}_remesh{self.seg:03d}.xdmf")
            self.f = io.XDMFFile(comm, name, "w")
            self.f.write_mesh(prob.msh)
            self.msh = prob.msh
            self.seg += 1
        for fn in prob.output_fields():
            self.f.write_function(fn, t)

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def peak_rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r * 1024 if sys.platform.startswith("linux") else r


def save_crack(prob, path):
    cd = prob.global_cell_damage()
    if rank == 0:
        np.savez(path, pts=prob.state.pts, cells=prob.state.cells, cd=cd,
                 dim=prob.dim, n_cells=prob.n_cells_global)


def load_state(mesh_path):
    if rank == 0:
        pts, cells = gmsh_io.read_msh(mesh_path, 2)
        cells = gmsh_io.reorder_structured(pts, cells, 2)
        payload = (pts, cells)
    else:
        payload = None
    pts, cells = comm.bcast(payload, root=0)
    return AMRState(pts, cells)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", dest="adaptive", action="store_true")
    ap.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    ap.set_defaults(adaptive=False)
    ap.add_argument("--delta-u", type=float, default=1e-6)
    ap.add_argument("--max-disp", type=float, default=0.007)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    P = Params()
    tag = "adaptive" if args.adaptive else "uniform"
    mesh_path = MESH[tag]
    out_dir = args.out or os.path.join(HERE, "output", f"AT2-{tag}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()

    n_steps = int(round(args.max_disp / args.delta_u))
    disps = np.arange(n_steps + 1, dtype=np.float64) * args.delta_u
    if args.nsteps is not None:
        disps = disps[: args.nsteps + 1]

    prob = PFFProblem(load_state(mesh_path), P)
    writer = XDMFWriter(os.path.join(out_dir, "AT2"))

    if rank == 0:
        print(f"\n{'=' * 70}\n"
              f"  FEniCSx phase-field fracture, AT2, 2-D {tag} mesh\n"
              f"  mesh={os.path.relpath(mesh_path, HERE)}  cells={prob.n_cells_global}  "
              f"dofs={prob.n_dofs_global}  hmin={P.h:.4g} mm\n"
              f"  E={P.E0:.6g} N/mm^2  nu={P.nu}  Gc={P.Gc} N/mm  l={P.l:.4g} mm  "
              f"k_res={P.k_res:.0e}  plane strain, spectral split, DG0 history\n"
              f"  loading: {len(disps) - 1} x {args.delta_u:.0e} mm "
              f"(to {disps[-1]:.4g} mm)  fields every {args.save_every} step(s)\n"
              f"  solver: {SOLVER_SUMMARY}\n"
              f"  MPI ranks={comm.size}\n{'=' * 70}", flush=True)

    save_crack(prob, os.path.join(out_dir, "crack_initial_AT2.npz"))
    csv = None
    if rank == 0:
        csv = open(os.path.join(out_dir, "results_AT2.csv"), "w")
        csv.write(CSV_HEADER + "\n")

    t0 = time.time()
    t_io = 0.0
    forces = [0.0]

    for i, disp in enumerate(disps[1:], start=1):
        n_remesh = 0
        while True:
            energy_old = prob.global_cell_tensile_energy() if args.adaptive else None
            s_its, ms_err, converged = prob.staggered_step(disp)
            if not converged:
                if rank == 0:
                    print(f"  [warn] staggered not converged at step {i}, "
                          f"disp={disp:.6g}: ms_err={ms_err:.3e} after {s_its} its",
                          flush=True)
                break
            if not args.adaptive:
                break
            cd = prob.global_cell_max_damage()
            energy = prob.global_cell_tensile_energy()
            eta = energy - energy_old
            marked = np.where(mark_cells(eta, cd, prob.state.cell_sizes(), P.h,
                                         alpha=P.amr_alpha,
                                         d_threshold=P.amr_d_threshold))[0]
            if rank == 0:
                print(f"  AMR: max_cell_d={cd.max():.3f}  max_dpsi={eta.max():.3e}  "
                      f"marked={len(marked)}", flush=True)
            if len(marked) == 0:
                break
            prob.state.refine(marked)
            new = PFFProblem(prob.state, P)
            new.transfer_from(prob)
            prob = new
            n_remesh += 1

        force = prob.reaction_force()
        maxd = prob.max_d()
        forces.append(force)
        row = (i, disp, force, force / 1000.0, maxd, prob.n_cells_global,
               prob.n_dofs_global, s_its, ms_err, n_remesh, time.time() - t0)
        if rank == 0:
            print(f"Step {i:5d}/{len(disps) - 1}  disp={disp:.4e}  "
                  f"|Fy|={abs(force):.3e} N/mm  max_d={maxd:.3f}  "
                  f"dofs={prob.n_dofs_global}  stag={s_its}  remesh={n_remesh}  "
                  f"t={time.time() - t0:.1f}s", flush=True)

        t_w = time.time()
        if rank == 0:
            csv.write(",".join(f"{v:.18e}" for v in row) + "\n")
            csv.flush()
        if i % args.save_every == 0 or i == len(disps) - 1:
            writer.write(prob, disp)
        t_io += time.time() - t_w

    writer.close()
    save_crack(prob, os.path.join(out_dir, "crack_final_AT2.npz"))

    dt = time.time() - t0
    prb = peak_rss_bytes()
    mem_sum_gb = comm.allreduce(prb, op=MPI.SUM) / 1e9
    if rank == 0:
        csv.close()
        np.savez(os.path.join(out_dir, "sol_AT2.npz"), disps=disps,
                 forces=np.asarray(forces))
        meta = {
            "model": "AT2", "mode": tag, "n_ranks": comm.size,
            "solver": "gmres-ilu", "solver_summary": SOLVER_SUMMARY,
            "units": "N-mm; force per unit thickness",
            "E_N_per_mm2": P.E0, "nu": P.nu, "Gc_N_per_mm": P.Gc, "l_mm": P.l,
            "Lx_mm": P.Lx, "Ly_mm": P.Ly, "notch_mm": P.x_tip,
            "mesh": os.path.relpath(mesh_path, HERE),
            "delta_u_mm": args.delta_u, "max_disp_mm": float(disps[-1]),
            "n_steps": len(disps) - 1, "save_every": args.save_every,
            "hmin_mm": P.h,
            "amr_marking": (f"energy increment > {P.amr_alpha} x max or damage > "
                            f"{P.amr_d_threshold}" if args.adaptive else "none"),
            "plane_assumption": "plane_strain",
            "elastic_split": "spectral_tension_compression",
            "element_u": "Q1", "element_d": "Q1",
            "history_space": "DG0", "quadrature_degree": P.quad_deg,
            "residual_stiffness": P.k_res,
            "max_stagger": P.max_stagger, "stagger_tolerance": P.ms_tol,
            "final_cells": int(prob.n_cells_global),
            "final_dofs": int(prob.n_dofs_global),
            "total_time_s": dt, "output_time_s": t_io, "solve_time_s": dt - t_io,
            "time_per_step_s": dt / max(len(disps) - 1, 1),
            "peak_mem_total_gb": mem_sum_gb,
        }
        with open(os.path.join(out_dir, "meta_AT2.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"\n{'=' * 70}\n  DONE  AT2  {tag}  final dofs={prob.n_dofs_global}  "
              f"cells={prob.n_cells_global}\n"
              f"  total {dt:.1f} s = output {t_io:.1f} s + solve {dt - t_io:.1f} s "
              f"({(dt - t_io) / 60:.2f} min)\n"
              f"  peak memory {mem_sum_gb:.2f} GB over {comm.size} rank(s)\n"
              f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
