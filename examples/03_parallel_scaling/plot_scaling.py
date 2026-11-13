"""Plot a scaling sweep written by run_scaling.sh (paper Fig. 4).

    python plot_scaling.py strong     # reads scaling_strong.csv
    python plot_scaling.py weak       # reads scaling_weak.csv

Strong scaling: fixed problem size; ideal wall-clock time T(1) / D and parallel
efficiency T(1) / (D T(D)). Weak scaling: constant work per GPU; ideal time
T(1) and efficiency T(1) / T(D). The times exclude compilation (PFF_WARMUP=1).
If several rows share the same D, the median is used.
"""
import csv
import os
import sys

import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("PFF_MODEL", os.environ.get("MODEL", "AT2"))


def load(path, model):
    if not os.path.exists(path):
        raise SystemExit(f"No {path}. Run the sweep first (run_scaling.sh).")
    by_d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                by_d.setdefault(int(row["D"]), []).append(row)
    if not by_d:
        raise SystemExit(f"No rows for model={model} in {path}.")
    D = sorted(by_d)
    med = lambda key: onp.array([onp.median([float(r[key]) for r in by_d[d]]) for d in D])
    return (onp.array(D), med("solve_time_s"), med("n_dofs"), med("dofs_per_gpu"),
            med("peak_mem_GiB"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "strong"
    if mode not in ("strong", "weak"):
        raise SystemExit("usage: plot_scaling.py strong|weak")
    D, time_s, dofs, dofs_per_gpu, mem = load(os.path.join(HERE, f"scaling_{mode}.csv"), MODEL)
    t_ref, d_ref = time_s[0], D[0]
    ideal = t_ref * d_ref / D if mode == "strong" else onp.full(len(D), t_ref)
    efficiency = 100.0 * ideal / time_s

    print(f"\n{mode} scaling ({MODEL})\n")
    print(f"  {'D':>3} | {'total DoFs':>12} | {'DoFs / GPU':>12} | {'time [min]':>10} | "
          f"{'efficiency':>10} | {'peak mem [GiB]':>14}")
    print("  " + "-" * 78)
    for i, d in enumerate(D):
        print(f"  {d:>3d} | {int(dofs[i]):>12d} | {int(dofs_per_gpu[i]):>12d} | "
              f"{time_s[i] / 60.0:>10.2f} | {efficiency[i]:>9.1f}% | {mem[i]:>14.2f}")

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.plot(D, ideal / 60.0, "--", color="0.5", lw=1.2, label="Ideal")
    ax.plot(D, time_s / 60.0, "o-", color="#1f3a68", lw=1.8, ms=5, label="measured")
    for d, t, e in zip(D, time_s, efficiency):
        ax.annotate(f"{t / 60.0:.1f} min\n$\\eta$ = {e:.1f}%", (d, t / 60.0),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(D)
    ax.set_xticklabels([str(d) for d in D])
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Wall-clock time [min]")
    ax.set_title(f"{mode.capitalize()} scaling")
    ax.grid(alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, f"scaling_{mode}_{MODEL}.png")
    fig.savefig(out, dpi=200)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
