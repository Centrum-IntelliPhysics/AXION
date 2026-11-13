"""Row-partitioned distributed solvers for an operator stored as (row, column,
value) triplets.

The rows of the operator are split into equal contiguous blocks, one per
device. Every device stores only its block of the entries and of the vectors;
the conjugate gradient (and projected gradient) iterations run under
``shard_map`` with the inner products reduced over all devices.

Two communication backends for the operator action:

  "gather"  all-gather the whole vector x in every iteration. Works for any
            sparsity (unstructured meshes, duplicated crack nodes). Costs O(N)
            memory and one all-gather per iteration.
  "halo"    strip decomposition of a structured grid: every device stores its
            strip and a one-row ghost layer exchanged with its neighbours.
            Minimal memory and communication, but assumes banded coupling.

This module distributes the LINEAR SOLVE only; the element evaluation of the
caller is unchanged. For the cell decomposition in which each device also
evaluates only its own elements, see ``axion.distributed``.

Every kernel is checked against a serial reference on 4 emulated CPU devices:

    python distributed_cg.py

Multi-node launch: one process per GPU (for example ``srun`` with
``--ntasks-per-node`` equal to the GPUs per node); ``init_distributed`` detects
SLURM.
"""
from __future__ import annotations
import os
import inspect as _inspect
from functools import partial

import numpy as onp
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# shard_map graduated to the top level in jax >= 0.8. Fall back for older jax.
try:
    from jax import shard_map
except Exception:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

# The replication checker kwarg was renamed check_rep (old) -> check_vma (new).
# Our CG/PG carries become "axis-varying" through all_gather/psum, which the
# checker rejects too conservatively, so we disable it in a version-portable way.
_SMKW = {}
if "check_vma" in _inspect.signature(shard_map).parameters:
    _SMKW["check_vma"] = False
elif "check_rep" in _inspect.signature(shard_map).parameters:
    _SMKW["check_rep"] = False


# =============================================================================
# SECTION 1.  Cluster bring-up + device meshes
# =============================================================================
def init_distributed():
    """Initialise the multi-process cluster ONCE, before any jax array exists.
    No-op when launched single-process. On SLURM this needs NO arguments:
    coordinator address, process count and this process's device are all read
    from the SLURM environment automatically."""
    if os.environ.get("SLURM_JOB_ID") or "JAX_COORDINATOR_ADDRESS" in os.environ:
        # Pin THIS process to the correct local GPU. With >1 task per node (e.g.
        # 2 GPUs/node) and no --gpu-bind, every task on a node sees ALL its GPUs,
        # so each must claim the SLURM_LOCALID-th one; if the cgroup already
        # isolated this task to a single GPU it shows up as ordinal 0 -> use [0].
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        _n_visible = len([x for x in _cvd.split(",") if x != ""])
        _local_id = int(os.environ.get("SLURM_LOCALID", "0"))
        if _n_visible > 1:
            jax.distributed.initialize(local_device_ids=[_local_id])
        else:
            jax.distributed.initialize()
    if jax.process_index() == 0:
        print(f"[dist] processes={jax.process_count()}  "
              f"global_devices={jax.device_count()}  "
              f"local_devices={jax.local_device_count()}", flush=True)
    return jax.device_count()


def make_domain_mesh(axis="i"):
    """1-D mesh spanning EVERY GPU on EVERY node (pure domain decomposition).
    Built DIRECTLY from jax.devices() because jax.make_mesh rejects multi-node
    ("multi-slice") topologies (e.g. 2 nodes x 1 GPU). This form works for both
    single-node and multi-node."""
    return Mesh(onp.asarray(jax.devices(), dtype=object), (axis,))


def make_hybrid_mesh(n_ensemble, ensemble_axis="e", domain_axis="d"):
    """2-D mesh (ensemble, domain) for HYBRID data x domain parallelism.
    n_ensemble independent cases, each domain-decomposed over the remaining
    devices.  Requires device_count % n_ensemble == 0."""
    ndev = jax.device_count()
    assert ndev % n_ensemble == 0, (
        f"device_count={ndev} not divisible by n_ensemble={n_ensemble}")
    devs = onp.asarray(jax.devices(), dtype=object).reshape(
        n_ensemble, ndev // n_ensemble)
    return Mesh(devs, (ensemble_axis, domain_axis))


# =============================================================================
# SECTION 2.  Host-side partitioning of the assembled COO system
# =============================================================================
# The operator is given as unassembled triplets: I, J index rows/columns (for an
# axion.Field: field.coo_rows, field.coo_cols) and V holds the values of the
# current Newton step (field.tangent_values).
# We split ROWS into equal contiguous blocks (one per domain shard). Equal row
# counts are REQUIRED so the solution/rhs vectors shard evenly; nnz imbalance is
# absorbed by padding to a common length (zero-padding contributes nothing).
#
# LOAD BALANCING / SMALLER HALOS (optional `perm`):
#   Contiguous blocks of the natural DOF numbering can give fat halos or uneven
#   nnz. Passing a global DOF permutation `perm` (e.g. from METIS / Scotch, or a
#   space-filling curve) relabels DOFs so each contiguous block is a compact,
#   balanced subdomain. We apply perm to (I,J,b) and invert it on the solution,
#   so the ANSWER is identical -- only the partition quality changes.
# -----------------------------------------------------------------------------
def partition_coo(I, J, V, Nd, num_shards, perm=None):
    """Split global COO rows into `num_shards` equal contiguous blocks.

    Returns (loc_rows, glb_cols, loc_vals) each shaped (num_shards, max_nnz),
    plus rows_per_shard.  If `perm` (length-Nd int array, a permutation) is
    given, DOFs are relabelled by it first (see note above).
    """
    I = onp.asarray(I).copy(); J = onp.asarray(J).copy(); V = onp.asarray(V)
    assert Nd % num_shards == 0, (
        f"Nd={Nd} not divisible by num_shards={num_shards}; pad the system "
        f"(append identity rows) so it divides evenly.")
    if perm is not None:
        # inv[old] = new position; relabel both row and column indices
        inv = onp.empty(Nd, onp.int64); inv[onp.asarray(perm)] = onp.arange(Nd)
        I = inv[I]; J = inv[J]
    rpd = Nd // num_shards
    order = onp.argsort(I, kind="stable")
    I, J, V = I[order], J[order], V[order]
    dev = I // rpd
    max_nnz = max(int((dev == d).sum()) for d in range(num_shards))
    loc_rows = onp.zeros((num_shards, max_nnz), onp.int32)
    glb_cols = onp.zeros((num_shards, max_nnz), onp.int32)
    loc_vals = onp.zeros((num_shards, max_nnz), onp.asarray(V).dtype)
    for d in range(num_shards):
        m = dev == d; k = int(m.sum())
        loc_rows[d, :k] = I[m] - d * rpd          # local row index in [0, rpd)
        glb_cols[d, :k] = J[m]                     # GLOBAL column index
        loc_vals[d, :k] = V[m]
    return loc_rows, glb_cols, loc_vals, rpd


def partition_strip(I, J, V, n_rows, n_cols, num_shards):
    """Structured STRIP partition for the "halo" backend.

    Splits a structured grid of shape (n_rows, n_cols) [node (r,c) -> dof
    r*n_cols+c] into `num_shards` horizontal strips.  Each shard's columns are
    remapped into a LOCAL extended layout
          [ top_halo(n_cols) | owned(strip*n_cols) | bottom_halo(n_cols) ]
    so the SpMV only needs a 1-row ghost layer from each neighbour.
    Assumes nearest-neighbour (banded) coupling -- true for Q1 quads / 5-9 pt
    stencils on a structured grid. Returns (loc_rows, loc_cols_ext, loc_vals,
    owned, n_cols).
    """
    I = onp.asarray(I); J = onp.asarray(J); V = onp.asarray(V)
    assert n_rows % num_shards == 0, "n_rows must divide num_shards for strips"
    rows_per = n_rows // num_shards
    owned = rows_per * n_cols
    dev_of_row = (I // n_cols) // rows_per
    max_nnz = max(int((dev_of_row == d).sum()) for d in range(num_shards))
    LR = onp.zeros((num_shards, max_nnz), onp.int32)
    LC = onp.zeros((num_shards, max_nnz), onp.int32)
    LV = onp.zeros((num_shards, max_nnz), onp.asarray(V).dtype)
    for d in range(num_shards):
        m = dev_of_row == d; k = int(m.sum())
        Id, Jd, Vd = I[m], J[m], V[m]
        r0 = d * rows_per                          # first grid row on shard d
        LR[d, :k] = Id - r0 * n_cols               # owned local row
        grow = Jd // n_cols                         # grid row of each column
        top = grow == (r0 - 1)                      # lives in the top ghost row
        bot = grow == (r0 + rows_per)               # lives in the bottom ghost row
        own = (~top) & (~bot)
        le = onp.empty(k, onp.int32)
        le[top] = (Jd[top] % n_cols)                              # [0, n_cols)
        le[own] = n_cols + (Jd[own] - r0 * n_cols)               # owned band
        le[bot] = n_cols + owned + (Jd[bot] % n_cols)            # bottom ghost
        LC[d, :k] = le; LV[d, :k] = Vd
    return LR, LC, LV, owned, n_cols


# =============================================================================
# SECTION 3.  Local SpMV backends (run INSIDE shard_map, one shard per device)
# =============================================================================
# Each backend returns a `matvec(x_local) -> y_local` closure plus the local
# Jacobi inverse-diagonal. The CG/PG loops (Section 4/5) are backend-agnostic.
# -----------------------------------------------------------------------------
def _gather_backend(loc_rows, glb_cols, loc_vals, rpd, axis):
    """all_gather backend: replicate x, multiply locally. Any sparsity."""
    row0 = jax.lax.axis_index(axis) * rpd
    def matvec(x_loc):
        x_full = jax.lax.all_gather(x_loc, axis, tiled=True)   # (Nd,) on device
        return jax.ops.segment_sum(loc_vals * x_full[glb_cols],
                                   loc_rows, num_segments=rpd)
    diag = jax.ops.segment_sum(jnp.where(glb_cols == loc_rows + row0, loc_vals, 0.0),
                               loc_rows, num_segments=rpd)
    inv = 1.0 / jnp.where(jnp.abs(diag) > 1e-14, diag, 1.0)
    return matvec, inv, diag


def _halo_backend(loc_rows, loc_cols_ext, loc_vals, owned, n_cols, axis, nshard):
    """ppermute halo backend: exchange only a 1-row ghost layer with neighbours.
    Memory-optimal; assumes the structured strip layout from partition_strip."""
    def matvec(x_owned):
        # send my BOTTOM owned row down (s -> s+1) => neighbour's TOP ghost
        send_bot = x_owned[owned - n_cols:owned]
        top_halo = jax.lax.ppermute(send_bot, axis,
                                    perm=[(s, s + 1) for s in range(nshard - 1)])
        # send my TOP owned row up (s -> s-1) => neighbour's BOTTOM ghost
        send_top = x_owned[0:n_cols]
        bot_halo = jax.lax.ppermute(send_top, axis,
                                    perm=[(s, s - 1) for s in range(1, nshard)])
        x_ext = jnp.concatenate([top_halo, x_owned, bot_halo])
        return jax.ops.segment_sum(loc_vals * x_ext[loc_cols_ext],
                                   loc_rows, num_segments=owned)
    # diagonal: owned column of row i sits at ext index n_cols + (i)  (own band)
    local_rows = jnp.arange(owned)
    diag = jax.ops.segment_sum(
        jnp.where(loc_cols_ext == (n_cols + loc_rows), loc_vals, 0.0),
        loc_rows, num_segments=owned)
    inv = 1.0 / jnp.where(jnp.abs(diag) > 1e-14, diag, 1.0)
    return matvec, inv, diag


# =============================================================================
# SECTION 4.  Distributed preconditioned CG  (backend-agnostic core)
# =============================================================================
def _pcg_local(matvec, dot, inv, b, tol, maxiter):
    """Block-Jacobi PCG operating on this shard's local vectors.
    `dot(u,v)` must already all-reduce over the domain axis (one psum)."""
    bnorm2 = dot(b, b)
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    z = inv * r
    p = z
    rz = dot(r, z)

    def body(state):
        x, r, p, rz, k = state
        Ap = matvec(p)
        alpha = rz / dot(p, Ap)                 # reduction #1 / iter
        x = x + alpha * p
        r = r - alpha * Ap
        z = inv * r
        rz_new = dot(r, z)                      # reduction #2 / iter
        p = z + (rz_new / rz) * p
        return x, r, p, rz_new, k + 1

    def cond(state):
        r = state[1]; k = state[4]
        # relative residual stop:  ||r||^2 <= tol^2 ||b||^2
        return (k < maxiter) & (dot(r, r) > (tol * tol) * bnorm2)

    final = jax.lax.while_loop(cond, body, (x, r, p, rz, 0))
    return final[0], final[4]                   # x_local, iteration count


def build_cg_gather(mesh, axis="i", tol=1e-6, maxiter=5000):
    """Distributed CG, all_gather backend (any mesh)."""
    @partial(shard_map, mesh=mesh, in_specs=(P(axis, None),) * 4,
             out_specs=P(axis, None), **_SMKW)
    def solve(loc_rows, glb_cols, loc_vals, b_shard):
        lr, gc, lv, b = loc_rows[0], glb_cols[0], loc_vals[0], b_shard[0]
        matvec, inv, _ = _gather_backend(lr, gc, lv, b.shape[0], axis)
        dot = lambda u, v: jax.lax.psum((u * v).sum(), axis)
        x, _ = _pcg_local(matvec, dot, inv, b, tol, maxiter)
        return x.reshape(1, -1)
    return jax.jit(solve)


def build_cg_halo(mesh, owned, n_cols, nshard, axis="i", tol=1e-6, maxiter=5000):
    """Distributed CG, ppermute halo backend (structured strip, memory-optimal)."""
    @partial(shard_map, mesh=mesh, in_specs=(P(axis, None),) * 4,
             out_specs=P(axis, None), **_SMKW)
    def solve(loc_rows, loc_cols_ext, loc_vals, b_shard):
        lr, lc, lv, b = loc_rows[0], loc_cols_ext[0], loc_vals[0], b_shard[0]
        matvec, inv, _ = _halo_backend(lr, lc, lv, owned, n_cols, axis, nshard)
        dot = lambda u, v: jax.lax.psum((u * v).sum(), axis)
        x, _ = _pcg_local(matvec, dot, inv, b, tol, maxiter)
        return x.reshape(1, -1)
    return jax.jit(solve)


# =============================================================================
# SECTION 5.  Distributed projected-gradient (bound-constrained damage solve)
# =============================================================================
def _pg_local(residual, dot, inv_Dm, y0, omega, max_it, tol, lo, hi):
    """Fixed-step projected gradient on this shard.  Same update as your
    serial pg_solve:  y <- clip(y - omega * R(y) / Dm, lo, hi)."""
    y = jnp.clip(y0, lo, hi)

    def body(state):
        y, _, k = state
        R = residual(y)                              # distributed residual
        y_new = jnp.clip(y - omega * R * inv_Dm, lo, hi)
        return y_new, y, k + 1

    def cond(state):
        y, y_prev, k = state
        change = jnp.sqrt(dot(y - y_prev, y - y_prev)) / (jnp.sqrt(dot(y, y)) + 1e-30)
        return (k < max_it) & (change > tol)

    y, _, k = jax.lax.while_loop(cond, body, (y, y + 1.0, 0))
    return y, k


def build_pg_gather(mesh, axis="i", omega=1.0, max_it=300, tol=1e-7, lo=0.0, hi=1.0):
    """Distributed PG for AT1/CZM damage, all_gather backend.
    Here the residual is the linear stand-in R(y)=A y - b (sharded COO); in your
    code you pass a residual closure wrapping the distributed cell-loop assembly
    -- see the WIRING GUIDE below."""
    @partial(shard_map, mesh=mesh, in_specs=(P(axis, None),) * 6,
             out_specs=P(axis, None), **_SMKW)
    def solve(loc_rows, glb_cols, loc_vals, b_shard, y0_shard, Dm_shard):
        lr, gc, lv = loc_rows[0], glb_cols[0], loc_vals[0]
        b, y0, Dm = b_shard[0], y0_shard[0], Dm_shard[0]
        matvec, _, _ = _gather_backend(lr, gc, lv, b.shape[0], axis)
        residual = lambda y: matvec(y) - b
        dot = lambda u, v: jax.lax.psum((u * v).sum(), axis)
        inv_Dm = 1.0 / jnp.where(jnp.abs(Dm) > 1e-14, Dm, 1.0)
        y, _ = _pg_local(residual, dot, inv_Dm, y0, omega, max_it, tol, lo, hi)
        return y.reshape(1, -1)
    return jax.jit(solve)


# =============================================================================
# SECTION 6.  HYBRID data x domain CG  (several big cases at once)
# =============================================================================
def build_hybrid_cg(mesh, e_axis="e", d_axis="d", tol=1e-6, maxiter=5000):
    """Solve a BATCH of independent SPD systems, each domain-decomposed.
    Operator arrays are shaped (E, D, max_nnz); rhs (E, D, rpd). The psum
    reduces over the DOMAIN axis ONLY, so ensemble members never communicate."""
    @partial(shard_map, mesh=mesh,
             in_specs=(P(e_axis, d_axis, None),) * 4,
             out_specs=P(e_axis, d_axis, None), **_SMKW)
    def solve(loc_rows, glb_cols, loc_vals, b_shard):
        lr, gc, lv, b = loc_rows[0, 0], glb_cols[0, 0], loc_vals[0, 0], b_shard[0, 0]
        matvec, inv, _ = _gather_backend(lr, gc, lv, b.shape[0], d_axis)
        dot = lambda u, v: jax.lax.psum((u * v).sum(), d_axis)   # reduce over d only
        x, _ = _pcg_local(matvec, dot, inv, b, tol, maxiter)
        return x.reshape(1, 1, -1)
    return jax.jit(solve)


# =============================================================================
# SECTION 7.  Convenience host wrappers (device_put + gather back)
# =============================================================================
def _put(mesh, spec, a):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def solve_cg(I, J, V, b, mesh=None, backend="gather", grid=None,
             tol=1e-6, maxiter=5000, perm=None):
    """Solve SPD A x = b across the cluster. `backend`='gather' (any mesh) or
    'halo' (structured; pass grid=(n_rows,n_cols)). Returns global x."""
    if mesh is None:
        mesh = make_domain_mesh()
    nshard = mesh.size
    Nd = b.shape[0]
    if backend == "gather":
        lr, gc, lv, rpd = partition_coo(I, J, V, Nd, nshard, perm=perm)
        b_sh = _reshape_rows(b, nshard, rpd, perm)
        solve = build_cg_gather(mesh, tol=tol, maxiter=maxiter)
        x_sh = solve(*(_put(mesh, P("i", None), a) for a in (lr, gc, lv, b_sh)))
        x = _gather_rows(x_sh, perm, Nd)
    elif backend == "halo":
        assert grid is not None, "halo backend needs grid=(n_rows, n_cols)"
        n_rows, n_cols = grid
        lr, lc, lv, owned, nc = partition_strip(I, J, V, n_rows, n_cols, nshard)
        b_sh = onp.asarray(b).reshape(nshard, owned)
        solve = build_cg_halo(mesh, owned, nc, nshard, tol=tol, maxiter=maxiter)
        x_sh = solve(*(_put(mesh, P("i", None), a) for a in (lr, lc, lv, b_sh)))
        x = onp.asarray(jax.experimental.multihost_utils.process_allgather(x_sh, tiled=True)).reshape(-1)
    else:
        raise ValueError(backend)
    return x


def solve_pg(I, J, V, b, Dm, y0=None, mesh=None, **kw):
    """Bound-constrained solve of A y = b, 0<=y<=1 (all_gather backend)."""
    if mesh is None:
        mesh = make_domain_mesh()
    nshard = mesh.size; Nd = b.shape[0]
    lr, gc, lv, rpd = partition_coo(I, J, V, Nd, nshard)
    if y0 is None:
        y0 = onp.zeros(Nd)
    r = lambda a: onp.asarray(a).reshape(nshard, rpd)
    solve = build_pg_gather(mesh, **kw)
    y_sh = solve(*(_put(mesh, P("i", None), a) for a in (lr, gc, lv, r(b), r(y0), r(Dm))))
    return onp.asarray(jax.experimental.multihost_utils.process_allgather(y_sh, tiled=True)).reshape(-1)


def solve_hybrid(I, J, V, B_rhs, n_ensemble, tol=1e-6, maxiter=5000):
    """Solve E independent systems (same A here; in practice each e can have its
    own V) with a 2-D (ensemble, domain) mesh. B_rhs shape (E, Nd)."""
    mesh = make_hybrid_mesh(n_ensemble)
    E, Dn = mesh.shape["e"], mesh.shape["d"]
    Nd = B_rhs.shape[1]; rpd = Nd // Dn
    lr, gc, lv, _ = partition_coo(I, J, V, Nd, Dn)
    bc = lambda a: onp.broadcast_to(a, (E,) + a.shape).copy()
    LR, GC, LV = bc(lr), bc(gc), bc(lv)                    # (E, D, max_nnz)
    Be = B_rhs.reshape(E, Dn, rpd)
    solve = build_hybrid_cg(mesh, tol=tol, maxiter=maxiter)
    x_sh = solve(*(_put(mesh, P("e", "d", None), a) for a in (LR, GC, LV, Be)))
    return onp.asarray(
        jax.experimental.multihost_utils.process_allgather(x_sh, tiled=True)).reshape(E, Nd)


# --- small helpers for the optional DOF permutation round-trip ---------------
def _reshape_rows(b, nshard, rpd, perm):
    b = onp.asarray(b)
    if perm is not None:
        inv = onp.empty(b.shape[0], onp.int64); inv[onp.asarray(perm)] = onp.arange(b.shape[0])
        b = b[onp.asarray(perm)]              # permuted rhs matches permuted rows
    return b.reshape(nshard, rpd)

def _gather_rows(x_sh, perm, Nd):
    x = onp.asarray(jax.experimental.multihost_utils.process_allgather(x_sh, tiled=True)).reshape(-1)
    if perm is not None:
        out = onp.empty(Nd, x.dtype); out[onp.asarray(perm)] = x   # undo relabelling
        return out
    return x


# =============================================================================
# SECTION 8.  USAGE NOTES
# =============================================================================
#   1. The index tables I, J are constant on a fixed mesh: build the partition
#      once. Pad the number of rows up to a multiple of the number of devices
#      with identity rows (entry (k, k, 1) and a zero right-hand side).
#   2. Dirichlet unknowns: eliminate them on the host before partitioning, so
#      that the distributed operator stays symmetric positive definite (this is
#      what examples/07_koyna_dam/koyna_dam_multigpu.py does).
#   3. Per Newton step only the values V change; gather them into the per-device
#      layout and call the distributed CG.
#   Possible extensions: a permutation of the unknowns from a graph partitioner
#   (argument `perm`) for compact, balanced blocks; a coarse-grid correction to
#   keep the iteration count flat for many devices.
# =============================================================================


# =============================================================================
# SECTION 9.  SELF-TESTS on FAKE CPU devices   (run:  python distributed_cg.py)
# =============================================================================
def _lap_coo(n):
    I, J, V = [], [], []
    idx = lambda i, j: i * n + j
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    I.append(k); J.append(idx(ii, jj)); V.append(-1.0)
            I.append(k); J.append(k); V.append(4.0)
    return onp.array(I), onp.array(J), onp.array(V)


def self_test():
    # Grid size is chosen as a multiple of the device count so that EVERY test
    # (gather / halo / hybrid) partitions evenly on any number of GPUs.
    ndev = jax.device_count()
    k = max(2, round(40 / ndev))
    n = ndev * k; N = n * n
    I, J, V = _lap_coo(n)
    A = onp.zeros((N, N)); A[I, J] += V
    onp.random.seed(0)
    b = onp.random.rand(N)
    x_ref = onp.linalg.solve(A, b)
    ndev = jax.device_count()
    results = []

    # 1. CG, gather backend
    x = solve_cg(I, J, V, b, backend="gather", tol=1e-12)
    e = onp.linalg.norm(x - x_ref) / onp.linalg.norm(x_ref)
    results.append(("CG gather", e));

    # 2. CG, gather backend WITH a random DOF permutation (round-trip must match)
    rng = onp.random.default_rng(3); perm = rng.permutation(N)
    xp = solve_cg(I, J, V, b, backend="gather", tol=1e-12, perm=perm)
    results.append(("CG gather+perm", onp.linalg.norm(xp - x_ref) / onp.linalg.norm(x_ref)))

    # 3. CG, halo backend (structured strip; grid = n x n)
    xh = solve_cg(I, J, V, b, backend="halo", grid=(n, n), tol=1e-12)
    results.append(("CG halo(ppermute)", onp.linalg.norm(xh - x_ref) / onp.linalg.norm(x_ref)))

    # 4. PG, bound-constrained, vs serial projected-gradient iteration
    Dm = onp.abs(onp.diag(A)); onp.random.seed(1); bpg = 0.5 * onp.random.rand(N)
    y = onp.zeros(N); yp = y + 1.0; k = 0
    while k < 500 and onp.linalg.norm(y - yp) / (onp.linalg.norm(y) + 1e-30) > 1e-10:
        yp = y; y = onp.clip(y - (A @ y - bpg) / Dm, 0, 1); k += 1
    yd = solve_pg(I, J, V, bpg, Dm, omega=1.0, max_it=500, tol=1e-10)
    results.append(("PG gather", onp.linalg.norm(yd - y) / (onp.linalg.norm(y) + 1e-30)))

    # 5. HYBRID data x domain (only if devices factor as E*D with E>1)
    if ndev % 2 == 0 and ndev >= 4:
        E = 2
        rng = onp.random.default_rng(4); Brhs = rng.random((E, N))
        xref2 = onp.stack([onp.linalg.solve(A, Brhs[e]) for e in range(E)])
        xd2 = solve_hybrid(I, J, V, Brhs, n_ensemble=E, tol=1e-12)
        results.append(("Hybrid ExD", onp.linalg.norm(xd2 - xref2) / onp.linalg.norm(xref2)))

    ok = all(err < 1e-8 for _, err in results)
    if jax.process_index() == 0:                 # print once, not once per process
        print(f"\n  self-test on {ndev} devices, N={N}:")
        for name, err in results:
            print(f"    {name:22s} rel_err={err:.3e}  {'PASS' if err < 1e-8 else 'FAIL'}")
        print("  ALL PASS\n" if ok else "  SOME FAILED\n")
    return ok


if __name__ == "__main__":
    if not (os.environ.get("SLURM_JOB_ID") or "JAX_COORDINATOR_ADDRESS" in os.environ):
        # fake 4 CPU devices so the SPMD paths actually execute on a laptop
        os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_enable_x64", True)
    init_distributed()
    self_test()
    # clean multi-process teardown -> avoids the benign 'connection refused' /
    # CANCELLED WatchTasks warnings printed when processes exit unsynchronised.
    if os.environ.get("SLURM_JOB_ID") or "JAX_COORDINATOR_ADDRESS" in os.environ:
        jax.experimental.multihost_utils.sync_global_devices("done")
        jax.distributed.shutdown()