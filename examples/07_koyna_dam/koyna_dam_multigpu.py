"""The Koyna gravity dam on several GPUs (paper Section 4.2.3).

``koyna_dam.py`` with one thing changed: the innermost linear solve (the CG of
Newton's method for the displacement and the AT2 damage subproblems) is
replaced by the row-partitioned distributed conjugate gradient of
``distributed_cg.py`` ("gather" backend), so that one large case is spread over
all GPUs.

What is distributed, and what is not
  * the rows of the operator are split into equal contiguous blocks, one per
    GPU, so each GPU stores about 1/P of the stored tangent entries;
  * every CG iteration does one all-gather of x and a local segment-sum, and
    the two inner products are reductions over the device axis;
  * the element evaluation (residual and tangent values) is left replicated.

Dirichlet conditions
  The dam base is clamped to zero. Homogeneous conditions allow the constrained
  rows and columns to be dropped, so the distributed operator is the plain SPD
  system of the free unknowns; the solution is scattered back with zeros at the
  fixed unknowns.

Launch (one process per GPU on a cluster, or all visible GPUs of one node):
    srun python koyna_dam_multigpu.py
    python koyna_dam_multigpu.py
Check the wiring without a GPU (4 emulated CPU devices; compares the
distributed solve of the Koyna operator with a dense solve):
    PFF_SELFTEST=1 python koyna_dam_multigpu.py
"""
from __future__ import annotations
import os
import sys
import time
from functools import partial

# --- env BEFORE jax is imported -------------------------------------------
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
_SELFTEST = os.environ.get('PFF_SELFTEST', '0') == '1'
if _SELFTEST and not (os.environ.get('SLURM_JOB_ID') or 'JAX_COORDINATOR_ADDRESS' in os.environ):
    # fake 4 CPU devices so the SPMD paths actually execute on a laptop
    os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

import numpy as onp
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import distributed_cg as dist             # the row-partitioned distributed CG

# Bring up the cluster ONCE, before any device array is created, then import the
# Koyna physics (which only defines functions/scalars at import -- no arrays).
dist.init_distributed()
jax.config.update('jax_enable_x64', True)

import koyna_dam as K                     # all Koyna geometry / loads / element evaluation


# =============================================================================
# Distributed reduced-SPD operator: partition ONCE (fixed mesh), reuse per step
# =============================================================================
def partition_reduced(field, fixed_idx, nshard):
    """Eliminate the (homogeneous) Dirichlet DOFs and split the resulting SPD
    free-system COO rows into `nshard` equal contiguous blocks for the gather CG.

    The mesh is fixed, so (I, J) -- hence this whole partition -- is CONSTANT.
    Only the tangent VALUES V change per Newton step; we return a gather index
    `vgather` so per step we do  loc_vals = V_ext[vgather]  (cheap, on device).

    Returns a dict with everything the per-step solve needs.
    """
    I = onp.asarray(field.coo_rows); J = onp.asarray(field.coo_cols)
    Nd = int(field.num_dofs)
    nnz = I.shape[0]

    fixed = onp.zeros(Nd, bool); fixed[onp.asarray(fixed_idx)] = True
    free = ~fixed
    free_idx = onp.where(free)[0]
    nfree = int(free_idx.size)

    rpd = -(-nfree // nshard)                 # ceil division
    Nf_pad = rpd * nshard
    new_id = onp.full(Nd, -1, onp.int64)
    new_id[free_idx] = onp.arange(nfree)      # global DOF -> compact free id

    keep = free[I] & free[J]                   # both endpoints free
    kidx = onp.where(keep)[0]                   # positions into V
    If = new_id[I[keep]]; Jf = new_id[J[keep]]

    # identity rows for the padded DOFs [nfree, Nf_pad): (k,k)=1, rhs 0 -> x=0
    pad_rows = onp.arange(nfree, Nf_pad)
    PAD, IDENT = nnz, nnz + 1                   # sentinels into V_ext=[V,0,1]
    If = onp.concatenate([If, pad_rows])
    Jf = onp.concatenate([Jf, pad_rows])
    vsrc = onp.concatenate([kidx, onp.full(pad_rows.size, IDENT, onp.int64)])

    order = onp.argsort(If, kind='stable')
    If, Jf, vsrc = If[order], Jf[order], vsrc[order]
    dev = If // rpd
    M = max(int((dev == d).sum()) for d in range(nshard))

    loc_rows = onp.zeros((nshard, M), onp.int32)
    glb_cols = onp.zeros((nshard, M), onp.int32)
    vgather = onp.full((nshard, M), PAD, onp.int64)      # pad -> value 0
    for d in range(nshard):
        m = dev == d; k = int(m.sum())
        loc_rows[d, :k] = If[m] - d * rpd                # local row in [0, rpd)
        glb_cols[d, :k] = Jf[m]                           # GLOBAL free column
        vgather[d, :k] = vsrc[m]

    return dict(loc_rows=jnp.asarray(loc_rows), glb_cols=jnp.asarray(glb_cols),
                vgather=jnp.asarray(vgather), free_idx=jnp.asarray(free_idx),
                nfree=nfree, Nf_pad=Nf_pad, rpd=rpd, nshard=nshard, Nd=Nd, nnz=nnz)


def build_distributed_solver(field, fixed_idx, mesh, tol=1e-4, maxiter=5000):
    """Return dsolve(V, rhs) -> du (global, zeros at fixed DOFs), solving the
    reduced SPD Newton system A_ff du_f = rhs_f across the device mesh."""
    pr = partition_reduced(field, fixed_idx, mesh.size)
    cg = dist.build_cg_gather(mesh, tol=tol, maxiter=maxiter)

    put = lambda a: jax.device_put(a, NamedSharding(mesh, P('i', None)))
    loc_rows = put(pr['loc_rows']); glb_cols = put(pr['glb_cols'])
    vgather = pr['vgather']; free_idx = pr['free_idx']
    nfree, Nf_pad, rpd, nshard, Nd = (pr[k] for k in
                                      ('nfree', 'Nf_pad', 'rpd', 'nshard', 'Nd'))

    def dsolve(V, rhs):
        # per-step local values: pad->0, identity->1
        V_ext = jnp.concatenate([V, jnp.array([0.0, 1.0], V.dtype)])
        loc_vals = put(V_ext[vgather])                      # (nshard, M) sharded
        b = jnp.zeros(Nf_pad, V.dtype).at[:nfree].set(rhs[free_idx])
        b_sh = put(b.reshape(nshard, rpd))
        x_sh = cg(loc_rows, glb_cols, loc_vals, b_sh)       # distributed CG
        x = jax.device_get(x_sh).reshape(-1)[:nfree]
        return jnp.zeros(Nd, V.dtype).at[free_idx].set(jnp.asarray(x))

    return dsolve


# =============================================================================
# Host-driven Newton (replaces the jitted newton with a distributed CG inside)
# =============================================================================
def newton_host(field, dsolve, u0, ivs, fixed_idx, f_ext,
                num_newton=20, nl_tol=1e-8):
    """Newton on a subproblem: assemble (device, replicated) -> distributed CG.
    All Dirichlet values are 0, so u stays 0 at fixed DOFs throughout."""
    fixed_idx = jnp.asarray(fixed_idx)
    u = jnp.asarray(u0).reshape(-1).at[fixed_idx].set(0.0)
    R, V = K.assemble_device(field, u, list(ivs), f_ext)
    R = R.at[fixed_idx].set(0.0)
    for _ in range(num_newton):
        if float(jnp.linalg.norm(R)) <= nl_tol:
            break
        du = dsolve(V, -R)
        u = u + du
        R, V = K.assemble_device(field, u, list(ivs), f_ext)
        R = R.at[fixed_idx].set(0.0)
    return u


# =============================================================================
# Host-driven staggered step (mirrors get_staggered_stepper's body, AT2 path)
# =============================================================================
def make_staggered_host(field_u, field_d, u_solver, d_solver,
                        u_fixed_idx, d_fixed_idx, mat_quad,
                        max_stagger=10, ms_tol=1e-4,
                        history_chunk_cells=100000):
    fe_u = field_u; Nu = fe_u.num_nodes
    fe_d = field_d; Nd = fe_d.num_nodes
    zero_fd = jnp.zeros(Nd * fe_d.ncomp)
    d_fixed_idx = jnp.asarray(d_fixed_idx)
    assert K.MODEL == 'AT2', ("this multi-GPU driver wires the AT2 Newton path; "
                              "CZM/AT1 use the projected-gradient solver -- see NOTES")

    def step(sol_u, sol_d, history_old, f_ext_u):
        su = jnp.asarray(sol_u); sd = jnp.asarray(sol_d)
        hist = jnp.asarray(history_old)
        for s in range(max_stagger):
            # ---- u-solve ----
            d_quad = fe_u.to_quad(sd)
            su_new = newton_host(field_u, u_solver, su, (d_quad, mat_quad),
                                 u_fixed_idx, f_ext_u).reshape(Nu, 2)
            # ---- history ----
            hist_new = compute_history_chunked(
                field_u, su_new, hist, mat_quad, history_chunk_cells)
            # ---- AT2 d-solve ----
            sd0 = sd.reshape(-1).at[d_fixed_idx].set(0.0)
            sd_new = newton_host(field_d, d_solver, sd0, (hist_new,),
                                 d_fixed_idx, zero_fd).reshape(Nd, 1)
            sd_new = jnp.clip(sd_new, 0., 1.).at[d_fixed_idx].set(0.0)
            # ---- convergence ----
            eu = float(jnp.linalg.norm(su_new - su) / (jnp.linalg.norm(su_new) + 1e-30))
            ed = float(jnp.linalg.norm(sd_new - sd) / (jnp.linalg.norm(sd_new) + 1e-30))
            su, sd, hist = su_new, sd_new, hist_new
            if max(eu, ed) <= ms_tol:
                break
        return su, sd, hist

    return step


def compute_E_pos_chunked(field_u, sol_u, mat_quad, dam_cell_mask,
                          chunk_cells=100000):
    """Compute the serial solver's ``int psi_plus dx`` diagnostic in chunks.

    ``K.compute_energy_fields`` applies ``eigh`` to every cell/quadrature point
    in one call.  On production meshes that creates an oversized cuSolver batch
    (``gpuSolverDnXsyevjBatched_bufferSize: invalid value``).  Chunking changes
    neither the quadrature nor the constitutive definition; it only bounds the
    temporary eigensolver batch.  This is a diagnostic and remains on rank 0.
    """
    fe = field_u
    su = jnp.asarray(sol_u)
    mask = jnp.asarray(dam_cell_mask)
    total = 0.0
    nc = int(fe.num_cells)
    chunk_cells = max(1, int(chunk_cells))
    for i0 in range(0, nc, chunk_cells):
        i1 = min(i0 + chunk_cells, nc)
        cell = fe.cells[i0:i1]
        u_grads = (jnp.take(su, cell, axis=0)[:, None, :, :, None]
                   * fe.shape_grads[i0:i1, :, :, None, :])
        u_grads = jnp.sum(u_grads, axis=2)
        psi_p = jax.vmap(jax.vmap(K.psi_plus_density))(
            u_grads, mat_quad[i0:i1])
        w = fe.JxW[i0:i1] * mask[i0:i1, None]
        total += float(jnp.sum(psi_p * w))
    return total


def compute_history_chunked(field_u, sol_u, history_old, mat_quad,
                            chunk_cells=100000):
    """Exact chunked equivalent of ``Displacement.compute_history``.

    The history is still ``max(psi_plus(u), H_old)`` independently at every
    quadrature point.  ``block_until_ready`` is intentional: without it JAX can
    queue all chunk eigensolves and recreate the same oversized cuSolver workload
    asynchronously, with the error then misleadingly reported by a later op.
    """
    fe = field_u
    su = jnp.asarray(sol_u)
    hold = jnp.asarray(history_old)
    vpsi = jax.vmap(jax.vmap(K.psi_plus_density))
    chunks = []
    nc = int(fe.num_cells)
    chunk_cells = max(1, int(chunk_cells))
    for i0 in range(0, nc, chunk_cells):
        i1 = min(i0 + chunk_cells, nc)
        cell = fe.cells[i0:i1]
        u_grads = (jnp.take(su, cell, axis=0)[:, None, :, :, None]
                   * fe.shape_grads[i0:i1, :, :, None, :])
        u_grads = jnp.sum(u_grads, axis=2)
        hnew = jnp.maximum(vpsi(u_grads, mat_quad[i0:i1]), hold[i0:i1])
        hnew.block_until_ready()
        chunks.append(hnew)
    return jnp.concatenate(chunks, axis=0)


# =============================================================================
# SELF-TEST: distributed solve of the REAL Koyna operator vs a dense solve
# =============================================================================
def self_test():
    """Build a small uniform Koyna, take one Newton tangent (V) at u=0, and check
    the distributed reduced-CG solution against onp.linalg.solve of the same
    reduced system.  Runs on the 4 fake CPU devices."""
    import numpy as _onp
    mesh = dist.make_domain_mesh()
    nshard = mesh.size

    pts, cells, y_notch, ys = K.koyna_mesh(nx=16, uniform_h=None, ny=24)
    pts, cells = K.insert_notch(pts, cells, y_notch, K.CRACK_LEN)
    _, field_d, field_u = K.build_fields(pts, cells)
    mat = K.material_quad(field_u, _onp.zeros(len(cells), bool))

    fixed_idx, _ = K.build_dirichlet_static(field_u, mat)
    fixed_idx = _onp.asarray(fixed_idx)

    # one tangent + residual at u = 0 (gravity only)
    f_grav = K.gravity_load(field_u, None)
    Nu = field_u.num_nodes
    u0 = jnp.zeros(Nu * 2)
    d_quad = field_u.to_quad(_onp.zeros((field_d.num_nodes, 1)))
    R, V = K.assemble_device(field_u, u0, [d_quad, mat], f_grav)
    R = R.at[fixed_idx].set(0.0)

    # reference: dense reduced solve  A_ff du_f = -R_f
    Nd = Nu * 2
    I = _onp.asarray(field_u.coo_rows); J = _onp.asarray(field_u.coo_cols); Vv = _onp.asarray(V)
    free = _onp.ones(Nd, bool); free[fixed_idx] = False
    fid = _onp.where(free)[0]; nf = fid.size
    nid = _onp.full(Nd, -1, _onp.int64); nid[fid] = _onp.arange(nf)
    A = _onp.zeros((nf, nf))
    keep = free[I] & free[J]
    _onp.add.at(A, (nid[I[keep]], nid[J[keep]]), Vv[keep])
    b = -_onp.asarray(R)[fid]
    du_ref = _onp.linalg.solve(A, b)

    # distributed
    dsolve = build_distributed_solver(field_u, fixed_idx, mesh, tol=1e-12, maxiter=20000)
    du = _onp.asarray(dsolve(V, -R))[fid]

    err = _onp.linalg.norm(du - du_ref) / (_onp.linalg.norm(du_ref) + 1e-30)
    if jax.process_index() == 0:
        print(f"\n  self-test: Koyna reduced operator, {nshard} devices, "
              f"nfree={nf}, nnz={int(keep.sum())}")
        print(f"    distributed CG vs dense solve   rel_err = {err:.3e}  "
              f"{'PASS' if err < 1e-6 else 'FAIL'}\n", flush=True)
    return err < 1e-6


# =============================================================================
# Driver
# =============================================================================
def main():
    t0 = time.time()
    mesh = dist.make_domain_mesh()
    nshard = mesh.size
    is0 = jax.process_index() == 0

    # ---- mesh (uniform) + upstream initial crack -------------------------
    UNIFORM_H = float(os.environ.get('PFF_UNIFORM_H', '0.5'))
    # Match the serial uniform driver: ``h`` is a GLOBAL size request, so NX
    # must be sized from the widest (70 m) row unless the user explicitly
    # overrides it.  Keeping NX=140 while reducing UNIFORM_H only refines the
    # vertical direction and produces a strongly anisotropic mesh.
    NX = int(os.environ.get(
        'PFF_NX', str(int(onp.ceil(K.GEOM['W_base'] / UNIFORM_H)))))
    pts, cells, y_notch, ys = K.koyna_mesh(nx=NX, uniform_h=UNIFORM_H)
    pts, cells = K.insert_notch(pts, cells, y_notch, K.CRACK_LEN)
    _, field_d, field_u = K.build_fields(pts, cells)

    is_rock = onp.zeros(len(cells), bool)          # no foundation in this driver
    mat_quad = K.material_quad(field_u, is_rock)
    f_grav = K.gravity_load(field_u, None)
    face_pre = K.upstream_face_precompute(field_u)

    u_fixed_idx, _ = K.build_dirichlet_static(field_u, mat_quad)
    u_fixed_idx = onp.asarray(u_fixed_idx)

    # Damage-admissible region -> pinned (d=0) DOFs.  This must be IDENTICAL to
    # get_staggered_stepper() in the serial solver.  In paper mode the admissible
    # region is the neck corridor, not merely everything above BASE_FREE_H.
    # Pinning only the base layer makes the first free row (y ~= 5.15 m) a
    # threshold-free AT2 nucleation line and recreates the spurious bottom crack.
    pts_d = onp.asarray(field_d.points)
    if K.PAPER_MODE:
        admissible = ((pts_d[:, 0] >= -1e-9)
                      & (pts_d[:, 0] <= K.FRACTURE_X_MAX + 1e-9)
                      & (pts_d[:, 1] >= K.FRACTURE_Y_MIN - 1e-9)
                      & (pts_d[:, 1] <= K.FRACTURE_Y_MAX + 1e-9))
        d_fixed_idx = onp.where(~admissible)[0]
    else:
        d_free_h = 0.0 if K.FOUNDATION else K.BASE_FREE_H
        d_fixed_idx = onp.where(pts_d[:, 1] <= d_free_h + 1e-9)[0]

    Nu = field_u.num_nodes
    Nd = field_d.num_nodes
    if is0:
        print(f"[koyna-mgpu] {nshard} devices | {len(pts)} nodes, {len(cells)} cells "
              f"| u-DOF {2*Nu} (free {2*Nu - u_fixed_idx.size}) | NX={NX}, h~{UNIFORM_H} m "
              f"| l={K.l} m | MODEL={K.MODEL}", flush=True)
        if K.PAPER_MODE:
            print(f"  damage corridor: x=[0,{K.FRACTURE_X_MAX:g}], "
                  f"y=[{K.FRACTURE_Y_MIN:g},{K.FRACTURE_Y_MAX:g}] m "
                  f"({Nd - d_fixed_idx.size} free, {d_fixed_idx.size} pinned damage DOFs)",
                  flush=True)
        else:
            print(f"  damage base exclusion: y <= {d_free_h:g} m "
                  f"({Nd - d_fixed_idx.size} free, {d_fixed_idx.size} pinned damage DOFs)",
                  flush=True)

    # ---- distributed solvers for u and AT2 d -----------------------------
    u_solver = build_distributed_solver(field_u, u_fixed_idx, mesh, tol=1e-4)
    d_solver = build_distributed_solver(field_d, d_fixed_idx, mesh, tol=1e-4)
    spectral_chunk = int(os.environ.get('PFF_SPECTRAL_CHUNK_CELLS',
                                        os.environ.get('PFF_ENERGY_CHUNK_CELLS',
                                                       '100000')))
    step = make_staggered_host(field_u, field_d, u_solver, d_solver,
                               u_fixed_idx, d_fixed_idx, mat_quad,
                               max_stagger=int(os.environ.get('PFF_MAX_STAGGER', '10')),
                               history_chunk_cells=spectral_chunk)

    # ---- reaction DOFs (diagnostic) --------------------------------------
    base_nodes = onp.where(onp.asarray(pts)[:, 1] < 1e-6)[0]
    base_dofs_x = jnp.asarray(base_nodes * 2 + 0)
    base_dofs_y = jnp.asarray(base_nodes * 2 + 1)

    # ---- load schedule -----------------------------------------------------
    # Two DELIBERATELY SEPARATE phases (matching koyna_dam.py):
    #   1. ELASTIC PRELOAD 0 -> H: displacement-only, damage FROZEN at 0.  The dam
    #      must not "crack" just from being filled -- only overflow drives fracture.
    #   2. OVERFLOW H -> H+ovf_max: full staggered u+d, fracture active.
    # After the preload, the irreversible history is INITIALISED (not accumulated)
    # from the equilibrated full-reservoir elastic state.
    # Accept the serial driver's documented names.  Retain the old spellings as
    # fallbacks so existing cluster scripts continue to work.
    n_fill = int(os.environ.get('PFF_N_FILL', os.environ.get('PFF_NFILL', '12')))
    n_ovf = int(os.environ.get('PFF_N_OVF', os.environ.get('PFF_NOVF', '50')))
    ovf_max = float(os.environ.get('PFF_OVF_MAX', os.environ.get('PFF_OVFMAX', '10.0')))
    preload_levels = onp.linspace(0.0, K.H_FULL, n_fill + 1)
    overflow_levels = onp.linspace(K.H_FULL, K.H_FULL + ovf_max, n_ovf + 1)

    sol_u = jnp.zeros((len(pts), 2))
    sol_d = jnp.zeros((len(pts), 1))
    dam_mask = jnp.asarray(~is_rock, dtype=jnp.float64)
    hist = {k: [] for k in ('level', 'overflow', 'max_d', 'E_pos', 'E_frac',
                            'base_Rx', 'base_Ry', 'crest_ux', 'solve_time',
                            'diagnostic_time', 'elapsed_time')}
    preload_step_time = []
    energy_chunk = int(os.environ.get('PFF_ENERGY_CHUNK_CELLS',
                                      str(spectral_chunk)))
    crest_node = int(onp.argmin((onp.asarray(pts)[:, 0]) ** 2
                                + (onp.asarray(pts)[:, 1] - K.GEOM['H']) ** 2))

    def _record(lv, su, sd, f_ext, solve_time=0.0):
        if not is0:
            return
        diagnostic_start = time.time()
        E_pos = compute_E_pos_chunked(field_u, su, mat_quad, dam_mask,
                                      energy_chunk)
        E_frac = K.compute_fracture_energy(field_d, sd)
        Rx, Ry = K.compute_base_reaction(field_u, su, sd, mat_quad,
                                         f_ext, base_dofs_x, base_dofs_y)
        md = float(jnp.max(sd)); cu = float(jnp.abs(su[crest_node, 0]))
        diagnostic_time = time.time() - diagnostic_start
        elapsed_time = time.time() - t0
        for key, val in (('level', lv), ('overflow', lv - K.H_FULL), ('max_d', md),
                         ('E_pos', E_pos), ('E_frac', E_frac),
                         ('base_Rx', Rx), ('base_Ry', Ry), ('crest_ux', cu),
                         ('solve_time', solve_time),
                         ('diagnostic_time', diagnostic_time),
                         ('elapsed_time', elapsed_time)):
            hist[key].append(val)
        print(f"  level={lv:7.2f}m ovf={lv-K.H_FULL:+5.2f}m  max_d={md:.4f}  "
              f"E_pos={E_pos:.3e}  Rx={Rx:8.2f} Ry={Ry:7.2f}  "
              f"t={time.time()-t0:.1f}s", flush=True)

    # ---- phase 1: elastic preload, damage frozen (distributed u-solve) -----
    d_quad0 = field_u.to_quad(sol_d)   # d = 0 throughout
    preload_start = time.time()
    for lv in preload_levels:
        lv = float(lv)
        f_ext = f_grav + K.hydro_load(face_pre, lv)
        load_start = time.time()
        sol_u = newton_host(field_u, u_solver, sol_u, (d_quad0, mat_quad),
                            u_fixed_idx, f_ext).reshape(len(pts), 2)
        preload_step_time.append(time.time() - load_start)
        if is0:
            print(f"  preload level={lv:7.2f}m  damage frozen  "
                  f"t={time.time()-t0:.1f}s", flush=True)
    preload_time = time.time() - preload_start

    # initialise (NOT accumulate) history from the full-reservoir elastic state
    history = compute_history_chunked(
        field_u, sol_u,
        jnp.zeros((field_u.num_cells, field_u.num_quads)),
        mat_quad, spectral_chunk)
    _record(float(K.H_FULL), sol_u, sol_d, f_grav + K.hydro_load(face_pre, K.H_FULL))

    # ---- phase 2: overflow, fracture active (distributed staggered) --------
    for k in range(1, len(overflow_levels)):
        lv = float(overflow_levels[k])
        f_ext = f_grav + K.hydro_load(face_pre, lv)
        load_start = time.time()
        sol_u, sol_d, history = step(sol_u, sol_d, history, f_ext)
        solve_time = time.time() - load_start
        _record(lv, sol_u, sol_d, f_ext, solve_time)

    if is0:
        out = K.output_dir
        total_wall_time = time.time() - t0
        u_dof_total = 2 * Nu
        u_dof_free = u_dof_total - int(u_fixed_idx.size)
        d_dof_total = Nd
        d_dof_free = d_dof_total - int(d_fixed_idx.size)
        onp.savez(os.path.join(out, f'history_koyna_mgpu_{K.MODEL}.npz'),
                  **{k: onp.asarray(v) for k, v in hist.items()},
                  preload_step_time=onp.asarray(preload_step_time),
                  preload_time=preload_time,
                  total_wall_time=total_wall_time,
                  n_nodes=int(len(pts)), n_cells=int(len(cells)),
                  u_dof_total=u_dof_total, u_dof_free=u_dof_free,
                  d_dof_total=d_dof_total, d_dof_free=d_dof_free,
                  total_dof=u_dof_total + d_dof_total,
                  total_free_dof=u_dof_free + d_dof_free,
                  n_gpus=int(nshard), uniform_h=UNIFORM_H, nx=NX,
                  n_fill=n_fill, n_ovf=n_ovf, ovf_max=ovf_max)
        K.plot_quad_mesh(pts, cells, os.path.join(out, f'solution_final_mgpu_{K.MODEL}.png'),
                         f'Koyna final damage (max d = {float(jnp.max(sol_d)):.2f})',
                         sol_d=onp.asarray(sol_d))
        print(f"  saved DOFs: total={u_dof_total + d_dof_total}, "
              f"free={u_dof_free + d_dof_free}; "
              f"preload={preload_time:.1f}s, solve wall={total_wall_time:.1f}s",
              flush=True)
        print(f"[koyna-mgpu] done in {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == '__main__':
    try:
        if _SELFTEST:
            self_test()
        else:
            main()
    finally:
        if os.environ.get('SLURM_JOB_ID') or 'JAX_COORDINATOR_ADDRESS' in os.environ:
            jax.experimental.multihost_utils.sync_global_devices('done')
            jax.distributed.shutdown()

# =============================================================================
# Notes
# =============================================================================
#  * Only the AT2 model (Newton + CG for both subproblems) is wired here.
#  * The element evaluation is replicated on every device; the cell
#    decomposition of axion.distributed evaluates only the owned cells.
#  * The "gather" backend is used, which handles the duplicated notch nodes.
