"""Mode-I phase-field fracture with fixed-capacity adaptive mesh refinement,
where the ENTIRE load loop -- including remeshing (mark -> refine -> 2:1 balance)
-- runs inside a single jax.lax.scan.

Because every mesh array has a fixed capacity (C_cap cells, N_cap nodes), the
scan body has a constant shape and compiles exactly ONCE; the mesh then evolves
purely by changing array *values* (which cells/nodes are live, their coords and
connectivity), never their shapes, so there is no recompilation and no host
round-trip across the whole simulation.

Scheme per load step (all on device):
  1. staggered solve (u given d, history update, d given history) on current mesh
  2. mark cells where phi = cell-mean damage exceeds a threshold and refine them
     1->4, then restore 2:1 balance -- fields are carried by exact P1 transfer
  3. staggered solve again on the (possibly) refined mesh
  4. record the top-face reaction force

Other phase-field models (AT1, PF-CZM) or refinement indicators can be used by
changing the phase-field element operators or the marker, without touching the
fixed-capacity and scan machinery.
"""
import jax
import jax.numpy as jnp
import numpy as onp

from amr_mesh import (Lattice, MeshState, initial_mesh, refine, balance,
                      hanging_constraints, compact, INVALID_KEY)
from amr_fem import (Material, assemble_elasticity, assemble_phasefield,
                     update_history, constrained_cg)


# ---- fixed structural bounds of the scan body (all compile-time constants) ----
MAX_STAGGER = 1000       # FEniCSx reference cap
MS_TOL = 1e-5            # FEniCSx relative staggered-update tolerance
ADAPT_SWEEPS = 8         # bounded equivalent of FEniCSx repeat-until-no-marks
BALANCE_SWEEPS = 4       # inner 2:1-balance sweeps per refine
PHI_TH = 0.5             # refine a cell when its mean damage exceeds this
CG_TOL = 1e-6            # same CG tolerance as the uniform-mesh solver
CG_MAXIT = 3000
NL_TOL = 1e-8            # subproblem (Newton) residual tolerance
NUM_NEWTON = 30          # FEniCSx displacement Newton cap


def linear_newton(matvec, rhs, diag, free, dir_val, hd, ad, bd, ww):
    """Solve a linear subproblem the way the uniform-mesh solver does: an initial constrained-CG
    solve (to CG_TOL), then Newton / iterative-refinement steps that solve a
    homogeneous-Dirichlet correction for the remaining residual until the free-dof
    residual < NL_TOL (capped at NUM_NEWTON).  A single CG only reaches CG_TOL, so
    this drives the sub-problem residual to the tighter NL_TOL and removes small
    fluctuations, exactly as the Newton while_loop of the uniform-mesh solver does."""
    from amr_fem import constrained_cg as _cg
    x = _cg(matvec, rhs, diag, free, dir_val, hd, ad, bd, ww, tol=CG_TOL, maxiter=CG_MAXIT)
    zero = jnp.zeros_like(dir_val)

    def cond(st):
        x, it = st
        return (it < NUM_NEWTON) & (jnp.linalg.norm((rhs - matvec(x)) * free) > NL_TOL)

    def body(st):
        x, it = st
        dx = _cg(matvec, rhs - matvec(x), diag, free, zero, hd, ad, bd, ww,
                 tol=CG_TOL, maxiter=CG_MAXIT)      # homogeneous-Dirichlet correction
        return x + dx, it + 1

    x, _ = jax.lax.while_loop(cond, body, (x, jnp.array(0)))
    return x


def split_elastic_newton(nx, cn, cv, u, d, mat, Ndof, free, dir_val, hc):
    """Newton solve for the nonlinear spectral split-stress equilibrium."""
    hd, ad, bd, ww = hc
    # Impose physical Dirichlet values, then restore homogeneous hanging-node
    # interpolation in the current nonlinear iterate.
    u = jnp.where(free, u, dir_val)
    real = ww != 0.0
    hs = jnp.where(real, hd, 0)
    aa = jnp.where(real, ad, 0)
    bb = jnp.where(real, bd, 0)
    u = u.at[hs].set(jnp.where(real, ww * (u[aa] + u[bb]), u[hs]))
    zero = jnp.zeros_like(dir_val)

    def cond(st):
        u, err, it = st
        return (it < NUM_NEWTON) & (err > NL_TOL)

    def body(st):
        u, _, it = st
        residual, tangent, diag = assemble_elasticity(
            nx, cn, cv, u.reshape(-1, 2), d[:, None], mat, Ndof)
        du = constrained_cg(tangent, -residual, diag, free, zero,
                            hd, ad, bd, ww, tol=CG_TOL, maxiter=CG_MAXIT)
        u = u + du
        residual2, _, _ = assemble_elasticity(
            nx, cn, cv, u.reshape(-1, 2), d[:, None], mat, Ndof)
        return u, jnp.linalg.norm(residual2 * free), it + 1

    residual0, _, _ = assemble_elasticity(
        nx, cn, cv, u.reshape(-1, 2), d[:, None], mat, Ndof)
    u, _, _ = jax.lax.while_loop(
        cond, body, (u, jnp.linalg.norm(residual0 * free), jnp.array(0)))
    return u


def node_IJ(node_key, lat):
    """Recover integer-lattice coordinates (I, J) from node keys."""
    J = node_key % jnp.int64(lat.NY + 1)
    I = node_key // jnp.int64(lat.NY + 1)
    return I, J


def expand_hang(hid, ma, mb, w, vec):
    """Expand node-level hanging constraints to dof level for a vec-field.
    Padding rows (hid < 0) are made inert: dof 0 with weight 0."""
    real = hid >= 0
    k = jnp.arange(vec)
    hd = jnp.where(real[:, None], hid[:, None] * vec + k, 0)
    ad = jnp.where(real[:, None], ma[:, None] * vec + k, 0)
    bd = jnp.where(real[:, None], mb[:, None] * vec + k, 0)
    ww = jnp.broadcast_to(jnp.where(real, w, 0.0)[:, None], (real.shape[0], vec))
    return hd.reshape(-1), ad.reshape(-1), bd.reshape(-1), ww.reshape(-1)


def build_constraints(state, lat, disp, hang_cap, Ly):
    """Build the fixed-shape Dirichlet + hanging constraint data for the current
    mesh, for both the displacement (vec=2) and damage (vec=1) sub-problems."""
    N_cap = state.node_key.shape[0]
    live = jnp.arange(N_cap) < state.n_nodes
    I, J = node_IJ(state.node_key, lat)
    bottom = live & (J == 0)
    top = live & (J == lat.NY)

    hid, ma, mb, w, n_flag = hanging_constraints(state, lat, hang_cap)
    hang_node = jnp.zeros(N_cap, dtype=bool).at[jnp.where(hid >= 0, hid, 0)].set(hid >= 0)

    # ---- displacement (vec = 2): bottom clamps ux,uy; top prescribes uy=disp --
    ux_dir = bottom
    uy_dir = bottom | top
    free_ux = live & (~ux_dir) & (~hang_node)
    free_uy = live & (~uy_dir) & (~hang_node)
    free_u = jnp.zeros(N_cap * 2, dtype=bool)
    free_u = free_u.at[0::2].set(free_ux).at[1::2].set(free_uy)
    dir_u = jnp.zeros(N_cap * 2)
    dir_u = dir_u.at[1::2].set(jnp.where(top, disp, 0.0))
    hu = expand_hang(hid, ma, mb, w, 2)

    # ---- damage (vec = 1): no Dirichlet, only hanging ----
    free_d = live & (~hang_node)
    dir_d = jnp.zeros(N_cap)
    hd = expand_hang(hid, ma, mb, w, 1)

    return (free_u, dir_u, hu), (free_d, dir_d, hd), (hid, ma, mb, w, top, n_flag)


def staggered_solve(state, u, d, d_old, hist, disp, mat, lat, hang_cap, Ly):
    """MAX_STAGGER alternating-minimization passes on the current (fixed) mesh."""
    N_cap = state.node_key.shape[0]
    Ndof_u, Ndof_d = N_cap * 2, N_cap
    (free_u, dir_u, hu), (free_d, dir_d, hd), extra = build_constraints(
        state, lat, disp, hang_cap, Ly)
    hid, ma, mb, w, top, n_flag = extra
    nx, cn, cv = state.node_xy, state.cell_nodes, state.cell_valid

    def one_pass(u, d, hist):
        # --- nonlinear u-solve: (g+k) sigma_plus + sigma_minus ---
        u = split_elastic_newton(nx, cn, cv, u, d, mat, Ndof_u,
                                 free_u, dir_u, hu)
        # --- history update (monotone) ---
        hist = update_history(nx, cn, cv, u.reshape(N_cap, 2), hist, mat)
        # --- d-solve (linear given history) ---
        mvd, diagd, rhsd = assemble_phasefield(nx, cn, cv, hist, mat, Ndof_d)
        d = linear_newton(mvd, rhsd, diagd, free_d, dir_d,
                          hd[0], hd[1], hd[2], hd[3])
        # FEniCSx reference default: history-field irreversibility, without the
        # optional --irreversible nodal lower bound.
        d = jnp.clip(d, 0., 1.)
        # re-project hanging damage so the mesh stays consistent after the clip
        real = hid >= 0
        d = d.at[jnp.where(real, hid, 0)].set(
            jnp.where(real, 0.5 * (d[jnp.where(real, ma, 0)] + d[jnp.where(real, mb, 0)]), d[jnp.where(real, hid, 0)]))
        return u, d, hist

    # alternating minimization until the staggered increment < MS_TOL (capped at
    # MAX_STAGGER) -- same convergence criterion as the staggered while_loop of the uniform-mesh solver.
    def _cond(st):
        u, d, hist, err, it = st
        return (it < MAX_STAGGER) & (err > MS_TOL)

    def _body(st):
        u, d, hist, _, it = st
        u2, d2, hist2 = one_pass(u, d, hist)
        eu = jnp.linalg.norm(u2 - u) / (jnp.linalg.norm(u2) + 1e-30)
        ed = jnp.linalg.norm(d2 - d) / (jnp.linalg.norm(d2) + 1e-30)
        return (u2, d2, hist2, jnp.maximum(eu, ed), it + 1)

    u, d, hist, _, _ = jax.lax.while_loop(
        _cond, _body, (u, d, hist, jnp.array(1e30), jnp.array(0)))

    # top-face reaction from the same split-stress internal residual
    f, _, _ = assemble_elasticity(
        nx, cn, cv, u.reshape(N_cap, 2), d[:, None], mat, Ndof_u)
    force_y = jnp.sum(jnp.where(top, f[1::2], 0.0))
    return u, d, hist, force_y, n_flag


def cell_mean_damage(state, d):
    """Mean nodal damage over each cell's four corners (the refinement estimator)."""
    return jnp.mean(d[state.cell_nodes], axis=1)


def make_step(mat, lat, hang_cap, Ly, new_node_cap, target_hmin):
    """Build the jitted per-load-step function used as the scan body."""

    def step(carry, disp):
        state, u, d, d_old, hist = carry
        N_cap = state.node_key.shape[0]

        # (1) solve on the current mesh
        u, d, hist, _, nf1 = staggered_solve(state, u, d, d_old, hist, disp,
                                             mat, lat, hang_cap, Ly)

        # (2) adapt: mark high-damage cells and refine + balance, carrying fields
        def adapt(_, carry):
            state, u, d, d_old, hist = carry
            phi = cell_mean_damage(state, d)
            h_cell = lat.Lx / (lat.NX0 * (2.0 ** state.cell_lv))   # physical cell size
            marker = (phi > PHI_TH) & (h_cell > target_hmin) & \
                     (state.cell_lv < lat.LMAX) & state.cell_valid & (disp > 0.0)
            state, nf, cf = refine(state, marker, lat,
                                   node_fields=(u.reshape(N_cap, 2), d[:, None], d_old[:, None]),
                                   cell_fields=(hist,), new_node_cap=new_node_cap)
            state, nf, cf = balance(state, lat, nf, cf,
                                    max_sweeps=BALANCE_SWEEPS, new_node_cap=new_node_cap)
            u = nf[0].reshape(-1)
            d = nf[1][:, 0]
            d_old = nf[2][:, 0]
            hist = cf[0]
            # FEniCSx resolves the same load again after each remesh before it
            # computes the next set of markers.
            u, d, hist, _, _ = staggered_solve(
                state, u, d, d_old, hist, disp, mat, lat, hang_cap, Ly)
            return state, u, d, d_old, hist

        state, u, d, d_old, hist = jax.lax.fori_loop(
            0, ADAPT_SWEEPS, adapt, (state, u, d, d_old, hist))

        # Peak append cursor BEFORE reclaiming dead parents -- this is the true
        # buffer high-water mark for the step.  The overflow guard MUST test this,
        # because compact() below resets n_cells to the live count and would
        # otherwise hide an intra-step overflow (clamped writes -> dropped cells).
        peak_ncells = state.n_cells

        # reclaim dead-parent slots (fixed-shape reorder -> no recompile) so the
        # buffer occupancy stays ~ live cells instead of the cumulative count
        state, (hist,) = compact(state, (hist,))

        # (3) solve again on the refined mesh; record the reaction force
        u, d, hist, force_y, nf2 = staggered_solve(state, u, d, d_old, hist, disp,
                                                   mat, lat, hang_cap, Ly)
        d_old = d
        n_live = jnp.sum(state.cell_valid)
        peak_hang = jnp.maximum(nf1, nf2)
        return (state, u, d, d_old, hist), (force_y, n_live, peak_ncells, peak_hang)

    return step


def repad(state, u, d, d_old, hist, C_new, N_new):
    """Copy the mesh + fields into LARGER fixed-capacity buffers (bucket growth).
    New slots are inert padding: INVALID_KEY nodes, dead cells, zero fields."""
    C_old = state.cell_valid.shape[0]
    N_old = state.node_key.shape[0]
    dC, dN = C_new - C_old, N_new - N_old
    st = MeshState(
        node_key=jnp.pad(state.node_key, (0, dN), constant_values=int(INVALID_KEY)),
        node_xy=jnp.pad(state.node_xy, ((0, dN), (0, 0))),
        n_nodes=state.n_nodes,
        cell_nodes=jnp.pad(state.cell_nodes, ((0, dC), (0, 0))),
        cell_lv=jnp.pad(state.cell_lv, (0, dC)),
        cell_ll=jnp.pad(state.cell_ll, ((0, dC), (0, 0))),
        cell_valid=jnp.pad(state.cell_valid, (0, dC), constant_values=False),
        n_cells=state.n_cells)
    u = jnp.pad(u, (0, 2 * dN)); d = jnp.pad(d, (0, dN))
    d_old = jnp.pad(d_old, (0, dN)); hist = jnp.pad(hist, ((0, dC), (0, 0)))
    return st, u, d, d_old, hist


def run_bucketed(mat, lat, disps, C0, N0, H0, Ly, y_crack, x_tip, target_hmin=0.0,
                 seg_len=25, grow=1.3, hi=0.75, max_retry=40):
    """Like run(), but GROWS the capacity on demand (bucketing).

    Runs the load steps in segments; between segments, if the append cursor
    passes ``hi * C_cap`` (or node cursor ``hi * N_cap``), re-pads the buffers to
    ``grow x`` larger and rebuilds the step -- one recompile per growth.  Start
    small; it enlarges itself.  Choose ``seg_len`` small enough (and C0/hi with
    enough headroom) that one segment cannot add more than ``(1 - hi)`` of the
    capacity -- the run raises if a segment overflows.
    """
    disps = onp.asarray(disps)
    C_cap, N_cap, hang_cap = int(C0), int(N0), int(H0)
    state = initial_mesh(lat, C_cap, N_cap, y_crack=y_crack, x_tip=x_tip)
    u = jnp.zeros(N_cap * 2); d = jnp.zeros(N_cap)
    d_old = jnp.zeros(N_cap); hist = jnp.zeros((C_cap, 4))

    def _grow(C, N, H):
        return (max(int(onp.ceil(C * grow)), C + 1),
                max(int(onp.ceil(N * grow)), N + 1),
                max(int(onp.ceil(H * grow)), H + 1))

    step = make_step(mat, lat, hang_cap, Ly, 5 * C_cap, target_hmin)
    forces_all, nlive_all = [], []
    i = 0
    while i < len(disps):
        seg = jnp.asarray(disps[i:i + seg_len])
        snap = (state, u, d, d_old, hist)
        while int(snap[0].n_cells) > hi * C_cap or int(snap[0].n_nodes) > hi * N_cap:
            C_cap, N_cap, hang_cap = _grow(C_cap, N_cap, hang_cap)
            snap = repad(*snap, C_cap, N_cap)
            step = make_step(mat, lat, hang_cap, Ly, 5 * C_cap, target_hmin)
            print(f"[bucket] pre-grow -> C_cap={C_cap} (recompile)")
        for _ in range(max_retry):
            carry, (f, nl, peak, peakh) = jax.lax.scan(step, snap, seg)
            # test the pre-compaction PEAK cursor + hanging count (both hidden after compaction)
            if int(jnp.max(peak)) <= C_cap and int(carry[0].n_nodes) <= N_cap and int(jnp.max(peakh)) <= hang_cap:
                break
            C_cap, N_cap, hang_cap = _grow(C_cap, N_cap, hang_cap)
            snap = repad(*snap, C_cap, N_cap)
            step = make_step(mat, lat, hang_cap, Ly, 5 * C_cap, target_hmin)
            print(f"[bucket] segment overflowed -> grow to C_cap={C_cap} & retry")
        else:
            raise RuntimeError(f"segment still overflowing after {max_retry} growths")
        state, u, d, d_old, hist = carry
        forces_all.append(onp.asarray(f)); nlive_all.append(onp.asarray(nl))
        i += seg_len
    return (state, u, d, d_old, hist), onp.concatenate(forces_all), onp.concatenate(nlive_all)


def run(mat, lat, disps, C_cap, N_cap, hang_cap, Ly, y_crack, x_tip,
        new_node_cap=None, target_hmin=0.0):
    """Drive the whole simulation as one scan and return forces + final state."""
    if new_node_cap is None:
        new_node_cap = 5 * C_cap
    state = initial_mesh(lat, C_cap, N_cap, y_crack=y_crack, x_tip=x_tip)

    u = jnp.zeros(N_cap * 2)
    d = jnp.zeros(N_cap)
    d_old = jnp.zeros(N_cap)
    hist = jnp.zeros((C_cap, 4))

    step = make_step(mat, lat, hang_cap, Ly, new_node_cap, target_hmin)
    carry0 = (state, u, d, d_old, hist)
    carry, (forces, n_live, peak, peakh) = jax.lax.scan(step, carry0, disps)
    peak_max = int(jnp.max(peak))
    if int(jnp.max(peakh)) > hang_cap:
        print(f"  !!! HANG OVERFLOW: peak hanging {int(jnp.max(peakh))} > hang_cap {hang_cap} -- raise hang_cap.")
    if peak_max > C_cap:
        print(f"  !!! OVERFLOW: peak append cursor {peak_max} > C_cap {C_cap} "
              f"-- mesh CORRUPTED (holes); raise C_cap.")
    return carry, onp.asarray(forces), onp.asarray(n_live)


if __name__ == "__main__":
    from run_demo import main
    main()
