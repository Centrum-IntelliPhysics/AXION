"""Verification of the automatically differentiated derivatives (paper Section 4.1.4).

Runs on top of the two-dimensional benchmark ``01_sent_2d/sent_2d_uniform.py``
(same mesh, element kernels, boundary conditions and staggered scheme) and
produces every number of that section:

  taylor   Taylor remainder test of the element tangents (displacement residual
           and damage residual) at the converged state at u_bar.
  newton   Residual history of Newton's method on the displacement subproblem
           of the last increment (tight CG tolerance).
  sens     dF/dG_c and dF/dE_0 at u_bar by forward-mode differentiation of the
           loading history against central differences (AT2 only).
  elastic  PF-CZM below the damage threshold: dF/dE_0 against F/E_0 (CZM only).

The sensitivity tests use parametrized copies of the two subproblems, in which
E_0 and G_c enter as data at the quadrature points. Each nonlinear subsolve is
differentiated by the implicit function theorem: the Newton iterates carry no
tangent, and one linearized solve at the converged state propagates the
tangents. Tangents flow through the staggered while_loop and the load-step
scan in forward mode. The central differences rerun exactly the same primal
algorithm.

Environment variables
  PFF_MODEL        AT2 (default) | AT1 | CZM
  PFF_L            regularization length in mm (default 0.02)
  PFF_NX, PFF_NY   mesh                      (default 150 x 150)
  VER_TESTS        comma list                (default: all that apply)
  VER_U_BAR        displacement u_bar [mm]   (default 5e-3)
  VER_DU           load increment [mm]       (default 1e-4)
  VER_U_EL         u_bar of the PF-CZM elastic check [mm] (default 1e-4)
  VER_LIN_TOL      relative CG tolerance     (default 1e-10)
  VER_NL_TOL       Newton residual tolerance (default 1e-8)
  VER_MS_TOL       staggered tolerance       (default 1e-10)
  VER_MAX_STAGGER  staggered iteration cap   (default 500)
  VER_TAG          suffix of the output files (default: lin<VER_LIN_TOL>)
  VER_OUTPUT_DIR   output folder (default ./output)

Units: N, mm, MPa.
"""
import os
import sys
import json
import time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
OUT = os.environ.get('VER_OUTPUT_DIR', os.path.join(HERE, 'output'))
os.makedirs(OUT, exist_ok=True)

# The verification uses l = 0.02 mm on a 150 x 150 mesh (the benchmark itself
# uses l = 0.03 mm on 100 x 100).
os.environ.setdefault('PFF_L', '0.02')
os.environ.setdefault('PFF_OUTPUT_DIR', OUT)

import jax
import jax.numpy as np
import numpy as onp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import axion
from axion import Mesh, isotropic_function, negative_part, positive_part

_spec = importlib.util.spec_from_file_location(
    'sent_2d_uniform', os.path.join(HERE, '..', '01_sent_2d', 'sent_2d_uniform.py'))
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

MODEL = bm.MODEL
nu, l, f_t, K_RES = bm.nu, bm.l, bm.f_t, bm.K_RES
P0 = onp.array([bm.E, bm.G_c])            # nominal parameters (E_0, G_c)
PNAMES = ('E0', 'Gc')

NX = int(os.environ.get('PFF_NX', '150'))
NY = int(os.environ.get('PFF_NY', '150'))
U_BAR = float(os.environ.get('VER_U_BAR', '5e-3'))
DU = float(os.environ.get('VER_DU', '1e-4'))
U_EL = float(os.environ.get('VER_U_EL', '1e-4'))
LIN_TOL = float(os.environ.get('VER_LIN_TOL', '1e-10'))
LIN_ATOL = float(os.environ.get('VER_LIN_ATOL', '1e-13'))
LIN_MAXITER = int(os.environ.get('VER_LIN_MAXITER', '20000'))
NL_TOL = float(os.environ.get('VER_NL_TOL', '1e-8'))
MS_TOL = float(os.environ.get('VER_MS_TOL', '1e-10'))
MAX_STAGGER = int(os.environ.get('VER_MAX_STAGGER', '500'))
NUM_NEWTON = 30
_default_tests = {'AT2': 'taylor,newton,sens', 'AT1': 'taylor,newton',
                  'CZM': 'taylor,newton,elastic'}[MODEL]
TESTS = [t.strip() for t in os.environ.get('VER_TESTS', _default_tests).split(',') if t.strip()]
TAG = os.environ.get('VER_TAG', f'lin{LIN_TOL:.0e}')

sg = jax.lax.stop_gradient


# =============================================================================
# Parametrized constitutive functions: p = (E_0, G_c)
# =============================================================================
def degradation_p(d, E, Gc):
    w = (1. - d) ** 2
    if MODEL in ('AT1', 'AT2'):
        return w
    a1 = 4. / np.pi * (E * Gc / f_t ** 2) / l
    return w / (w + a1 * d - 0.5 * a1 * d ** 2)


_degradation_prime_p = jax.grad(degradation_p)
_crack_geo_prime = jax.grad(bm.crack_geo)
# The degradation g is an explicit argument of the spectral function, so its
# tangent is propagated.
_split = isotropic_function(lambda lam, g: g * positive_part(lam) + negative_part(lam))


def lame(E):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2. * (1. + nu))


def stress_p(grad_u, d, E, Gc):
    lmbda, mu = lame(E)
    g = degradation_p(d[0], E, Gc) + K_RES
    eps = 0.5 * (grad_u + grad_u.T)
    tr = np.trace(eps)
    return (lmbda * (g * positive_part(tr) + negative_part(tr)) * np.eye(2)
            + 2. * mu * _split(eps, g))


def psi_plus_p(grad_u, E):
    lmbda, mu = lame(E)
    eps = 0.5 * (grad_u + grad_u.T)
    evals, _ = np.linalg.eigh(eps)
    return lmbda / 2. * positive_part(np.trace(eps)) ** 2 + mu * np.sum(positive_part(evals) ** 2)


class DisplacementP(bm.Displacement):
    """theta = (d, E_0, G_c) at the quadrature points."""
    def flux(self):
        return stress_p


class DamageP(bm.Damage):
    """theta = (history, E_0, G_c) at the quadrature points."""
    def flux(self):
        def fn(grad_d, history, E, Gc):
            return 2. * Gc * l / bm.c_alpha() * grad_d
        return fn

    def source(self):
        def fn(d, x, history, E, Gc):
            dd = d[0]
            return np.array([Gc / (bm.c_alpha() * l) * _crack_geo_prime(dd)
                             + _degradation_prime_p(dd, E, Gc) * history])
        return fn


# =============================================================================
# Differentiable loading history
# =============================================================================
def cg_solve(matvec, inv_jacobi, V, rhs, fixed, lin_tol):
    inv = inv_jacobi(V, fixed)
    x, _ = jax.scipy.sparse.linalg.cg(lambda d: matvec(V, d, fixed), rhs,
                                      tol=lin_tol, atol=LIN_ATOL, maxiter=LIN_MAXITER,
                                      M=lambda r: r * inv)
    return x


def get_implicit_newton(field, lin_tol):
    """Newton + PCG whose result is differentiated by the implicit function theorem."""
    matvec, inv_jacobi = axion.make_operator(field)

    def solve(x0, theta, fixed):
        x0c, theta_c = sg(x0), [sg(v) for v in theta]

        def cond(state):
            x, R, V, it = state
            return (it < NUM_NEWTON) & (np.linalg.norm(R) > NL_TOL)

        def body(state):
            x, R, V, it = state
            x_new = x + cg_solve(matvec, inv_jacobi, V, -R, fixed, lin_tol)
            res, V_new = axion.assemble(field, x_new, theta_c)
            return (x_new, res.at[fixed].set(0.0), V_new, it + 1)

        res0, V0 = axion.assemble(field, x0c, theta_c)
        x_s, R_s, V_s, _ = jax.lax.while_loop(cond, body, (x0c, res0.at[fixed].set(0.0), V0, 0))
        x_s, V_s = sg(x_s), sg(V_s)
        # One linearized solve at the converged state: the primal correction is
        # O(NL_TOL); the tangent is  -K^{-1} (dR/dp + dR/d(coupled field)).
        R = field.residual(x_s, theta).at[fixed].set(0.0)
        return x_s + cg_solve(matvec, inv_jacobi, V_s, -R, fixed, lin_tol), np.linalg.norm(R_s)

    return solve


def get_projected_jacobi(field, max_it=300, tol=1e-7, relax=1.0):
    """Projected Jacobi iteration of the benchmark (AT1 / PF-CZM), parametrized residual."""
    def solve(y0, theta):
        D = axion.abs_diagonal(field, sg(y0), [sg(v) for v in theta])

        def cond(state):
            y, y_prev, it = state
            return (it < max_it) & (np.linalg.norm(y - y_prev) / (np.linalg.norm(y) + 1e-30) > tol)

        def body(state):
            y, _, it = state
            R = field.residual(y, theta)
            return (np.clip(y - relax * R / D, 0., 1.), y, it + 1)

        y0c = np.clip(y0, 0., 1.)
        return jax.lax.while_loop(cond, body, (y0c, y0c + 1.0, 0))[0]

    return solve


def get_history_runner(field_u, field_d, lin_tol):
    """run(p, disps, carry0) -> (final carry, carry before the last step, per-step diagnostics)."""
    Nn = field_u.num_nodes
    ones_q = np.ones((field_u.num_cells, field_u.num_quads))
    fixed = field_u.dirichlet_dofs
    is_top_dof = field_u.dirichlet_mask(bm.TOP_BC)
    no_dofs = np.array([], dtype=np.int32)
    top_faces = field_u.face_set(bm.top)
    top_cell_nodes = field_u.cells[np.asarray(top_faces.cell_ids)]
    newton_u = get_implicit_newton(field_u, lin_tol)
    if MODEL == 'AT2':
        newton_d = get_implicit_newton(field_d, lin_tol)
    else:
        solve_d = get_projected_jacobi(field_d)

    def cell_psi(su, E):
        grad_center = np.mean(field_u.grad_at_quad(su), axis=1)        # one value per cell
        return jax.vmap(lambda g: psi_plus_p(g, E))(grad_center)

    def reaction(su, sd, p):
        u_gf = np.einsum('sak,sqad->sqkd', su[top_cell_nodes], top_faces.shape_grads)
        d_f = np.einsum('sqa,sak->sqk', top_faces.shape_vals, sd[top_cell_nodes])
        n = u_gf.shape[0] * u_gf.shape[1]
        sig = jax.vmap(stress_p)(u_gf.reshape(n, 2, 2), d_f.reshape(n, 1),
                                 p[0] * np.ones(n), p[1] * np.ones(n))
        t = sig.reshape(u_gf.shape) @ np.array([0., 1.])
        return np.sum(t * top_faces.dS[:, :, None], axis=(0, 1))[1]

    def step(carry, disp, p):
        su0, sd0, h0 = carry[:3]
        E_q, Gc_q = p[0] * ones_q, p[1] * ones_q
        fixed_vals = np.where(is_top_dof, disp, 0.0)

        def cond(state):
            return (state[4] < MAX_STAGGER) & (state[3] > MS_TOL)

        def body(state):
            su, sd, hist, err, s, _ = state
            d_quad = field_u.to_quad(sd)
            u0 = su.reshape(-1).at[fixed].set(fixed_vals)
            u_new, rn = newton_u(u0, [d_quad, E_q, Gc_q], fixed)
            su_new = u_new.reshape(Nn, 2)
            h_c = np.maximum(cell_psi(su_new, p[0]), hist[:, 0])
            hist_new = np.broadcast_to(h_c[:, None], hist.shape)
            theta_d = [hist_new, E_q, Gc_q]
            if MODEL == 'AT2':
                d_raw, _ = newton_d(sd.reshape(-1), theta_d, no_dofs)
                n_low = np.sum(d_raw < 0.)
                sd_new = np.clip(d_raw, 0., 1.).reshape(Nn, 1)
            else:
                sd_new = solve_d(sd.reshape(-1), theta_d).reshape(Nn, 1)
                n_low = np.sum(sd_new <= 0.)
            eu = np.linalg.norm(su_new - su) / (np.linalg.norm(su_new) + 1e-30)
            ed = np.linalg.norm(sd_new - sd) / (np.linalg.norm(sd_new) + 1e-30)
            return (su_new, sd_new, hist_new, np.maximum(eu, ed), s + 1, (rn, n_low))

        init = (su0, sd0, h0, np.array(1e30), np.array(0), (np.array(0.), np.array(0)))
        su, sd, hist, err, s, (rn, n_low) = jax.lax.while_loop(cond, body, init)
        # H is the running maximum over the staggered iterates, so it can exceed
        # psi+ of the converged displacement; report by how much.
        psi = cell_psi(su, p[0])
        gap = (hist[:, 0] - psi) / np.max(psi)
        diag = dict(F=reaction(su, sd, p), stagger=s, err=err, newton_res=rn,
                    max_d=np.max(sd), n_upper=np.sum(sd >= 1.), n_lower=n_low,
                    n_hist_lag=np.sum(gap > 1e-8), hist_gap=np.max(gap),
                    n_unload=np.sum(psi < h0[:, 0]))      # H of the previous increment still active
        return (su, sd, hist, su0, sd0, h0), diag

    @jax.jit
    def run(p, disps, carry0):
        return jax.lax.scan(lambda c, x: step(c, x, p), carry0, disps)

    return run, fixed, is_top_dof


# =============================================================================
# Tests
# =============================================================================
def orders(r):
    r = onp.asarray(r)
    with onp.errstate(divide='ignore', invalid='ignore'):
        return onp.log10(r[:-1] / r[1:])


def taylor_test(res_fn, V, rows, cols, z, dz, mask, hs, smooth_mask_fn=None):
    """r0, r1 of the Taylor test. The spectral split makes the displacement
    residual only piecewise smooth, so if smooth_mask_fn is given, r1 is also
    evaluated on the unknowns whose elements stay on the same tension/compression
    branch between z and z + h dz (r1s), with the fraction of unknowns kept."""
    Kdz = np.zeros_like(z).at[rows].add(V * dz[cols]) * mask
    R0 = res_fn(z) * mask
    out = dict(h=list(hs), r0=[], r1=[], r1s=[], kept=[])
    for h in hs:
        dR = res_fn(z + h * dz) * mask - R0
        out['r0'].append(float(np.linalg.norm(dR)))
        out['r1'].append(float(np.linalg.norm(dR - h * Kdz)))
        if smooth_mask_fn is not None:
            m = smooth_mask_fn(z, z + h * dz)
            out['r1s'].append(float(np.linalg.norm((dR - h * Kdz) * m)))
            out['kept'].append(float(np.sum(m * mask) / np.sum(mask)))
    for k in ('r0', 'r1', 'r1s'):
        out['order_' + k] = orders(out[k]).tolist() if out[k] else []
    return out


def print_taylor(name, t):
    print(f"\n--- Taylor remainder test: {name} ---")
    head = f"{'h':>8} {'r0':>12} {'order':>6} {'r1':>12} {'order':>6}"
    print(head + (f" {'r1 (same branch)':>17} {'order':>6} {'DoFs kept':>10}" if t['r1s'] else ""))
    od = lambda k, i: f"{t['order_' + k][i - 1]:6.2f}" if i else "     -"
    for i, h in enumerate(t['h']):
        line = f"{h:8.0e} {t['r0'][i]:12.4e} {od('r0', i)} {t['r1'][i]:12.4e} {od('r1', i)}"
        if t['r1s']:
            line += f" {t['r1s'][i]:17.4e} {od('r1s', i)} {t['kept'][i]:10.4f}"
        print(line)


def build_fields(pts, cells):
    """Parametrized fields and the benchmark's own fields on the same mesh."""
    mesh = Mesh(pts, cells, 'QUAD4')
    par_d = DamageP(mesh, ncomp=1)
    par_u = DisplacementP(mesh, ncomp=2, dirichlet=bm.dirichlet_u)
    _, bm_d, bm_u = bm.build_fields(pts, cells)
    return par_u, par_d, bm_u, bm_d


def main():
    t0 = time.time()
    results = dict(model=MODEL, NX=NX, NY=NY, l=l, u_bar=U_BAR, du=DU, lin_tol=LIN_TOL,
                   nl_tol=NL_TOL, ms_tol=MS_TOL, max_stagger=MAX_STAGGER, E0=P0[0], Gc=P0[1])
    n_steps = int(round(U_BAR / DU))
    disps = np.arange(1, n_steps + 1) * DU

    pts, cells = bm.rectangle_mesh(bm.Lx, bm.Ly, NX, NY)
    pts, cells = bm.insert_crack(pts, cells, bm.y_crack, bm.x_tip)
    par_u, par_d, bm_u, bm_d = build_fields(pts, cells)
    Nn = par_u.num_nodes
    n_cells, n_quads = par_u.num_cells, par_u.num_quads
    ones_q = np.ones((n_cells, n_quads))
    p0 = np.array(P0)
    print(f"Model {MODEL} | {NX}x{NY} QUAD4, {3 * Nn} DoFs, l = {l} | u_bar={U_BAR:.3e} in "
          f"{n_steps} increments of {DU:.1e} | CG tol {LIN_TOL:.0e}, Newton tol {NL_TOL:.0e}, "
          f"staggered tol {MS_TOL:.0e}")
    print(f"Tests: {TESTS}")

    run, fixed, is_top_dof = get_history_runner(par_u, par_d, LIN_TOL)
    carry0 = (np.zeros((Nn, 2)), np.zeros((Nn, 1)), np.zeros((n_cells, n_quads))) * 2

    # ---- nominal loading history ----
    t1 = time.time()
    carry, diag = run(p0, disps, carry0)
    su, sd, hist, su_prev, sd_prev, _ = carry
    diag = {k: onp.asarray(v) for k, v in diag.items()}
    F0 = float(diag['F'][-1])
    print(f"\nNominal history: {time.time() - t1:.1f} s (incl. compilation)")
    print(f"  F(u_bar) = {F0:.10e} N/mm, max d = {diag['max_d'][-1]:.6f}")
    print(f"  staggered iterations per increment: max {diag['stagger'].max()}, last {diag['stagger'][-1]} "
          f"(cap {MAX_STAGGER}); max final staggered error {diag['err'].max():.2e}")
    print(f"  max Newton residual at exit {diag['newton_res'].max():.2e}")
    print(f"  differentiability: nodes with d>=1: {diag['n_upper'].max()}, "
          f"nodes at the lower bound (last increment): {diag['n_lower'][-1]}, "
          f"cells with H - psi+ > 1e-8 max(psi+) (max over increments): {diag['n_hist_lag'].max()} "
          f"of {n_cells}, max (H - psi+)/max(psi+) = {diag['hist_gap'].max():.2e}, "
          f"cells with psi+ below H of the previous increment: {diag['n_unload'].max()}")
    results.update(F=F0, max_d=float(diag['max_d'][-1]), stagger_max=int(diag['stagger'].max()),
                   n_upper=int(diag['n_upper'].max()), n_lower_last=int(diag['n_lower'][-1]),
                   n_hist_lag=int(diag['n_hist_lag'].max()), hist_gap=float(diag['hist_gap'].max()),
                   n_unload=int(diag['n_unload'].max()), disps=onp.asarray(disps).tolist(),
                   forces=diag['F'].tolist())

    # ---- consistency of the parametrized subproblems with the benchmark's ----
    d_quad = par_u.to_quad(sd)
    Ru_b, Vu_b = axion.assemble(bm_u, su.reshape(-1), [d_quad])
    Ru_p, Vu_p = axion.assemble(par_u, su.reshape(-1), [d_quad, P0[0] * ones_q, P0[1] * ones_q])
    _, Vd_b = axion.assemble(bm_d, sd.reshape(-1), [hist])
    # (R_d vanishes at the converged d, so compare it at a shifted state)
    Rd_b, Vd_s = axion.assemble(bm_d, sd.reshape(-1) + 0.1, [hist])
    Rd_p, Vd_p = axion.assemble(par_d, sd.reshape(-1) + 0.1, [hist, P0[0] * ones_q, P0[1] * ones_q])
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))
    print(f"  parametrized vs benchmark kernels (rel. diff): R_u {rel(Ru_p, Ru_b):.1e}, "
          f"K_u {rel(Vu_p, Vu_b):.1e}, R_d {rel(Rd_p, Rd_b):.1e}, K_d {rel(Vd_p, Vd_s):.1e}")

    free_u = np.ones(2 * Nn).at[fixed].set(0.0)
    rows_u, cols_u = np.asarray(bm_u.coo_rows), np.asarray(bm_u.coo_cols)
    rows_d, cols_d = np.asarray(bm_d.coo_rows), np.asarray(bm_d.coo_cols)

    # ---- (1) Taylor remainder tests, on the benchmark's own kernels ----
    if 'taylor' in TESTS:
        hs = [10. ** (-k) for k in range(1, 9)]
        key_u, key_d = jax.random.split(jax.random.PRNGKey(0))
        z = su.reshape(-1)
        dz = jax.random.normal(key_u, z.shape) * free_u
        dz = dz * np.max(np.abs(z)) / np.max(np.abs(dz))     # |dz|_inf = |z|_inf
        res_u = jax.jit(lambda x: bm_u.residual(x, [d_quad]))
        nodes_per_cell = bm_u.nodes_per_cell

        def branch(x):          # signs of (eigenvalues, trace) of the strain at the quadrature points
            grads = bm_u.grad_at_quad(x)
            eps = 0.5 * (grads + np.swapaxes(grads, -1, -2))
            return np.sign(np.concatenate([np.linalg.eigvalsh(eps),
                                           np.trace(eps, axis1=-2, axis2=-1)[..., None]], -1))

        @jax.jit
        def same_branch(xa, xb):
            flipped = np.any(branch(xa) != branch(xb), axis=(1, 2))                     # per cell
            node_bad = np.zeros(Nn).at[bm_u.cells.reshape(-1)].add(
                np.repeat(flipped, nodes_per_cell) * 1.0)
            return np.repeat(node_bad == 0., 2) * 1.0

        t = taylor_test(res_u, Vu_b, rows_u, cols_u, z, dz, free_u, hs, same_branch)
        print_taylor(f"displacement residual ({MODEL})", t)
        results['taylor_u'] = t

        zd = sd.reshape(-1)
        dzd = jax.random.normal(key_d, zd.shape)
        dzd = dzd / np.max(np.abs(dzd))                      # |dz|_inf = 1
        res_d = jax.jit(lambda x: bm_d.residual(x, [hist]))
        t = taylor_test(res_d, Vd_b, rows_d, cols_d, zd, dzd, np.ones_like(zd), hs)
        print_taylor(f"damage residual ({MODEL})", t)
        results['taylor_d'] = t

    # ---- (2) Newton convergence, displacement subproblem of the last increment ----
    if 'newton' in TESTS:
        matvec, inv_jacobi = axion.make_operator(bm_u)
        dq = par_u.to_quad(sd_prev)
        asm = jax.jit(lambda x: axion.assemble(bm_u, x, [dq]))
        u = su_prev.reshape(-1).at[fixed].set(np.where(is_top_dof, disps[-1], 0.0))
        hist_res = []
        for it in range(15):
            R, V = asm(u)
            R = R.at[fixed].set(0.0)
            hist_res.append(float(np.linalg.norm(R)))
            if hist_res[-1] < 1e-11 * hist_res[0] or (it > 1 and hist_res[-1] >= hist_res[-2]):
                break
            inv = inv_jacobi(V, fixed)
            du, _ = jax.scipy.sparse.linalg.cg(lambda d: matvec(V, d, fixed), -R, tol=1e-13,
                                               atol=0.0, maxiter=LIN_MAXITER, M=lambda r: r * inv)
            u = u + du
        print(f"\n--- Newton, displacement subproblem, increment {n_steps} (CG tol 1e-13) ---")
        for it, r in enumerate(hist_res):
            print(f"  it {it}: ||R|| = {r:.4e}   (relative {r / hist_res[0]:.4e})")
        results['newton'] = dict(increment=n_steps, residual=hist_res)

    # ---- (3) dF/dp: forward mode vs central differences ----
    if 'sens' in TESTS:
        F_of = lambda p: run(p, disps, carry0)[1]['F'][-1]
        jvp_fn = jax.jit(lambda p, t: jax.jvp(F_of, (p,), (t,)))
        deltas = [10. ** (-0.5 * k) for k in range(2, 15)]   # 1e-1 ... 1e-7, half decades
        sens = {}
        for i, name in enumerate(PNAMES):
            t1 = time.time()
            e = onp.zeros(2); e[i] = 1.0
            # Tangent seeded with p_i (derivative w.r.t. ln p_i), so that the tangent
            # linear systems have right-hand sides of the same magnitude as the
            # primal ones and see the same effective CG tolerance.
            F_ad, dF_ad = (float(x) for x in jvp_fn(p0, np.array(P0[i] * e)))
            dF_ad /= P0[i]
            t_ad = time.time() - t1
            rows = []
            for dl in deltas:
                Fp = float(F_of(np.array(P0 * (1. + dl * e))))
                Fm = float(F_of(np.array(P0 * (1. - dl * e))))
                dF_fd = (Fp - Fm) / (2. * P0[i] * dl)
                rows.append((dl, dF_fd, abs(dF_fd - dF_ad) / abs(dF_ad)))
            print(f"\n--- dF/d{name}: forward mode = {dF_ad:.10e}  (F = {F_ad:.10e}, {t_ad:.1f} s; "
                  f"normalized p/F dF/dp = {dF_ad * P0[i] / F_ad:.6f}) ---")
            print(f"{'Delta':>10} {'central diff':>18} {'rel. diff':>11} {'order':>6}")
            errs = [r[2] for r in rows]
            od = orders(errs) / 0.5
            for k, (dl, fd, er) in enumerate(rows):
                print(f"{dl:10.3e} {fd:18.10e} {er:11.3e} " + (f"{od[k-1]:6.2f}" if k else "     -"))
            kmin = int(onp.argmin(errs))
            # An isolated minimum can be a sign change of (FD - AD); the plateau
            # over 1e-6 <= Delta <= 1e-4 is the level set by the solver tolerances.
            plateau = float(onp.median([er for dl, _, er in rows if 0.99e-6 <= dl <= 1.01e-4]))
            print(f"  minimum relative difference {errs[kmin]:.3e} at Delta = {deltas[kmin]:.3e}; "
                  f"plateau (median, 1e-6 <= Delta <= 1e-4) {plateau:.3e}")
            sens[name] = dict(dF_ad=dF_ad, delta=deltas, dF_fd=[r[1] for r in rows], rel_diff=errs,
                              min_rel_diff=errs[kmin], delta_at_min=deltas[kmin], plateau=plateau,
                              time_ad=t_ad)
        results['sens'] = sens

    # ---- (4) PF-CZM below the damage threshold: dF/dE_0 = F/E_0 ----
    if 'elastic' in TESTS:
        n_el = max(int(round(U_EL / DU)), 1)
        disps_el = np.arange(1, n_el + 1) * (U_EL / n_el)
        _, dg = run(p0, disps_el, carry0)
        F_el_fn = lambda p: run(p, disps_el, carry0)[1]['F'][-1]
        F_el, dF_el = (float(x) for x in jax.jvp(F_el_fn, (p0,), (np.array([P0[0], 0.]),)))
        dF_el /= P0[0]
        err = abs(dF_el - F_el / P0[0]) / abs(F_el / P0[0])
        print(f"\n--- {MODEL}, elastic check at u = {U_EL:.2e} ({n_el} increments) ---")
        print(f"  max d = {float(onp.asarray(dg['max_d']).max()):.3e}  (must be 0)")
        print(f"  dF/dE0 forward mode = {dF_el:.12e},  F/E0 = {F_el / P0[0]:.12e},  relative error = {err:.3e}")
        results['elastic'] = dict(u=U_EL, max_d=float(onp.asarray(dg['max_d']).max()), F=F_el,
                                  dF_dE_ad=dF_el, F_over_E=F_el / P0[0], rel_err=err)

    with open(os.path.join(OUT, f'verify_{MODEL}_{TAG}.json'), 'w') as fh:
        json.dump(results, fh, indent=1, default=float)

    # ---- figure: (a) Newton residual, (b) AD vs central difference ----
    if 'newton' in results or 'sens' in results:
        plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 10,
                             'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))
        if 'newton' in results:
            r = results['newton']['residual']
            axs[0].semilogy(range(len(r)), r, 'k-o', ms=4)
            axs[0].set_xlabel('Newton iteration')
            axs[0].set_ylabel(r'$\|\mathbf{R}\|_2$')
        if 'sens' in results:
            for name, lab, st in (('Gc', r'$G_{\mathrm{f}}$', 'o-'), ('E0', r'$E_0$', 's--')):
                s = results['sens'][name]
                axs[1].loglog(s['delta'], s['rel_diff'], st, ms=4, label=lab)
            dl = onp.array([1e-3, 1e-1])
            axs[1].loglog(dl, 0.5 * results['sens']['Gc']['rel_diff'][0] * (dl / 1e-1) ** 2, 'k:', label='slope 2')
            axs[1].set_xlabel(r'relative step $\Delta$')
            axs[1].set_ylabel('relative difference')
            axs[1].legend(frameon=False)
        for ax, t in zip(axs, ('(a)', '(b)')):
            ax.set_title(t, loc='left')
            ax.grid(which='major', linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'verify_{MODEL}_{TAG}.png'), dpi=300)
        plt.close(fig)

    print(f"\nDone in {time.time() - t0:.1f} s. Results: {os.path.join(OUT, f'verify_{MODEL}_{TAG}.json')}")


if __name__ == '__main__':
    main()
