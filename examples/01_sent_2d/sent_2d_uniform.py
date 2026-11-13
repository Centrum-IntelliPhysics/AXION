"""Two-dimensional single-edge-notched tension on a uniform mesh (paper Section 4.1.1).

A unit square with a horizontal pre-crack from the left edge to the centre is
clamped at the bottom and pulled at the top under plane strain. The mesh is a
fixed QUAD4 grid and the pre-crack is a geometric slit (duplicated nodes), so
the mesh never changes: every kernel is compiled once and the whole loading
history runs as one program on the device,

    scan over load increments [ while staggered [ while Newton [ while CG ] ] ] .

Three phase-field models share one element definition and differ only in the
crack geometric function alpha(d) and the degradation function omega(d):

    AT2   damage subproblem linear and SPD      -> one Newton step with CG, then clip
    AT1   elastic threshold, bounded            -> projected Jacobi
    CZM   PF-CZM (Wu), indefinite tangent       -> projected Jacobi

The displacement subproblem uses the spectral tension-compression split and is
solved by Newton's method with Jacobi-preconditioned CG.

Units: N, mm, MPa.

Environment variables
    PFF_MODEL        AT2 (default) | AT1 | CZM
    PFF_L            regularization length in mm (default 0.03)
    PFF_NX, PFF_NY   cells per direction, even (default 100 x 100, h = l/3)
    PFF_NSTEPS       number of load increments (default 700)
    PFF_DU           displacement increment in mm (default 1e-5)
    PFF_MAX_STAGGER  cap on staggered iterations per increment (default 1000)
    PFF_MS_TOL       staggered tolerance (default 1e-5)
    PFF_OUTPUT_DIR   output folder (default ./output_uniform)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import jax
import jax.numpy as np
import numpy as onp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import AutoMinorLocator

import axion
from axion import DirichletBC, Field, Mesh, isotropic_function, negative_part, positive_part

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
output_dir = os.environ.get('PFF_OUTPUT_DIR', os.path.join(here, 'output_uniform'))
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Material and model parameters (Miehe et al. 2010)
# =============================================================================
MODEL = os.environ.get('PFF_MODEL', 'AT2')
assert MODEL in ('AT1', 'AT2', 'CZM'), f"PFF_MODEL must be AT1/AT2/CZM, got {MODEL}"

E = 210e3             # Young's modulus [MPa]
nu = 0.3              # Poisson's ratio
G_c = 2.7             # fracture energy [N/mm]
l = float(os.environ.get('PFF_L', '0.03'))   # regularization length [mm]
f_t = 2445.42         # tensile strength [MPa] (PF-CZM only)
mu = E / (2. * (1. + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

l_ch = E * G_c / (f_t ** 2)   # Irwin's characteristic length
K_RES = 1e-8                  # residual stiffness of fully broken material

# Geometry: unit square, horizontal pre-crack from the left edge to the centre.
Lx, Ly = 1.0, 1.0
y_crack, x_tip = 0.5, 0.5


def crack_geo(d):
    """Crack geometric function alpha(d)."""
    if MODEL == 'AT1':
        return d
    if MODEL == 'AT2':
        return d ** 2
    return 2. * d - d ** 2                      # PF-CZM


def degradation(d):
    """Degradation function omega(d)."""
    w = (1. - d) ** 2
    if MODEL in ('AT1', 'AT2'):
        return w
    a1 = 4. / np.pi * l_ch / l                  # PF-CZM, linear softening
    a2 = -0.5
    a3 = 0.0
    Q = a1 * d + a1 * a2 * d ** 2 + a1 * a2 * a3 * d ** 3
    return w / (w + Q)


def c_alpha():
    """Normalization constant c_alpha = 4 int_0^1 sqrt(alpha)."""
    return {'AT1': 8. / 3., 'AT2': 2., 'CZM': np.pi}[MODEL]


_degradation_prime = jax.grad(degradation)
_crack_geo_prime = jax.grad(crack_geo)


# =============================================================================
# Damage subproblem
# =============================================================================
class Damage(Field):
    """omega'(d) H + G_c / c_alpha ( alpha'(d) / l - 2 l lap d ) = 0,
    with the driving energy frozen at the history field H."""

    def flux(self):
        coeff = 2. * G_c * l / c_alpha()

        def fn(grad_d, history):
            return coeff * grad_d
        return fn

    def source(self):
        coeff = G_c / (c_alpha() * l)

        def fn(d, x, history):
            dd = d[0]
            return np.array([coeff * _crack_geo_prime(dd) + _degradation_prime(dd) * history])
        return fn


# =============================================================================
# Displacement subproblem
# =============================================================================
def strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)


def psi_plus(grad_u):
    """Tensile part of the undamaged strain energy density (spectral split)."""
    eps = strain(grad_u)
    eigen_vals, _ = np.linalg.eigh(eps)
    tr_plus = positive_part(np.trace(eps))
    return lmbda / 2. * tr_plus ** 2 + mu * np.sum(positive_part(eigen_vals) ** 2)


# deviatoric part of the split stress as an isotropic tensor function of the
# strain; g is the degradation at the quadrature point
_split_stress = isotropic_function(
    lambda lam, g: 2. * mu * (g * positive_part(lam) + negative_part(lam)))


def stress(grad_u, d):
    """sigma = (omega(d) + k_res) d psi+ / d eps + d psi- / d eps."""
    eps = strain(grad_u)
    g = degradation(d[0]) + K_RES
    tr = np.trace(eps)
    volumetric = lmbda * (g * positive_part(tr) + negative_part(tr)) * np.eye(2)
    return volumetric + _split_stress(eps, g)


class Displacement(Field):
    def flux(self):
        return stress

    def compute_history(self, sol_u, history_old):
        """H = max(H_old, psi+), with psi+ evaluated at the cell centre and
        stored as one value per cell (broadcast to the quadrature points)."""
        grad_center = np.mean(self.grad_at_quad(sol_u), axis=1)
        psi_center = jax.vmap(psi_plus)(grad_center)
        h_center = np.maximum(psi_center, history_old[:, 0])
        return np.broadcast_to(h_center[:, None], history_old.shape)

    def reaction_force(self, faces, sol_u, sol_d):
        """Resultant of sigma n over a set of element faces."""
        cell_nodes = self.cells[np.asarray(faces.cell_ids)]
        grad_u = np.einsum('sak,sqad->sqkd', sol_u[cell_nodes], faces.shape_grads)
        d_face = np.einsum('sqa,sak->sqk', faces.shape_vals, sol_d[cell_nodes])
        sigma = jax.vmap(jax.vmap(stress))(grad_u, d_face)
        return np.einsum('sqkd,sqd,sq->k', sigma, faces.normals, faces.dS)


# =============================================================================
# Boundary conditions: bottom clamped, top pulled in y
# =============================================================================
def bottom(point):
    return np.isclose(point[1], 0., atol=1e-6)


def top(point):
    return np.isclose(point[1], Ly, atol=1e-6)


# The top condition carries the unit value; it is scaled by the applied
# displacement inside the load loop.
dirichlet_u = [DirichletBC(bottom, 0, 0.), DirichletBC(bottom, 1, 0.), DirichletBC(top, 1, 1.)]
TOP_BC = 2


# =============================================================================
# Uniform structured QUAD4 mesh with a geometric slit
# =============================================================================
def rectangle_mesh(Lx, Ly, nx, ny):
    """QUAD4 grid whose rows are laid out in four blocks, with a row of nodes
    exactly on the crack line y = 0.5."""
    xs = onp.linspace(0., Lx, nx + 1)
    n_band = int(round(0.1 * ny))
    n_outer = (ny - 2 * n_band) // 2
    counts = (n_outer, n_band, n_band, ny - n_outer - 2 * n_band)
    bounds = (0.0, 0.4 - 1e-4, 0.5, 0.6 + 1e-4, Ly)
    ys = onp.concatenate([
        onp.linspace(bounds[k], bounds[k + 1], counts[k] + 1)[:-1]
        for k in range(4)
    ] + [onp.array([Ly])])
    nxp = len(xs)
    X, Y = onp.meshgrid(xs, ys, indexing="xy")
    points = onp.stack([X.ravel(), Y.ravel()], axis=1)

    def nid(i, j):
        return j * nxp + i

    cells = []
    for j in range(len(ys) - 1):
        for i in range(nxp - 1):
            cells.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)])
    return points, onp.array(cells, dtype=onp.int32)


def insert_crack(points, cells, y_crack, x_tip, tol=1e-9, crack_half_opening=1e-4):
    """Open a slit on {y = y_crack, x < x_tip} by duplicating the nodes on the
    line (tip excluded) and attaching the elements above it to the copies."""
    points = onp.asarray(points)
    cells = onp.asarray(cells).copy()
    on_line = onp.isclose(points[:, 1], y_crack, atol=1e-8) & (points[:, 0] < x_tip - tol)
    old_ids = onp.where(on_line)[0]
    if len(old_ids) == 0:
        raise ValueError("no nodes on the crack line; need a node row exactly at y = y_crack "
                         "(use an even NY)")
    cent_y = points[cells][:, :, 1].mean(axis=1)
    upper_cells = onp.where(cent_y > y_crack)[0]
    new_ids = onp.arange(len(points), len(points) + len(old_ids))
    remap = dict(zip(old_ids.tolist(), new_ids.tolist()))
    upper = points[old_ids].copy()
    opening = crack_half_opening * (1.0 - points[old_ids, 0] / x_tip)
    points[old_ids, 1] -= opening
    upper[:, 1] += opening
    points = onp.vstack([points, upper])
    for c in upper_cells:
        for j in range(cells.shape[1]):
            if cells[c, j] in remap:
                cells[c, j] = remap[cells[c, j]]
    return points, cells


def build_fields(points, cells):
    mesh = Mesh(points, cells, 'QUAD4')
    field_d = Damage(mesh, ncomp=1)
    field_u = Displacement(mesh, ncomp=2, dirichlet=dirichlet_u)
    return mesh, field_d, field_u


# =============================================================================
# Staggered solve of one load increment
# =============================================================================
def get_staggered_stepper(field_u, field_d, max_stagger=1000, ms_tol=1e-5):
    newton_u = axion.make_newton(field_u, max_iter=30, tol=1e-9, cg_tol=1e-8, cg_maxiter=5000)
    Nu, Nd = field_u.num_nodes, field_d.num_nodes

    if MODEL in ('CZM', 'AT1'):
        solve_d = axion.make_projected_jacobi(field_d, max_iter=300, tol=1e-7, relax=1.0)
    else:
        newton_d = axion.make_newton(field_d, max_iter=30, tol=1e-9, cg_tol=1e-8, cg_maxiter=5000)
    no_dofs = np.array([], dtype=np.int32)

    @jax.jit
    def stepper(sol_u, sol_d, history_old, fixed, fixed_vals):
        def cond(state):
            su, sd, hist, err, s = state
            return (s < max_stagger) & (err > ms_tol)

        def body(state):
            su, sd, hist, err, s = state
            # ---- displacement ----
            d_quad = field_u.to_quad(sd)
            u0 = su.reshape(-1).at[fixed].set(fixed_vals)
            R0, V0 = axion.assemble(field_u, u0, (d_quad,))
            su_new = newton_u(u0, R0.at[fixed].set(0.0), V0, (d_quad,), fixed).reshape(Nu, 2)
            # ---- history (updated after every displacement solve) ----
            hist_new = field_u.compute_history(su_new, hist)
            # ---- damage ----
            if MODEL in ('CZM', 'AT1'):
                D = axion.abs_diagonal(field_d, sd, (hist_new,))
                sd_new = solve_d(sd.reshape(-1), D, (hist_new,)).reshape(Nd, 1)
            else:
                sd0 = sd.reshape(-1)
                R0d, V0d = axion.assemble(field_d, sd0, (hist_new,))
                sd_new = newton_d(sd0, R0d, V0d, (hist_new,), no_dofs).reshape(Nd, 1)
                sd_new = np.clip(sd_new, 0., 1.)
            # ---- convergence of the staggered iteration ----
            eu = np.linalg.norm(su_new - su) / (np.linalg.norm(su_new) + 1e-30)
            ed = np.linalg.norm(sd_new - sd) / (np.linalg.norm(sd_new) + 1e-30)
            return (su_new, sd_new, hist_new, np.maximum(eu, ed), s + 1)

        init = (sol_u, sol_d, history_old, np.array(1e30), np.array(0))
        su, sd, hist, err, s = jax.lax.while_loop(cond, body, init)
        return su, sd, hist

    return stepper


# =============================================================================
# Load loop: one compiled scan over all increments
# =============================================================================
def get_loading_stepper(field_u, field_d, max_stagger=1000, ms_tol=1e-5):
    stepper = get_staggered_stepper(field_u, field_d, max_stagger, ms_tol)
    fixed = field_u.dirichlet_dofs
    is_top_dof = field_u.dirichlet_mask(TOP_BC)
    top_faces = field_u.face_set(top)

    @jax.jit
    def step_fn(carry, disp):
        sol_u, sol_d, history_old = carry
        fixed_vals = np.where(is_top_dof, disp, 0.0)
        sol_u_new, sol_d_new, history_new = stepper(sol_u, sol_d, history_old, fixed, fixed_vals)
        force = field_u.reaction_force(top_faces, sol_u_new, sol_d_new)
        return (sol_u_new, sol_d_new, history_new), force[1]

    return step_fn


def run_loading(field_u, field_d, disps, sol_u, sol_d, history_old,
                max_stagger=1000, ms_tol=1e-5):
    step_fn = get_loading_stepper(field_u, field_d, max_stagger=max_stagger, ms_tol=ms_tol)
    carry0 = (np.asarray(sol_u), np.asarray(sol_d), np.asarray(history_old))
    (sol_u, sol_d, history), forces = jax.lax.scan(step_fn, carry0, np.asarray(disps[1:]))
    return sol_u, sol_d, history, onp.asarray(forces)


# =============================================================================
# Plotting
# =============================================================================
def plot_quad_mesh(points, cells, fname, title, sol_d=None, show_crack=False):
    points = onp.asarray(points)
    cells = onp.asarray(cells)
    polys = points[cells].astype(float)
    fig, ax = plt.subplots(figsize=(7, 7))
    if sol_d is None:
        ax.add_collection(PolyCollection(polys, facecolors='#cfe2f3', edgecolors='k',
                                         linewidths=0.25))
    else:
        d_cell = onp.asarray(sol_d).reshape(-1)[cells].mean(axis=1)
        pc = PolyCollection(polys, array=d_cell, cmap='inferno', edgecolors='face', linewidths=0.0)
        pc.set_clim(0., 1.)
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('damage  d')
    if show_crack:
        ax.plot([0., x_tip], [y_crack, y_crack], 'r-', lw=2., label='pre-crack')
        ax.legend(loc='upper right', frameon=False)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================
def main():
    t_wall0 = time.time()

    # h = l/3 = 0.01 mm -> 100 x 100 cells. Both counts must be even so that
    # grid lines fall on the crack line y = 0.5 and on the crack tip x = 0.5.
    NX = int(os.environ.get('PFF_NX', '100'))
    NY = int(os.environ.get('PFF_NY', '100'))
    NSTEPS = int(os.environ.get('PFF_NSTEPS', '700'))
    DU = float(os.environ.get('PFF_DU', '1e-5'))
    MAX_STAGGER = int(os.environ.get('PFF_MAX_STAGGER', '1000'))
    MS_TOL = float(os.environ.get('PFF_MS_TOL', '1e-5'))

    if NX <= 0 or NY <= 0 or NX % 2 or NY % 2:
        raise ValueError(f'PFF_NX and PFF_NY must be positive and even, got NX={NX}, NY={NY}')
    if NSTEPS <= 0 or DU <= 0.0:
        raise ValueError(f'PFF_NSTEPS and PFF_DU must be positive, got {NSTEPS}, {DU}')

    disps = onp.arange(NSTEPS + 1, dtype=onp.float64) * DU
    hx, hy = Lx / NX, Ly / NY

    pts, cells = rectangle_mesh(Lx, Ly, NX, NY)
    n_before = len(pts)
    pts, cells = insert_crack(pts, cells, y_crack, x_tip)
    n_slit = len(pts) - n_before

    mesh, field_d, field_u = build_fields(pts, cells)

    sol_u = onp.zeros((len(pts), 2))
    sol_d = onp.zeros((len(pts), 1))
    history_old = onp.zeros((field_u.num_cells, field_u.num_quads))

    n_dofs = 3 * len(pts)        # two displacement and one damage unknown per node
    print(f"Axion | model = {MODEL}. Uniform fixed mesh: {n_dofs} DoFs, {len(pts)} nodes, "
          f"{len(cells)} cells (NX={NX}, NY={NY}, hx={hx:.4f}, hy={hy:.4f}, l={l:.4f}). "
          f"Loading: {NSTEPS} increments of {DU:.4e} mm, final displacement {disps[-1]:.4e} mm. "
          f"Pre-crack: {n_slit} duplicated nodes.")

    plot_quad_mesh(pts, cells, os.path.join(output_dir, f'mesh_start_{MODEL}.png'),
                   f'Mesh ({n_dofs} DoFs, hx={hx:.4f}, hy={hy:.4f})', show_crack=True)

    start = time.time()
    sol_u, sol_d, history, forces = run_loading(field_u, field_d, disps, sol_u, sol_d,
                                                history_old, max_stagger=MAX_STAGGER,
                                                ms_tol=MS_TOL)
    solve_time = time.time() - start
    wall_time = time.time() - t_wall0

    forces = onp.concatenate([[0.0], forces])
    onp.savez(os.path.join(output_dir, f'sol_{MODEL}.npz'), disps=disps, forces=forces)
    onp.savez(os.path.join(output_dir, f'final_{MODEL}.npz'),
              points=onp.asarray(pts), cells=onp.asarray(cells),
              sol_d=onp.asarray(sol_d).reshape(-1), disps=disps, forces=forces,
              n_cells=int(len(cells)), n_nodes=int(len(pts)), n_dofs=int(n_dofs),
              solve_time=float(solve_time), wall_time=float(wall_time),
              peak_load=float(onp.max(onp.abs(forces))), Lx=float(Lx), Ly=float(Ly))
    axion.write_vtu(os.path.join(output_dir, f'final_{MODEL}.vtu'), mesh,
                    point_data={'d': onp.asarray(sol_d), 'u': onp.asarray(sol_u)},
                    cell_data={'history': onp.mean(onp.asarray(history), axis=1)})

    peak_i = int(onp.abs(forces).argmax())
    summary = (
        f"=============== SUMMARY ({MODEL}, uniform mesh) ===============\n"
        f"  DoF              : {n_dofs}  ({len(cells)} cells, {len(pts)} nodes)\n"
        f"  load increments  : {NSTEPS} (du = {DU:.4e} mm)\n"
        f"  solve time [s]   : {solve_time:.3f}  ({solve_time / 60.:.2f} min, includes compilation)\n"
        f"  wall time [s]    : {wall_time:.3f}\n"
        f"  time / increment : {solve_time / NSTEPS:.3f} s\n"
        f"  peak load        : {abs(forces[peak_i]):.6f} N/mm at u = {disps[peak_i]:.6e} mm\n"
        f"===============================================================")
    print("\n" + summary)
    with open(os.path.join(output_dir, f'summary_{MODEL}.txt'), 'w') as fh:
        fh.write(summary + "\n")

    plot_quad_mesh(pts, cells, os.path.join(output_dir, f'solution_final_{MODEL}.png'),
                   f'Final damage ({n_dofs} DoFs, max d = {onp.max(onp.asarray(sol_d)):.2f})',
                   sol_d=sol_d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(disps, forces, 'b-', marker='o', ms=4, label=f'Axion ({MODEL}, uniform)')
    ax.set_xlabel('displacement [mm]')
    ax.set_ylabel('reaction force [N/mm]')
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', linewidth=0.6, alpha=0.6)
    ax.grid(which='minor', linewidth=0.3, alpha=0.3, linestyle=':')
    ax.legend(frameon=True, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'ForceVsDisp_{MODEL}.png'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
