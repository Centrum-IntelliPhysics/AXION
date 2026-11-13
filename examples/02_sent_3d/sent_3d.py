"""Three-dimensional single-edge-notched tension on a uniform mesh (paper Section 4.1.2).

A slab of side 1 mm with a through-thickness pre-crack at mid-height is clamped
at the bottom face and pulled at the top face. The domain is discretized with a
uniform grid of trilinear hexahedra (HEX8) and the pre-crack is a geometric
slit (duplicated nodes on the crack plane).

The hybrid formulation of Ambati et al. (2015) is used: the stress is degraded
isotropically, sigma = (omega(d) + k_res) d psi_0 / d eps, so the displacement
subproblem is linear at fixed damage, while the crack-driving history field is
built from the tensile energy psi+ of the spectral split. The principal
strains needed for psi+ are evaluated in closed form.

As in two dimensions the whole loading history is one compiled program,

    scan over load increments [ while staggered [ while Newton [ while CG ] ] ] .

Units: N, mm, MPa.

Environment variables
    PFF_MODEL                AT2 (default) | AT1 | CZM
    PFF_CRACK                slit (default: duplicated nodes) | phi (d = 1 on the crack plane)
    PFF_NX, PFF_NY, PFF_NZ   cells per direction (default 150 x 150 x 15; NX, NY even)
    PFF_NSTEPS               number of load increments (default 160)
    PFF_DU                   displacement increment in mm (default 5e-5)
    PFF_MAX_STAGGER          cap on staggered iterations per increment (default 8)
    PFF_MS_TOL               staggered tolerance (default 1e-3)
    PFF_OUTPUT_DIR           output folder (default ./output)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import h5py
import jax
import jax.numpy as np
import numpy as onp

import axion
from axion import DirichletBC, Field, Mesh, eigvalsh_3x3, positive_part

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
out_dir = os.environ.get('PFF_OUTPUT_DIR', os.path.join(HERE, 'output'))
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Material and model parameters (Miehe et al. 2010)
# -----------------------------------------------------------------------------
MODEL = os.environ.get('PFF_MODEL', 'AT2').upper()
assert MODEL in ('AT1', 'AT2', 'CZM'), f"PFF_MODEL must be AT1/AT2/CZM, got {MODEL}"

E = 210e3            # Young's modulus [MPa]
nu = 0.3             # Poisson's ratio
G_c = 2.7            # fracture energy [N/mm]
l = 0.0313           # regularization length [mm]
f_t = 2445.42        # tensile strength [MPa] (PF-CZM only)
mu = E / (2. * (1. + nu))
lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
l_ch = E * G_c / (f_t ** 2)
K_RES = 1e-8         # residual stiffness of fully broken material


def crack_geo(d):
    """Crack geometric function alpha(d)."""
    if MODEL == 'AT1':
        return d
    if MODEL == 'AT2':
        return d ** 2
    return 2. * d - d ** 2          # PF-CZM


def degradation(d):
    """Degradation function omega(d)."""
    w = (1. - d) ** 2
    if MODEL in ('AT1', 'AT2'):
        return w
    a1 = 4. / np.pi * l_ch / l       # PF-CZM, linear softening (a2 = -1/2, a3 = 0)
    Q = a1 * (d - 0.5 * d ** 2)
    return w / (w + Q)


def c_alpha():
    return {'AT1': 8. / 3., 'AT2': 2., 'CZM': np.pi}[MODEL]


_degradation_prime = jax.grad(degradation)
_crack_geo_prime = jax.grad(crack_geo)

# -----------------------------------------------------------------------------
# Geometry and mesh
# -----------------------------------------------------------------------------
Lx, Ly, Lz = 1.0, 1.0, 0.2
Nx = int(os.environ.get('PFF_NX', 150))
Ny = int(os.environ.get('PFF_NY', 150))
Nz = int(os.environ.get('PFF_NZ', 15))
if Nx <= 0 or Ny <= 0 or Nz <= 0:
    raise ValueError(f'PFF_NX/PFF_NY/PFF_NZ must be positive, got {Nx}/{Ny}/{Nz}')
if Nx % 2 or Ny % 2:
    raise ValueError(f'PFF_NX and PFF_NY must be even, got {Nx}/{Ny}')
hx, hy, hz = Lx / Nx, Ly / Ny, Lz / Nz
y_crack, x_tip = 0.5 * Ly, 0.5 * Lx     # pre-crack plane y = Ly/2, x < Lx/2

# Pre-crack:
#   'slit' (default)  duplicate the nodes on the crack plane and attach the
#                     elements above it to the copies: a true displacement
#                     discontinuity, with no condition on the damage
#   'phi'             prescribe d = 1 on the nodes of the crack plane
CRACK_TYPE = os.environ.get('PFF_CRACK', 'slit').lower()
assert CRACK_TYPE in ('slit', 'phi')


def insert_crack_3d(points, cells, y_crack, x_tip, tol=1e-9):
    """Open a slit on {y = y_crack, x < x_tip, all z} by duplicating the nodes
    of the plane (crack front excluded) and attaching the elements above the
    plane to the copies."""
    points = onp.asarray(points, dtype=float)
    cells = onp.asarray(cells).copy()
    on_plane = (onp.abs(points[:, 1] - y_crack) < 1e-8) & (points[:, 0] < x_tip - tol)
    old_ids = onp.where(on_plane)[0]
    if len(old_ids) == 0:
        raise ValueError("no nodes on the crack plane; need an even NY so that a "
                         "grid plane lies at y = y_crack")
    new_ids = onp.arange(len(points), len(points) + len(old_ids))
    remap = dict(zip(old_ids.tolist(), new_ids.tolist()))
    points = onp.vstack([points, points[old_ids]])
    cent_y = points[cells][:, :, 1].mean(axis=1)
    for c in onp.where(cent_y > y_crack)[0]:
        for j in range(cells.shape[1]):
            if cells[c, j] in remap:
                cells[c, j] = remap[cells[c, j]]
    return points, cells, len(new_ids)


# =============================================================================
# Damage subproblem
# =============================================================================
class Damage(Field):
    """omega'(d) H + G_c / c_alpha ( alpha'(d) / l - 2 l lap d ) = 0."""

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
# Displacement subproblem (hybrid formulation)
# =============================================================================
def stress(grad_u, d):
    """sigma = (omega(d) + k_res) (lambda tr(eps) I + 2 mu eps)."""
    eps = 0.5 * (grad_u + grad_u.T)
    g = degradation(d[0]) + K_RES
    return g * (lmbda * np.trace(eps) * np.eye(3) + 2. * mu * eps)


def psi_plus(grad_u):
    """Tensile part of the undamaged strain energy density (spectral split)."""
    eps = 0.5 * (grad_u + grad_u.T)
    eigvals = eigvalsh_3x3(eps)
    tr_plus = positive_part(np.trace(eps))
    return lmbda / 2. * tr_plus ** 2 + mu * np.sum(positive_part(eigvals) ** 2)


class Displacement(Field):
    def flux(self):
        return stress

    def compute_history(self, sol_u, history_old):
        """H = max(H_old, psi+) at every quadrature point."""
        psi = jax.vmap(jax.vmap(psi_plus))(self.grad_at_quad(sol_u))
        return np.maximum(psi, history_old)

    def reaction_force(self, faces, sol_u, sol_d):
        """Resultant of sigma n over a set of element faces."""
        cell_nodes = self.cells[np.asarray(faces.cell_ids)]
        grad_u = np.einsum('sak,sqad->sqkd', sol_u[cell_nodes], faces.shape_grads)
        d_face = np.einsum('sqa,sak->sqk', faces.shape_vals, sol_d[cell_nodes])
        sigma = jax.vmap(jax.vmap(stress))(grad_u, d_face)
        return np.einsum('sqkd,sqd,sq->k', sigma, faces.normals, faces.dS)


# =============================================================================
# Boundary conditions: bottom face clamped, top face pulled in +y
# =============================================================================
def bottom(point):
    return np.isclose(point[1], 0., atol=1e-6)


def top(point):
    return np.isclose(point[1], Ly, atol=1e-6)


# The top condition carries the unit value; it is scaled by the applied
# displacement inside the load loop.
dirichlet_u = [DirichletBC(bottom, 0, 0.), DirichletBC(bottom, 1, 0.),
               DirichletBC(bottom, 2, 0.), DirichletBC(top, 1, 1.)]
TOP_BC = 3


def crack_dofs(points):
    """Damage unknowns on the crack plane, for PFF_CRACK=phi."""
    cn = onp.where((onp.abs(points[:, 1] - y_crack) < 1e-6) & (points[:, 0] <= x_tip + 1e-6))[0]
    return (np.array(cn, dtype=np.int32), np.ones(len(cn)))


# =============================================================================
# Staggered solve of one load increment
# =============================================================================
def get_staggered_stepper(field_u, field_d, crack, max_stagger=8, ms_tol=1e-3):
    Nu, Nd = field_u.num_nodes, field_d.num_nodes
    crack_idx, crack_vals = crack

    newton_u = axion.make_newton(field_u, max_iter=20, tol=1e-8, cg_tol=1e-6, cg_maxiter=10000)
    if MODEL in ('AT1', 'CZM'):
        solve_d = axion.make_active_set_newton(field_d, max_iter=50, tol=1e-8,
                                               cg_tol=1e-6, cg_maxiter=2000)
    else:
        newton_d = axion.make_newton(field_d, max_iter=20, tol=1e-8, cg_tol=1e-6,
                                     cg_maxiter=10000)

    @jax.jit
    def stepper(sol_u, sol_d, history_old, fixed, fixed_vals):
        def cond(state):
            su, sd, hist, err, s = state
            return (s < max_stagger) & (err > ms_tol)

        def body(state):
            su, sd, hist, err, s = state
            # ---- displacement (linear at fixed damage) ----
            d_quad = field_u.to_quad(sd)
            u0 = su.reshape(-1).at[fixed].set(fixed_vals)
            R0, V0 = axion.assemble(field_u, u0, (d_quad,))
            su_new = newton_u(u0, R0.at[fixed].set(0.0), V0, (d_quad,), fixed).reshape(Nu, 3)

            # ---- history ----
            hist_new = field_u.compute_history(su_new, history_old)

            # ---- damage ----
            if MODEL in ('AT1', 'CZM'):
                D = axion.abs_diagonal(field_d, sd, (hist_new,))
                sd_new = solve_d(sd.reshape(-1), D, (hist_new,), crack_idx,
                                 crack_vals).reshape(Nd, 1)
            else:
                sd0 = sd.reshape(-1)
                R0d, V0d = axion.assemble(field_d, sd0, (hist_new,))
                sd_new = newton_d(sd0, R0d, V0d, (hist_new,), crack_idx).reshape(Nd, 1)
                sd_new = np.clip(sd_new, 0., 1.)

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
def get_loading_stepper(field_u, field_d, crack, max_stagger=8, ms_tol=1e-3):
    stepper = get_staggered_stepper(field_u, field_d, crack, max_stagger, ms_tol)
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


def run_loading(field_u, field_d, disps, sol_u, sol_d, history_old, crack,
                max_stagger=8, ms_tol=1e-3):
    step_fn = get_loading_stepper(field_u, field_d, crack, max_stagger=max_stagger, ms_tol=ms_tol)
    carry0 = (np.asarray(sol_u), np.asarray(sol_d), np.asarray(history_old))
    (sol_u, sol_d, history), forces = jax.lax.scan(step_fn, carry0, np.asarray(disps))
    return sol_u, sol_d, history, onp.asarray(forces)


def build_fields(nx, ny, nz):
    grid = axion.box_mesh(nx, ny, nz, Lx, Ly, Lz)
    points, cells, n_dup = grid.points, grid.cells, 0
    if CRACK_TYPE == 'slit':
        points, cells, n_dup = insert_crack_3d(points, cells, y_crack, x_tip)
    mesh = Mesh(points, cells, 'HEX8')
    field_d = Damage(mesh, ncomp=1)
    field_u = Displacement(mesh, ncomp=3, dirichlet=dirichlet_u)
    return mesh, field_d, field_u, n_dup


# =============================================================================
# Driver
# =============================================================================
def main():
    t_wall0 = time.time()

    mesh, field_d, field_u, n_dup = build_fields(Nx, Ny, Nz)
    n_nodes, n_cells, n_quads = field_u.num_nodes, field_u.num_cells, field_u.num_quads

    sol_u = onp.zeros((n_nodes, 3))
    sol_d = onp.zeros((n_nodes, 1))
    history_old = onp.zeros((n_cells, n_quads))

    if CRACK_TYPE == 'slit':
        crack = (np.array([], dtype=np.int32), np.zeros((0,)))
        crack_desc = f"geometric slit ({n_dup} duplicated nodes)"
    else:
        crack = crack_dofs(mesh.points)
        crack_desc = f"d = 1 on y = {y_crack}, x <= {x_tip} ({len(onp.asarray(crack[0]))} nodes)"

    DU = float(os.environ.get('PFF_DU', '5e-5'))
    NSTEPS = int(os.environ.get('PFF_NSTEPS', '160'))
    MAX_STAGGER = int(os.environ.get('PFF_MAX_STAGGER', '8'))
    MS_TOL = float(os.environ.get('PFF_MS_TOL', '1e-3'))
    if DU <= 0.0 or NSTEPS <= 0:
        raise ValueError(f'PFF_DU and PFF_NSTEPS must be positive, got {DU}, {NSTEPS}')
    if MAX_STAGGER <= 0 or MS_TOL <= 0.0:
        raise ValueError('PFF_MAX_STAGGER and PFF_MS_TOL must be positive')
    disps = onp.arange(1, NSTEPS + 1, dtype=onp.float64) * DU
    full_disps = onp.concatenate([[0.], disps])

    n_dofs = 4 * n_nodes       # three displacement and one damage unknown per node
    print(f"Axion | 3D single-edge-notched tension [{MODEL}]: {n_dofs} DoFs, {n_nodes} nodes, "
          f"{n_cells} HEX8 cells, mesh {Nx}x{Ny}x{Nz}, l={l}, h=({hx:.4g},{hy:.4g},{hz:.4g}). "
          f"Loading: {NSTEPS} increments of {DU:.4e} mm, final displacement {disps[-1]:.4e} mm. "
          f"Pre-crack: {crack_desc}.", flush=True)

    start = time.time()
    sol_u, sol_d, history, forces = run_loading(field_u, field_d, disps, sol_u, sol_d,
                                                history_old, crack,
                                                max_stagger=MAX_STAGGER, ms_tol=MS_TOL)
    solve_time = time.time() - start
    wall_time = time.time() - t_wall0
    forces = onp.concatenate([[0.0], forces])

    def write_attrs(f):
        f.attrs['n_cells'] = int(n_cells)
        f.attrs['n_nodes'] = int(n_nodes)
        f.attrs['n_dofs'] = int(n_dofs)
        f.attrs['solve_time'] = float(solve_time)
        f.attrs['wall_time'] = float(wall_time)
        f.attrs['peak_load'] = float(onp.max(onp.abs(forces)))
        f.attrs['Lx'] = float(Lx); f.attrs['Ly'] = float(Ly); f.attrs['Lz'] = float(Lz)
        f.attrs['Nx'] = Nx; f.attrs['Ny'] = Ny; f.attrs['Nz'] = Nz
        f.attrs['du'] = DU; f.attrs['nsteps'] = NSTEPS

    with h5py.File(os.path.join(out_dir, f'force_disp_{MODEL}.h5'), 'w') as f:
        f.create_dataset('disps', data=full_disps)
        f.create_dataset('forces', data=forces)
        write_attrs(f)
    with h5py.File(os.path.join(out_dir, f'final_{MODEL}.h5'), 'w') as f:
        f.create_dataset('points', data=mesh.points)
        f.create_dataset('cells', data=mesh.cells)
        f.create_dataset('sol_d', data=onp.asarray(sol_d).reshape(-1))
        f.create_dataset('disps', data=full_disps)
        f.create_dataset('forces', data=forces)
        write_attrs(f)
    axion.write_vtu(os.path.join(out_dir, f'final_{MODEL}.vtu'), mesh,
                    point_data={'d': onp.asarray(sol_d), 'u': onp.asarray(sol_u)},
                    cell_data={'history': onp.mean(onp.asarray(history), axis=1)})

    peak_i = int(onp.abs(forces).argmax())
    summary = (
        f"=============== SUMMARY ({MODEL}, 3D uniform mesh) ===============\n"
        f"  DoF              : {n_dofs}  ({n_cells} HEX8 cells, {n_nodes} nodes)\n"
        f"  load increments  : {NSTEPS} (du = {DU:.4e} mm)\n"
        f"  solve time [s]   : {solve_time:.1f}  ({solve_time / 60.:.2f} min, includes compilation)\n"
        f"  wall time [s]    : {wall_time:.1f}\n"
        f"  time / increment : {solve_time / NSTEPS:.3f} s\n"
        f"  peak load        : {abs(forces[peak_i]):.6f} N at u = {full_disps[peak_i]:.6e} mm\n"
        f"==================================================================")
    print("\n" + summary)
    with open(os.path.join(out_dir, f'summary_{MODEL}.txt'), 'w') as fh:
        fh.write(summary + "\n")
    print(f"[{MODEL}] output in {out_dir}/")


if __name__ == '__main__':
    main()
