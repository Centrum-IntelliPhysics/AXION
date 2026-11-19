import os
import sys
import glob
import time
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import jax
import jax.numpy as np
import numpy as onp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import axion
from axion import DirichletBC, Field, Mesh, isotropic_function, negative_part, positive_part

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
crt_file_path = os.path.dirname(os.path.abspath(__file__))
# PFF_OUTDIR lets several cases write to separate folders.
output_dir = os.path.join(crt_file_path, os.environ.get('PFF_OUTDIR', 'output_notch'))
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vtk_dir, exist_ok=True)

safe_plus = positive_part
safe_minus = negative_part

# =============================================================================
# Material / model parameters  (Zhou et al. 2023, Table 2 -- SI units)
# =============================================================================
# Solid (mu, lam given -> E=60 GPa, nu=0.3, K_Vr = lam + 2mu/3 = 50 GPa)
mu = 23.08e9           # shear modulus [Pa]
lmbda = 34.62e9        # Lame lambda [Pa]
E = mu * (3. * lmbda + 2. * mu) / (lmbda + mu)      # ~60 GPa
nu = lmbda / (2. * (lmbda + mu))                    # ~0.3
K_Vr = lmbda + 2. * mu / 3.                          # drained bulk modulus ~50 GPa

G_c = 500.0            # critical energy release rate [N/m]
l = float(os.environ.get('PFF_L0', '0.1'))   # phase-field length scale l0 [m]
K_RES = 1e-9           # residual stiffness floor k

# Biot / indicator interpolation thresholds (c1, c2)
d1, d2 = 0.4, 1.0
alpha_R, alpha_F = 0.05, 1.0     # Biot coefficient reservoir / fracture

# Hydraulic properties (reservoir R / fracture F)
eps_p = 0.05                     # porosity
rho_R, rho_F = 1.0e3, 1.0e3      # fluid density [kg/m^3]
k_R, k_F = 1.0e-15, 8.333e-4     # permeability [m^2]
c_R, c_F = 1.0e-8, 1.0e-8        # fluid compressibility [1/Pa]
mu_R, mu_F = 1.0e-3, 1.0e-3      # fluid viscosity [Pa s]

# Initial in-situ stress (in-plane), env-configurable [MPa] via PFF_SIGX0/PFF_SIGY0.
# Fig. 2 shows confining (compressive) stress; tension-positive convention -> the
# applied magnitudes enter as NEGATIVE (SIG0_SIGN = -1).
#   sec 4.1 (horizontal notch, isotropic): sigx0 = sigy0 = 0.5 MPa   (defaults)
#   sec 4.2 (inclined notch, contrast):    sigy0 = 0.5, sigx0 = ratio*0.5 MPa
SIG0_SIGN = float(os.environ.get('PFF_SIG0_SIGN', '-1.0'))   # -1 compression
sig_x0 = SIG0_SIGN * float(os.environ.get('PFF_SIGX0', '0.5')) * 1e6
sig_y0 = SIG0_SIGN * float(os.environ.get('PFF_SIGY0', '0.5')) * 1e6
SIGMA0 = np.array([[sig_x0, 0.0], [0.0, sig_y0]])

# Section 4.5 / Fig. 19 uses spatially varying compression. With y in [-5,5] m,
# sigma_x changes from -5 MPa at the bottom to -1 MPa at the top, and sigma_y
# changes from -2.25 MPa to -2 MPa. The latter gradient is equilibrated by a
# downward self-weight body force of 25 kN/m^3 (approximately rho_s=2500 kg/m^3
# with g=10 m/s^2). Gravity is mechanical here; Eq. (26) in the paper omits it
# from Darcy velocity.
STRESS_PROFILE = os.environ.get('PFF_STRESS_PROFILE', 'constant').strip().lower()
if STRESS_PROFILE not in ('constant', 'fig19_linear'):
    raise ValueError(
        'PFF_STRESS_PROFILE must be constant or fig19_linear, '
        f'got {STRESS_PROFILE!r}')
FIG19_SIGX_TOP_MPA = 1.0
FIG19_SIGX_BOTTOM_MPA = 5.0
FIG19_SIGY_TOP_MPA = 2.0
FIG19_SIGY_BOTTOM_MPA = 2.25
GRAVITY_BODY_FORCE_Y = -25.0e3 if STRESS_PROFILE == 'fig19_linear' else 0.0


def initial_stress_at(points):
    """Tension-positive in-situ stress tensor at physical points."""
    y = points[..., 1]
    if STRESS_PROFILE == 'fig19_linear':
        eta = (y - Y0) / (Y1 - Y0)
        sigx_mag = FIG19_SIGX_BOTTOM_MPA + eta * (
            FIG19_SIGX_TOP_MPA - FIG19_SIGX_BOTTOM_MPA)
        sigy_mag = FIG19_SIGY_BOTTOM_MPA + eta * (
            FIG19_SIGY_TOP_MPA - FIG19_SIGY_BOTTOM_MPA)
        sx = -1.0e6 * sigx_mag
        sy = -1.0e6 * sigy_mag
    else:
        sx = np.full_like(y, sig_x0)
        sy = np.full_like(y, sig_y0)
    result = np.zeros(points.shape[:-1] + (2, 2), dtype=points.dtype)
    result = result.at[..., 0, 0].set(sx)
    result = result.at[..., 1, 1].set(sy)
    return result

# Diagnostic toggle: where the initial stress sigma0 acts. The sec 4.2 crack was
# observed to deflect toward S_min (vertical) instead of S_max (horizontal) with
# mode='both'; this isolates which term (stress g(phi)sigma0 vs history sigma0:eps)
# drives the (wrong) directional bias.  PFF_SIG0_MODE in {both, stress, hist, off}.
_SIG0_MODE = os.environ.get('PFF_SIG0_MODE', 'both').lower()
SIG0_IN_STRESS = _SIG0_MODE in ('both', 'stress')   # g(phi) sigma0 in total stress
SIG0_IN_HIST = _SIG0_MODE in ('both', 'hist')       # sigma0:eps in history H

# Geometry: FULL 10 m x 10 m plate, centred at origin. Notch length 0.8 m. The
# single layout places one notch through the centre at inclination THETA:
#   sec 4.1 -> THETA = 0   (horizontal notch)
#   sec 4.2 -> THETA = 45  (inclined notch)
# The Fig. 18 layout places two horizontal notches at y=+/-spacing/2.
Lx, Ly = 10.0, 10.0
X0, X1 = -Lx / 2., Lx / 2.
Y0, Y1 = -Ly / 2., Ly / 2.
lc = 0.4               # notch half-length [m] (full notch length 0.8 m)
THETA = onp.deg2rad(float(os.environ.get('PFF_THETA', '0.0')))
NOTCH_LAYOUT = os.environ.get('PFF_NOTCH_LAYOUT', 'single').strip().lower()
NOTCH_SPACING = float(os.environ.get('PFF_NOTCH_SPACING', '1.0'))
SOURCE_HALF_WIDTH = float(os.environ.get(
    'PFF_SOURCE_HALF_WIDTH', str(0.5 * l)))
if NOTCH_LAYOUT not in ('single', 'double_horizontal'):
    raise ValueError(
        'PFF_NOTCH_LAYOUT must be single or double_horizontal, '
        f'got {NOTCH_LAYOUT!r}')
if NOTCH_LAYOUT == 'double_horizontal' and NOTCH_SPACING <= 0.0:
    raise ValueError(f'PFF_NOTCH_SPACING must be positive, got {NOTCH_SPACING}')
if NOTCH_LAYOUT == 'double_horizontal' and not onp.isclose(THETA, 0.0):
    raise ValueError('Fig. 18 double_horizontal notches require PFF_THETA=0')
if SOURCE_HALF_WIDTH <= 0.0:
    raise ValueError(
        'PFF_SOURCE_HALF_WIDTH must be positive, '
        f'got {SOURCE_HALF_WIDTH}')

if NOTCH_LAYOUT == 'single':
    NOTCH_CENTERS = onp.array([[0.0, 0.0]])
    NOTCH_ANGLES = onp.array([THETA])
else:
    NOTCH_CENTERS = onp.array(
        [[0.0, -0.5 * NOTCH_SPACING], [0.0, 0.5 * NOTCH_SPACING]])
    NOTCH_ANGLES = onp.zeros(2)

H0_B = 1.0e3           # notch seed amplitude factor

# Equation (20) defines H as a maximum over accepted physical time.  During a
# nonlinear solve, every trial state must therefore be compared with the fixed
# history from the preceding converged time step.  Accumulating maxima from
# unconverged stagger iterates creates artificial damage that cannot be undone.
HISTORY_RULE = 'physical_time_converged_history_v4'
SOLVER_RULE = 'coupled_up_physical_history_phi_anderson_rms_v7'

# =============================================================================
# phi-dependent model functions
# =============================================================================
def crack_geo(d):
    return d ** 2                       # AT2 w(d) = d^2


def degradation(d):
    return (1. - K_RES) * (1. - d) ** 2 + K_RES   # g(d) = (1-k)(1-d)^2 + k


def c_w():
    return 2.0                           # AT2


def _chi_F(d):
    return np.clip((d - d1) / (d2 - d1), 0., 1.)


def alpha_biot(d):
    cf = _chi_F(d)
    return alpha_R * (1. - cf) + alpha_F * cf


def fluid_coeffs(d, dt):
    """phi-interpolated hydraulic coefficients per quad (Eqs. 23-25).

    Returns (mob, coeff_S, rho_a_chiR):
      mob        = rho K_eff / mu_eff            -> Darcy mobility (tensor_map)
      coeff_S    = rho S / dt                    -> storage/time coeff (mass_map)
      rho_a_chiR = rho * alpha * chi_R           -> poro-mech coupling factor
    """
    cf = _chi_F(d)
    cr = 1. - cf
    alpha = alpha_R * cr + alpha_F * cf
    K_eff = k_R * cr + k_F * cf
    mu_eff = mu_R * cr + mu_F * cf
    rho = rho_R * cr + rho_F * cf
    c = c_R * cr + c_F * cf
    S = eps_p * c + (alpha - eps_p) * (1. - alpha) / K_Vr
    mob = rho * K_eff / mu_eff
    coeff_S = rho * S / dt
    return mob, coeff_S, rho * alpha * cr


_g_prime = jax.grad(degradation)
_w_prime = jax.grad(crack_geo)

# =============================================================================
# Phase-field (damage) subproblem  -- AT2, driven by history H (Eq. 30)
# =============================================================================
class Damage(Field):
    def flux(self):
        coeff = 2. * G_c * l / c_w()          # -> Gc l0 grad d
        def fn(d_grad, history):
            return coeff * d_grad
        return fn

    def source(self):
        coeff = G_c / (c_w() * l)             # -> Gc/l0 d
        def fn(d, x, history):
            dd = d[0]
            return np.array([coeff * _w_prime(dd) + _g_prime(dd) * history])
        return fn

# =============================================================================
# Displacement subproblem  -- spectral split + initial stress + fluid pressure
# =============================================================================
def strain(u_grad):
    return 0.5 * (u_grad + u_grad.T)


def psi_plus(epsilon):
    eigen_vals, _ = np.linalg.eigh(epsilon)
    tr_plus = safe_plus(np.trace(epsilon))
    return lmbda / 2. * tr_plus ** 2 + mu * np.sum(safe_plus(eigen_vals) ** 2)


def psi_plus_fn(u_grad):
    return psi_plus(strain(u_grad))


# deviatoric part of the split effective stress; g is the degradation at the
# quadrature point
_split_stress = isotropic_function(
    lambda lam, g: 2. * mu * (g * safe_plus(lam) + safe_minus(lam)))


def stress_fn(u_grad, d, p, sigma0):
    """sigma = sigma^e + g(d) sigma0(x) - alpha(d) p I (Eqs. 15-16)."""
    epsilon = strain(u_grad)
    g = degradation(d[0])
    tr_plus = safe_plus(np.trace(epsilon))
    tr_minus = safe_minus(np.trace(epsilon))
    sigma_e = lmbda * (g * tr_plus + tr_minus) * np.eye(2) + _split_stress(epsilon, g)
    # + initial-stress load g(d) sigma0(x) and - fluid-pressure load alpha p I
    sigma0_term = g * sigma0 if SIG0_IN_STRESS else np.zeros((2, 2))
    sigma_p = alpha_biot(d[0]) * p[0] * np.eye(2)
    return sigma_e + sigma0_term - sigma_p


class Displacement(Field):
    """theta = (d, p, sigma0) at the quadrature points."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma0_quad = initial_stress_at(self.quad_points)

    def flux(self):
        return stress_fn

    def boundary_sources(self):
        """Residual contributions of the prescribed in-situ stress tractions.

        The stress boundary condition is ``ft = sigma0(x) @ n``. Boundary terms
        are added to the residual, so each function returns ``-ft`` to reproduce
        the weak form of Eq. (29). These tractions are deliberately *not*
        degraded: only the internal initial-stress energy is multiplied by g(d),
        while the remote boundary stress remains prescribed.
        """
        normals = (
            np.array([-1.0, 0.0]),  # left
            np.array([1.0, 0.0]),   # right
            np.array([0.0, -1.0]),  # bottom
            np.array([0.0, 1.0]),   # top
        )

        def make_map(normal):
            def boundary_source(u, x):
                del u
                return -(initial_stress_at(x) @ normal)

            return boundary_source

        return [make_map(normal) for normal in normals]

    def source(self):
        """Mechanical gravity residual -b for the self-weight of the varying-stress case."""
        def fn(u, x, d, p, sigma0):
            del u, x, d, p, sigma0
            return np.array([0.0, -GRAVITY_BODY_FORCE_Y])
        return fn

    def _grads_center(self, sol_u):
        return np.mean(self.grad_at_quad(sol_u), axis=1)      # (ncell, vec, dim), cell centre

    def compute_history(self, sol_u, history_old):
        """H = max( psi+(eps) + sigma0:eps , H_old )  (Eq. 20)."""
        u_grad_center = self._grads_center(sol_u)
        eps_center = 0.5 * (u_grad_center + np.transpose(u_grad_center, (0, 2, 1)))
        psi_center = jax.vmap(psi_plus_fn)(u_grad_center)
        cell_centers = np.mean(self.points[self.cells], axis=1)
        sigma0_center = initial_stress_at(cell_centers)
        sig0_eps = (sigma0_center[:, 0, 0] * eps_center[:, 0, 0]
                    + sigma0_center[:, 1, 1] * eps_center[:, 1, 1])
        drive = psi_center + (sig0_eps if SIG0_IN_HIST else 0.)
        h_center = np.maximum(drive, history_old[:, 0])
        return np.broadcast_to(h_center[:, None], history_old.shape)

    def compute_evol_quad(self, sol_u):
        """Volumetric strain eps_vol = tr(eps) at the quadrature points, (ncell, nquad)."""
        u_grads = self.grad_at_quad(sol_u)             # (ncell, nquad, vec, dim)
        return u_grads[:, :, 0, 0] + u_grads[:, :, 1, 1]

# =============================================================================
# Fluid-pressure subproblem  -- linear Darcy diffusion + storage (Eq. 31)
# =============================================================================
class Pressure(Field):
    """int B^T mob grad p  +  int N (coeff_S p - rhs) = 0,  all coefficients per
    quadrature point: theta = (mob, coeff_S, rhs)."""
    def flux(self):
        def fn(p_grad, mob, coeff_S, rhs):
            return mob * p_grad
        return fn

    def source(self):
        def fn(p, x, mob, coeff_S, rhs):
            return np.array([coeff_S * p[0] - rhs])
        return fn

# =============================================================================
# Boundary conditions
#   displacement: zero TANGENTIAL displacement on the outer edges
#                 (left/right -> u_y=0 ; top/bottom -> u_x=0)   [Fig. 4]
#                 and prescribed normal traction ft=SIGMA0@n
#   pressure    : p = 0 on all outer edges
#   phase field : natural (grad phi . n = 0)  -> no Dirichlet
# =============================================================================
TOL = 1e-8


def on_left_right(point):
    return np.isclose(point[0], X0, atol=TOL) | np.isclose(point[0], X1, atol=TOL)


def on_top_bottom(point):
    return np.isclose(point[1], Y0, atol=TOL) | np.isclose(point[1], Y1, atol=TOL)


def on_left(point):
    return np.isclose(point[0], X0, atol=TOL)


def on_right(point):
    return np.isclose(point[0], X1, atol=TOL)


def on_bottom(point):
    return np.isclose(point[1], Y0, atol=TOL)


def on_top(point):
    return np.isclose(point[1], Y1, atol=TOL)


def on_outer(point):
    return on_left_right(point) | on_top_bottom(point)


# left/right fix u_y (component 1); top/bottom fix u_x (component 0)
dirichlet_u = [DirichletBC(on_left_right, 1, 0.), DirichletBC(on_top_bottom, 0, 0.)]
# pressure p = 0 on the whole outer boundary
dirichlet_p = [DirichletBC(on_outer, 0, 0.)]

# =============================================================================
# Structured mesh (centred). TRI3 uses alternating diagonals to avoid a uniform
# directional bias; QUAD4 remains available for backward-compatible runs.
# =============================================================================
def rectangle_mesh(nx, ny, ele_type='QUAD4'):
    xs = onp.linspace(X0, X1, nx + 1)
    ys = onp.linspace(Y0, Y1, ny + 1)
    nxp = len(xs)
    X, Y = onp.meshgrid(xs, ys, indexing="xy")
    points = onp.stack([X.ravel(), Y.ravel()], axis=1)

    def nid(i, j):
        return j * nxp + i

    cells = []
    for j in range(len(ys) - 1):
        for i in range(nxp - 1):
            n00 = nid(i, j)
            n10 = nid(i + 1, j)
            n11 = nid(i + 1, j + 1)
            n01 = nid(i, j + 1)
            if ele_type == 'QUAD4':
                cells.append([n00, n10, n11, n01])
            elif ele_type == 'TRI3':
                # Checkerboard diagonals avoid imposing one preferred diagonal
                # throughout the domain while retaining exact outer boundaries.
                if (i + j) % 2 == 0:
                    cells.extend(([n00, n10, n11], [n00, n11, n01]))
                else:
                    cells.extend(([n00, n10, n01], [n10, n11, n01]))
            else:
                raise ValueError(f'PFF_ELE_TYPE must be TRI3 or QUAD4, got {ele_type!r}')
    return points, onp.array(cells, dtype=onp.int32)


def unstructured_rectangle_mesh(h_target):
    """Deterministic, reflection-symmetric Gmsh TRI3 mesh.

    Gmsh discretizes one quadrant; reflecting that triangulation about both axes
    preserves the exact symmetry of Fig. 18 without reintroducing the directional
    locking of a Cartesian checkerboard mesh.
    """
    if h_target <= 0.0:
        raise ValueError(f'PFF_MESH_H must be positive, got {h_target}')
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            'PFF_MESH_KIND=unstructured_gmsh requires the gmsh Python package') from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.option.setNumber('Mesh.Algorithm', 6)  # Frontal-Delaunay
        gmsh.option.setNumber('Mesh.MeshSizeMin', h_target)
        gmsh.option.setNumber('Mesh.MeshSizeMax', h_target)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.add('two_horizontal_notches_quadrant')
        corners = [
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_target),
            gmsh.model.geo.addPoint(X1, 0.0, 0.0, h_target),
            gmsh.model.geo.addPoint(X1, Y1, 0.0, h_target),
            gmsh.model.geo.addPoint(0.0, Y1, 0.0, h_target),
        ]
        lines = [
            gmsh.model.geo.addLine(corners[i], corners[(i + 1) % 4])
            for i in range(4)
        ]
        loop = gmsh.model.geo.addCurveLoop(lines)
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        quadrant_points = onp.asarray(
            coordinates, dtype=float).reshape(-1, 3)[:, :2]
        element_tags, element_nodes = gmsh.model.mesh.getElementsByType(2)
        del element_tags
        element_nodes = onp.asarray(element_nodes, dtype=onp.int64).reshape(-1, 3)
        tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
        quadrant_cells = onp.asarray(
            [[tag_to_index[int(tag)] for tag in row] for row in element_nodes],
            dtype=onp.int32)
    finally:
        gmsh.finalize()

    points_list = []
    cells_list = []
    point_index = {}
    for sx, sy in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)):
        local_to_global = onp.empty(len(quadrant_points), dtype=onp.int32)
        for local_id, (x, y) in enumerate(quadrant_points):
            xr = 0.0 if abs(x) <= 1e-13 else sx * x
            yr = 0.0 if abs(y) <= 1e-13 else sy * y
            key = (round(float(xr), 13), round(float(yr), 13))
            if key not in point_index:
                point_index[key] = len(points_list)
                points_list.append((xr, yr))
            local_to_global[local_id] = point_index[key]
        cells_list.append(local_to_global[quadrant_cells])
    points = onp.asarray(points_list, dtype=float)
    cells = onp.concatenate(cells_list, axis=0).astype(onp.int32, copy=False)

    # Use a consistent counter-clockwise TRI3 orientation.
    triangle_points = points[cells]
    signed_twice_area = (
        (triangle_points[:, 1, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 2, 1] - triangle_points[:, 0, 1])
        - (triangle_points[:, 1, 1] - triangle_points[:, 0, 1])
        * (triangle_points[:, 2, 0] - triangle_points[:, 0, 0]))
    clockwise = signed_twice_area < 0.0
    cells[clockwise, 1], cells[clockwise, 2] = (
        cells[clockwise, 2].copy(), cells[clockwise, 1].copy())
    if onp.any(onp.isclose(signed_twice_area, 0.0)):
        raise RuntimeError('Gmsh generated a degenerate triangle')
    return points, cells


def mesh_fingerprint(points, cells):
    digest = hashlib.sha256()
    digest.update(onp.ascontiguousarray(points, dtype=onp.float64).tobytes())
    digest.update(onp.ascontiguousarray(cells, dtype=onp.int32).tobytes())
    return digest.hexdigest()


def reflection_symmetry_maps(points, cells):
    """Index maps for the identity, x, y, and xy reflections.

    Fig. 18 has reflection-symmetric geometry, loads, sources, and initial
    notches. These maps let the nonlinear solver remain on that exact symmetric
    solution branch through the abrupt crack-jump step.
    """
    points = onp.asarray(points, dtype=float)
    cells = onp.asarray(cells, dtype=onp.int32)

    def make_maps(coordinates, label, decimals):
        lookup = {
            (round(float(x), decimals), round(float(y), decimals)): index
            for index, (x, y) in enumerate(coordinates)
        }
        maps = []
        for sx, sy in ((1.0, 1.0), (-1.0, 1.0),
                       (1.0, -1.0), (-1.0, -1.0)):
            reflected = []
            for x, y in coordinates:
                key = (round(float(sx * x), decimals),
                       round(float(sy * y), decimals))
                if key not in lookup:
                    raise RuntimeError(
                        f'{label} mesh is not reflection-symmetric at {key}')
                reflected.append(lookup[key])
            maps.append(reflected)
        return onp.asarray(maps, dtype=onp.int32)

    node_maps = make_maps(points, 'node', 12)
    # Reflected triangles may sum their three vertices in a different order;
    # ten decimals safely absorbs that roundoff while remaining far below h.
    cell_maps = make_maps(points[cells].mean(axis=1), 'cell-centroid', 10)
    return node_maps, cell_maps


def reflection_projectors(node_maps=None, cell_maps=None):
    """Return JAX-compatible projectors onto the Fig. 18 symmetry subspace."""
    if node_maps is None or cell_maps is None:
        return (lambda values: values,
                lambda values: values,
                lambda values: values)

    node_maps = np.asarray(node_maps, dtype=np.int32)
    cell_maps = np.asarray(cell_maps, dtype=np.int32)
    vector_signs = np.asarray(
        [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])

    def scalar_project(values):
        return np.mean(values[node_maps], axis=0)

    def vector_project(values):
        reflected_back = values[node_maps] * vector_signs[:, None, :]
        return np.mean(reflected_back, axis=0)

    def cell_scalar_project(values):
        # History is deliberately cellwise constant; rebuild that representation
        # after averaging each four-cell reflection orbit.
        cell_values = np.mean(values, axis=1)
        symmetric_values = np.mean(cell_values[cell_maps], axis=0)
        return np.broadcast_to(symmetric_values[:, None], values.shape)

    return scalar_project, vector_project, cell_scalar_project


def _notch_frames(cell_centroids):
    """Coordinates relative to every seeded notch (point, notch arrays)."""
    points = onp.asarray(cell_centroids)
    offsets = points[:, None, :] - NOTCH_CENTERS[None, :, :]
    ct = onp.cos(NOTCH_ANGLES)[None, :]
    st = onp.sin(NOTCH_ANGLES)[None, :]
    xp = offsets[..., 0] * ct + offsets[..., 1] * st
    yp = -offsets[..., 0] * st + offsets[..., 1] * ct
    return xp, yp


def compute_H0(cell_centroids):
    """High initial history in an l0/2 band around every notch segment."""
    xp, yp = _notch_frames(cell_centroids)
    dx = onp.maximum(onp.abs(xp) - lc, 0.0)
    d_s = onp.min(onp.sqrt(dx ** 2 + yp ** 2), axis=1)
    amp = H0_B * G_c / (2.0 * l)
    return onp.where(d_s <= 0.5 * l, amp * (1.0 - 2.0 * d_s / l), 0.0)


def compute_source(cell_centroids, q_F):
    """Fluid mass source q_m=q_F in the union of all notch footprints."""
    xp, yp = _notch_frames(cell_centroids)
    if NOTCH_LAYOUT == 'double_horizontal':
        # Fig. 18: inject over a rounded segment band. Its half-width is an
        # explicit input because the paper specifies the 0.8 m notch length but
        # not the finite aperture needed to integrate volumetric q_F. The
        # default l0/2 matches the history-seed footprint.
        dx = onp.maximum(onp.abs(xp) - lc, 0.0)
        distance_to_segment = onp.sqrt(dx ** 2 + yp ** 2)
        per_notch = distance_to_segment <= SOURCE_HALF_WIDTH + TOL
    else:
        # Preserve the validated Sections 4.1-4.2 single-notch source rule.
        per_notch = (onp.abs(xp) <= lc + TOL) & (onp.abs(yp) <= 0.5 * l)
    in_notch = onp.any(per_notch, axis=1)
    return onp.where(in_notch, q_F, 0.0)


def build_fields(points, cells, ele_type='QUAD4', quadrature_order=None):
    mesh = Mesh(onp.asarray(points), onp.asarray(cells), ele_type)
    field_d = Damage(mesh, ncomp=1, quad_degree=quadrature_order)
    field_u = Displacement(mesh, ncomp=2, quad_degree=quadrature_order,
                           dirichlet=dirichlet_u,
                           boundaries=[on_left, on_right, on_bottom, on_top])
    field_p = Pressure(mesh, ncomp=1, quad_degree=quadrature_order,
                       dirichlet=dirichlet_p)
    return mesh, field_d, field_u, field_p


def mesh_max_edge(points, cells):
    """Return the actual longest element edge, not the Cartesian grid spacing."""
    cell_points = onp.asarray(points)[onp.asarray(cells)]
    rolled = onp.roll(cell_points, shift=-1, axis=1)
    return float(onp.linalg.norm(rolled - cell_points, axis=2).max())


def dirichlet_data(field):
    return field.dirichlet_dofs, field.dirichlet_values

# =============================================================================
# Operators (fixed mesh -> index tables constant -> compile once)
# =============================================================================
def get_assembled_ops(field):
    cache = getattr(field, '_asm_ops', None)
    if cache is not None:
        return cache
    matvec, inv_jacobi = axion.make_operator(field)
    cache = (matvec, inv_jacobi, field.ncomp, field.num_nodes)
    field._asm_ops = cache
    return cache


def assemble_device(field, u_flat, internal_vars):
    """Residual and stored tangent entries of ``field`` at ``u_flat``."""
    return axion.assemble(field, u_flat, internal_vars)


def get_newton_solver(field, num_newton=30, nl_tol=1e-8, nl_atol=1e-7,
                      lin_tol=1e-8, lin_maxiter=5000, project_flat=None):
    cache = getattr(field, '_newton_solver', None)
    if cache is not None:
        return cache
    matvec, inv_jacobi, vec, N = get_assembled_ops(field)
    if project_flat is None:
        project_flat = lambda values: values

    @jax.jit
    def newton(u0, R0, V0, iv, fixed_idx, fixed_vals):
        # A mixed absolute/relative tolerance is essential for a balanced stress
        # boundary. Use an RMS residual so the acceptance criterion does not get
        # stricter merely because a refined mesh has more algebraic unknowns.
        # The former global L2 norm rejected a converged 324900-DOF state whose
        # residual consisted only of accumulated assembly roundoff.
        u0 = project_flat(u0).at[fixed_idx].set(fixed_vals)
        R0 = project_flat(R0).at[fixed_idx].set(0.0)
        residual_scale = np.sqrt(np.asarray(R0.size, dtype=R0.dtype))
        residual_rms0 = np.linalg.norm(R0) / residual_scale
        newton_tol = nl_atol + nl_tol * residual_rms0

        def cond(state):
            u, R, V, it = state
            residual_rms = np.linalg.norm(R) / residual_scale
            return (it < num_newton) & (residual_rms > newton_tol)
        def body(state):
            u, R, V, it = state
            inv = inv_jacobi(V, fixed_idx)
            du, _ = jax.scipy.sparse.linalg.cg(
                lambda d: project_flat(
                    matvec(V, project_flat(d), fixed_idx)), -R,
                tol=lin_tol, atol=0.0, maxiter=lin_maxiter,
                M=lambda r: project_flat(r * inv))
            # For Fig. 18, keep every Newton iterate in the reflection-symmetric
            # subspace. The convergence residual is then evaluated on the actual
            # projected state returned to the staggered solver.
            u_new = project_flat(u + du).at[fixed_idx].set(fixed_vals)
            res_new, V_new = assemble_device(field, u_new, iv)
            R_new = project_flat(res_new).at[fixed_idx].set(0.0)
            return (u_new, R_new, V_new, it + 1)
        u, R, V, it = jax.lax.while_loop(cond, body, (u0, R0, V0, 0))
        del V
        residual_rms = np.linalg.norm(R) / residual_scale
        residual_ratio = residual_rms / np.maximum(newton_tol, 1e-30)
        converged = np.isfinite(residual_ratio) & np.all(np.isfinite(u)) \
            & (residual_ratio <= 1.0)
        return u, residual_ratio, it, converged

    field._newton_solver = newton
    return newton


# =============================================================================
# Pre-injection equilibrium for the seeded notch (paper's t = 0 state)
# =============================================================================
def get_initial_state_solver(field_u, field_d,
                             max_stagger=200, ms_tol=1e-4,
                             symmetry_node_maps=None,
                             symmetry_cell_maps=None):
    """Equilibrate ``u`` and ``phi`` at p=0 before fluid injection.

    The pre-existing notch is first introduced through H0.  The fixed remote
    stress tractions then produce only the local excavation displacement seen in
    the paper's t=0 state.  Pressure remains exactly zero during this solve; the
    resulting volumetric strain becomes ``evol_old`` for the first flow step.
    """
    cache = getattr(field_u, '_initial_state_solver', None)
    if cache is not None:
        return cache

    fe_u = field_u
    Nu = fe_u.num_nodes
    Nd = field_d.num_nodes
    ncell, nq = fe_u.num_cells, fe_u.num_quads
    scalar_project, vector_project, cell_project = reflection_projectors(
        symmetry_node_maps, symmetry_cell_maps)
    newton_u = get_newton_solver(
        field_u,
        project_flat=lambda values: vector_project(
            values.reshape(Nu, 2)).reshape(-1))
    newton_d = get_newton_solver(
        field_d,
        project_flat=lambda values: scalar_project(
            values.reshape(Nd, 1)).reshape(-1))

    zero_p_quad = np.zeros((ncell, nq, 1))
    empty_idx = np.array([], dtype=np.int32)
    empty_vals = np.zeros((0,))

    @jax.jit
    def equilibrate(sol_u, sol_d, history_old, fixed_u_idx, fixed_u_vals):
        def cond(state):
            su, sd, hist, err, s, inner_ratio, inner_ok = state
            del hist, inner_ratio, inner_ok
            return (s < max_stagger) & (err > ms_tol)

        def body(state):
            su, sd, hist, err, s, inner_ratio, inner_ok = state
            del err, inner_ratio, inner_ok

            d_quad = fe_u.to_quad(sd)
            iv_u = [d_quad, zero_p_quad, field_u.sigma0_quad]
            u0 = su.reshape(-1).at[fixed_u_idx].set(fixed_u_vals)
            r0, V0 = assemble_device(field_u, u0, iv_u)
            R0 = r0.at[fixed_u_idx].set(0.0)
            su_flat, ratio_u, _, ok_u = newton_u(
                u0, R0, V0, iv_u, fixed_u_idx, fixed_u_vals)
            su_map = vector_project(su_flat.reshape(Nu, 2))

            # Equation (20): renew the trial field from the fixed history at
            # the preceding accepted state, never from another trial iterate.
            hist_map = cell_project(
                field_u.compute_history(su_map, history_old))
            d0 = sd.reshape(-1)
            r0d, V0d = assemble_device(field_d, d0, [hist_map])
            sd_flat, ratio_d, _, ok_d = newton_d(
                d0, r0d, V0d, [hist_map], empty_idx, empty_vals)
            sd_map = scalar_project(
                np.clip(sd_flat.reshape(Nd, 1), 0.0, 1.0))

            # Floors make a zero-state equilibrium insensitive to floating-point
            # roundoff while retaining a relative test once a field is nonzero.
            eu = np.linalg.norm(su_map - su) / np.maximum(np.linalg.norm(su_map), 1e-12)
            ed = np.linalg.norm(sd_map - sd) / np.maximum(np.linalg.norm(sd_map), 1e-12)
            map_error = np.maximum(eu, ed)
            map_inner_ok = ok_u & ok_d
            validated = map_inner_ok & np.isfinite(map_error) & (map_error <= ms_tol)

            # The residual above belongs to the current state (su, sd). If it is
            # converged, keep that exact tested state; do not return the untested
            # fixed-point image computed after it.
            su_candidate = vector_project(np.where(validated, su, su_map))
            sd_candidate = scalar_project(np.where(validated, sd, sd_map))
            # Never advance any state from an unconverged inner solve.
            su_out = np.where(map_inner_ok, su_candidate, su)
            sd_out = np.where(map_inner_ok, sd_candidate, sd)
            # Commit history corresponding to the state actually retained.
            hist_accepted = cell_project(
                field_u.compute_history(su_out, history_old))
            hist_out = np.where(map_inner_ok, hist_accepted, hist)
            error_out = np.where(map_inner_ok, map_error, np.inf)
            ratio_out = np.array([ratio_u, ratio_d])
            return (su_out, sd_out, hist_out, error_out, s + 1,
                    ratio_out, map_inner_ok)

        init = (sol_u, sol_d, history_old, np.array(1e30), np.array(0),
                np.full((2,), np.inf), np.array(False))
        return jax.lax.while_loop(cond, body, init)

    field_u._initial_state_solver = equilibrate
    return equilibrate

# =============================================================================
# One staggered time step (backward Euler), jitted
# =============================================================================
def get_time_stepper(field_u, field_d, field_p, dt,
                     max_stagger=200, ms_tol=1e-4, anderson_depth=5,
                     anderson_start=2, anderson_beta=1.0,
                     anderson_regularization=1e-8,
                     up_max_iter=50, up_tol=1e-6,
                     stagger_relax=0.5, relax_after=20,
                     symmetry_node_maps=None,
                     symmetry_cell_maps=None):
    cache = getattr(field_u, '_time_stepper', None)
    if cache is not None:
        return cache
    fe_u = field_u; Nu = fe_u.num_nodes
    fe_d = field_d; Nd = fe_d.num_nodes
    fe_p = field_p; Np = fe_p.num_nodes
    ncell, nq = fe_u.num_cells, fe_u.num_quads
    scalar_project, vector_project, cell_project = reflection_projectors(
        symmetry_node_maps, symmetry_cell_maps)
    newton_u = get_newton_solver(
        field_u,
        project_flat=lambda values: vector_project(
            values.reshape(Nu, 2)).reshape(-1))
    newton_d = get_newton_solver(
        field_d,
        project_flat=lambda values: scalar_project(
            values.reshape(Nd, 1)).reshape(-1))
    newton_p = get_newton_solver(
        field_p,
        project_flat=lambda values: scalar_project(
            values.reshape(Np, 1)).reshape(-1))

    empty_idx = np.array([], dtype=np.int32)
    empty_vals = np.zeros((0,))

    total_dofs = Nu * 2 + Np + Nd
    memory_rows = np.arange(anderson_depth)

    @jax.jit
    def time_step(sol_u, sol_p, sol_d, history_old, p_old_quad, evol_old,
                  fixed_u_idx, fixed_u_vals, fixed_p_idx, fixed_p_vals, q_m):
        def cond(state):
            err = state[4]
            s = state[6]
            return (s < max_stagger) & (err > ms_tol)

        def body(state):
            (su, sp, sd, hist, err, field_errors, s,
             x_memory, g_memory, r_memory, memory_count,
             scale_u_ref, scale_p_ref, scale_d_ref,
             inner_ratio, inner_ok) = state
            del err, field_errors, inner_ratio, inner_ok
            d_quad = fe_u.to_quad(sd)              # (ncell,nq,1)

            # ---- coupled (u,p) block: phase field frozen ----
            # Figure 3 of the paper solves the displacement and pressure
            # equations together before renewing H. The two fields are separate
            # subproblems here, so converge their block Gauss-Seidel map to
            # ``up_tol``. Renewing H after a single u-then-p sweep changes the
            # path-dependent crack-jump solution and can suppress one tip.
            def up_cond(up_state):
                _, _, up_error, up_it, _, _, up_ok = up_state
                return (up_it < up_max_iter) & (up_error > up_tol) & up_ok

            def up_body(up_state):
                up_u, up_p, _, up_it, _, _, _ = up_state

                p_quad = fe_p.to_quad(up_p)
                iv_u = [d_quad, p_quad, field_u.sigma0_quad]
                u0 = up_u.reshape(-1).at[fixed_u_idx].set(fixed_u_vals)
                r0, V0 = assemble_device(field_u, u0, iv_u)
                R0 = r0.at[fixed_u_idx].set(0.0)
                u_flat, ratio_u_i, _, ok_u_i = newton_u(
                    u0, R0, V0, iv_u, fixed_u_idx, fixed_u_vals)
                u_new = vector_project(u_flat.reshape(Nu, 2))

                dq = d_quad[..., 0]
                mob, coeff_S, rho_a_chiR = fluid_coeffs(dq, dt)
                evol = field_u.compute_evol_quad(u_new)
                rhs = coeff_S * p_old_quad + q_m \
                    - rho_a_chiR * (evol - evol_old) / dt
                iv_p = [mob, coeff_S, rhs]
                p0 = up_p.reshape(-1).at[fixed_p_idx].set(fixed_p_vals)
                r0p, V0p = assemble_device(field_p, p0, iv_p)
                R0p = r0p.at[fixed_p_idx].set(0.0)
                p_flat, ratio_p_i, _, ok_p_i = newton_p(
                    p0, R0p, V0p, iv_p, fixed_p_idx, fixed_p_vals)
                p_new = scalar_project(p_flat.reshape(Np, 1))

                scale_u = np.maximum(
                    np.maximum(np.linalg.norm(up_u), np.linalg.norm(u_new)),
                    1e-12)
                scale_p = np.maximum(
                    np.maximum(np.linalg.norm(up_p), np.linalg.norm(p_new)),
                    1.0)
                up_error = np.maximum(
                    np.linalg.norm(u_new - up_u) / scale_u,
                    np.linalg.norm(p_new - up_p) / scale_p)
                up_ok = ok_u_i & ok_p_i & np.isfinite(up_error)
                return (u_new, p_new, up_error, up_it + 1,
                        ratio_u_i, ratio_p_i, up_ok)

            up_init = (su, sp, np.array(np.inf), np.array(0),
                       np.array(np.inf), np.array(np.inf), np.array(True))
            (su_map, sp_map, up_error, up_iterations,
             ratio_u, ratio_p, up_inner_ok) = jax.lax.while_loop(
                 up_cond, up_body, up_init)
            del up_iterations
            up_converged = up_inner_ok & np.isfinite(up_error) \
                & (up_error <= up_tol)

            # Preserve the fast Anderson/full-map iteration initially. If a
            # crack jump has not settled after ``relax_after`` sweeps, recover
            # the delayed damping used by the earlier robust implementation.
            # The raw maps remain available below for the exact fixed-point
            # residual, so relaxation cannot falsely declare convergence.
            omega = np.where(s >= relax_after, stagger_relax, 1.0)
            su_coupled = su + omega * (su_map - su)
            sp_coupled = sp + omega * (sp_map - sp)

            # ---- history (AT2, includes sigma0:eps) ----
            # Equation (20) takes a maximum over accepted physical time.  Use
            # the fixed preceding-time history in every nonlinear trial; using
            # ``hist`` here would irreversibly retain overshoots from states
            # that never satisfied the coupled fixed-point equations.
            hist_map = cell_project(
                field_u.compute_history(su_coupled, history_old))

            # ---- phi-solve: linear AT2 + clip ----
            sd0 = sd.reshape(-1)
            r0d, V0d = assemble_device(field_d, sd0, [hist_map])
            sd_flat, ratio_d, _, ok_d = newton_d(
                sd0, r0d, V0d, [hist_map], empty_idx, empty_vals)
            sd_map = scalar_project(
                np.clip(sd_flat.reshape(Nd, 1), 0., 1.))
            sd_coupled = np.clip(sd + omega * (sd_map - sd), 0., 1.)

            du = su_map - su
            dp = sp_map - sp
            dd = sd_map - sd
            first_scale_u = np.maximum(
                np.maximum(np.linalg.norm(su), np.linalg.norm(su_map)), 1e-12)
            first_scale_p = np.maximum(
                np.maximum(np.linalg.norm(sp), np.linalg.norm(sp_map)), 1.0)
            first_scale_d = np.maximum(
                np.maximum(np.linalg.norm(sd), np.linalg.norm(sd_map)), 1e-12)
            scale_u_ref = np.where(s == 0, first_scale_u, scale_u_ref)
            scale_p_ref = np.where(s == 0, first_scale_p, scale_p_ref)
            scale_d_ref = np.where(s == 0, first_scale_d, scale_d_ref)
            eu = np.linalg.norm(du) / scale_u_ref
            ep = np.linalg.norm(dp) / scale_p_ref
            ed = np.linalg.norm(dd) / scale_d_ref
            e = np.maximum(np.maximum(eu, ep), ed)
            map_inner_ok = up_converged & ok_d
            map_inner_ratios = np.array([ratio_u, ratio_p, ratio_d])
            validated = map_inner_ok & np.isfinite(e) & (e <= ms_tol)

            # Anderson acceleration of the coupled Gauss-Seidel fixed-point map.
            # Residual blocks are nondimensionalized separately so pressure does
            # not dominate displacement and phase field solely by its SI scale.
            x = np.concatenate((su.reshape(-1), sp.reshape(-1), sd.reshape(-1)))
            g = np.concatenate((su_coupled.reshape(-1),
                                sp_coupled.reshape(-1),
                                sd_coupled.reshape(-1)))
            r_scaled = np.concatenate((du.reshape(-1) / scale_u_ref,
                                       dp.reshape(-1) / scale_p_ref,
                                       dd.reshape(-1) / scale_d_ref))
            slot = s % anderson_depth
            x_memory_updated = x_memory.at[slot].set(x)
            g_memory_updated = g_memory.at[slot].set(g)
            r_memory_updated = r_memory.at[slot].set(r_scaled)
            x_memory = np.where(map_inner_ok, x_memory_updated,
                                np.zeros_like(x_memory))
            g_memory = np.where(map_inner_ok, g_memory_updated,
                                np.zeros_like(g_memory))
            r_memory = np.where(map_inner_ok, r_memory_updated,
                                np.zeros_like(r_memory))
            memory_count = np.where(
                map_inner_ok, np.minimum(memory_count + 1, anderson_depth), 0)

            valid = memory_rows < memory_count
            valid_outer = valid[:, None] & valid[None, :]
            gram_raw = r_memory @ r_memory.T
            gram = np.where(valid_outer, gram_raw, 0.0)
            trace_scale = np.trace(gram) / np.maximum(memory_count, 1)
            ridge = anderson_regularization * np.maximum(trace_scale, 1e-16)
            system = gram + np.diag(np.where(valid, ridge, 1.0))
            rhs_coeff = valid.astype(np.float64)
            coeff_raw = np.linalg.solve(system, rhs_coeff)
            coeff_sum = np.sum(coeff_raw)
            coeff = coeff_raw / np.where(np.abs(coeff_sum) > 1e-14,
                                         coeff_sum, 1.0)
            x_average = coeff @ x_memory
            g_average = coeff @ g_memory
            accelerated = (1.0 - anderson_beta) * x_average \
                + anderson_beta * g_average
            predicted_residual = coeff @ r_memory
            prediction_safe = np.linalg.norm(predicted_residual) \
                <= 1.05 * np.linalg.norm(r_scaled)
            coeff_safe = np.all(np.isfinite(coeff)) \
                & (np.max(np.abs(coeff)) < 50.0)
            use_anderson = (s >= anderson_start) & (s < relax_after) \
                & (memory_count >= 2) \
                & map_inner_ok & coeff_safe & prediction_safe \
                & np.all(np.isfinite(accelerated))
            candidate = np.where(use_anderson, accelerated, g)
            candidate = np.where(map_inner_ok, candidate, x)

            # e is the residual of the exact current state x. Once it satisfies
            # the requested tolerance, retain x. Returning candidate here would
            # accept an Anderson extrapolation whose residual was never tested.
            x_new = np.where(validated, x, candidate)

            u_end = Nu * 2
            p_end = u_end + Np
            su_new = vector_project(
                x_new[:u_end].at[fixed_u_idx].set(fixed_u_vals).reshape(Nu, 2))
            sp_new = scalar_project(
                x_new[u_end:p_end].at[fixed_p_idx].set(fixed_p_vals).reshape(Np, 1))
            sd_new = scalar_project(
                np.clip(x_new[p_end:].reshape(Nd, 1), 0.0, 1.0))
            # Commit history for the state actually retained.  If an inner
            # solve failed, preserve the last valid trial reference; it is not
            # returned as a converged physical-time state in that case.
            hist_accepted = cell_project(
                field_u.compute_history(su_new, history_old))
            hist_new = cell_project(
                np.where(map_inner_ok, hist_accepted, hist))
            error_out = np.where(map_inner_ok, e, np.inf)
            return (su_new, sp_new, sd_new, hist_new, error_out,
                    np.array([eu, ep, ed]), s + 1,
                    x_memory, g_memory, r_memory, memory_count,
                    scale_u_ref, scale_p_ref, scale_d_ref,
                    map_inner_ratios, map_inner_ok)

        sol_u = vector_project(sol_u)
        sol_p = scalar_project(sol_p)
        sol_d = scalar_project(sol_d)
        history_old = cell_project(history_old)
        init = (sol_u, sol_p, sol_d, history_old, np.array(1e30),
                np.full((3,), np.inf), np.array(0),
                np.zeros((anderson_depth, total_dofs)),
                np.zeros((anderson_depth, total_dofs)),
                np.zeros((anderson_depth, total_dofs)), np.array(0),
                np.array(0.0), np.array(0.0), np.array(0.0),
                np.full((3,), np.inf), np.array(False))
        state = jax.lax.while_loop(cond, body, init)
        su, sp, sd, hist, err, field_errors, s = state[:7]
        inner_ratio, inner_ok = state[-2:]
        # advance time-lagged quantities (keep p_old as per-quad scalars)
        p_old_new = fe_p.to_quad(sp)[..., 0]
        evol_new = field_u.compute_evol_quad(su)
        return (su, sp, sd, hist, p_old_new, evol_new, err, field_errors, s,
                inner_ratio, inner_ok)

    field_u._time_stepper = time_step
    return time_step

# =============================================================================
# Plotting
# =============================================================================
def plot_field(points, cells, fname, title, cell_vals, cmap, clim, label):
    points = onp.asarray(points); cells = onp.asarray(cells)
    polys = points[cells].astype(float)
    fig, ax = plt.subplots(figsize=(6, 6))
    pc = PolyCollection(polys, array=onp.asarray(cell_vals), cmap=cmap,
                        edgecolors='face', linewidths=0.0)
    if clim is not None:
        pc.set_clim(*clim)
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04); cb.set_label(label)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_aspect('equal'); ax.set_title(title)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def node_to_cell(vals, cells):
    return onp.asarray(vals).reshape(-1)[onp.asarray(cells)].mean(axis=1)

# =============================================================================
# Driver
# =============================================================================
def main():
    t_wall0 = time.time()

    # A Cartesian spacing of 0.05 m produces TRI3 diagonal edges of 0.0707 m.
    # NX=NY=284 keeps the *actual* longest edge below the paper's h_max=0.05 m.
    NX = int(os.environ.get('PFF_NX', '284'))
    NY = int(os.environ.get('PFF_NY', '284'))
    NT = int(os.environ.get('PFF_NT', '100'))
    DT = float(os.environ.get('PFF_DT', '0.05'))
    Q_F = float(os.environ.get('PFF_QF', '10.0'))
    SAVE_EVERY = int(os.environ.get('PFF_SAVE_EVERY', '5'))
    # Crack jumps can require substantially more fixed-point iterations than
    # smooth loading steps.  More headroom does not slow easy steps because the
    # JAX loop exits as soon as MS_TOL is met.
    MAX_STAGGER = int(os.environ.get('PFF_MAX_STAGGER', '500'))
    MS_TOL = float(os.environ.get('PFF_MS_TOL', '1e-4'))
    ANDERSON_DEPTH = int(os.environ.get('PFF_ANDERSON_DEPTH', '5'))
    ANDERSON_START = int(os.environ.get('PFF_ANDERSON_START', '2'))
    ANDERSON_BETA = float(os.environ.get('PFF_ANDERSON_BETA', '1.0'))
    ANDERSON_REG = float(os.environ.get('PFF_ANDERSON_REG', '1e-8'))
    UP_MAX_ITER = int(os.environ.get('PFF_UP_MAX_ITER', '50'))
    UP_TOL = float(os.environ.get('PFF_UP_TOL', '1e-6'))
    STAGGER_RELAX = float(os.environ.get('PFF_STAGGER_RELAX', '0.5'))
    RELAX_AFTER = int(os.environ.get('PFF_RELAX_AFTER', '20'))
    QUADRATURE_ORDER = int(os.environ.get('PFF_QUADRATURE_ORDER', '2'))
    RESUME = os.environ.get('PFF_RESUME', '0').lower() in ('1', 'true', 'yes', 'on')
    MESH_KIND_REQUESTED = os.environ.get(
        'PFF_MESH_KIND', 'structured_checkerboard').lower()
    MESH_H_TARGET = float(os.environ.get('PFF_MESH_H', '0.035'))
    symmetry_default = '1' if NOTCH_LAYOUT == 'double_horizontal' else '0'
    ENFORCE_REFLECTION_SYMMETRY = os.environ.get(
        'PFF_ENFORCE_REFLECTION_SYMMETRY', symmetry_default
    ).lower() in ('1', 'true', 'yes', 'on')
    if MESH_KIND_REQUESTED not in ('structured_checkerboard', 'unstructured_gmsh'):
        raise ValueError(
            'PFF_MESH_KIND must be structured_checkerboard or unstructured_gmsh, '
            f'got {MESH_KIND_REQUESTED!r}')
    if ANDERSON_DEPTH < 2:
        raise ValueError(f'PFF_ANDERSON_DEPTH must be at least 2, got {ANDERSON_DEPTH}')
    if ANDERSON_START < 0:
        raise ValueError(f'PFF_ANDERSON_START must be nonnegative, got {ANDERSON_START}')
    if not (0.0 < ANDERSON_BETA <= 1.0):
        raise ValueError(f'PFF_ANDERSON_BETA must be in (0,1], got {ANDERSON_BETA}')
    if ANDERSON_REG <= 0.0:
        raise ValueError(f'PFF_ANDERSON_REG must be positive, got {ANDERSON_REG}')
    if UP_MAX_ITER < 1:
        raise ValueError(f'PFF_UP_MAX_ITER must be positive, got {UP_MAX_ITER}')
    if UP_TOL <= 0.0:
        raise ValueError(f'PFF_UP_TOL must be positive, got {UP_TOL}')
    if not (0.0 < STAGGER_RELAX <= 1.0):
        raise ValueError(
            f'PFF_STAGGER_RELAX must be in (0,1], got {STAGGER_RELAX}')
    if RELAX_AFTER < 1:
        raise ValueError(f'PFF_RELAX_AFTER must be positive, got {RELAX_AFTER}')
    ELE_TYPE = os.environ.get('PFF_ELE_TYPE', 'QUAD4').upper()
    if ELE_TYPE not in ('TRI3', 'QUAD4'):
        raise ValueError(f'PFF_ELE_TYPE must be TRI3 or QUAD4, got {ELE_TYPE!r}')
    if ELE_TYPE == 'TRI3' and QUADRATURE_ORDER < 2:
        raise ValueError(
            'TRI3 requires PFF_QUADRATURE_ORDER>=2 for the pressure-storage and '
            'phase-reaction matrices; one-point integration is not valid here.')

    if MESH_KIND_REQUESTED == 'unstructured_gmsh':
        if ELE_TYPE != 'TRI3':
            raise ValueError('PFF_MESH_KIND=unstructured_gmsh requires PFF_ELE_TYPE=TRI3')
        mesh_kind = 'unstructured_gmsh'
        mesh_rule = 'gmsh_quadrant_reflection_v1'
        mesh_path = os.path.join(output_dir, 'mesh.npz')
        loaded_mesh = False
        if os.path.exists(mesh_path):
            with onp.load(mesh_path, allow_pickle=False) as saved_mesh:
                required_mesh = {
                    'points', 'cells', 'mesh_kind', 'mesh_rule', 'mesh_h_target'}
                if required_mesh.issubset(saved_mesh.files):
                    saved_kind = str(saved_mesh['mesh_kind'].item())
                    saved_rule = str(saved_mesh['mesh_rule'].item())
                    saved_h = float(saved_mesh['mesh_h_target'])
                    if (saved_kind == mesh_kind
                            and saved_rule == mesh_rule
                            and onp.isclose(saved_h, MESH_H_TARGET)):
                        pts = onp.asarray(saved_mesh['points'], dtype=float)
                        cells = onp.asarray(saved_mesh['cells'], dtype=onp.int32)
                        loaded_mesh = True
        if not loaded_mesh:
            pts, cells = unstructured_rectangle_mesh(MESH_H_TARGET)
            mesh_tmp_path = mesh_path + '.tmp.npz'
            onp.savez(mesh_tmp_path, points=pts, cells=cells,
                      mesh_kind=mesh_kind, mesh_rule=mesh_rule,
                      mesh_h_target=MESH_H_TARGET)
            os.replace(mesh_tmp_path, mesh_path)
        hx = MESH_H_TARGET
        hy = MESH_H_TARGET
        mesh_description = (
            f'Gmsh unstructured target_h={MESH_H_TARGET:.4f} m, '
            f'mesh_file={os.path.basename(mesh_path)}')
    else:
        if NX % 2 != 0 or NY % 2 != 0:
            raise ValueError(
                f'NX and NY must be even so the mesh contains the notch midpoint; '
                f'got NX={NX}, NY={NY}.')
        hx = Lx / NX
        hy = Ly / NY
        pts, cells = rectangle_mesh(NX, NY, ELE_TYPE)
        mesh_kind = 'structured_checkerboard'
        mesh_rule = 'alternating_diagonal_checkerboard_v1'
        mesh_description = f'NX={NX},NY={NY}, grid=({hx:.4f},{hy:.4f}) m'

    h_max = mesh_max_edge(pts, cells)
    mesh_digest = mesh_fingerprint(pts, cells)
    if ENFORCE_REFLECTION_SYMMETRY:
        symmetry_node_maps, symmetry_cell_maps = reflection_symmetry_maps(
            pts, cells)
        symmetry_rule = 'scalar_even_vector_reflection_xy_v1'
    else:
        symmetry_node_maps = None
        symmetry_cell_maps = None
        symmetry_rule = 'off'
    if NOTCH_LAYOUT == 'single':
        source_rule = 'fixed_initial_notch_band_v1'
    else:
        source_rule = (
            'fixed_double_horizontal_rounded_notch_band_'
            f'spacing_{NOTCH_SPACING:g}_v2')
    profile_metadata = {}
    if STRESS_PROFILE == 'fig19_linear':
        profile_metadata = dict(
            stress_profile=STRESS_PROFILE,
            gravity_body_force_y=GRAVITY_BODY_FORCE_Y,
            sigx_top_mpa=FIG19_SIGX_TOP_MPA,
            sigx_bottom_mpa=FIG19_SIGX_BOTTOM_MPA,
            sigy_top_mpa=FIG19_SIGY_TOP_MPA,
            sigy_bottom_mpa=FIG19_SIGY_BOTTOM_MPA)
    mesh, field_d, field_u, field_p = build_fields(
        pts, cells, ELE_TYPE, quadrature_order=QUADRATURE_ORDER)
    if ELE_TYPE == 'TRI3' and h_max > 0.05 + 1e-12:
        print(f'WARNING: actual TRI3 h_max={h_max:.6f} m exceeds the paper value '
              '0.05 m; this run is diagnostic, not paper-resolution validation.',
              flush=True)

    centroids = onp.asarray(pts)[onp.asarray(cells)].mean(axis=1)
    H0_cell = compute_H0(centroids)
    q_m_cell = compute_source(centroids, Q_F)
    nq = field_u.num_quads
    history = onp.broadcast_to(H0_cell[:, None], (len(cells), nq)).copy()
    q_m = np.asarray(onp.broadcast_to(q_m_cell[:, None], (len(cells), nq)).copy())

    sol_u = np.zeros((len(pts), 2))
    sol_p = np.zeros((len(pts), 1))
    sol_d = np.zeros((len(pts), 1))
    p_old_quad = np.zeros((len(cells), nq))
    evol_old = np.zeros((len(cells), nq))
    history = np.asarray(history)

    # Dirichlet structure (constant across steps)
    fixed_u_idx, fixed_u_vals = dirichlet_data(field_u)
    fixed_p_idx, fixed_p_vals = dirichlet_data(field_p)

    n_dofs = 4 * len(pts)
    theta_deg = onp.rad2deg(THETA)
    notch_description = (
        f"one notch, len={2*lc} m @ theta={theta_deg:.1f} deg"
        if NOTCH_LAYOUT == 'single' else
        f"two horizontal notches, len={2*lc} m, spacing={NOTCH_SPACING:g} m")
    if STRESS_PROFILE == 'fig19_linear':
        stress_description = (
            'Fig.19 linear compression: '
            f'sigx(top,bottom)=({FIG19_SIGX_TOP_MPA:g},'
            f'{FIG19_SIGX_BOTTOM_MPA:g}) MPa, '
            f'sigy(top,bottom)=({FIG19_SIGY_TOP_MPA:g},'
            f'{FIG19_SIGY_BOTTOM_MPA:g}) MPa, '
            f'body_y={GRAVITY_BODY_FORCE_Y:g} N/m^3')
    else:
        stress_description = (
            f'sig0=({sig_x0/1e6:.2f},{sig_y0/1e6:.2f}) MPa '
            f'(ratio sigx0/sigy0={sig_x0/sig_y0:.2f})')
    print(f"Axion | coupled hydraulic fracture (Zhou et al. 2023). out={output_dir.split('/')[-1]}. "
          f"{len(pts)} nodes, {len(cells)} {ELE_TYPE} cells "
          f"({mesh_description}, "
          f"actual h_max={h_max:.4f} m, l0={l} m, l0/h_max={l/h_max:.2f}, "
          f"quadrature_order={QUADRATURE_ORDER}). "
          f"E={E/1e9:.1f} GPa, nu={nu:.3f}, Gc={G_c}. "
          f"layout={NOTCH_LAYOUT}: {notch_description}. "
          f"{stress_description}, mode={_SIG0_MODE}. "
          f"dt={DT}s, NT={NT} (t_end={NT*DT:.2f}s), q_F={Q_F}, "
          f"source half-width={SOURCE_HALF_WIDTH:g} m. "
          f"stagger: max={MAX_STAGGER}, tol={MS_TOL:g}; "
          f"Anderson: depth={ANDERSON_DEPTH}, start={ANDERSON_START}, "
          f"beta={ANDERSON_BETA:g}, reg={ANDERSON_REG:g}; "
          f"coupled (u,p): max={UP_MAX_ITER}, tol={UP_TOL:g}; "
          f"relax={STAGGER_RELAX:g} after {RELAX_AFTER}; "
          f"reflection_symmetry={symmetry_rule}; resume={RESUME}.",
          flush=True)

    time_step = get_time_stepper(field_u, field_d, field_p, DT,
                                 max_stagger=MAX_STAGGER, ms_tol=MS_TOL,
                                 anderson_depth=ANDERSON_DEPTH,
                                 anderson_start=ANDERSON_START,
                                 anderson_beta=ANDERSON_BETA,
                                 anderson_regularization=ANDERSON_REG,
                                 up_max_iter=UP_MAX_ITER,
                                 up_tol=UP_TOL,
                                 stagger_relax=STAGGER_RELAX,
                                 relax_after=RELAX_AFTER,
                                 symmetry_node_maps=symmetry_node_maps,
                                 symmetry_cell_maps=symmetry_cell_maps)

    history_path = os.path.join(output_dir, 'notch_history.npz')
    checkpoint_path = os.path.join(output_dir, 'state_checkpoint.npz')
    lengths = []
    start_step = 0

    def solver_array(value):
        """Convert persisted fields to the fp64 dtype used by this solver.

        meshio commonly reads legacy VTU point fields as float32. Mixing those
        fields with the fp64 constitutive/assembly arrays changes a Newton loop
        carry from float32 to float64, which JAX correctly rejects.
        """
        return np.asarray(value, dtype=np.float64)

    def save_vtk(n, t):
        axion.write_vtu(os.path.join(vtk_dir, f'step_{n:04d}.vtu'), mesh,
                        point_data={'sol': onp.asarray(sol_d), 'u': onp.asarray(sol_u),
                                    'p': onp.asarray(sol_p)},
                        cell_data={'history': onp.mean(onp.asarray(history), axis=1)})
        plot_field(pts, cells, os.path.join(output_dir, f'phi_{n:04d}.png'),
                   f'phase field  phi   (t = {t:.2f} s)',
                   node_to_cell(sol_d, cells), 'inferno', (0., 1.), 'phi')

    def measure_state():
        dv = onp.asarray(sol_d).reshape(-1)
        broken = dv > 0.9
        pb = onp.asarray(pts)[broken]
        upper_increment = 0.0
        lower_increment = 0.0
        if pb.size:
            distance_to_centers = onp.linalg.norm(
                pb[:, None, :] - NOTCH_CENTERS[None, :, :], axis=2)
            half_len = float(onp.min(distance_to_centers, axis=1).max())
            if STRESS_PROFILE == 'fig19_linear':
                upper_increment = max(float(pb[:, 1].max() - lc), 0.0)
                lower_increment = max(float(-lc - pb[:, 1].min()), 0.0)
        else:
            half_len = 0.0
        all_points = onp.asarray(pts)
        p_values = onp.asarray(sol_p).reshape(-1)
        notch_node_ids = [
            onp.argmin(onp.sum((all_points - center) ** 2, axis=1))
            for center in NOTCH_CENTERS
        ]
        p_notch = float(onp.mean(p_values[notch_node_ids]))
        return dv, half_len, p_notch, upper_increment, lower_increment

    def save_history(completed):
        onp.savez(
            history_path,
            data=onp.asarray(lengths), NX=NX, NY=NY, nt=NT, dt=DT, q_F=Q_F,
            element_type=ELE_TYPE, quadrature_order=QUADRATURE_ORDER,
            mesh_kind=mesh_kind, mesh_h_max=h_max, grid_hx=hx, grid_hy=hy,
            mesh_rule=mesh_rule, mesh_fingerprint=mesh_digest,
            reflection_symmetry_rule=symmetry_rule,
            l0=l, history_seed_factor=H0_B,
            source_rule=source_rule, source_half_width=SOURCE_HALF_WIDTH,
            notch_layout=NOTCH_LAYOUT, notch_spacing=NOTCH_SPACING,
            notch_centers=NOTCH_CENTERS,
            pressure_measure='mean_at_notch_centers',
            theta_deg=theta_deg, sig_x0_mpa=abs(sig_x0) / 1e6,
            sig_y0_mpa=abs(sig_y0) / 1e6, sig0_sign=SIG0_SIGN,
            sig0_mode=_SIG0_MODE, history_rule=HISTORY_RULE,
            solver_rule=SOLVER_RULE,
            ms_tol=MS_TOL, anderson_depth=ANDERSON_DEPTH,
            anderson_start=ANDERSON_START, anderson_beta=ANDERSON_BETA,
            anderson_regularization=ANDERSON_REG,
            up_max_iter=UP_MAX_ITER, up_tol=UP_TOL,
            stagger_relax=STAGGER_RELAX, relax_after=RELAX_AFTER,
            completed=bool(completed), **profile_metadata)

    def checkpoint_metadata():
        metadata = dict(
            NX=NX, NY=NY, nt=NT, dt=DT, q_F=Q_F, element_type=ELE_TYPE,
            quadrature_order=QUADRATURE_ORDER, mesh_kind=mesh_kind,
            mesh_h_max=h_max, grid_hx=hx, grid_hy=hy, l0=l,
            history_seed_factor=H0_B, source_rule=source_rule,
            source_half_width=SOURCE_HALF_WIDTH,
            theta_deg=theta_deg, sig_x0_mpa=abs(sig_x0) / 1e6,
            sig_y0_mpa=abs(sig_y0) / 1e6, sig0_sign=SIG0_SIGN,
            sig0_mode=_SIG0_MODE, history_rule=HISTORY_RULE,
            solver_rule=SOLVER_RULE, ms_tol=MS_TOL,
            anderson_depth=ANDERSON_DEPTH, anderson_start=ANDERSON_START,
            anderson_beta=ANDERSON_BETA,
            anderson_regularization=ANDERSON_REG,
            up_max_iter=UP_MAX_ITER, up_tol=UP_TOL,
            stagger_relax=STAGGER_RELAX, relax_after=RELAX_AFTER)
        # Keep legacy single-notch checkpoints restart-compatible. The source rule
        # already distinguishes old single-notch data; Fig. 18 additionally needs
        # its spacing to be an exact part of restart validation.
        if NOTCH_LAYOUT != 'single':
            metadata.update(notch_layout=NOTCH_LAYOUT,
                            notch_spacing=NOTCH_SPACING,
                            reflection_symmetry_rule=symmetry_rule)
        if mesh_kind == 'unstructured_gmsh':
            metadata.update(mesh_h_target=MESH_H_TARGET,
                            mesh_rule=mesh_rule,
                            mesh_fingerprint=mesh_digest)
        metadata.update(profile_metadata)
        return metadata

    def save_checkpoint(step):
        """Atomically save the complete converged state needed for an exact resume."""
        tmp_path = checkpoint_path + '.tmp.npz'
        onp.savez(
            tmp_path, step=int(step), sol_u=onp.asarray(sol_u),
            sol_p=onp.asarray(sol_p), sol_d=onp.asarray(sol_d),
            history=onp.asarray(history), p_old_quad=onp.asarray(p_old_quad),
            evol_old=onp.asarray(evol_old), lengths=onp.asarray(lengths),
            **checkpoint_metadata())
        os.replace(tmp_path, checkpoint_path)

    def validate_restart_metadata(saved, source):
        expected = checkpoint_metadata()
        missing = [name for name in expected if name not in saved.files]
        if missing:
            raise RuntimeError(f'{source} lacks restart metadata: {missing}')

        mismatches = []
        for name in ('NX', 'NY', 'nt', 'quadrature_order', 'anderson_depth',
                     'anderson_start', 'up_max_iter', 'relax_after'):
            if int(saved[name]) != int(expected[name]):
                mismatches.append(f'{name}={int(saved[name])} (expected {expected[name]})')
        for name in ('dt', 'q_F', 'mesh_h_max', 'grid_hx', 'grid_hy', 'l0',
                     'history_seed_factor', 'source_half_width', 'theta_deg',
                     'sig_x0_mpa', 'sig_y0_mpa', 'sig0_sign', 'ms_tol',
                     'anderson_beta', 'anderson_regularization',
                     'up_tol', 'stagger_relax'):
            if not onp.isclose(float(saved[name]), float(expected[name])):
                mismatches.append(
                    f'{name}={float(saved[name]):g} (expected {expected[name]:g})')
        if 'mesh_h_target' in expected:
            if not onp.isclose(float(saved['mesh_h_target']),
                               float(expected['mesh_h_target'])):
                mismatches.append(
                    f"mesh_h_target={float(saved['mesh_h_target']):g} "
                    f"(expected {float(expected['mesh_h_target']):g})")
        if 'notch_spacing' in expected:
            if not onp.isclose(float(saved['notch_spacing']),
                               float(expected['notch_spacing'])):
                mismatches.append(
                    f"notch_spacing={float(saved['notch_spacing']):g} "
                    f"(expected {float(expected['notch_spacing']):g})")
        for name in ('element_type', 'mesh_kind', 'source_rule', 'sig0_mode',
                     'history_rule', 'solver_rule'):
            value = str(saved[name].item())
            if value != str(expected[name]):
                mismatches.append(f'{name}={value!r} (expected {expected[name]!r})')
        for name in ('gravity_body_force_y', 'sigx_top_mpa',
                     'sigx_bottom_mpa', 'sigy_top_mpa',
                     'sigy_bottom_mpa'):
            if name in expected and not onp.isclose(
                    float(saved[name]), float(expected[name])):
                mismatches.append(
                    f'{name}={float(saved[name]):g} '
                    f'(expected {float(expected[name]):g})')
        if 'stress_profile' in expected:
            value = str(saved['stress_profile'].item())
            if value != str(expected['stress_profile']):
                mismatches.append(
                    f'stress_profile={value!r} '
                    f'(expected {expected["stress_profile"]!r})')
        if 'mesh_fingerprint' in expected:
            value = str(saved['mesh_fingerprint'].item())
            if value != str(expected['mesh_fingerprint']):
                mismatches.append('mesh_fingerprint does not match the current mesh')
        if 'mesh_rule' in expected:
            value = str(saved['mesh_rule'].item())
            if value != str(expected['mesh_rule']):
                mismatches.append(
                    f'mesh_rule={value!r} (expected {expected["mesh_rule"]!r})')
        if 'reflection_symmetry_rule' in expected:
            value = str(saved['reflection_symmetry_rule'].item())
            if value != str(expected['reflection_symmetry_rule']):
                mismatches.append(
                    f'reflection_symmetry_rule={value!r} '
                    f'(expected {expected["reflection_symmetry_rule"]!r})')
        if 'notch_layout' in expected:
            value = str(saved['notch_layout'].item())
            if value != str(expected['notch_layout']):
                mismatches.append(
                    f"notch_layout={value!r} (expected {expected['notch_layout']!r})")
        if mismatches:
            raise RuntimeError(f'{source} does not match this run:\n  - '
                               + '\n  - '.join(mismatches))

    def load_exact_checkpoint():
        nonlocal sol_u, sol_p, sol_d, history, p_old_quad, evol_old
        nonlocal lengths, start_step
        if not os.path.exists(checkpoint_path):
            return False
        with onp.load(checkpoint_path, allow_pickle=False) as saved:
            saved_rule = (str(saved['history_rule'].item())
                          if 'history_rule' in saved.files else None)
            if saved_rule != HISTORY_RULE:
                print(
                    f'  checkpoint uses obsolete history update {saved_rule!r}; '
                    'starting this case fresh', flush=True)
                return False
            saved_solver_rule = (str(saved['solver_rule'].item())
                                 if 'solver_rule' in saved.files else None)
            if saved_solver_rule != SOLVER_RULE:
                print(
                    f'  checkpoint uses obsolete nonlinear solver rule '
                    f'{saved_solver_rule!r}; starting this case fresh',
                    flush=True)
                return False
            saved_source_rule = (str(saved['source_rule'].item())
                                 if 'source_rule' in saved.files else None)
            if saved_source_rule != source_rule:
                print(
                    f'  checkpoint uses obsolete notch source rule '
                    f'{saved_source_rule!r}; starting this case fresh',
                    flush=True)
                return False
            if NOTCH_LAYOUT == 'double_horizontal':
                saved_symmetry_rule = (
                    str(saved['reflection_symmetry_rule'].item())
                    if 'reflection_symmetry_rule' in saved.files else None)
                if saved_symmetry_rule != symmetry_rule:
                    print(
                        f'  checkpoint uses obsolete reflection-symmetry rule '
                        f'{saved_symmetry_rule!r}; starting this case fresh',
                        flush=True)
                    return False
            saved_mesh_kind = (str(saved['mesh_kind'].item())
                               if 'mesh_kind' in saved.files else None)
            if saved_mesh_kind != mesh_kind:
                print(
                    f'  checkpoint uses incompatible mesh {saved_mesh_kind!r}; '
                    f'starting fresh with {mesh_kind!r}', flush=True)
                return False
            if mesh_kind == 'unstructured_gmsh':
                saved_mesh_rule = (str(saved['mesh_rule'].item())
                                   if 'mesh_rule' in saved.files else None)
                if saved_mesh_rule != mesh_rule:
                    print(
                        f'  checkpoint uses obsolete mesh-generation rule '
                        f'{saved_mesh_rule!r}; starting this case fresh',
                        flush=True)
                    return False
            validate_restart_metadata(saved, checkpoint_path)
            required = ('step', 'sol_u', 'sol_p', 'sol_d', 'history',
                        'p_old_quad', 'evol_old', 'lengths')
            missing = [name for name in required if name not in saved.files]
            if missing:
                raise RuntimeError(f'{checkpoint_path} lacks state arrays: {missing}')
            step = int(saved['step'])
            su = onp.asarray(saved['sol_u'])
            sp = onp.asarray(saved['sol_p'])
            sd = onp.asarray(saved['sol_d'])
            hist = onp.asarray(saved['history'])
            pold = onp.asarray(saved['p_old_quad'])
            eold = onp.asarray(saved['evol_old'])
            saved_lengths = onp.asarray(saved['lengths'])

        expected_shapes = {
            'sol_u': (len(pts), 2), 'sol_p': (len(pts), 1),
            'sol_d': (len(pts), 1), 'history': (len(cells), nq),
            'p_old_quad': (len(cells), nq), 'evol_old': (len(cells), nq),
        }
        arrays = {'sol_u': su, 'sol_p': sp, 'sol_d': sd, 'history': hist,
                  'p_old_quad': pold, 'evol_old': eold}
        bad_shapes = [f'{name}: {value.shape} != {expected_shapes[name]}'
                      for name, value in arrays.items()
                      if value.shape != expected_shapes[name]]
        history_columns = 5 if STRESS_PROFILE == 'fig19_linear' else 3
        if bad_shapes or saved_lengths.shape != (step + 1, history_columns):
            details = bad_shapes + [
                f'lengths: {saved_lengths.shape} != '
                f'{(step + 1, history_columns)}']
            raise RuntimeError(f'Invalid state in {checkpoint_path}:\n  - '
                               + '\n  - '.join(details))
        if step < 0 or step > NT:
            raise RuntimeError(f'Invalid checkpoint step {step}; expected 0..{NT}')

        sol_u = solver_array(su); sol_p = solver_array(sp)
        sol_d = solver_array(sd); history = solver_array(hist)
        p_old_quad = solver_array(pold); evol_old = solver_array(eold)
        lengths = [tuple(row) for row in saved_lengths]
        start_step = step
        print(f'  resumed exact checkpoint at step {start_step}/{NT} '
              f'(t={start_step*DT:.2f} s)', flush=True)
        return True

    def load_legacy_vtk_checkpoint():
        """Recover older interrupted runs from their latest saved VTU snapshot.

        The history implementation is cellwise constant, so the VTU cell mean is
        the full history state. p_old and evol_old are reconstructed from p and u.
        Steps after the latest VTU are intentionally rerun.
        """
        nonlocal sol_u, sol_p, sol_d, history, p_old_quad, evol_old
        nonlocal lengths, start_step
        if not os.path.exists(history_path):
            return False
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, 'step_*.vtu')))
        if not vtk_files:
            return False

        with onp.load(history_path, allow_pickle=False) as saved:
            legacy_missing = [name for name in checkpoint_metadata()
                              if name not in saved.files]
            if legacy_missing:
                print(
                    f'  legacy output {os.path.basename(history_path)} predates '
                    'validated restart metadata; starting this case fresh '
                    f'(missing: {", ".join(legacy_missing)})',
                    flush=True)
                return False
            saved_source_rule = str(saved['source_rule'].item())
            if saved_source_rule != source_rule:
                print(
                    f'  legacy output uses obsolete notch source rule '
                    f'{saved_source_rule!r}; starting this case fresh',
                    flush=True)
                return False
            saved_mesh_kind = str(saved['mesh_kind'].item())
            if saved_mesh_kind != mesh_kind:
                print(
                    f'  legacy output uses incompatible mesh {saved_mesh_kind!r}; '
                    f'starting fresh with {mesh_kind!r}', flush=True)
                return False
            if mesh_kind == 'unstructured_gmsh':
                saved_mesh_rule = (str(saved['mesh_rule'].item())
                                   if 'mesh_rule' in saved.files else None)
                if saved_mesh_rule != mesh_rule:
                    print(
                        f'  legacy output uses obsolete mesh-generation rule '
                        f'{saved_mesh_rule!r}; starting this case fresh',
                        flush=True)
                    return False
            validate_restart_metadata(saved, history_path)
            if 'data' not in saved.files:
                raise RuntimeError(f'{history_path} has no data array')
            history_data = onp.asarray(saved['data'])
        history_columns = 5 if STRESS_PROFILE == 'fig19_linear' else 3
        if (history_data.ndim != 2
                or history_data.shape[1] != history_columns):
            raise RuntimeError(f'Invalid history shape in {history_path}: '
                               f'{history_data.shape}')

        last_history_step = len(history_data) - 1
        candidates = []
        for filename in vtk_files:
            stem = os.path.splitext(os.path.basename(filename))[0]
            try:
                step = int(stem.rsplit('_', 1)[1])
            except (IndexError, ValueError):
                continue
            if step <= last_history_step:
                candidates.append((step, filename))
        if not candidates:
            return False
        step, filename = max(candidates)

        import meshio
        saved_mesh = meshio.read(filename)
        cell_type = 'triangle' if ELE_TYPE == 'TRI3' else 'quad'
        saved_cells = saved_mesh.cells_dict.get(cell_type)
        if saved_cells is None:
            raise RuntimeError(f'{filename} has no {cell_type} cells')
        if not onp.allclose(saved_mesh.points[:, :2], onp.asarray(pts)):
            raise RuntimeError(f'{filename} points do not match the current mesh')
        if not onp.array_equal(saved_cells, onp.asarray(cells)):
            raise RuntimeError(f'{filename} connectivity does not match the current mesh')
        required_points = ('sol', 'u', 'p')
        missing = [name for name in required_points if name not in saved_mesh.point_data]
        if missing or 'history' not in saved_mesh.cell_data:
            raise RuntimeError(f'{filename} lacks restart fields: '
                               f'{missing + (["history"] if "history" not in saved_mesh.cell_data else [])}')

        su = onp.asarray(saved_mesh.point_data['u']).reshape(len(pts), 2)
        sp = onp.asarray(saved_mesh.point_data['p']).reshape(len(pts), 1)
        sd = onp.asarray(saved_mesh.point_data['sol']).reshape(len(pts), 1)
        hist_cell = onp.concatenate(saved_mesh.cell_data['history']).reshape(len(cells))
        hist = onp.broadcast_to(hist_cell[:, None], (len(cells), nq)).copy()

        sol_u = solver_array(su); sol_p = solver_array(sp)
        sol_d = solver_array(sd); history = solver_array(hist)
        p_old_quad = field_p.to_quad(sol_p)[..., 0]
        evol_old = field_u.compute_evol_quad(sol_u)
        lengths = [tuple(row) for row in history_data[:step + 1]]
        start_step = step
        next_action = (f'resuming at step {start_step + 1}/{NT}'
                       if start_step < NT else 'run is already complete')
        print(f'  recovered legacy VTU checkpoint {os.path.basename(filename)}; '
              f'{next_action}', flush=True)
        save_checkpoint(start_step)
        save_history(completed=False)
        return True

    resumed = False
    if RESUME:
        resumed = load_exact_checkpoint()
        if not resumed:
            resumed = load_legacy_vtk_checkpoint()

    if not resumed:
        # A fresh run must not inherit a completed flag or final snapshot from an
        # older parameter set. Clearing only this case's generated files makes an
        # aborted rerun visibly incomplete to the plotting scripts.
        stale_patterns = (
            os.path.join(vtk_dir, '*'),
            os.path.join(output_dir, 'phi_*.png'),
            history_path,
            checkpoint_path,
            checkpoint_path + '.tmp.npz',
        )
        for pattern in stale_patterns:
            for stale_file in glob.glob(pattern):
                if os.path.isfile(stale_file):
                    os.remove(stale_file)

        # Establish the paper's t=0 state before injection: the seeded notch is
        # equilibrated under ft=SIGMA0@n at p=0 and q_m=0.
        initial_state = get_initial_state_solver(
            field_u, field_d, max_stagger=MAX_STAGGER, ms_tol=MS_TOL,
            symmetry_node_maps=symmetry_node_maps,
            symmetry_cell_maps=symmetry_cell_maps)
        (sol_u, sol_d, history, init_err, init_s,
         init_inner_ratio, init_inner_ok) = initial_state(
             sol_u, sol_d, history, fixed_u_idx, fixed_u_vals)
        sol_d.block_until_ready()
        init_err_host = float(onp.asarray(init_err))
        init_s_host = int(onp.asarray(init_s))
        init_inner_ratios_host = onp.asarray(init_inner_ratio, dtype=float)
        init_inner_ok_host = bool(onp.asarray(init_inner_ok))
        if (not onp.isfinite(init_err_host) or init_err_host > MS_TOL
                or not init_inner_ok_host):
            raise RuntimeError(
                f't=0 notch equilibrium did not converge: iterations={init_s_host}, '
                f'fixed-point error={init_err_host:.3e}, '
                f'inner residual ratios (u,phi)={init_inner_ratios_host}, '
                f'tolerance={MS_TOL:.3e}')
        evol_old = field_u.compute_evol_quad(sol_u)
        print(f"  initial equilibrium  stagger={init_s_host:3d}  "
              f"error={init_err_host:.3e}  inner/tol(u,d)="
              f"({init_inner_ratios_host[0]:.1e},"
              f"{init_inner_ratios_host[1]:.1e})  "
              f"max d={float(onp.asarray(sol_d).max()):.3f}",
              flush=True)
        (dv, initial_half_len, initial_p_mid,
         initial_upper_increment, initial_lower_increment) = measure_state()
        if STRESS_PROFILE == 'fig19_linear':
            lengths = [(0.0, initial_half_len, initial_p_mid,
                        initial_upper_increment, initial_lower_increment)]
        else:
            lengths = [(0.0, initial_half_len, initial_p_mid)]
        start_step = 0
        save_checkpoint(0)
        save_history(completed=False)
        save_vtk(0, 0.0)

    for n in range(start_step + 1, NT + 1):
        t0 = time.time()
        (sol_u, sol_p, sol_d, history, p_old_quad, evol_old, err,
         field_errors, s, inner_ratio, inner_ok) = time_step(
             sol_u, sol_p, sol_d, history, p_old_quad, evol_old,
             fixed_u_idx, fixed_u_vals, fixed_p_idx, fixed_p_vals, q_m)
        sol_d.block_until_ready()
        err_host = float(onp.asarray(err))
        field_errors_host = onp.asarray(field_errors, dtype=float)
        s_host = int(onp.asarray(s))
        inner_ratios_host = onp.asarray(inner_ratio, dtype=float)
        inner_ok_host = bool(onp.asarray(inner_ok))
        if (not onp.isfinite(err_host) or err_host > MS_TOL
                or not onp.all(onp.isfinite(field_errors_host))
                or not inner_ok_host):
            save_history(completed=False)
            raise RuntimeError(
                f'time step {n}/{NT} did not converge: iterations={s_host}, '
                f'fixed-point error={err_host:.3e}, '
                f'field errors (u,p,phi)={field_errors_host}, '
                f'inner residual ratios (u,p,phi)={inner_ratios_host}, '
                f'tolerance={MS_TOL:.3e}. '
                'The last converged state is checkpointed; rerun with '
                'PFF_RESUME=1 and, if needed, a larger PFF_MAX_STAGGER.')

        # Crack half-length estimate: maximum radial distance from the centre
        # among broken nodes, valid for an inclined/deflecting fracture.
        dv, half_len, p_mid, upper_increment, lower_increment = measure_state()
        if STRESS_PROFILE == 'fig19_linear':
            lengths.append((n * DT, half_len, p_mid,
                            upper_increment, lower_increment))
        else:
            lengths.append((n * DT, half_len, p_mid))
        tip_message = (
            f"  tip increments(up,down)=({upper_increment:.3f},"
            f"{lower_increment:.3f}) m"
            if STRESS_PROFILE == 'fig19_linear' else '')
        print(f"  step {n:3d}/{NT}  t={n*DT:5.2f}s  stagger={s_host:3d}  "
              f"error={err_host:.2e} (u,p,d)=({field_errors_host[0]:.1e},"
              f"{field_errors_host[1]:.1e},{field_errors_host[2]:.1e})  "
              f"inner/tol(u,p,d)=({inner_ratios_host[0]:.1e},"
              f"{inner_ratios_host[1]:.1e},{inner_ratios_host[2]:.1e})  "
              f"half-len={half_len:6.3f} m  p_notch={p_mid/1e6:7.3f} MPa  "
              f"max d={dv.max():.3f}{tip_message}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        save_checkpoint(n)
        save_history(completed=(n == NT))
        if n % SAVE_EVERY == 0 or n == NT:
            save_vtk(n, n * DT)

    # Also repairs the completion flag if a previous process stopped after the
    # final atomic state checkpoint but before updating the lightweight history.
    save_history(completed=True)

    wall = time.time() - t_wall0
    if STRESS_PROFILE == 'fig19_linear':
        paper_reference = (
            'Figs.20-21 target: upper tip grows ~4 m while lower tip '
            'arrests near ~0.8 m by t=14 s')
    elif NOTCH_LAYOUT == 'double_horizontal':
        paper_reference = (
            'Fig.18 target: symmetric outward deflection from both notches at t=2.5 s')
    elif abs(theta_deg) > 1e-9:
        paper_reference = (
            'Fig.14 target: t=6 s path bends toward horizontal as sigx0/sigy0 increases')
    else:
        paper_reference = (
            'Fig.5 ref: half-len ~0.42/0.89/1.79 m at t=2.4/3.6/4.8 s')
    final_tip_summary = (
        f"  final tip increments (up,down) : "
        f"({lengths[-1][3]:.3f},{lengths[-1][4]:.3f}) m\n"
        if STRESS_PROFILE == 'fig19_linear' else '')
    print(f"\n==================== SUMMARY ====================\n"
          f"  model      : coupled (u,p,phi) HF, Zhou et al. 2023; "
          f"layout={NOTCH_LAYOUT}\n"
          f"  mesh       : {n_dofs} DOFs total, {len(cells)} cells, "
          f"h_max={h_max:.5f} m, l0/h_max={l/h_max:.2f}, "
          f"quadrature order={QUADRATURE_ORDER}\n"
          f"  time       : NT={NT}, dt={DT}s, t_end={NT*DT:.2f}s\n"
          f"  final half-length : {lengths[-1][1]:.3f} m\n"
          f"  final p_notch     : {lengths[-1][2]/1e6:.3f} MPa\n"
          f"{final_tip_summary}"
          f"  wall       : {wall:.1f} s\n"
          f"  paper ref  : {paper_reference}\n"
          f"================================================", flush=True)


if __name__ == '__main__':
    main()
