"""The Koyna gravity dam (paper Section 4.2.3).

Geometry, boundary conditions, material properties and the initial notch follow
Santillan, Mosquera & Cueto-Felgueroso, "Phase-field model for brittle fracture.
Validation with experimental results and extension to dam engineering problems",
Eng. Fract. Mech. 178 (2017) 109-125, Section 4.1. In the comments and option
names below, "the paper" and PAPER_MODE refer to that reference, not to the
Axion paper.

Same numerics as the two-dimensional benchmark -- spectral split, staggered
solution, Newton with Jacobi-preconditioned CG, a staggered stepper compiled
once -- applied to the dam under self-weight, reservoir pressure and overflow.

The run of the paper (uniform element size h = 0.015 m over the whole
cross-section, l = 0.03 m, 9.6e7 degrees of freedom) is launched by
``run_koyna.sh``; without it the script builds a graded mesh that is refined
around the expected crack path, which runs on a small GPU.

SETUP (from the reference)
-------------------------
1. Geometry (Fig. 4a): H=103 m, crest 14.8 m, base 70 m, downstream slope break
   at (x=19.3 m, y=66.5 m); vertical upstream face.  A boundary-conforming mapped
   QUAD4/TRI3 mesh (``koyna_mesh``).
2. Material: E=25 GPa, nu=0.2, g_c=100 Pa*m, rho=2450 kg/m^3.
3. Initial crack: horizontal, on the UPSTREAM face, at the slope-change elevation
   (y=66.5 m), length = 10% of the dam thickness there = 1.93 m (``insert_notch``).
4. Boundary conditions: base RIGIDLY FIXED (ux=uy=0).
5. Loading: self-weight + hydrostatic.  The control parameter is the reservoir
   WATER LEVEL, ramped 0 -> H (filling) and then H -> H + 10 m (overflow over the
   crest; the paper's snap-through is at ~6.7 m).  These enter the displacement
   residual as *consistent nodal load vectors* (constant in u and d, so they never
   touch the SPD tangent or the eigen-split):
       R(u) = R_internal(u,d) - ( f_grav + hydro_load(level) ).
   Self-weight is assembled once; the hydrostatic load is re-assembled per level
   (cheap host numpy) because a partly filled reservoir is not a scaled full one.

NOTE ON AT2 vs the base:  AT2 has w(d)=d^2, so w'(0)=0 -> NO elastic threshold;
its effective nucleation strength is governed by the length scale, not by f_t:
sigma_c ~ (9/16) sqrt(E*Gc/(6 l)).  Lower l (finer mesh) to raise it, or switch
to PF-CZM for a true f_t.

Units: [MPa], [m], [g = 9.81 m/s^2].
"""
import os
import sys
import glob
import time

# These must be set before importing JAX/h5py (meshio imports h5py).  Avoid JAX
# reserving nearly the entire GPU and allow XDMF output on shared HPC filesystems
# whose advisory file locking is unreliable.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import jax
import jax.numpy as np
import numpy as onp
import meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import AutoMinorLocator

import axion
from axion import DirichletBC, Field, Mesh, isotropic_function, negative_part, positive_part

# Global plot style: Times New Roman, font size 10.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
for _k in ('axes.titlesize', 'axes.labelsize', 'xtick.labelsize',
           'ytick.labelsize', 'legend.fontsize'):
    plt.rcParams[_k] = 10

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
crt_file_path = os.path.dirname(os.path.abspath(__file__))
_output_setting = os.environ.get('PFF_OUTPUT_DIR', 'output_koyna')
output_dir = (_output_setting if os.path.isabs(_output_setting)
              else os.path.join(crt_file_path, _output_setting))
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vtk_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Bracket operators
# -----------------------------------------------------------------------------
safe_plus = positive_part
safe_minus = negative_part

# region Parameters
# =============================================================================
# Material / model parameters (CONCRETE, SI-consistent [MPa], [m])
# =============================================================================
MODEL = os.environ.get('PFF_MODEL', 'AT2')
assert MODEL in ('AT1', 'AT2', 'CZM'), f"PFF_MODEL must be AT1/AT2/CZM, got {MODEL}"

# Element type: 'TRI3' (triangles) or 'QUAD4' (quads). Override via PFF_MESH.
MESH_TYPE = os.environ.get('PFF_MESH', 'TRI3')
assert MESH_TYPE in ('TRI3', 'QUAD4'), f"PFF_MESH must be TRI3/QUAD4, got {MESH_TYPE}"
MESHIO_TYPE = {'TRI3': 'triangle', 'QUAD4': 'quad'}[MESH_TYPE]

# Material properties: Koyna dam concrete, from Santillan, Mosquera &
# Cueto-Felgueroso (2017), Eng. Fract. Mech. 178, 109-125, Section 4.1.
E = 2.5e4             # Young's modulus [MPa]  (25 GPa)
nu = 0.2              # Poisson's ratio
G_c = 1.0e-4          # Griffith g_c = 100 Pa*m = 1e-4 MPa*m  (= 100 J/m^2)
l = float(os.environ.get('PFF_L', 0.3))   # Phase-field length scale [m], override
                      # via PFF_L.  The paper uses l = 0.2-0.4 m (Fig 4d).
                      # CRITICAL: l must be well BELOW the initial crack length
                      # (1.93 m), otherwise the regularised band (~2l) swallows
                      # the seeded crack and the model cannot see it at all.
                      # The graded mesh below sizes itself from l (h_fine = l/2).
# NOTE: f_t is NOT from the paper -- it has no tensile strength in it at all.
# The paper uses AT2 (its Eq. 4 is gamma_l = d^2/(2l) + (l/2)|grad d|^2, i.e.
# w(d)=d^2, c_w=2), whose effective strength is set by the LENGTH SCALE:
#     sigma_c = (9/16) sqrt(E*g_c/(6l))  =  0.81 / 0.66 / 0.57 MPa at l = 0.2/0.3/0.4
# and Section 4.1 says so explicitly ("l ... also behaves as an effective material
# parameter"), sweeping l over that range.  The dam cracks at ~6.7 m of overflow
# BECAUSE the effective strength is ~0.66 MPa.  Running PF-CZM with a realistic
# f_t = 2.9 MPa makes the dam ~4x stronger than the benchmark and NOTHING cracks
# (verified: max_d = 0.03 at 10 m overflow).  So use AT2 to reproduce the paper;
# f_t below is only consumed by the PF-CZM law and is not a paper value.
f_t = 2.9             # Tensile strength [MPa] (PF-CZM only; NOT from the paper)
mu = E / (2. * (1. + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
MU_C, LMBDA_C = mu, lmbda     # concrete Lame constants

l_ch = E * G_c / (f_t ** 2)   # material (Irwin) length, for the PF-CZM law
K_RES = 1e-8                  # residual stiffness floor (keeps u-operator nonsingular)
LIGHT_OUTPUT = os.environ.get('PFF_LIGHT_OUTPUT', '0') == '1'

# --- elastic foundation (rock) ----------------------------------------------
# A rigidly clamped dam base has a stress SINGULARITY at the heel corner (where
# the fixed base meets the loaded upstream face).  Refining the mesh only makes
# the peak stress higher, so it eventually exceeds f_t and nucleates a spurious
# crack straight along the base -- no matter how good the rest of the model is.
# Modelling the rock removes that singularity at the source: the dam base is then
# simply BONDED to a compliant foundation (shared nodes, no Dirichlet BC at all),
# and the supports move out to the far rock boundary where they do no harm.
# Rock properties are the paper's own (Section 4.2): E=20 GPa, nu=0.25.
# Paper-reproduction mode uses the rigid support and the refined fracture region
# shown in Fig. 4.  The optional compliant foundation remains available for
# sensitivity studies, but it is not the boundary condition used in Section 4.1.
PAPER_MODE = os.environ.get('PFF_PAPER_MODE', '1') == '1'
FOUNDATION = (os.environ.get('PFF_FOUND', '0') == '1') and not PAPER_MODE
E_ROCK = 2.0e4                # 20 GPa
NU_ROCK = 0.25
MU_R = E_ROCK / (2. * (1. + NU_ROCK))
LMBDA_R = E_ROCK * NU_ROCK / ((1 + NU_ROCK) * (1 - 2 * NU_ROCK))
# The foundation is MASSLESS -- standard practice for dam analysis.  Giving the
# rock self-weight would make it settle under its own gravity and produce a bogus
# initial stress/displacement field unless a geostatic initialisation step is
# added first.  We want the rock's COMPLIANCE, not its weight.
ROCK_EXT = 1.0 * 103.0        # lateral extent beyond heel and toe [m] (~1 dam height)
ROCK_DEPTH = 1.0 * 103.0      # depth below the dam base [m]
ROCK_H_MAX = 5.0              # max rock element size in the far field [m]

# --- loads (consistent units: [MPa], [m]) ---
G_ACC = 9.81                          # gravity [m/s^2]
RHO_C = 2450.0                        # concrete density [kg/m^3]
RHO_W = 1000.0                        # water density   [kg/m^3]
# body force / pressure gradients converted to [MPa/m]:  N/m^3 * 1e-6 = MPa/m
RHO_G_C = RHO_C * G_ACC * 1e-6        # ~0.02403 MPa/m  (concrete self weight)
RHO_G_W = RHO_W * G_ACC * 1e-6        # ~0.00981 MPa/m  (hydrostatic gradient)

# --- Koyna dam geometry [m] (origin at the upstream heel), from Fig. 4a ---
# Upstream face vertical (x=0).  Downstream face: mild batter from the crest
# corner (14.8, 103) down to the slope break (19.3, 66.5), then the steep slope
# from the break down to the toe (70, 0).
GEOM = dict(H=103.0,        # total height
            W_base=70.0,    # base width (heel->toe)
            W_crest=14.8,   # crest width
            y_break=66.5,   # elevation of the downstream slope break
            x_break=19.3)   # downstream x at the slope break
# Loading (Section 4.1): self-weight + hydrostatic.  The control parameter is the
# WATER LEVEL measured from the base: it is ramped 0 -> H (filling the reservoir)
# and then H -> H + overflow_max (overflow over the crest).  Overflow = level - H.
H_FULL = GEOM['H']          # full reservoir = water at crest level

Lx = GEOM['W_base']
Ly = GEOM['H']
y_break = GEOM['y_break']

# Initial crack (Section 4.1): horizontal, on the UPSTREAM face, at the slope-
# change elevation, length = 10% of the dam thickness there.
CRACK_Y = GEOM['y_break']                       # elevation of the initial crack
CRACK_LEN = 0.10 * GEOM['x_break']              # 10% of thickness = 0.10*19.3 = 1.93 m
assert l < 0.5 * CRACK_LEN, (
    f"length scale l={l} m is too large for the {CRACK_LEN:.2f} m initial crack: "
    f"the regularised band (~2l) would swallow it. Use PFF_L <~ {0.5 * CRACK_LEN:.2f}.")

# --- mesh grading -----------------------------------------------------------
# Like the paper (Fig. 4b-c) we refine only a REGION OF INTEREST around the crack
# and leave the rest coarse.  Row spacing is fine (h = l/2) in y in [Y_FINE_LO,
# Y_FINE_HI] and grows geometrically to H_COARSE away from it.  This is what makes
# a small l affordable: uniform refinement at l=0.3 m would need ~650k elements.
# Coarse rows at the base are a FEATURE: they stop the clamped-corner stress
# singularity at the heel from being resolved sharply enough to nucleate.
# h = l/5 in the crack band, NOT l/2.  The paper meshes its region of interest at
# 4 cm with l = 0.2-0.4 m, i.e. l/h = 5-10.  At the l/h = 2 "minimum" the crack TIP
# is too coarse to localise, so the crack initiates late -- measured: the neck went
# at 8.8 m of overflow instead of the paper's 6.7 m, which let the (diffuse, and
# always-damaging under AT2) base tension localise first at 8.4 m and steal the run.
H_FINE = float(os.environ.get('PFF_HFINE', l / 5.0))   # row spacing in the crack band
H_COARSE = max(8.0 * H_FINE, 1.0)                      # row spacing far away
Y_FINE_LO, Y_FINE_HI = 50.0, 80.0                  # region of interest [m]
GRADE_GROWTH = 1.2                                 # geometric growth per row

# In paper mode the phase field is solved only in the refined crack-path region
# visible in Fig. 4b-c.  This prevents the threshold-free AT2 model from using
# the rigid heel singularity as a competing nucleation site.  Displacements are
# still solved over the complete dam and the full base remains clamped.
FRACTURE_X_MAX = float(os.environ.get('PFF_FRAC_XMAX', 30.0))
FRACTURE_Y_MIN = float(os.environ.get('PFF_FRAC_YMIN', Y_FINE_LO))
FRACTURE_Y_MAX = float(os.environ.get('PFF_FRAC_YMAX', Y_FINE_HI))

# --- damage-free base layer -------------------------------------------------
# Impose d = 0 for nodes with y <= BASE_FREE_H.  The concrete-rock interface is
# NOT modelled here, and the fully clamped base has a corner stress singularity at
# the heel that otherwise nucleates a spurious crack straight along the base.
#
# NOTE: size this against the BASE ROW SPACING, not against l.  The rows near the
# base are deliberately COARSE (H_COARSE ~ 1.2 m), so a layer of a few *l* would
# pin only the single y=0 row and the crack would simply form one row higher up.
# 5% of the dam height spans ~5 coarse rows, which is enough to keep it out.
#
# With FOUNDATION=1 this crutch is NOT needed: there is no clamped-corner
# singularity to suppress, so we pin damage only in the rock and at the
# concrete-rock interface (y <= 0) and let the whole dam body crack freely.
BASE_FREE_H = float(os.environ.get(
    'PFF_BASE_FREE', 0.0 if FOUNDATION else 0.05 * GEOM['H']))


def x_right_of_y(y):
    """Downstream face x-coordinate at elevation y (host/onp scalar or array)."""
    Wb, Wc, yb, xb = (GEOM['W_base'], GEOM['W_crest'], GEOM['y_break'], GEOM['x_break'])
    H = GEOM['H']
    lower = Wb + (xb - Wb) * (y / yb)                    # toe (y=0) -> break (y=yb)
    upper = xb + (Wc - xb) * ((y - yb) / (H - yb))       # break -> crest corner
    return onp.where(y <= yb, lower, upper)


def crack_geo(d):
    """Crack geometric function w(d)."""
    if MODEL == 'AT1':
        return d
    if MODEL == 'AT2':
        return d ** 2
    if MODEL == 'CZM':
        return 2. * d - d ** 2
    raise ValueError(MODEL)


def degradation(d):
    """Energy degradation function g(d)."""
    w = (1. - d) ** 2
    if MODEL in ('AT1', 'AT2'):
        return w
    if MODEL == 'CZM':
        a1 = 4. / np.pi * l_ch / l
        a2 = -0.5
        a3 = 0.0
        Q = a1 * d + a1 * a2 * d ** 2 + a1 * a2 * a3 * d ** 3
        return w / (w + Q)
    raise ValueError(MODEL)


def c_w():
    return {'AT1': 8. / 3., 'AT2': 2., 'CZM': np.pi}[MODEL]


_g_prime = jax.grad(degradation)
_w_prime = jax.grad(crack_geo)

# region damage subproblem
# =============================================================================
# Damage subproblem
# =============================================================================
class Damage(Field):
    """omega'(d) H + Gc/c_w ( alpha'(d)/l - 2 l lap d ) = 0, with the driving
    term frozen at the irreversible history field H."""
    def flux(self):
        coeff = 2. * G_c * l / c_w()

        def fn(d_grad, history):
            return coeff * d_grad
        return fn

    def source(self):
        coeff = G_c / (c_w() * l)

        def fn(d, x, history):
            dd = d[0]
            return np.array([coeff * _w_prime(dd) + _g_prime(dd) * history])
        return fn

# region displacement subproblem
# =============================================================================
# Displacement subproblem
# =============================================================================
# deviatoric part of the split stress divided by 2 mu; g is the degradation at
# the quadrature point
_split = isotropic_function(lambda lam, g: g * safe_plus(lam) + safe_minus(lam))


def stress_fn(u_grad, d, mat):
    """sigma = (g(d) + k_res) dPsi+/deps + dPsi-/deps (spectral split). ``mat``
    holds [mu, lambda] at the quadrature point, so the dam (concrete) and the
    foundation (rock) share one operator."""
    mu_q, lmbda_q = mat[0], mat[1]
    epsilon = 0.5 * (u_grad + u_grad.T)
    g = degradation(d[0]) + K_RES
    tr_plus = safe_plus(np.trace(epsilon))
    tr_minus = safe_minus(np.trace(epsilon))
    sigma1 = lmbda_q * (g * tr_plus + tr_minus) * np.eye(2)
    return sigma1 + 2. * mu_q * _split(epsilon, g)


class Displacement(Field):
    """theta = (d, mat) at the quadrature points."""
    def flux(self):
        return stress_fn

    def compute_history(self, sol_u, history_old, mat_quad):
        u_grads = self.grad_at_quad(sol_u)
        psi = jax.vmap(jax.vmap(psi_plus_density))(u_grads, mat_quad)
        return np.maximum(psi, history_old)

# region BC
# =============================================================================
# Boundary conditions: base fully clamped (foundation bond).  No prescribed
# displacement -- the dam is load-controlled.
# =============================================================================
def base(point):
    return np.isclose(point[1], 0., atol=1e-6)


def upstream(point):
    """The dam's wetted upstream face: x = 0 AND 0 <= y <= H.

    The y-bounds are ESSENTIAL, not cosmetic.  Field.select_faces takes every
    face whose nodes satisfy the predicate -- including INTERIOR faces.  With a foundation, the plane
    x = 0 continues down through the rock, so a bare `isclose(x, 0)` also picks up
    the interior faces there; each is shared by two cells, so the pressure gets
    applied twice, on an interior plane, and the total horizontal load comes out
    at ~364 MN instead of the correct 52 MN.  Bounding y to the dam fixes it.
    """
    return (np.isclose(point[0], 0., atol=1e-6)
            & (point[1] > -1e-6)
            & (point[1] < GEOM['H'] + 1e-6))


def rock_bottom(point):
    return np.isclose(point[1], -ROCK_DEPTH, atol=1e-6)


def rock_side(point):
    return (np.isclose(point[0], -ROCK_EXT, atol=1e-6)
            | np.isclose(point[0], GEOM['W_base'] + ROCK_EXT, atol=1e-6))


if FOUNDATION:
    # The dam base carries NO Dirichlet condition at all -- it is bonded to the
    # rock through shared nodes, so continuity is automatic and there is no
    # clamped corner to go singular.  The supports sit far away on the rock:
    #   bottom of the rock : fully fixed
    #   sides of the rock  : rollers (ux = 0, free to settle vertically)
    dirichlet_u = [DirichletBC(rock_bottom, 0, 0.), DirichletBC(rock_bottom, 1, 0.),
                   DirichletBC(rock_side, 0, 0.)]
else:
    # No foundation: the dam base is RIGIDLY FIXED (reference Section 4.1,
    # "rigidly supported").  This is what produces the heel corner singularity,
    # hence the damage-free base layer above.
    dirichlet_u = [DirichletBC(base, 0, 0.), DirichletBC(base, 1, 0.)]

# region mesh
# =============================================================================
# Structured, boundary-conforming (mapped) QUAD4 mesh of the Koyna profile.
# At each elevation y the section spans x in [0, x_right(y)]; we place nx+1 nodes
# linearly across it, giving a logically-rectangular quad grid that conforms to
# the sloped downstream face.
# =============================================================================
def graded_ys():
    """Row elevations, refined around the crack region and coarse elsewhere.

    The fine block is anchored EXACTLY on y_break so the crack line always lands
    on a node row.  Outside it the spacing grows geometrically up to H_COARSE.
    """
    H = GEOM['H']
    # fine block, anchored on y_break
    n_dn = int(onp.ceil((y_break - Y_FINE_LO) / H_FINE))
    n_up = int(onp.ceil((Y_FINE_HI - y_break) / H_FINE))
    fine = y_break + H_FINE * onp.arange(-n_dn, n_up + 1)
    fine = fine[(fine > 0.0) & (fine < H)]

    def grade(start, stop, sgn):
        out, y, h = [], float(start), H_FINE
        while True:
            h = min(h * GRADE_GROWTH, H_COARSE)
            y = y + sgn * h
            if sgn > 0 and y >= stop - 0.5 * h:
                break
            if sgn < 0 and y <= stop + 0.5 * h:
                break
            out.append(y)
        return out

    below = grade(fine[0], 0.0, -1)
    above = grade(fine[-1], H, +1)
    ys = onp.concatenate([[0.0], onp.array(below), fine, onp.array(above), [H]])
    return onp.unique(onp.round(ys, 9))


def uniform_ys(h_target):
    """Nearly uniform elevations with an exact row at the notch elevation.

    Each side of y_break is divided independently, so every row spacing differs
    from h_target only by the small rounding needed to land exactly on y_break
    and H.  This avoids the single distorted row produced by moving a node from
    an otherwise uniform linspace.
    """
    if h_target <= 0.0:
        raise ValueError('PFF_UNIFORM_H must be positive')
    n_below = max(1, int(onp.ceil(y_break / h_target)))
    n_above = max(1, int(onp.ceil((GEOM['H'] - y_break) / h_target)))
    below = onp.linspace(0.0, y_break, n_below + 1)
    above = onp.linspace(y_break, GEOM['H'], n_above + 1)[1:]
    return onp.concatenate([below, above])


def koyna_mesh(nx, ny=None, uniform_h=None):
    """Boundary-conforming mapped mesh.  If ny is None the rows are GRADED
    (region-of-interest refinement); otherwise they are uniform with ny rows."""
    H = GEOM['H']
    if uniform_h is not None:
        ys = uniform_ys(float(uniform_h))
    elif ny is None:
        ys = graded_ys()
    else:
        ys = onp.linspace(0., H, ny + 1)
        # force one node row EXACTLY onto the slope break so the notch line exists
        j0 = int(onp.argmin(onp.abs(ys - y_break)))
        ys[j0] = y_break
    ny = len(ys) - 1
    xr = x_right_of_y(ys)                     # downstream x at each row
    xi = onp.linspace(0., 1., nx + 1)         # normalised span coordinate

    nxp = nx + 1
    X = (xi[None, :] * xr[:, None]).ravel()   # (ny+1, nx+1)
    Y = onp.repeat(ys, nxp)
    points = onp.stack([X, Y], axis=1)

    def nid(i, j):
        return j * nxp + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if MESH_TYPE == 'QUAD4':
                cells.append([n0, n1, n2, n3])
            else:  # TRI3: split each quad into two CCW triangles
                cells.append([n0, n1, n2])
                cells.append([n0, n2, n3])
    # y_break is a node row by construction in both branches
    return points, onp.array(cells, dtype=onp.int32), float(y_break), ys


def _grade_out(start, stop, h0, h_max, growth=1.25):
    """Node coords from `start` to `stop` (exclusive of start, inclusive of stop),
    first step h0 growing geometrically to h_max."""
    sgn = 1.0 if stop > start else -1.0
    out, y, h = [], float(start), h0
    while True:
        h = min(h * growth, h_max)
        y = y + sgn * h
        if (sgn > 0 and y >= stop - 0.5 * h) or (sgn < 0 and y <= stop + 0.5 * h):
            break
        out.append(y)
    out.append(float(stop))
    return out


def add_foundation(points, cells, dam_ys):
    """Append an elastic rock block under the dam and bond it by SHARING the dam's
    base nodes (no tied constraints needed -- the interface nodes are literally the
    same nodes, so displacement continuity is automatic).

    The rock's column positions inside [0, W_base] are exactly the dam's base-row
    x-coordinates, which is what makes the shared-node merge possible.  Outside
    that span, and downward, the rock is graded coarse.

    Returns (points, cells, is_rock_cell).
    """
    points = onp.asarray(points)
    cells = onp.asarray(cells)
    Wb = GEOM['W_base']
    n_dam_cells = len(cells)

    # --- rock grid lines ----------------------------------------------------
    base_mask = onp.abs(points[:, 1]) < 1e-9
    xs_dam = onp.unique(onp.round(points[base_mask, 0], 9))    # dam base columns
    h0 = float(onp.min(onp.diff(xs_dam)))                      # match dam spacing
    xs_left = _grade_out(0.0, -ROCK_EXT, h0, ROCK_H_MAX)
    xs_right = _grade_out(Wb, Wb + ROCK_EXT, h0, ROCK_H_MAX)
    xs = onp.unique(onp.round(onp.concatenate([xs_left, xs_dam, xs_right]), 9))

    h0y = float(dam_ys[1] - dam_ys[0])                         # match first dam row
    ys = onp.unique(onp.round(
        onp.concatenate([[0.0], _grade_out(0.0, -ROCK_DEPTH, h0y, ROCK_H_MAX)]), 9))
    ys = ys[::-1] if ys[0] > ys[-1] else ys                    # ascending: -D .. 0

    nxp, nyp = len(xs), len(ys)

    # --- node indices: reuse the dam's base nodes on the top row -------------
    dam_base_id = {}
    for nid_ in onp.where(base_mask)[0]:
        dam_base_id[round(float(points[nid_, 0]), 9)] = int(nid_)

    new_pts = []
    gid = onp.full((nyp, nxp), -1, dtype=onp.int64)
    next_id = len(points)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            key = round(float(xv), 9)
            if abs(yv) < 1e-9 and key in dam_base_id:
                gid[j, i] = dam_base_id[key]        # SHARED with the dam
            else:
                gid[j, i] = next_id
                new_pts.append([xv, yv])
                next_id += 1
    if new_pts:
        points = onp.vstack([points, onp.array(new_pts)])

    # --- rock cells ---------------------------------------------------------
    rock_cells = []
    for j in range(nyp - 1):
        for i in range(nxp - 1):
            n0, n1, n2, n3 = gid[j, i], gid[j, i + 1], gid[j + 1, i + 1], gid[j + 1, i]
            if MESH_TYPE == 'QUAD4':
                rock_cells.append([n0, n1, n2, n3])
            else:
                rock_cells.append([n0, n1, n2])
                rock_cells.append([n0, n2, n3])
    cells = onp.vstack([cells, onp.array(rock_cells, dtype=cells.dtype)])

    is_rock_cell = onp.zeros(len(cells), dtype=bool)
    is_rock_cell[n_dam_cells:] = True
    return points, cells, is_rock_cell


def insert_notch(points, cells, y_notch, x_tip, tol=1e-9):
    """Open a horizontal slit on {y = y_notch, x < x_tip} (an edge crack from the
    UPSTREAM face x=0 inward to the tip at x = x_tip) by duplicating the line
    nodes (tip excluded) and remapping the elements above the line to the copies.
    This is the paper's initial fracture: horizontal, on the upstream face, at the
    slope-change elevation."""
    points = onp.asarray(points)
    cells = onp.asarray(cells).copy()
    on_line = onp.isclose(points[:, 1], y_notch, atol=1e-6) & (points[:, 0] < x_tip - tol)
    old_ids = onp.where(on_line)[0]
    if len(old_ids) == 0:
        raise ValueError("no nodes on the crack line; check y_break lands on a row")
    new_ids = onp.arange(len(points), len(points) + len(old_ids))
    remap = dict(zip(old_ids.tolist(), new_ids.tolist()))
    points = onp.vstack([points, points[old_ids]])
    cent_y = points[cells][:, :, 1].mean(axis=1)
    for c in onp.where(cent_y > y_notch)[0]:
        for j in range(cells.shape[1]):
            if cells[c, j] in remap:
                cells[c, j] = remap[cells[c, j]]
    return points, cells


# =============================================================================
# Field builder
# =============================================================================
def build_fields(points, cells):
    mesh = Mesh(onp.asarray(points), onp.asarray(cells), MESH_TYPE)
    field_d = Damage(mesh, ncomp=1)
    field_u = Displacement(mesh, ncomp=2, dirichlet=dirichlet_u)
    return mesh, field_d, field_u


# =============================================================================
# Dirichlet data: (global unknown index, value) pairs
# =============================================================================
def dirichlet_data(field):
    return field.dirichlet_dofs, field.dirichlet_values

# region external loads
# =============================================================================
# Precomputed consistent nodal load vectors (host, once).  These are the whole
# reason the code can stay load-controlled without touching the SPD tangent or
# the eigen split: they are constant in u and d.
# =============================================================================
def gravity_load(field_u, is_rock_cell=None):
    """Consistent nodal self-weight, body force b = (0, -rho_c*g).  Flat (N*vec,).

    The foundation is MASSLESS: rock cells get zero body force.  Otherwise the
    rock would settle under its own weight and contaminate the dam's stress field
    with a bogus initial state (you'd need a geostatic initialisation step first).
    """
    fe = field_u
    vec = fe.ncomp
    Nd = fe.num_nodes * vec
    cells = onp.asarray(fe.cells)                        # (num_cells, num_nodes)

    shape_vals = onp.asarray(fe.shape_vals)             # (num_quads, num_nodes)
    JxW = onp.asarray(fe.JxW)                            # (num_cells, num_quads)
    Na_int = onp.einsum('qa,cq->ca', shape_vals, JxW)   # = int N_a dV, per element
    if is_rock_cell is not None:                        # massless foundation
        Na_int = Na_int * (~onp.asarray(is_rock_cell))[:, None]

    b_grav = onp.array([0.0, -RHO_G_C])                 # self weight [MPa/m]
    f_grav = onp.zeros(Nd)
    for comp in range(vec):
        gdof = (cells * vec + comp).reshape(-1)
        onp.add.at(f_grav, gdof, (Na_int * b_grav[comp]).reshape(-1))
    return np.asarray(f_grav)


def upstream_face_precompute(field_u):
    """Static geometry of the upstream (x=0) face, so the hydrostatic load can be
    re-assembled cheaply at ANY water level (a partially filled reservoir is NOT
    a scaled full one -- the wetted height changes, so the load is not linear in
    the level until the face is fully submerged)."""
    vec = field_u.ncomp
    faces = field_u.face_set(upstream)
    face_vals = onp.asarray(faces.shape_vals)                             # (nf, nfq, num_nodes)
    cells_b = onp.asarray(field_u.mesh.cells)[faces.cell_ids]           # (nf, num_nodes)
    xq = onp.asarray(faces.quad_points)                                   # (nf, nfq, dim)
    return dict(face_vals=face_vals, cells_b=cells_b, nanson=onp.asarray(faces.dS),
                yq=xq[:, :, 1], shape=xq.shape, Nd=field_u.num_nodes * vec, vec=vec)


def hydro_load(pre, water_level):
    """Consistent nodal hydrostatic load on the upstream face for a reservoir
    filled to `water_level` (measured from the base):

        p(y) = rho_w * g * (water_level - y),  clipped at 0 above the waterline

    pushing in +x (downstream).  Flat (N*vec,)."""
    p = onp.clip(RHO_G_W * (water_level - pre['yq']), 0.0, None)   # (nf, nfq) [MPa]
    t = onp.zeros(pre['shape'])
    t[:, :, 0] = p                                                 # traction +x
    fvec = onp.einsum('fqa,fqd,fq->fad', pre['face_vals'], t, pre['nanson'])
    f = onp.zeros(pre['Nd'])
    for comp in range(pre['vec']):
        gdof = (pre['cells_b'] * pre['vec'] + comp).reshape(-1)
        onp.add.at(f, gdof, fvec[:, :, comp].reshape(-1))
    return np.asarray(f)

# region diagnostics
# =============================================================================
# Energy densities (module level so both the solver and the diagnostics use the
# exact same definitions) and per-step diagnostic quantities for debugging the
# base-crack question: WHERE is the tensile driving energy psi+ largest?
# =============================================================================
def _strain(u_grad):
    return 0.5 * (u_grad + u_grad.T)


def psi_plus_density(u_grad, mat):
    """Tensile (crack-driving) elastic energy density psi+ (undegraded).
    `mat` = [mu, lmbda] at this quadrature point."""
    mu_q, lmbda_q = mat[0], mat[1]
    eps = _strain(u_grad)
    ev, _ = np.linalg.eigh(eps)
    tr_plus = safe_plus(np.trace(eps))
    return lmbda_q / 2. * tr_plus ** 2 + mu_q * np.sum(safe_plus(ev) ** 2)


def psi_minus_density(u_grad, mat):
    """Compressive elastic energy density psi- (never degraded)."""
    mu_q, lmbda_q = mat[0], mat[1]
    eps = _strain(u_grad)
    ev, _ = np.linalg.eigh(eps)
    tr_minus = safe_minus(np.trace(eps))
    return lmbda_q / 2. * tr_minus ** 2 + mu_q * np.sum(safe_minus(ev) ** 2)


def material_quad(field_u, is_rock_cell):
    """Per-quad [mu, lmbda] array, shape (num_cells, num_quads, 2).  This is the
    ONLY thing that distinguishes concrete from rock in the operator."""
    nq = field_u.num_quads
    is_rock = onp.asarray(is_rock_cell)
    mu_c = onp.where(is_rock, MU_R, MU_C)
    lm_c = onp.where(is_rock, LMBDA_R, LMBDA_C)
    mat = onp.stack([onp.repeat(mu_c[:, None], nq, axis=1),
                     onp.repeat(lm_c[:, None], nq, axis=1)], axis=-1)
    return np.asarray(mat)


def _cell_u_grads(fe, sol_u):
    """(num_cells, num_quads, vec, dim) displacement gradients at quad points."""
    return fe.grad_at_quad(sol_u)


def compute_energy_fields(field_u, sol_u, sol_d, mat_quad, dam_cell_mask=None):
    """Return (scalars dict, per-cell field dict).

    scalars: E_pos = int psi+ dx        (total tensile / crack-driving energy)
             E_min = int psi- dx        (total compressive energy)
             E_ela = int [g(d)psi+ + psi-] dx   (degraded elastic strain energy)
    fields (per cell, for XDMF): psi_plus, psi_minus, strain_energy (degraded),
             d_cell.
    """
    fe = field_u
    u_grads = _cell_u_grads(fe, sol_u)                              # (nc, nq, vec, dim)
    psi_p = jax.vmap(jax.vmap(psi_plus_density))(u_grads, mat_quad)  # (nc, nq)
    psi_m = jax.vmap(jax.vmap(psi_minus_density))(u_grads, mat_quad)
    d_quad = fe.to_quad(sol_d)[:, :, 0]            # (nc, nq)
    g_q = degradation(d_quad) + K_RES
    JxW = fe.JxW                                                    # (nc, nq)

    # Integrate the energies over the DAM only -- the rock is a numerical device,
    # and its (large, elastic) energy would swamp the dam's fracture signal.
    w = JxW if dam_cell_mask is None else JxW * dam_cell_mask[:, None]
    E_pos = float(np.sum(psi_p * w))
    E_min = float(np.sum(psi_m * w))
    E_ela = float(np.sum((g_q * psi_p + psi_m) * w))

    scalars = dict(E_pos=E_pos, E_min=E_min, E_ela=E_ela)
    fields = dict(
        psi_plus=onp.asarray(np.mean(psi_p, axis=1)),
        psi_minus=onp.asarray(np.mean(psi_m, axis=1)),
        strain_energy=onp.asarray(np.mean(g_q * psi_p + psi_m, axis=1)),
        d_cell=onp.asarray(np.mean(d_quad, axis=1)),
    )
    return scalars, fields


def compute_fracture_energy(field_d, sol_d):
    """Total surface/fracture energy int Gc/cw ( w(d)/l + l |grad d|^2 ) dx."""
    fe = field_d
    d_grads = fe.grad_at_quad(sol_d)[:, :, 0, :]             # (nc, nq, dim)
    grad_sq = np.sum(d_grads ** 2, axis=-1)                  # (nc, nq)
    d_quad = fe.to_quad(sol_d)[:, :, 0]     # (nc, nq)
    integrand = G_c / c_w() * (crack_geo(d_quad) / l + l * grad_sq)
    return float(np.sum(integrand * fe.JxW))


def compute_base_reaction(field_u, sol_u, sol_d, mat_quad, f_ext_u,
                          base_dofs_x, base_dofs_y):
    """Total support reaction.

    The (unconstrained) residual is  R_int(u,d) - f_ext.  At the free DOFs it is
    zero; at the supported DOFs the leftover equals the reaction the supports must
    supply.  Summing the x/y supported DOFs gives the shear (Rx) and vertical
    reaction (Ry).  By global equilibrium Rx ~ -(applied horizontal) and
    Ry ~ +(self weight), so this doubles as a solver sanity check.

    Without a foundation the supported DOFs are the clamped dam base; with one
    they are the outer rock boundary -- the totals are the same either way.
    """
    d_quad = field_u.to_quad(sol_d)
    zero_f = np.zeros_like(f_ext_u)
    R_int, _ = assemble_device(field_u, np.asarray(sol_u).reshape(-1),
                               [d_quad, mat_quad], zero_f)
    react = R_int - f_ext_u
    return float(np.sum(react[base_dofs_x])), float(np.sum(react[base_dofs_y]))

# region CG solver
# =============================================================================
# Operators for the SPD subproblems
# =============================================================================
def get_assembled_ops(field):
    cache = getattr(field, '_asm_ops', None)
    if cache is not None:
        return cache
    matvec, inv_jacobi = axion.make_operator(field)
    cache = (matvec, inv_jacobi, field.ncomp, field.num_nodes)
    field._asm_ops = cache
    return cache


def assemble_device(field, u_flat, internal_vars, f_ext):
    """Residual and stored tangent entries.  The external load vector f_ext
    (constant in u and d) is subtracted from the internal residual:
        R = R_internal(u) - f_ext.
    For subproblems with no external load, pass zeros."""
    res, V = axion.assemble(field, u_flat, internal_vars)
    return res - f_ext, V


def get_newton_solver(field, num_newton=20, nl_tol=1e-8, lin_tol=1e-4, lin_maxiter=5000):
    cache = getattr(field, '_newton_solver', None)
    if cache is not None:
        return cache
    matvec, inv_jacobi, vec, N = get_assembled_ops(field)

    @jax.jit
    def newton(u0, R0, V0, ivs, fixed_idx, fixed_vals, f_ext):
        """`ivs` is a TUPLE of internal variables: (d_quad, mat_quad) for the
        displacement subproblem, (history,) for the damage subproblem."""
        ivs = ivs if isinstance(ivs, (tuple, list)) else (ivs,)
        def cond(state):
            u, R, V, it = state
            return (it < num_newton) & (np.linalg.norm(R) > nl_tol)
        def body(state):
            u, R, V, it = state
            inv = inv_jacobi(V, fixed_idx)
            du, _ = jax.scipy.sparse.linalg.cg(lambda d: matvec(V, d, fixed_idx), -R,
                                               tol=lin_tol, atol=0.0, maxiter=lin_maxiter,
                                               M=lambda r: r * inv)
            u_new = u + du
            res_new, V_new = assemble_device(field, u_new, list(ivs), f_ext)
            return (u_new, res_new.at[fixed_idx].set(0.0), V_new, it + 1)
        return jax.lax.while_loop(cond, body, (u0, R0, V0, 0))[0]

    field._newton_solver = newton
    return newton


# region PG solver
# =============================================================================
# Bound-constrained (0 <= d <= 1) damage solve by projected gradient (CZM/AT1)
# =============================================================================
def get_damage_solver(field, d_fixed_idx, max_it=300, tol=1e-7, omega=1.0):
    """Projected gradient for the bound-constrained damage solve (CZM / AT1).
    Nodes in `d_fixed_idx` are pinned at d = 0 (the damage-free base layer)."""
    cache = getattr(field, '_damage_solver', None)
    if cache is not None:
        return cache
    @jax.jit
    def pg_solve(y0, Dm, history):
        def residual_fn(y):
            return field.residual(y, [history])
        def cond(state):
            y, y_prev, it = state
            change = np.linalg.norm(y - y_prev) / (np.linalg.norm(y) + 1e-30)
            return (it < max_it) & (change > tol)
        def body(state):
            y, _, it = state
            R = residual_fn(y)
            y_new = np.clip(y - omega * R / Dm, 0., 1.)
            y_new = y_new.at[d_fixed_idx].set(0.0)      # damage-free base layer
            return (y_new, y, it + 1)
        y0c = np.clip(y0, 0., 1.).at[d_fixed_idx].set(0.0)
        y, _, _ = jax.lax.while_loop(cond, body, (y0c, y0c + 1.0, 0))
        return y

    field._damage_solver = pg_solve
    return pg_solve


def compute_damage_Dm(field, sol, history):
    return axion.abs_diagonal(field, sol, [history])


# region Plotting
# =============================================================================
# Plotting  (metres; no gap-opening -- the notch is already a real slit)
# =============================================================================
def plot_quad_mesh(points, cells, fname, title, sol_d=None, show_notch=False,
                   notch_tip=None, clip_to_dam=True):
    points = onp.asarray(points)
    cells = onp.asarray(cells)
    polys = points[cells].astype(float)
    fig, ax = plt.subplots(figsize=(6, 8))
    if sol_d is None:
        pc = PolyCollection(polys, facecolors='#cfe2f3', edgecolors='k', linewidths=0.15)
        ax.add_collection(pc)
    else:
        d_cell = onp.asarray(sol_d).reshape(-1)[cells].mean(axis=1)
        pc = PolyCollection(polys, array=d_cell, cmap='inferno',
                            edgecolors='face', linewidths=0.0)
        pc.set_clim(0., 1.)
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('damage  d')
    if show_notch and notch_tip is not None:
        # initial crack on the UPSTREAM face: x=0 -> notch_tip at y=y_break
        ax.plot([0., notch_tip], [y_break, y_break], 'r-', lw=2., label='initial crack')
        ax.legend(loc='upper right', frameon=False)
    if clip_to_dam:
        # the foundation block is much larger than the dam; frame the dam
        ax.set_xlim(-2, GEOM['W_base'] + 2)
        ax.set_ylim(-2, GEOM['H'] + 2)
    else:
        ax.set_xlim(points[:, 0].min() - 2, points[:, 0].max() + 2)
        ax.set_ylim(points[:, 1].min() - 2, points[:, 1].max() + 2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# region staggered solver
# =============================================================================
# Staggered (alternating-minimization) solve on the fixed mesh
# =============================================================================
def get_staggered_stepper(field_u, field_d, max_stagger=5, ms_tol=1e-4):
    cache = getattr(field_u, '_staggered_stepper', None)
    if cache is not None:
        return cache
    newton_u = get_newton_solver(field_u)
    fe_u = field_u; Nu = fe_u.num_nodes
    fe_d = field_d; Nd = fe_d.num_nodes

    zero_fd = np.zeros(Nd * field_d.ncomp)   # damage has no external load

    # Damage-free region: vec=1 for the phase field, so the flat DOF index IS the
    # node index.  In paper mode damage is admissible only in the refined region
    # surrounding the prescribed notch and expected downward crack path.  This
    # removes the rigid-heel singularity from the fracture minimisation without
    # changing the displacement boundary condition.
    pts_d = onp.asarray(fe_d.points)
    if PAPER_MODE:
        admissible = ((pts_d[:, 0] >= -1e-9)
                      & (pts_d[:, 0] <= FRACTURE_X_MAX + 1e-9)
                      & (pts_d[:, 1] >= FRACTURE_Y_MIN - 1e-9)
                      & (pts_d[:, 1] <= FRACTURE_Y_MAX + 1e-9))
        d_fixed_idx = np.array(onp.where(~admissible)[0], dtype=np.int32)
    else:
        d_free_h = 0.0 if FOUNDATION else BASE_FREE_H
        d_fixed_idx = np.array(onp.where(pts_d[:, 1] <= d_free_h + 1e-9)[0],
                               dtype=np.int32)
    d_fixed_vals = np.zeros(d_fixed_idx.shape)

    if MODEL in ('CZM', 'AT1'):
        pg_solve = get_damage_solver(field_d, d_fixed_idx)
    else:                                   # AT2: Newton solver with CG
        newton_d = get_newton_solver(field_d)

    @jax.jit
    def stepper(sol_u, sol_d, history_old, fixed_idx, fixed_vals, f_ext_u, mat_quad):
        def cond(state):
            su, sd, hist, err, s = state
            return (s < max_stagger) & (err > ms_tol)
        def body(state):
            su, sd, hist, err, s = state
            # ---- u-solve (load-controlled: external f_ext_u in residual) ----
            d_quad = fe_u.to_quad(sd)
            u0 = su.reshape(-1).at[fixed_idx].set(fixed_vals)
            res0, V0 = assemble_device(field_u, u0, [d_quad, mat_quad], f_ext_u)
            R0 = res0.at[fixed_idx].set(0.0)
            su_new = newton_u(u0, R0, V0, (d_quad, mat_quad),
                              fixed_idx, fixed_vals, f_ext_u).reshape(Nu, 2)
            # ---- history ----
            hist_new = field_u.compute_history(su_new, history_old, mat_quad)
            # ---- d-solve ----
            if MODEL in ('CZM', 'AT1'):
                Dm = compute_damage_Dm(field_d, sd, hist_new)
                sd_new = pg_solve(sd.reshape(-1), Dm, hist_new).reshape(Nd, 1)
            else:                                       # AT2: Newton + CG + clip
                sd0 = sd.reshape(-1).at[d_fixed_idx].set(0.0)
                res0d, V0d = assemble_device(field_d, sd0, [hist_new], zero_fd)
                res0d = res0d.at[d_fixed_idx].set(0.0)
                sd_new = newton_d(sd0, res0d, V0d, (hist_new,),
                                  d_fixed_idx, d_fixed_vals, zero_fd).reshape(Nd, 1)
                sd_new = np.clip(sd_new, 0., 1.).at[d_fixed_idx].set(0.0)
            # ---- converge ----
            eu = np.linalg.norm(su_new - su) / (np.linalg.norm(su_new) + 1e-30)
            ed = np.linalg.norm(sd_new - sd) / (np.linalg.norm(sd_new) + 1e-30)
            return (su_new, sd_new, hist_new, np.maximum(eu, ed), s + 1)
        init = (sol_u, sol_d, history_old, np.array(1e30), np.array(0))
        su, sd, hist, err, s = jax.lax.while_loop(cond, body, init)
        return su, sd, hist

    field_u._staggered_stepper = stepper
    return stepper

# region loading stepper
# ---------------------------------------------------------------------------
# Static Dirichlet structure.  The base is fully clamped at zero; there is no
# load-carrying Dirichlet DOF, so fixed_vals is always zero.
# ---------------------------------------------------------------------------
def build_dirichlet_static(field_u, mat_quad):
    fixed_idx, vals = dirichlet_data(field_u)
    fixed_idx_static = np.array(onp.asarray(fixed_idx), dtype=np.int32)
    fixed_vals_static = np.array(onp.asarray(vals))     # all zeros (clamped base)
    return fixed_idx_static, fixed_vals_static


def get_elastic_preload_stepper(field_u):
    """Return a displacement-only step used for self-weight/reservoir preload.

    Damage and its irreversible history are deliberately not updated here.  The
    final full-reservoir elastic state is used to initialise the history once,
    immediately before overflow fracture evolution starts.
    """
    newton_u = get_newton_solver(field_u)
    fe_u = field_u
    Nu = fe_u.num_nodes

    @jax.jit
    def preload(sol_u, sol_d, fixed_idx, fixed_vals, f_ext_u, mat_quad):
        d_quad = fe_u.to_quad(sol_d)
        u0 = sol_u.reshape(-1).at[fixed_idx].set(fixed_vals)
        res0, V0 = assemble_device(field_u, u0, [d_quad, mat_quad], f_ext_u)
        R0 = res0.at[fixed_idx].set(0.0)
        return newton_u(u0, R0, V0, (d_quad, mat_quad), fixed_idx,
                        fixed_vals, f_ext_u).reshape(Nu, 2)

    return preload


def regional_maxima(points, cells, sol_d, fields=None):
    """Damage and tensile-energy maxima in heel, notch corridor, and remainder."""
    pts = onp.asarray(points)
    ctr = pts[onp.asarray(cells)].mean(axis=1)
    d = onp.asarray(sol_d).reshape(-1)
    psi = None if fields is None else onp.asarray(fields['psi_plus'])

    node_heel = (pts[:, 1] >= -1e-9) & (pts[:, 1] < 10.0)
    node_notch = ((pts[:, 0] <= FRACTURE_X_MAX) &
                  (pts[:, 1] >= FRACTURE_Y_MIN) &
                  (pts[:, 1] <= FRACTURE_Y_MAX))
    cell_heel = (ctr[:, 1] >= -1e-9) & (ctr[:, 1] < 10.0)
    cell_notch = ((ctr[:, 0] <= FRACTURE_X_MAX) &
                  (ctr[:, 1] >= FRACTURE_Y_MIN) &
                  (ctr[:, 1] <= FRACTURE_Y_MAX))

    def mx(a, mask):
        return float(a[mask].max()) if onp.any(mask) else 0.0

    return dict(
        max_d_heel=mx(d, node_heel),
        max_d_notch=mx(d, node_notch),
        max_d_rest=mx(d, ~(node_heel | node_notch)),
        max_psi_heel=onp.nan if psi is None else mx(psi, cell_heel),
        max_psi_notch=onp.nan if psi is None else mx(psi, cell_notch),
        max_psi_rest=onp.nan if psi is None else mx(psi, ~(cell_heel | cell_notch)),
    )


# region debug loop
# =============================================================================
# DEBUG loading loop: a plain Python for-loop (not lax.scan) so we can do host
# I/O every step -- compute energies / base reaction and stream fields into a
# single XDMF (+ .h5) time series.  Slower per step than scan, but the coarse
# mesh keeps it fast and this is what you want for debugging the base crack.
# =============================================================================
def run_loading_debug(field_u, field_d, preload_levels, levels,
                      sol_u, sol_d, history_old,
                      f_grav, face_pre, points, cells, mat_quad, dam_cell_mask,
                      base_dofs_x, base_dofs_y, crest_node,
                      max_stagger=10, save_every=1, output_dir=None, MODEL=None):
    """Quasi-static load loop driven by the reservoir WATER LEVEL.

    `preload_levels` ramps the elastic problem from empty to full reservoir while
    damage is frozen.  `levels` then starts at H and advances overflow with the
    staggered fracture solve active.

        f_ext = f_grav + hydro_load(level)
    """
    stepper = get_staggered_stepper(field_u, field_d, max_stagger)
    preload_stepper = get_elastic_preload_stepper(field_u)
    fixed_idx, fixed_vals = build_dirichlet_static(field_u, mat_quad)

    sol_u = np.asarray(sol_u)
    sol_d = np.asarray(sol_d)
    history_old = np.asarray(history_old)

    hist = {k: [] for k in (
        'level', 'overflow', 'E_pos', 'E_min', 'E_ela', 'E_frac',
        'base_Rx', 'base_Ry', 'max_d', 'crest_ux',
        'max_d_heel', 'max_d_notch', 'max_d_rest',
        'max_psi_heel', 'max_psi_notch', 'max_psi_rest')}

    pts3 = onp.asarray(points)
    cell_block = [(MESHIO_TYPE, onp.asarray(cells))]
    start = time.time()

    # Elastic/geostatic preload.  The same gravity vector is present throughout;
    # reservoir pressure is raised gradually to avoid a single large Newton jump.
    for j, lv in enumerate(preload_levels):
        f_ext_u = f_grav + hydro_load(face_pre, float(lv))
        sol_u = preload_stepper(sol_u, sol_d, fixed_idx, fixed_vals,
                                f_ext_u, mat_quad)
        print(f"Preload {j + 1:3d}/{len(preload_levels)}  level={float(lv):6.2f}m  "
              f"damage frozen  t={time.time() - start:.1f}s", flush=True)

    # Initialise (do not accumulate) tensile history from the equilibrated
    # self-weight + full-reservoir state.
    history_old = field_u.compute_history(
        sol_u, np.zeros_like(history_old), mat_quad)

    # meshio writes the .h5 with a basename relative to the CWD and references it
    # by basename in the .xdmf.  chdir into output_dir so BOTH land together (and
    # the .xdmf -> .h5 relative link resolves in ParaView).
    prev_cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        with meshio.xdmf.TimeSeriesWriter(f'koyna_{MODEL}.xdmf') as writer:
            writer.write_points_cells(pts3, cell_block)

            def _write(t, su, sd, fields):
                u = onp.asarray(su)
                u3 = onp.concatenate([u, onp.zeros((u.shape[0], 1))], axis=1)  # pad to 3D
                point_data = {'u': u3, 'd': onp.asarray(sd).reshape(-1, 1)}
                if LIGHT_OUTPUT:
                    writer.write_data(float(t), point_data=point_data)
                else:
                    writer.write_data(
                        float(t), point_data=point_data,
                        cell_data={'psi_plus': [fields['psi_plus']],
                                   'psi_minus': [fields['psi_minus']],
                                   'strain_energy': [fields['strain_energy']],
                                   'damage': [fields['d_cell']]})

            # Full-reservoir elastic preload, immediately before fracture is active.
            if LIGHT_OUTPUT:
                sc0, fld0 = None, None
            else:
                sc0, fld0 = compute_energy_fields(field_u, sol_u, sol_d, mat_quad,
                                                  dam_cell_mask)
            _write(H_FULL, sol_u, sol_d, fld0)

            for k, level in enumerate(levels):
                lv = float(level)
                ov = lv - H_FULL                         # overflow (<0 while filling)
                f_ext_u = f_grav + hydro_load(face_pre, lv)
                sol_u, sol_d, history_old = stepper(
                    sol_u, sol_d, history_old, fixed_idx, fixed_vals, f_ext_u, mat_quad)

                if LIGHT_OUTPUT:
                    sc = dict(E_pos=onp.nan, E_min=onp.nan, E_ela=onp.nan)
                    fld = None
                    E_frac = onp.nan
                    Rx, Ry = onp.nan, onp.nan
                else:
                    sc, fld = compute_energy_fields(field_u, sol_u, sol_d, mat_quad,
                                                    dam_cell_mask)
                    E_frac = compute_fracture_energy(field_d, sol_d)
                    Rx, Ry = compute_base_reaction(field_u, sol_u, sol_d, mat_quad,
                                                   f_ext_u, base_dofs_x, base_dofs_y)
                max_d = float(np.max(sol_d))
                region = regional_maxima(points, cells, sol_d, fld)
                # horizontal displacement AT THE CREST (the paper's Fig. 4d
                # abscissa).  Must be a specific node: max|u_x| over all nodes
                # would pick up the foundation, which moves more than the dam.
                crest_ux = float(np.abs(sol_u[crest_node, 0]))

                for key, val in (('level', lv), ('overflow', ov),
                                 ('E_pos', sc['E_pos']), ('E_min', sc['E_min']),
                                 ('E_ela', sc['E_ela']), ('E_frac', E_frac),
                                 ('base_Rx', Rx), ('base_Ry', Ry),
                                 ('max_d', max_d), ('crest_ux', crest_ux)):
                    hist[key].append(val)
                for key, val in region.items():
                    hist[key].append(val)

                print(f"Step {k + 1:4d}/{len(levels)}  ovfl level={lv:6.2f}m  "
                      f"ovf={ov:+5.2f}m  max_d={max_d:.3f}  "
                      f"d(heel/notch)={region['max_d_heel']:.3f}/"
                      f"{region['max_d_notch']:.3f}  "
                      f"crest_ux={crest_ux * 1e3:6.2f}mm  E_pos={sc['E_pos']:.3e}  "
                      f"Rx={Rx:.3e}  Ry={Ry:.3e}  t={time.time() - start:.1f}s",
                      flush=True)

                if (k % save_every == 0) or (k == len(levels) - 1):
                    _write(lv, sol_u, sol_d, fld)
    finally:
        os.chdir(prev_cwd)

    hist = {k: onp.asarray(v) for k, v in hist.items()}
    onp.savez(os.path.join(output_dir, f'history_koyna_{MODEL}.npz'), **hist)
    return sol_u, sol_d, history_old, hist


# region Main
# =============================================================================
# Driver
# =============================================================================
def main():
    t_wall0 = time.time()

    # --- resolution ---------------------------------------------------------
    # NX sets the number of columns across the section.  Note the section is only
    # 19.3 m wide at the neck, so NX=140 already gives dx = 19.3/140 = 0.14 m there
    # (fine enough for l=0.3) while the base stays coarse at 70/140 = 0.5 m.
    # The ROWS are graded around the crack (see graded_ys), so a small l is cheap.
    COARSE = os.environ.get('PFF_COARSE', '0' if PAPER_MODE else '1') == '1'
    uniform_h = float(os.environ.get('PFF_UNIFORM_H', '0'))
    if COARSE:
        NX = 70
        n_fill, n_ovf, ovf_max = 8, 20, 8.0
        SAVE_EVERY = 1
    else:
        # Keep the largest horizontal element in the entire fracture corridor at
        # or below the paper's coarse 8 cm resolution.  Because this mapped mesh
        # widens downward, sizing NX only at the 19.3 m notch elevation is not
        # sufficient; the widest corridor row occurs at FRACTURE_Y_MIN.
        h_target = float(os.environ.get('PFF_H_TARGET', 0.08))
        corridor_width = float(x_right_of_y(onp.array(FRACTURE_Y_MIN)))
        NX = int(os.environ.get('PFF_NX',
                                max(140, int(onp.ceil(corridor_width / h_target)))))
        n_fill, n_ovf, ovf_max = 12, 50, 10.0
        SAVE_EVERY = 2

    if uniform_h > 0.0:
        # A genuinely global size requirement must be based on the widest row,
        # i.e. the 70 m base, rather than only the narrow fracture corridor.
        NX = int(os.environ.get('PFF_NX',
                                int(onp.ceil(GEOM['W_base'] / uniform_h))))

    # Runtime overrides.  Increasing PFF_N_OVF reduces the load increment;
    # increasing PFF_OVF_MAX extends the simulation to a higher water level.
    n_fill = int(os.environ.get('PFF_N_FILL', n_fill))
    n_ovf = int(os.environ.get('PFF_N_OVF', n_ovf))
    ovf_max = float(os.environ.get('PFF_OVF_MAX', ovf_max))
    SAVE_EVERY = int(os.environ.get('PFF_SAVE_EVERY', SAVE_EVERY))
    MAX_STAGGER = int(os.environ.get('PFF_MAX_STAGGER', 20))
    if n_fill < 1 or n_ovf < 1 or ovf_max <= 0.0 or SAVE_EVERY < 1:
        raise ValueError('PFF_N_FILL, PFF_N_OVF, PFF_OVF_MAX and '
                         'PFF_SAVE_EVERY must all be positive')

    # Elastic preload and fracture loading are deliberately separate.  Damage is
    # frozen during reservoir filling, then activated at zero overflow.
    preload_levels = onp.linspace(0.0, H_FULL, n_fill + 1)
    levels = onp.linspace(H_FULL, H_FULL + ovf_max, n_ovf + 1)

    # --- Koyna mesh (graded rows) + upstream-face initial crack (built ONCE) ---
    pts, cells, y_notch, ys = koyna_mesh(
        NX, None, uniform_h=uniform_h if uniform_h > 0.0 else None)
    n_before = len(pts)
    pts, cells = insert_notch(pts, cells, y_notch, CRACK_LEN)   # crack: x=0 -> CRACK_LEN
    n_slit = len(pts) - n_before
    n_dam_cells = len(cells)

    # --- elastic foundation, bonded to the dam by shared base nodes -----------
    if FOUNDATION:
        pts, cells, is_rock_cell = add_foundation(pts, cells, ys)
    else:
        is_rock_cell = onp.zeros(len(cells), dtype=bool)
    dam_cell_mask = np.asarray(~is_rock_cell, dtype=np.float64)

    _, field_d, field_u = build_fields(pts, cells)

    # --- per-quad material: concrete in the dam, rock in the foundation -------
    mat_quad = material_quad(field_u, is_rock_cell)

    # --- external loads: self-weight once, hydrostatic rebuilt per water level --
    f_grav = gravity_load(field_u, is_rock_cell)      # massless foundation
    face_pre = upstream_face_precompute(field_u)

    # --- supported DOFs (for the reaction diagnostic): rock boundary if there is
    #     a foundation, otherwise the clamped dam base --------------------------
    P = onp.asarray(pts)
    if FOUNDATION:
        sup = (onp.abs(P[:, 1] + ROCK_DEPTH) < 1e-6) | (onp.abs(P[:, 0] + ROCK_EXT) < 1e-6) \
            | (onp.abs(P[:, 0] - (GEOM['W_base'] + ROCK_EXT)) < 1e-6)
    else:
        sup = P[:, 1] < 1e-6
    base_nodes = onp.where(sup)[0]
    base_dofs_x = np.array(base_nodes * 2 + 0, dtype=np.int32)
    base_dofs_y = np.array(base_nodes * 2 + 1, dtype=np.int32)

    # crest node (upstream crest corner) for the Fig. 4d displacement readout
    crest_node = int(onp.argmin((P[:, 0] - 0.0) ** 2 + (P[:, 1] - GEOM['H']) ** 2))

    sol_u = onp.zeros((len(pts), 2))
    sol_d = onp.zeros((len(pts), 1))
    history_old = onp.zeros((field_u.num_cells, field_u.num_quads))

    dy = onp.diff(ys)
    dx_neck = float(x_right_of_y(onp.array(y_break))) / NX
    n_dfree = int(onp.sum(onp.asarray(pts)[:, 1] <= BASE_FREE_H))
    print(f"Koyna dam | Model = {MODEL} | mesh = {MESH_TYPE} "
          f"({'UNIFORM h=' + format(uniform_h, '.3f') + ' m' if uniform_h > 0.0 else ('COARSE test' if COARSE else 'graded production')})")
    print(f"  mesh      : {len(pts)} nodes, {len(cells)} cells, {len(ys) - 1} graded rows "
          f"(dy {dy.min():.3f}-{dy.max():.3f} m), dx at neck = {dx_neck:.3f} m")
    print(f"  length    : l = {l:.3f} m  ->  "
          f"{'h_uniform' if uniform_h > 0.0 else 'h_fine'} = {dy.min():.3f} m (l/h = "
          f"{l / dy.min():.2f}), band ~2l = {2 * l:.2f} m vs crack {CRACK_LEN:.2f} m "
          f"({CRACK_LEN / (2 * l):.1f}x)")
    print(f"  crack     : {n_slit} duplicated nodes, UPSTREAM face, y={y_notch:.2f} m, "
          f"length={CRACK_LEN:.2f} m")
    if PAPER_MODE:
        print(f"  paper mode: ON -- rigid base; damage admissible only in "
              f"x=[0,{FRACTURE_X_MAX:.1f}], "
              f"y=[{FRACTURE_Y_MIN:.1f},{FRACTURE_Y_MAX:.1f}] m")
    if PAPER_MODE:
        print(f"  support   : full dam base rigidly clamped ({len(base_nodes)} nodes)")
        print("              no artificial bottom damage layer; phase field is fixed "
              "outside the crack corridor")
    elif FOUNDATION:
        n_rock = int(is_rock_cell.sum())
        print(f"  foundation: ON  -- rock E={E_ROCK/1e3:.0f} GPa, nu={NU_ROCK}, MASSLESS; "
              f"extent +-{ROCK_EXT:.0f} m, depth {ROCK_DEPTH:.0f} m ({n_rock} cells)")
        print(f"              dam base is BONDED (shared nodes, no Dirichlet BC); "
              f"supports on the outer rock: {len(base_nodes)} nodes")
        print(f"              damage pinned to 0 in the rock + interface (y <= 0)")
    else:
        n_rows_pinned = int(onp.sum(ys <= BASE_FREE_H))
        y_first_free = ys[n_rows_pinned] if n_rows_pinned < len(ys) else float('nan')
        print(f"  foundation: OFF -- dam base rigidly clamped ({len(base_nodes)} nodes); "
              f"damage pinned to 0 for y <= {BASE_FREE_H:.2f} m -> {n_rows_pinned} rows "
              f"({n_dfree} nodes), first free row at y = {y_first_free:.2f} m")
        if n_rows_pinned < 3:
            print("  WARNING: the damage-free base layer spans fewer than 3 node rows; "
                  "the crack will just form on the first free row. Raise PFF_BASE_FREE.")
    print(f"  schedule  : elastic preload 0->{H_FULL:.0f} m in {n_fill} steps "
          f"(damage frozen), then overflow 0->{ovf_max:.0f} m in {n_ovf} steps")
    print(f"  output    : {'LIGHT (u+d point fields; energy/reaction diagnostics skipped)' if LIGHT_OUTPUT else 'FULL diagnostics'}")
    if MODEL == 'AT2':
        sig_c = (9. / 16.) * onp.sqrt(E * G_c / (6. * l))
        print(f"  AT2 effective nucleation strength ~ {sig_c:.3f} MPa "
              f"(vs f_t = {f_t:.2f} MPa). Lower l to raise it.")

    plot_quad_mesh(pts, cells, os.path.join(output_dir, f'mesh_start_{MODEL}.png'),
                   f'Koyna mesh ({MESH_TYPE}, {len(cells)} cells, graded)',
                   show_notch=True, notch_tip=CRACK_LEN)
    if FOUNDATION:   # full domain incl. the rock, so the foundation can be eyeballed
        plot_quad_mesh(pts, cells, os.path.join(output_dir, f'mesh_full_{MODEL}.png'),
                       f'Dam + foundation ({len(cells)} cells)', clip_to_dam=False)

    start = time.time()

    # region run loading (DEBUG: energies + reaction + XDMF time series)
    sol_u, sol_d, history, hist = run_loading_debug(
        field_u, field_d, preload_levels, levels, sol_u, sol_d, history_old,
        f_grav, face_pre, pts, cells, mat_quad, dam_cell_mask,
        base_dofs_x, base_dofs_y, crest_node,
        max_stagger=MAX_STAGGER, save_every=SAVE_EVERY,
        output_dir=output_dir, MODEL=MODEL)

    # region save
    solve_time = time.time() - start
    wall_time = time.time() - t_wall0
    n_steps = (len(preload_levels) - 1) + len(levels)

    onp.savez(os.path.join(output_dir, f'final_koyna_{MODEL}.npz'),
              points=onp.asarray(pts), cells=onp.asarray(cells),
              sol_d=onp.asarray(sol_d).reshape(-1),
              n_cells=int(len(cells)), n_nodes=int(len(pts)),
              solve_time=float(solve_time), wall_time=float(wall_time),
              H=float(GEOM['H']), W_base=float(GEOM['W_base']), **hist)

    summary = (
        "==================== KOYNA DAM / TIMING SUMMARY ====================\n"
        f"  model                : {MODEL}\n"
        f"  mesh                 : {MESH_TYPE} graded ({len(cells)} cells, {len(pts)} nodes)\n"
        f"  length scale         : l = {l:.3f} m  (band ~2l = {2 * l:.2f} m, "
        f"crack = {CRACK_LEN:.2f} m)\n"
        f"  loading              : elastic preload to full reservoir (damage frozen), "
        f"then overflow 0->{ovf_max:.1f} m with fracture active\n"
        f"  initial crack        : upstream face, y={CRACK_Y:.1f} m, len={CRACK_LEN:.2f} m\n"
        f"  base                 : {'rigidly clamped (paper)' if PAPER_MODE else ('elastic foundation' if FOUNDATION else 'rigidly clamped')}\n"
        f"  fracture region      : x=[0,{FRACTURE_X_MAX:.1f}], "
        f"y=[{FRACTURE_Y_MIN:.1f},{FRACTURE_Y_MAX:.1f}] m\n"
        f"  scheme               : staggered (alternating minimization)\n"
        f"  d-solver             : {'projected gradient (bound-constrained)' if MODEL in ('AT1','CZM') else 'assembled CG + clip'}\n"
        f"  output               : koyna_{MODEL}.xdmf (+ .h5) time series\n"
        f"  output mode          : {'light' if LIGHT_OUTPUT else 'full diagnostics'}\n"
        f"  load steps           : {n_steps}\n"
        f"  final max damage     : {onp.max(onp.asarray(sol_d)):.3f}\n"
        f"  TOTAL solve time     : {solve_time:.1f} s  ({solve_time / 60.:.2f} min)\n"
        f"  TOTAL wall time      : {wall_time:.1f} s  ({wall_time / 60.:.2f} min)\n"
        "===================================================================")
    print("\n" + summary)
    with open(os.path.join(output_dir, f'timing_{MODEL}.txt'), 'w') as fh:
        fh.write(summary + "\n")

    plot_quad_mesh(pts, cells, os.path.join(output_dir, f'solution_final_{MODEL}.png'),
                   f'Koyna final damage (max d = {onp.max(onp.asarray(sol_d)):.2f})',
                   sol_d=sol_d)

    # --- diagnostics.  Panel 1 uses the paper's Fig. 4d axes and shows only the
    #     overflow phase; the rest show the whole history vs water level. --------
    lv = hist['level']
    ov = hist['overflow']
    m = ov >= -1e-9                                   # overflow phase only
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
    axs = axs.ravel()
    axs[0].plot(hist['crest_ux'][m] * 1e3, ov[m], 'b-o', ms=3)
    axs[0].set_xlabel('crest displacement [mm]'); axs[0].set_ylabel('overflow [m]')
    axs[0].set_title('Overflow vs crest disp. (cf. Fig. 4d)')
    axs[1].plot(ov, hist['max_d_heel'], 'r-o', ms=3, label='heel $y<10$ m')
    axs[1].plot(ov, hist['max_d_notch'], 'b-s', ms=3, label='notch corridor')
    axs[1].plot(ov, hist['max_d_rest'], 'k-^', ms=3, label='remainder')
    axs[1].set_xlabel('overflow [m]'); axs[1].set_ylabel('regional max damage $d$')
    axs[1].set_title('Competing crack regions'); axs[1].legend(fontsize=8)
    axs[2].plot(lv, hist['E_pos'], 'r-o', ms=3, label='$\\int\\psi^+$ (driving)')
    axs[2].plot(lv, hist['E_ela'], 'g-s', ms=3, label='$\\int[g\\psi^+{+}\\psi^-]$ (elastic)')
    axs[2].plot(lv, hist['E_frac'], 'm-^', ms=3, label='fracture energy')
    axs[2].axvline(H_FULL, color='k', ls=':', lw=1)
    axs[2].set_xlabel('water level [m]'); axs[2].set_ylabel('energy [MPa$\\cdot$m$^2$]')
    axs[2].set_title('Energies'); axs[2].legend(fontsize=8)
    axs[3].plot(lv, hist['base_Rx'], 'b-o', ms=3, label='base shear $R_x$')
    axs[3].plot(lv, hist['base_Ry'], 'k-s', ms=3, label='vertical $R_y$')
    axs[3].axvline(H_FULL, color='k', ls=':', lw=1)
    axs[3].set_xlabel('water level [m]'); axs[3].set_ylabel('base reaction [MN]')
    axs[3].set_title('Base reaction'); axs[3].legend(fontsize=8)
    axs[4].plot(ov, hist['max_psi_heel'], 'r-o', ms=3, label='heel $y<10$ m')
    axs[4].plot(ov, hist['max_psi_notch'], 'b-s', ms=3, label='notch corridor')
    axs[4].plot(ov, hist['max_psi_rest'], 'k-^', ms=3, label='remainder')
    axs[4].set_xlabel('overflow [m]'); axs[4].set_ylabel(r'max $\psi^+$ [MPa]')
    axs[4].set_title('Tensile driving-energy maxima'); axs[4].legend(fontsize=8)
    axs[5].axis('off')
    for ax in axs:
        ax.grid(which='major', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'diagnostics_{MODEL}.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
