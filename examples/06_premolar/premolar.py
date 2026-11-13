"""Three-dimensional premolar under occlusal loading (paper Section 4.2.2).

A maxillary second premolar, anchored in alveolar bone through the periodontal
ligament (PDL), is loaded on its occlusal surface by a rigid sphere of 4 mm
diameter driven vertically under displacement control. The configuration
follows the intact-tooth study of Zhu et al. (Eng. Fract. Mech. 343, 2026).
In the comments below, "the paper", "the PD paper" and its figure, section and
equation numbers (for example "Fig. 5") refer to that peridynamic study, not to
the Axion paper; "an earlier run" refers to computations made while the example
was set up.

Geometry. The patient-specific geometry of the reference study is not
available, so an idealized premolar with the dimensions of an average maxillary
second premolar is built here with Gmsh: crown 8 mm high and 9 x 7 mm wide with
two cusps separated by a central fissure 1.7 mm deep with flanks inclined at
45 degrees, enamel 1.2 mm thick thinning to 0.3 mm at the cemento-enamel
junction, root 14 mm, PDL 0.2 mm, bone block with a cortical shell. Tooth, PDL
and bone are meshed as conforming volumes with linear tetrahedra (TET4), so
every interface coincides with element faces. The six constituents (enamel,
dentin, pulp, PDL, cancellous and cortical bone) enter only as material data at
the quadrature points; only enamel and dentin of the crown can damage.

Model. Hybrid phase-field formulation: the stress is degraded isotropically,
sigma = (omega(d) + k_res) (lambda tr(eps) I + 2 mu eps), and the crack is
driven by the Rankine-type measure <sigma_1>^2 / 2E of the largest principal
effective stress, which vanishes under the compression beneath the indenter
(PFF_DRIVING=spectral selects the spectral strain split instead). The PF-CZM
model is used; its damage subproblem is solved by projected Jacobi iteration.

Contact. Frictionless unilateral spring acting on the occlusal surface within a
short range d_s of the sphere; its contribution to the tangent is obtained by
automatic differentiation like the bulk terms. No crack, weak zone or preferred
path is introduced.

The loading schedule of the paper (increments of 0.004 mm, refined to 0.002 mm
for delta in [0.164, 0.176] mm) is set in run_premolar.sh.

Units: N, mm, MPa.
"""
import json
import os
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import jax
import jax.numpy as np
import numpy as onp
import gmsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

import axion
from axion import DirichletBC, Field, Mesh, negative_part, positive_part
from axion import eigvalsh_3x3 as eigvalsh_sym3x3

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10

crt_file_path = os.path.dirname(os.path.abspath(__file__))
_output_setting = os.environ.get('PFF_OUTPUT_DIR', 'output_premolar')
output_dir = (_output_setting if os.path.isabs(_output_setting)
              else os.path.join(crt_file_path, _output_setting))
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vtk_dir, exist_ok=True)

safe_plus = positive_part
safe_minus = negative_part

# region Parameters
MODEL = os.environ.get('PFF_MODEL', 'CZM')
assert MODEL in ('AT1', 'AT2', 'CZM'), f"PFF_MODEL must be AT1/AT2/CZM, got {MODEL}"
ELE = 'TET4'                        # 3D linear tetrahedra

# --- tissue properties from Table 1 [MPa], with phase-field Gc [MPa.mm] ---
E_ENAMEL, NU_ENAMEL = 84.1e3, 0.30
E_DENTIN, NU_DENTIN = 18.1e3, 0.31
E_PULP, NU_PULP = 2.0, 0.45
E_PDL, NU_PDL = 10.0, 0.45
E_CANCEL, NU_CANCEL = 1.37e3, 0.30
E_CORT, NU_CORT = 13.7e3, 0.26
# Fracture energy: MEASURED values (the aim is a realistic crack
# pattern, not a replica of the PD paper's calibration).
#   enamel  Gc = 13.1 J/m2 = 0.0131 MPa.mm   (K_IC ~ 1.0 MPa sqrt(m): Bechtle et
#           al. 2010; the value used by Yang et al., JMBBM 157 (2024), Table 1)
#   dentin  Gc = 161.8 J/m2 = 0.1618 MPa.mm  (K_IC ~ 1.8 MPa sqrt(m): Imbeni et
#           al. 2005; Yang et al. 2024, Table 1)
# The PD paper's own s0-based pair (s0 = sqrt(5 G0/(9 K delta)) with the 0.45 mm
# horizon: 0.0817 / 0.1575 MPa.mm) is what earlier runs used; restore it with
#   PFF_GC_ENAMEL=0.0817 PFF_GC_DENTIN=0.1575 PFF_FT_ENAMEL=100.9 PFF_FT_DENTIN=63.4
# Note the dentin/enamel ratio of 12: Yang et al. report that with it the AT2
# crack arrests at the DEJ.  With PF-CZM the DEJ crossing is set by the dentin
# strength below, so the enamel crack arrests and the dentin cracks later, at
# a higher load -- the clinical "cracked tooth" sequence.
GC_ENAMEL = float(os.environ.get('PFF_GC_ENAMEL', 0.0131))
GC_DENTIN = float(os.environ.get('PFF_GC_DENTIN', 0.1618))
# Nonfracturing tissues receive a large numerical Gc in addition to d=0 nodal BCs.
GC_NONFRACTURE = float(os.environ.get('PFF_GC_NONFRACTURE', 1000.0))
# Tensile strength per tissue (PF-CZM needs f_t; AT1/AT2 ignore it).
#   enamel  40 MPa  (measured UTS 11-42 MPa, Giannini et al. 2004; the AT2
#           model of Yang et al. 2024 with l0 = 0.1 mm has sigma_c = 34 MPa)
#   dentin  60 MPa  (bulk UTS 50-100 MPa, Sano et al. 1994; Yang's sigma_c = 58)
# The PD paper's critical stretch s0 (Table 1) maps to f_t = E*s0 = 100.9 /
# 63.4 MPa; PFF_S0_* are kept so that pair can be restored (see above).
# The resulting characteristic lengths l_ch = E Gc / f_t^2 are 0.69 mm (enamel)
# and 0.81 mm (dentin); PF-CZM softening is stable for l < 2 l_ch / pi = 0.44 /
# 0.52 mm, so l = 0.25 mm is comfortably inside that regime.
S0_ENAMEL = float(os.environ.get('PFF_S0_ENAMEL', 1.2e-3))
S0_DENTIN = float(os.environ.get('PFF_S0_DENTIN', 3.5e-3))
F_T_ENAMEL = float(os.environ.get('PFF_FT_ENAMEL', 40.0))
F_T_DENTIN = float(os.environ.get('PFF_FT_DENTIN', 60.0))
f_t = F_T_ENAMEL                      # kept for back-compatible reporting

def _lame(E, nu):
    return E / (2. * (1. + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

MU_E, LMBDA_E = _lame(E_ENAMEL, NU_ENAMEL)
MU_D, LMBDA_D = _lame(E_DENTIN, NU_DENTIN)
MU_P, LMBDA_P = _lame(E_PULP, NU_PULP)
MU_L, LMBDA_L = _lame(E_PDL, NU_PDL)
MU_CB, LMBDA_CB = _lame(E_CANCEL, NU_CANCEL)
MU_CT, LMBDA_CT = _lame(E_CORT, NU_CORT)

# Length scale.  0.25 mm is ~1/3 of l_ch for both tissues (regularisation, not
# a material parameter in PF-CZM), resolves the 1.2-1.5 mm enamel cap with
# ~5 elements across the band at h = l/2-l/3, and matches the runs since 1406.
l = float(os.environ.get('PFF_L', 0.25))


def _a1(E, gc, ft):
    """PF-CZM (Wu) softening parameter a1 = 4 l_ch / (pi l), l_ch = E Gc / f_t^2.

    a1 is what makes the band reproduce the prescribed strength, so it is a
    MATERIAL constant: enamel and dentin have different E and Gc and therefore
    different a1.  Using the enamel value everywhere left dentin with an
    effective strength of ~76 MPa instead of f_t.
    """
    return 4. / onp.pi * (E * gc / (ft ** 2)) / l


L_CH_ENAMEL = E_ENAMEL * GC_ENAMEL / (F_T_ENAMEL ** 2)
L_CH_DENTIN = E_DENTIN * GC_DENTIN / (F_T_DENTIN ** 2)
A1_ENAMEL = _a1(E_ENAMEL, GC_ENAMEL, F_T_ENAMEL)
A1_DENTIN = _a1(E_DENTIN, GC_DENTIN, F_T_DENTIN)
H_CRIT_ENAMEL = F_T_ENAMEL ** 2 / (2. * E_ENAMEL)
H_CRIT_DENTIN = F_T_DENTIN ** 2 / (2. * E_DENTIN)
# Residual stiffness. At d->1 the element stiffness falls to K_RES*E; 1e-8 spans
# 8 orders of magnitude across the mesh and makes the Jacobi-preconditioned CG
# very ill-conditioned. Raise to 1e-6 if the solve produces NaN.
K_RES = float(os.environ.get('PFF_K_RES', 1e-8))

# --- idealised full premolar + supporting bone geometry [mm] ---
# Sized as an average maxillary second premolar (crown 8 mm tall, 9 mm
# bucco-lingual (x) by 7 mm mesio-distal (z), root 14 mm) and checked against
# the paper's Figs 4-5, where the 4 mm ball is the scale bar: crown height
# ~0.8 x crown width, root ~1.3 x, pulp horn ~0.6 x below the cusps, alveolar
# crest ~1.5 mm below the CEJ, bone block ~1.2 x crown width.  The previous
# outline had a 12 mm crown, a 9 mm root, cusp tips only 3.2 mm apart and --
# decisive for a phase-field model -- the ball bearing on a 27 deg cusp
# incline.  Yang et al. (JMBBM 157 (2024) 106655, Fig. 14) show with an AT2
# phase field on a CBCT molar that the crack starts at the fissure tip for a
# cusp angle of 50 deg (incline 40 deg from horizontal) but at the BALL
# CONTACT / DEJ for 70 deg (incline 20 deg): "stress no longer concentrates at
# the concavity".  That is precisely where earlier runs nucleated.  The
# fissure is therefore a 45 deg V (cusp angle 45 deg) wide enough for the ball
# to bear on its flanks, with a rounded floor.  Cusp tips stay at y = 20.
GEOM = dict(H=20.0, W=9.0, y_cej=12.0, y_apex=-2.0)
H_TOOTH, Y_CEJ, Y_APEX = GEOM['H'], GEOM['y_cej'], GEOM['y_apex']
Z_MIN = float(os.environ.get('PFF_Z_MIN', -3.5))
Z_MAX = float(os.environ.get('PFF_Z_MAX', 3.5))
if Z_MAX <= Z_MIN:
    raise ValueError('PFF_Z_MAX must be greater than PFF_Z_MIN')
LZ = Z_MAX - Z_MIN
# Enamel: T_ENAMEL over the occlusal part of the crown, tapering to
# T_ENAMEL_CEJ over the cervical ENAMEL_TAPER_H mm (real enamel thins to
# nothing at the CEJ).
T_ENAMEL = float(os.environ.get('PFF_T_ENAMEL', 1.2))
T_ENAMEL_CEJ = float(os.environ.get('PFF_T_ENAMEL_CEJ', 0.3))
ENAMEL_TAPER_H = float(os.environ.get('PFF_ENAMEL_TAPER_H', 3.0))
PDL_THICKNESS = float(os.environ.get('PFF_PDL_THICKNESS', 0.20))
# The alveolar crest sits below the CEJ (paper Fig. 5a shows a band of exposed
# root dentin between enamel and bone).  The old model put the bone top AT the
# CEJ, which is also what produced the re-entrant-edge mode there.
Y_CREST = float(os.environ.get('PFF_Y_CREST', Y_CEJ - 1.5))
BONE_X_HALF = float(os.environ.get('PFF_BONE_X_HALF', 6.0))
BONE_BOTTOM = float(os.environ.get('PFF_BONE_BOTTOM', -5.0))
CORTICAL_THICKNESS = float(os.environ.get('PFF_CORTICAL_THICKNESS', 1.0))
# Pulp horns 5 mm below the cusp tips (paper Fig. 5b: ~0.6 x crown width), so
# the crack crosses ~1.2 mm enamel + ~2 mm dentin before reaching the pulp, as
# in Fig. 5 (0.24 mm: through enamel; 0.30 mm: at the pulp).
PULP_TOP = float(os.environ.get('PFF_PULP_TOP', H_TOOTH - 5.0))
PULP_BOTTOM = Y_APEX + 1.5
# Figure 5 concerns crown fracture.  Keeping the phase field above the
# cemento-enamel junction removes the non-physical re-entrant-edge mode at the
# alveolar crest while leaving the elastic root/PDL/bone load path intact.
FRACTURE_Y_MIN = float(os.environ.get('PFF_FRACTURE_Y_MIN', Y_CEJ + 1.0))
# Rounded fissure floor.  A perfectly sharp V-notch (zero tip radius) is a
# stress singularity that made the crack initiate at 297 N / delta=0.074 mm
# instead of the paper's 693 N / 0.21 mm.  Real occlusal fissures are rounded,
# typically 0.1-0.3 mm.  Set PFF_FISSURE_RADIUS=0 to restore the sharp notch.
FISSURE_RADIUS = float(os.environ.get('PFF_FISSURE_RADIUS', 0.20))
# V-fissure between the two cusps: vertex FISSURE_DEPTH below the tips, tips at
# x = +-CUSP_HALF_SPAN.  Default 1.7 / 1.7 -> 45 deg incline: the R = 2 mm ball
# bears on the flanks at |x| = 2 sin 45 = 1.41 mm, inside the V and 0.8 mm
# above the floor, so the wedge action splits the crown at the fissure.
# PFF_FISSURE_DEPTH=0.62 reproduces Yang's flat 70 deg case (incline 20 deg),
# where the crack is expected to move to the contact point instead.
FISSURE_DEPTH = float(os.environ.get('PFF_FISSURE_DEPTH', 1.7))
CUSP_HALF_SPAN = float(os.environ.get('PFF_CUSP_HALF_SPAN', 1.7))
if not 0.5 <= CUSP_HALF_SPAN <= 2.4:
    raise ValueError('PFF_CUSP_HALF_SPAN must lie in [0.5, 2.4] mm for this outline')
CUSP_INCLINE_DEG = float(onp.degrees(onp.arctan2(FISSURE_DEPTH, CUSP_HALF_SPAN)))
CUSP_ANGLE_DEG = 90.0 - CUSP_INCLINE_DEG        # Yang et al. 2024: from the vertical
_FISSURE_V = (0.0, H_TOOTH - FISSURE_DEPTH)
_FISSURE_A = (-CUSP_HALF_SPAN, H_TOOTH)
_FISSURE_B = (CUSP_HALF_SPAN, H_TOOTH)


def _fillet(A, V, B, r, n=9):
    """Circular fillet of radius r replacing corner V between rays V->A, V->B.

    Returns the arc points ordered A-side first.  The tangent points lie ON the
    original legs, so surface locators keyed to the straight segments still
    match the faces either side of the arc.
    """
    A, V, B = (onp.asarray(x, dtype=onp.float64) for x in (A, V, B))
    if r <= 0.0:
        return [tuple(V)]
    ua = (A - V) / onp.linalg.norm(A - V)
    ub = (B - V) / onp.linalg.norm(B - V)
    half = 0.5 * onp.arccos(onp.clip(ua @ ub, -1.0, 1.0))   # half the corner angle
    d = r / onp.tan(half)                                    # vertex -> tangent point
    bis = (ua + ub) / onp.linalg.norm(ua + ub)
    C = V + bis * (r / onp.sin(half))                        # arc centre
    Ta, Tb = V + ua * d, V + ub * d
    a0 = onp.arctan2(*(Ta - C)[::-1])
    a1 = onp.arctan2(*(Tb - C)[::-1])
    if a1 < a0:
        a1 += 2 * onp.pi
    if a1 - a0 > onp.pi:                                     # take the minor arc
        a0, a1 = a1 - 2 * onp.pi, a0
    ang = onp.linspace(a0, a1, n)
    return [(float(C[0] + r * onp.cos(t)), float(C[1] + r * onp.sin(t))) for t in ang]


_FISSURE_ARC = _fillet(_FISSURE_A, _FISSURE_V, _FISSURE_B, FISSURE_RADIUS)
# Deepest point of the (possibly rounded) fissure -- the Fig. 5 initiation site.
FISSURE_XY = min(_FISSURE_ARC, key=lambda p: p[1])
CUSP_TIP = (CUSP_HALF_SPAN, H_TOOTH)
# Fissure "column" used by the per-step H_fis report and the diagnostics.
FISSURE_COL_HALF_X = 0.45
FISSURE_COL_Y_MIN = FISSURE_XY[1] - 1.3
# Right half of the outline beyond the cusp tip: outer cusp slope down to the
# shoulder, crown side (max half-width 4.5 at y = 16.4), CEJ at 4.0, root
# tapering to the apex.
_CROWN_TOP_RIGHT = [(CUSP_HALF_SPAN + 0.9, 19.6), (CUSP_HALF_SPAN + 1.8, 18.8)]
_CROWN_SIDE_RIGHT = [(4.2, 17.6), (4.5, 16.4), (4.4, 14.5), (4.1, 12.6), (4.0, Y_CEJ)]
_ROOT_RIGHT = [(3.4, 8.5), (2.9, 5.5), (2.2, 2.0), (1.2, -1.0), (0.0, Y_APEX)]


def outline_half_width(y):
    """Half-width x(y) of the outline below the shoulder (monotone in y there)."""
    pts = onp.asarray(_CROWN_SIDE_RIGHT + _ROOT_RIGHT, dtype=onp.float64)[::-1]
    return float(onp.interp(y, pts[:, 1], pts[:, 0]))


ROOT_HALF_CREST = outline_half_width(Y_CREST)
# Occlusal polyline the ball can touch: left shoulder -> cusp tip -> V flank ->
# rounded floor -> V flank -> cusp tip -> right shoulder.  Its segments are the
# contact-surface locators (arc chords included, so the floor is loaded if the
# ball ever reaches it).
_OCCLUSAL_PROFILE = onp.asarray(
    [(-x, y) for (x, y) in reversed(_CROWN_TOP_RIGHT)] + [_FISSURE_A]
    + [tuple(p) for p in _FISSURE_ARC] + [_FISSURE_B] + _CROWN_TOP_RIGHT,
    dtype=onp.float64)
BALL_DIAMETER = 4.0
BALL_RADIUS = 0.5 * BALL_DIAMETER


def _ball_touch(profile, R, n=801):
    """Lowest centre height on x = 0 at which a sphere of radius R just touches
    the polyline, and the touch point.  The centre must satisfy
    yc >= y + sqrt(R^2 - x^2) at every profile point with |x| < R; the binding
    point is found by dense sampling of each segment."""
    P = onp.asarray(profile, dtype=onp.float64)
    best, at = -onp.inf, None
    t = onp.linspace(0.0, 1.0, n)[:, None]
    for a, b in zip(P[:-1], P[1:]):
        q = a + t * (b - a)
        ok = onp.abs(q[:, 0]) < R
        if not ok.any():
            continue
        yc = q[ok, 1] + onp.sqrt(R ** 2 - q[ok, 0] ** 2)
        i = int(onp.argmax(yc))
        if yc[i] > best:
            best, at = float(yc[i]), (float(q[ok][i, 0]), float(q[ok][i, 1]))
    return best, at


_Y0_AUTO, _TOUCH = _ball_touch(_OCCLUSAL_PROFILE, BALL_RADIUS)
# Smallest centre height whose spherical surface does not penetrate the cusp
# profile at delta=0 -- computed from the profile (it was 21.4361 mm for the
# old outline, hard-coded).
BALL_CENTER_Y0 = float(os.environ.get('PFF_BALL_CENTER_Y0', _Y0_AUTO))
CONTACT_X, CONTACT_Y = abs(_TOUCH[0]), _TOUCH[1]
# Contact law, following the paper's sec. 2.3: a SHORT-RANGE spring that acts
# once a surface point is within d_s of the ball (Eq. 14, d_s = 0.3 mm), not a
# hard penalty at zero gap.  A hard rigid-on-rigid contact gives a Hertzian
# pressure peak in the GPa range whose edge tension (1-2nu)p0/3 exceeds
# f_t=101 MPa already at ~50 N -- an earlier run (CZM, Gc=82/158, every other fix in)
# still formed a Hertzian RING crack at the contact edge at 339 N instead of a
# fissure crack.  Smearing the contact over d_s spreads the load over a patch
# of radius ~sqrt(2 R d_s) ~ 1.1 mm and removes that concentration; the paper
# says explicitly the 4 mm ball was chosen "to prevent stress concentration
# from directly affecting crack propagation".
# Penalty 5e3 N/mm^3: set by an elastic sweep (nucleation_map.py, k = 2e3 /
# 5e3 / 1e4 / 2e4 with d_s = 0.3).  At 5e3 the reaction is 606 N at delta =
# 0.21 mm (paper: 693 N), the FISSURE reaches the CZM threshold between delta =
# 0.21 and 0.24 (paper initiation 0.21), and the contact edge stays below
# threshold until 0.25 (fissure/contact ratio 1.23 at 0.21).  2e3 was too soft
# (385 N, fissure 0.39); the hard contact 2.5e5 gave 881 N and a Hertzian ring
# crack at the contact edge (an earlier run).  PFF_CONTACT_DS=0 PFF_CONTACT_PENALTY=2.5e5
# restores the old hard contact.
CONTACT_DS = float(os.environ.get('PFF_CONTACT_DS', 0.30))                # mm
CONTACT_PENALTY = float(os.environ.get('PFF_CONTACT_PENALTY', 5.0e3))     # N/mm^3
# The ball starts d_s ABOVE the just-touching height, so that force onset
# (surface point enters the d_s shell) is still at delta = 0.
BALL_Y_START = BALL_CENTER_Y0 + CONTACT_DS
R_EFF = BALL_RADIUS + CONTACT_DS          # radius at which the spring engages
# Optional: forbid damage inside the ball's contact shadow.  Once the occlusal
# surface can damage, the penalty traction acts on material that softens under
# it (damage -> deeper penetration -> larger traction -> more damage).  The
# paper's initiation site is the fissure at x=0, which lies OUTSIDE the contact
# band around |x| = CONTACT_X, so this guard does not suppress it.  Default off.
# Optional pre-existing crack, as in the paper's sec. 4.2 Cases I-III, where
# initial cracks are "represented by initially broken bonds within the
# prescribed crack region".  PFF_SEED_CRACK=1 plants the phase-field equivalent:
# the irreversible history H is set high in a thin plane so d -> 1 there.
# Default is OFF -- Fig. 5 itself is the INTACT tooth.
# Case-III of the paper = a crack on the occlusal surface at the central
# fissure, running into dentine: that is SEED_DEPTH ~ 2-3 mm.
SEED_CRACK = os.environ.get('PFF_SEED_CRACK', '0') == '1'
SEED_DEPTH = float(os.environ.get('PFF_SEED_DEPTH', 1.0))      # mm below fissure
SEED_HALF_THICK = float(os.environ.get('PFF_SEED_HALF_THICK', 0.0))  # 0 -> h/2
SEED_HALF_Z = float(os.environ.get('PFF_SEED_HALF_Z', 1e9))    # mesio-distal half-length
SEED_H_FACTOR = float(os.environ.get('PFF_SEED_H_FACTOR', 1e3))  # x H_crit

CONTACT_NO_DAMAGE = os.environ.get('PFF_CONTACT_NO_DAMAGE', '0') == '1'
# Diagnostic: pin d=0 everywhere, leaving pure elasticity + penalty contact.  If
# a run still produces NaN with this set, the fault is in the contact/elastic
# tangent, not in the phase field.
ELASTIC_ONLY = os.environ.get('PFF_ELASTIC_ONLY', '0') == '1'
# Margin MUST stay below the fissure's clearance from the ball (printed at
# start-up; 0.296 mm for the old outline, ~0.5 mm for the 45 deg V), or the
# guard swallows the fissure floor -- the very site where Fig. 5 nucleates
# the crack.  0.30 did exactly that on the old outline, by 4 microns.
CONTACT_SHADOW_MARGIN = float(os.environ.get('PFF_CONTACT_SHADOW_MARGIN', 0.05))
# VOLUMETRIC guard.  Pinning only the surface nodes does not work: the damage
# simply forms one band-width (2 l) BENEATH the pinned layer.  An earlier run (l=0.25,
# h=0.125) put its damage at |x|~1.5, y in [18.95,19.47] -- the cusp incline,
# 0.48 mm = 2 l below the occlusal surface, not at the fissure.  Refining makes
# it worse, because the contact-edge concentration sharpens with h while the
# rounded fissure does not.  So exclude a SHELL of depth ~3 l under the contact
# band, keyed on radius (not gap) so the fissure at rho < RHO_MIN stays free.
# PFF_CONTACT_SHADOW_DEPTH=0 restores the old surface-only guard.
CONTACT_SHADOW_DEPTH = float(os.environ.get('PFF_CONTACT_SHADOW_DEPTH', 3.0 * l))
CONTACT_SHADOW_RHO_MIN = float(os.environ.get('PFF_CONTACT_SHADOW_RHO_MIN', 0.30))
CONTACT_SHADOW_RHO_MAX = float(os.environ.get('PFF_CONTACT_SHADOW_RHO_MAX',
                                              CONTACT_X + 1.2))   # contact + patch
# Solver iteration caps.  These dominate the cost: every stagger runs a Newton
# loop whose iterations each do up to PFF_CG_MAXITER CG iterations, plus a
# projected-Jacobi damage solve of up to PFF_PG_MAXIT sweeps.  Lower them for a
# quick smoke test; the defaults are the production values.
NEWTON_MAX = int(os.environ.get('PFF_NEWTON_MAX', 20))
CG_MAXITER = int(os.environ.get('PFF_CG_MAXITER', 8000))
# 300 was never enough: an earlier run ended every step at 300/300 with the sweep
# still changing d by 5e-2 (tolerance 1e-7), i.e. an unconverged damage field.
PG_MAXIT = int(os.environ.get('PFF_PG_MAXIT', 2000))
# Damage sub-solver for CZM/AT1.  'pg' = projected Jacobi (the original).  It
# converges fast only while the LOCAL term dominates; with the paper-derived
# Gc = 82 J/m2 the gradient term Gc*l*lap(d) is 6.8x stronger than at Gc = 12
# and Jacobi crawls -- an earlier run had 109 fissure nodes 4.6x over threshold, all
# admissible, and d never passed 0.021 in 300 sweeps.  'newton' uses the same
# Newton+CG path as AT2 (with the [0,1] clip), which is insensitive to that.
# An earlier run settled it: Newton+CG gives d=1 over 49% of the enamel at delta=0.002
# (CZM's softening tangent is indefinite; CG diverges silently).  Stay on pg.
DAMAGE_SOLVER = os.environ.get('PFF_DAMAGE_SOLVER', 'pg').lower()
assert DAMAGE_SOLVER in ('pg', 'newton'), "PFF_DAMAGE_SOLVER must be pg|newton"
USE_PG = DAMAGE_SOLVER == 'pg' and MODEL in ('CZM', 'AT1')
CG_TOL = float(os.environ.get('PFF_CG_TOL', 1e-4))
# Crack driving force for the history H: 'rankine' (max principal effective
# stress, PF-CZM standard, zero in compression) or 'spectral' (Miehe strain
# split, which damages the contact zone -- see Elasticity.psi_plus).
DRIVING = os.environ.get('PFF_DRIVING', 'rankine').lower()
assert DRIVING in ('rankine', 'spectral'), "PFF_DRIVING must be rankine|spectral"
# Cap on the irreversible history H, in multiples of each tissue's H_crit.
# H is built from the EFFECTIVE (undegraded) stress, so inside a fully open
# crack the strain is huge and the effective stress with it: an earlier run reported
# H/H_crit at the fissure growing 18 -> 1.2e4 -> 1.4e5 -> 9.6e7 -> 4.2e9.  A
# quadrature point that absurd drags every node of its element with it, which
# is what turned the clean planar crack at delta=0.064 into a diffuse blob by
# delta=0.104.  d -> 1 needs only H = (2 l_ch / (pi l))^2 ~ 3 x H_crit, so a cap
# An earlier run's INTACT fissure notch reached a genuine elastic 18.6 x H_crit at
# delta=0.004, so the cap must sit well above that: 100 preserves every real
# stress concentration while cutting the open-crack artefact by seven orders.
# PFF_H_CAP=0 disables the cap.
H_CAP = float(os.environ.get('PFF_H_CAP', 100.0))
# Freeze H once a point is fully broken.  A cap alone still lets an open crack
# sit at the cap value forever and bleed damage into its neighbours through the
# shared element nodes; a broken point has no meaningful driving force left, so
# its history simply stops updating.  PFF_D_FREEZE=1 disables the freeze.
D_FREEZE = float(os.environ.get('PFF_D_FREEZE', 0.99))
# Stop once the specimen has failed: further steps only smear damage through the
# remaining ligament at a load of ~0 (an earlier run spent 11 h of wall clock there).
STOP_AFTER_FAILURE = os.environ.get('PFF_STOP_AFTER_FAILURE', '1') == '1'
STOP_LOAD_FRAC = float(os.environ.get('PFF_STOP_LOAD_FRAC', 0.05))
# Per-iteration progress lines from inside the jitted solver (Newton residual,
# stagger change).  A step of this size can be silent for 30-60 min otherwise.
VERBOSE = os.environ.get('PFF_VERBOSE', '1') == '1'
DELTA_MAX = float(os.environ.get('PFF_DELTA_MAX', 0.34))
DELTA_STEP = float(os.environ.get('PFF_DELTA_STEP', 0.01))
FIG5_TARGETS = onp.array([0.21, 0.24, 0.30, 0.34])
# Uniform mesh. Production default gives l/h=3; use 4 for a convergence run.
COARSE = os.environ.get('PFF_COARSE', '0') == '1'
H_OVER_L = float(os.environ.get('PFF_H_OVER_L', 3.0))
H_UNIFORM = float(os.environ.get('PFF_H', l / H_OVER_L))
if COARSE:
    H_UNIFORM = float(os.environ.get('PFF_H', 1.0))
H_FINE = H_UNIFORM
# The phase-field band only has to be resolved where damage is admissible (the
# crown).  Meshing the whole bone block at l/3 as well is what makes the default
# production run ~3M tets / ~2M u-DOFs, i.e. tens of GB for the assembled
# operator alone.  PFF_H_BONE coarsens everything below the crown box; it
# defaults to H_UNIFORM, so the uniform mesh is still the default.
H_COARSE = float(os.environ.get('PFF_H_BONE', H_UNIFORM))


# region model + geometry helpers
def crack_geo(d):
    if MODEL == 'AT1':
        return d
    if MODEL == 'AT2':
        return d ** 2
    return 2. * d - d ** 2                        # CZM


def degradation(d, a1):
    w = (1. - d) ** 2
    if MODEL in ('AT1', 'AT2'):
        return w
    Q = a1 * (d - 0.5 * d ** 2)                   # PF-CZM (Wu), a2=-0.5, a3=0
    return w / (w + Q)


def c_w():
    return {'AT1': 8. / 3., 'AT2': 2., 'CZM': np.pi}[MODEL]


_g_prime = jax.grad(degradation, argnums=0)
_w_prime = jax.grad(crack_geo)


def tooth_outline():
    """Idealised single-root premolar section, starting at the fissure floor."""
    right = ([tuple(p) for p in _FISSURE_ARC[len(_FISSURE_ARC) // 2:]]   # floor -> right flank
             + [_FISSURE_B] + _CROWN_TOP_RIGHT + _CROWN_SIDE_RIGHT + _ROOT_RIGHT)
    return right + [(-x, y) for (x, y) in reversed(right[1:-1])]


def outer_root_outline():
    """PDL outer boundary around the embedded part of the root.

    The PDL is a real geometric shell, rather than a coordinate label painted
    onto tetrahedra that cross the tooth/bone interface.  This is deliberately
    a simple idealised offset of the root used for the Figure-5 analogue.
    """
    t = PDL_THICKNESS
    right = ([(ROOT_HALF_CREST + t, Y_CREST)]
             + [(x + t, y) for (x, y) in _ROOT_RIGHT if y < Y_CREST and x > 0.0]
             + [(0.0, Y_APEX - t)])
    return right + [(-x, y) for (x, y) in reversed(right[:-1])]


_OCC_A = _OCCLUSAL_PROFILE[:-1]
_OCC_B = _OCCLUSAL_PROFILE[1:]


def _segment_surface_locator(a_host, b_host):
    """Return a locator for one planar extrusion of an occlusal segment."""
    a_host = onp.asarray(a_host, dtype=onp.float64)
    b_host = onp.asarray(b_host, dtype=onp.float64)

    def locator(point):
        a = np.asarray(a_host)
        b = np.asarray(b_host)
        p = point[:2]
        ab = b - a
        t = np.clip(np.sum((p - a) * ab) / (np.sum(ab * ab) + 1e-30), 0.0, 1.0)
        closest = a + t * ab
        return np.sum((p - closest) ** 2) <= 1e-12

    return locator


# Keep the planes separate.  A single union locator can accidentally select
# internal tetrahedron faces whose vertices sit on two adjoining cusp planes.
OCCLUSAL_LOCATION_FNS = [_segment_surface_locator(a, b) for a, b in zip(_OCC_A, _OCC_B)]


def _seg_dist(P, A, B):
    """Min distance from points P (M,2) to segment A-B."""
    AB = B - A
    t = onp.clip(((P - A) @ AB) / (onp.dot(AB, AB) + 1e-30), 0.0, 1.0)
    proj = A[None, :] + t[:, None] * AB[None, :]
    return onp.linalg.norm(P - proj, axis=1)


def dist_to_outline(P):
    """Min distance from points P (M,2) to the tooth boundary polyline."""
    verts = onp.asarray(tooth_outline())
    n = len(verts)
    d = onp.full(len(P), onp.inf)
    for i in range(n):
        d = onp.minimum(d, _seg_dist(P, verts[i], verts[(i + 1) % n]))
    return d


def root_boundary_distance(P):
    """Distance to tooth boundary segments below the cemento-enamel junction."""
    P = onp.asarray(P)
    verts = onp.asarray(tooth_outline())
    d = onp.full(len(P), onp.inf)
    for i in range(len(verts)):
        A, B = verts[i], verts[(i + 1) % len(verts)]
        if min(A[1], B[1]) <= Y_CEJ + 1e-8:
            d = onp.minimum(d, _seg_dist(P, A, B))
    return d


def _ccw(poly):
    """Return the polygon with counter-clockwise (positive-area) orientation.

    ``matplotlib.path.Path.contains_points`` interprets a positive ``radius`` as
    *growing* the path only for a CCW ring; for a CW ring the same value shrinks
    it and every point lying exactly on the boundary is reported as outside.
    ``tooth_outline``/``outer_root_outline`` are written clockwise, so the raw
    call excluded the whole extruded tooth surface -- including the occlusal
    fissure where the Figure-5 crack starts.
    """
    V = onp.asarray(poly, dtype=onp.float64)
    area = 0.5 * onp.sum(V[:, 0] * onp.roll(V[:, 1], -1)
                         - onp.roll(V[:, 0], -1) * V[:, 1])
    return V if area > 0. else V[::-1]


def tissue_masks(P):
    """Classify arbitrary 3D points into the six Figure-5 material domains."""
    P = onp.asarray(P).reshape(-1, 3)
    xy = P[:, :2]
    y = P[:, 1]
    zc = P[:, 2] - 0.5 * (Z_MIN + Z_MAX)
    inside_tooth = MplPath(_ccw(tooth_outline())).contains_points(xy, radius=1e-9)
    inside_outer_root = MplPath(_ccw(outer_root_outline())).contains_points(
        xy, radius=1e-9)

    # Pulp: a canal widening into the chamber, with a domed roof whose apex is
    # the pulp horn at PULP_TOP.  Half-axes in x (bucco-lingual) and z
    # (mesio-distal) grow from 0.5/0.6 mm at the apical end to 1.6/2.0 mm.
    s = onp.clip((y - PULP_BOTTOM) / max(PULP_TOP - 2.0 - PULP_BOTTOM, 1e-9), 0.0, 1.0)
    ax = 0.5 + 1.1 * s
    az = 0.6 + 1.4 * s
    roof = onp.where(y > PULP_TOP - 2.0,
                     onp.sqrt(onp.clip(1.0 - ((y - (PULP_TOP - 2.0)) / 2.0) ** 2, 0.0, 1.0)),
                     1.0)
    roof = onp.maximum(roof, 1e-6)
    pulp = (inside_tooth & (y > PULP_BOTTOM) & (y < PULP_TOP)
            & ((P[:, 0] / (ax * roof)) ** 2 + (zc / (az * roof)) ** 2 <= 1.0))

    crown = inside_tooth & (y >= Y_CEJ)
    # enamel thickness: T_ENAMEL occlusally, tapering to T_ENAMEL_CEJ at the CEJ
    t_y = T_ENAMEL_CEJ + (T_ENAMEL - T_ENAMEL_CEJ) * onp.clip(
        (y - Y_CEJ) / max(ENAMEL_TAPER_H, 1e-9), 0.0, 1.0)
    xy_surface = dist_to_outline(xy) <= t_y
    z_surface = onp.minimum(P[:, 2] - Z_MIN, Z_MAX - P[:, 2]) <= t_y
    enamel = crown & (xy_surface | z_surface) & ~pulp
    dentin = inside_tooth & ~enamel & ~pulp

    pdl = ~inside_tooth & inside_outer_root & (y <= Y_CREST + 1e-8)
    cortical = (~inside_tooth & ~pdl & (y <= Y_CREST)
                & ((y <= BONE_BOTTOM + CORTICAL_THICKNESS)
                   | (onp.abs(P[:, 0]) >= BONE_X_HALF - CORTICAL_THICKNESS)
                   | (onp.abs(zc) >= 0.5 * LZ - CORTICAL_THICKNESS)))
    cancellous = ~inside_tooth & ~pdl & ~cortical
    return {'enamel': enamel, 'dentin': dentin, 'pulp': pulp, 'pdl': pdl,
            'cancellous_bone': cancellous, 'cortical_bone': cortical,
            'tooth': inside_tooth}


# region subproblems
class Damage(Field):
    """omega'(d) H + Gc/c_w ( alpha'(d)/l - 2 l lap d ) = 0, with Gc and the
    PF-CZM constant a1 given per quadrature point: theta = (history, gc, a1)."""
    def flux(self):
        def fn(d_grad, history, gc, a1):
            return (2. * gc * l / c_w()) * d_grad
        return fn

    def source(self):
        def fn(d, x, history, gc, a1):
            dd = d[0]
            return np.array([gc / (c_w() * l) * _w_prime(dd)
                             + _g_prime(dd, a1) * history])
        return fn


class Displacement(Field):
    """Hybrid model: isotropically degraded stiffness for equilibrium; a tensile
    measure is used only to build the history H. theta = (d, mat) with
    mat = [mu, lambda, a1, H_crit] per quadrature point, so all constituents
    share one operator."""
    def flux(self):
        def stress(u_grad, d, mat):
            mu_q, lmbda_q, a1_q = mat[0], mat[1], mat[2]
            eps = 0.5 * (u_grad + u_grad.T)
            g = degradation(d[0], a1_q) + K_RES
            return g * (lmbda_q * np.trace(eps) * np.eye(3) + 2. * mu_q * eps)
        return stress

    def boundary_sources(self):
        """One-sided, frictionless penalty contact with a descending 4 mm ball.

        Boundary terms are added to the residual, so the function returns minus
        the physical ball-on-tooth traction. The contact is inactive for a
        positive gap and has no tangential penalty. theta_b = (delta,), the
        ball displacement at the face quadrature points.
        """
        zmid = 0.5 * (Z_MIN + Z_MAX)

        def rigid_ball_contact(u, x, delta):
            # Gap measured along the sphere NORMAL (true signed distance), not
            # vertically.  The old form took a vertical overlap but applied the
            # traction along n, giving a tangent K*(n (x) e_y) that is NOT
            # symmetric -- measured at 1.5e-3 relative asymmetry -- while the
            # step is solved with CG, which assumes symmetry.  That produced
            # inf/NaN once enough contact points sat away from the ball's
            # bottom, and got worse with mesh refinement.  Here the leading
            # tangent term is -K n (x) n, which is symmetric, and the vertical
            # measure's overestimate of penetration away from the bottom is
            # gone.  BALL_CENTER_Y0 is unchanged: both measures agree at zero
            # penetration, so the just-touching height BALL_CENTER_Y0 is unchanged.
            xd = x + u
            r = np.array([xd[0],
                          xd[1] - (BALL_Y_START - delta),
                          xd[2] - zmid])
            dist = np.sqrt(np.sum(r * r) + 1e-30)
            # Short-range spring: active for dist < R + d_s, linear in the
            # shortfall.  d_s = 0 reduces to hard penalty contact.
            penetration = np.maximum(R_EFF - dist, 0.0)
            outward_normal = r / dist        # ball centre -> surface point
            traction_on_tooth = CONTACT_PENALTY * penetration * outward_normal
            return -traction_on_tooth

        return [rigid_ball_contact for _ in OCCLUSAL_LOCATION_FNS]

    @staticmethod
    def psi_plus(u_grad, mat):
        """Miehe spectral STRAIN split (PFF_DRIVING=spectral).

        NOT suitable under an indenter: uniaxial compression sigma gives lateral
        Poisson strains +nu*sigma/E, i.e. psi+ = 2 mu (nu sigma/E)^2 > 0 with no
        tension anywhere.  With f_t = 40 MPa that exceeds H_crit at only 107 MPa
        of COMPRESSION (271 MPa at f_t = 101), so an earlier run damaged the whole
        contact flank (d = 1 on both cusp inclines) while the fissure stayed
        intact -- the same artefact the earlier contact "guards" were masking.
        """
        mu_q, lmbda_q = mat[0], mat[1]
        eps = 0.5 * (u_grad + u_grad.T)
        eigs = eigvalsh_sym3x3(eps)
        tr_plus = safe_plus(np.trace(eps))
        return lmbda_q / 2. * tr_plus ** 2 + mu_q * np.sum(safe_plus(eigs) ** 2)

    @staticmethod
    def psi_rankine(u_grad, mat):
        """Rankine driving force of PF-CZM (Wu 2017; Wu & Nguyen 2018):
        Y = <sigma_1>^2 / (2E) with sigma_1 the largest principal EFFECTIVE
        (undamaged) stress.  Identical to psi+ in uniaxial tension, so the
        threshold H_crit = f_t^2/2E is unchanged, and exactly zero under
        triaxial compression -- the contact zone cannot damage; the fissure
        floor, in wedge tension sigma_xx, can.  PFF_DRIVING=rankine (default).
        """
        mu_q, lmbda_q = mat[0], mat[1]
        eps = 0.5 * (u_grad + u_grad.T)
        sig = lmbda_q * np.trace(eps) * np.eye(3) + 2. * mu_q * eps
        s1 = eigvalsh_sym3x3(sig)[2]                      # ascending -> largest
        E_q = mu_q * (3. * lmbda_q + 2. * mu_q) / (lmbda_q + mu_q)
        return safe_plus(s1) ** 2 / (2. * E_q)

    def compute_history(self, sol_u, history_old, mat_quad, d_quad=None):
        u_grads = self.grad_at_quad(sol_u)
        fn = self.psi_rankine if DRIVING == 'rankine' else self.psi_plus
        vmap_psi = jax.vmap(jax.vmap(fn))
        psi = vmap_psi(u_grads, mat_quad)
        if H_CAP > 0.0:
            # mat_quad[..., 3] is the tissue's H_crit.  See the PFF_H_CAP note.
            psi = np.minimum(psi, H_CAP * mat_quad[..., 3])
        if d_quad is not None and D_FREEZE < 1.0:
            # Already broken: keep whatever H it had, stop feeding it more.
            psi = np.where(d_quad[..., 0] >= D_FREEZE, 0.0, psi)
        return np.maximum(psi, history_old)


# region BC
def bone_support(point):
    """Fix the external bone block without clamping tooth/PDL end faces.

    The z-end condition is limited to side strips outside the PDL envelope.
    Applying it to every point below the CEJ also clamps the extruded root faces
    and produces the false cervical crack seen in the original result.
    """
    zmid = 0.5 * (Z_MIN + Z_MAX)
    on_bone_height = point[1] <= Y_CREST + 1e-6
    on_z_bone_strip = ((np.abs(point[2] - zmid) >= 0.5 * LZ - 1e-6)
                       & (np.abs(point[0]) >= ROOT_HALF_CREST + 2.0 * PDL_THICKNESS))
    return on_bone_height & (
        (point[1] <= BONE_BOTTOM + 1e-6)
        | (np.abs(point[0]) >= BONE_X_HALF - 1e-6)
        | on_z_bone_strip)


# Bone base and four lateral faces fixed in all three directions.
dirichlet_u = [DirichletBC(bone_support, component, 0.) for component in range(3)]


# region mesh
def tooth_mesh_3d(h_fine, h_coarse, Lz):
    """Mesh conforming tooth, PDL, and bone volumes with TET4.

    Boolean cuts make the root/PDL and PDL/bone interfaces coincide exactly.
    The previous single union volume allowed tetrahedra to straddle those
    interfaces and created a spurious high-energy line at the alveolar crest.
    Returns (points[N,3], cells[M,4])."""

    def add_prism(poly, height):
        point_tags = [gmsh.model.occ.addPoint(x, y, 0.0) for x, y in poly]
        line_tags = [gmsh.model.occ.addLine(point_tags[i],
                                           point_tags[(i + 1) % len(point_tags)])
                     for i in range(len(point_tags))]
        wire = gmsh.model.occ.addWire(line_tags)
        surface = gmsh.model.occ.addPlaneSurface([wire])
        out = gmsh.model.occ.extrude([(2, surface)], 0.0, 0.0, height)
        return next(tag for dim, tag in out if dim == 3)

    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('premolar')
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeMin', min(h_fine, h_coarse))
        gmsh.option.setNumber('Mesh.MeshSizeMax', max(h_fine, h_coarse))

        tooth_volume = add_prism(tooth_outline(), Lz)
        root_outer_volume = add_prism(outer_root_outline(), Lz)
        bone_volume = gmsh.model.occ.addBox(-BONE_X_HALF, BONE_BOTTOM, 0.0,
                                            2.0 * BONE_X_HALF,
                                            Y_CREST - BONE_BOTTOM, Lz)

        bone_cut, _ = gmsh.model.occ.cut([(3, bone_volume)],
                                         [(3, root_outer_volume)],
                                         removeObject=True, removeTool=False)
        pdl_cut, _ = gmsh.model.occ.cut([(3, root_outer_volume)],
                                        [(3, tooth_volume)],
                                        removeObject=True, removeTool=False)
        if not bone_cut or not pdl_cut:
            raise RuntimeError('Gmsh failed to construct conforming bone/PDL volumes')

        # Imprint matching interface surfaces and shared nodes on every volume.
        # Capture fragment's output map (rather than discarding it) so the tooth/
        # PDL/bone volumes can be tagged as gmsh physical groups by their final
        # (possibly renumbered) entity tags.
        frag_input = [(3, tooth_volume)] + pdl_cut + bone_cut
        frag_out, frag_map = gmsh.model.occ.fragment(frag_input, [])
        gmsh.model.occ.synchronize()

        tooth_tags = [t for _, t in frag_map[0]]
        pdl_tags = [t for grp in frag_map[1:1 + len(pdl_cut)] for _, t in grp]
        bone_tags = [t for grp in frag_map[1 + len(pdl_cut):] for _, t in grp]
        vol_tags = {'tooth': tooth_tags, 'pdl': pdl_tags, 'bone': bone_tags}
        for name, tags in vol_tags.items():
            gmsh.model.addPhysicalGroup(3, tags, name=name)

        # Boundary surfaces owned by exactly one volume are that volume's free
        # (exterior) surface; surfaces owned by two are the interface between
        # them (tooth/PDL, PDL/bone) -- read directly off gmsh's own topology
        # rather than re-deriving BC/contact regions from geometry thresholds.
        owners = {}
        for name, tags in vol_tags.items():
            for vt in tags:
                for _, s in gmsh.model.getBoundary([(3, vt)], oriented=False,
                                                    combined=False):
                    owners.setdefault(s, set()).add(name)
        surface_groups = {}
        for s, names in owners.items():
            key = (f'{next(iter(names))}_free' if len(names) == 1
                   else '_'.join(sorted(names)) + '_interface')
            surface_groups.setdefault(key, []).append(s)
        for name, surfs in surface_groups.items():
            gmsh.model.addPhysicalGroup(2, surfs, name=name)

        # h_fine inside the fracture-admissible crown, h_coarse elsewhere.  With
        # h_coarse == h_fine (the default) the box still covers the whole model
        # and the mesh stays uniform.
        f = gmsh.model.mesh.field.add('Box')
        gmsh.model.mesh.field.setNumber(f, 'VIn', h_fine)
        gmsh.model.mesh.field.setNumber(f, 'VOut', h_coarse)
        gmsh.model.mesh.field.setNumber(f, 'XMin', -BONE_X_HALF - 1.0)
        gmsh.model.mesh.field.setNumber(f, 'XMax', BONE_X_HALF + 1.0)
        gmsh.model.mesh.field.setNumber(f, 'YMin', (BONE_BOTTOM - 1.0 if h_coarse <= h_fine
                                                   else FRACTURE_Y_MIN - 2.0 * l))
        gmsh.model.mesh.field.setNumber(f, 'YMax', H_TOOTH + 1.0)
        gmsh.model.mesh.field.setNumber(f, 'ZMin', -1.0)
        gmsh.model.mesh.field.setNumber(f, 'ZMax', Lz + 1.0)
        gmsh.model.mesh.field.setNumber(f, 'Thickness', 4.0 * l)
        gmsh.model.mesh.field.setAsBackgroundMesh(f)

        gmsh.model.mesh.generate(3)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        order = onp.argsort(node_tags)
        tag2idx = onp.zeros(int(node_tags.max()) + 1, dtype=onp.int64)
        tag2idx[node_tags[order]] = onp.arange(len(node_tags))
        points = coords[order]
        points[:, 2] += Z_MIN
        elem_tags, enodes = gmsh.model.mesh.getElementsByType(4)   # 4 = TET4
        cells = tag2idx[enodes.reshape(-1, 4)].astype(onp.int32)

        # Cell field: which gmsh physical volume (tooth/PDL/bone) each tet came
        # from, read straight off the physical groups tagged above.
        order_e = onp.argsort(elem_tags)
        sorted_etags = elem_tags[order_e]

        def _elem_idx(tags3d):
            etags_list = [gmsh.model.mesh.getElementsByType(4, vt)[0] for vt in tags3d]
            if not etags_list:
                return onp.array([], dtype=onp.int64)
            all_et = onp.concatenate(etags_list)
            return order_e[onp.searchsorted(sorted_etags, all_et)]

        cell_volume_id = onp.zeros(len(elem_tags), dtype=onp.float64)
        for i, name in enumerate(('tooth', 'pdl', 'bone'), start=1):
            cell_volume_id[_elem_idx(vol_tags[name])] = float(i)
        gmsh_cell_fields = [('gmsh_volume_id', cell_volume_id)]

        # Point fields: node membership in each gmsh physical surface (free
        # exterior of tooth/PDL/bone, and the tooth-PDL / PDL-bone interfaces).
        gmsh_point_fields = []
        for name, surfs in surface_groups.items():
            node_ids = set()
            for s in surfs:
                ntags, _, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
                node_ids.update(int(t) for t in ntags)
            mask = onp.zeros(len(node_tags), dtype=onp.float64)
            if node_ids:
                mask[tag2idx[onp.array(sorted(node_ids), dtype=onp.int64)]] = 1.0
            gmsh_point_fields.append((f'gmsh_{name}', mask))
    finally:
        gmsh.finalize()
    return (onp.asarray(points), onp.asarray(cells),
            gmsh_cell_fields, gmsh_point_fields)


def contact_shadow(P, delta):
    """Nodes the rigid ball reaches (undeformed) at this indentation, + margin."""
    P = onp.asarray(P).reshape(-1, 3)
    rx = P[:, 0]
    rz = P[:, 2] - 0.5 * (Z_MIN + Z_MAX)
    rho2 = rx * rx + rz * rz
    radial_y = onp.sqrt(onp.maximum(R_EFF ** 2 - rho2, 0.0))
    gap = (BALL_Y_START - delta - radial_y) - P[:, 1]
    surface = (rho2 < R_EFF ** 2) & (gap <= CONTACT_SHADOW_MARGIN)
    if CONTACT_SHADOW_DEPTH <= 0.0:
        return surface
    # Everything within CONTACT_SHADOW_DEPTH of the occlusal surface whose
    # in-plane radius falls in the contact annulus.  RHO_MIN keeps the fissure
    # (rho = 0) admissible; the paper's crack starts there.
    rho = onp.sqrt(rho2)
    depth = dist_to_outline(P[:, :2])
    volume = ((rho >= CONTACT_SHADOW_RHO_MIN) & (rho <= CONTACT_SHADOW_RHO_MAX)
              & (depth <= CONTACT_SHADOW_DEPTH) & (P[:, 1] > 0.5 * (Y_CEJ + H_TOOTH)))
    return surface | volume


def cell_tissue_masks(points, cells):
    """Diagnostic cell-centre tissue masks."""
    ctr = onp.asarray(points)[onp.asarray(cells)].mean(axis=1)
    return tissue_masks(ctr)


def boundary_faces(cells):
    """Triangular faces of the tet mesh that belong to exactly one cell."""
    floc = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    faces = onp.concatenate([cells[:, f] for f in floc], axis=0)      # (4M,3)
    key = onp.sort(faces, axis=1)
    uniq, idx, cnt = onp.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[cnt == 1]]                                       # oriented boundary tris


def tet_volumes(points, cells):
    """Positive TET4 volumes [mm^3], used for a mesh-independent damage metric."""
    X = onp.asarray(points)[onp.asarray(cells)]
    return onp.abs(onp.einsum('ij,ij->i', X[:, 1] - X[:, 0],
                              onp.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]))) / 6.0


# region field builder
def build_fields(points, cells):
    mesh = Mesh(onp.asarray(points), onp.asarray(cells), ELE)
    field_d = Damage(mesh, ncomp=1)
    field_u = Displacement(mesh, ncomp=3, dirichlet=dirichlet_u,
                           boundaries=OCCLUSAL_LOCATION_FNS)
    return mesh, field_d, field_u


def dirichlet_data(field):
    return field.dirichlet_dofs, field.dirichlet_values


def save_sol(mesh, sol, path, cell_infos=None, point_infos=None):
    """Write the damage field (point field 'sol') and further fields to a VTU file."""
    point_data = {'sol': sol}
    point_data.update(dict(point_infos or []))
    axion.write_vtu(path, mesh, point_data=point_data, cell_data=dict(cell_infos or []))


def material_quads(field_u):
    """Build six-tissue elastic fields and fracture energy at quadrature points."""
    xq = onp.asarray(field_u.quad_points)
    shape = xq.shape[:2]
    masks_flat = tissue_masks(xq.reshape(-1, 3))
    masks = {k: v.reshape(shape) for k, v in masks_flat.items()}

    mu = onp.full(shape, MU_CB)
    lm = onp.full(shape, LMBDA_CB)
    gc = onp.full(shape, GC_NONFRACTURE)
    # Non-fracturing tissues keep the enamel softening constant: their damage is
    # already suppressed by GC_NONFRACTURE and by the d=0 nodal constraints, and
    # deriving a1 from GC_NONFRACTURE would give a huge (unstable) driving force.
    a1 = onp.full(shape, A1_ENAMEL)
    hc = onp.full(shape, H_CRIT_ENAMEL)          # per-quad H_crit, for the H cap
    for name, mu_i, lm_i, gc_i, a1_i, hc_i in (
            ('cortical_bone', MU_CT, LMBDA_CT, GC_NONFRACTURE, A1_ENAMEL, H_CRIT_ENAMEL),
            ('pdl', MU_L, LMBDA_L, GC_NONFRACTURE, A1_ENAMEL, H_CRIT_ENAMEL),
            ('pulp', MU_P, LMBDA_P, GC_NONFRACTURE, A1_ENAMEL, H_CRIT_ENAMEL),
            ('dentin', MU_D, LMBDA_D, GC_DENTIN, A1_DENTIN, H_CRIT_DENTIN),
            ('enamel', MU_E, LMBDA_E, GC_ENAMEL, A1_ENAMEL, H_CRIT_ENAMEL)):
        mu = onp.where(masks[name], mu_i, mu)
        lm = onp.where(masks[name], lm_i, lm)
        gc = onp.where(masks[name], gc_i, gc)
        a1 = onp.where(masks[name], a1_i, a1)
        hc = onp.where(masks[name], hc_i, hc)
    mat = onp.stack([mu, lm, a1, hc], axis=-1)
    return np.asarray(mat), np.asarray(gc), np.asarray(a1)


# region unilateral rigid-ball contact
def contact_surface_info(field_u):
    """Number and reference area of the candidate occlusal contact faces."""
    nfaces = sum(faces.num_faces for faces in field_u.boundaries)
    if nfaces == 0:
        raise ValueError('no occlusal contact faces found; check the crown geometry')
    area = sum(float(onp.asarray(faces.dS).sum()) for faces in field_u.boundaries)
    return nfaces, area


def contact_surface_vars(field_u, delta):
    """Ball displacement at every quadrature point of the contact surface."""
    return [[np.full(faces.quad_points.shape[:2], delta)]
            for faces in field_u.boundaries]


# region seeding
def seed_crack_history(field_u, mat_quad):
    """Irreversible history H at t=0.

    Fig. 5 is the INTACT tooth, so this is zero unless PFF_SEED_CRACK=1, which
    plants a vertical crack on the plane x=0 running down from the fissure --
    the phase-field equivalent of the paper's "initially broken bonds".
    """
    H = onp.zeros((field_u.num_cells, field_u.num_quads))
    if not SEED_CRACK:
        return np.asarray(H)

    xq = onp.asarray(field_u.quad_points)
    half = SEED_HALF_THICK if SEED_HALF_THICK > 0.0 else 0.5 * H_UNIFORM
    y_top = FISSURE_XY[1]
    zc = xq[..., 2] - 0.5 * (Z_MIN + Z_MAX)
    band = ((onp.abs(xq[..., 0]) <= half)
            & (xq[..., 1] <= y_top)
            & (xq[..., 1] >= y_top - SEED_DEPTH)
            & (onp.abs(zc) <= SEED_HALF_Z))
    # H must exceed the damage threshold by enough to drive d -> 1.
    H[band] = SEED_H_FACTOR * max(H_CRIT_ENAMEL, H_CRIT_DENTIN)
    print(f"  seed      : initial crack on x=0, {SEED_DEPTH:.2f} mm below the "
          f"fissure (y={y_top:.3f}), half-thickness {half:.3f} mm, "
          f"{int(band.sum())} quadrature points", flush=True)
    return np.asarray(H)


# region solver core
def get_assembled_ops(field):
    cache = getattr(field, '_asm_ops', None)
    if cache is not None:
        return cache
    matvec, inv_jacobi = axion.make_operator(field)
    cache = (matvec, inv_jacobi, field.ncomp, field.num_nodes)
    field._asm_ops = cache
    return cache


def assemble_device(field, u_flat, internal_vars, f_ext, surface_ivs=None):
    """Residual (minus the external load) and stored tangent entries."""
    if surface_ivs is None:
        surface_ivs = getattr(field, 'default_surface_ivs', None)
    res, V = axion.assemble(field, u_flat, internal_vars, surface_ivs)
    return res - f_ext, V


def get_newton_solver(field, num_newton=None, nl_rtol=1e-8, nl_atol=1e-12,
                      lin_tol=None, lin_maxiter=None):
    cache = getattr(field, '_newton_solver', None)
    if cache is not None:
        return cache
    num_newton = NEWTON_MAX if num_newton is None else num_newton
    lin_maxiter = CG_MAXITER if lin_maxiter is None else lin_maxiter
    lin_tol = CG_TOL if lin_tol is None else lin_tol
    matvec, inv_jacobi, vec, N = get_assembled_ops(field)

    @jax.jit
    def newton(u0, R0, V0, ivs, fixed_idx, fixed_vals, f_ext, surface_ivs):
        ivs = ivs if isinstance(ivs, (tuple, list)) else (ivs,)
        # Relative test.  A fixed absolute 1e-8 is unreachable here: one load
        # step of the ball leaves a residual of order 1e2-1e3 N, so the loop
        # always burnt all num_newton iterations (and all lin_maxiter CG
        # iterations inside each) even after the step had fully converged.
        tol = np.maximum(nl_rtol * np.linalg.norm(R0), nl_atol)
        def cond(state):
            u, R, V, it = state
            return (it < num_newton) & (np.linalg.norm(R) > tol)
        def body(state):
            u, R, V, it = state
            inv = inv_jacobi(V, fixed_idx)
            du, _ = jax.scipy.sparse.linalg.cg(lambda d: matvec(V, d, fixed_idx), -R,
                                               tol=lin_tol, atol=0.0, maxiter=lin_maxiter,
                                               M=lambda r: r * inv)
            u_new = u + du
            res_new, V_new = assemble_device(field, u_new, list(ivs), f_ext,
                                             surface_ivs)
            R_new = res_new.at[fixed_idx].set(0.0)
            if VERBOSE:
                # linear-solve quality: |A du + R| / |R| should be ~lin_tol; a value
                # near 1 means CG hit lin_maxiter without converging.
                lin_res = np.linalg.norm(matvec(V, du, fixed_idx) + R) / (np.linalg.norm(R) + 1e-30)
                jax.debug.print("      newton it={it}  |R| {r0:.3e} -> {r1:.3e}  (tol {tol:.1e}, "
                                "CG rel.res {lr:.1e})", it=it + 1, r0=np.linalg.norm(R),
                                r1=np.linalg.norm(R_new), tol=tol, lr=lin_res)
            return (u_new, R_new, V_new, it + 1)
        return jax.lax.while_loop(cond, body, (u0, R0, V0, 0))[0]

    field._newton_solver = newton
    return newton


def get_damage_solver(field, d_fixed_idx, max_it=None, tol=1e-7, omega=1.0):
    cache = getattr(field, '_damage_solver', None)
    if cache is not None:
        return cache
    max_it = PG_MAXIT if max_it is None else max_it

    @jax.jit
    def pg_solve(y0, Dm, history, gc_quad, a1_quad):
        def residual_fn(y):
            return field.residual(y, [history, gc_quad, a1_quad])
        def cond(state):
            y, y_prev, it = state
            change = np.linalg.norm(y - y_prev) / (np.linalg.norm(y) + 1e-30)
            return (it < max_it) & (change > tol)
        def body(state):
            y, _, it = state
            R = residual_fn(y)
            y_new = np.clip(y - omega * R / Dm, 0., 1.)
            y_new = y_new.at[d_fixed_idx].set(0.0)
            return (y_new, y, it + 1)
        y0c = np.clip(y0, 0., 1.).at[d_fixed_idx].set(0.0)
        y, y_prev, it = jax.lax.while_loop(cond, body, (y0c, y0c + 1.0, 0))
        change = np.linalg.norm(y - y_prev) / (np.linalg.norm(y) + 1e-30)
        return y, it, change

    field._damage_solver = pg_solve
    return pg_solve


def compute_damage_Dm(field, sol, history, gc_quad, a1_quad):
    return axion.abs_diagonal(field, sol, [history, gc_quad, a1_quad])


# region staggered stepper
def get_staggered_stepper(field_u, field_d, d_fixed_idx, max_stagger=10, ms_tol=1e-4):
    cache = getattr(field_u, '_staggered_stepper', None)
    if cache is not None:
        return cache
    newton_u = get_newton_solver(field_u)
    fe_u = field_u; Nu = fe_u.num_nodes
    fe_d = field_d; Nd = fe_d.num_nodes

    zero_fd = np.zeros(Nd * fe_d.ncomp)
    d_fixed_vals = np.zeros(d_fixed_idx.shape)

    if USE_PG:
        pg_solve = get_damage_solver(field_d, d_fixed_idx)
    else:
        newton_d = get_newton_solver(field_d)

    @jax.jit
    def stepper(sol_u, sol_d, history_old, fixed_idx, fixed_vals,
                f_ext_u, mat_quad, gc_quad, a1_quad, ball_delta):
        surface_ivs_u = contact_surface_vars(field_u, ball_delta)
        surface_ivs_d = None                    # the damage field has no boundary terms

        def cond(state):
            su, sd, hist, err, s = state
            return (s < max_stagger) & (err > ms_tol)
        def body(state):
            su, sd, hist, err, s, _, _ = state
            d_quad = fe_u.to_quad(sd)
            u0 = su.reshape(-1).at[fixed_idx].set(fixed_vals)
            res0, V0 = assemble_device(field_u, u0, [d_quad, mat_quad], f_ext_u,
                                       surface_ivs_u)
            R0 = res0.at[fixed_idx].set(0.0)
            su_new = newton_u(u0, R0, V0, (d_quad, mat_quad),
                              fixed_idx, fixed_vals, f_ext_u,
                              surface_ivs_u).reshape(Nu, 3)
            hist_new = field_u.compute_history(su_new, history_old, mat_quad,
                                                 d_quad)
            if USE_PG:
                Dm = compute_damage_Dm(field_d, sd, hist_new, gc_quad, a1_quad)
                sd_new, pg_it, pg_chg = pg_solve(sd.reshape(-1), Dm, hist_new,
                                                 gc_quad, a1_quad)
                sd_new = sd_new.reshape(Nd, 1)
            else:
                sd0 = sd.reshape(-1).at[d_fixed_idx].set(0.0)
                res0d, V0d = assemble_device(field_d, sd0,
                                             [hist_new, gc_quad, a1_quad], zero_fd)
                res0d = res0d.at[d_fixed_idx].set(0.0)
                sd_new = newton_d(sd0, res0d, V0d, (hist_new, gc_quad, a1_quad),
                                  d_fixed_idx, d_fixed_vals, zero_fd,
                                  surface_ivs_d).reshape(Nd, 1)
                sd_new = np.clip(sd_new, 0., 1.).at[d_fixed_idx].set(0.0)
                pg_it, pg_chg = np.array(-1), np.array(-1.0)
            eu = np.linalg.norm(su_new - su) / (np.linalg.norm(su_new) + 1e-30)
            ed = np.linalg.norm(sd_new - sd) / (np.linalg.norm(sd_new) + 1e-30)
            if VERBOSE:
                jax.debug.print("    stagger {s}: du/u {eu:.2e}  dd/d {ed:.2e}  max d {md:.4f}"
                                "  pg {pi}", s=s + 1, eu=eu, ed=ed, md=np.max(sd_new), pi=pg_it)
            # Convergence is judged on the DAMAGE change only.  u is always the
            # converged solution for the d it was solved with, so once d stops
            # changing there is nothing left to iterate.  The old max(eu, ed)
            # always forced a second stagger (eu ~ 1 on the first pass, since u
            # moves from the previous load level), i.e. a second full Newton+CG
            # solve of the same problem -- doubling every elastic step.
            return (su_new, sd_new, hist_new, ed, s + 1, pg_it, pg_chg)
        def cond2(state):
            return cond(state[:5])
        init = (sol_u, sol_d, history_old, np.array(1e30), np.array(0),
                np.array(-1), np.array(-1.0))
        su, sd, hist, err, s, pg_it, pg_chg = jax.lax.while_loop(cond2, body, init)
        return su, sd, hist, s, err, pg_it, pg_chg

    field_u._staggered_stepper = stepper
    return stepper


def build_dirichlet_static(field_u, mat_quad):
    """Return the fixed bone-support DOFs and their zero values."""
    field_u.default_surface_ivs = contact_surface_vars(field_u, 0.0)
    support_idx, support_vals = dirichlet_data(field_u)
    support_idx_host = onp.asarray(support_idx, dtype=onp.int32)
    support_y_dofs = support_idx_host[support_idx_host % 3 == 1]
    return (np.array(support_idx_host, dtype=np.int32), np.array(support_vals),
            np.array(support_y_dofs, dtype=np.int32))


def cell_output_fields(cell_masks, gc_quad):
    """ParaView cell fields shared by every saved state."""
    names = ('enamel', 'dentin', 'pulp', 'pdl', 'cancellous_bone', 'cortical_bone')
    tissue_id = onp.zeros(len(next(iter(cell_masks.values()))), dtype=onp.float64)
    for i, name in enumerate(names, start=1):
        tissue_id[cell_masks[name]] = float(i)
    fields = [('tissue_id', tissue_id), ('Gc_mean', onp.asarray(gc_quad).mean(axis=1))]
    fields.extend((name, onp.asarray(cell_masks[name], dtype=onp.float64)) for name in names)
    return fields


# region driver
def main():
    t_wall0 = time.time()
    SAVE_EVERY = int(os.environ.get('PFF_SAVE_EVERY', 1 if COARSE else 5))
    MAX_STAGGER = int(os.environ.get('PFF_MAX_STAGGER', 20))
    levels = onp.arange(DELTA_STEP, DELTA_MAX + 0.5 * DELTA_STEP, DELTA_STEP)
    # Optional local refinement window: replace the coarse levels inside
    # [PFF_REFINE_START, PFF_REFINE_END] with a finer PFF_REFINE_STEP spacing,
    # leaving the schedule elsewhere untouched. Every level inside the window
    # is also force-saved (see REFINE_WINDOW use below), regardless of
    # PFF_SAVE_EVERY, so the refined region isn't thinned out by the normal
    # save cadence.
    REFINE_START = os.environ.get('PFF_REFINE_START')
    REFINE_END = os.environ.get('PFF_REFINE_END')
    REFINE_STEP = os.environ.get('PFF_REFINE_STEP')
    REFINE_WINDOW = None
    if REFINE_START is not None and REFINE_END is not None and REFINE_STEP is not None:
        r0, r1, rs = float(REFINE_START), float(REFINE_END), float(REFINE_STEP)
        REFINE_WINDOW = (r0, r1)
        levels = levels[(levels < r0 - 1e-9) | (levels > r1 + 1e-9)]
        refine_levels = onp.arange(r0, r1 + 0.5 * rs, rs)
        levels = onp.concatenate([levels, refine_levels])
        print(f"  refine    : [{r0:.3f}, {r1:.3f}] mm at step={rs:.4f} mm "
              f"({len(refine_levels)} levels, every one saved)")
    targets = FIG5_TARGETS[FIG5_TARGETS <= DELTA_MAX + 1e-12]
    levels = onp.unique(onp.round(onp.concatenate([levels, targets]), 10))

    pts, cells, gmsh_cell_fields, gmsh_point_fields = tooth_mesh_3d(H_FINE, H_COARSE, LZ)
    cell_masks = cell_tissue_masks(pts, cells)
    mesh, field_d, field_u = build_fields(pts, cells)
    mat_quad, gc_quad, a1_quad = material_quads(field_u)
    contact_faces, contact_area = contact_surface_info(field_u)

    P = onp.asarray(pts)
    disp_node = int(onp.argmin((P[:, 0] - CUSP_TIP[0]) ** 2 + (P[:, 1] - CUSP_TIP[1]) ** 2
                               + (P[:, 2] - 0.5 * (Z_MIN + Z_MAX)) ** 2))

    pts_d = onp.asarray(mesh.points)
    node_masks = tissue_masks(pts_d)
    admissible = ((node_masks['enamel'] | node_masks['dentin'])
                  & (pts_d[:, 1] >= FRACTURE_Y_MIN - 1e-10))
    if CONTACT_NO_DAMAGE:
        shadow = contact_shadow(pts_d, DELTA_MAX)
        admissible = admissible & ~shadow
    if ELASTIC_ONLY:
        admissible = onp.zeros_like(admissible)
    d_fixed_idx = np.array(onp.where(~admissible)[0], dtype=np.int32)
    fracture_admissible = onp.asarray(admissible, dtype=onp.float64)

    sol_u = onp.zeros((len(pts), 3))
    sol_d = onp.zeros((len(pts), 1))
    history_old = seed_crack_history(field_u, mat_quad)

    u_dofs = 3 * len(pts)
    nnz = 144 * len(cells)                       # TET4 u-tangent: (12x12) per cell
    gb = nnz * (8 + 8 + 8) / 1e9                 # stored values (f64) + row/column indices
    print(f"Axion | intact premolar | Model = {MODEL} | mesh = {ELE} "
          f"({'COARSE test' if COARSE else 'production'})")
    print(f"  mesh      : {len(pts)} nodes, {len(cells)} tets, z=[{Z_MIN:.2f}, {Z_MAX:.2f}] mm, "
          f"{'UNIFORM' if H_COARSE <= H_FINE else 'GRADED'} h={H_UNIFORM:.4f} mm "
          f"(l/h={l / H_UNIFORM:.1f}, band ~2l spans {2 * l / H_UNIFORM:.0f} elems"
          + (f"; bone h={H_COARSE:.4f} mm" if H_COARSE > H_FINE else "") + ")")
    print(f"  size      : u-DOFs ~{u_dofs/1e6:.2f}M, assembled u-tangent ~{nnz/1e6:.0f}M "
          f"nnz (~{gb:.1f} GB)")
    if gb > 6.0:
        print(f"  WARNING   : ~{gb:.1f} GB for the assembled u-operator alone, and the "
              f"tangent evaluation peaks at several times that. Raise PFF_L, "
              f"lower PFF_H_OVER_L, or coarsen the bone with PFF_H_BONE.", flush=True)
    print("  tissues   : " + ", ".join(
        f"{name}={int(cell_masks[name].sum())}" for name in
        ('enamel', 'dentin', 'pulp', 'pdl', 'cancellous_bone', 'cortical_bone')))
    print(f"  fracture  : enamel/dentin crown, y>={FRACTURE_Y_MIN:.2f} mm; "
          f"Gc={GC_ENAMEL*1e3:.0f}/{GC_DENTIN*1e3:.0f} J/m2; "
          f"no initial crack or weak path")
    print(f"  length    : l = {l:.3f} mm  (band ~2l = {2*l:.2f} mm); "
          f"l_ch = {L_CH_ENAMEL:.3f}/{L_CH_DENTIN:.3f} mm (enamel/dentin), "
          f"a1 = {A1_ENAMEL:.2f}/{A1_DENTIN:.2f}")
    print(f"  strength  : f_t = {F_T_ENAMEL:.1f}/{F_T_DENTIN:.1f} MPa (enamel/dentin; the PD "
          f"paper's E*s0 would be {E_ENAMEL*S0_ENAMEL:.1f}/{E_DENTIN*S0_DENTIN:.1f}); "
          f"damage threshold H_crit = f_t^2/2E = "
          f"{H_CRIT_ENAMEL:.4f}/{H_CRIT_DENTIN:.4f} MPa")
    if MODEL == 'CZM':
        # Linear-softening PF-CZM: the point response H(d) is monotone (stable,
        # gradual damage) only for l < 2 l_ch / pi.  Above that the H needed to
        # hold damage FALLS with d -- snap-back: once threshold is crossed the
        # point jumps to d=1 and the crown fails in one step (earlier runs at
        # l=0.5, l_ch=0.675).  H(d->1)/H_crit = (2 l_ch/(pi l))^2.
        l_snap = 2.0 * min(L_CH_ENAMEL, L_CH_DENTIN) / onp.pi
        hd1 = (2.0 * min(L_CH_ENAMEL, L_CH_DENTIN) / (onp.pi * l)) ** 2
        if l >= l_snap:
            print(f"  WARNING   : PF-CZM SNAP-BACK regime: l={l:.3f} mm >= 2 l_ch/pi = "
                  f"{l_snap:.3f} mm (H(d=1)/H_crit = {hd1:.2f} < 1). Damage will jump "
                  f"to 1 once threshold is crossed. Lower PFF_L below {l_snap:.3f}.")
        else:
            print(f"  regime    : PF-CZM stable softening, l={l:.3f} < 2 l_ch/pi = "
                  f"{l_snap:.3f} mm; d -> 1 needs H = {hd1:.2f} x H_crit at a point")
    print(f"  indenter  : unilateral frictionless 4 mm ball; {contact_faces} candidate faces, "
          f"surface area ~{contact_area:.2f} mm2, penalty={CONTACT_PENALTY:.2e} N/mm3, "
          f"short-range d_s={CONTACT_DS:.2f} mm (paper Eq.14)"
          + ("" if CONTACT_DS > 0 else "  [HARD contact: Hertzian ring-crack risk]"))
    clearance0 = BALL_CENTER_Y0 - BALL_RADIUS - FISSURE_XY[1]
    print(f"  geometry  : crown {H_TOOTH - Y_CEJ:.1f} mm tall, {GEOM['W']:.1f} (x) by "
          f"{LZ:.1f} (z) mm wide; root {Y_CEJ - Y_APEX:.1f} mm; alveolar crest "
          f"{Y_CEJ - Y_CREST:.1f} mm below the CEJ; pulp horn {H_TOOTH - PULP_TOP:.1f} mm "
          f"below the cusps; enamel {T_ENAMEL:.2f} -> {T_ENAMEL_CEJ:.2f} mm")
    print(f"  fissure   : V between cusp tips at x=+-{CUSP_HALF_SPAN:.2f}, "
          f"{FISSURE_DEPTH:.2f} mm deep, floor radius {FISSURE_RADIUS:.2f} mm -> incline "
          f"{CUSP_INCLINE_DEG:.0f} deg = cusp angle {CUSP_ANGLE_DEG:.0f} deg "
          f"(Yang 2024: 50 deg -> fissure crack, 70 deg -> contact crack)")
    print(f"  ball      : just touches the flanks at x=+-{CONTACT_X:.3f}, y={CONTACT_Y:.3f} "
          f"(centre y0={BALL_CENTER_Y0:.4f}); fissure floor y={FISSURE_XY[1]:.3f}, "
          f"ball-floor clearance {clearance0:.3f} mm at delta=0", flush=True)
    with open(os.path.join(output_dir, 'geometry.json'), 'w') as fh:
        json.dump(dict(
            h_tooth=H_TOOTH, y_cej=Y_CEJ, y_crest=Y_CREST, y_apex=Y_APEX,
            z_min=Z_MIN, z_max=Z_MAX, fissure_xy=list(FISSURE_XY),
            cusp_tip=list(CUSP_TIP), cusp_half_span=CUSP_HALF_SPAN,
            fissure_depth=FISSURE_DEPTH, cusp_incline_deg=CUSP_INCLINE_DEG,
            cusp_angle_deg=CUSP_ANGLE_DEG, ball_radius=BALL_RADIUS,
            ball_center_y0=BALL_CENTER_Y0, contact_x=CONTACT_X, contact_y=CONTACT_Y,
            fissure_column_half_x=FISSURE_COL_HALF_X,
            fissure_column_y_min=FISSURE_COL_Y_MIN,
            contact_rho=[CONTACT_SHADOW_RHO_MIN, CONTACT_SHADOW_RHO_MAX],
            contact_y_min=FISSURE_XY[1] - 0.2, pulp_top=PULP_TOP,
            gc_enamel=GC_ENAMEL, gc_dentin=GC_DENTIN, l=l), fh, indent=1)
    if CONTACT_NO_DAMAGE:
        print(f"  guard     : damage forbidden in the ball contact shadow at "
              f"delta={DELTA_MAX:.3f} mm (margin {CONTACT_SHADOW_MARGIN:.2f} mm), "
              f"{int(contact_shadow(pts_d, DELTA_MAX).sum())} nodes")
        # The guard must never cover the fissure floor -- that is the Fig. 5
        # initiation site.  Check the actual point, not an assumption.
        fissure = onp.array([[FISSURE_XY[0], FISSURE_XY[1],
                              0.5 * (Z_MIN + Z_MAX)]])
        f_gap = float((BALL_Y_START - DELTA_MAX - R_EFF) - FISSURE_XY[1])
        if bool(contact_shadow(fissure, DELTA_MAX)[0]):
            print(f"  ERROR     : the contact guard covers the fissure at "
                  f"{FISSURE_XY} -- its clearance is {f_gap:.4f} mm but the "
                  f"margin is {CONTACT_SHADOW_MARGIN:.4f} mm. The crack cannot "
                  f"nucleate where Fig. 5 starts it. Lower "
                  f"PFF_CONTACT_SHADOW_MARGIN below {f_gap:.4f}.", flush=True)
            raise SystemExit(2)
        print(f"            : fissure clearance {f_gap:.4f} mm > margin "
              f"{CONTACT_SHADOW_MARGIN:.4f} mm, so the fissure stays free")
        if CONTACT_SHADOW_DEPTH > 0.0:
            print(f"            : volumetric shell {CONTACT_SHADOW_DEPTH:.3f} mm deep "
                  f"(= {CONTACT_SHADOW_DEPTH / l:.1f} l) over rho in "
                  f"[{CONTACT_SHADOW_RHO_MIN:.2f}, {CONTACT_SHADOW_RHO_MAX:.2f}] mm")
    print(f"  driving   : {DRIVING} -- "
          + ("<sigma_1>^2/2E on the effective stress; zero under the ball"
             if DRIVING == 'rankine' else
             "Miehe spectral strain split; WARNING damages the contact zone by Poisson expansion"))
    print(f"  damage    : {'projected-Jacobi' if USE_PG else 'Newton+CG (clipped to [0,1])'} "
          f"sub-solver  [PFF_DAMAGE_SOLVER={DAMAGE_SOLVER}]")
    print(f"  solver    : max {MAX_STAGGER} staggers x {NEWTON_MAX} Newton x "
          f"{CG_MAXITER} CG (rtol {CG_TOL:.0e}); damage {PG_MAXIT} "
          f"projected-Jacobi sweeps")
    print(f"  H freeze  : " + (f"H stops updating once d >= {D_FREEZE:.2f}"
          if D_FREEZE < 1.0 else "DISABLED -- open cracks keep accumulating H"))
    print(f"  H cap     : " + (f"H <= {H_CAP:.0f} x H_crit per tissue "
          f"({H_CAP * H_CRIT_ENAMEL:.3f}/{H_CAP * H_CRIT_DENTIN:.3f} MPa); d -> 1 needs "
          f"{(2.0 * min(L_CH_ENAMEL, L_CH_DENTIN) / (onp.pi * l)) ** 2:.2f} x"
          if H_CAP > 0 else "DISABLED -- H can blow up inside an open crack"))
    print(f"  stop rule : " + (f"post-failure stop when the load falls below "
          f"{100 * STOP_LOAD_FRAC:.0f}% of its peak" if STOP_AFTER_FAILURE
          else "run to delta_max regardless of failure"))
    print(f"  loading   : imposed displacement 0 -> {DELTA_MAX:.3f} mm; "
          f"base and four lateral bone surfaces fixed", flush=True)

    # initial mesh with tissue + seed + gmsh physical-group fields, for ParaView
    save_sol(mesh, onp.asarray(sol_d),
             os.path.join(output_dir, f'mesh_start_{MODEL}.vtu'),
             point_infos=[('fracture_admissible', fracture_admissible)]
                         + gmsh_point_fields,
             cell_infos=cell_output_fields(cell_masks, gc_quad)
                        + [('seed_history', onp.asarray(history_old.mean(axis=1)))]
                        + gmsh_cell_fields)

    if os.environ.get('PFF_MESH_ONLY', '0') == '1':
        print(f"  mesh-only : wrote {os.path.join(output_dir, f'mesh_start_{MODEL}.vtu')}")
        return

    stepper = get_staggered_stepper(field_u, field_d, d_fixed_idx, MAX_STAGGER)
    fixed_idx, fixed_vals_zero, support_y_dofs = build_dirichlet_static(field_u, mat_quad)
    sol_u = np.asarray(sol_u); sol_d = np.asarray(sol_d)
    history_old = np.asarray(history_old)

    volumes = tet_volumes(pts, cells)
    _ctr = onp.asarray(pts)[onp.asarray(cells)].mean(axis=1)
    fissure_cells = ((onp.abs(_ctr[:, 0]) < FISSURE_COL_HALF_X)
                     & (_ctr[:, 1] > FISSURE_COL_Y_MIN)
                     & (cell_masks['enamel'] | cell_masks['dentin']))
    hist = {k: [] for k in ('delta_mm', 'reaction_N', 'max_d', 'cusp_uy_mm',
                             'damage_volume_mm3', 'enamel_damage_pct', 'dentin_damage_pct')}
    mvtk = os.path.join(vtk_dir, MODEL)
    os.makedirs(mvtk, exist_ok=True)
    zero_f = np.zeros(3 * len(pts))
    start = time.time()
    for k, delta in enumerate(levels):
        delta = float(delta)
        (sol_u, sol_d, history_old, n_stag, stag_err,
         pg_it, pg_chg) = stepper(
            sol_u, sol_d, history_old, fixed_idx, fixed_vals_zero,
            zero_f, mat_quad, gc_quad, a1_quad, delta)
        # The SOLVER'S history H, normalised by each tissue's onset threshold.
        # Written to every VTU so the driving force the damage equation actually
        # sees can be compared with the damage it produced -- an earlier run had an
        # externally computed psi+/H_crit of 4.6 at the fissure yet d = 0.02.
        _H = onp.asarray(history_old).max(axis=1)
        _Hc = onp.where(cell_masks['enamel'], H_CRIT_ENAMEL,
                        onp.where(cell_masks['dentin'], H_CRIT_DENTIN, onp.inf))
        H_ratio = _H / _Hc
        h_fis = float(onp.max(H_ratio[fissure_cells])) if fissure_cells.any() else 0.0

        # Fail fast on a destroyed solve.  Without this the NaN propagates and
        # every later step exits its while_loop instantly, so the run "finishes"
        # and writes a full set of all-NaN VTUs plus an empty diagnostics plot.
        u_bad = int(onp.count_nonzero(~onp.isfinite(onp.asarray(sol_u))))
        d_bad = int(onp.count_nonzero(~onp.isfinite(onp.asarray(sol_d))))
        if u_bad or d_bad:
            nu = onp.asarray(sol_u).size
            nd = onp.asarray(sol_d).size
            print(f"\n*** SOLVE DESTROYED at delta={delta:.3f} mm "
                  f"(step {k + 1}/{len(levels)}) ***", flush=True)
            print(f"    non-finite u : {u_bad}/{nu} components", flush=True)
            print(f"    non-finite d : {d_bad}/{nd} nodes", flush=True)
            if hist['delta_mm']:
                print(f"    last good    : delta={hist['delta_mm'][-1]:.3f} mm, "
                      f"reaction={hist['reaction_N'][-1]:.1f} N, "
                      f"max_d={hist['max_d'][-1]:.3f}", flush=True)
            else:
                print("    no completed step", flush=True)
            print("    Try, in order:  PFF_K_RES=1e-6 (ill-conditioned tangent "
                  "at d->1);  PFF_ELASTIC_ONLY=1 (is the contact tangent alone "
                  "at fault?);  lower PFF_CONTACT_PENALTY;  smaller "
                  "PFF_DELTA_STEP.", flush=True)
            print("    Writing the partial history and stopping.\n", flush=True)
            break

        d_quad_u = field_u.to_quad(sol_d)
        residual_internal, _ = assemble_device(
            field_u, sol_u.reshape(-1), [d_quad_u, mat_quad], zero_f,
            contact_surface_vars(field_u, delta))
        reaction = float(onp.abs(
            onp.asarray(residual_internal)[onp.asarray(support_y_dofs)].sum()))
        max_d = float(np.max(sol_d))
        disp_uy = float(sol_u[disp_node, 1])
        d_host = onp.asarray(sol_d).reshape(-1)
        d_cell = d_host[onp.asarray(cells)].mean(axis=1)
        damage_volume = float(onp.sum(volumes * d_cell))
        enamel_volume = max(float(volumes[cell_masks['enamel']].sum()), 1e-30)
        dentin_volume = max(float(volumes[cell_masks['dentin']].sum()), 1e-30)
        enamel_damage = 100.0 * float(onp.sum(volumes[cell_masks['enamel']]
                                              * d_cell[cell_masks['enamel']])) / enamel_volume
        dentin_damage = 100.0 * float(onp.sum(volumes[cell_masks['dentin']]
                                              * d_cell[cell_masks['dentin']])) / dentin_volume
        hist['delta_mm'].append(delta); hist['reaction_N'].append(reaction)
        hist['max_d'].append(max_d); hist['cusp_uy_mm'].append(disp_uy)
        hist['damage_volume_mm3'].append(damage_volume)
        hist['enamel_damage_pct'].append(enamel_damage)
        hist['dentin_damage_pct'].append(dentin_damage)
        print(f"Step {k + 1:4d}/{len(levels)}  delta={delta:6.3f} mm  "
              f"reaction={reaction:8.1f} N  max_d={max_d:.3f}  "
              f"damage E/D={enamel_damage:.3f}/{dentin_damage:.3f}%  "
              f"H_fis={h_fis:.2f}  "
              f"stag={int(n_stag)}/{MAX_STAGGER} err={float(stag_err):.1e}  "
              + (f"pg={int(pg_it)}/{PG_MAXIT} chg={float(pg_chg):.1e}  " if USE_PG else "")
              + f"t={time.time()-start:.1f}s", flush=True)
        is_target = onp.any(onp.isclose(delta, FIG5_TARGETS, atol=1e-10))
        in_refine_window = (REFINE_WINDOW is not None
                            and REFINE_WINDOW[0] - 1e-9 <= delta <= REFINE_WINDOW[1] + 1e-9)
        if (k % SAVE_EVERY == 0) or (k == len(levels) - 1) or is_target or in_refine_window:
            save_sol(mesh, onp.asarray(sol_d),
                     os.path.join(mvtk, f'premolar_delta_{delta:0.3f}.vtu'),
                     point_infos=[('u', onp.asarray(sol_u)),
                                  ('fracture_admissible', fracture_admissible)],
                     cell_infos=cell_output_fields(cell_masks, gc_quad)
                                + [('H_ratio', H_ratio)])
        if is_target:
            save_sol(mesh, onp.asarray(sol_d),
                     os.path.join(output_dir, f'target_delta_{delta:0.2f}_{MODEL}.vtu'),
                     point_infos=[('u', onp.asarray(sol_u)),
                                  ('fracture_admissible', fracture_admissible)],
                     cell_infos=cell_output_fields(cell_masks, gc_quad)
                                + [('H_ratio', H_ratio)])

        peak = max(hist['reaction_N'])
        if (STOP_AFTER_FAILURE and max_d > 0.9 and peak > 1.0
                and reaction < STOP_LOAD_FRAC * peak):
            d_peak = hist['delta_mm'][int(onp.argmax(onp.asarray(hist['reaction_N'])))]
            print(f"\n*** POST-FAILURE STOP at delta={delta:.3f} mm ***", flush=True)
            print(f"    reaction {reaction:.1f} N is below "
                  f"{100 * STOP_LOAD_FRAC:.0f}% of the peak {peak:.1f} N "
                  f"(reached at delta={d_peak:.3f} mm).", flush=True)
            print(f"    The crown has separated; every later state is diffuse "
                  f"post-peak damage, not crack growth.", flush=True)
            print(f"    Set PFF_STOP_AFTER_FAILURE=0 to run to delta_max anyway.\n",
                  flush=True)
            break

    solve_time = time.time() - start
    wall_time = time.time() - t_wall0
    n_done = len(hist['delta_mm'])
    early_note = ' (STOPPED EARLY - solve destroyed)' if n_done < len(levels) else ''
    final_reaction = (f"{hist['reaction_N'][-1]:.1f} N" if n_done else "n/a")
    # Report the last COMPLETED step; sol_d may hold the destroyed state.
    final_max_d = (f"{hist['max_d'][-1]:.3f}" if n_done else "n/a")
    hist = {k: onp.asarray(v) for k, v in hist.items()}
    onp.savez(os.path.join(output_dir, f'history_{MODEL}.npz'),
              points=onp.asarray(pts), cells=onp.asarray(cells),
              sol_d=onp.asarray(sol_d).reshape(-1),
              solve_time=float(solve_time), wall_time=float(wall_time), **hist)

    summary = (
        "==================== PREMOLAR / TIMING SUMMARY ====================\n"
        f"  model            : {MODEL}\n"
        f"  mesh             : {ELE} ({len(cells)} tets, {len(pts)} nodes), z=[{Z_MIN:.2f}, {Z_MAX:.2f}] mm\n"
        f"  tissues          : enamel (E={E_ENAMEL/1e3:.0f} GPa, Gc={GC_ENAMEL*1e3:.0f} J/m2) "
        f"/ dentin (E={E_DENTIN/1e3:.0f} GPa, Gc={GC_DENTIN*1e3:.0f} J/m2)\n"
        f"  model type       : hybrid (isotropic degraded stress; {DRIVING} history)\n"
        f"  geometry         : conforming tooth/PDL/bone volumes (idealised; CBCT unavailable)\n"
        f"  initial flaw     : {'none (intact tooth)' if not SEED_CRACK else f'seeded crack on x=0, {SEED_DEPTH:.2f} mm below the fissure'}\n"
        f"  fracture domain  : crown enamel/dentin, y>={FRACTURE_Y_MIN:.2f} mm\n"
        f"  loading          : unilateral frictionless 4 mm rigid ball, delta=0 -> {DELTA_MAX:.3f} mm\n"
        f"  steps completed  : {n_done}/{len(levels)}{early_note}\n"
        f"  final reaction   : {final_reaction}\n"
        f"  support          : bone base and four lateral surfaces fixed\n"
        f"  final max damage : {final_max_d}\n"
        f"  TOTAL solve time : {solve_time:.1f} s  ({solve_time/60.:.2f} min)\n"
        f"  TOTAL wall time  : {wall_time:.1f} s  ({wall_time/60.:.2f} min)\n"
        "==================================================================")
    print("\n" + summary, flush=True)
    with open(os.path.join(output_dir, f'timing_{MODEL}.txt'), 'w') as fh:
        fh.write(summary + "\n")

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(hist['delta_mm'], hist['reaction_N'], 'b-o', ms=3)
    axs[0].axhline(693.104, color='0.5', ls='--', lw=1, label='Zhu et al.: initiation')
    axs[0].axhline(1356.0, color='0.2', ls=':', lw=1, label='Zhu et al.: fracture')
    axs[0].set_xlabel('imposed ball displacement [mm]')
    axs[0].set_ylabel('reaction force [N]')
    axs[0].set_title('Load-displacement'); axs[0].legend(fontsize=7)
    axs[1].plot(hist['delta_mm'], hist['enamel_damage_pct'], 'r-o', ms=3, label='enamel')
    axs[1].plot(hist['delta_mm'], hist['dentin_damage_pct'], 'b-o', ms=3, label='dentin')
    axs[1].set_xlabel('imposed ball displacement [mm]')
    axs[1].set_ylabel('volume-averaged damage [%]')
    axs[1].set_title('Tissue damage'); axs[1].legend(fontsize=7)
    for ax in axs:
        ax.grid(which='major', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'diagnostics_{MODEL}.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
