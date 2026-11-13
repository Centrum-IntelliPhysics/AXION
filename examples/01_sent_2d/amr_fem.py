"""Matrix-free QUAD4 finite-element operators for the fixed-capacity AMR mesh,
plus a constrained conjugate-gradient solver that folds hanging-node and
Dirichlet constraints directly into the mat-vec.

Everything here is fully on device and shape-static: the operators loop over the
FULL C_cap cell buffer and mask out the dead slots, so a jitted caller compiles
once and never recompiles when the mesh changes.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


# QUAD4 reference element: 2x2 Gauss quadrature.
# Node order matches the mesh: 0=(-1,-1) 1=(1,-1) 2=(1,1) 3=(-1,1).
_G = 1.0 / jnp.sqrt(jnp.array(3.0))
_GP = jnp.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]])
_GW = jnp.ones(4)
_XI = jnp.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])


def _shape(xi, eta):
    return 0.25 * (1. + _XI[:, 0] * xi) * (1. + _XI[:, 1] * eta)


def _shape_grad_ref(xi, eta):
    dN_dxi = 0.25 * _XI[:, 0] * (1. + _XI[:, 1] * eta)
    dN_deta = 0.25 * _XI[:, 1] * (1. + _XI[:, 0] * xi)
    return jnp.stack([dN_dxi, dN_deta], axis=1)


_N_GP = jnp.stack([_shape(gp[0], gp[1]) for gp in _GP])
_DNREF_GP = jnp.stack([_shape_grad_ref(gp[0], gp[1]) for gp in _GP])
_DNREF_CENTER = _shape_grad_ref(0.0, 0.0)


def cell_geometry(node_xy4):
    """Per-cell geometry at the 4 Gauss points.

    Returns gradN_phys (4qp, 4nod, 2) and detJxW (4qp,).  Uses a division-free
    guarded inverse so a degenerate (dead/padding) cell yields gradN = 0 and
    detJ = 0 with no NaN, contributing exactly zero to every integral.
    """
    def per_gp(dNref):
        J = node_xy4.T @ dNref
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        safe = jnp.where(detJ == 0., 1., detJ)
        inv_detJ = jnp.where(jnp.abs(detJ) > 1e-30, 1.0 / safe, 0.0)
        adjJ = jnp.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])
        gradN = (dNref @ adjJ.T) * inv_detJ
        return gradN, detJ
    gradN, detJ = jax.vmap(per_gp)(_DNREF_GP)
    return gradN, detJ * _GW


class Material(NamedTuple):
    E: float
    nu: float
    Gc: float
    l: float
    k_res: float

    @property
    def mu(self):
        return self.E / (2. * (1. + self.nu))

    @property
    def lmbda(self):
        return self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))


def _split_density(eps, d, mat):
    """Miehe spectral split: (g(d)+k) psi_plus + psi_minus (plane strain)."""
    tr = jnp.trace(eps)
    a, b, c = eps[0, 0], eps[1, 1], eps[0, 1]
    # A tiny regularisation makes the algorithmic tangent finite at repeated
    # eigenvalues and at the positive/negative Macaulay-bracket corner.
    disc = jnp.sqrt(((a - b) * 0.5) ** 2 + c ** 2 + 1e-24)
    eig = jnp.array([0.5 * (a + b) + disc, 0.5 * (a + b) - disc])
    pos = lambda x: 0.5 * (x + jnp.sqrt(x * x + 1e-24))
    neg = lambda x: 0.5 * (x - jnp.sqrt(x * x + 1e-24))
    psi_p = 0.5 * mat.lmbda * pos(tr) ** 2 + mat.mu * jnp.sum(pos(eig) ** 2)
    psi_m = 0.5 * mat.lmbda * neg(tr) ** 2 + mat.mu * jnp.sum(neg(eig) ** 2)
    return ((1. - d) ** 2 + mat.k_res) * psi_p + psi_m


def elasticity_cell_residual_tangent(node_xy4, u4, d4, mat):
    """Internal residual and consistent 8x8 tangent of split elastic energy."""
    gradN, detJxW = cell_geometry(node_xy4)

    def energy(ue_flat):
        ue = ue_flat.reshape(4, 2)
        def per_gp(qp):
            gu = ue.T @ gradN[qp]
            eps = 0.5 * (gu + gu.T)
            d_qp = jnp.dot(_N_GP[qp], d4)
            return _split_density(eps, d_qp, mat) * detJxW[qp]
        return jnp.sum(jax.vmap(per_gp)(jnp.arange(4)))

    ue = u4.reshape(-1)
    return jax.grad(energy)(ue), jax.hessian(energy)(ue)


def phasefield_cell_operators(node_xy4, hist4, mat):
    """AT2: return (A_d 4x4, b_d 4) with A_d = Gc*l K_lap + (Gc/l+2H) M, b_d=2H(M@1)."""
    gradN, detJxW = cell_geometry(node_xy4)
    Gc, l = mat.Gc, mat.l

    def per_gp(qp):
        g = gradN[qp]
        N = _N_GP[qp]
        H = hist4[qp]
        K_lap = (g @ g.T) * (Gc * l)
        M = jnp.outer(N, N)
        A = (K_lap + (Gc / l + 2. * H) * M) * detJxW[qp]
        b = (2. * H * N) * detJxW[qp]
        return A, b
    A, b = jax.vmap(per_gp)(jnp.arange(4))
    return jnp.sum(A, axis=0), jnp.sum(b, axis=0)


def _psi_plus(eps, mat):
    """Spectral tensile strain energy of a 2x2 symmetric strain tensor."""
    tr = eps[0, 0] + eps[1, 1]
    a, b, c = eps[0, 0], eps[1, 1], eps[0, 1]
    disc = jnp.sqrt(((a - b) * 0.5) ** 2 + c ** 2)
    e1 = 0.5 * (a + b) + disc
    e2 = 0.5 * (a + b) - disc
    pos = lambda x: 0.5 * (x + jnp.abs(x))
    return 0.5 * mat.lmbda * pos(tr) ** 2 + mat.mu * (pos(e1) ** 2 + pos(e2) ** 2)


def update_history(node_xy, cell_nodes, cell_valid, u_nodal, hist_old, mat):
    """FEniCSx-reference DG0 history, broadcast to the four Gauss points.

    FEniCSx stores one history value per cell and evaluates psi+ at the DG0
    interpolation point (the reference-cell centre).  Keep the existing
    fixed-shape ``(cell, 4)`` buffer, but make all four entries identical.
    """
    def per_cell(nodes, valid, h_old):
        X = node_xy[nodes]
        ue = u_nodal[nodes]
        J = X.T @ _DNREF_CENTER
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        # Padding cells are degenerate.  A guarded explicit inverse prevents a
        # NaN from being formed before the validity mask is applied.
        safe_detJ = jnp.where(jnp.abs(detJ) > 1e-30, detJ, 1.0)
        invJ = jnp.array([[J[1, 1], -J[0, 1]],
                          [-J[1, 0], J[0, 0]]]) / safe_detJ
        gradN = _DNREF_CENTER @ invJ
        gu = ue.T @ gradN
        eps = 0.5 * (gu + gu.T)
        H = jnp.maximum(_psi_plus(eps, mat), h_old[0])
        return jnp.where(valid, jnp.full((4,), H), h_old)
    return jax.vmap(per_cell)(cell_nodes, cell_valid, hist_old)


def build_cell_dofs(cell_nodes, vec):
    base = cell_nodes[:, :, None] * vec + jnp.arange(vec)[None, None, :]
    return base.reshape(cell_nodes.shape[0], -1)


def assemble_elasticity(node_xy, cell_nodes, cell_valid, u_nodal, d_nodal, mat, Ndof):
    """Global split-stress residual and matrix-free consistent Newton tangent."""
    def per_cell(nod, valid):
        re, Ke = elasticity_cell_residual_tangent(
            node_xy[nod], u_nodal[nod], d_nodal[nod, 0], mat)
        return jnp.where(valid, re, 0.0), jnp.where(valid, Ke, 0.0)
    re, Ke = jax.vmap(per_cell)(cell_nodes, cell_valid)
    cell_dofs = build_cell_dofs(cell_nodes, 2)

    residual = jnp.zeros(Ndof).at[cell_dofs.reshape(-1)].add(re.reshape(-1))

    def matvec(u):
        ul = u[cell_dofs]
        fl = jnp.einsum('cij,cj->ci', Ke, ul)
        return jnp.zeros(Ndof).at[cell_dofs.reshape(-1)].add(fl.reshape(-1))

    diag = jnp.zeros(Ndof).at[cell_dofs.reshape(-1)].add(
        jax.vmap(jnp.diagonal)(Ke).reshape(-1))
    return residual, matvec, diag


def assemble_phasefield(node_xy, cell_nodes, cell_valid, history, mat, Ndof):
    """Matrix-free AT2 phase-field operator; returns (matvec, diag, rhs)."""
    def per_cell(nod, h, valid):
        A, b = phasefield_cell_operators(node_xy[nod], h, mat)
        return jnp.where(valid, A, 0.0), jnp.where(valid, b, 0.0)
    A, b = jax.vmap(per_cell)(cell_nodes, history, cell_valid)
    cell_dofs = build_cell_dofs(cell_nodes, 1)

    def matvec(d):
        dl = d[cell_dofs]
        fl = jnp.einsum('cij,cj->ci', A, dl)
        return jnp.zeros(Ndof).at[cell_dofs.reshape(-1)].add(fl.reshape(-1))

    rhs = jnp.zeros(Ndof).at[cell_dofs.reshape(-1)].add(b.reshape(-1))
    diag = jnp.zeros(Ndof).at[cell_dofs.reshape(-1)].add(
        jax.vmap(jnp.diagonal)(A).reshape(-1))
    return matvec, diag, rhs


def constrained_cg(matvec_raw, b, diag, free_mask, dir_val,
                   h_dof, a_dof, b_dof, w, tol=1e-8, maxiter=5000):
    """Solve A x = b with x[dir]=dir_val, x[hang]=w*(x[a]+x[b]).

    Reduced system  (Tt A T) z = Tt (b - A c),  x = T z + c.
    Padding hanging rows must be inert: h_dof=a_dof=b_dof=0, w=0.
    """
    def T(z):
        return z.at[h_dof].add(w * (z[a_dof] + z[b_dof]))

    def Tt(y):
        out = y * free_mask
        out = out.at[a_dof].add(w * y[h_dof])
        out = out.at[b_dof].add(w * y[h_dof])
        return out * free_mask

    # Offset c: Dirichlet values, PLUS hanging dofs whose master is a Dirichlet
    # node (its prescribed value must enter here since z is zero on Dirichlet
    # dofs and T only picks up free masters).
    c = dir_val.at[h_dof].add(w * (dir_val[a_dof] + dir_val[b_dof]))
    rhs = Tt(b - matvec_raw(c))
    dfree = jnp.where(free_mask & (jnp.abs(diag) > 1e-30), diag, 1.0)
    precond = lambda r: r / dfree

    def A_red(z):
        return Tt(matvec_raw(T(z)))

    z, _ = jax.scipy.sparse.linalg.cg(A_red, rhs, x0=jnp.zeros_like(b),
                                      tol=tol, atol=0.0, maxiter=maxiter, M=precond)
    return T(z) + c
