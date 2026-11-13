"""Self-contained checks of the axion core on tiny meshes (CPU, a few seconds).

    python tests/test_core.py        # or:  pytest tests
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as onp

import axion
from axion import DirichletBC, Field, Mesh, box_mesh, rectangle_mesh
from axion.elements import cell_quadrature, reference_element


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def tet_box(n):
    """Unit cube split into tetrahedra (six per hexahedron)."""
    grid = box_mesh(n, n, n)
    splits = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
    cells = onp.concatenate([grid.cells[:, t] for t in splits])
    return Mesh(grid.points, cells, 'TET4')


def distorted(mesh, amplitude=0.03, seed=0):
    rng = onp.random.default_rng(seed)
    pts = mesh.points.copy()
    interior = onp.all((pts > 1e-9) & (pts < 1. - 1e-9), axis=1)
    pts[interior] += amplitude * rng.standard_normal((int(interior.sum()), mesh.dim))
    return Mesh(pts, mesh.cells, mesh.ele_type)


def all_meshes():
    return [distorted(rectangle_mesh(4, 3)), distorted(rectangle_mesh(4, 3, ele_type='TRI3')),
            distorted(box_mesh(3, 2, 2)), distorted(tet_box(2))]


LAMBDA, MU = 1.3, 0.8


def linear_stress(grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    return LAMBDA * jnp.trace(eps) * jnp.eye(grad_u.shape[0]) + 2. * MU * eps


class LinearElasticity(Field):
    def flux(self):
        return linear_stress


class NonlinearScalar(Field):
    """A smooth nonlinear weak form with every kind of term."""
    def flux(self):
        return lambda g, a: (1. + a + jnp.sum(g * g)) * g

    def source(self):
        return lambda u, x, a: jnp.sin(u) + a * u ** 3 + x[0]

    def boundary_sources(self):
        return [lambda u, x: u ** 2 - 1.]


# ----------------------------------------------------------------------------
# reference elements
# ----------------------------------------------------------------------------
def test_quadrature_integrates_polynomials():
    # int over the unit triangle of x^a y^b = a! b! / (a + b + 2)!
    pts, w = cell_quadrature('TRI3', 2)
    assert abs(w.sum() - 0.5) < 1e-14
    assert abs((w * pts[:, 0] * pts[:, 1]).sum() - 1. / 24.) < 1e-14
    pts, w = cell_quadrature('TET4', 2)
    assert abs(w.sum() - 1. / 6.) < 1e-14
    assert abs((w * pts[:, 0] * pts[:, 1]).sum() - 1. / 120.) < 1e-14
    pts, w = cell_quadrature('HEX8', 2)        # exact up to degree 3 per direction
    assert abs(w.sum() - 8.) < 1e-14
    assert abs((w * pts[:, 0] ** 2 * pts[:, 2] ** 2).sum() - 8. / 9.) < 1e-14


def test_shape_functions_are_a_nodal_basis():
    for ele in ('TRI3', 'QUAD4', 'TET4', 'HEX8'):
        ref = reference_element(ele)
        N, dN = axion.elements.shape_functions(ele, ref.node_coords)
        assert onp.allclose(N, onp.eye(ref.num_nodes), atol=1e-14)
        assert onp.allclose(ref.shape_vals.sum(axis=1), 1.)
        assert onp.allclose(ref.shape_grads_ref.sum(axis=1), 0., atol=1e-14)
        assert onp.allclose(ref.face_shape_vals.sum(axis=2), 1.)


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def test_volume_and_closed_surface():
    for mesh in all_meshes():
        field = Field(mesh, ncomp=1)
        assert abs(float(jnp.sum(field.JxW)) - 1.) < 1e-12, mesh.ele_type
        faces = field.face_set(cell_ids=mesh.boundary_faces()[0], face_ids=mesh.boundary_faces()[1])
        # sum of n dS over a closed surface vanishes; flux of x equals dim * volume
        assert float(jnp.linalg.norm(jnp.einsum('sqd,sq->d', faces.normals, faces.dS))) < 1e-12
        flux = float(jnp.einsum('sqd,sqd,sq->', faces.quad_points, faces.normals, faces.dS))
        assert abs(flux - mesh.dim) < 1e-12, mesh.ele_type


def test_gradient_of_a_linear_field_is_exact():
    for mesh in all_meshes():
        field = Field(mesh, ncomp=mesh.dim)
        A = onp.random.default_rng(1).standard_normal((mesh.dim, mesh.dim))
        u = mesh.points @ A.T
        grads = field.grad_at_quad(jnp.asarray(u))
        assert float(jnp.max(jnp.abs(grads - A))) < 1e-12, mesh.ele_type


# ----------------------------------------------------------------------------
# residual and tangent
# ----------------------------------------------------------------------------
def test_patch_test():
    """A linear displacement field is an exact solution: zero residual at the
    interior nodes of a distorted mesh."""
    for mesh in all_meshes():
        field = LinearElasticity(mesh, ncomp=mesh.dim)
        A = onp.random.default_rng(2).standard_normal((mesh.dim, mesh.dim))
        R = field.residual(jnp.asarray(mesh.points @ A.T).reshape(-1)).reshape(mesh.num_nodes, -1)
        interior = onp.all((mesh.points > 1e-9) & (mesh.points < 1. - 1e-9), axis=1)
        assert float(jnp.max(jnp.abs(R[interior]))) < 1e-12, mesh.ele_type


def test_tangent_is_the_derivative_of_the_residual():
    on_side = lambda p: jnp.isclose(p[0], 1., atol=1e-8)
    rng = onp.random.default_rng(3)
    for mesh in all_meshes():
        field = NonlinearScalar(mesh, ncomp=1, boundaries=[on_side])
        a = jnp.asarray(rng.uniform(0.2, 1., (mesh.num_cells, field.num_quads)))
        z = jnp.asarray(rng.standard_normal(mesh.num_nodes))
        R, tangents = field.residual_and_tangent(z, (a,))
        dense = jax.jacfwd(lambda x: field.residual(x, (a,)))(z)
        # stored entries, operator action and diagonal against the dense Jacobian
        K = onp.zeros((field.num_dofs, field.num_dofs))
        onp.add.at(K, (field.coo_rows, field.coo_cols), onp.asarray(field.tangent_values(tangents)))
        assert onp.allclose(K, dense, atol=1e-11), mesh.ele_type
        x = jnp.asarray(rng.standard_normal(mesh.num_nodes))
        assert jnp.allclose(field.apply(tangents, x), dense @ x, atol=1e-11)
        assert jnp.allclose(field.diagonal(tangents), jnp.diagonal(dense), atol=1e-11)
        # Taylor remainder: second order
        dz = jnp.asarray(rng.standard_normal(mesh.num_nodes))
        r1 = [float(jnp.linalg.norm(field.residual(z + h * dz, (a,)) - R - h * (dense @ dz)))
              for h in (1e-2, 1e-3)]
        assert 1.9 < onp.log10(r1[0] / r1[1]) < 2.1, (mesh.ele_type, r1)


def test_dirichlet_sets():
    mesh = rectangle_mesh(4, 4)
    bottom = lambda p: jnp.isclose(p[1], 0.)
    left = lambda p: jnp.isclose(p[0], 0.)
    field = Field(mesh, ncomp=2, dirichlet=[DirichletBC(bottom, 0, 0.), DirichletBC(bottom, 1, 0.),
                                             DirichletBC(left, 0, lambda p: 2. * p[1])])
    dofs = onp.asarray(field.dirichlet_dofs)
    assert len(dofs) == len(set(dofs.tolist())) == 5 + 5 + 4       # the corner is not repeated
    vals = dict(zip(dofs.tolist(), onp.asarray(field.dirichlet_values).tolist()))
    node = 2 * 5                                                   # (x, y) = (0, 0.5)
    assert abs(vals[2 * node] - 1.0) < 1e-14
    assert int(field.dirichlet_mask(2).sum()) == 5


# ----------------------------------------------------------------------------
# spectral functions
# ----------------------------------------------------------------------------
def test_isotropic_function_derivative():
    F = axion.isotropic_function(lambda lam, g: g * axion.positive_part(lam) ** 2
                                 + axion.negative_part(lam))
    rng = onp.random.default_rng(4)
    for dim in (2, 3):
        A = rng.standard_normal((dim, dim)); A = jnp.asarray(0.5 * (A + A.T))
        V = rng.standard_normal((dim, dim)); V = jnp.asarray(0.5 * (V + V.T))
        g, dg, h = jnp.asarray(0.7), jnp.asarray(0.3), 1e-6
        _, jvp = jax.jvp(F, (A, g), (V, dg))
        fd = (F(A + h * V, g + h * dg) - F(A - h * V, g - h * dg)) / (2. * h)
        assert float(jnp.max(jnp.abs(jvp - fd))) < 1e-8
    # repeated eigenvalues: f(lam) = lam^2 gives F(A) = A^2 and DF[V] = A V + V A
    square = axion.isotropic_function(lambda lam: lam ** 2)
    for A in (jnp.zeros((3, 3)), 0.4 * jnp.eye(3), jnp.diag(jnp.array([1., 1., -2.]))):
        V = rng.standard_normal((3, 3)); V = jnp.asarray(0.5 * (V + V.T))
        value, jvp = jax.jvp(square, (A,), (V,))
        assert bool(jnp.all(jnp.isfinite(jvp)))
        assert jnp.allclose(value, A @ A, atol=1e-13)
        assert jnp.allclose(jvp, A @ V + V @ A, atol=1e-12)


def test_closed_form_eigenvalues():
    rng = onp.random.default_rng(5)
    for _ in range(5):
        A = rng.standard_normal((3, 3)); A = 0.5 * (A + A.T)
        assert onp.allclose(axion.eigvalsh_3x3(jnp.asarray(A)), onp.linalg.eigvalsh(A), atol=1e-12)
    D = onp.diag([0.3, -1.2, 0.3])
    assert onp.allclose(axion.eigvalsh_3x3(jnp.asarray(D)), [-1.2, 0.3, 0.3], atol=1e-14)


# ----------------------------------------------------------------------------
# solvers
# ----------------------------------------------------------------------------
def test_newton_cg_against_a_dense_solve():
    mesh = distorted(rectangle_mesh(6, 6), amplitude=0.01)
    bottom = lambda p: jnp.isclose(p[1], 0.)
    top = lambda p: jnp.isclose(p[1], 1.)
    field = LinearElasticity(mesh, ncomp=2, dirichlet=[
        DirichletBC(bottom, 0, 0.), DirichletBC(bottom, 1, 0.), DirichletBC(top, 1, 0.1)])
    fixed, values = field.dirichlet_dofs, field.dirichlet_values
    z0 = jnp.zeros(field.num_dofs).at[fixed].set(values)
    R0, V0 = axion.assemble(field, z0)
    newton = axion.make_newton(field, max_iter=5, tol=1e-12, cg_tol=1e-13, cg_maxiter=2000)
    z = newton(z0, R0.at[fixed].set(0.), V0, (), fixed)

    K = onp.asarray(jax.jacfwd(field.residual)(jnp.zeros(field.num_dofs)))
    free = onp.setdiff1d(onp.arange(field.num_dofs), onp.asarray(fixed))
    ref = onp.asarray(z0).copy()
    ref[free] = onp.linalg.solve(K[onp.ix_(free, free)], -(K @ onp.asarray(z0))[free])
    assert onp.linalg.norm(onp.asarray(z) - ref) / onp.linalg.norm(ref) < 1e-9


def test_bound_constrained_solvers_agree():
    """Obstacle-type problem: both bound-constrained solvers reach the same state,
    which satisfies the optimality conditions."""
    class Obstacle(Field):
        def flux(self):
            return lambda g, f: 0.05 * g

        def source(self):
            return lambda u, x, f: u - f

    mesh = rectangle_mesh(8, 8)
    field = Obstacle(mesh, ncomp=1)
    xq = field.quad_points
    f = 2.5 * jnp.sin(3. * xq[..., 0]) * jnp.cos(2. * xq[..., 1])      # pushes beyond [0, 1]
    z0 = jnp.full(field.num_dofs, 0.5)
    D = axion.abs_diagonal(field, z0, (f,))
    z_pj = axion.make_projected_jacobi(field, max_iter=5000, tol=1e-13)(z0, D, (f,))
    none = jnp.array([], dtype=jnp.int32)
    z_as = axion.make_active_set_newton(field, max_iter=50, tol=1e-12, cg_tol=1e-13)(
        z0, D, (f,), none, jnp.zeros(0))
    assert float(jnp.max(jnp.abs(z_pj - z_as))) < 1e-7
    R = field.residual(z_as, (f,))
    inside = (z_as > 1e-9) & (z_as < 1. - 1e-9)
    assert float(jnp.max(jnp.abs(jnp.where(inside, R, 0.)))) < 1e-8
    assert bool(jnp.all(jnp.where(z_as <= 1e-9, R, 0.) >= -1e-8))       # lower bound: R >= 0
    assert bool(jnp.all(jnp.where(z_as >= 1. - 1e-9, R, 0.) <= 1e-8))   # upper bound: R <= 0
    assert int(jnp.sum(~inside)) > 0                                    # bounds are active


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"{len(tests)} tests passed")
