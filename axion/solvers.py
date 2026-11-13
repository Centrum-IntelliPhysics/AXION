"""Linear and nonlinear solvers that run inside one compiled program.

All iterations are written with JAX control-flow primitives, so a solver can be
called from a staggered ``while_loop`` inside a load-step ``scan`` without
leaving the device:

* :func:`make_operator`         element-by-element operator action and Jacobi
                                preconditioner, with Dirichlet unknowns
                                eliminated symmetrically
* :func:`make_newton`           Newton iterations with Jacobi-preconditioned CG
* :func:`make_projected_jacobi` bound-constrained subproblem by projected
                                Jacobi iteration (needs only residuals)
* :func:`make_active_set_newton` bound-constrained subproblem by a primal-dual
                                active-set (semismooth Newton) iteration
"""
import jax
import jax.numpy as jnp


def make_operator(field):
    """Global operator of ``field`` applied from its stored element tangents.

    Returns ``(matvec, inv_jacobi)``:

    ``matvec(V, x, fixed)``   y = K x, where ``V`` holds the element tangent
        entries (``field.tangent_values``) and ``fixed`` the indices of the
        prescribed unknowns. Their columns are dropped and their rows replaced
        by the identity, which keeps the operator symmetric.
    ``inv_jacobi(V, fixed)``  inverse of the diagonal of the same operator.

    No sparse matrix is assembled: every stored entry multiplies the gathered
    value of x and is scatter-added to its row.
    """
    n = field.num_dofs
    rows = jnp.asarray(field.coo_rows)
    cols = jnp.asarray(field.coo_cols)
    on_diagonal = rows == cols

    @jax.jit
    def matvec(V, x, fixed):
        x_free = x.at[fixed].set(0.)
        y = jnp.zeros(n, dtype=x.dtype).at[rows].add(V * x_free[cols])
        return y.at[fixed].set(x[fixed], unique_indices=True)

    @jax.jit
    def inv_jacobi(V, fixed):
        diag = jnp.zeros(n, dtype=V.dtype).at[rows].add(jnp.where(on_diagonal, V, 0.))
        diag = diag.at[fixed].set(1.)
        return 1. / jnp.where(jnp.abs(diag) > 1e-12, diag, 1.)

    return matvec, inv_jacobi


def assemble(field, z, theta=(), theta_boundaries=None):
    """Residual and stored tangent entries of ``field`` at the flat state ``z``."""
    R, tangents = field.residual_and_tangent(z, theta, theta_boundaries)
    return R, field.tangent_values(tangents)


def make_newton(field, max_iter=30, tol=1e-9, cg_tol=1e-8, cg_maxiter=5000):
    """Newton's method for one field with the other data ``theta`` held fixed.

    Returns the compiled function ``newton(z0, R0, V0, theta, fixed)``; ``z0``
    already carries the prescribed values, and ``R0``, ``V0`` are the residual
    (zero on the prescribed unknowns) and tangent entries at ``z0``. Every
    update solves K dz = -R by conjugate gradients with a Jacobi
    preconditioner, to the relative tolerance ``cg_tol``. Iterations stop when
    ||R||_2 <= ``tol`` or after ``max_iter`` updates.
    """
    matvec, inv_jacobi = make_operator(field)

    @jax.jit
    def newton(z0, R0, V0, theta, fixed):
        def cond(state):
            _, R, _, it = state
            return (it < max_iter) & (jnp.linalg.norm(R) > tol)

        def body(state):
            z, R, V, it = state
            inv = inv_jacobi(V, fixed)
            dz, _ = jax.scipy.sparse.linalg.cg(lambda x: matvec(V, x, fixed), -R,
                                               tol=cg_tol, atol=0., maxiter=cg_maxiter,
                                               M=lambda r: r * inv)
            z_new = z + dz
            R_new, V_new = assemble(field, z_new, theta)
            return z_new, R_new.at[fixed].set(0.), V_new, it + 1

        return jax.lax.while_loop(cond, body, (z0, R0, V0, 0))[0]

    return newton


def abs_diagonal(field, z, theta=()):
    """|diag K| of ``field`` at ``z``, with vanishing entries replaced by one.
    This is the scaling D of the projected Jacobi iteration."""
    _, K = field.evaluate_cells(z, theta, tangent=True)
    diag = jnp.zeros(field.num_dofs, dtype=K.dtype).at[field.cell_dofs.reshape(-1)].add(
        jnp.diagonal(K, axis1=1, axis2=2).reshape(-1))
    diag = jnp.abs(diag)
    return jnp.where(diag > 1e-12, diag, 1.)


def make_projected_jacobi(field, max_iter=300, tol=1e-7, relax=1., lower=0., upper=1.):
    """Bound-constrained solve  lower <= z <= upper  by projected Jacobi sweeps,

        z <- clip(z - relax * R(z) / D, lower, upper),    D = |diag K| .

    Only residual evaluations are needed and the absolute value keeps the
    scaling positive, so the iteration applies to non-convex subproblems whose
    tangent is indefinite. Returns the compiled function ``solve(z0, D, theta)``;
    it stops when the relative change of z falls below ``tol`` or after
    ``max_iter`` sweeps.
    """
    @jax.jit
    def solve(z0, D, theta):
        def cond(state):
            z, z_prev, it = state
            change = jnp.linalg.norm(z - z_prev) / (jnp.linalg.norm(z) + 1e-30)
            return (it < max_iter) & (change > tol)

        def body(state):
            z, _, it = state
            R = field.residual(z, theta)
            return jnp.clip(z - relax * R / D, lower, upper), z, it + 1

        z_start = jnp.clip(z0, lower, upper)
        return jax.lax.while_loop(cond, body, (z_start, z_start + 1., 0))[0]

    return solve


def make_active_set_newton(field, max_iter=50, tol=1e-8, cg_tol=1e-6, cg_maxiter=2000,
                           lower=0., upper=1.):
    """Bound-constrained solve  lower <= z <= upper  by a primal-dual active-set
    (semismooth Newton) iteration.

    In every iteration the unknowns whose projected Jacobi trial value
    z - R / D leaves the admissible interval are held at the bound they hit;
    the remaining ones are updated by one Newton step, whose linear system is
    solved by Jacobi-preconditioned CG with the held unknowns eliminated.
    Unknowns listed in ``pinned`` are always held at ``pinned_vals`` (for
    example a pre-crack imposed as d = 1).

    Returns the compiled function ``solve(z0, D, theta, pinned, pinned_vals)``;
    it stops when the relative change of z falls below ``tol`` or after
    ``max_iter`` iterations. The Newton step requires a positive definite
    tangent on the free unknowns.
    """
    n = field.num_dofs
    n_cell_entries = field.num_cells * field.dofs_per_cell ** 2
    rows = jnp.asarray(field.coo_rows[:n_cell_entries])
    cols = jnp.asarray(field.coo_cols[:n_cell_entries])
    on_diagonal = rows == cols

    def tangent_values(z, theta):
        _, K = field.evaluate_cells(z, theta, tangent=True)
        return K.reshape(-1)

    def masked_matvec(V, held, p):
        p_free = jnp.where(held, 0., p)
        Ap = jnp.zeros(n, dtype=p.dtype).at[rows].add(V * p_free[cols])
        return jnp.where(held, p, Ap)

    def inv_jacobi_masked(V, held):
        diag = jnp.zeros(n, dtype=V.dtype).at[rows].add(jnp.where(on_diagonal, V, 0.))
        diag = jnp.where(held, 1., diag)
        return 1. / jnp.where(jnp.abs(diag) > 1e-12, diag, 1.)

    @jax.jit
    def solve(z0, D, theta, pinned, pinned_vals):
        pinned_mask = jnp.zeros(n, dtype=bool).at[pinned].set(True)
        pinned_target = jnp.zeros(n).at[pinned].set(pinned_vals)
        z_start = jnp.clip(jnp.asarray(z0), lower, upper).at[pinned].set(pinned_vals)
        inv_D = 1. / D

        def cond(state):
            _, it, converged = state
            return (it < max_iter) & (~converged)

        def body(state):
            z, it, _ = state
            R = field.residual(z, theta)

            # active set from the projected Jacobi trial value
            trial = z - inv_D * R
            at_lower = trial <= lower
            at_upper = trial >= upper
            target = jnp.where(at_lower, lower, jnp.where(at_upper, upper, z))

            held = at_lower | at_upper | pinned_mask
            target = jnp.where(pinned_mask, pinned_target, target)

            # move the held unknowns to their bounds before linearizing, so that
            # the jump enters the residual of the free unknowns
            z_shift = jnp.where(held, target, z)
            R_shift = jnp.where(held, 0., field.residual(z_shift, theta))
            V_shift = tangent_values(z_shift, theta)

            inv = inv_jacobi_masked(V_shift, held)
            delta, _ = jax.scipy.sparse.linalg.cg(
                lambda p: masked_matvec(V_shift, held, p), -R_shift,
                tol=cg_tol, atol=0., maxiter=cg_maxiter, M=lambda r: r * inv)

            z_new = jnp.clip(z_shift + delta, lower, upper).at[pinned].set(pinned_vals)
            change = jnp.linalg.norm(z_new - z) / (jnp.linalg.norm(z_new) + 1e-30)
            return z_new, it + 1, change < tol

        return jax.lax.while_loop(cond, body, (z_start, jnp.array(0), jnp.array(False)))[0]

    return solve
