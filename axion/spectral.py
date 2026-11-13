"""Isotropic tensor functions of a symmetric tensor and the spectral
tension-compression split.

For a symmetric tensor A = sum_a lambda_a n_a (x) n_a and a scalar function f,

    F(A) = sum_a f(lambda_a) n_a (x) n_a .

Differentiating the eigenvectors directly fails when two eigenvalues coincide,
which happens at every quadrature point of an unloaded or hydrostatically
strained body. The directional derivative is therefore supplied in closed form
(Daleckii-Krein),

    DF(A)[V] = Q ( Theta o (Q^T V Q) ) Q^T ,
    Theta_ab = (f(lambda_a) - f(lambda_b)) / (lambda_a - lambda_b)   if lambda_a != lambda_b
             =  f'(lambda_a)                                         otherwise,

with Q the matrix of eigenvectors and o the entrywise product. This is the
tangent of the spectral split used in phase-field fracture (Miehe et al. 2010).
"""
import jax
import jax.numpy as jnp


def positive_part(x):
    """<x>_+ written so that its derivative at zero is 1/2 rather than 0 or NaN."""
    return 0.5 * (x + jnp.abs(x))


def negative_part(x):
    """<x>_- = x - <x>_+ ."""
    return 0.5 * (x - jnp.abs(x))


def isotropic_function(fn):
    """Lift ``fn(lam, *params) -> scalar`` to symmetric tensors.

    Returns ``F(A, *params)`` with a forward-mode rule that is valid for
    repeated eigenvalues. ``params`` are extra arguments of ``fn`` (for example
    the degradation at the quadrature point); tangents with respect to them are
    propagated as well, so nothing has to be closed over.
    """
    dfn = jax.grad(fn, argnums=0)

    def _eig(A, params):
        lam, Q = jnp.linalg.eigh(A)
        f = jax.vmap(lambda e: fn(e, *params))(lam)
        return lam, Q, f

    @jax.custom_jvp
    def F(A, *params):
        _, Q, f = _eig(A, params)
        return (Q * f[None, :]) @ Q.T

    @F.defjvp
    def F_jvp(primals, tangents):
        A, *params = primals
        V, *dparams = tangents
        lam, Q, f = _eig(A, params)
        df = jax.vmap(lambda e: dfn(e, *params))(lam)

        gap = lam[:, None] - lam[None, :]
        equal = gap == 0.
        theta = jnp.where(equal, df[:, None],
                          (f[:, None] - f[None, :]) / jnp.where(equal, 1., gap))
        V_sym = 0.5 * (V + V.T)
        out = (Q * f[None, :]) @ Q.T
        dout = Q @ (theta * (Q.T @ V_sym @ Q)) @ Q.T

        if params:
            df_params = jax.vmap(
                lambda e: jax.jvp(lambda *p: fn(e, *p), tuple(params), tuple(dparams))[1])(lam)
            dout = dout + (Q * df_params[None, :]) @ Q.T
        return out, dout

    return F


def eigvalsh_3x3(A):
    """Eigenvalues (ascending) of a symmetric 3 x 3 matrix in closed form.

    Trigonometric solution of the characteristic cubic (Smith 1961). It uses
    only elementwise arithmetic, so it vectorizes over any number of
    quadrature points without calling a batched eigensolver.
    """
    a11, a22, a33 = A[0, 0], A[1, 1], A[2, 2]
    a12, a13, a23 = A[0, 1], A[0, 2], A[1, 2]

    off = a12 ** 2 + a13 ** 2 + a23 ** 2
    q = (a11 + a22 + a33) / 3.
    p = jnp.sqrt(((a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2. * off) / 6.)
    scale = 1. / jnp.where(p > 1e-30, p, 1.)

    b11, b22, b33 = (a11 - q) * scale, (a22 - q) * scale, (a33 - q) * scale
    b12, b13, b23 = a12 * scale, a13 * scale, a23 * scale
    half_det = 0.5 * (b11 * (b22 * b33 - b23 * b23)
                      - b12 * (b12 * b33 - b23 * b13)
                      + b13 * (b12 * b23 - b22 * b13))
    phi = jnp.arccos(jnp.clip(half_det, -1., 1.)) / 3.

    largest = q + 2. * p * jnp.cos(phi)
    smallest = q + 2. * p * jnp.cos(phi + 2. * jnp.pi / 3.)
    middle = 3. * q - smallest - largest
    # a diagonal matrix has p == 0 in the limit; its eigenvalues are the diagonal
    return jnp.where(off > 1e-30, jnp.array([smallest, middle, largest]),
                     jnp.sort(jnp.array([a11, a22, a33])))
