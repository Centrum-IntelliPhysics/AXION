"""Reference elements and quadrature rules.

Linear Lagrange elements are provided: TRI3 and QUAD4 in two dimensions, TET4
and HEX8 in three. Local node numbering follows the VTK convention, so meshes
read with meshio or written to VTU need no reordering:

    TRI3   (0,0) (1,0) (0,1)                      on the unit triangle
    QUAD4  (-1,-1) (1,-1) (1,1) (-1,1)            on the biunit square
    TET4   (0,0,0) (1,0,0) (0,1,0) (0,0,1)        on the unit tetrahedron
    HEX8   bottom face as QUAD4 at zeta = -1, then the top face at zeta = +1

Everything here is plain NumPy and is evaluated once, before compilation.
"""
from dataclasses import dataclass
from math import ceil

import numpy as onp

ELEMENT_TYPES = ('TRI3', 'QUAD4', 'TET4', 'HEX8')

# Polynomial degree integrated exactly when the user does not ask for one.
DEFAULT_QUAD_DEGREE = {'TRI3': 1, 'QUAD4': 2, 'TET4': 1, 'HEX8': 2}

# meshio / VTK cell names
MESHIO_CELL_TYPE = {'TRI3': 'triangle', 'QUAD4': 'quad',
                    'TET4': 'tetra', 'HEX8': 'hexahedron'}

_NODES = {
    'TRI3': onp.array([[0., 0.], [1., 0.], [0., 1.]]),
    'QUAD4': onp.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
    'TET4': onp.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    'HEX8': onp.array([[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.],
                       [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]]),
}

# Local node numbers of every face (edge in 2D).
_FACES = {
    'TRI3': onp.array([[0, 1], [1, 2], [2, 0]]),
    'QUAD4': onp.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
    'TET4': onp.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]),
    'HEX8': onp.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]),
}


# ----------------------------------------------------------------------------
# Shape functions
# ----------------------------------------------------------------------------
def shape_functions(ele_type, xi):
    """Shape functions and their reference gradients at the points ``xi``.

    Parameters
    ----------
    ele_type : str
    xi : array (n_points, dim), reference coordinates

    Returns
    -------
    N : (n_points, n_nodes)
    dN : (n_points, n_nodes, dim), derivatives with respect to xi
    """
    xi = onp.atleast_2d(onp.asarray(xi, dtype=onp.float64))
    n_pts = xi.shape[0]
    if ele_type in ('TRI3', 'TET4'):
        dim = xi.shape[1]
        N = onp.concatenate([1. - xi.sum(axis=1, keepdims=True), xi], axis=1)
        dN_one = onp.concatenate([-onp.ones((1, dim)), onp.eye(dim)], axis=0)
        dN = onp.broadcast_to(dN_one, (n_pts, dim + 1, dim)).copy()
        return N, dN
    if ele_type in ('QUAD4', 'HEX8'):
        corners = _NODES[ele_type]                       # (n_nodes, dim), entries +-1
        n_nodes, dim = corners.shape
        # one factor (1 + xi_a xi) / 2 per direction and node
        factors = 0.5 * (1. + corners[None, :, :] * xi[:, None, :])   # (n_pts, n_nodes, dim)
        N = factors.prod(axis=2)
        dN = onp.empty((n_pts, n_nodes, dim))
        for k in range(dim):
            others = [j for j in range(dim) if j != k]
            dN[:, :, k] = 0.5 * corners[None, :, k] * factors[:, :, others].prod(axis=2)
        return N, dN
    raise ValueError(f"unknown element type {ele_type!r}; choose from {ELEMENT_TYPES}")


# ----------------------------------------------------------------------------
# Quadrature
# ----------------------------------------------------------------------------
def _gauss_points_per_direction(degree):
    return max(1, ceil((degree + 1) / 2))


def gauss_line(degree):
    """Gauss-Legendre rule on [-1, 1], exact for polynomials of ``degree``."""
    return onp.polynomial.legendre.leggauss(_gauss_points_per_direction(degree))


def gauss_tensor(dim, degree):
    """Tensor-product Gauss-Legendre rule on [-1, 1]^dim."""
    x, w = gauss_line(degree)
    grids = onp.meshgrid(*([x] * dim), indexing='ij')
    wgrids = onp.meshgrid(*([w] * dim), indexing='ij')
    points = onp.stack([g.ravel() for g in grids], axis=1)
    weights = onp.prod(onp.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    return points, weights


def simplex_rule(dim, degree):
    """Symmetric rule on the unit triangle (dim = 2) or tetrahedron (dim = 3)."""
    if dim == 1:
        x, w = gauss_line(degree)                 # mapped from [-1, 1] to [0, 1]
        return 0.5 * (x[:, None] + 1.), 0.5 * w
    if dim == 2:
        if degree <= 1:
            return onp.array([[1. / 3., 1. / 3.]]), onp.array([0.5])
        if degree == 2:
            a, b = 1. / 6., 2. / 3.
            return onp.array([[a, a], [a, b], [b, a]]), onp.full(3, 1. / 6.)
    if dim == 3:
        if degree <= 1:
            return onp.array([[0.25, 0.25, 0.25]]), onp.array([1. / 6.])
        if degree == 2:
            a = (5. + 3. * onp.sqrt(5.)) / 20.
            b = (5. - onp.sqrt(5.)) / 20.
            pts = onp.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])
            return pts, onp.full(4, 1. / 24.)
    raise NotImplementedError(
        f"simplex quadrature of degree {degree} in {dim}D is not implemented (use degree <= 2)")


def cell_quadrature(ele_type, degree):
    if ele_type == 'TRI3':
        return simplex_rule(2, degree)
    if ele_type == 'TET4':
        return simplex_rule(3, degree)
    if ele_type == 'QUAD4':
        return gauss_tensor(2, degree)
    if ele_type == 'HEX8':
        return gauss_tensor(3, degree)
    raise ValueError(f"unknown element type {ele_type!r}")


def _face_parametrization(ele_type, degree):
    """Quadrature points of the reference face and the face shape functions
    that map them into the element (M has shape (n_face_points, n_face_nodes))."""
    if ele_type in ('TRI3', 'QUAD4'):            # faces are 2-node edges, s in [0, 1]
        s, w = simplex_rule(1, degree)
        M = onp.concatenate([1. - s, s], axis=1)
        dM = onp.broadcast_to(onp.array([[-1.], [1.]]), (len(w), 2, 1))
        return M, dM, w
    if ele_type == 'TET4':                        # triangular faces
        s, w = simplex_rule(2, degree)
        M, dM = shape_functions('TRI3', s)
        return M, dM, w
    if ele_type == 'HEX8':                        # quadrilateral faces
        s, w = gauss_tensor(2, degree)
        M, dM = shape_functions('QUAD4', s)
        return M, dM, w
    raise ValueError(f"unknown element type {ele_type!r}")


# ----------------------------------------------------------------------------
# Reference element
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceElement:
    """Tabulated data of one element type for a fixed quadrature degree.

    Cell quadrature (n_q points):
        quad_points (n_q, dim), quad_weights (n_q,),
        shape_vals (n_q, n_nodes), shape_grads_ref (n_q, n_nodes, dim)

    Face quadrature (n_f faces with n_fq points each); the points are given in
    the reference coordinates of the element and the weights already contain
    the measure of the reference face:
        face_nodes (n_f, n_face_nodes), face_quad_points (n_f, n_fq, dim),
        face_quad_weights (n_f, n_fq), face_shape_vals (n_f, n_fq, n_nodes),
        face_shape_grads_ref (n_f, n_fq, n_nodes, dim), face_normals_ref (n_f, dim)
    """
    ele_type: str
    dim: int
    num_nodes: int
    node_coords: onp.ndarray
    quad_degree: int
    quad_points: onp.ndarray
    quad_weights: onp.ndarray
    shape_vals: onp.ndarray
    shape_grads_ref: onp.ndarray
    face_quad_degree: int
    face_nodes: onp.ndarray
    face_quad_points: onp.ndarray
    face_quad_weights: onp.ndarray
    face_shape_vals: onp.ndarray
    face_shape_grads_ref: onp.ndarray
    face_normals_ref: onp.ndarray

    @property
    def num_quads(self):
        return len(self.quad_weights)

    @property
    def num_faces(self):
        return len(self.face_nodes)

    @property
    def num_face_quads(self):
        return self.face_quad_weights.shape[1]


def reference_element(ele_type, quad_degree=None, face_quad_degree=None):
    """Build the :class:`ReferenceElement` of ``ele_type``.

    ``quad_degree`` is the polynomial degree integrated exactly over the cell
    (default: 1 for simplices, i.e. one point; 2 for QUAD4/HEX8, i.e. two Gauss
    points per direction). ``face_quad_degree`` defaults to ``quad_degree``.
    """
    if ele_type not in ELEMENT_TYPES:
        raise ValueError(f"unknown element type {ele_type!r}; choose from {ELEMENT_TYPES}")
    if quad_degree is None:
        quad_degree = DEFAULT_QUAD_DEGREE[ele_type]
    if face_quad_degree is None:
        face_quad_degree = quad_degree

    nodes = _NODES[ele_type]
    num_nodes, dim = nodes.shape
    qp, qw = cell_quadrature(ele_type, quad_degree)
    N, dN = shape_functions(ele_type, qp)

    faces = _FACES[ele_type]
    M, dM, sw = _face_parametrization(ele_type, face_quad_degree)
    centroid = nodes.mean(axis=0)
    f_points, f_weights, f_N, f_dN, f_normals = [], [], [], [], []
    for face in faces:
        xf = nodes[face]                                  # (n_face_nodes, dim)
        pts = M @ xf                                      # face points in element coordinates
        tangents = onp.einsum('qas,ad->qsd', dM, xf)      # (n_fq, dim - 1, dim)
        if dim == 2:
            t = tangents[:, 0, :]
            normal = onp.stack([t[:, 1], -t[:, 0]], axis=1)
        else:
            normal = onp.cross(tangents[:, 0, :], tangents[:, 1, :])
        measure = onp.linalg.norm(normal, axis=1)         # reference surface element
        n_ref = normal[0] / measure[0]                    # faces of these elements are flat
        if n_ref @ (xf.mean(axis=0) - centroid) < 0.:
            n_ref = -n_ref
        Nf, dNf = shape_functions(ele_type, pts)
        f_points.append(pts)
        f_weights.append(sw * measure)
        f_N.append(Nf)
        f_dN.append(dNf)
        f_normals.append(n_ref)

    return ReferenceElement(
        ele_type=ele_type, dim=dim, num_nodes=num_nodes, node_coords=nodes,
        quad_degree=quad_degree, quad_points=qp, quad_weights=qw,
        shape_vals=N, shape_grads_ref=dN,
        face_quad_degree=face_quad_degree, face_nodes=faces,
        face_quad_points=onp.stack(f_points), face_quad_weights=onp.stack(f_weights),
        face_shape_vals=onp.stack(f_N), face_shape_grads_ref=onp.stack(f_dN),
        face_normals_ref=onp.stack(f_normals))
