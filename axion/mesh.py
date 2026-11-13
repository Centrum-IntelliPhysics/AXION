"""Mesh container and structured mesh generators.

A mesh is a pair (points, cells) with one element type. Unstructured meshes
are created elsewhere (for example with Gmsh) and wrapped in :class:`Mesh`;
the generators below cover the box-shaped benchmark domains.
"""
import numpy as onp

from .elements import ELEMENT_TYPES, MESHIO_CELL_TYPE, _FACES, _NODES, shape_functions


class Mesh:
    """Nodes and connectivity of a single-element-type mesh.

    Parameters
    ----------
    points : (num_nodes, dim) float array
    cells : (num_cells, nodes_per_cell) int array, VTK local numbering
    ele_type : 'TRI3', 'QUAD4', 'TET4' or 'HEX8'
    """

    def __init__(self, points, cells, ele_type):
        if ele_type not in ELEMENT_TYPES:
            raise ValueError(f"unknown element type {ele_type!r}; choose from {ELEMENT_TYPES}")
        self.points = onp.ascontiguousarray(points, dtype=onp.float64)
        self.cells = onp.ascontiguousarray(cells, dtype=onp.int32)
        self.ele_type = ele_type
        n_nodes_ref, dim_ref = _NODES[ele_type].shape
        if self.points.ndim != 2 or self.points.shape[1] != dim_ref:
            raise ValueError(f"{ele_type} needs points of shape (num_nodes, {dim_ref}), "
                             f"got {self.points.shape}")
        if self.cells.ndim != 2 or self.cells.shape[1] != n_nodes_ref:
            raise ValueError(f"{ele_type} needs cells of shape (num_cells, {n_nodes_ref}), "
                             f"got {self.cells.shape}")

    @property
    def dim(self):
        return self.points.shape[1]

    @property
    def num_nodes(self):
        return self.points.shape[0]

    @property
    def num_cells(self):
        return self.cells.shape[0]

    @property
    def meshio_cell_type(self):
        return MESHIO_CELL_TYPE[self.ele_type]

    def cell_centroids(self):
        return self.points[self.cells].mean(axis=1)

    def signed_cell_measures(self):
        """Jacobian determinant at the cell centre times the reference measure.
        Negative values mean that the local node numbering is inverted."""
        centre = _NODES[self.ele_type].mean(axis=0, keepdims=True)
        _, dN = shape_functions(self.ele_type, centre)
        jac = onp.einsum('cad,ak->cdk', self.points[self.cells], dN[0])
        ref_measure = {'TRI3': 0.5, 'QUAD4': 4., 'TET4': 1. / 6., 'HEX8': 8.}[self.ele_type]
        return onp.linalg.det(jac) * ref_measure

    def boundary_faces(self):
        """Faces that belong to exactly one cell.

        Returns (cell_ids, local_face_ids), each of shape (num_boundary_faces,).
        """
        faces = _FACES[self.ele_type]
        nodes = self.cells[:, faces]                              # (num_cells, n_faces, n_face_nodes)
        keys = onp.sort(nodes.reshape(-1, faces.shape[1]), axis=1)
        _, inverse, counts = onp.unique(keys, axis=0, return_inverse=True, return_counts=True)
        exterior = (counts[inverse.reshape(-1)] == 1).reshape(nodes.shape[:2])
        cell_ids, face_ids = onp.nonzero(exterior)
        return cell_ids.astype(onp.int32), face_ids.astype(onp.int32)


def rectangle_mesh(nx, ny, Lx=1., Ly=1., ele_type='QUAD4', origin=(0., 0.), xs=None, ys=None):
    """Structured mesh of the rectangle [0, Lx] x [0, Ly] (shifted by ``origin``).

    Nodes are numbered row by row (x fastest). ``xs`` / ``ys`` may be given to
    use a graded set of grid lines instead of the uniform one. With
    ``ele_type='TRI3'`` every quadrilateral is split along alternating
    diagonals, which avoids a directional bias of the triangulation.
    """
    xs = onp.linspace(0., Lx, nx + 1) if xs is None else onp.asarray(xs, dtype=onp.float64)
    ys = onp.linspace(0., Ly, ny + 1) if ys is None else onp.asarray(ys, dtype=onp.float64)
    nx, ny = len(xs) - 1, len(ys) - 1
    X, Y = onp.meshgrid(xs + origin[0], ys + origin[1], indexing='xy')
    points = onp.stack([X.ravel(), Y.ravel()], axis=1)

    i, j = onp.meshgrid(onp.arange(nx), onp.arange(ny), indexing='xy')
    i, j = i.ravel(), j.ravel()
    n00 = j * (nx + 1) + i
    n10, n01, n11 = n00 + 1, n00 + nx + 1, n00 + nx + 2
    if ele_type == 'QUAD4':
        cells = onp.stack([n00, n10, n11, n01], axis=1)
    elif ele_type == 'TRI3':
        even = ((i + j) % 2 == 0)[:, None]
        lower = onp.where(even, onp.stack([n00, n10, n11], axis=1), onp.stack([n00, n10, n01], axis=1))
        upper = onp.where(even, onp.stack([n00, n11, n01], axis=1), onp.stack([n10, n11, n01], axis=1))
        cells = onp.stack([lower, upper], axis=1).reshape(-1, 3)
    else:
        raise ValueError(f"rectangle_mesh supports QUAD4 and TRI3, got {ele_type!r}")
    return Mesh(points, cells, ele_type)


def box_mesh(nx, ny, nz, Lx=1., Ly=1., Lz=1.):
    """Structured HEX8 mesh of the box [0, Lx] x [0, Ly] x [0, Lz].

    Nodes are numbered with x fastest, then y, then z.
    """
    xs, ys, zs = (onp.linspace(0., L, n + 1) for L, n in ((Lx, nx), (Ly, ny), (Lz, nz)))
    Z, Y, X = onp.meshgrid(zs, ys, xs, indexing='ij')
    points = onp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    k, j, i = onp.meshgrid(onp.arange(nz), onp.arange(ny), onp.arange(nx), indexing='ij')
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    sx, sy = 1, nx + 1
    sz = (nx + 1) * (ny + 1)
    n0 = k * sz + j * sy + i
    cells = onp.stack([n0, n0 + sx, n0 + sx + sy, n0 + sy,
                       n0 + sz, n0 + sz + sx, n0 + sz + sx + sy, n0 + sz + sy], axis=1)
    return Mesh(points, cells, 'HEX8')
