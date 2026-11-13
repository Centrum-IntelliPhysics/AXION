"""One field equation of a coupled problem.

A :class:`Field` holds the discretization of one unknown field (displacement,
damage, pressure, ...) and turns a weak form written at a quadrature point
into the element residuals, the element tangents and the global operator
action used by the solvers.

The weak form of a field z with test function v is written as

    R(z; v) = int_Omega  flux(grad z, theta) : grad v  dV
            + int_Omega  source(z, x, theta) . v       dV
            + sum_b int_Gamma_b  boundary_source_b(z, x, theta_b) . v  dA  =  0 ,

and a subclass supplies ``flux``, ``source`` and ``boundary_sources`` as plain
JAX functions of the values at ONE quadrature point. ``theta`` collects the
data that stay fixed while this field is solved: material parameters,
internal variables and the other fields, all sampled at the quadrature
points.

From that single definition the class generates

* the element residual r_e = F_e(z_e; theta_e), vectorized over the elements,
* the element tangent K_e = d r_e / d z_e, whose columns are forward-mode
  Jacobian-vector products of the same kernel along the unit vectors,
* the global residual by scatter-add, and the operator action
  y = sum_e P_e^T K_e P_e x without assembling a sparse matrix.
"""
import warnings
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp

from .elements import reference_element


class DirichletBC(NamedTuple):
    """Prescribed value of one component on a set of nodes.

    where : function(point) -> bool, JAX-traceable, selects the nodes
    component : index of the constrained component
    value : float, or function(point) -> float
    """
    where: Callable
    component: int
    value: object = 0.


class FaceSet(NamedTuple):
    """Geometry of a set of element faces (one row per selected face).

    cell_ids (n,)                 cell that owns the face
    face_ids (n,)                 local face number in that cell
    cell_dofs (n, n_e)            global unknowns of the owning cell
    shape_vals (n, n_fq, n_nodes) shape functions at the face quadrature points
    shape_grads (n, n_fq, n_nodes, dim)  their physical gradients
    dS (n, n_fq)                  surface element times quadrature weight
    normals (n, n_fq, dim)        outward unit normals
    quad_points (n, n_fq, dim)    physical quadrature points
    """
    cell_ids: onp.ndarray
    face_ids: onp.ndarray
    cell_dofs: jnp.ndarray
    shape_vals: jnp.ndarray
    shape_grads: jnp.ndarray
    dS: jnp.ndarray
    normals: jnp.ndarray
    quad_points: jnp.ndarray

    @property
    def num_faces(self):
        return len(self.cell_ids)


def _node_mask(where, points):
    """Evaluate a point predicate on every node."""
    return onp.asarray(jax.vmap(where)(jnp.asarray(points)), dtype=bool)


def _chunk_bounds(n, num_chunks):
    num_chunks = max(1, min(int(num_chunks), n))
    edges = onp.linspace(0, n, num_chunks + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _residual_and_tangent(kernel):
    """Return f(z_e, *args) -> (r_e, K_e) with K_e[i, j] = d r_i / d z_j.

    Every column of K_e is the Jacobian-vector product of the element kernel
    along one unit vector, so the tangent is the derivative of the implemented
    residual to round-off."""
    def fn(z_e, *args):
        f = lambda z: kernel(z, *args)
        basis = jnp.eye(z_e.shape[0], dtype=z_e.dtype)
        r, columns = jax.vmap(lambda q: jax.jvp(f, (z_e,), (q,)), out_axes=(None, -1))(basis)
        return r, columns
    return fn


class Field:
    """Discretization and weak form of one field.

    Parameters
    ----------
    mesh : axion.Mesh
    ncomp : number of components of the field (1 for a scalar)
    dirichlet : list of :class:`DirichletBC`
    boundaries : list of functions(point) -> bool. Each one selects the element
        faces (all nodes must satisfy it) of one boundary integral, in the same
        order as the functions returned by :meth:`boundary_sources`.
    quad_degree : polynomial degree integrated exactly on the cells (default:
        one point for TRI3/TET4, 2 x 2 (x 2) Gauss points for QUAD4/HEX8)
    face_quad_degree : the same for the faces (default: ``quad_degree``)
    num_chunks : the element loop is evaluated in this many slices, which
        bounds the temporary memory of the tangent evaluation
    """

    def __init__(self, mesh, ncomp, dirichlet=(), boundaries=(), quad_degree=None,
                 face_quad_degree=None, num_chunks=20):
        self.mesh = mesh
        self.ncomp = int(ncomp)
        self.dim = mesh.dim
        self.ele_type = mesh.ele_type
        self.num_chunks = int(num_chunks)
        ref = reference_element(mesh.ele_type, quad_degree, face_quad_degree)
        self.ref = ref

        self.num_nodes = mesh.num_nodes
        self.num_cells = mesh.num_cells
        self.num_dofs = self.num_nodes * self.ncomp
        self.nodes_per_cell = ref.num_nodes
        self.dofs_per_cell = ref.num_nodes * self.ncomp
        self.num_quads = ref.num_quads
        self.num_face_quads = ref.num_face_quads

        # ---- cell geometry (host, once) -----------------------------------
        X = mesh.points[mesh.cells]                                    # (C, n_nodes, dim)
        jac = onp.einsum('cad,qak->cqdk', X, ref.shape_grads_ref)      # dx_d / dxi_k
        det = onp.linalg.det(jac)
        if onp.any(det <= 0.):
            share = float(onp.mean(det <= 0.))
            warnings.warn(
                f"{100. * share:.1f}% of the quadrature points have a non-positive Jacobian "
                "determinant: the local node numbering of those cells is inverted. Integrals "
                "over them enter with a negative sign.", stacklevel=2)
        inv = onp.linalg.inv(jac)                                      # dxi_k / dx_d
        shape_grads = onp.einsum('qak,cqkd->cqad', ref.shape_grads_ref, inv)
        quad_points = onp.einsum('qa,cad->cqd', ref.shape_vals, X)
        del X, jac, inv

        node_dofs = (mesh.cells[:, :, None] * self.ncomp
                     + onp.arange(self.ncomp, dtype=onp.int32)[None, None, :])
        cell_dofs = node_dofs.reshape(self.num_cells, self.dofs_per_cell).astype(onp.int32)

        self.points = jnp.asarray(mesh.points)
        self.cells = jnp.asarray(mesh.cells)
        self.cell_dofs = jnp.asarray(cell_dofs)                        # gather map P_e
        self.shape_vals = jnp.asarray(ref.shape_vals)                  # (n_q, n_nodes)
        self.shape_grads = jnp.asarray(shape_grads)                    # (C, n_q, n_nodes, dim)
        self.JxW = jnp.asarray(det * ref.quad_weights[None, :])        # (C, n_q)
        self.quad_points = jnp.asarray(quad_points)                    # (C, n_q, dim)

        # ---- Dirichlet data -----------------------------------------------
        self.dirichlet = [bc if isinstance(bc, DirichletBC) else DirichletBC(*bc)
                          for bc in dirichlet]
        self._set_dirichlet()

        # ---- boundary integrals -------------------------------------------
        self.boundaries = [self.face_set(where) for where in boundaries]

        # ---- weak form -> element kernels ---------------------------------
        self._build_kernels()

    # ------------------------------------------------------------------
    # Weak form: override in a subclass
    # ------------------------------------------------------------------
    def flux(self):
        """Return fn(grad_z, *theta) -> array (ncomp, dim), paired with grad v."""
        return None

    def source(self):
        """Return fn(z, x, *theta) -> array (ncomp,), paired with v."""
        return None

    def boundary_sources(self):
        """Return a list of fn(z, x, *theta_b) -> array (ncomp,), one for every
        entry of ``boundaries``; each is paired with v on that boundary."""
        return []

    def cell_term(self):
        """Return fn(z_e, x_q, N, dN, JxW, *theta_e) -> array (n_nodes, ncomp) for
        contributions that do not fit the flux/source form. ``z_e`` has shape
        (n_nodes, ncomp); the remaining arguments are the data of one cell."""
        return None

    # ------------------------------------------------------------------
    # Dirichlet conditions
    # ------------------------------------------------------------------
    def _set_dirichlet(self):
        points = onp.asarray(self.mesh.points)
        merged = {}
        sets = []
        for bc in self.dirichlet:
            nodes = onp.nonzero(_node_mask(bc.where, points))[0]
            dofs = (nodes * self.ncomp + bc.component).astype(onp.int32)
            if callable(bc.value):
                vals = onp.asarray(jax.vmap(bc.value)(jnp.asarray(points[nodes])),
                                   dtype=onp.float64).reshape(-1)
            else:
                vals = onp.full(len(nodes), float(bc.value))
            sets.append((dofs, vals))
            merged.update(zip(dofs.tolist(), vals.tolist()))      # a later condition wins
        self.dirichlet_sets = sets
        dofs = onp.array(sorted(merged), dtype=onp.int32)
        self.dirichlet_dofs = jnp.asarray(dofs)
        self.dirichlet_values = jnp.asarray(onp.array([merged[i] for i in dofs.tolist()],
                                                      dtype=onp.float64))

    def dirichlet_mask(self, index):
        """Boolean array over ``dirichlet_dofs`` marking the unknowns of the
        ``index``-th Dirichlet condition (useful to scale one of them by a load)."""
        dofs = onp.asarray(self.dirichlet_dofs)
        return jnp.asarray(onp.isin(dofs, self.dirichlet_sets[index][0]))

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------
    def select_faces(self, where):
        """Element faces all of whose nodes satisfy ``where``.
        Returns (cell_ids, local_face_ids)."""
        mask = _node_mask(where, self.mesh.points)
        face_nodes = self.mesh.cells[:, self.ref.face_nodes]           # (C, n_faces, n_face_nodes)
        cell_ids, face_ids = onp.nonzero(mask[face_nodes].all(axis=2))
        return cell_ids.astype(onp.int32), face_ids.astype(onp.int32)

    def face_set(self, where=None, cell_ids=None, face_ids=None):
        """Geometry of the faces selected by ``where`` (or given explicitly)."""
        if where is not None:
            cell_ids, face_ids = self.select_faces(where)
        ref = self.ref
        cell_ids = onp.asarray(cell_ids, dtype=onp.int32)
        face_ids = onp.asarray(face_ids, dtype=onp.int32)
        X = self.mesh.points[self.mesh.cells[cell_ids]]                # (n, n_nodes, dim)
        dN = ref.face_shape_grads_ref[face_ids]                        # (n, n_fq, n_nodes, dim)
        N = ref.face_shape_vals[face_ids]                              # (n, n_fq, n_nodes)
        jac = onp.einsum('sad,sqak->sqdk', X, dN)
        det = onp.linalg.det(jac)
        inv = onp.linalg.inv(jac)
        grads = onp.einsum('sqak,sqkd->sqad', dN, inv)
        # Nanson: n dA = det(J) J^-T n_ref dA_ref
        v = onp.einsum('sk,sqkd->sqd', ref.face_normals_ref[face_ids], inv)
        norm = onp.linalg.norm(v, axis=-1)
        dS = norm * det * ref.face_quad_weights[face_ids]
        normals = v / norm[..., None]
        xq = onp.einsum('sqa,sad->sqd', N, X)
        return FaceSet(cell_ids=cell_ids, face_ids=face_ids,
                       cell_dofs=self.cell_dofs[jnp.asarray(cell_ids)],
                       shape_vals=jnp.asarray(N), shape_grads=jnp.asarray(grads),
                       dS=jnp.asarray(dS), normals=jnp.asarray(normals),
                       quad_points=jnp.asarray(xq))

    # ------------------------------------------------------------------
    # Element kernels
    # ------------------------------------------------------------------
    def _build_kernels(self):
        nn, nc = self.nodes_per_cell, self.ncomp
        N = self.shape_vals
        flux, source, cell_term = self.flux(), self.source(), self.cell_term()

        def element_residual(z_e, x_q, dN, JxW, *theta_e):
            z = z_e.reshape(nn, nc)
            r = jnp.zeros((nn, nc), dtype=z_e.dtype)
            if flux is not None:
                grad = jnp.einsum('ac,qad->qcd', z, dN)
                P = jax.vmap(flux)(grad, *theta_e)
                r = r + jnp.einsum('qcd,qad,q->ac', P, dN, JxW)
            if source is not None:
                s = jax.vmap(source)(N @ z, x_q, *theta_e)
                r = r + jnp.einsum('qc,qa,q->ac', s, N, JxW)
            if cell_term is not None:
                r = r + cell_term(z, x_q, N, dN, JxW, *theta_e)
            return r.reshape(-1)

        def make_face_residual(fn):
            def face_residual(z_e, x_q, N_f, dS, *theta_f):
                z = z_e.reshape(nn, nc)
                t = jax.vmap(fn)(N_f @ z, x_q, *theta_f)
                return jnp.einsum('qc,qa,q->ac', t, N_f, dS).reshape(-1)
            return face_residual

        self.element_residual = element_residual
        self._cell_res = jax.vmap(element_residual)
        self._cell_res_tan = jax.vmap(_residual_and_tangent(element_residual))

        face_fns = list(self.boundary_sources())
        if len(face_fns) != len(self.boundaries):
            raise ValueError(f"{len(self.boundaries)} boundaries were declared but "
                             f"boundary_sources() returns {len(face_fns)} functions")
        self.face_residuals = [make_face_residual(fn) for fn in face_fns]
        self._face_res = [jax.vmap(f) for f in self.face_residuals]
        self._face_res_tan = [jax.vmap(_residual_and_tangent(f)) for f in self.face_residuals]

        self._coo = None

    def _coo_indices(self):
        """Position of every stored tangent entry in the global operator: all
        cell tangents first, then the face tangents of each boundary in turn.
        Built on first use (the tables hold n_e^2 integers per cell)."""
        if self._coo is None:
            blocks = [self.cell_dofs] + [b.cell_dofs for b in self.boundaries]
            n_e = self.dofs_per_cell
            rows = [onp.broadcast_to(onp.asarray(b)[:, :, None],
                                     (b.shape[0], n_e, n_e)).reshape(-1) for b in blocks]
            cols = [onp.broadcast_to(onp.asarray(b)[:, None, :],
                                     (b.shape[0], n_e, n_e)).reshape(-1) for b in blocks]
            self._coo = (onp.concatenate(rows).astype(onp.int32),
                         onp.concatenate(cols).astype(onp.int32))
        return self._coo

    @property
    def coo_rows(self):
        """Global row of every stored tangent entry (see :meth:`tangent_values`)."""
        return self._coo_indices()[0]

    @property
    def coo_cols(self):
        """Global column of every stored tangent entry."""
        return self._coo_indices()[1]

    def release_geometry(self):
        """Drop the full-mesh geometry arrays held by this field.

        Use after the geometry has been copied elsewhere (for example sharded
        over several devices by ``axion.distributed``): the element kernels take
        the geometry as arguments and remain usable, but methods that evaluate
        on the whole mesh (``evaluate_cells``, ``residual``, ...) do not."""
        self.quad_points = None
        self.shape_grads = None
        self.JxW = None
        self._coo = None

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def gather(self, z):
        """Element unknowns z_e = P_e z, shape (num_cells, dofs_per_cell)."""
        return jnp.reshape(z, (-1,))[self.cell_dofs]

    def to_quad(self, z):
        """Values of a nodal array at the quadrature points, (num_cells, n_q, k).

        ``z`` is any nodal array on this mesh, (num_nodes, k) or flat; it need
        not be the unknown of this field (the displacement field interpolates
        the damage to its quadrature points this way)."""
        z = jnp.reshape(z, (self.num_nodes, -1))
        return jnp.einsum('qa,cak->cqk', self.shape_vals, z[self.cells])

    def grad_at_quad(self, z):
        """Gradients of a nodal array at the quadrature points,
        (num_cells, n_q, k, dim)."""
        z = jnp.reshape(z, (self.num_nodes, -1))
        return jnp.einsum('cak,cqad->cqkd', z[self.cells], self.shape_grads)

    # ------------------------------------------------------------------
    # Element evaluation
    # ------------------------------------------------------------------
    def evaluate_cells(self, z, theta=(), tangent=False):
        """Element residuals r_e, shape (num_cells, n_e); with ``tangent=True``
        also the element tangents K_e, shape (num_cells, n_e, n_e)."""
        args = (self.gather(z), self.quad_points, self.shape_grads, self.JxW, *theta)
        fn = self._cell_res_tan if tangent else self._cell_res
        outs = [fn(*[a[lo:hi] for a in args])
                for lo, hi in _chunk_bounds(self.num_cells, self.num_chunks)]
        if tangent:
            return (jnp.concatenate([o[0] for o in outs]), jnp.concatenate([o[1] for o in outs]))
        return jnp.concatenate(outs)

    def evaluate_boundaries(self, z, theta_boundaries=None, tangent=False):
        """Face residuals (and tangents) of every boundary integral, as lists."""
        z = jnp.reshape(z, (-1,))
        if theta_boundaries is None:
            theta_boundaries = [()] * len(self.boundaries)
        res, tan = [], []
        for b, f, f_tan, theta_b in zip(self.boundaries, self._face_res, self._face_res_tan,
                                        theta_boundaries):
            args = (z[b.cell_dofs], b.quad_points, b.shape_vals, b.dS, *theta_b)
            if tangent:
                r, K = f_tan(*args)
                res.append(r)
                tan.append(K)
            else:
                res.append(f(*args))
        return (res, tan) if tangent else res

    # ------------------------------------------------------------------
    # Global residual and operator
    # ------------------------------------------------------------------
    def assemble(self, cell_res, face_res=()):
        """Global residual R = sum_e P_e^T r_e, flat of length num_dofs."""
        R = jnp.zeros(self.num_dofs, dtype=cell_res.dtype)
        R = R.at[self.cell_dofs.reshape(-1)].add(cell_res.reshape(-1))
        for b, r in zip(self.boundaries, face_res):
            R = R.at[b.cell_dofs.reshape(-1)].add(r.reshape(-1))
        return R

    def residual(self, z, theta=(), theta_boundaries=None):
        """Global residual of the weak form at z (flat, length num_dofs)."""
        return self.assemble(self.evaluate_cells(z, theta),
                             self.evaluate_boundaries(z, theta_boundaries))

    def residual_and_tangent(self, z, theta=(), theta_boundaries=None):
        """Global residual and the element tangents ``(K_cells, [K_faces, ...])``."""
        r_c, K_c = self.evaluate_cells(z, theta, tangent=True)
        r_f, K_f = self.evaluate_boundaries(z, theta_boundaries, tangent=True)
        return self.assemble(r_c, r_f), (K_c, K_f)

    def tangent_values(self, tangents):
        """Stored tangent entries as one flat vector aligned with
        ``coo_rows`` / ``coo_cols``."""
        K_c, K_f = tangents
        return jnp.concatenate([K_c.reshape(-1)] + [K.reshape(-1) for K in K_f])

    def apply(self, tangents, x):
        """Operator action y = sum_e P_e^T K_e P_e x (gather, local product,
        scatter-add); ``x`` and the result are flat vectors of length num_dofs."""
        K_c, K_f = tangents
        y = jnp.zeros_like(x)
        y = y.at[self.cell_dofs.reshape(-1)].add(
            jnp.einsum('cij,cj->ci', K_c, x[self.cell_dofs]).reshape(-1))
        for b, K in zip(self.boundaries, K_f):
            y = y.at[b.cell_dofs.reshape(-1)].add(
                jnp.einsum('cij,cj->ci', K, x[b.cell_dofs]).reshape(-1))
        return y

    def diagonal(self, tangents):
        """Diagonal of the global tangent, scatter-added from the element diagonals."""
        K_c, K_f = tangents
        diag = jnp.zeros(self.num_dofs, dtype=K_c.dtype)
        diag = diag.at[self.cell_dofs.reshape(-1)].add(
            jnp.diagonal(K_c, axis1=1, axis2=2).reshape(-1))
        for b, K in zip(self.boundaries, K_f):
            diag = diag.at[b.cell_dofs.reshape(-1)].add(
                jnp.diagonal(K, axis1=1, axis2=2).reshape(-1))
        return diag
