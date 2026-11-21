import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import ufl
from dolfinx import fem, mesh as dmesh
import dolfinx.fem.petsc
import dolfinx_mpc

import models as M

comm = MPI.COMM_WORLD
rank = comm.rank

KSP_U = {"ksp_type": "gmres", "pc_type": "hypre", "pc_hypre_type": "euclid"}
KSP_D = {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "euclid"}
NEWTON_ATOL, NEWTON_RTOL = 1e-5, 1e-7
U_MAXIT, D_MAXIT, D_STOL = 999, 1000, 1e-5
STAGGER_TOL = 1e-6
SOLVER_SUMMARY = ("u: Newton + GMRES/hypre Euclid ILU; d: bounded Newton + CG/Euclid; "
                  "Newton |R|<1e-5 or rel 1e-7; staggered L2 change < 1e-6")

ROUND = 9


def _rk(p):
    return tuple(round(float(c), ROUND) for c in p)


def build_mesh(state, dim=2):
    cell_name = "quadrilateral" if dim == 2 else "hexahedron"
    ce = ufl.Mesh(basix.ufl.element("Lagrange", cell_name, 1, shape=(dim,)))
    if rank == 0:
        cells = state.cells.astype(np.int64)
        pts = state.pts.astype(np.float64)
    else:
        k = 4 if dim == 2 else 8
        cells = np.zeros((0, k), dtype=np.int64)
        pts = np.zeros((0, dim), dtype=np.float64)
    return dmesh.create_mesh(comm, cells, ce, pts)


def _hanging_coord_maps(V, state, hanging):
    hcoords = set()
    for s_node, mnodes in hanging.items():
        hcoords.add(_rk(state.pts[s_node]))
        for mn in mnodes:
            hcoords.add(_rk(state.pts[mn]))

    im = V.dofmap.index_map
    n_owned = im.size_local
    gdim = V.mesh.geometry.dim
    coords = V.tabulate_dof_coordinates()[:, :gdim]

    local, contrib = {}, []
    for b in range(n_owned):
        key = _rk(coords[b])
        if key in hcoords:
            local[key] = b
    if local:
        owned_blocks = np.fromiter(local.values(), dtype=np.int32)
        g = im.local_to_global(owned_blocks)
        contrib = list(zip(local.keys(), (int(x) for x in g)))

    gathered = comm.allgather(contrib)
    glob = {}
    for r, chunk in enumerate(gathered):
        for key, gblk in chunk:
            glob[key] = (gblk, r)
    return glob, local


def build_hanging_mpc(V, state, hanging):
    bs = V.dofmap.index_map_bs
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    if not hanging:
        mpc.add_constraint(V, np.zeros(0, np.int32), np.zeros(0, np.int64),
                           np.zeros(0, np.float64), np.zeros(0, np.int32),
                           np.zeros(1, np.int32))
        mpc.finalize()
        return mpc

    glob, local = _hanging_coord_maps(V, state, hanging)

    slaves, masters, coeffs, owners, offsets = [], [], [], [], [0]
    for s_node, mnodes in hanging.items():
        cs = _rk(state.pts[s_node])
        if cs not in local:
            continue
        ls = local[cs]
        master_g = []
        for mn in mnodes:
            cm = _rk(state.pts[mn])
            if cm not in glob:
                master_g = None
                break
            master_g.append(glob[cm])
        if master_g is None:
            continue
        w = 1.0 / len(master_g)
        for c in range(bs):
            slaves.append(bs * ls + c)
            for gblk, own in master_g:
                masters.append(bs * gblk + c)
                coeffs.append(w)
                owners.append(own)
            offsets.append(len(masters))

    mpc.add_constraint(
        V,
        np.asarray(slaves, dtype=np.int32),
        np.asarray(masters, dtype=np.int64),
        np.asarray(coeffs, dtype=np.float64),
        np.asarray(owners, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    mpc.finalize()
    return mpc


def _gnorm(a_local):
    return np.sqrt(comm.allreduce(float(np.dot(a_local, a_local)), op=MPI.SUM))


def _interp(f_new, f_old):
    Vn = f_new.function_space
    msh_n = Vn.mesh
    im = msh_n.topology.index_map(msh_n.topology.dim)
    cells = np.arange(im.size_local + im.num_ghosts, dtype=np.int32)
    data = fem.create_interpolation_data(Vn, f_old.function_space, cells, padding=1e-10)
    f_new.interpolate_nonmatching(f_old, cells, interpolation_data=data)
    f_new.x.scatter_forward()


class PFFProblem:
    def __init__(self, state, P):
        self.state = state
        self.P = P
        self.dim = 2

        msh = build_mesh(state, self.dim)
        self.msh = msh
        tdim = msh.topology.dim

        self.V = fem.functionspace(msh, ("Lagrange", 1, (self.dim,)))
        self.S = fem.functionspace(msh, ("Lagrange", 1))
        self.DG0 = fem.functionspace(msh, ("DG", 0))
        self.H = fem.Function(self.DG0, name="history")
        self.H.x.array[:] = 0.0

        hanging = state.hanging()
        self.mpcV = build_hanging_mpc(self.V, state, hanging)
        self.mpcS = build_hanging_mpc(self.S, state, hanging)

        self.u = fem.Function(self.V, name="u")
        self.d = fem.Function(self.S, name="d")

        def bottom(x):
            return np.isclose(x[1], 0.0, atol=1e-6)

        def top(x):
            return np.isclose(x[1], P.Ly, atol=1e-6)

        bdofs_b = fem.locate_dofs_geometrical(self.V, bottom)
        bc_b = fem.dirichletbc(np.zeros(self.dim), bdofs_b, self.V)
        self.Vy, _ = self.V.sub(1).collapse()
        tdofs = fem.locate_dofs_geometrical((self.V.sub(1), self.Vy), top)
        self.uDy = fem.Function(self.Vy)
        bc_t = fem.dirichletbc(self.uDy, tdofs, self.V.sub(1))
        self.bcs_u = [bc_b, bc_t]
        self.uDy_zero = fem.Function(self.Vy)
        self.uDy_zero.x.array[:] = 0.0
        self.bcs_du = [bc_b, fem.dirichletbc(self.uDy_zero, tdofs, self.V.sub(1))]

        md = {"quadrature_degree": P.quad_deg}
        dx = ufl.Measure("dx", domain=msh, metadata=md)

        u = self.u
        v, du = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        d = self.d
        q, dd = ufl.TestFunction(self.S), ufl.TrialFunction(self.S)

        eps = M.strain(u)
        psi_p, psi_m = M.psi_split(eps, P.lmbda, P.mu)
        self.psi_p = psi_p

        d_v = ufl.variable(d)
        g = M.degradation(d_v)
        g_d = g + P.k_res
        g_p = ufl.diff(g, d_v)
        w_p = ufl.diff(M.crack_geo(d_v), d_v)
        cw = M.C_W

        self.F_u_ufl = ufl.derivative((g_d * psi_p + psi_m) * dx, u, v)
        self.J_u_ufl = ufl.derivative(self.F_u_ufl, u, du)
        self.negF_u_ufl = -self.F_u_ufl

        self.F_d_ufl = (g_p * self.H * q
                        + P.Gc / (cw * P.l) * w_p * q
                        + 2.0 * P.Gc * P.l / cw * ufl.dot(ufl.grad(d), ufl.grad(q))) * dx
        self.J_d_ufl = ufl.derivative(self.F_d_ufl, d, dd)
        self.negF_d_ufl = -self.F_d_ufl

        eps_v = ufl.variable(eps)
        psi_p_v, psi_m_v = M.psi_split(eps_v, P.lmbda, P.mu)
        sig = ufl.diff(g_d * psi_p_v + psi_m_v, eps_v)
        top_f = dmesh.locate_entities_boundary(msh, tdim - 1, top)
        mt = dmesh.meshtags(msh, tdim - 1, np.sort(top_f),
                            np.full(len(top_f), 2, dtype=np.int32))
        ds = ufl.Measure("ds", domain=msh, subdomain_data=mt, metadata=md)
        self.force_form = fem.form(sig[1, 1] * ds(2))
        self.eps_ufl = eps
        self.sig_ufl = sig
        self._out = None

        self.F_u_form = fem.form(self.F_u_ufl)
        self.F_d_form = fem.form(self.F_d_ufl)
        self.u_prev = fem.Function(self.V)
        self.d_prev = fem.Function(self.S)
        self.err_u_form = fem.form(ufl.inner(u - self.u_prev, u - self.u_prev) * dx)
        self.err_d_form = fem.form((d - self.d_prev) ** 2 * dx)

        self.n_u = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        self.n_d = self.S.dofmap.index_map.size_local
        self.n_dofs_global = (
            self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
            + self.S.dofmap.index_map.size_global * self.S.dofmap.index_map_bs)
        self.n_cells_local = msh.topology.index_map(tdim).size_local
        self.n_cells_global = msh.topology.index_map(tdim).size_global
        self.orig_cell = msh.topology.original_cell_index[:self.n_cells_local]

    def _residual(self, form, mpc, bcs):
        b = dolfinx_mpc.assemble_vector(form, mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if bcs:
            fem.petsc.set_bc(b, bcs)
        r = b.array.copy()
        b.destroy()
        return r

    def solve_u(self):
        fem.set_bc(self.u.x.array, self.bcs_u)
        self.mpcV.backsubstitution(self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        r0 = None
        for _ in range(U_MAXIT):
            r = _gnorm(self._residual(self.F_u_form, self.mpcV, self.bcs_du))
            if r0 is None:
                r0 = r if r > 0.0 else 1.0
            if r < NEWTON_ATOL or r / r0 < NEWTON_RTOL:
                return True
            prob = dolfinx_mpc.LinearProblem(
                self.J_u_ufl, self.negF_u_ufl, self.mpcV, bcs=self.bcs_du,
                petsc_options_prefix="pff_u_", petsc_options=KSP_U)
            step = prob.solve()
            self.u.x.array[:] += step.x.array
            self.mpcV.backsubstitution(self.u.x.petsc_vec)
            self.u.x.scatter_forward()
        return False

    def update_history(self):
        trial = fem.Function(self.DG0)
        points = self.DG0.element.interpolation_points
        if callable(points):
            points = points()
        trial.interpolate(fem.Expression(self.psi_p, points))
        np.maximum(self.H.x.array, trial.x.array, out=self.H.x.array)
        self.H.x.scatter_forward()

    def solve_d(self):
        r0 = None
        for _ in range(D_MAXIT):
            R = self._residual(self.F_d_form, self.mpcS, [])[:self.n_d]
            dd = self.d.x.array[:self.n_d]
            active = ((dd <= 1e-14) & (R > 0)) | ((dd >= 1.0 - 1e-14) & (R < 0))
            r = _gnorm(np.where(active, 0.0, R))
            if r0 is None:
                r0 = r if r > 0.0 else 1.0
            if r < NEWTON_ATOL or r / r0 < NEWTON_RTOL:
                return True
            d0 = self.d.x.array.copy()
            prob = dolfinx_mpc.LinearProblem(
                self.J_d_ufl, self.negF_d_ufl, self.mpcS, bcs=[],
                petsc_options_prefix="pff_d_", petsc_options=KSP_D)
            step = prob.solve()
            self.d.x.array[:] += step.x.array
            np.clip(self.d.x.array, 0.0, 1.0, out=self.d.x.array)
            self.mpcS.backsubstitution(self.d.x.petsc_vec)
            self.d.x.scatter_forward()
            dx_n = _gnorm(self.d.x.array[:self.n_d] - d0[:self.n_d])
            if dx_n < D_STOL * _gnorm(self.d.x.array[:self.n_d]):
                return True
        return False

    def _l2_err(self, form):
        return np.sqrt(comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))

    def staggered_step(self, disp):
        P = self.P
        self.uDy.x.array[:] = disp
        err, s = np.inf, 0
        while s < P.max_stagger and err > P.ms_tol:
            self.u_prev.x.array[:] = self.u.x.array
            self.d_prev.x.array[:] = self.d.x.array
            if not self.solve_u():
                return s + 1, np.inf, False
            self.update_history()
            if not self.solve_d():
                return s + 1, np.inf, False
            err = max(self._l2_err(self.err_u_form), self._l2_err(self.err_d_form))
            s += 1
        return s, err, bool(err <= P.ms_tol)

    def output_fields(self):
        if self._out is None:
            pts = self.DG0.element.interpolation_points
            if callable(pts):
                pts = pts()
            comps = [(0, 0, "xx"), (1, 1, "yy"), (0, 1, "xy")]
            exprs = [("psi_plus", self.psi_p)]
            exprs += [(f"stress_{n}", self.sig_ufl[i, j]) for i, j, n in comps]
            exprs += [(f"strain_{n}", self.eps_ufl[i, j]) for i, j, n in comps]
            self._out = []
            for name, e in exprs:
                f = fem.Function(self.DG0, name=name)
                self._out.append((f, fem.Expression(e, pts)))
        for f, e in self._out:
            f.interpolate(e)
            f.x.scatter_forward()
        return [self.u, self.d, self.H] + [f for f, _ in self._out]

    def reaction_force(self):
        loc = fem.assemble_scalar(self.force_form)
        return comm.allreduce(float(loc), op=MPI.SUM)

    def max_d(self):
        loc = self.d.x.array[:self.n_d].max() if self.n_d else -np.inf
        return comm.allreduce(float(loc), op=MPI.MAX)

    def _to_global_cells(self, loc, op):
        g = np.zeros(self.state.n_cells(), dtype=np.float64)
        g[self.orig_cell] = loc
        out = np.zeros_like(g)
        comm.Allreduce(g, out, op=op)
        return out

    def global_cell_damage(self):
        d0 = fem.Function(self.DG0)
        d0.interpolate(self.d)
        return self._to_global_cells(d0.x.array[:self.n_cells_local], MPI.SUM)

    def global_cell_tensile_energy(self):
        e0 = fem.Function(self.DG0)
        points = self.DG0.element.interpolation_points
        if callable(points):
            points = points()
        e0.interpolate(fem.Expression(self.psi_p, points))
        return self._to_global_cells(e0.x.array[:self.n_cells_local], MPI.SUM)

    def global_cell_max_damage(self):
        loc = np.empty(self.n_cells_local, dtype=np.float64)
        for c in range(self.n_cells_local):
            loc[c] = np.max(self.d.x.array[self.S.dofmap.cell_dofs(c)])
        return self._to_global_cells(loc, MPI.MAX)

    def transfer_from(self, old):
        _interp(self.u, old.u)
        _interp(self.d, old.d)
        _interp(self.H, old.H)
        np.clip(self.d.x.array, 0.0, 1.0, out=self.d.x.array)
        self.mpcS.backsubstitution(self.d.x.petsc_vec)
        self.d.x.scatter_forward()
