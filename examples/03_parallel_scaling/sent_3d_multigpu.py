"""Three-dimensional single-edge-notched tension on several GPUs
(paper Sections 2.7 and 4.1.3).

Same physics as ``02_sent_3d/sent_3d.py`` (hybrid AT2 / AT1 / PF-CZM model on a
HEX8 slab with a geometric pre-crack), but the mesh is split across all devices
by the cell decomposition of ``axion.distributed``:

  * every device evaluates only its own ~1/D cells (residual, tangent), so the
    memory and the element work per device fall as 1/D;
  * the CG iteration runs on the "owned + one ghost layer" layout; each operator
    action is preceded by a halo exchange and the inner products are global
    reductions, so the convergence history coincides with the single-device one.

The whole computation is still one compiled program: the load loop is a scan,
the staggered iteration a while_loop, and every displacement / damage solve a
shard_map (element evaluation + halo-PCG) inside that control flow.

Per-device data (sharded over the device axis): geometry (shape gradients,
JxW, ...), index tables, the fields in owned layout, the history field and the
Dirichlet masks. Global fields are reconstructed once, at the end, for output.

Environment variables
    PFF_MODEL    AT2 (default) | AT1 | CZM
    PFF_CRACK    slit (default) | none
    PFF_NX, PFF_NY, PFF_NZ   cells per direction (default 150 x 150 x 15; NX, NY even)
    PFF_NSTEPS   number of load increments (default 160)
    PFF_DU       displacement increment in mm (default 5e-5)
    PFF_WARMUP   1 (default): compile on a zero-load schedule before the timed run
    PFF_PARTITION    metis (default, needs pymetis) | sfc | slab
    PFF_CELL_BATCH   cells per element-evaluation batch on each device (default 200000)
    PFF_TAG      label of the output folder (default n<processes>_gpu<devices>)
    PFF_OUTPUT_DIR   parent folder of the outputs (default: this folder)
    PFF_VERIFY   if set, run a small distributed-vs-dense comparison instead

Run on the visible GPUs of one node:
    PFF_MODEL=AT2 python sent_3d_multigpu.py
Self-check on any machine (4 emulated CPU devices):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    PFF_VERIFY=1 PFF_NSTEPS=2 python sent_3d_multigpu.py
"""
import os
import sys
import time
import gc
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Initialize the multi-process runtime BEFORE any JAX backend initialization
# (a no-op in a single process; SLURM-aware on several nodes). Importing axion
# does not touch the backend.
from axion import distributed as cp
cp.init_distributed()

import numpy as onp
import jax
import jax.numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

import axion
from axion import Field, Mesh, eigvalsh_3x3, positive_part


# =============================================================================
# 1. Material and model parameters (same as 02_sent_3d/sent_3d.py)
# =============================================================================
E     = 210e3        # Young's modulus            [MPa]
nu    = 0.3          # Poisson's ratio            [-]
G_c   = 2.7          # critical energy release    [N/mm]
l     = 0.0313         # phase-field length scale   [mm]   (needs h <~ l/2 to converge)
f_t   = 2445.42      # tensile strength           [MPa]  (only used by PF-CZM)
K_RES = 1e-8         # residual stiffness floor   [-]    (avoids exact 0 at d=1)

mu    = E / (2.0 * (1.0 + nu))                     # shear modulus
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))   # Lame's first parameter
l_ch  = E * G_c / (f_t ** 2)                        # Irwin characteristic length (CZM)

MODEL = os.environ.get("PFF_MODEL", "AT2")
assert MODEL in ("AT1", "AT2", "CZM")

# Partition of the cell dual graph: metis (needs pymetis) | sfc (Morton curve) | slab
PARTITION = os.environ.get("PFF_PARTITION", "metis")
if PARTITION == "metis":
    try:
        import pymetis  # noqa: F401
    except ImportError as exc:
        raise ImportError("PFF_PARTITION=metis needs pymetis (pip install pymetis); "
                          "or set PFF_PARTITION=sfc") from exc

ROOT = (jax.process_index() == 0)                  # only rank 0 prints / writes files


def _memory_report(tag):
    """Print live/peak allocator bytes (rank 0 only) and RETURN a
    {device: stats} dict so callers can save the byte counts.

    Returns an empty-stats dict on backends without memory_stats() (e.g. CPU),
    so callers can safely ``.get(..., 0)`` the values.
    """
    collected = {}
    mib = 1024.0 ** 2
    do_print = ROOT and os.environ.get("PFF_MEM_DIAG", "1") != "0"
    for dev in jax.local_devices():
        stats = dev.memory_stats() or {}
        collected[str(dev)] = stats
        if not do_print:
            continue
        values = []
        for key in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "bytes_limit"):
            if key in stats:
                values.append(f"{key}={stats[key] / mib:.1f} MiB")
        print(f"[gpu-mem] {tag} {dev}: " + (", ".join(values) or "unavailable"),
              flush=True)
    return collected


# --- crack geometric function w(d), degradation g(d), and normalisation c_w.
#     These select the model (AT1 / AT2 / PF-CZM). ---
def crack_geo(d):
    """w(d): AT1 -> d ,  AT2 -> d^2 ,  PF-CZM -> 2d - d^2 ."""
    if MODEL == "AT1":
        return d
    if MODEL == "AT2":
        return d ** 2
    return 2.0 * d - d ** 2                         # CZM


def degradation(d):
    """g(d): stiffness degradation.  AT1/AT2 -> (1-d)^2 ;  PF-CZM -> Wu's rational g."""
    w = (1.0 - d) ** 2
    if MODEL in ("AT1", "AT2"):
        return w
    a1 = 4.0 / np.pi * l_ch / l                     # PF-CZM (Wu), a2 = -0.5, a3 = 0
    Q = a1 * (d - 0.5 * d ** 2)
    return w / (w + Q)


def c_w():
    """Normalising constant so the fracture energy integrates to G_c."""
    return {"AT1": 8.0 / 3.0, "AT2": 2.0, "CZM": np.pi}[MODEL]


# jax.grad gives w'(d) and g'(d) exactly (used inside the PhaseField mass term).
_w_prime = jax.grad(crack_geo)
_g_prime = jax.grad(degradation)


# =============================================================================
# 2. Fields: the same weak forms as the single-device example
# =============================================================================
def stress(grad_u, d):
    """Hybrid model: sigma = (g(d) + K_RES) (lmbda tr(eps) I + 2 mu eps)."""
    eps = 0.5 * (grad_u + grad_u.T)
    g = degradation(d[0]) + K_RES
    return g * (lmbda * np.trace(eps) * np.eye(3) + 2.0 * mu * eps)


def psi_plus(grad_u):
    """Tensile part of the undamaged strain energy density -> crack driving force."""
    eps = 0.5 * (grad_u + grad_u.T)
    eigvals = eigvalsh_3x3(eps)                     # 3 principal strains, closed form
    tr_plus = positive_part(np.trace(eps))
    return lmbda / 2.0 * tr_plus ** 2 + mu * np.sum(positive_part(eigvals) ** 2)


class Displacement(Field):
    """Displacement subproblem; the spectral split enters only through the
    crack-driving history H."""

    def flux(self):
        return stress

    def compute_history(self, sol_u, history_old):
        """H = max(psi_plus(grad u), H_old) on the whole mesh. Used only by the
        dense self-check; the sharded driver updates the history per owned cell
        inside shard_map."""
        psi = jax.vmap(jax.vmap(psi_plus))(self.grad_at_quad(sol_u))
        return np.maximum(psi, history_old)


class Damage(Field):
    """Damage subproblem of the chosen MODEL:
        g'(d) H + Gc/c_w ( w'(d)/l - 2 l lap d ) = 0 ."""

    def flux(self):
        coeff = 2.0 * G_c * l / c_w()
        return lambda grad_d, H: coeff * grad_d

    def source(self):
        coeff = G_c / (c_w() * l)

        def fn(d, x, H):
            dd = d[0]
            return np.array([coeff * _w_prime(dd) + _g_prime(dd) * H])
        return fn


# =============================================================================
# 3. Geometry: slab, mode-I boundary conditions, geometric pre-crack
# =============================================================================
Lx, Ly, Lz = 1.0, 1.0, 0.2                          # thin slab -> quasi-2D mode I


def dirichlet_bottom_top(points):
    """Mode-I Dirichlet: bottom face (y=0) fully clamped (all 3 comps), top face
    (y=1) pulled in +y.  Returns (fixed_mask, is_top_ydof, top_nodes).
      fixed_mask   : (Nd,) bool  -- constrained global dofs
      is_top_ydof  : (Nd,) float -- 1.0 on the top y-dofs (times disp = the load)
      top_nodes    : node ids on the top face (for the reaction force)"""
    Nd = points.shape[0] * 3
    y = points[:, 1]
    bottom = onp.where(onp.isclose(y, 0.0))[0]
    top = onp.where(onp.isclose(y, Ly))[0]
    fixed = onp.zeros(Nd, bool)
    for comp in range(3):                            # clamp all 3 comps at the bottom
        fixed[bottom * 3 + comp] = True
    fixed[top * 3 + 1] = True                        # prescribe u_y on the top
    is_top = onp.zeros(Nd)
    is_top[top * 3 + 1] = 1.0
    return fixed, is_top, top


def insert_crack_3d(points, cells, y_crack, x_tip, tol=1e-9):
    """Open a GEOMETRIC slit on {y = y_crack, x < x_tip, all z} by DUPLICATING the
    crack-plane nodes (tip excluded) and remapping the elements ABOVE the plane to
    the copies -> a true zero-width displacement discontinuity (mesh-independent,
    no phi=1 Dirichlet).

    Returns points, cells (with duplicates appended) and n_dup.  Note: the original
    and its duplicate are two INDEPENDENT nodes; the cell partition gives each its
    own owner naturally -- they must NOT be pinned to the same owner."""
    points = onp.asarray(points, dtype=float)
    cells = onp.asarray(cells).copy()
    on_plane = (onp.abs(points[:, 1] - y_crack) < 1e-8) & (points[:, 0] < x_tip - tol)
    old_ids = onp.where(on_plane)[0]
    if len(old_ids) == 0:
        raise ValueError("no nodes on the crack plane; need EVEN Ny so a row lands on y=0.5")
    new_ids = onp.arange(len(points), len(points) + len(old_ids))
    remap = dict(zip(old_ids.tolist(), new_ids.tolist()))
    points = onp.vstack([points, points[old_ids]])
    cent_y = points[cells][:, :, 1].mean(axis=1)     # element centroids
    for c in onp.where(cent_y > y_crack)[0]:          # elements above the crack plane
        for j in range(cells.shape[1]):
            if cells[c, j] in remap:
                cells[c, j] = remap[cells[c, j]]
    return points, cells, len(new_ids)


# =============================================================================
# 4. The cell-partition solver driver (this is where partition happens)
# =============================================================================
class Driver:
    """Builds the cell partition ONCE (host), packs the static per-shard data, and
    exposes a fully jitted `run(disps)` that scans over load steps.  Each load step
    runs a staggered (alternating u <-> damage) while_loop; both solves are the
    distributed cell-partition solvers of axion.distributed."""

    def __init__(self, points, cells, field_u, field_d, method="metis"):
        self.DMESH = cp.make_domain_mesh()            # 1-D device mesh over ALL GPUs
        D = self.DMESH.shape["d"]
        self.D = D
        self.N = field_u.num_nodes
        self.nq = field_u.num_quads
        N, NdU, NdD = self.N, self.N * 3, self.N     # #dofs for u (vec=3) and d (vec=1)

        # ---- 4a. Partition the cell dual graph into D parts (topology-agnostic) ----
        # cell_part[c] = which shard owns cell c.  part_u/part_d hold, PER SHARD:
        # owned cells, local node numbering [owned | ghost], local COO (LR/GC),
        # owner->ghost / ghost->owner halo tables, and owned<->global dof maps.
        # (u and d share the SAME cell/node split; only the dof layout differs by vec.)
        cell_part = cp.partition_cells(points, cells, D, method, axis=2)
        part_u = cp.build_mesh_partition(points, cells, D, vec=3, method=method, axis=2,
                                         cell_part=cell_part)
        part_d = cp.build_mesh_partition(points, cells, D, vec=1, method=method, axis=2,
                                         cell_part=cell_part)
        C = part_u.shard.cell_valid.shape[1]          # padded #owned-cells per shard
        self.C = C
        self.part_u, self.part_d = part_u, part_d

        # ---- 4c. Pack and SHARD the per-cell geometry (large arrays -> 1/D per
        #          device). Every geometry array of a field is sliced to the cells
        #          owned by each shard, padded to (D, C, ...) and sharded over 'd'
        #          so that device d stores only its slice. ----
        def pack_and_shard(arr):
            a = onp.asarray(arr)
            out = onp.zeros((D, C) + a.shape[1:], a.dtype)
            for d in range(D):
                oc = onp.where(cell_part == d)[0]
                out[d, :len(oc)] = a[oc]
            return cp.shard(out, self.DMESH)          # <-- 1/D per device (memory win)

        # (quad_points, shape_grads, JxW)
        self.GEO_u = tuple(pack_and_shard(g) for g in cp.field_geometry(field_u))
        self.GEO_d = tuple(pack_and_shard(g) for g in cp.field_geometry(field_d))
        n_geom = len(self.GEO_u) + 1                   # geometry + one internal variable

        # ---- 4d. Host-only owned<->global maps are kept in `part_*` and are used
        #          only after the solve when rank/processes gather the final fields
        #          for output.  They are deliberately not placed on every GPU.

        # ---- 4e. The distributed solvers (built once; compiled on first call).
        #   make_elem_fn wraps the element kernels of a field;
        #   make_distributed_newton = sharded element evaluation + halo-PCG
        #                             (displacement, AT2 damage);
        #   make_distributed_ssn    = the same with a bound-constrained active set
        #                             (AT1 / PF-CZM damage). ----
        elem_u = cp.make_elem_fn(field_u)
        elem_d = cp.make_elem_fn(field_d)
        self.elem_u = elem_u
        self.elem_res_u = elem_u.residual
        self.newton_u = cp.make_distributed_newton(
            part_u, self.DMESH, elem_u, tol=1e-10, maxiter=20000, n_geom=n_geom, n_newton=20)
        if MODEL == "AT2":                             # AT2 damage is linear -> Newton + clip
            self.solve_d = cp.make_distributed_newton(
                part_d, self.DMESH, elem_d, tol=1e-10, maxiter=20000, n_geom=n_geom, n_newton=20)
        else:                                          # AT1/CZM damage is bound-constrained -> SSN
            self.solve_d = cp.make_distributed_ssn(
                part_d, self.DMESH, elem_d, cg_tol=1e-10, cg_maxiter=20000,
                max_ssn=50, tol=1e-8, n_geom=n_geom)
        self.n_geom = n_geom

        # The distributed solvers hold their SHARDED ShardArrays as `.sd0`.  In a
        # multi-process run these span non-addressable devices, so the jitted
        # step_fn must NOT close over them.  Grab them here and thread them through
        # the scan carry as explicit arguments instead (see _build_step / run).
        self.SD_u = self.newton_u.sd0
        self.SD_d = self.solve_d.sd0

        # ---- 4f. Dirichlet + reaction bookkeeping in OWNED layout.  These are
        #          sharded over D as well, so field masks do not replicate O(N). ----
        fixed_u, is_top, top = dirichlet_bottom_top(points)
        self.top = top
        fixed_u_owned_np = onp.where(part_u.owned_valid_np,
                                     fixed_u.astype(float)[part_u.owned_global_dof], 0.0) > 0.5
        is_top_owned_np = onp.where(part_u.owned_valid_np,
                                    is_top[part_u.owned_global_dof], 0.0)
        self.fixed_u_owned = cp.shard(fixed_u_owned_np, self.DMESH)
        self.is_top_owned = cp.shard(is_top_owned_np, self.DMESH)
        self.zerosU = cp.shard(onp.zeros((D, part_u.rpd)), self.DMESH)
        self.zerosD = cp.shard(onp.zeros((D, part_d.rpd)), self.DMESH)
        self.no_fix_d = cp.shard(onp.zeros((D, part_d.rpd), bool), self.DMESH)
        self.crack_target = cp.shard(onp.zeros((D, part_d.rpd)), self.DMESH)
        self.shape_vals = cp.replicate(onp.asarray(field_d.shape_vals), self.DMESH)
        self.top_owned = cp.shard(is_top_owned_np > 0.5, self.DMESH)
        self._build_sharded_field_ops()
        self._build_step()

        # All runtime geometry is now in GEO_u/GEO_d and all partition tables in
        # SD_u/SD_d. The element kernels take the geometry as arguments, so the
        # full-mesh copies held by the fields are no longer needed; keeping them
        # would leave one complete mesh on the first GPU.
        field_u.release_geometry()
        field_d.release_geometry()

        # Runtime/output only needs the scalar layout metadata and host global
        # maps.  The complete host-side stacked tables are no longer needed after
        # SD_u/SD_d and the shard_map kernels have been built.
        self.part_u = self.part_u._replace(shard=None)
        self.part_d = self.part_d._replace(shard=None)
        del part_u, part_d, cell_part
        gc.collect()

        if ROOT and os.environ.get("PFF_MEM_DIAG", "1") != "0":
            lr_local = self.SD_u.LR.addressable_shards[0].data.shape
            geo_local = self.GEO_u[1].addressable_shards[0].data.shape
            print(f"[sharding] D={self.D}, cells_per_shard(padded)={self.C}; "
                  f"cell_batch={cp.configured_cell_batch_size(self.C)}; "
                  f"SD_u.LR global={self.SD_u.LR.shape}, local={lr_local}; "
                  f"GEO_u.shape_grads global={self.GEO_u[1].shape}, local={geo_local}",
                  flush=True)

    # ---- sharded field helpers -------------------------------------------------
    def _build_sharded_field_ops(self):
        """Build shard_map kernels that keep fields/history in owned layouts."""
        axis = "d"
        sd_spec = cp.ShardArrays(*[P(axis, None)] * len(cp.ShardArrays._fields))
        geom_spec = tuple(P(axis, None) for _ in range(self.n_geom))
        rpd_u, rpd_d = self.part_u.rpd, self.part_d.rpd
        nloc_u, nloc_d = self.part_u.nloc, self.part_d.nloc
        NIF_u, NIF_d = self.part_u.n_iface + 1, self.part_d.n_iface + 1
        vec_u, ndof_u = self.part_u.vec, self.part_u.ndof_c
        vec_d, ndof_d = self.part_d.vec, self.part_d.ndof_c
        elem_res_u = self.elem_res_u

        @partial(cp.shard_map, mesh=self.DMESH,
                 in_specs=(sd_spec, P(axis, None), P()),
                 out_specs=P(axis, None), **cp._SMKW)
        def _d_to_quad(sd, D0, shape_vals):
            local_cells = sd.local_cells[0]
            own_slot, own_iloc = sd.own_slot[0], sd.own_iface_local[0]
            gslot, cell_valid = sd.ghost_slot[0], sd.cell_valid[0]
            d0 = D0[0]
            nc = local_cells.shape[0]
            batch = cp.configured_cell_batch_size(nc)
            nblocks = nc // batch

            def gather(x):
                send = np.zeros(NIF_d, x.dtype).at[own_slot].set(x[own_iloc])
                return np.concatenate([x, jax.lax.psum(send, axis)[gslot]])

            d_ext = gather(d0).reshape(-1, vec_d)
            q0 = np.zeros((nc, shape_vals.shape[0], vec_d), dtype=d0.dtype)

            def body(ib, q):
                c0 = ib * batch
                lc = jax.lax.dynamic_slice_in_dim(local_cells, c0, batch, axis=0)
                valid = jax.lax.dynamic_slice_in_dim(cell_valid, c0, batch, axis=0)
                cell_d = d_ext[lc]
                qb = np.sum(cell_d[:, None, :, :] *
                            shape_vals[None, :, :, None], axis=2)
                qb = np.where(valid[:, None, None], qb, 0.0)
                return jax.lax.dynamic_update_slice_in_dim(q, qb, c0, axis=0)

            return jax.lax.fori_loop(0, nblocks, body, q0)[None, ...]

        @partial(cp.shard_map, mesh=self.DMESH,
                 in_specs=(sd_spec, P(axis, None), P(axis, None), P(axis, None)),
                 out_specs=P(axis, None), **cp._SMKW)
        def _history_update(sd, U0, H0, SG):
            local_cells = sd.local_cells[0]
            own_slot, own_iloc = sd.own_slot[0], sd.own_iface_local[0]
            gslot, cell_valid = sd.ghost_slot[0], sd.cell_valid[0]
            u0, h0, sg = U0[0], H0[0], SG[0]
            nc = local_cells.shape[0]
            batch = cp.configured_cell_batch_size(nc)
            nblocks = nc // batch

            def gather(x):
                send = np.zeros(NIF_u, x.dtype).at[own_slot].set(x[own_iloc])
                return np.concatenate([x, jax.lax.psum(send, axis)[gslot]])

            u_ext = gather(u0).reshape(-1, vec_u)
            vmap_psi = jax.vmap(jax.vmap(psi_plus))

            def body(ib, history):
                c0 = ib * batch
                lc = jax.lax.dynamic_slice_in_dim(local_cells, c0, batch, axis=0)
                sgb = jax.lax.dynamic_slice_in_dim(sg, c0, batch, axis=0)
                hb = jax.lax.dynamic_slice_in_dim(h0, c0, batch, axis=0)
                valid = jax.lax.dynamic_slice_in_dim(cell_valid, c0, batch, axis=0)
                cell_u = u_ext[lc]
                u_grads = cell_u[:, None, :, :, None] * sgb[:, :, :, None, :]
                u_grads = np.sum(u_grads, axis=2)
                h_new = np.maximum(vmap_psi(u_grads), hb)
                h_new = np.where(valid[:, None], h_new, 0.0)
                return jax.lax.dynamic_update_slice_in_dim(history, h_new, c0, axis=0)

            return jax.lax.fori_loop(0, nblocks, body, np.zeros_like(h0))[None, ...]

        @partial(cp.shard_map, mesh=self.DMESH,
                 in_specs=(sd_spec, P(axis, None), geom_spec, P(axis, None)),
                 out_specs=P(), **cp._SMKW)
        def _reaction(sd, U0, GEOM, TOP):
            local_cells = sd.local_cells[0]
            cell_valid = sd.cell_valid[0]
            own_slot, own_iloc = sd.own_slot[0], sd.own_iface_local[0]
            gslot, ownval = sd.ghost_slot[0], sd.owned_valid[0]
            u0, top = U0[0], TOP[0]
            geom0 = tuple(g[0] for g in GEOM)
            nc = local_cells.shape[0]
            gdof = (local_cells[:, :, None] * vec_u + np.arange(vec_u)).reshape(nc, -1)

            def gather(x):
                send = np.zeros(NIF_u, x.dtype).at[own_slot].set(x[own_iloc])
                return np.concatenate([x, jax.lax.psum(send, axis)[gslot]])

            def halo(y):
                contrib = np.zeros(NIF_u, y.dtype).at[gslot].add(y[rpd_u:])
                total = jax.lax.psum(contrib, axis)
                return y[:rpd_u].at[own_iloc].add(total[own_slot])

            u_ext = gather(u0)
            cell_sol = u_ext.reshape(-1, vec_u)[local_cells].reshape(nc, ndof_u)
            res_ext = cp.batch_element_residual(
                elem_res_u, cell_sol, geom0, cell_valid, gdof, nloc_u, ndof_u)
            res_owned = halo(res_ext)
            return jax.lax.psum(np.sum(np.where(top & ownval, res_owned, 0.0)), axis)

        self.d_to_quad = lambda d_owned, shards: _d_to_quad(shards, d_owned, self.shape_vals)
        self.history_update = lambda u_owned, h_owned, shape_grads, shards: \
            _history_update(shards, u_owned, h_owned, shape_grads)
        self.reaction_Fy = lambda u_owned, geom_u, shards: \
            _reaction(shards, u_owned, tuple(geom_u), self.top_owned)

    def _owned_to_global_host(self, part, owned):
        """Gather final owned-layout field to a host global vector, once per run."""
        local_ids = []
        local_vals = []
        for sh in owned.addressable_shards:
            # At D=1 the single shard spans the whole array, so index[0] is
            # slice(None) and .start is None -> that shard begins at row 0.
            start = sh.index[0].start
            d = 0 if start is None else int(start)
            local_ids.append(d)
            local_vals.append(onp.asarray(jax.device_get(sh.data)).reshape(-1, part.rpd)[0])
        local_ids = onp.asarray(local_ids, dtype=onp.int32)
        local_vals = onp.asarray(local_vals)
        if jax.process_count() > 1:
            local_ids = onp.asarray(multihost_utils.process_allgather(local_ids, tiled=True)).reshape(-1)
            local_vals = onp.asarray(multihost_utils.process_allgather(local_vals, tiled=True)).reshape(-1, part.rpd)
        full = onp.zeros(part.Nd_real, dtype=local_vals.dtype)
        for d, vals in zip(local_ids.tolist(), local_vals):
            valid = part.owned_valid_np[d]
            full[part.owned_global_dof[d, valid]] = vals[valid]
        return full

    def _build_step(self):
        """Compile the per-load-step function with sharded field/history carry."""
        MAX_STAGGER = 8
        MS_TOL = 1e-3                                  # staggered tolerance

        def step_fn(carry, disp):
            sol_u, sol_d, history, GEO_u, GEO_d, SD_u, SD_d = carry

            def cond(state):
                _, _, _, err, s = state
                return (s < MAX_STAGGER) & (err > MS_TOL)

            def body(state):
                su, sd, H, _, s = state

                d_quad = self.d_to_quad(sd, SD_d)                    # (D, C, nq, 1)
                u0 = np.where(self.fixed_u_owned, self.is_top_owned * disp, su)
                geom_u = (*GEO_u, d_quad)
                su_new = self.newton_u(u0, geom_u, self.zerosU, self.fixed_u_owned,
                                       shards=SD_u)                  # owned layout

                H_new = self.history_update(su_new, H, GEO_u[1], SD_u)
                geom_d = (*GEO_d, H_new)
                if MODEL == "AT2":
                    sd_new = np.clip(self.solve_d(sd, geom_d, self.zerosD, self.no_fix_d,
                                                  shards=SD_d), 0.0, 1.0)
                else:
                    sd_new = self.solve_d(sd, geom_d, self.no_fix_d, self.crack_target,
                                          shards=SD_d)

                eu = np.linalg.norm(su_new - su) / (np.linalg.norm(su_new) + 1e-30)
                ed = np.linalg.norm(sd_new - sd) / (np.linalg.norm(sd_new) + 1e-30)
                return (su_new, sd_new, H_new, np.maximum(eu, ed), s + 1)

            init = (sol_u, sol_d, history, np.array(1e30), np.array(0))
            sol_u, sol_d, history, _, _ = jax.lax.while_loop(cond, body, init)
            d_quad = self.d_to_quad(sol_d, SD_d)
            Fy = self.reaction_Fy(sol_u, (*GEO_u, d_quad), SD_u)
            return (sol_u, sol_d, history, GEO_u, GEO_d, SD_u, SD_d), \
                np.array([np.max(sol_d), Fy])

        self.step_fn = step_fn

    def run(self, disps):
        """Scan the load schedule. Fields stay sharded until this final gather."""
        rep = (lambda a: np.asarray(a)) if jax.process_count() == 1 \
            else (lambda a: cp.replicate(a, self.DMESH))
        carry0 = (cp.shard(onp.zeros((self.D, self.part_u.rpd)), self.DMESH),
                  cp.shard(onp.zeros((self.D, self.part_d.rpd)), self.DMESH),
                  cp.shard(onp.zeros((self.D, self.C, self.nq)), self.DMESH),
                  self.GEO_u, self.GEO_d, self.SD_u, self.SD_d)
        (sol_u_owned, sol_d_owned, _, _, _, _, _), out = jax.lax.scan(self.step_fn, carry0, rep(disps))
        sol_u = self._owned_to_global_host(self.part_u, sol_u_owned)
        sol_d = self._owned_to_global_host(self.part_d, sol_d_owned)
        return sol_u, sol_d, onp.asarray(out)

# =============================================================================
# 5. Serial DENSE reference (only for PFF_VERIFY self-check; not used in a real run)
# =============================================================================
def _dense_assemble(field, iv, u_flat, jac):
    theta = (np.asarray(iv),)
    z = np.asarray(u_flat)
    n = field.num_dofs
    cell_dofs = onp.asarray(field.cell_dofs).reshape(-1)
    if jac:
        r_c, K_c = field.evaluate_cells(z, theta, tangent=True)
        A = onp.zeros((n, n))
        onp.add.at(A, (field.coo_rows, field.coo_cols), onp.asarray(K_c).reshape(-1))
        R = onp.zeros(n); onp.add.at(R, cell_dofs, onp.asarray(r_c).reshape(-1))
        return A, R
    r_c = field.evaluate_cells(z, theta)
    R = onp.zeros(n); onp.add.at(R, cell_dofs, onp.asarray(r_c).reshape(-1))
    return R


def _dense_newton(prob, iv, u0, fixed, tgt=None):
    u0 = u0.copy()
    if tgt is not None:
        u0[fixed] = tgt[fixed]
    A, R = _dense_assemble(prob, iv, u0, True)
    free = onp.where(~fixed)[0]
    du = onp.zeros_like(u0); du[free] = onp.linalg.solve(A[free][:, free], -R[free])
    return u0 + du


def _dense_ssn(prob, iv, d0, max_it=50, tol=1e-8):
    N = d0.shape[0]
    A0, _ = _dense_assemble(prob, iv, d0, True)
    Dm = onp.abs(onp.diag(A0)); Dm = onp.where(Dm > 1e-12, Dm, 1.0); iD = 1.0 / Dm
    d = onp.clip(d0, 0.0, 1.0)
    for _ in range(max_it):
        R = _dense_assemble(prob, iv, d, False)
        trial = d - iD * R
        fixed = (trial <= 0.0) | (trial >= 1.0)
        tgt = onp.where(trial <= 0.0, 0.0, onp.where(trial >= 1.0, 1.0, d))
        ds = onp.where(fixed, tgt, d)
        A, R2 = _dense_assemble(prob, iv, ds, True); R2 = onp.where(fixed, 0.0, R2)
        free = onp.where(~fixed)[0]
        de = onp.zeros(N); de[free] = onp.linalg.solve(A[free][:, free], -R2[free])
        dn = onp.clip(ds + de, 0.0, 1.0)
        ch = onp.linalg.norm(dn - d) / (onp.linalg.norm(dn) + 1e-30)
        rn = onp.linalg.norm(_dense_assemble(prob, iv, dn, False)) / (onp.linalg.norm(Dm) + 1e-30)
        d = dn
        if ch < tol and rn < tol:
            break
    return d


def _dense_staggered(points, cells, pu, pd, disps, max_stagger=8, tol=1e-4):
    N, nc, nq = pu.num_nodes, pu.num_cells, pu.num_quads
    fixed, _, top = dirichlet_bottom_top(points)
    su = onp.zeros(N * 3); sd = onp.zeros(N); H = onp.zeros((nc, nq))
    c2q = pd.to_quad
    for disp in disps:
        tgt = onp.zeros(N * 3); tgt[top * 3 + 1] = disp
        for _ in range(max_stagger):
            su0, sd0 = su.copy(), sd.copy()
            dq = onp.asarray(c2q(sd.reshape(N, 1)))
            su = _dense_newton(pu, dq, su, fixed, tgt)
            H = onp.asarray(pu.compute_history(su.reshape(N, 3), H))
            if MODEL == "AT2":
                sd = onp.clip(_dense_newton(pd, H, sd, onp.zeros(N, bool)), 0.0, 1.0)
            else:
                sd = _dense_ssn(pd, H, sd)
            if max(onp.linalg.norm(su - su0) / (onp.linalg.norm(su) + 1e-30),
                   onp.linalg.norm(sd - sd0) / (onp.linalg.norm(sd) + 1e-30)) < tol:
                break
    return su, sd


# =============================================================================
# 6. Driver entry point
# =============================================================================
def _build_mesh(Nx, Ny, Nz, crack):
    """Structured HEX8 box -> optional geometric slit -> the two fields."""
    assert Nx % 2 == 0 and Ny % 2 == 0, "Nx, Ny must be EVEN (node row on y=0.5)"
    grid = axion.box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    points, cells = grid.points, grid.cells
    if crack == "slit":
        points, cells, _ = insert_crack_3d(points, cells, 0.5 * Ly, 0.5 * Lx)
    mesh = Mesh(points, cells, "HEX8")
    field_u = Displacement(mesh, ncomp=3)
    field_d = Damage(mesh, ncomp=1)
    return points, cells, field_u, field_d


def _self_check():
    """Small distributed cell-partition vs serial-dense staggered comparison."""
    points, cells, pu, pd = _build_mesh(4, 4, 2, "slit")
    disps = onp.array([0.001, 0.002])                  # tiny load -> stable, damage small
    su_ref, sd_ref = _dense_staggered(points, cells, pu, pd, disps)
    su, sd, _ = Driver(points, cells, pu, pd, method=PARTITION).run(disps)
    eu = onp.linalg.norm(su - su_ref) / onp.linalg.norm(su_ref)
    ed = onp.linalg.norm(sd - sd_ref) / (onp.linalg.norm(sd_ref) + 1e-30)
    if ROOT:
        ok = eu < 1e-6 and ed < 1e-6
        print(f"[{MODEL}] SELF-CHECK cell-partition vs dense: "
              f"sol_u={eu:.2e} sol_d={ed:.2e}  {'PASS' if ok else 'FAIL'}")


def main():
    Nx = int(os.environ.get("PFF_NX", 150))
    Ny = int(os.environ.get("PFF_NY", 150))
    Nz = int(os.environ.get("PFF_NZ", 15))
    nsteps = int(os.environ.get("PFF_NSTEPS", 160))
    du = float(os.environ.get("PFF_DU", 5e-5))
    crack = os.environ.get("PFF_CRACK", "slit")

    t_wall0 = time.time()
    points, cells, field_u, field_d = _build_mesh(Nx, Ny, Nz, crack)
    n_nodes = field_u.num_nodes
    n_cells = field_u.num_cells
    solver = Driver(points, cells, field_u, field_d, method=PARTITION)
    _memory_report("after Driver construction and full-geometry release")
    if ROOT:
        print(f"[{MODEL}] {n_nodes} nodes, {n_cells} HEX8 cells, crack={crack}, "
              f"D={solver.D} devices, {jax.process_count()} process(es)", flush=True)

    disps = onp.arange(1, nsteps + 1) * du

    # Optional compile warmup (weak-scaling timings must NOT count one-off JIT
    # compilation). Running the SAME-length scan once on a zero-displacement
    # schedule compiles every kernel; the timed run below then hits the XLA
    # cache, so solve_time is pure execution. Enabled by default; PFF_WARMUP=0
    # disables it.
    if os.environ.get("PFF_WARMUP", "1") != "0":
        if ROOT:
            print(f"[{MODEL}] compile warmup (zero-disp, {nsteps} steps)...", flush=True)
        _tw = time.time()
        solver.run(onp.zeros_like(disps))              # blocks (host gather at end)
        if ROOT:
            print(f"[{MODEL}] warmup/compile done in {time.time() - _tw:.1f}s", flush=True)

    t0 = time.time()
    sol_u, sol_d, out = solver.run(disps)              # out[:, 0] = max_d, out[:, 1] = Fy
    mem_after = _memory_report("after solve")
    solve_time = time.time() - t0
    wall_time = time.time() - t_wall0

    # ---- GPU memory (bytes) after the solve.  peak_bytes_in_use is a since-start
    #      high-water mark, so reading it here covers the whole run.  Take the max
    #      over this rank's local devices; sum(...) if you want the per-node total. ----
    mib = 1024.0 ** 2
    gib = 1024.0 ** 3
    peak_bytes = max((s.get("peak_bytes_in_use", 0) for s in mem_after.values()),
                     default=0)
    inuse_bytes = max((s.get("bytes_in_use", 0) for s in mem_after.values()),
                      default=0)
    limit_bytes = max((s.get("bytes_limit", 0) for s in mem_after.values()),
                      default=0)


    if not ROOT:                                        # final gather is done; only rank 0 writes / prints
       return

    # ---- assemble the force-displacement curve (fields were gathered once after scan) ----
    forces = onp.concatenate([[0.0], onp.asarray(out[:, 1])])
    full_disps = onp.concatenate([[0.0], disps])
    sol_d_full = onp.asarray(sol_d).reshape(-1)

    # ---- save to HDF5 (same layout as 02_sent_3d/sent_3d.py) ----
    import h5py
    # Tag the output dir by the partition size so a weak-/strong-scaling sweep
    # (D = 1, 2, 4, ...) does not overwrite itself. Override with PFF_TAG.
    tag = os.environ.get("PFF_TAG", f"n{jax.process_count()}_gpu{solver.D}")
    out_root = os.environ.get("PFF_OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(out_root, f"output_{tag}_{MODEL}")
    os.makedirs(out_dir, exist_ok=True)

    def _attrs(f):
        f.attrs["n_cells"] = int(n_cells); f.attrs["n_nodes"] = int(n_nodes)
        f.attrs["solve_time"] = float(solve_time); f.attrs["wall_time"] = float(wall_time)
        f.attrs["peak_load"] = float(onp.max(onp.abs(forces)))
        f.attrs["peak_bytes_in_use"] = int(peak_bytes)
        f.attrs["bytes_in_use"] = int(inuse_bytes)
        f.attrs["bytes_limit"] = int(limit_bytes)

    with h5py.File(os.path.join(out_dir, f"force_disp_{MODEL}.h5"), "w") as f:
        f.create_dataset("disps", data=full_disps)
        f.create_dataset("forces", data=forces)
        _attrs(f)
    with h5py.File(os.path.join(out_dir, f"final_{MODEL}.h5"), "w") as f:
        f.create_dataset("points", data=onp.asarray(points))
        f.create_dataset("cells", data=onp.asarray(cells))
        f.create_dataset("sol_d", data=sol_d_full)
        f.create_dataset("disps", data=full_disps)
        f.create_dataset("forces", data=forces)
        _attrs(f)
        f.attrs["Lx"] = float(Lx); f.attrs["Ly"] = float(Ly); f.attrs["Lz"] = float(Lz)

    # ---- per-step log + timing summary ----
    for i, (md, fy) in enumerate(zip(out[:, 0], out[:, 1]), 1):
        print(f"  step {i:4d}/{nsteps}  disp={disps[i-1]:.4e}  "
              f"|Fy|={abs(float(fy)):.4e}  max_d={float(md):.3f}")
    dsolver = ("distributed semismooth Newton (SSN, bound-constrained)"
               if MODEL in ("AT1", "CZM") else "distributed Newton + clip")
    summary = (
        "==================== TIMING / METHOD SUMMARY ====================\n"
        f"  model                : {MODEL}\n"
        f"  mesh                 : UNIFORM fixed HEX8 ({n_cells} cells, {n_nodes} nodes)\n"
        f"  partition            : cell partition over {solver.D} device(s), "
        f"{jax.process_count()} process(es)\n"
        f"  scheme               : staggered (alternating minimization), jax.lax.scan\n"
        f"  u-solver             : distributed Newton (cell-partition assembly, halo-CG)\n"
        f"  d-solver             : {dsolver}\n"
        f"  jit                  : compiled once (fixed mesh, no recompilation)\n"
        f"  load steps           : {nsteps}\n"
        f"  TOTAL solve time     : {solve_time:.1f} s  ({solve_time / 60.:.2f} min)\n"
        f"  TOTAL wall time      : {wall_time:.1f} s  ({wall_time / 60.:.2f} min)\n"
        f"  avg time / load step : {solve_time / max(nsteps, 1):.3f} s\n"
        f"  peak |Fy|            : {float(onp.max(onp.abs(forces))):.4e}   "
        f"final max_d : {float(sol_d_full.max()):.3f}\n"
        f"  peak GPU mem in use  : {peak_bytes / mib:.1f} MiB  "
        f"({peak_bytes / gib:.2f} GiB)  [per device, rank 0]\n"
        f"  live GPU mem in use  : {inuse_bytes / mib:.1f} MiB  "
        f"({inuse_bytes / gib:.2f} GiB)\n"
        f"  GPU mem limit        : {limit_bytes / mib:.1f} MiB  "
        f"({limit_bytes / gib:.2f} GiB)\n"
        "=================================================================")
    print("\n" + summary)
    with open(os.path.join(out_dir, f"timing_{MODEL}.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(f"[{MODEL}] output in {out_dir}/  "
          f"(force_disp_{MODEL}.h5, final_{MODEL}.h5, timing_{MODEL}.txt)")

    # ---- scaling record: one CSV row per run, shared across a sweep ----
    # DOFs = 3 displacement + 1 damage per node. time_per_step excludes the
    # one-off compile (see PFF_WARMUP above), so it is the steady-state cost.
    n_dofs = 4 * int(n_nodes)
    D = int(solver.D)
    csv_path = os.path.join(out_root, os.environ.get("PFF_CSV", "scaling.csv"))
    header = ("model,D,NX,NY,NZ,n_cells,n_nodes,n_dofs,cells_per_gpu,dofs_per_gpu,"
              "nsteps,solve_time_s,time_per_step_s,wall_time_s,peak_mem_GiB\n")
    row = (f"{MODEL},{D},{Nx},{Ny},{Nz},{int(n_cells)},{int(n_nodes)},{n_dofs},"
           f"{int(n_cells) // D},{n_dofs // D},{nsteps},{solve_time:.4f},"
           f"{solve_time / max(nsteps, 1):.6f},{wall_time:.4f},{peak_bytes / gib:.4f}\n")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as fh:
        if write_header:
            fh.write(header)
        fh.write(row)
    print(f"[{MODEL}] appended scaling row (D={D}) to {csv_path}")


if __name__ == "__main__":
    if os.environ.get("PFF_VERIFY"):
        _self_check()
    else:
        main()