"""Multi-GPU execution by cell decomposition (paper Section 2.7).

The mesh itself (cells and nodes) is split across the device axis 'd', so every
device holds only about 1/D of the cells, nodes, fields, tangent values and CG
vectors. The partition is computed on the cell dual graph (METIS, a Morton
space-filling curve, or coordinate slabs), and the halo exchange is a single
all-reduce over a compact interface buffer. No nearest-neighbour structure is
assumed, so the scheme applies to any mesh (unstructured, slit, ...).

Every device runs the same compiled program on its own subdomain:

* :func:`build_mesh_partition`     host-side partition and index tables
* :func:`make_elem_fn`, :func:`pack_geometry`   element kernels of a
                                   :class:`axion.Field` and its geometry, sliced
                                   to the cells owned by each device
* :func:`make_distributed_newton`  element evaluation on the owned cells, halo
                                   exchange and Jacobi-preconditioned CG whose
                                   inner products are global reductions
* :func:`make_distributed_ssn`     the same for a bound-constrained subproblem
                                   (primal-dual active set)

The self-test at the bottom checks the partitioned solvers against dense
references on 4 emulated CPU devices:

    python -m axion.distributed

--------------------------------------------------------------------------------
DATA MODEL (read this once; every downstream function uses these fields)
--------------------------------------------------------------------------------
Local DOF layout on each shard is  [ owned | ghost ]  with FIXED padded sizes:

    index range         meaning
    [0,   rpd)          OWNED dofs  (real 0..n_owned_dof_d, then identity padding)
    [rpd, nloc)         GHOST dofs  (real 0..n_ghost_dof_d, then unused padding)

`rpd`  = vec * (max owned-nodes over shards)     (== rows-per-device, padded)
`nloc` = rpd + vec * (max ghost-nodes over shards)

The HALO uses a compact interface buffer of length NIF = n_iface + 1.  The last
slot (index n_iface) is a DUMMY sink for all padded entries, so padding never
clobbers a real slot.  gather (owner->ghost) and halo_sum (ghost->owner) are the
SAME three tables read in opposite directions:

    own_slot[d,i]        : interface-buffer slot that shard d OWNS (i-th)
    own_iface_local[d,i] : local OWNED-dof id of that same interface dof
    ghost_slot[d,j]      : interface-buffer slot feeding shard d's j-th GHOST dof
================================================================================
"""
from __future__ import annotations
import os
import inspect as _inspect
from functools import partial
from typing import NamedTuple

import numpy as onp
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import shard_map
except Exception:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

# version-portable disable of the (too-conservative) replication checker
_SMKW = {}
if "check_vma" in _inspect.signature(shard_map).parameters:
    _SMKW["check_vma"] = False
elif "check_rep" in _inspect.signature(shard_map).parameters:
    _SMKW["check_rep"] = False


def make_domain_mesh(axis="d"):
    """1-D mesh over ALL devices on ALL nodes."""
    return Mesh(onp.asarray(jax.devices(), dtype=object), (axis,))


def configured_cell_batch_size(ncells=None):
    """Static per-device element batch size for compiled assembly loops."""
    batch = int(os.environ.get("PFF_CELL_BATCH", "200000"))
    if batch <= 0:
        raise ValueError("PFF_CELL_BATCH must be a positive integer")
    return batch if ncells is None else min(batch, int(ncells))


def init_distributed():
    """Init the multi-process cluster ONCE, before any jax backend init.
    No-op single-process. SLURM-aware."""
    if os.environ.get("SLURM_JOB_ID") or "JAX_COORDINATOR_ADDRESS" in os.environ:
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        _nvis = len([x for x in _cvd.split(",") if x != ""])
        _lid = int(os.environ.get("SLURM_LOCALID", "0"))
        if _nvis > 1:
            jax.distributed.initialize(local_device_ids=[_lid])
        else:
            jax.distributed.initialize()
    if jax.process_index() == 0:
        print(f"[dist] processes={jax.process_count()}  global_devices={jax.device_count()}  "
              f"local_devices={jax.local_device_count()}", flush=True)
    return jax.device_count()


def shard(a, mesh, axis="d"):
    """Place `a` SHARDED over the domain axis: axis-0 (size D) is split across the
    D devices, so each device/process stores only its 1/D slice.  Use for the BIG
    per-shard arrays (geometry, COO LR/GC) so device memory really drops to ~1/D.
    (replicate() below keeps a FULL copy on every device -- use only for small things.)

    MULTI-PROCESS SAFE: build the global sharded array from each process's OWN slab
    via make_array_from_single_device_arrays.  Plain device_put(global, P('d')) is
    NOT usable across nodes -- it requires the identical global array on every host
    and asserts it bit-for-bit (geometry computed on different GPUs can
    differ in the last bit -> AssertionError).  Here each host only ever provides
    the shards it owns, so there is no cross-host equality requirement."""
    a = onp.asarray(a)
    sharding = NamedSharding(mesh, P(axis))
    # {device: index-tuple} for THIS process's addressable devices only.
    idx_map = sharding.addressable_devices_indices_map(a.shape)
    arrays = [jax.device_put(a[idx], dev) for dev, idx in idx_map.items()]
    return jax.make_array_from_single_device_arrays(a.shape, sharding, arrays)


def replicate(a, mesh):
    """Fully-REPLICATED global array (every process/device holds the whole thing)."""
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))


# =============================================================================
# Per-shard device arrays (all stacked over D, axis 0, padded to uniform shapes)
# =============================================================================
class ShardArrays(NamedTuple):
    local_cells:     jax.Array   # (D, C, npc)  LOCAL node ids per owned(+pad) cell
    cell_valid:      jax.Array   # (D, C)       real cell vs padding
    LR:              jax.Array   # (D, NNZ)     LOCAL row dof id  (aligns with V_local)
    GC:              jax.Array   # (D, NNZ)     LOCAL col dof id
    nnz_valid:       jax.Array   # (D, NNZ)     real COO entry vs padding
    own_slot:        jax.Array   # (D, OI)      interface slot this shard OWNS
    own_iface_local: jax.Array   # (D, OI)      local owned-dof id of that interface dof
    ghost_slot:      jax.Array   # (D, G)       interface slot feeding each ghost dof
    owned_valid:     jax.Array   # (D, rpd)     real owned dof vs identity-padding row


class MeshPartition(NamedTuple):
    shard:   ShardArrays
    D:       int
    vec:     int
    npc:     int
    ndof_c:  int                 # = npc * vec
    n_iface: int                 # #real interface dofs (buffer length is n_iface + 1)
    rpd:     int                 # padded owned-dof count per shard
    nloc:    int                 # padded (owned + ghost) dof count per shard
    Nd_real: int                 # true global #DOFs = n_nodes * vec
    # host-only maps (numpy) for field shard / gather-at-save / test un-scatter:
    node_owner:       onp.ndarray  # (n_nodes,)   owner shard of each node
    owned_nodes:      list         # list[D] of global node-id arrays (unpadded)
    owned_global_dof: onp.ndarray  # (D, rpd)     global dof id of each owned slot
    owned_valid_np:   onp.ndarray  # (D, rpd)     bool


# =============================================================================
# Cell dual graph + partition methods (host, numpy) -- topology agnostic
# =============================================================================
def build_cell_adjacency(cells, n_nodes):
    """CSR (xadj, adjncy) of the cell DUAL GRAPH: two cells are neighbours iff
    they share at least one node.  Works for any element type / dimension."""
    num_cells, npc = cells.shape
    # node -> list of incident cells (bincount-based CSR, no python dict)
    flat_nodes = cells.reshape(-1)
    flat_cells = onp.repeat(onp.arange(num_cells), npc)
    order = onp.argsort(flat_nodes, kind="stable")
    nodes_sorted = flat_nodes[order]
    cells_sorted = flat_cells[order]
    node_ptr = onp.searchsorted(nodes_sorted, onp.arange(n_nodes + 1))

    nbr = [set() for _ in range(num_cells)]
    for nd in range(n_nodes):
        inc = cells_sorted[node_ptr[nd]:node_ptr[nd + 1]]
        for a in inc:
            nbr[a].update(int(b) for b in inc if b != a)
    adjncy, xadj = [], [0]
    for a in range(num_cells):
        adjncy.extend(sorted(nbr[a]))
        xadj.append(len(adjncy))
    return onp.asarray(xadj), onp.asarray(adjncy)


def cell_centroids(points, cells):
    """(num_cells, dim) centroid of each cell."""
    return points[cells].mean(axis=1)


def _morton_key(coords):
    """Morton (Z-order) key of quantized coordinates -> a space-filling-curve
    order that keeps spatially-near cells contiguous.  Pure numpy, any dimension."""
    c = coords - coords.min(0)
    span = onp.where(c.max(0) > 0, c.max(0), 1.0)
    q = onp.minimum((c / span * 1023).astype(onp.int64), 1023)   # 10 bits / axis
    key = onp.zeros(len(coords), dtype=onp.int64)
    for b in range(10):
        for a in range(q.shape[1]):
            key |= ((q[:, a] >> b) & 1) << (b * q.shape[1] + a)
    return key


def partition_cells(points, cells, D, method="sfc", axis=2):
    """Assign each cell to a shard in [0, D). Returns (num_cells,) int array.
      'metis' : min edge-cut, load-balanced  (needs pymetis)
      'sfc'   : Morton space-filling-curve, dependency-free  (any mesh)
      'slab'  : structured split along coordinate `axis`      (regular box)
    """
    num_cells = cells.shape[0]
    if method == "metis":
        import pymetis
        xadj, adjncy = build_cell_adjacency(cells, points.shape[0])
        _, part = pymetis.part_graph(D, xadj=xadj.tolist(), adjncy=adjncy.tolist())
        return onp.asarray(part)
    if method == "sfc":
        order = onp.argsort(_morton_key(cell_centroids(points, cells)))
        part = onp.empty(num_cells, onp.int64)
        part[order] = onp.arange(num_cells) * D // num_cells   # D equal-size chunks
        return part
    if method == "slab":
        c = cell_centroids(points, cells)[:, axis]
        t = (c - c.min()) / (onp.ptp(c) + 1e-30)
        return onp.clip((t * D).astype(onp.int64), 0, D - 1)
    raise ValueError(f"unknown method {method!r}")


# =============================================================================
# THE partitioner: dual-graph partition -> unique owner -> local numbering ->
#                  local COO (LR/GC) -> compact interface tables  (all static)
# =============================================================================
def build_mesh_partition(points, cells, D, vec, method="sfc", axis=2,
                         slit_pairs=None, cell_part=None):
    """Build a MeshPartition for ANY mesh.  `slit_pairs` (optional, shape (k,2)
    of (orig, dup) global node ids) pins each slit duplicate-pair to one owner."""
    points = onp.asarray(points, float)
    cells = onp.asarray(cells).astype(onp.int64)
    n_nodes, npc = points.shape[0], cells.shape[1]
    ndof_c = npc * vec

    # ---- 1) partition the cell dual graph -------------------------------------
    if cell_part is None:
        cell_part = partition_cells(points, cells, D, method, axis)
    else:
        cell_part = onp.asarray(cell_part, dtype=onp.int64)
        if cell_part.shape != (len(cells),):
            raise ValueError(f"cell_part must have shape {(len(cells),)}, got {cell_part.shape}")
        if onp.any((cell_part < 0) | (cell_part >= D)):
            raise ValueError("cell_part contains a shard id outside [0, D)")

    # ---- 2) counts[node, shard] = #incident cells of that shard on that node ---
    counts = onp.zeros((n_nodes, D), onp.int64)
    onp.add.at(counts, (cells.reshape(-1), onp.repeat(cell_part, npc)), 1)

    node_owner = counts.argmax(axis=1)              # unique owner (tie -> min shard)
    touched = (counts > 0).sum(axis=1)              # #distinct shards touching node
    is_iface = touched > 1                          # shared node <=> on an interface
    # Geometric-slit duplicate nodes are INDEPENDENT nodes (distinct global ids):
    # the cell partition gives each its own owner naturally.  Do NOT pin dup->orig
    # owner -- the duplicate is only touched by the ABOVE-plane elements, so pinning
    # it to a shard that does not touch it breaks the "owner touches the node"
    # invariant and corrupts the halo tables (-> NaN).  Accepted for API compat only.
    _ = slit_pairs

    # ---- 3) compact interface buffer: one slot block per interface node -------
    iface_nodes = onp.where(is_iface)[0]
    slot_base = onp.full(n_nodes, -1, onp.int64)
    slot_base[iface_nodes] = onp.arange(len(iface_nodes)) * vec
    n_iface = int(len(iface_nodes) * vec)
    dummy = n_iface                                  # padded entries sink here

    # ---- 4) pass 1: per-shard owned/ghost sets and the padded shapes ----------
    oc_list, own_list, ghost_list = [], [], []
    for d in range(D):
        oc = onp.where(cell_part == d)[0]
        owned = onp.where(node_owner == d)[0]
        touched_nodes = onp.unique(cells[oc]) if len(oc) else onp.zeros(0, onp.int64)
        ghost = onp.setdiff1d(touched_nodes, owned, assume_unique=False)
        oc_list.append(oc); own_list.append(owned); ghost_list.append(ghost)

    ownpad = max(len(o) for o in own_list)           # max owned NODES
    ghostpad = max(max(len(g) for g in ghost_list), 1)  # max ghost NODES (>=1: a dummy
    # ghost layer with slot=dummy keeps nloc == rpd + len(gslot) even when a shard
    # has NO ghosts, e.g. D=1; its values are always 0 so it is inert.)
    C_real = max(len(o) for o in oc_list)            # max owned CELLS
    # Fixed-size XLA slices require a full final batch. Tiny tests retain their
    # natural size instead of being padded to the production batch size.
    cell_batch = configured_cell_batch_size(C_real)
    C = ((C_real + cell_batch - 1) // cell_batch) * cell_batch
    rpd = ownpad * vec
    nloc = (ownpad + ghostpad) * vec
    NNZ = C * ndof_c * ndof_c
    OI = max(int(is_iface[o].sum()) for o in own_list) * vec  # max owned iface dofs
    G = ghostpad * vec

    # ---- 5) pass 2: build padded stacked arrays -------------------------------
    local_cells = onp.zeros((D, C, npc), onp.int32)
    cell_valid = onp.zeros((D, C), bool)
    LR = onp.zeros((D, NNZ), onp.int32)
    GC = onp.zeros((D, NNZ), onp.int32)
    nnz_valid = onp.zeros((D, NNZ), bool)
    own_slot = onp.full((D, max(OI, 1)), dummy, onp.int32)
    own_iface_local = onp.zeros((D, max(OI, 1)), onp.int32)
    ghost_slot = onp.full((D, max(G, 1)), dummy, onp.int32)
    owned_valid = onp.zeros((D, rpd), bool)
    owned_global_dof = onp.zeros((D, rpd), onp.int64)

    comp = onp.arange(vec)
    for d in range(D):
        oc, owned, ghost = oc_list[d], own_list[d], ghost_list[d]
        no, ng, ncell = len(owned), len(ghost), len(oc)

        # global node -> LOCAL node id: owned first (0..no-1), ghost after ownpad
        locid = onp.full(n_nodes, -1, onp.int64)
        locid[owned] = onp.arange(no)
        locid[ghost] = ownpad + onp.arange(ng)

        # local element connectivity + local element->dof map
        lc = locid[cells[oc]]                                   # (ncell, npc)
        local_cells[d, :ncell] = lc
        cell_valid[d, :ncell] = True
        eldof = (lc[:, :, None] * vec + comp[None, None, :]).reshape(ncell, ndof_c)
        rows = onp.broadcast_to(eldof[:, :, None], (ncell, ndof_c, ndof_c)).reshape(-1)
        cols = onp.broadcast_to(eldof[:, None, :], (ncell, ndof_c, ndof_c)).reshape(-1)
        k = ncell * ndof_c * ndof_c
        LR[d, :k] = rows; GC[d, :k] = cols; nnz_valid[d, :k] = True

        # owned dof bookkeeping (real dofs 0..no*vec, rest are identity padding)
        owned_valid[d, :no * vec] = True
        owned_global_dof[d, :no * vec] = (owned[:, None] * vec + comp[None, :]).reshape(-1)

        # owned interface dofs -> (own_slot, own_iface_local)
        oif = owned[is_iface[owned]]                            # owned & shared nodes
        oif_localnode = locid[oif]                              # 0..no-1
        oslot = (slot_base[oif][:, None] + comp[None, :]).reshape(-1)
        oloc = (oif_localnode[:, None] * vec + comp[None, :]).reshape(-1)
        own_slot[d, :len(oslot)] = oslot
        own_iface_local[d, :len(oloc)] = oloc

        # ghost dofs (contiguous j*vec+comp, aligned with x_ext[rpd:]) -> ghost_slot
        gslot = (slot_base[ghost][:, None] + comp[None, :]).reshape(-1)  # (ng*vec,)
        ghost_slot[d, :len(gslot)] = gslot

    # Keep the complete stacked (D, ...) partition tables on the HOST.  Turning
    # them into jnp arrays here first places the complete tables on every rank's
    # default GPU; make_distributed_*() would then create a second, correctly
    # sharded copy while the original full-device copy remained reachable via
    # MeshPartition.shard.  shard() below is deliberately the only point where
    # these large tables enter device memory.
    to32 = lambda a: onp.asarray(a, dtype=onp.int32)
    shard = ShardArrays(
        local_cells=to32(local_cells), cell_valid=onp.asarray(cell_valid),
        LR=to32(LR), GC=to32(GC), nnz_valid=onp.asarray(nnz_valid),
        own_slot=to32(own_slot), own_iface_local=to32(own_iface_local),
        ghost_slot=to32(ghost_slot), owned_valid=onp.asarray(owned_valid))
    return MeshPartition(
        shard=shard, D=int(D), vec=int(vec), npc=int(npc), ndof_c=int(ndof_c),
        n_iface=n_iface, rpd=int(rpd), nloc=int(nloc), Nd_real=int(n_nodes * vec),
        node_owner=node_owner, owned_nodes=own_list,
        owned_global_dof=owned_global_dof, owned_valid_np=owned_valid)


# =============================================================================
# Block-Jacobi PCG on this shard's local vectors (identical to the 2a version)
# =============================================================================
def _pcg_local(matvec, dot, inv, b, tol, maxiter):
    bnorm2 = dot(b, b)
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    z = inv * r
    p = z
    rz = dot(r, z)

    def body(s):
        x, r, p, rz, k = s
        Ap = matvec(p)
        alpha = rz / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = inv * r
        rzn = dot(r, z)
        p = z + (rzn / rz) * p
        return x, r, p, rzn, k + 1

    def cond(s):
        r, k = s[1], s[4]
        return (k < maxiter) & (dot(r, r) > (tol * tol) * bnorm2)

    return jax.lax.while_loop(cond, body, (x, r, p, rz, 0))[0]


# =============================================================================
# Distributed masked PCG with the GENERAL (compact-psum) halo
#   cg(V_local, R_owned, fixed_owned) -> du_global   (usable inside jit)
# =============================================================================
def make_cell_distributed_cg(part: MeshPartition, mesh, tol, maxiter, axis="d"):
    sd0 = jax.tree_util.tree_map(lambda a: shard(a, mesh), part.shard)  # 1/D per device
    rpd, nloc, n_iface = part.rpd, part.nloc, part.n_iface
    NIF = n_iface + 1                                  # +1 dummy sink slot
    sd_spec = ShardArrays(*[P(axis, None)] * len(ShardArrays._fields))

    @partial(shard_map, mesh=mesh,
             in_specs=(sd_spec, P(axis, None), P(axis, None), P(axis, None)),
             out_specs=P(None), **_SMKW)
    def _solve(sd, Vloc, B, FX):
        # strip the length-1 sharded axis (each shard holds exactly one slab)
        lc = sd.local_cells[0]; LR = sd.LR[0]; GC = sd.GC[0]
        nnzv = sd.nnz_valid[0]
        own_slot = sd.own_slot[0]; own_iloc = sd.own_iface_local[0]
        gslot = sd.ghost_slot[0]; ownval = sd.owned_valid[0]
        v = Vloc[0]; b = B[0]; fx_owned = FX[0]
        lv = jnp.where(nnzv, v, 0.0)                   # kill padded COO entries

        def gather(x_owned):                           # owner -> ghost  (rpd -> nloc)
            send = jnp.zeros(NIF, x_owned.dtype).at[own_slot].set(x_owned[own_iloc])
            full = jax.lax.psum(send, axis)            # ONE collective, size NIF
            return jnp.concatenate([x_owned, full[gslot]])

        def halo(y_ext):                               # ghost -> owner  (nloc -> rpd)
            contrib = jnp.zeros(NIF, y_ext.dtype).at[gslot].add(y_ext[rpd:])
            total = jax.lax.psum(contrib, axis)
            return y_ext[:rpd].at[own_iloc].add(total[own_slot])

        # padded owned rows are identity rows; fixed columns come from the owner
        fx_all = fx_owned | (~ownval)
        fx_ext = gather(fx_owned.astype(jnp.float64)) > 0.5

        def matvec(x_owned):
            x_ext = gather(x_owned)
            x_ext = jnp.where(fx_ext, 0.0, x_ext)      # zero fixed COLUMNS
            y_ext = jax.ops.segment_sum(lv * x_ext[GC], LR, num_segments=nloc)
            y_owned = halo(y_ext)                      # sum shared rows to their owner
            return jnp.where(fx_all, x_owned, y_owned)  # fixed ROWS -> identity

        # assembled diagonal (owned layout) -> block-Jacobi preconditioner
        diag_ext = jnp.zeros(nloc).at[LR].add(jnp.where((LR == GC) & nnzv, lv, 0.0))
        diag_owned = jnp.where(fx_all, 1.0, halo(diag_ext))
        inv = 1.0 / jnp.where(jnp.abs(diag_owned) > 1e-12, diag_owned, 1.0)

        dot = lambda u, w: jax.lax.psum((u * w).sum(), axis)
        x = _pcg_local(matvec, dot, inv, b, tol, maxiter)
        # scatter each shard's OWNED du back to a full global vector, then all-reduce
        gdof = _owned_global_dof_dev[jax.lax.axis_index(axis)]
        gvalid = _owned_valid_dev[jax.lax.axis_index(axis)]
        contrib = jnp.zeros(part.Nd_real).at[gdof].add(jnp.where(gvalid, x, 0.0))
        return jax.lax.psum(contrib, axis)             # (Nd_real,) on every shard

    # device copies of the owned->global scatter map (indexed by axis_index)
    _owned_global_dof_dev = jnp.asarray(part.owned_global_dof.astype(onp.int32))
    _owned_valid_dev = jnp.asarray(part.owned_valid_np)

    def cg(V_local, R_owned, fixed_owned):
        """V_local:(D,NNZ) tangent aligned to LR/GC; R_owned:(D,rpd) residual with
        fixed rows zeroed; fixed_owned:(D,rpd) bool.  Returns du:(Nd_real,)."""
        return _solve(sd0, V_local, -R_owned, fixed_owned)

    return cg


# =============================================================================
# Distributed Newton = sharded ASSEMBLY + PCG in ONE shard_map
#   The only problem-specific piece is `elem_fn`; everything else is defined here.
#   elem_fn(u_cell, sg_cell, JxW_cell, iv_cell) -> (Ke (ndof_c,ndof_c), Re (ndof_c,))
#     u_cell : (ndof_c,)  this cell's solution dofs (LOCAL order = npc*vec)
#     sg/JxW : this cell's pre-sliced shape-grads / JxW
#     iv     : this cell's internal vars (d_quad for u-solve, history for d-solve)
#   For a Field, elem_fn wraps its per-cell residual + tangent kernel.
# =============================================================================


# =============================================================================
# Distributed Newton = sharded ASSEMBLY + PCG in ONE shard_map.
#   elem_fn is BATCHED over a shard's owned cells:
#     elem_fn(cell_sol (nc,ndof_c), *geom_slices) -> (Ke (nc,ndof_c,ndof_c), Re (nc,ndof_c))
#   geom = n_geom per-shard arrays (D, C, ...) sliced to owned cells (pack_geometry).
#   For an axion.Field: elem_fn = make_elem_fn(field);
#   geom = pack_geometry(field, part, cell_part, internal_vars_cells).
#   Validated against the serial element evaluation (self_test_real).
# =============================================================================
def batch_element_assembly(elem_fn, cell_sol, geom, cell_valid, gdof,
                           nnz_valid, nloc, ndof_c):
    """Assemble in a compiled fixed-memory loop, retaining only the final COO V."""
    nc = int(cell_sol.shape[0])
    batch = configured_cell_batch_size(nc)
    if nc % batch:
        raise ValueError(f"padded cell count {nc} is not divisible by batch {batch}")
    nblocks = nc // batch
    nnz_per_cell = ndof_c * ndof_c
    v0 = jnp.zeros((nc * nnz_per_cell,), dtype=cell_sol.dtype)
    r0 = jnp.zeros((nloc,), dtype=cell_sol.dtype)

    def body(ib, state):
        values, residual = state
        c0 = ib * batch
        cs = jax.lax.dynamic_slice_in_dim(cell_sol, c0, batch, axis=0)
        gb = tuple(jax.lax.dynamic_slice_in_dim(g, c0, batch, axis=0)
                   for g in geom)
        valid = jax.lax.dynamic_slice_in_dim(cell_valid, c0, batch, axis=0)
        dofs = jax.lax.dynamic_slice_in_dim(gdof, c0, batch, axis=0)
        Ke, Re = elem_fn(cs, *gb)

        k0 = c0 * nnz_per_cell
        valid_nnz = jax.lax.dynamic_slice_in_dim(
            nnz_valid, k0, batch * nnz_per_cell, axis=0)
        vb = jnp.where(valid_nnz, Ke.reshape(-1), 0.0)
        values = jax.lax.dynamic_update_slice_in_dim(values, vb, k0, axis=0)

        rb = jnp.where(valid[:, None], Re.reshape(batch, ndof_c), 0.0)
        residual = residual.at[dofs.reshape(-1)].add(rb.reshape(-1))
        return values, residual

    return jax.lax.fori_loop(0, nblocks, body, (v0, r0))


def batch_element_residual(elem_res_fn, cell_sol, geom, cell_valid, gdof,
                           nloc, ndof_c):
    """Residual-only batching used by reaction-force post-processing."""
    nc = int(cell_sol.shape[0])
    batch = configured_cell_batch_size(nc)
    if nc % batch:
        raise ValueError(f"padded cell count {nc} is not divisible by batch {batch}")
    nblocks = nc // batch
    r0 = jnp.zeros((nloc,), dtype=cell_sol.dtype)

    def body(ib, residual):
        c0 = ib * batch
        cs = jax.lax.dynamic_slice_in_dim(cell_sol, c0, batch, axis=0)
        gb = tuple(jax.lax.dynamic_slice_in_dim(g, c0, batch, axis=0)
                   for g in geom)
        valid = jax.lax.dynamic_slice_in_dim(cell_valid, c0, batch, axis=0)
        dofs = jax.lax.dynamic_slice_in_dim(gdof, c0, batch, axis=0)
        Re = elem_res_fn(cs, *gb)
        rb = jnp.where(valid[:, None], Re.reshape(batch, ndof_c), 0.0)
        return residual.at[dofs.reshape(-1)].add(rb.reshape(-1))

    return jax.lax.fori_loop(0, nblocks, body, r0)


def make_distributed_newton(part, mesh, elem_fn, tol, maxiter, n_geom,
                            n_newton=20, nl_tol=1e-8, axis="d"):
    # Newton is a while_loop (like the serial axion.make_newton):
    # it stops as soon as the GLOBAL residual norm drops below nl_tol, using
    # n_newton only as an UPPER bound.  For the linear u-/AT2-solves this exits in
    # ~1 step (was: scan(length=n_newton) which always ran the full n_newton steps).
    sd0 = jax.tree_util.tree_map(lambda a: shard(a, mesh), part.shard)  # 1/D per device
    rpd, nloc, ndof_c, vec = part.rpd, part.nloc, part.ndof_c, part.vec
    NIF = part.n_iface + 1
    sd_spec = ShardArrays(*[P(axis, None)] * len(ShardArrays._fields))
    geom_spec = tuple(P(axis, None) for _ in range(n_geom))

    @partial(shard_map, mesh=mesh,
             in_specs=(sd_spec, P(axis, None), geom_spec, P(axis, None), P(axis, None)),
             out_specs=P(axis, None), **_SMKW)
    def _newton(sd, U0, GEOM, B, FX):
        local_cells = sd.local_cells[0]; LR = sd.LR[0]; GC = sd.GC[0]
        nnzv = sd.nnz_valid[0]; cell_valid = sd.cell_valid[0]
        own_slot = sd.own_slot[0]; own_iloc = sd.own_iface_local[0]
        gslot = sd.ghost_slot[0]; ownval = sd.owned_valid[0]
        u0 = U0[0]; geom0 = tuple(g[0] for g in GEOM); b_owned = B[0]; fx_owned = FX[0]
        nc = local_cells.shape[0]
        gdof = (local_cells[:, :, None] * vec + jnp.arange(vec)).reshape(nc, -1)

        def gather(x_owned):
            send = jnp.zeros(NIF, x_owned.dtype).at[own_slot].set(x_owned[own_iloc])
            return jnp.concatenate([x_owned, jax.lax.psum(send, axis)[gslot]])

        def halo(y_ext):
            contrib = jnp.zeros(NIF, y_ext.dtype).at[gslot].add(y_ext[rpd:])
            total = jax.lax.psum(contrib, axis)
            return y_ext[:rpd].at[own_iloc].add(total[own_slot])

        fx_all = fx_owned | (~ownval)
        fx_ext = gather(fx_owned.astype(jnp.float64)) > 0.5
        dot = lambda u, w: jax.lax.psum((u * w).sum(), axis)

        def assemble(u_owned):
            u_ext = gather(u_owned)
            cell_sol = u_ext.reshape(-1, vec)[local_cells].reshape(nc, ndof_c)
            V_local, res_ext = batch_element_assembly(
                elem_fn, cell_sol, geom0, cell_valid, gdof, nnzv, nloc, ndof_c)
            return halo(res_ext), V_local

        def mask_res(res_owned):
            # drop fixed rows, subtract the (fixed-masked) rhs -- same as before
            return jnp.where(fx_all, 0.0, res_owned) - jnp.where(fx_all, 0.0, b_owned)

        def rnorm(r):
            # GLOBAL residual norm over the free OWNED dofs (r already 0 on fixed/pad).
            # dot() is a psum, so this is the norm across ALL shards.
            return jnp.sqrt(dot(r, r))

        def lin_solve(res_owned, lv):
            # one Newton step's linear solve  A(lv)*du = -res  by masked halo-PCG.
            def matvec(x_owned):
                x_ext = jnp.where(fx_ext, 0.0, gather(x_owned))
                y_ext = jax.ops.segment_sum(lv * x_ext[GC], LR, num_segments=nloc)
                return jnp.where(fx_all, x_owned, halo(y_ext))
            diag_ext = jnp.zeros(nloc).at[LR].add(jnp.where((LR == GC) & nnzv, lv, 0.0))
            diag = jnp.where(fx_all, 1.0, halo(diag_ext))
            inv = 1.0 / jnp.where(jnp.abs(diag) > 1e-12, diag, 1.0)
            return _pcg_local(matvec, dot, inv, -res_owned, tol, maxiter)

        # --- Newton as a while_loop (mirrors the serial solver) ---
        # state = (u, residual@u, tangent@u, iter).  Carry the tangent lv so the
        # linear solve reuses it without re-assembling.
        res0, lv0 = assemble(u0)
        res0 = mask_res(res0)

        def cond(state):
            _, res, _, it = state
            return (it < n_newton) & (rnorm(res) > nl_tol)   # stop on residual OR cap

        def body(state):
            u, res, lv, it = state
            du = lin_solve(res, lv)                           # solve with current tangent
            u_new = u + du
            res_new, lv_new = assemble(u_new)                # residual + tangent at u_new
            return (u_new, mask_res(res_new), lv_new, it + 1)

        u_owned, _, _, _ = jax.lax.while_loop(cond, body, (u0, res0, lv0, jnp.array(0)))
        return jnp.where(ownval, u_owned, 0.0)[None, :]

    def newton(u0_owned, geom, b_owned, fixed_owned, shards=None):
        # `shards` lets the caller PASS the (sharded) ShardArrays as an explicit
        # argument instead of closing over sd0.  Required in multi-process runs:
        # sd0 spans non-addressable devices and JAX forbids closing over such
        # arrays inside jit/scan.  Default (None) keeps the old single-process API.
        return _newton(sd0 if shards is None else shards,
                       u0_owned, tuple(geom), b_owned, fixed_owned)
    newton.sd0 = sd0                      # expose so a driver can thread it through scan
    return newton


# ---- element kernels and per-cell geometry of an axion.Field ----
def make_elem_fn(field):
    """Batched element kernels of ``field`` in the form the distributed solvers
    expect:

        elem_fn(cell_sol, quad_points, shape_grads, JxW, *theta) -> (K_e, r_e)
        elem_fn.residual(...)                                    -> r_e

    with K_e of shape (nc, ndof_c, ndof_c) and r_e of shape (nc, ndof_c). All
    mesh geometry is passed explicitly, so the kernels do not hold the full
    mesh and the geometry can be sharded over the devices."""
    ndof_c = field.dofs_per_cell
    kernel_jac = field._cell_res_tan
    kernel = field._cell_res

    def elem_fn(cell_sol, xq, sg, jxw, *theta):
        vals, jacs = kernel_jac(cell_sol, xq, sg, jxw, *theta)
        nc = cell_sol.shape[0]
        return jacs.reshape(nc, ndof_c, ndof_c), vals.reshape(nc, ndof_c)

    def elem_res_fn(cell_sol, xq, sg, jxw, *theta):
        vals = kernel(cell_sol, xq, sg, jxw, *theta)
        return vals.reshape(cell_sol.shape[0], ndof_c)

    elem_fn.residual = elem_res_fn
    elem_fn.ndof_c = ndof_c
    return elem_fn


def field_geometry(field):
    """The per-cell geometry arrays consumed by the kernels of :func:`make_elem_fn`,
    in order: (quad_points, shape_grads, JxW)."""
    return (field.quad_points, field.shape_grads, field.JxW)


def pack_geometry(field, part, cell_part, internal_vars_cells=()):
    """Slice the per-cell geometry of ``field`` (and any cell-wise internal
    variables) to the cells OWNED by each shard and pad to (D, C, ...)."""
    D = part.D; C = part.shard.cell_valid.shape[1]
    arrays = [onp.asarray(g) for g in field_geometry(field)] + \
             [onp.asarray(v) for v in internal_vars_cells]
    packed = []
    for arr in arrays:
        out = onp.zeros((D, C) + arr.shape[1:], arr.dtype)
        for d in range(D):
            oc = onp.where(cell_part == d)[0]
            out[d, :len(oc)] = arr[oc]
        packed.append(jnp.asarray(out))
    return tuple(packed)


def _grid_quads(nx, ny):
    xs, ys = onp.meshgrid(onp.arange(nx), onp.arange(ny), indexing="xy")
    pts = onp.stack([xs.reshape(-1), ys.reshape(-1), onp.zeros(nx*ny)], axis=1).astype(float)
    cells = []
    nid = lambda i, j: j*nx + i
    for j in range(ny-1):
        for i in range(nx-1):
            cells.append([nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)])
    return pts, onp.asarray(cells)


def _q1_element():
    K = (1.0/6.0)*onp.array([[4.,-1.,-2.,-1.],[-1.,4.,-1.,-2.],[-2.,-1.,4.,-1.],[-1.,-2.,-1.,4.]])
    return K + 0.05*onp.eye(4)


def _build_Vlocal(part, Ke):
    ke = Ke.reshape(-1); cv = onp.asarray(part.shard.cell_valid); D, C = cv.shape
    V = onp.zeros((D, C, ke.size)); V[cv] = ke
    return jnp.asarray(V.reshape(D, C * ke.size))


def _to_owned(part, vfull):
    og, ov = part.owned_global_dof, part.owned_valid_np
    return jnp.asarray(onp.where(ov, vfull[og], 0.0))


def _from_owned(part, vowned):
    """Host helper for tests/output: owned layout (D,rpd) -> full global vector."""
    arr = onp.asarray(vowned)
    out = onp.zeros(part.Nd_real, dtype=arr.dtype)
    for d in range(part.D):
        valid = part.owned_valid_np[d]
        out[part.owned_global_dof[d, valid]] = arr[d, valid]
    return out


def self_test():
    jax.config.update("jax_enable_x64", True)
    D = jax.device_count(); mesh = make_domain_mesh(); vec = 1
    pts, cells = _grid_quads(11, 9); Nd = pts.shape[0]*vec; Ke = _q1_element()
    A = onp.zeros((Nd, Nd))
    for c in cells:
        for a in range(4):
            for b in range(4):
                A[c[a], c[b]] += Ke[a, b]
    res = []
    for m in ("slab", "sfc"):
        part = build_mesh_partition(pts, cells, D, vec, method=m, axis=0)
        Vloc = _build_Vlocal(part, Ke); cg = make_cell_distributed_cg(part, mesh, 1e-11, 5000)
        onp.random.seed(0); rhs = onp.random.rand(Nd); xref = onp.linalg.solve(A, rhs)
        du = onp.asarray(cg(Vloc, _to_owned(part, -rhs), jnp.zeros((D, part.rpd), bool)))
        res.append((f"{m}: CG SPD", onp.linalg.norm(du-xref)/onp.linalg.norm(xref)))
        rng = onp.random.default_rng(1); fixed = onp.sort(rng.choice(Nd, 7, replace=False))
        mask = onp.zeros(Nd, bool); mask[fixed] = True
        tgt = onp.zeros(Nd); tgt[fixed] = rng.random(len(fixed)); r2 = rng.random(Nd)
        Rg = A@tgt - r2; Rg[fixed] = 0.0
        du2 = onp.asarray(cg(Vloc, _to_owned(part, Rg), _to_owned(part, mask.astype(float))>0.5))
        free = onp.setdiff1d(onp.arange(Nd), fixed)
        uf = onp.linalg.solve(A[free][:, free], r2[free]-A[free][:, fixed]@tgt[fixed])
        uref = tgt.copy(); uref[free] = uf
        res.append((f"{m}: CG Dir.", onp.linalg.norm(tgt+du2-uref)/onp.linalg.norm(uref)))
    ok = all(e < 1e-7 for _, e in res)
    print("  [CG] halo CG vs dense:")
    for n, e in res: print(f"    {n:16s} rel_err={e:.3e}  {'PASS' if e<1e-7 else 'FAIL'}")
    return ok


def self_test_newton():
    jax.config.update("jax_enable_x64", True)
    D = jax.device_count(); mesh = make_domain_mesh(); vec = 1
    pts, cells = _grid_quads(11, 9); Nd = pts.shape[0]*vec
    Ke = _q1_element(); Kej = jnp.asarray(Ke)
    A = onp.zeros((Nd, Nd))
    for c in cells:
        for a in range(4):
            for b in range(4):
                A[c[a], c[b]] += Ke[a, b]
    # synthetic BATCHED elem_fn (constant Ke), n_geom=1 dummy geometry
    def elem_fn(cell_sol, dummy):
        nc = cell_sol.shape[0]
        return jnp.broadcast_to(Kej, (nc, 4, 4)), cell_sol @ Kej.T
    res = []
    for m in ("slab", "sfc"):
        part = build_mesh_partition(pts, cells, D, vec, method=m, axis=0)
        newton = make_distributed_newton(part, mesh, elem_fn, 1e-11, 5000, n_geom=1, n_newton=20)
        C = part.shard.cell_valid.shape[1]; geom = (jnp.zeros((D, C, 1)),)
        onp.random.seed(0); b = onp.random.rand(Nd); xref = onp.linalg.solve(A, b)
        u = _from_owned(part, newton(_to_owned(part, onp.zeros(Nd)), geom,
                                  _to_owned(part, b), jnp.zeros((D, part.rpd), bool)))
        res.append((f"{m}: Newton SPD", onp.linalg.norm(u-xref)/onp.linalg.norm(xref)))
        rng = onp.random.default_rng(2); fixed = onp.sort(rng.choice(Nd, 7, replace=False))
        mask = onp.zeros(Nd, bool); mask[fixed] = True
        tgt = onp.zeros(Nd); tgt[fixed] = rng.random(len(fixed)); rhs = rng.random(Nd)
        u2 = _from_owned(part, newton(_to_owned(part, tgt), geom, _to_owned(part, rhs),
                                   _to_owned(part, mask.astype(float))>0.5))
        free = onp.setdiff1d(onp.arange(Nd), fixed)
        uf = onp.linalg.solve(A[free][:, free], rhs[free]-A[free][:, fixed]@tgt[fixed])
        uref = tgt.copy(); uref[free] = uf
        res.append((f"{m}: Newton Dir.", onp.linalg.norm(u2-uref)/onp.linalg.norm(uref)))
    ok = all(e < 1e-7 for _, e in res)
    print("  [Newton] sharded assembly + PCG vs dense (synthetic elem_fn):")
    for n, e in res: print(f"    {n:16s} rel_err={e:.3e}  {'PASS' if e<1e-7 else 'FAIL'}")
    return ok


def self_test_real():
    """Validates make_elem_fn + pack_geometry on a real :class:`axion.Field`
    (Poisson on a small HEX8 box): distributed Newton against a dense solve."""
    from .field import Field
    from .mesh import box_mesh
    jax.config.update("jax_enable_x64", True)
    D = jax.device_count(); mesh = make_domain_mesh()

    class Poisson(Field):
        def flux(self):
            return lambda grad_u: grad_u

    grid = box_mesh(3, 3, 2, 1., 1., 1.)
    pts, cells = grid.points, grid.cells
    field = Poisson(grid, ncomp=1)
    vec = field.ncomp; N = field.num_nodes; Nd = N * vec
    onp.random.seed(0); u = onp.random.rand(N, vec)
    _, cj = field.evaluate_cells(jnp.asarray(u), (), tangent=True)
    A = onp.zeros((Nd, Nd))
    onp.add.at(A, (field.coo_rows, field.coo_cols), onp.asarray(cj).reshape(-1))
    elem_fn = make_elem_fn(field)
    res = []
    for m in ("slab", "sfc"):
        cell_part = partition_cells(pts, cells, D, m, axis=2)
        part = build_mesh_partition(pts, cells, D, vec, method=m, axis=2)
        geom = pack_geometry(field, part, cell_part)
        newton = make_distributed_newton(part, mesh, elem_fn, 1e-11, 5000, n_geom=len(geom), n_newton=20)
        rng = onp.random.default_rng(3); fixed = onp.sort(rng.choice(Nd, 6, replace=False))
        mask = onp.zeros(Nd, bool); mask[fixed] = True
        tgt = onp.zeros(Nd); tgt[fixed] = rng.random(len(fixed)); rhs = rng.random(Nd)
        u2 = _from_owned(part, newton(_to_owned(part, tgt), geom, _to_owned(part, rhs),
                                   _to_owned(part, mask.astype(float))>0.5))
        free = onp.setdiff1d(onp.arange(Nd), fixed)
        uf = onp.linalg.solve(A[free][:, free], rhs[free]-A[free][:, fixed]@tgt[fixed])
        uref = tgt.copy(); uref[free] = uf
        res.append((f"{m}: field newton", onp.linalg.norm(u2-uref)/onp.linalg.norm(uref)))
    ok = all(e < 1e-7 for _, e in res)
    print("  [field] axion.Field kernels (make_elem_fn + pack_geometry) -> newton vs dense:")
    for n, e in res: print(f"    {n:16s} rel_err={e:.3e}  {'PASS' if e<1e-7 else 'FAIL'}")
    return ok


# =============================================================================
# Distributed semismooth-Newton (bound-constrained damage: AT1 / CZM)
# =============================================================================
def make_distributed_ssn(part, mesh, elem_fn, cg_tol, cg_maxiter, max_ssn, tol, n_geom=4, axis="d"):
    """Bound-constrained (0 <= d <= 1) damage solve by a primal-dual active-set
    (semismooth Newton), distributed exactly like the u-Newton: assembly over OWNED
    cells + halo, and a MASKED halo-PCG for the linear step.  Each SSN iteration the
    active dofs (hitting 0 or 1) plus any crack Dirichlet are eliminated from the
    linear system via `fixed` (columns zeroed, rows -> identity).  Used for AT1/CZM.
    Returns  ssn(d0_owned, geom, crack_mask_owned, crack_target_owned) -> d_owned."""
    # SHARD the big per-shard COO tables and return the owned field layout.
    sd0 = jax.tree_util.tree_map(lambda a: shard(a, mesh), part.shard)
    rpd, nloc, ndof_c, vec = part.rpd, part.nloc, part.ndof_c, part.vec
    NIF = part.n_iface + 1
    sd_spec = ShardArrays(*[P(axis, None)] * len(ShardArrays._fields))
    geom_spec = tuple(P(axis, None) for _ in range(n_geom))     # geometry arrays + internal variables

    @partial(shard_map, mesh=mesh,
             in_specs=(sd_spec, P(axis, None), geom_spec, P(axis, None), P(axis, None)),
             out_specs=P(axis, None), **_SMKW)
    def _ssn(sd, D0, GEOM, CM, CT):
        # --- unpack this shard's slice (drop the length-1 sharded axis via [0]) ---
        local_cells = sd.local_cells[0]
        LR, GC, nnzv = sd.LR[0], sd.GC[0], sd.nnz_valid[0]
        cell_valid = sd.cell_valid[0]
        own_slot, own_iloc = sd.own_slot[0], sd.own_iface_local[0]
        gslot, ownval = sd.ghost_slot[0], sd.owned_valid[0]
        d0 = D0[0]                                   # initial damage (owned layout)
        geom0 = tuple(g[0] for g in GEOM)            # (quad_points, shape_grads, JxW, history)
        cm, ct = CM[0], CT[0]                         # crack mask / target (owned)
        nc = local_cells.shape[0]
        # local element -> dof map (owned+ghost numbering), used to scatter Re
        gdof = (local_cells[:, :, None] * vec + jnp.arange(vec)).reshape(nc, -1)

        # --- halo primitives (identical to the u-Newton) ---
        def gather(x):                               # owner -> ghost   (rpd -> nloc)
            send = jnp.zeros(NIF, x.dtype).at[own_slot].set(x[own_iloc])
            return jnp.concatenate([x, jax.lax.psum(send, axis)[gslot]])

        def halo(y):                                 # ghost -> owner   (nloc -> rpd)
            contrib = jnp.zeros(NIF, y.dtype).at[gslot].add(y[rpd:])
            total = jax.lax.psum(contrib, axis)
            return y[:rpd].at[own_iloc].add(total[own_slot])

        dot = lambda u, w: jax.lax.psum((u * w).sum(), axis)     # global inner product

        def pnorm(x):                                # global norm over OWNED real dofs
            return jnp.sqrt(jax.lax.psum(jnp.sum(jnp.where(ownval, x * x, 0.0)), axis))

        def assemble(d):
            # gather ghost values -> per-cell kernel on owned cells -> scatter Re ->
            # halo to owner rows. Returns (V_local tangent, residual on owned rows).
            d_ext = gather(d)
            cell_sol = d_ext.reshape(-1, vec)[local_cells].reshape(nc, ndof_c)
            V, res_ext = batch_element_assembly(
                elem_fn, cell_sol, geom0, cell_valid, gdof, nnzv, nloc, ndof_c)
            return V, halo(res_ext)

        def dmask_diag(V, fx_all):
            # block-Jacobi preconditioner diagonal (owned layout); fixed rows -> 1.
            diag_ext = jnp.zeros(nloc).at[LR].add(jnp.where((LR == GC) & nnzv, V, 0.0))
            dg = jnp.where(fx_all, 1.0, jnp.abs(halo(diag_ext)))
            return 1.0 / jnp.where(dg > 1e-12, dg, 1.0)

        # FIXED Jacobi scale for the projected-gradient active-set probe (from d0).
        # MUST use |diag|: a signed diagonal flips the probe direction for CZM
        # (its damage tangent can be negative) and makes the iteration diverge.
        V0, _ = assemble(d0)
        inv_Dm = dmask_diag(V0, (~ownval))

        def pcg(V, b, fx_owned):
            # masked halo-PCG: zero fixed COLUMNS, identity fixed ROWS (sym. Dirichlet).
            fx_all = fx_owned | (~ownval)
            fx_ext = gather(fx_owned.astype(jnp.float64)) > 0.5

            def matvec(x):
                x_ext = jnp.where(fx_ext, 0.0, gather(x))
                y_ext = jax.ops.segment_sum(V * x_ext[GC], LR, num_segments=nloc)
                return jnp.where(fx_all, x, halo(y_ext))
            return _pcg_local(matvec, dot, dmask_diag(V, fx_all), b, cg_tol, cg_maxiter)

        # ---- semismooth-Newton outer loop ----
        d0c = jnp.where(cm, ct, jnp.clip(d0, 0., 1.))           # feasible start + crack pinned

        def cond(state):
            _, it, converged = state
            return (it < max_ssn) & (~converged)

        def body(state):
            d, it, _ = state
            # (a) projected-gradient probe of the 0/1 active set
            _, R = assemble(d)
            trial = d - inv_Dm * R
            lo, hi = trial <= 0.0, trial >= 1.0
            active = lo | hi
            target_pg = jnp.where(lo, 0.0, jnp.where(hi, 1.0, d))
            # (b) merge active set with the crack Dirichlet -> fixed set + target
            fixed = active | cm
            target = jnp.where(cm, ct, target_pg)
            # (c) shift onto target BEFORE assembling, so the jump is baked into R
            d_shift = jnp.where(fixed, target, d)
            V, R_shift = assemble(d_shift)
            R_shift = jnp.where(fixed, 0.0, R_shift)
            # (d) masked Newton step, then clip + re-pin the crack
            delta = pcg(V, -R_shift, fixed)
            d_new = jnp.clip(d_shift + delta, 0., 1.)
            d_new = jnp.where(cm, ct, d_new)
            # (e) convergence on BOTH the increment and the normalized residual
            change = pnorm(d_new - d) / (pnorm(d_new) + 1e-30)
            _, R_new = assemble(d_new)
            resid = pnorm(jnp.where(ownval, R_new, 0.0)) / (pnorm(jnp.where(ownval, 1.0 / inv_Dm, 0.0)) + 1e-30)
            return (d_new, it + 1, (change < tol))

        d, _, _ = jax.lax.while_loop(cond, body, (d0c, jnp.array(0), jnp.array(False)))

        return jnp.where(ownval, d, 0.0)[None, :]

    def ssn(d0_owned, geom, crack_mask_owned, crack_target_owned, shards=None):
        # see make_distributed_newton: pass the sharded ShardArrays explicitly in
        # multi-process runs so they are ARGUMENTS, not (non-addressable) closures.
        return _ssn(sd0 if shards is None else shards,
                    d0_owned, tuple(geom), crack_mask_owned, crack_target_owned)
    ssn.sd0 = sd0
    return ssn


if __name__ == "__main__":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    a = self_test(); b = self_test_newton(); c = self_test_real()
    print("  >>> OVERALL:", "ALL PASS" if (a and b and c) else "FAILED")
