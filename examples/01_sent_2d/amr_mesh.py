"""Fixed-capacity, device-side quadtree AMR for JAX.

Design goals
------------
Every array in the mesh state has a FIXED shape (a capacity), so a jitted
kernel that consumes the mesh compiles exactly ONCE and is reused across all
refinement states -- there is no recompilation when the mesh changes, and the
whole refine/balance/hanging update can live INSIDE a ``jax.lax.scan``.

The two ideas that make this possible
-------------------------------------
1. Fixed capacity ("arena").  Nodes live in buffers of length ``N_cap`` and
   cells in buffers of length ``C_cap``.  The live counts (``n_nodes``,
   ``n_cells``) are scalars carried alongside; unused slots are inert padding.
   Refinement is APPEND-ONLY: children are written at the live cursor and the
   parent slot is flipped invalid.  Because phase-field damage is irreversible
   we never coarsen, so append-only never needs compaction.

2. Integer-lattice node identity.  Every node sits on the finest virtual grid
   at integer coordinates ``(I, J)`` with ``0 <= I <= NX``, ``0 <= J <= NY``
   where ``NX = NX0 * 2**LMAX``.  Its identity is the single integer key
   ``key = I * (NY + 1) + J``.  Two cells that share an edge produce the SAME
   midpoint key, so duplicate nodes are removed by a key comparison -- no
   persistent edge->midpoint registry is needed, and the whole thing is a
   sort / searchsorted, which JAX runs on device with static shapes.  Nodes on
   the upper face of a geometric slit receive a SIDE key offset, so coincident
   upper/lower crack-face nodes remain topologically distinct.

The mesh state is a flat NamedTuple of arrays so it can be a scan carry.

Coordinate / cell conventions
------------------------------
A leaf cell at refinement level ``lv`` spans ``s = 2**(LMAX - lv)`` integer
units.  It is identified by its lower-left corner ``(I, J)`` and ``lv``.  Its
four corner nodes, counter-clockwise, are::

    n3 (I,   J+s) ---- n2 (I+s, J+s)
      |                     |
    n0 (I,   J)   ---- n1 (I+s, J)

Refining it 1->4 creates 5 new candidate nodes (4 edge midpoints + 1 centre)
and 4 children of level ``lv + 1``.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as onp


# A sentinel key for "empty / invalid" node slots (larger than any real key).
INVALID_KEY = jnp.int64(1) << jnp.int64(60)


class MeshState(NamedTuple):
    """Fixed-shape mesh arena.  All arrays keep their shape for the whole run."""
    node_key:  jnp.ndarray   # (N_cap,)  int64  integer-lattice key, INVALID_KEY if empty
    node_xy:   jnp.ndarray   # (N_cap, 2) float physical coordinates
    n_nodes:   jnp.ndarray   # ()        int32  number of live nodes (append cursor)

    cell_nodes: jnp.ndarray  # (C_cap, 4) int32 compact corner-node ids per leaf
    cell_lv:    jnp.ndarray  # (C_cap,)   int32 refinement level
    cell_ll:    jnp.ndarray  # (C_cap, 2) int32 lower-left integer coordinate (I, J)
    cell_valid: jnp.ndarray  # (C_cap,)   bool  is this slot a live leaf?
    n_cells:    jnp.ndarray  # ()         int32 append cursor (live + dead-parent slots)


# -----------------------------------------------------------------------------
# Static geometry helpers (closed over by the jitted kernels; pure Python ints).
# -----------------------------------------------------------------------------
class Lattice(NamedTuple):
    """Static integer-lattice description.  Holds only Python ints / floats, so
    it can be closed over inside jit without being a traced value."""
    NX0: int
    NY0: int
    LMAX: int
    Lx: float
    Ly: float

    @property
    def NX(self) -> int:
        return self.NX0 * (2 ** self.LMAX)

    @property
    def NY(self) -> int:
        return self.NY0 * (2 ** self.LMAX)

    @property
    def SIDE(self) -> int:
        """Offset used to distinguish upper-face nodes of the geometric slit."""
        return (self.NX + 1) * (self.NY + 1)

    def key(self, I, J):
        """Integer-lattice key for node coordinate (I, J)."""
        return I.astype(jnp.int64) * jnp.int64(self.NY + 1) + J.astype(jnp.int64)

    def xy(self, I, J):
        """Physical coordinate for integer node coordinate (I, J)."""
        x = I.astype(jnp.float64) * (self.Lx / self.NX)
        y = J.astype(jnp.float64) * (self.Ly / self.NY)
        return jnp.stack([x, y], axis=-1)


# -----------------------------------------------------------------------------
# Node lookup: map a batch of integer keys to compact node ids (or -1 if absent)
# -----------------------------------------------------------------------------
def lookup_nodes(state: MeshState, query_keys):
    """Vectorised key -> node-id lookup against the live node table.

    Implemented as a sort of the live keys + ``searchsorted`` (both device ops,
    static shapes).  Returns the compact node id for each query key, or -1 when
    the key is not present.
    """
    live = jnp.arange(state.node_key.shape[0]) < state.n_nodes
    # Push dead slots to +inf so they never match and sort to the end.
    keys = jnp.where(live, state.node_key, INVALID_KEY)
    order = jnp.argsort(keys)                    # ascending; dead slots last
    sorted_keys = keys[order]

    pos = jnp.searchsorted(sorted_keys, query_keys.astype(jnp.int64))
    pos = jnp.clip(pos, 0, sorted_keys.shape[0] - 1)
    found = sorted_keys[pos] == query_keys.astype(jnp.int64)
    node_id = jnp.where(found, order[pos], -1)
    return node_id


# -----------------------------------------------------------------------------
# One refinement round (append-only, fixed capacity)
# -----------------------------------------------------------------------------
def refine(state: MeshState, marker, lat: Lattice,
           node_fields=(), cell_fields=(),
           new_node_cap: int = None):
    """Refine every marked LIVE leaf 1->4.  Append-only, shape-preserving.

    Parameters
    ----------
    marker : (C_cap,) bool
        Cells to refine.  Only slots that are both marked and live are acted on.
    node_fields : tuple of (N_cap, k) arrays
        P1 nodal fields; new nodes get the average of their edge endpoints
        (midpoints) or the cell corners (centres).
    cell_fields : tuple of (C_cap, k) arrays
        Per-cell fields; the four children inherit the parent value.
    new_node_cap : int
        Static upper bound on the number of DISTINCT new nodes created this
        round (needed by ``jnp.unique``'s ``size=``).  Defaults to ``5 * C_cap``.

    Returns
    -------
    new_state, new_node_fields, new_cell_fields
    """
    C_cap = state.cell_valid.shape[0]
    N_cap = state.node_key.shape[0]
    if new_node_cap is None:
        new_node_cap = 5 * C_cap

    do = marker & state.cell_valid                      # (C_cap,) cells to split
    lv = state.cell_lv                                  # (C_cap,)
    I0 = state.cell_ll[:, 0]
    J0 = state.cell_ll[:, 1]
    s = (jnp.int32(1) << (lat.LMAX - lv))               # (C_cap,) integer span
    h = s // 2                                          # half span (>=1 when lv<LMAX)

    # --- the 5 candidate new nodes of every cell (only meaningful where `do`) ---
    # integer coordinates of: mid01, mid12, mid23, mid30, centre
    cand_I = jnp.stack([I0 + h, I0 + s, I0 + h, I0,     I0 + h], axis=1)  # (C_cap, 5)
    cand_J = jnp.stack([J0,     J0 + h, J0 + s, J0 + h, J0 + h], axis=1)
    # Only nodes lying on the bottom edge of an upper slit-face cell receive
    # the SIDE offset.  Interior/top-edge nodes retain their normal key, which
    # avoids creating an artificial discontinuity above the slit after refine.
    has_side = jnp.any(
        state.node_key[state.cell_nodes] >= jnp.int64(lat.SIDE), axis=1)
    cand_side = has_side[:, None] & (cand_J == J0[:, None])
    cand_key = lat.key(cand_I, cand_J) + \
        cand_side.astype(jnp.int64) * jnp.int64(lat.SIDE)                  # (C_cap, 5)
    cand_key = jnp.where(do[:, None], cand_key, INVALID_KEY)

    flat_key = cand_key.reshape(-1)                                       # (5 C_cap,)

    # existing id for each candidate (>=0 if the node already exists)
    exist_id = lookup_nodes(state, flat_key)                             # (5 C_cap,)
    is_new = (flat_key != INVALID_KEY) & (exist_id < 0)

    # dedup the genuinely-new keys and assign fresh sequential compact ids
    new_only = jnp.where(is_new, flat_key, INVALID_KEY)
    uniq_keys, inv = jnp.unique(new_only, size=new_node_cap,
                                fill_value=INVALID_KEY, return_inverse=True)
    inv = inv.reshape(-1)
    # slot 0..(n_new-1) of uniq_keys are real; the INVALID_KEY fill sits somewhere.
    uniq_valid = uniq_keys != INVALID_KEY
    # rank of each unique real key among the reals -> its fresh local index
    fresh_local = jnp.cumsum(uniq_valid) - 1                             # (new_node_cap,)
    n_new = jnp.sum(uniq_valid).astype(jnp.int32)

    fresh_id_of_cand = state.n_nodes + fresh_local[inv]                  # (5 C_cap,)
    cand_id = jnp.where(is_new, fresh_id_of_cand,
                        jnp.where(exist_id >= 0, exist_id, 0)).astype(jnp.int32)
    cand_id = cand_id.reshape(C_cap, 5)                                  # (C_cap, 5)

    # --- write the new nodes into the arena (key + xy), append-only ---
    # For each fresh local index we need one representative (I, J).  Build it by
    # scattering candidate coordinates into fresh slots (duplicates overwrite the
    # same slot with identical data, so a plain set is fine).
    fresh_slot = jnp.where(is_new, fresh_local[inv], -1)                 # (5 C_cap,)
    flat_I = cand_I.reshape(-1)
    flat_J = cand_J.reshape(-1)
    safe_slot = jnp.where(fresh_slot >= 0, fresh_slot, new_node_cap)     # dump invalid at pad tail
    key_by_local = jnp.full((new_node_cap + 1,), INVALID_KEY, dtype=jnp.int64)
    I_by_local = jnp.zeros((new_node_cap + 1,), dtype=jnp.int32)
    J_by_local = jnp.zeros((new_node_cap + 1,), dtype=jnp.int32)
    key_by_local = key_by_local.at[safe_slot].set(flat_key)
    I_by_local = I_by_local.at[safe_slot].set(flat_I.astype(jnp.int32))
    J_by_local = J_by_local.at[safe_slot].set(flat_J.astype(jnp.int32))
    key_by_local = key_by_local[:new_node_cap]
    I_by_local = I_by_local[:new_node_cap]
    J_by_local = J_by_local[:new_node_cap]

    dst = state.n_nodes + jnp.arange(new_node_cap)                       # target slots
    in_range = dst < N_cap
    dst = jnp.where(in_range, dst, N_cap - 1)                            # clamp (assumes capacity ok)
    # Physical midpoint refinement, matching FEniCSx geometry transfer.  This
    # also preserves the small tapered notch instead of snapping new nodes back
    # to the zero-width integer lattice.
    cn0 = state.cell_nodes
    xy = state.node_xy
    cand_xy = jnp.stack([
        0.5 * (xy[cn0[:, 0]] + xy[cn0[:, 1]]),
        0.5 * (xy[cn0[:, 1]] + xy[cn0[:, 2]]),
        0.5 * (xy[cn0[:, 2]] + xy[cn0[:, 3]]),
        0.5 * (xy[cn0[:, 3]] + xy[cn0[:, 0]]),
        0.25 * (xy[cn0[:, 0]] + xy[cn0[:, 1]] + xy[cn0[:, 2]] + xy[cn0[:, 3]])
    ], axis=1).reshape(-1, 2)
    xy_store = jnp.zeros((new_node_cap + 1, 2))
    xy_store = xy_store.at[safe_slot].set(cand_xy)
    xy_by_local = xy_store[:new_node_cap]
    node_key = state.node_key.at[dst].set(
        jnp.where(key_by_local != INVALID_KEY, key_by_local, state.node_key[dst]))
    node_xy = state.node_xy.at[dst].set(
        jnp.where((key_by_local != INVALID_KEY)[:, None], xy_by_local, state.node_xy[dst]))
    n_nodes = state.n_nodes + n_new

    # --- transfer P1 node fields to the new nodes ---
    # midpoints: average of the two edge endpoints; centre: average of 4 corners.
    cn = state.cell_nodes                                               # (C_cap, 4)
    out_node_fields = []
    for f in node_fields:
        # value at each candidate = average of the relevant parent corners
        v_mid01 = 0.5 * (f[cn[:, 0]] + f[cn[:, 1]])
        v_mid12 = 0.5 * (f[cn[:, 1]] + f[cn[:, 2]])
        v_mid23 = 0.5 * (f[cn[:, 2]] + f[cn[:, 3]])
        v_mid30 = 0.5 * (f[cn[:, 3]] + f[cn[:, 0]])
        v_cen = 0.25 * (f[cn[:, 0]] + f[cn[:, 1]] + f[cn[:, 2]] + f[cn[:, 3]])
        v_cand = jnp.stack([v_mid01, v_mid12, v_mid23, v_mid30, v_cen], axis=1)  # (C_cap,5,k)
        v_flat = v_cand.reshape(-1, f.shape[1])
        val_by_local = jnp.zeros((new_node_cap + 1, f.shape[1]), dtype=f.dtype)
        val_by_local = val_by_local.at[safe_slot].set(v_flat)
        val_by_local = val_by_local[:new_node_cap]
        f_new = f.at[dst].set(
            jnp.where((key_by_local != INVALID_KEY)[:, None], val_by_local, f[dst]))
        out_node_fields.append(f_new)

    # --- build the 4 children and append them; flip parents invalid ---
    n0 = cn[:, 0]; n1 = cn[:, 1]; n2 = cn[:, 2]; n3 = cn[:, 3]
    m01, m12, m23, m30, cc = (cand_id[:, 0], cand_id[:, 1], cand_id[:, 2],
                              cand_id[:, 3], cand_id[:, 4])
    # child corner tables (each (C_cap, 4)); ordering matches n0..n3 convention
    child0 = jnp.stack([n0,  m01, cc,  m30], axis=1)
    child1 = jnp.stack([m01, n1,  m12, cc ], axis=1)
    child2 = jnp.stack([cc,  m12, n2,  m23], axis=1)
    child3 = jnp.stack([m30, cc,  m23, n3 ], axis=1)
    # child lower-left integer coords
    ll0 = jnp.stack([I0,     J0    ], axis=1)
    ll1 = jnp.stack([I0 + h, J0    ], axis=1)
    ll2 = jnp.stack([I0 + h, J0 + h], axis=1)
    ll3 = jnp.stack([I0,     J0 + h], axis=1)

    # each marked cell contributes 4 children; give them append offsets 0..3.
    # position of a marked cell among the marked ones:
    rank = (jnp.cumsum(do.astype(jnp.int32)) - 1)                       # (C_cap,)
    base = state.n_cells + 4 * rank                                     # first child slot
    n_ref = jnp.sum(do.astype(jnp.int32)).astype(jnp.int32)

    cell_nodes = state.cell_nodes
    cell_lv = state.cell_lv
    cell_ll = state.cell_ll
    cell_valid = state.cell_valid
    out_cell_fields = [c for c in cell_fields]

    for k, (childk, llk) in enumerate([(child0, ll0), (child1, ll1),
                                       (child2, ll2), (child3, ll3)]):
        dstc = jnp.where(do, base + k, C_cap - 1)                      # inert target if not marked
        put = do & (base + k < C_cap)
        cell_nodes = cell_nodes.at[dstc].set(
            jnp.where(put[:, None], childk, cell_nodes[dstc]))
        cell_lv = cell_lv.at[dstc].set(jnp.where(put, lv + 1, cell_lv[dstc]))
        cell_ll = cell_ll.at[dstc].set(jnp.where(put[:, None], llk, cell_ll[dstc]))
        cell_valid = cell_valid.at[dstc].set(jnp.where(put, True, cell_valid[dstc]))
        for ci, c in enumerate(cell_fields):
            out_cell_fields[ci] = out_cell_fields[ci].at[dstc].set(
                jnp.where(put[:, None] if c.ndim == 2 else put, c[dstc], out_cell_fields[ci][dstc]))

    # flip the refined parents invalid
    cell_valid = jnp.where(do, False, cell_valid)
    n_cells = state.n_cells + 4 * n_ref

    new_state = MeshState(node_key=node_key, node_xy=node_xy, n_nodes=n_nodes,
                          cell_nodes=cell_nodes, cell_lv=cell_lv, cell_ll=cell_ll,
                          cell_valid=cell_valid, n_cells=n_cells)
    return new_state, tuple(out_node_fields), tuple(out_cell_fields)


# -----------------------------------------------------------------------------
# 2:1 balance: no live leaf may be more than one level finer than its neighbour
# -----------------------------------------------------------------------------
def needs_balance(state: MeshState, lat: Lattice):
    """Mark live leaves that must be refined to restore 2:1 balance.

    A leaf is unbalanced if one of its edges carries a node that is itself the
    midpoint of a *finer* subdivision on the neighbour side -- i.e. the edge's
    own midpoint node exists AND one of the two half-edge midpoints also exists.
    That means the neighbour is two levels finer, which is illegal.
    """
    C_cap = state.cell_valid.shape[0]
    lv = state.cell_lv
    I0 = state.cell_ll[:, 0]; J0 = state.cell_ll[:, 1]
    s = (jnp.int32(1) << (lat.LMAX - lv))
    h = s // 2
    q = s // 4                                        # quarter span (0 if lv>=LMAX-1)
    has_side = jnp.any(
        state.node_key[state.cell_nodes] >= jnp.int64(lat.SIDE), axis=1)

    def full_key(I, J):
        side = has_side & (J == J0)
        return lat.key(I, J) + side.astype(jnp.int64) * jnp.int64(lat.SIDE)

    # For each of the 4 edges: does the edge midpoint exist, and does a quarter
    # point on that edge exist (=> neighbour is 2 levels finer)?
    def edge_unbalanced(mid_I, mid_J, qa_I, qa_J, qb_I, qb_J):
        mid = lookup_nodes(state, full_key(mid_I, mid_J)) >= 0
        qa = lookup_nodes(state, full_key(qa_I, qa_J)) >= 0
        qb = lookup_nodes(state, full_key(qb_I, qb_J)) >= 0
        return mid & (qa | qb)

    e0 = edge_unbalanced(I0 + h, J0,     I0 + q,     J0,     I0 + h + q, J0)      # bottom
    e1 = edge_unbalanced(I0 + s, J0 + h, I0 + s, J0 + q,     I0 + s, J0 + h + q)  # right
    e2 = edge_unbalanced(I0 + h, J0 + s, I0 + q,     J0 + s, I0 + h + q, J0 + s)  # top
    e3 = edge_unbalanced(I0,     J0 + h, I0,     J0 + q,     I0,     J0 + h + q)   # left

    # A cell can only be 2:1-violated if a neighbour is TWO levels finer, which
    # requires lv+2 <= LMAX, i.e. lv <= LMAX-2.  (For lv = LMAX-1 the quarter
    # span s//4 is 0 and the quarter-point test degenerates onto the corner
    # node -- which always exists -- spuriously flagging legal 1-level jumps
    # and marching refinement across the whole domain.)
    valid_and_refinable = state.cell_valid & (lv <= lat.LMAX - 2)
    return (e0 | e1 | e2 | e3) & valid_and_refinable


def balance(state, lat, node_fields=(), cell_fields=(), max_sweeps: int = 8,
            new_node_cap: int = None):
    """Repeatedly refine unbalanced leaves until the mesh is 2:1 balanced.

    A fixed ``max_sweeps`` bound keeps this jit/scan friendly.  Each sweep is a
    normal ``refine`` on the leaves flagged by ``needs_balance``.
    """
    def body(i, carry):
        st, nf, cf = carry
        mk = needs_balance(st, lat)
        st2, nf2, cf2 = refine(st, mk, lat, nf, cf, new_node_cap=new_node_cap)
        return (st2, nf2, cf2)

    return jax.lax.fori_loop(0, max_sweeps, body, (state, node_fields, cell_fields))


# -----------------------------------------------------------------------------
# Hanging nodes: a node that is an edge midpoint of a live leaf but sits on that
# leaf's (coarser) edge is constrained to the average of the two edge endpoints.
# -----------------------------------------------------------------------------
def hanging_constraints(state: MeshState, lat: Lattice, hang_cap: int):
    """Return fixed-capacity hanging-node constraints.

    For each live leaf edge we test whether its midpoint node exists in the
    table.  If it does, that midpoint is a hanging node whose value must equal
    the average of the edge's two corner nodes.

    Returns (hang_id, master_a, master_b, weight) each of shape (hang_cap,),
    padded with -1 ids / 0 weight.  Every hanging node has exactly two masters
    with weight 0.5, so a single (a, b) pair per row suffices.
    """
    C_cap = state.cell_valid.shape[0]
    cn = state.cell_nodes
    lv = state.cell_lv
    I0 = state.cell_ll[:, 0]; J0 = state.cell_ll[:, 1]
    s = (jnp.int32(1) << (lat.LMAX - lv))
    h = s // 2

    # 4 edges: (corner_a, corner_b, midpoint integer coord)
    edges_a = jnp.stack([cn[:, 0], cn[:, 1], cn[:, 2], cn[:, 3]], axis=1)   # (C_cap,4)
    edges_b = jnp.stack([cn[:, 1], cn[:, 2], cn[:, 3], cn[:, 0]], axis=1)
    mid_I = jnp.stack([I0 + h, I0 + s, I0 + h, I0    ], axis=1)
    mid_J = jnp.stack([J0,     J0 + h, J0 + s, J0 + h], axis=1)
    has_side = jnp.any(
        state.node_key[state.cell_nodes] >= jnp.int64(lat.SIDE), axis=1)
    mid_side = has_side[:, None] & (mid_J == J0[:, None])
    mid_key = lat.key(mid_I, mid_J) + \
        mid_side.astype(jnp.int64) * jnp.int64(lat.SIDE)
    mid_id = lookup_nodes(state, mid_key)                                   # (C_cap,4)

    is_hang = (mid_id >= 0) & state.cell_valid[:, None]
    # a midpoint that also happens to be a corner of THIS leaf is not hanging
    is_hang = is_hang & (mid_id != edges_a) & (mid_id != edges_b)

    flat_hang = is_hang.reshape(-1)
    n_flag = jnp.sum(flat_hang).astype(jnp.int32)   # true hanging count (2D: no dups)
    flat_hid = mid_id.reshape(-1)
    flat_a = edges_a.reshape(-1)
    flat_b = edges_b.reshape(-1)

    # gather the flagged rows into the first slots of fixed-size output arrays
    order = jnp.argsort(~flat_hang)              # True(=hanging) first
    sel = order[:hang_cap]
    keep = flat_hang[sel]
    hang_id = jnp.where(keep, flat_hid[sel], -1).astype(jnp.int32)
    master_a = jnp.where(keep, flat_a[sel], 0).astype(jnp.int32)
    master_b = jnp.where(keep, flat_b[sel], 0).astype(jnp.int32)
    weight = jnp.where(keep, 0.5, 0.0)
    return hang_id, master_a, master_b, weight, n_flag


# -----------------------------------------------------------------------------
# Compaction: reclaim dead-parent slots WITHOUT changing array shapes
# -----------------------------------------------------------------------------
def compact(state: MeshState, cell_fields=()):
    """Reclaim the slots of dead-parent cells so the buffer occupancy tracks the
    number of LIVE cells instead of the cumulative count.

    When a cell is refined it is split into four appended children and the
    original parent is marked invalid (a "dead parent"); in the append-only
    scheme its slot is never freed, so the append cursor keeps growing.  This
    routine gathers the live cells to the FRONT of the fixed-capacity cell
    buffers and turns every freed tail slot into an inert dummy cell.

    Crucially this is a pure REORDER within the SAME C_cap arrays -- the shapes
    never change -- so it triggers NO recompilation and can be called every step
    (device-side, inside the scan).  Node arrays are left untouched: refinement
    never orphans a node (a refined parent's corners are still corners of its
    children), so the node buffers carry no dead entries.  Any per-cell fields
    (e.g. the history) are reordered to match.  ``n_cells`` is reset to the live
    count, so subsequent refinement appends right after the live block.
    """
    C_cap = state.cell_valid.shape[0]
    # stable order: live cells (valid=True -> ~valid=0) sort to the front
    order = jnp.argsort(~state.cell_valid)
    n_live = jnp.sum(state.cell_valid).astype(jnp.int32)
    keep = jnp.arange(C_cap) < n_live                # front block = the live cells

    cell_nodes = jnp.where(keep[:, None], state.cell_nodes[order], 0)   # tail -> [0,0,0,0]
    cell_lv = jnp.where(keep, state.cell_lv[order], 0)
    cell_ll = jnp.where(keep[:, None], state.cell_ll[order], 0)
    cell_valid = keep                                # exactly the front n_live are live

    new_state = state._replace(cell_nodes=cell_nodes, cell_lv=cell_lv,
                               cell_ll=cell_ll, cell_valid=cell_valid,
                               n_cells=n_live)
    out_fields = tuple(
        jnp.where(keep[:, None] if f.ndim == 2 else keep, f[order], 0)
        for f in cell_fields)
    return new_state, out_fields


# -----------------------------------------------------------------------------
# Build an initial uniform coarse mesh (level 0) in the arena
# -----------------------------------------------------------------------------
def initial_mesh(lat: Lattice, C_cap: int, N_cap: int,
                 y_crack=None, x_tip=None, crack_half_opening=1e-4) -> MeshState:
    """Level-0 QUAD4 mesh with an optional side-tagged geometric slit.

    The slit is ``y=y_crack, x<x_tip``.  Nodes on its upper face are duplicated
    and upper cells are remapped to those duplicates.  No damage is seeded.
    """
    NX0, NY0, LMAX = lat.NX0, lat.NY0, lat.LMAX
    s0 = 2 ** LMAX                                   # level-0 cell span

    # nodes of the coarse grid, integer coords
    ii, jj = onp.meshgrid(onp.arange(NX0 + 1), onp.arange(NY0 + 1), indexing="xy")
    I = (ii.ravel() * s0).astype(onp.int64)
    J = (jj.ravel() * s0).astype(onp.int64)
    nkey = I * (lat.NY + 1) + J
    nx = I.astype(float) * (lat.Lx / lat.NX)
    n_band = int(round(0.1 * NY0))
    n_outer = (NY0 - 2 * n_band) // 2
    counts = (n_outer, n_band, n_band, NY0 - n_outer - 2 * n_band)
    bounds = (0.0, 0.4 - crack_half_opening, 0.5,
              0.6 + crack_half_opening, lat.Ly)
    y_rows = onp.concatenate([
        onp.linspace(bounds[k], bounds[k + 1], counts[k] + 1)[:-1]
        for k in range(4)
    ] + [onp.array([lat.Ly])])
    ny = y_rows[(J // s0).astype(onp.int64)]
    n_nodes0 = len(nkey)
    assert n_nodes0 <= N_cap, "N_cap too small for the coarse mesh"

    # node id from (col, row) on the coarse grid
    def nid(col, row):
        return row * (NX0 + 1) + col

    cells = []
    ll = []
    for row in range(NY0):
        for col in range(NX0):
            cells.append([nid(col, row), nid(col + 1, row),
                          nid(col + 1, row + 1), nid(col, row + 1)])
            ll.append([col * s0, row * s0])
    cells = onp.array(cells, dtype=onp.int32)
    ll = onp.array(ll, dtype=onp.int32)

    if y_crack is not None and x_tip is not None:
        SIDE = int(lat.SIDE)
        on_line = (onp.abs(ny - y_crack) < 1e-9) & (nx < x_tip - 1e-9)
        old = onp.where(on_line)[0]
        if len(old) == 0:
            raise ValueError(
                "no coarse nodes on the requested slit; Y_CRACK must lie on "
                "an initial-grid row and X_TIP must be inside the domain")
        upper_cells = onp.where(ny[cells].mean(axis=1) > y_crack)[0]
        new = onp.arange(n_nodes0, n_nodes0 + len(old))
        remap = {int(o): int(n) for o, n in zip(old, new)}
        nkey = onp.concatenate([nkey, nkey[old] + SIDE])
        nx = onp.concatenate([nx, nx[old]])
        # Match the FEniCSx Gmsh notch: +/-1e-4 mm at x=0, tapering
        # linearly to zero opening at the shared crack tip.
        opening = crack_half_opening * (1.0 - nx[old] / x_tip)
        ny_upper = ny[old] + opening
        ny[old] -= opening
        ny = onp.concatenate([ny, ny_upper])
        n_nodes0 = len(nkey)

        for cc in upper_cells:
            for local in range(4):
                old_id = int(cells[cc, local])
                if old_id in remap:
                    cells[cc, local] = remap[old_id]

    n_cells0 = len(cells)
    assert n_nodes0 <= N_cap and n_cells0 <= C_cap, \
        "capacity too small for the coarse mesh + geometric slit"

    node_key = onp.full(N_cap, int(INVALID_KEY), dtype=onp.int64)
    node_key[:n_nodes0] = nkey
    node_xy = onp.zeros((N_cap, 2))
    node_xy[:n_nodes0, 0] = nx
    node_xy[:n_nodes0, 1] = ny

    cell_nodes = onp.zeros((C_cap, 4), dtype=onp.int32)
    cell_nodes[:n_cells0] = cells
    cell_lv = onp.zeros(C_cap, dtype=onp.int32)
    cell_ll = onp.zeros((C_cap, 2), dtype=onp.int32)
    cell_ll[:n_cells0] = ll
    cell_valid = onp.zeros(C_cap, dtype=bool)
    cell_valid[:n_cells0] = True

    return MeshState(
        node_key=jnp.asarray(node_key), node_xy=jnp.asarray(node_xy),
        n_nodes=jnp.int32(n_nodes0),
        cell_nodes=jnp.asarray(cell_nodes), cell_lv=jnp.asarray(cell_lv),
        cell_ll=jnp.asarray(cell_ll), cell_valid=jnp.asarray(cell_valid),
        n_cells=jnp.int32(n_cells0))
