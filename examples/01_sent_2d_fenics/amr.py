import numpy as np
from collections import defaultdict

ROUND = 9


def _rk(p):
    return (round(float(p[0]), ROUND), round(float(p[1]), ROUND))


class PointPool:
    def __init__(self, pts):
        self.pts = [np.asarray(p, float) for p in pts]
        self.idx = {}
        for i, p in enumerate(self.pts):
            self.idx.setdefault(_rk(p), i)

    def get(self, p):
        k = _rk(p)
        j = self.idx.get(k)
        if j is None:
            j = len(self.pts)
            self.idx[k] = j
            self.pts.append(np.asarray(p, float))
        return j

    def array(self):
        return np.array(self.pts, float)


def split_quad(pool, cell):
    P = pool.array()
    sw, se, nw, ne = P[cell[0]], P[cell[1]], P[cell[2]], P[cell[3]]
    a, b, c, d = int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])
    m_s = pool.get(0.5 * (sw + se))
    m_w = pool.get(0.5 * (sw + nw))
    m_e = pool.get(0.5 * (se + ne))
    m_n = pool.get(0.5 * (nw + ne))
    mc = pool.get(0.25 * (sw + se + nw + ne))
    return [
        [a,   m_s, m_w, mc],
        [m_s, b,   mc,  m_e],
        [m_w, mc,  c,   m_n],
        [mc,  m_e, m_n, d],
    ]


_EDGE_ORDER = [(0, 1), (1, 3), (3, 2), (2, 0)]


def _edge_neighbor_pairs(pts, cells):
    sides = defaultdict(lambda: ([], []))
    for ci, cell in enumerate(cells):
        p = pts[cell]
        x0, x1 = p[:, 0].min(), p[:, 0].max()
        y0, y1 = p[:, 1].min(), p[:, 1].max()
        sides[(0, round(float(x0), ROUND))][0].append((y0, y1, ci))
        sides[(0, round(float(x1), ROUND))][1].append((y0, y1, ci))
        sides[(1, round(float(y0), ROUND))][0].append((x0, x1, ci))
        sides[(1, round(float(y1), ROUND))][1].append((x0, x1, ci))
    pairs = set()
    tol = 10.0 ** (-ROUND)
    for neg, pos in sides.values():
        for a0, a1, ca in neg:
            for b0, b1, cb in pos:
                if ca != cb and min(a1, b1) - max(a0, b0) > tol:
                    pairs.add(tuple(sorted((ca, cb))))
    return pairs


def balance_marks(pts, cells, level, marked):
    marked = set(int(m) for m in marked)
    pairs = _edge_neighbor_pairs(pts, cells)
    changed = True
    while changed:
        changed = False
        tgt = level.copy()
        for m in marked:
            tgt[m] = level[m] + 1
        for a, b in pairs:
            if abs(tgt[a] - tgt[b]) > 1:
                lo = a if tgt[a] < tgt[b] else b
                if lo not in marked:
                    marked.add(lo)
                    changed = True
    return marked


def refine(pts, cells, level, marked):
    marked = balance_marks(pts, cells, level, marked)
    pool = PointPool(pts)
    new_cells, new_level = [], []
    for ci, cell in enumerate(cells):
        if ci in marked:
            for child in split_quad(pool, cell):
                new_cells.append(child)
                new_level.append(level[ci] + 1)
        else:
            new_cells.append([int(v) for v in cell])
            new_level.append(int(level[ci]))
    return (pool.array(),
            np.array(new_cells, np.int64),
            np.array(new_level, int))


def find_hanging(pts, cells):
    key2idx = {_rk(p): i for i, p in enumerate(pts)}
    hanging = {}
    for cell in cells:
        for a, b in _EDGE_ORDER:
            ia, ib = int(cell[a]), int(cell[b])
            mid = 0.5 * (pts[ia] + pts[ib])
            j = key2idx.get(_rk(mid))
            if j is not None and j not in (ia, ib):
                hanging[int(j)] = (ia, ib)
    return hanging


class AMRState:
    def __init__(self, pts, cells):
        self.pts = np.asarray(pts, float)
        self.cells = np.asarray(cells, np.int64)
        self.level = self._levels_from_size()

    def _levels_from_size(self):
        h = self.cell_sizes()
        hmax = h.max()
        lvl = np.round(np.log2(hmax / np.maximum(h, 1e-30))).astype(int)
        return np.maximum(lvl, 0)

    def cell_sizes(self):
        c = self.pts[self.cells]
        dx = c[:, :, 0].max(1) - c[:, :, 0].min(1)
        dy = c[:, :, 1].max(1) - c[:, :, 1].min(1)
        return np.maximum(dx, dy)

    def refine(self, marked):
        self.pts, self.cells, self.level = refine(
            self.pts, self.cells, self.level, marked)

    def hanging(self):
        return find_hanging(self.pts, self.cells)

    def n_cells(self):
        return len(self.cells)


def mark_cells(energy_increment, damage, cell_size, hmin, alpha=0.8, d_threshold=0.5):
    eta = np.asarray(energy_increment, float)
    d = np.asarray(damage, float)
    marked = np.zeros(len(eta), dtype=bool)
    if len(eta):
        emax = np.max(eta)
        if emax > 0.0:
            marked |= eta > alpha * emax
    marked |= d > d_threshold
    marked &= np.asarray(cell_size) > hmin * 1.001
    return marked
