import numpy as np

_QUAD = 3
_HEX = 5


def _elem_type(dim):
    return _QUAD if dim == 2 else _HEX


def _nnodes(dim):
    return 4 if dim == 2 else 8


def _read_msh22(lines, dim):
    etype = _elem_type(dim)
    i, n = 0, len(lines)
    node_tag, node_xyz = [], []
    cells = []
    while i < n:
        line = lines[i].strip()
        if line == "$Nodes":
            count = int(lines[i + 1])
            for k in range(count):
                parts = lines[i + 2 + k].split()
                node_tag.append(int(parts[0]))
                node_xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 2 + count
        elif line == "$Elements":
            count = int(lines[i + 1])
            for k in range(count):
                parts = lines[i + 2 + k].split()
                t = int(parts[1])
                if t != etype:
                    continue
                ntags = int(parts[2])
                conn = parts[3 + ntags:]
                cells.append([int(v) for v in conn])
            i += 2 + count
        else:
            i += 1
    return node_tag, np.asarray(node_xyz, float), cells


def _read_msh41(lines, dim):
    etype = _elem_type(dim)
    i, n = 0, len(lines)
    node_tag, node_xyz = [], []
    cells = []
    while i < n:
        line = lines[i].strip()
        if line == "$Nodes":
            hdr = lines[i + 1].split()
            n_blocks = int(hdr[0])
            j = i + 2
            for _ in range(n_blocks):
                b = lines[j].split()
                nb = int(b[3])
                tags = [int(lines[j + 1 + t]) for t in range(nb)]
                coords = [[float(v) for v in lines[j + 1 + nb + t].split()]
                          for t in range(nb)]
                node_tag.extend(tags)
                node_xyz.extend(coords)
                j += 1 + 2 * nb
            i = j
        elif line == "$Elements":
            hdr = lines[i + 1].split()
            n_blocks = int(hdr[0])
            j = i + 2
            for _ in range(n_blocks):
                b = lines[j].split()
                b_etype = int(b[2])
                nb = int(b[3])
                for t in range(nb):
                    parts = lines[j + 1 + t].split()
                    if b_etype == etype:
                        cells.append([int(v) for v in parts[1:]])
                j += 1 + nb
            i = j
        else:
            i += 1
    return node_tag, np.asarray(node_xyz, float), cells


def read_msh(path, dim):
    with open(path) as fh:
        lines = fh.readlines()
    ver = "2.2"
    for k, line in enumerate(lines):
        if line.strip() == "$MeshFormat":
            ver = lines[k + 1].split()[0]
            break
    if ver.startswith("2"):
        node_tag, node_xyz, cells = _read_msh22(lines, dim)
    elif ver.startswith("4"):
        node_tag, node_xyz, cells = _read_msh41(lines, dim)
    else:
        raise ValueError(f"unsupported .msh version {ver} (need 2.2 or 4.1 ASCII)")

    if not cells:
        raise ValueError(
            f"no {'quad' if dim == 2 else 'hex'} elements found in {path} — "
            f"was it meshed with the right dimension and recombined?")

    tag2idx = {t: k for k, t in enumerate(node_tag)}
    cells = np.array([[tag2idx[v] for v in c] for c in cells], dtype=np.int64)
    points = node_xyz[:, :dim].astype(np.float64)

    used = np.unique(cells)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    points = points[used]
    cells = remap[cells]
    return points, cells


def reorder_structured(points, cells, dim):
    cells = np.asarray(cells)
    pc = points[cells]
    c = pc.mean(axis=1, keepdims=True)
    gt = (pc > c).astype(np.int64)
    if dim == 2:
        key = gt[:, :, 1] * 2 + gt[:, :, 0]
    else:
        key = gt[:, :, 2] * 4 + gt[:, :, 1] * 2 + gt[:, :, 0]

    k = 2 ** dim
    if not np.array_equal(np.sort(key, axis=1), np.tile(np.arange(k), (len(cells), 1))):
        bad = np.where(~np.all(np.sort(key, axis=1) == np.arange(k), axis=1))[0]
        raise ValueError(
            f"{len(bad)} cell(s) are not structured {'quads' if dim == 2 else 'hexes'} "
            f"(first: cell {bad[0]}); runtime quad/hex AMR requires a "
            f"transfinite/structured mesh")
    order = np.argsort(key, axis=1)
    return np.take_along_axis(cells, order, axis=1)
