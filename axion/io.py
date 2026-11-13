"""Output of fields for visualization."""
import os

import numpy as onp


def write_vtu(path, mesh, point_data=None, cell_data=None):
    """Write nodal and cell-wise fields on ``mesh`` to a VTU file (ParaView).

    point_data : dict name -> array (num_nodes,) or (num_nodes, k)
    cell_data : dict name -> array (num_cells,) or (num_cells, k)
    """
    import meshio

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    points = onp.asarray(mesh.points)
    if points.shape[1] == 2:                       # VTU stores three coordinates
        points = onp.concatenate([points, onp.zeros((len(points), 1))], axis=1)
    out = meshio.Mesh(points=points, cells={mesh.meshio_cell_type: mesh.cells})
    for name, values in (point_data or {}).items():
        out.point_data[name] = onp.asarray(values, dtype=onp.float32)
    for name, values in (cell_data or {}).items():
        out.cell_data[name] = [onp.asarray(values, dtype=onp.float32)]
    out.write(path)
