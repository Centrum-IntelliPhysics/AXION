"""Geometry and computed phase field at t = 2.5 s of the two-notch case
(Zhou et al. 2023, Fig. 18). Reads the output folder of hydraulic_fracture.py."""

import argparse
import hashlib
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = os.path.dirname(os.path.abspath(__file__))


def scalar(saved, name):
    return saved[name].item()


def require_close(name, value, expected, atol=1e-12):
    if not np.isclose(float(value), float(expected), rtol=0.0, atol=atol):
        raise RuntimeError(f'{name}={value!r}; expected {expected!r} for Fig. 18')


def checkerboard_mesh(nx, ny, element_type):
    xs = np.linspace(-5.0, 5.0, nx + 1)
    ys = np.linspace(-5.0, 5.0, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    points = np.column_stack((xx.ravel(), yy.ravel()))
    nxp = nx + 1
    triangles = []

    def node(i, j):
        return j * nxp + i

    for j in range(ny):
        for i in range(nx):
            n00, n10 = node(i, j), node(i + 1, j)
            n11, n01 = node(i + 1, j + 1), node(i, j + 1)
            if element_type == 'TRI3':
                if (i + j) % 2 == 0:
                    triangles.extend(((n00, n10, n11), (n00, n11, n01)))
                else:
                    triangles.extend(((n00, n10, n01), (n10, n11, n01)))
            elif element_type == 'QUAD4':
                triangles.extend(((n00, n10, n11), (n00, n11, n01)))
            else:
                raise RuntimeError(f'Unsupported element_type={element_type!r}')
    return points, np.asarray(triangles, dtype=np.int32)


def mesh_fingerprint(points, cells):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(cells, dtype=np.int32).tobytes())
    return digest.hexdigest()


def reflection_node_maps(points):
    lookup = {
        (round(float(x), 12), round(float(y), 12)): index
        for index, (x, y) in enumerate(points)
    }
    maps = []
    for sx, sy in ((1.0, 1.0), (-1.0, 1.0),
                   (1.0, -1.0), (-1.0, -1.0)):
        maps.append([
            lookup[(round(float(sx * x), 12), round(float(sy * y), 12))]
            for x, y in points
        ])
    return np.asarray(maps, dtype=np.int32)


def load_completed_case(output_dir):
    checkpoint_path = os.path.join(output_dir, 'state_checkpoint.npz')
    history_path = os.path.join(output_dir, 'notch_history.npz')
    if not os.path.exists(checkpoint_path) or not os.path.exists(history_path):
        raise RuntimeError(
            f'Fig. 18 output is incomplete: expected {checkpoint_path} and {history_path}')

    with np.load(history_path, allow_pickle=False) as history:
        required_history = {
            'completed', 'notch_layout', 'notch_spacing', 'notch_centers',
            'nt', 'dt', 'q_F', 'sig_x0_mpa', 'sig_y0_mpa', 'sig0_sign',
            'sig0_mode', 'theta_deg', 'l0', 'mesh_h_max', 'element_type',
            'mesh_kind', 'mesh_rule', 'mesh_fingerprint',
            'reflection_symmetry_rule', 'source_rule',
            'source_half_width',
        }
        missing = sorted(required_history - set(history.files))
        if missing:
            raise RuntimeError(f'{history_path} lacks Fig. 18 metadata: {missing}')
        if not bool(scalar(history, 'completed')):
            raise RuntimeError(f'{history_path} is marked incomplete')
        if str(scalar(history, 'notch_layout')) != 'double_horizontal':
            raise RuntimeError('Output is not the double_horizontal Fig. 18 case')
        require_close('notch_spacing', scalar(history, 'notch_spacing'), 1.0)
        require_close('theta_deg', scalar(history, 'theta_deg'), 0.0)
        require_close('sig_x0_mpa', scalar(history, 'sig_x0_mpa'), 0.5)
        require_close('sig_y0_mpa', scalar(history, 'sig_y0_mpa'), 0.5)
        require_close('sig0_sign', scalar(history, 'sig0_sign'), -1.0)
        require_close('q_F', scalar(history, 'q_F'), 10.0)
        require_close('l0', scalar(history, 'l0'), 0.1)
        if str(scalar(history, 'sig0_mode')) != 'both':
            raise RuntimeError('Fig. 18 requires both initial-stress terms (sig0_mode=both)')
        mesh_kind = str(scalar(history, 'mesh_kind'))
        if mesh_kind != 'unstructured_gmsh':
            raise RuntimeError(
                'Fig. 18 validation requires the paper-like unstructured Gmsh TRI3 mesh; '
                f'got mesh_kind={mesh_kind!r}')
        expected_mesh_rule = 'gmsh_quadrant_reflection_v1'
        if str(scalar(history, 'mesh_rule')) != expected_mesh_rule:
            raise RuntimeError('Fig. 18 output uses an obsolete mesh-generation rule')
        expected_symmetry_rule = 'scalar_even_vector_reflection_xy_v1'
        if str(scalar(history, 'reflection_symmetry_rule')) != expected_symmetry_rule:
            raise RuntimeError(
                'Fig. 18 output was not solved on the reflection-symmetric branch')
        expected_mesh_fingerprint = str(scalar(history, 'mesh_fingerprint'))
        expected_source = 'fixed_double_horizontal_rounded_notch_band_spacing_1_v2'
        if str(scalar(history, 'source_rule')) != expected_source:
            raise RuntimeError(
                'Two-notch output uses an obsolete or incompatible injection region')
        source_half_width = float(scalar(history, 'source_half_width'))
        if source_half_width <= 0.0:
            raise RuntimeError(
                f'Invalid source_half_width={source_half_width:g} m')
        nt = int(scalar(history, 'nt'))
        dt = float(scalar(history, 'dt'))
        require_close('t_end', nt * dt, 2.5)
        h_max = float(scalar(history, 'mesh_h_max'))
        if h_max > 0.05 + 1e-12:
            raise RuntimeError(
                f'mesh_h_max={h_max:g} m exceeds the paper value 0.05 m')

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        required_checkpoint = {
            'step', 'nt', 'element_type', 'mesh_kind', 'mesh_rule',
            'reflection_symmetry_rule',
            'mesh_fingerprint', 'sol_d'}
        missing = sorted(required_checkpoint - set(checkpoint.files))
        if missing:
            raise RuntimeError(f'{checkpoint_path} lacks arrays/metadata: {missing}')
        step = int(scalar(checkpoint, 'step'))
        checkpoint_nt = int(scalar(checkpoint, 'nt'))
        if step != checkpoint_nt or step != nt:
            raise RuntimeError(
                f'Checkpoint stops at step {step}; completed Fig. 18 requires step {nt}')
        element_type = str(scalar(checkpoint, 'element_type'))
        if element_type != 'TRI3':
            raise RuntimeError(f'Fig. 18 requires TRI3, got {element_type!r}')
        if str(scalar(checkpoint, 'mesh_kind')) != mesh_kind:
            raise RuntimeError('Checkpoint and history use different mesh kinds')
        if str(scalar(checkpoint, 'mesh_rule')) != expected_mesh_rule:
            raise RuntimeError('Checkpoint uses an obsolete mesh-generation rule')
        if str(scalar(checkpoint, 'reflection_symmetry_rule')) != expected_symmetry_rule:
            raise RuntimeError('Checkpoint was not solved on the symmetric branch')
        if str(scalar(checkpoint, 'mesh_fingerprint')) != expected_mesh_fingerprint:
            raise RuntimeError('Checkpoint and history use different meshes')
        phi = np.asarray(checkpoint['sol_d'], dtype=float).reshape(-1)

    mesh_path = os.path.join(output_dir, 'mesh.npz')
    if not os.path.exists(mesh_path):
        raise RuntimeError(f'Missing unstructured Fig. 18 mesh: {mesh_path}')
    with np.load(mesh_path, allow_pickle=False) as saved_mesh:
        required_mesh = {
            'points', 'cells', 'mesh_kind', 'mesh_rule', 'mesh_h_target'}
        missing = sorted(required_mesh - set(saved_mesh.files))
        if missing:
            raise RuntimeError(f'{mesh_path} lacks mesh arrays/metadata: {missing}')
        if str(scalar(saved_mesh, 'mesh_kind')) != mesh_kind:
            raise RuntimeError(f'{mesh_path} has the wrong mesh kind')
        if str(scalar(saved_mesh, 'mesh_rule')) != expected_mesh_rule:
            raise RuntimeError(f'{mesh_path} uses an obsolete generation rule')
        points = np.asarray(saved_mesh['points'], dtype=float)
        triangles = np.asarray(saved_mesh['cells'], dtype=np.int32)
    if mesh_fingerprint(points, triangles) != expected_mesh_fingerprint:
        raise RuntimeError(f'{mesh_path} does not match the completed solver state')
    if phi.shape != (len(points),):
        raise RuntimeError(
            f'Phase field has shape {phi.shape}; expected {(len(points),)}')
    if not np.all(np.isfinite(phi)):
        raise RuntimeError('Phase field contains non-finite values')
    node_maps = reflection_node_maps(points)
    symmetry_error = float(np.max(np.abs(phi[node_maps] - phi[None, :])))
    if symmetry_error > 1e-10:
        raise RuntimeError(
            f'Completed Fig. 18 phase field broke reflection symmetry: '
            f'max error={symmetry_error:.3e}')
    return points, triangles, np.clip(phi, 0.0, 1.0), step * dt


def boundary_arrows(ax):
    positions = np.linspace(-4.5, 4.5, 11)
    arrow = dict(arrowstyle='-|>', mutation_scale=12, lw=1.4, color='red')
    for value in positions:
        ax.add_patch(FancyArrowPatch((value, 5.9), (value, 5.05), **arrow))
        ax.add_patch(FancyArrowPatch((value, -5.9), (value, -5.05), **arrow))
        ax.add_patch(FancyArrowPatch((-5.9, value), (-5.05, value), **arrow))
        ax.add_patch(FancyArrowPatch((5.9, value), (5.05, value), **arrow))


def draw_geometry(ax):
    ax.add_patch(Rectangle((-5.0, -5.0), 10.0, 10.0,
                           facecolor='#dff5fb', edgecolor='black', lw=1.4))
    boundary_arrows(ax)
    for yc in (-0.5, 0.5):
        ax.plot([-0.4, 0.4], [yc, yc], color='black', lw=3.0,
                solid_capstyle='butt')

    ax.annotate('', xy=(0.4, 1.15), xytext=(-0.4, 1.15),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax.text(0.0, 1.32, '0.8 m', ha='center', va='bottom')
    ax.annotate('', xy=(0.75, 0.5), xytext=(0.75, -0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax.text(0.92, 0.0, '1 m', ha='left', va='center')
    ax.text(0.0, 0.72, 'notch', ha='center', va='bottom', fontweight='bold')
    ax.text(0.0, -0.72, 'notch', ha='center', va='top', fontweight='bold')
    ax.text(0.0, 6.05, '0.5 MPa', ha='center', va='bottom')
    ax.text(0.0, -6.05, '0.5 MPa', ha='center', va='top')
    ax.text(-6.05, 0.0, '0.5 MPa', ha='right', va='center')
    ax.text(6.05, 0.0, '0.5 MPa', ha='left', va='center')
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-7.0, 7.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Geometry and boundary condition', y=-0.08)


def draw_phase(ax, points, triangles, phi, time_value):
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    ax.tripcolor(triangulation, phi, shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'(b) Phase field at $t={time_value:g}$ s', y=-0.08)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        default=os.environ.get('PFF_OUTDIR', 'output_two_notch'))
    parser.add_argument(
        '--figure', default=os.path.join(HERE, 'two_notch_case.png'))
    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(HERE, output_dir)

    points, triangles, phi, time_value = load_completed_case(output_dir)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), gridspec_kw={'wspace': 0.14})
    draw_geometry(axes[0])
    draw_phase(axes[1], points, triangles, phi, time_value)
    fig.suptitle('Fig. 18  Fracture propagation from two horizontal notches',
                 y=0.04, fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.025, right=0.99, top=0.98, bottom=0.15)
    fig.savefig(args.figure, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {args.figure}')


if __name__ == '__main__':
    main()
