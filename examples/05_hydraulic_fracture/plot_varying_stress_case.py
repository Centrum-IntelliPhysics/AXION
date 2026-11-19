import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = os.path.dirname(os.path.abspath(__file__))


def scalar(saved, name):
    return saved[name].item()


def require_close(name, value, expected, atol=1e-12):
    if not np.isclose(float(value), float(expected), rtol=0.0, atol=atol):
        raise RuntimeError(f'{name}={value!r}; expected {expected!r}')


def load_case(output_dir):
    history_path = os.path.join(output_dir, 'notch_history.npz')
    if not os.path.exists(history_path):
        raise RuntimeError(f'Missing varying-stress history: {history_path}')
    with np.load(history_path, allow_pickle=False) as saved:
        required = {
            'completed', 'data', 'nt', 'dt', 'q_F', 'l0', 'theta_deg',
            'notch_layout', 'stress_profile', 'gravity_body_force_y',
            'sigx_top_mpa', 'sigx_bottom_mpa', 'sigy_top_mpa',
            'sigy_bottom_mpa', 'mesh_kind', 'mesh_rule', 'mesh_h_max',
            'element_type', 'source_rule', 'sig0_mode', 'history_rule',
            'solver_rule',
            'up_max_iter', 'up_tol',
        }
        missing = sorted(required - set(saved.files))
        if missing:
            raise RuntimeError(f'{history_path} lacks metadata: {missing}')
        if not bool(scalar(saved, 'completed')):
            raise RuntimeError('The varying-stress run is incomplete')
        if str(scalar(saved, 'stress_profile')) != 'fig19_linear':
            raise RuntimeError('Output does not use the Fig. 19 stress profile')
        if str(scalar(saved, 'notch_layout')) != 'single':
            raise RuntimeError('Figs. 19-21 require one central notch')
        if str(scalar(saved, 'element_type')) != 'TRI3':
            raise RuntimeError('Paper comparison requires TRI3 elements')
        if str(scalar(saved, 'mesh_kind')) != 'unstructured_gmsh':
            raise RuntimeError('Paper comparison requires the unstructured Gmsh mesh')
        if str(scalar(saved, 'mesh_rule')) != 'gmsh_quadrant_reflection_v1':
            raise RuntimeError('Output uses an obsolete mesh rule')
        if str(scalar(saved, 'source_rule')) != 'fixed_initial_notch_band_v1':
            raise RuntimeError('Output uses an incompatible notch source')
        if str(scalar(saved, 'sig0_mode')) != 'both':
            raise RuntimeError('Initial stress must act in stress and history terms')
        if str(scalar(saved, 'history_rule')) != (
                'physical_time_converged_history_v4'):
            raise RuntimeError(
                'Output predates the physical-time history correction')
        if str(scalar(saved, 'solver_rule')) != (
                'coupled_up_physical_history_phi_anderson_rms_v7'):
            raise RuntimeError(
                'Output predates the corrected nonlinear/history solver')
        if int(scalar(saved, 'up_max_iter')) != 50:
            raise RuntimeError('Paper comparison requires PFF_UP_MAX_ITER=50')
        require_close('up_tol', scalar(saved, 'up_tol'), 1.0e-6)
        require_close('theta_deg', scalar(saved, 'theta_deg'), 90.0)
        require_close('q_F', scalar(saved, 'q_F'), 10.0)
        require_close('l0', scalar(saved, 'l0'), 0.1)
        require_close('gravity_body_force_y',
                      scalar(saved, 'gravity_body_force_y'), -25.0e3)
        require_close('sigx_top_mpa', scalar(saved, 'sigx_top_mpa'), 1.0)
        require_close('sigx_bottom_mpa', scalar(saved, 'sigx_bottom_mpa'), 5.0)
        require_close('sigy_top_mpa', scalar(saved, 'sigy_top_mpa'), 2.0)
        require_close('sigy_bottom_mpa', scalar(saved, 'sigy_bottom_mpa'), 2.25)
        nt = int(scalar(saved, 'nt'))
        dt = float(scalar(saved, 'dt'))
        require_close('dt', dt, 0.05)
        require_close('t_end', nt * dt, 14.0)
        if float(scalar(saved, 'mesh_h_max')) > 0.05 + 1e-12:
            raise RuntimeError('Mesh is coarser than the paper h_max=0.05 m')
        history = np.asarray(saved['data'], dtype=float)
    if history.ndim != 2 or history.shape[1] < 5:
        raise RuntimeError(
            f'Expected time/length/pressure/upper/lower history, got {history.shape}')
    if len(history) != nt + 1 or not np.isclose(history[-1, 0], 14.0):
        raise RuntimeError('History does not contain every completed time step')
    return history, dt


def load_phase(output_dir, time_value, dt):
    step = int(round(time_value / dt))
    path = os.path.join(output_dir, 'vtk', f'step_{step:04d}.vtu')
    if not os.path.exists(path):
        raise RuntimeError(
            f'Missing t={time_value:g} s snapshot: {path}. '
            'Use PFF_SAVE_EVERY=20.')
    mesh = meshio.read(path)
    triangles = None
    for block in mesh.cells:
        if block.type == 'triangle':
            triangles = np.asarray(block.data, dtype=np.int32)
            break
    if triangles is None:
        raise RuntimeError(f'{path} contains no triangle cells')
    if 'sol' not in mesh.point_data:
        raise RuntimeError(f'{path} contains no phase-field point data')
    points = np.asarray(mesh.points[:, :2], dtype=float)
    phi = np.asarray(mesh.point_data['sol'], dtype=float).reshape(-1)
    return points, triangles, np.clip(phi, 0.0, 1.0)


def arrow(ax, start, end, scale=12, color='red', lw=1.4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=scale,
        lw=lw, color=color))


def draw_geometry(ax):
    ax.add_patch(Rectangle((-5.0, -5.0), 10.0, 10.0,
                           facecolor='#dff5fb', edgecolor='black', lw=1.5))
    ys = np.linspace(-4.5, 4.5, 11)
    for y in ys:
        magnitude = 3.0 - 0.4 * y
        length = 0.65 + 0.14 * magnitude
        arrow(ax, (-5.0 - length, y), (-5.05, y))
        arrow(ax, (5.0 + length, y), (5.05, y))
    for x in np.linspace(-4.5, 4.5, 11):
        arrow(ax, (x, 6.0), (x, 5.05))
        arrow(ax, (x, -6.0), (x, -5.05))

    ax.plot([0.0, 0.0], [-0.4, 0.4], color='black', lw=4,
            solid_capstyle='butt')
    ax.annotate('', xy=(-0.65, 0.4), xytext=(-0.65, -0.4),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    ax.text(-0.82, 0.0, '0.8 m', ha='right', va='center')
    ax.text(0.18, 0.0, 'notch', ha='left', va='center', fontweight='bold')
    arrow(ax, (-3.8, 3.6), (-3.8, 2.25), scale=15, color='blue', lw=2.2)
    ax.text(-3.55, 3.0, 'Gravity', ha='left', va='center')
    ax.text(0.0, 6.1, '2 MPa', ha='center', va='bottom')
    ax.text(0.0, -6.1, '2.25 MPa', ha='center', va='top')
    ax.text(-6.15, 4.8, '1 MPa', ha='right', va='center')
    ax.text(6.15, 4.8, '1 MPa', ha='left', va='center')
    ax.text(-6.15, -4.8, '5 MPa', ha='right', va='center')
    ax.text(6.15, -4.8, '5 MPa', ha='left', va='center')
    ax.set_xlim(-7.4, 7.4)
    ax.set_ylim(-7.2, 7.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Linearly varying stress boundary condition', pad=8)


def draw_phase(ax, points, triangles, phi, time_value):
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    ax.tripcolor(tri, phi, shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'$t={time_value:g}$ s', y=-0.10)


def make_figures(output_dir, history, dt, geometry_path, results_path):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    draw_geometry(ax)
    fig.tight_layout()
    fig.savefig(geometry_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    times = (4.0, 9.0, 14.0)
    fields = [load_phase(output_dir, value, dt) for value in times]
    fig = plt.figure(figsize=(12.0, 9.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.9),
                            hspace=0.28, wspace=0.12)
    for index, (time_value, field) in enumerate(zip(times, fields)):
        ax = fig.add_subplot(grid[0, index])
        draw_phase(ax, *field, time_value)

    ax = fig.add_subplot(grid[1, :])
    t = history[:, 0]
    upper = history[:, 3]
    lower = history[:, 4]
    ax.plot(t, upper, color='blue', lw=2.0, label='From the upper tip')
    ax.plot(t, lower, color='red', lw=2.0, label='From the lower tip')
    marker_ids = np.arange(0, len(t), max(int(round(1.0 / dt)), 1))
    ax.scatter(t[marker_ids], upper[marker_ids], color='blue', edgecolor='white',
               s=38, zorder=3)
    ax.scatter(t[marker_ids], lower[marker_ids], color='red', edgecolor='white',
               s=38, zorder=3)
    ax.set_xlim(3.0, 14.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('Fracture length increment [m]')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left')
    fig.suptitle('Hydraulic fracture under a linearly varying stress field',
                 y=0.98, fontsize=15, fontweight='bold')
    fig.savefig(results_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        default=os.environ.get(
            'PFF_OUTDIR', 'output_varying_stress'))
    parser.add_argument(
        '--geometry',
        default=os.path.join(HERE, 'varying_stress_case_geometry.png'))
    parser.add_argument(
        '--results',
        default=os.path.join(HERE, 'varying_stress_case_results.png'))
    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(HERE, output_dir)
    history, dt = load_case(output_dir)
    make_figures(output_dir, history, dt, args.geometry, args.results)
    print(f'Wrote {args.geometry}')
    print(f'Wrote {args.results}')


if __name__ == '__main__':
    main()
