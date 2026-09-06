"""Plotting routines for simulation results and 3D terrain visualizations."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from rough_slope_sim.simulation import Trajectory, _get_terrain_normal
from rough_slope_sim.terrain import Terrain


def plot_ball_surface_closeup(
    terrain,
    ball_radius: float,
    ball_x: float,
    ball_y: float,
    window_factor: float = 2.5,
    dense_grid_size: int = 150,
):
    """Plots a 1:1 true-scale close-up by analytically rendering true-size grains inside the window."""
    win_size = window_factor * ball_radius
    x_min, x_max = ball_x - win_size, ball_x + win_size
    y_min, y_max = ball_y - win_size, ball_y + win_size

    # Local dense spatial grid
    sub_x = np.linspace(x_min, x_max, dense_grid_size)
    sub_y = np.linspace(y_min, y_max, dense_grid_size)
    sub_X, sub_Y = np.meshgrid(sub_x, sub_y, indexing="ij")

    # Local macroscopic incline baseline
    z_base = -sub_X * np.tan(np.radians(15.0))  # Reference incline

    # Determine local grit particle diameter d50
    # Query local elevation variance to extract physical d50
    sampled_z = terrain.get_height(sub_X.ravel(), sub_Y.ravel()).reshape((dense_grid_size, dense_grid_size))
    d50_local = max(float(np.ptp(sampled_z - z_base)), 0.001)

    # Generate analytical grains inside the local window based on true particle density
    grain_radius = d50_local / 2.0
    grain_area = np.pi * (grain_radius**2)
    window_area = (x_max - x_min) * (y_max - y_min)
    num_local_grains = int(np.clip(0.80 * window_area / grain_area, 50, 8000))

    rng = np.random.default_rng(42)
    gx = rng.uniform(x_min - grain_radius, x_max + grain_radius, size=num_local_grains)
    gy = rng.uniform(y_min - grain_radius, y_max + grain_radius, size=num_local_grains)
    gh = rng.uniform(0.8, 1.2, size=num_local_grains) * d50_local

    # Rasterize local paraboloid grains
    z_micro_local = np.zeros_like(sub_X)
    for i in range(num_local_grains):
        dist_sq = (sub_X - gx[i]) ** 2 + (sub_Y - gy[i]) ** 2
        r_sq = grain_radius**2
        profile = np.maximum(0.0, gh[i] * (1.0 - dist_sq / r_sq))
        z_micro_local = np.maximum(z_micro_local, profile)

    sub_Z = z_base + z_micro_local

    # Calculate exact contact height under ball footprint
    footprint_dist_sq = (sub_X - ball_x) ** 2 + (sub_Y - ball_y) ** 2
    contact_mask = footprint_dist_sq <= (ball_radius**2)
    z_contact = np.max(sub_Z[contact_mask]) if np.any(contact_mask) else np.max(sub_Z)

    ball_z = z_contact + ball_radius

    # 3D sphere mesh
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sphere_x = ball_x + ball_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = ball_y + ball_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = ball_z + ball_radius * np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(sub_X, sub_Y, sub_Z, cmap="gist_earth", alpha=0.9, edgecolor="none", antialiased=True)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color="crimson", alpha=0.95, edgecolor="darkred", lw=0.3)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(ball_z - win_size, ball_z + win_size)

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    p_val_est = terrain.p_value_rough if ball_y < 15.0 else terrain.p_value_smooth
    ax.set_title(f"Contact Close-up (R = {ball_radius * 10.0:.1f} mm | Grit ~ {p_val_est:.0f}P)")

    plt.tight_layout()
    return fig


def plot_terrain_3d(terrain: Terrain, *, quiver_skip: int = 28) -> Figure:
    """Generates 3D surface plot with sparse, clean surface normal vectors."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="summer", alpha=0.75, rstride=2, cstride=2)

    # Sparse grid sampling for normal vectors
    s = max(1, quiver_skip)
    x_sub = terrain.X[::s, ::s]
    y_sub = terrain.Y[::s, ::s]
    z_sub = terrain.Z[::s, ::s]

    nx_arr = np.zeros_like(x_sub)
    ny_arr = np.zeros_like(y_sub)
    nz_arr = np.zeros_like(z_sub)

    for i in range(x_sub.shape[0]):
        for j in range(x_sub.shape[1]):
            nx, ny, nz = _get_terrain_normal(terrain, float(x_sub[i, j]), float(y_sub[i, j]))
            nx_arr[i, j], ny_arr[i, j], nz_arr[i, j] = nx, ny, nz

    ax.quiver(
        x_sub,
        y_sub,
        z_sub,
        nx_arr,
        ny_arr,
        nz_arr,
        length=0.25,
        color="crimson",
        alpha=0.9,
        linewidth=1.0,
        label="Surface Normals",
    )

    ax.set_title("3D Terrain Surface with Sparse Normal Vectors")
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_zlabel("Z Height (cm)")
    ax.view_init(elev=35, azim=-50)
    return fig


def plot_trajectories_3d(terrain: Terrain, trajectories: list[Trajectory]) -> Figure:
    """Renders simulated trajectories over 3D rough terrain."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="summer", alpha=0.5)

    colors = cm.inferno(np.linspace(0, 1, max(len(trajectories), 1)))
    for k, traj in enumerate(trajectories):
        x_arr = traj.x if isinstance(traj, Trajectory) else traj[:, 1]
        y_arr = traj.y if isinstance(traj, Trajectory) else traj[:, 2]
        z_arr = traj.z if isinstance(traj, Trajectory) else np.zeros_like(x_arr)
        ax.plot(x_arr, y_arr, z_arr, "-", color=colors[k], linewidth=1.1, zorder=1)

    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_zlabel("Z Height (cm)")
    ax.view_init(elev=30, azim=-45)
    return fig


def plot_experiment_vs_sim_distribution(y_exp: np.ndarray, y_sim: np.ndarray, x_slice: float) -> Figure:
    """Plots cross-sectional Y distribution comparison at X slice."""
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_exp = y_exp[~np.isnan(y_exp)]
    valid_sim = y_sim[~np.isnan(y_sim)]

    ax.hist(valid_exp, bins=12, density=True, alpha=0.5, color="#1e88e5", label="Exp")
    ax.hist(valid_sim, bins=12, density=True, alpha=0.5, color="#ffb300", label="Sim")

    if len(valid_exp) > 3:
        g_e = np.linspace(min(valid_exp), max(valid_exp), 150)
        ax.plot(g_e, gaussian_kde(valid_exp)(g_e), color="#0d47a1", lw=2)
    if len(valid_sim) > 3:
        g_s = np.linspace(min(valid_sim), max(valid_sim), 150)
        ax.plot(g_s, gaussian_kde(valid_sim)(g_s), color="#ff6f00", lw=2)

    ax.set_title(f"Histogram at X = {x_slice:.1f} cm")
    ax.set_xlabel("Y Position (cm)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig


def plot_trajectories_and_three_slices(
    exp_trajs: list,
    sim_trajs: list,
    x_slices: list[float],
    is_dual: bool = False,
    interface_y: float = 11.5,
) -> Figure:
    """Multi-panel plot: 90° CW rotated trajectories and 3 cross-section histograms."""
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1.0])

    ax_traj = fig.add_subplot(gs[:, 0])

    for t in exp_trajs:
        x_arr = t[:, 1] if isinstance(t, np.ndarray) else t.x
        y_arr = t[:, 2] if isinstance(t, np.ndarray) else t.y
        ax_traj.plot(y_arr, x_arr, color="#1e88e5", alpha=0.25, lw=1.0)

    for t in sim_trajs:
        x_arr = t[:, 1] if isinstance(t, np.ndarray) else t.x
        y_arr = t[:, 2] if isinstance(t, np.ndarray) else t.y
        ax_traj.plot(y_arr, x_arr, color="#ffb300", alpha=0.20, lw=0.8)

    colors = ["#e53935", "#8e24aa", "#43a047"]
    for x_s, c in zip(x_slices, colors):
        ax_traj.axhline(x_s, color=c, linestyle="--", lw=1.8, label=f"Slice X={x_s:.1f} cm")

    if is_dual:
        ax_traj.axvline(
            interface_y, color="black", linestyle="-.", lw=2.0, label=f"Interface (Y={interface_y:.1f} cm)"
        )

    ax_traj.set_title("Trajectories (Rotated 90° CW: Downslope Downwards)")
    ax_traj.set_xlabel("Y Position (cm) [Lateral]")
    ax_traj.set_ylabel("X Position (cm) [Downslope]")
    ax_traj.invert_yaxis()
    ax_traj.grid(True, linestyle=":", alpha=0.5)
    ax_traj.legend(loc="upper right", fontsize=9)

    from rough_slope_sim.analysis import trajectories_at_x_slice

    for idx, (x_s, c) in enumerate(zip(x_slices, colors)):
        ax_slice = fig.add_subplot(gs[idx, 1])
        y_exp = trajectories_at_x_slice(exp_trajs, x_s)
        y_sim = trajectories_at_x_slice(sim_trajs, x_s)

        valid_e = y_exp[~np.isnan(y_exp)]
        valid_s = y_sim[~np.isnan(y_sim)]

        if len(valid_e) > 0:
            ax_slice.hist(valid_e, bins=12, density=True, alpha=0.4, color="#1e88e5", label="Exp")
        if len(valid_s) > 0:
            ax_slice.hist(valid_s, bins=12, density=True, alpha=0.4, color="#ffb300", label="Sim")

        if is_dual:
            ax_slice.axvline(interface_y, color="black", linestyle="-.", lw=1.2)

        ax_slice.set_title(f"Slice at X = {x_s:.1f} cm", color=c, fontweight="bold")
        ax_slice.set_xlabel("Y Position (cm)")
        ax_slice.set_ylabel("Density")
        ax_slice.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_surface_center_analysis(
    exp_trajs: list, sim_trajs: list, center_y: float = 11.5, x_eval: float = 15.0
) -> Figure:
    """Center line comparison for dual-grit surfaces."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    from rough_slope_sim.analysis import trajectories_at_x_slice

    y_exp = trajectories_at_x_slice(exp_trajs, x_eval)
    y_sim = trajectories_at_x_slice(sim_trajs, x_eval)

    mean_exp, median_exp = np.mean(y_exp), np.median(y_exp)
    mean_sim, median_sim = np.mean(y_sim), np.median(y_sim)

    ax.axhline(center_y, color="black", linestyle="-.", label=f"Interface Boundary (Y={center_y:.1f} cm)")
    ax.axhline(mean_exp, color="#1e88e5", linestyle="--", label=f"Exp Mean Y={mean_exp:.2f} cm")
    ax.axhline(median_exp, color="#0d47a1", linestyle=":", label=f"Exp Median Y={median_exp:.2f} cm")
    ax.axhline(mean_sim, color="#ffb300", linestyle="--", label=f"Sim Mean Y={mean_sim:.2f} cm")

    ax.set_title(f"Dual-Surface Trajectory Center Analysis (X = {x_eval:.1f} cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig
