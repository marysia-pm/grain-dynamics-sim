"""Plotting routines for simulation results, 3D terrain visualizations, and Nanovea benchmark panels."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from .analysis import trajectories_at_x_slice
from .simulation import Trajectory
from .terrain import NANOVEA_DATA, Terrain, nanovea_d50_from_grit


def plot_calibrated_sandpaper_panel(
    patch_size_cm: float = 0.20, grid_res: int = 500, seed: int = 42
) -> Figure:
    """Generates a 2x5 comparison figure across all standard Nanovea benchmark grit levels."""
    fig = plt.figure(figsize=(22, 9))
    grits = list(NANOVEA_DATA.keys())

    for col, target_grit in enumerate(grits):
        target_diam_um = NANOVEA_DATA[target_grit]["d50"] * 10000.0
        target_Sa_um = NANOVEA_DATA[target_grit]["sa"] * 10000.0

        x_grid = np.linspace(0, patch_size_cm, grid_res)
        y_grid = np.linspace(0, patch_size_cm, grid_res)
        X_m, Y_m = np.meshgrid(x_grid, y_grid)

        rng = np.random.default_rng(seed + col)
        z_local = rng.normal(0, NANOVEA_DATA[target_grit]["sa"], size=(grid_res, grid_res))

        ax1 = fig.add_subplot(2, 5, col + 1)
        ax1.imshow(
            z_local * 10.0,
            extent=[0, patch_size_cm * 10, 0, patch_size_cm * 10],
            origin="lower",
            cmap="terrain",
        )
        ax1.set_title(
            f"Simulated P{target_grit}\nFixed Patch: {patch_size_cm * 10:.2f} mm | Diam: {target_diam_um:.1f} µm\n$S_a$: {target_Sa_um:.2f} µm",
            fontsize=9,
        )
        if col == 0:
            ax1.set_ylabel("y (mm)")
        ax1.set_xlabel("x (mm)")

        ax2 = fig.add_subplot(2, 5, col + 6, projection="3d")
        ds = 4
        ax2.plot_surface(
            X_m[::ds, ::ds] * 10,
            Y_m[::ds, ::ds] * 10,
            z_local[::ds, ::ds] * 10,
            cmap="terrain",
            linewidth=0,
            antialiased=True,
        )
        ax2.set_title(f"3D Topography (P{target_grit})", fontsize=9)
        if col == 0:
            ax2.set_zlabel("Z (mm)")
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Y (mm)")

        ax2.set_box_aspect((1, 1, 0.35))
        ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()
    return fig


def _rasterize_grains(
    sub_X: np.ndarray,
    sub_Y: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gh: np.ndarray,
    grain_radius: float,
    batch_size: int = 500,
) -> np.ndarray:
    """Vectorized max-height rasterization of overlapping paraboloid grains."""
    r_sq = grain_radius**2
    flat_x = sub_X.ravel()
    flat_y = sub_Y.ravel()
    z_flat = np.zeros_like(flat_x)

    n = len(gx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dx = flat_x[None, :] - gx[start:end, None]
        dy = flat_y[None, :] - gy[start:end, None]
        dist_sq = dx * dx + dy * dy
        profile = np.maximum(0.0, gh[start:end, None] * (1.0 - dist_sq / r_sq))
        np.maximum(z_flat, profile.max(axis=0), out=z_flat)

    return z_flat.reshape(sub_X.shape)


def plot_ball_surface_closeup(
    terrain: Terrain,
    ball_radius: float,
    ball_x: float,
    ball_y: float,
    window_factor: float = 2.5,
    dense_grid_size: int = 150,
) -> Figure:
    """Plots a 1:1 true-scale close-up by rendering true-size grains inside the window."""
    win_size = window_factor * ball_radius
    x_min, x_max = ball_x - win_size, ball_x + win_size
    y_min, y_max = ball_y - win_size, ball_y + win_size

    sub_x = np.linspace(x_min, x_max, dense_grid_size)
    sub_y = np.linspace(y_min, y_max, dense_grid_size)
    sub_X, sub_Y = np.meshgrid(sub_x, sub_y, indexing="ij")

    z_base = -sub_X * np.tan(np.radians(terrain.slope_angle))
    sampled_z = terrain.get_height(sub_X.ravel(), sub_Y.ravel()).reshape((dense_grid_size, dense_grid_size))
    d50_local = max(float(np.ptp(sampled_z - z_base)), 0.001)

    grain_radius = d50_local / 2.0
    grain_area = np.pi * (grain_radius**2)
    window_area = (x_max - x_min) * (y_max - y_min)
    num_local_grains = int(np.clip(0.80 * window_area / grain_area, 50, 8000))

    rng = np.random.default_rng(42)
    gx = rng.uniform(x_min - grain_radius, x_max + grain_radius, size=num_local_grains)
    gy = rng.uniform(y_min - grain_radius, y_max + grain_radius, size=num_local_grains)
    gh = rng.uniform(0.8, 1.2, size=num_local_grains) * d50_local

    z_micro_local = _rasterize_grains(sub_X, sub_Y, gx, gy, gh, grain_radius)
    sub_Z = z_base + z_micro_local

    footprint_dist_sq = (sub_X - ball_x) ** 2 + (sub_Y - ball_y) ** 2
    contact_mask = footprint_dist_sq <= (ball_radius**2)
    z_contact = np.max(sub_Z[contact_mask]) if np.any(contact_mask) else np.max(sub_Z)

    ball_z = z_contact + ball_radius

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
    """Generates 3D surface plot with sparse surface normal vectors."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="summer", alpha=0.75, rstride=2, cstride=2)

    s = max(1, quiver_skip)
    x_sub = terrain.X[::s, ::s]
    y_sub = terrain.Y[::s, ::s]
    z_sub = terrain.Z[::s, ::s]

    nx_flat, ny_flat, nz_flat = terrain.get_normal(x_sub.ravel(), y_sub.ravel())
    nx_arr = nx_flat.reshape(x_sub.shape)
    ny_arr = ny_flat.reshape(x_sub.shape)
    nz_arr = nz_flat.reshape(x_sub.shape)

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


def plot_variance_over_time(time_axis: np.ndarray, variance: np.ndarray) -> Figure:
    """Plots lateral (Y) position variance across the ensemble as a function of time."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, variance, color="#1e88e5", lw=1.8)
    ax.set_title("Ensemble Lateral Variance Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Var(Y) (cm²)")
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig
