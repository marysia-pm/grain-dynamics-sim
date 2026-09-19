"""Plotting routines for simulation results, 3D terrain visualizations, and Nanovea benchmark panels."""

from __future__ import annotations

from typing import Any, Sequence
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import gaussian_kde, norm

from .analysis import trajectories_at_x_slice
from .simulation import Trajectory
from .terrain import (
    NANOVEA_DATA,
    Terrain,
    _tiled_grain_bump_field,
    galton_board_bump_field,
    nanovea_d50_from_grit,
    nanovea_sa_from_d50,
)

try:
    plt.style.use("https://raw.githubusercontent.com/turczyneq/softmatter-style/main/softmatter.mplstyle")
except Exception:
    # The style sheet is fetched over the network at import time with no fallback,
    # so importing this module (or anything that imports the package) hard-crashes
    # anywhere without live internet access (CI, sandboxes, restricted networks).
    plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _add_info_box(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    """Adds a standardized high-contrast physics metadata box."""
    x, y = (0.03, 0.95) if loc == "upper left" else (0.97, 0.95)
    ha = "left" if loc == "upper left" else "right"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment=ha,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray", lw=1.0),
        zorder=100,
    )


def _format_run_title(terrain: Terrain | None, ball_radius: float | None) -> str:
    """Builds a 'Trajectories (...)' title summarizing the run's key physical
    parameters (ball radius, slope angle, surface type/grit) instead of a
    generic description."""
    parts = []
    if ball_radius is not None:
        parts.append(f"R={ball_radius * 10.0:.2f}mm")
    if terrain is not None:
        parts.append(rf"$\theta$={terrain.slope_angle:.0f}$^\circ$")
        if terrain.surface_type == "galton":
            parts.append("Galton Board")
        else:
            gr, gs = terrain.grit_rough, terrain.grit_smooth
            if np.isclose(gr, gs):
                parts.append(f"Sandpaper P{gr:.0f}")
            else:
                parts.append(f"Sandpaper P{gr:.0f}/P{gs:.0f}")
    if not parts:
        return "Trajectories"
    return "Trajectories (" + ", ".join(parts) + ")"


def _auto_limits(
    *value_groups: Sequence[float] | np.ndarray,
    pad_frac: float = 0.15,
    min_pad: float = 0.5,
    fallback: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Data-driven axis limits: min/max across all given value arrays plus a padding
    margin, instead of a domain-wide or otherwise hardcoded range that leaves narrow
    data clusters looking like a thin sliver in a mostly-empty plot. Falls back to
    `fallback` when no finite values are available at all."""
    finite_chunks = []
    for values in value_groups:
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)

    if not finite_chunks:
        return fallback

    all_vals = np.concatenate(finite_chunks)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = max((hi - lo) * pad_frac, min_pad)
    return lo - pad, hi + pad


def plot_calibrated_sandpaper_panel(
    patch_size_cm: float = 0.20, grid_res: int = 500, seed: int = 42
) -> Figure:
    """Generates a 2x5 comparison figure across all standard Nanovea benchmark grit levels.

    Uses the same tiled-grain generator as the real terrain (not a placeholder), so
    this panel's grain shapes and Sa calibration match what generate_terrain actually
    produces -- just at the fine, dedicated per-grit resolution needed to resolve
    grains that can be a few micrometers across.
    """
    fig = plt.figure(figsize=(22, 9))
    grits = list(NANOVEA_DATA.keys())

    for col, target_grit in enumerate(grits):
        d50_cm = NANOVEA_DATA[target_grit]["d50_cm"]
        target_diam_um = d50_cm * 10000.0
        target_Sa_um = nanovea_sa_from_d50(d50_cm) * 10000.0

        x_grid = np.linspace(0, patch_size_cm, grid_res)
        y_grid = np.linspace(0, patch_size_cm, grid_res)
        X_m, Y_m = np.meshgrid(x_grid, y_grid, indexing="ij")

        z_local = _tiled_grain_bump_field(
            x_grid, y_grid, d50_cm=d50_cm, slope_angle=0.0, tile_size_cm=patch_size_cm, seed=seed + col
        )

        ax1 = fig.add_subplot(2, 5, col + 1)
        ax1.imshow(
            z_local.T * 10.0,
            extent=[0, patch_size_cm * 10, 0, patch_size_cm * 10],
            origin="lower",
            cmap="viridis",
        )
        ax1.set_title(
            f"Simulated P{target_grit}\n | Diam: {target_diam_um:.1f} $\\mu$m | $S_a$: {target_Sa_um:.2f} $\\mu$m",
            fontsize=11,
            fontweight="bold",
        )
        if col == 0:
            ax1.set_ylabel("y (mm)", fontweight="bold")
        ax1.set_xlabel("x (mm)", fontweight="bold")

        ax2 = fig.add_subplot(2, 5, col + 6, projection="3d")
        ds = 4
        ax2.plot_surface(
            X_m[::ds, ::ds] * 10,
            Y_m[::ds, ::ds] * 10,
            z_local[::ds, ::ds] * 10,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        ax2.set_title(f"3D Topography (P{target_grit})", fontsize=11, fontweight="bold")
        if col == 0:
            ax2.set_zlabel("Z (mm)", fontweight="bold")
        ax2.set_xlabel("X (mm)", fontweight="bold")
        ax2.set_ylabel("Y (mm)", fontweight="bold")

        ax2.set_box_aspect((1, 1, 0.35))
        ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()
    return fig


def _tilted_cylinder_mesh(
    rs: float, ry: float, radius: float, height: float, slope_angle: float, z_offset: float, n_theta: int = 28
):
    """Builds an exact 3D mesh (tube + flat top cap) for a cylinder of given
    `radius`/`height`, based at surface coordinate (rs, ry) [rs measured along
    the incline], with its axis along the surface NORMAL. Used to draw pegs as
    true cylinders in the closeup plot -- a heightfield (single Z per X, Y) can
    only approximate a tilted cylinder, and that approximation's flat top
    shrinks toward a point as height/radius grows (more of what a vertical ray
    sees is the lateral wall, not the cap). Drawing the actual mesh sidesteps
    that entirely; it's for this plot's visuals only, not the physics grid.
    """
    rad = np.radians(slope_angle)
    cos_th, sin_th = np.cos(rad), np.sin(rad)
    x0 = rs * cos_th
    y0 = ry
    z0 = z_offset - x0 * np.tan(rad)

    phi = np.linspace(0.0, 2.0 * np.pi, n_theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    n_vals = np.array([0.0, height])
    N, COS_PHI = np.meshgrid(n_vals, cos_phi, indexing="ij")
    _, SIN_PHI = np.meshgrid(n_vals, sin_phi, indexing="ij")
    tube_x = x0 + N * sin_th + radius * COS_PHI * cos_th
    tube_y = y0 + radius * SIN_PHI
    tube_z = z0 + N * cos_th - radius * COS_PHI * sin_th

    rho = np.linspace(0.0, radius, 6)
    RHO, COS_PHI2 = np.meshgrid(rho, cos_phi, indexing="ij")
    _, SIN_PHI2 = np.meshgrid(rho, sin_phi, indexing="ij")
    cap_x = x0 + height * sin_th + RHO * COS_PHI2 * cos_th
    cap_y = y0 + RHO * SIN_PHI2
    cap_z = z0 + height * cos_th - RHO * COS_PHI2 * sin_th

    return (tube_x, tube_y, tube_z), (cap_x, cap_y, cap_z)


def _peg_centers_in_window(terrain: Terrain, x_min: float, x_max: float, y_min: float, y_max: float):
    """Enumerates (rs, ry) peg-center coordinates (rs along the incline) whose
    footprint could be visible within the given (x, y) window, using the same
    row/column placement as galton_board_bump_field."""
    rad = np.radians(terrain.slope_angle)
    cos_th, sin_th = np.cos(rad), np.sin(rad)
    peg_r = terrain.peg_radius
    peg_h = terrain.peg_height

    s_start = terrain.peg_x_start
    s_end = terrain.peg_x_end if terrain.peg_x_end is not None else terrain.x_bounds[1] / cos_th
    rows_s = np.arange(s_start, s_end + 1e-9, terrain.peg_dx)

    y_lo, y_hi = terrain.y_bounds
    x_pad = peg_r + peg_h * sin_th  # a leaning peg's footprint extends downslope by h*sin(theta)

    centers = []
    for i, rs in enumerate(rows_s):
        rx = rs * cos_th
        if rx < x_min - x_pad or rx > x_max + peg_r:
            continue
        y_offset = (terrain.peg_dy / 2.0) if (terrain.peg_staggered and i % 2 == 1) else 0.0
        first_y = y_lo + peg_r + y_offset
        cols_y = np.arange(first_y, y_hi - peg_r + 1e-9, terrain.peg_dy)
        for ry in cols_y:
            if y_min - peg_r <= ry <= y_max + peg_r:
                centers.append((rs, ry))
    return centers


def plot_ball_surface_closeup(
    terrain: Terrain,
    ball_radius: float,
    ball_x: float,
    ball_y: float,
    window_factor: float = 1.0,
    dense_grid_size: int = 150,
    z_offset: float | None = None,
) -> Figure:
    """3D surface contact closeup around the contact point.

    For sandpaper, the local grain texture is regenerated fresh at a resolution
    matched to the terrain's own d50 (via the same _tiled_grain_bump_field used by
    generate_terrain), instead of interpolating from the coarse full-ramp
    simulation grid. That coarse grid only has ~1500 points across a ~25cm ramp,
    which cannot resolve grains a few micrometers across at high grit -- this is
    why closeups used to look progressively less like real grains as grit
    increased. Regenerating locally keeps the closeup grit-independent, the same
    way exploring_surfaces.py always renders a small dedicated high-resolution
    patch regardless of grit.
    """
    if z_offset is None:
        # Use the same baseline the terrain was actually built with, rather than a
        # guessed constant (the baseline now scales with ramp length, see terrain.py).
        z_offset = getattr(terrain, "z_offset", 15.0)

    win_size = max(window_factor * ball_radius, 3.0)
    x_min, x_max = ball_x - win_size, ball_x + win_size
    y_min, y_max = ball_y - win_size, ball_y + win_size

    surface_type = getattr(terrain, "surface_type", "sandpaper").lower()
    rad = np.radians(terrain.slope_angle)

    if surface_type == "galton":
        # Resolve each peg with enough points across its diameter that the vertical
        # sidewall renders as a true 90-degree wall. Interpolating the coarse global
        # grid instead smears the wall into a slanted ribbon (the peg geometry is
        # genuinely vertical; it was only the closeup mesh that was too coarse).
        peg_r = getattr(terrain, "peg_radius", 0.30)
        needed_spacing = max(2.0 * peg_r, 1e-6) / 30.0
        dense_grid_size = int(np.clip((2 * win_size) / needed_spacing, dense_grid_size, 600))
    else:
        # Pick a grid fine enough to resolve the local grain diameter (at least
        # ~8 points across one grain), capped so runtime stays bounded.
        transition_y = getattr(terrain, "roughness_transition_y", (y_min + y_max) / 2.0)
        local_d50 = (
            terrain.roughness_amplitude_rough if ball_y < transition_y else terrain.roughness_amplitude_smooth
        )
        local_d50 = max(local_d50, 1e-6)
        target_pts_per_grain = 8.0
        needed_spacing = local_d50 / target_pts_per_grain
        dense_grid_size = int(np.clip((2 * win_size) / needed_spacing, dense_grid_size, 400))

    sub_x = np.linspace(x_min, x_max, dense_grid_size)
    sub_y = np.linspace(y_min, y_max, dense_grid_size)
    sub_X, sub_Y = np.meshgrid(sub_x, sub_y, indexing="ij")

    z_base_local = z_offset - sub_X * np.tan(rad)

    peg_meshes: list[tuple] = []
    if surface_type == "galton":
        # Contact placement still uses the real (heightfield) peg geometry, so the
        # ball rests correctly whether it's over a peg tip or the flat board.
        peg_relief = galton_board_bump_field(
            sub_X,
            sub_Y,
            slope_angle=terrain.slope_angle,
            peg_radius=getattr(terrain, "peg_radius", 0.30),
            peg_height=getattr(terrain, "peg_height", 0.20),
            peg_dx=getattr(terrain, "peg_dx", 2.0),
            peg_dy=getattr(terrain, "peg_dy", 1.5),
            staggered=getattr(terrain, "peg_staggered", True),
            peg_x_start=getattr(terrain, "peg_x_start", 5.0),
            peg_x_end=getattr(terrain, "peg_x_end", None),
            shape=getattr(terrain, "peg_shape", "cylinder"),
        )
        sub_Z = np.maximum(0.0, z_base_local + peg_relief)

        # The BOARD itself renders flat (a heightfield can only approximate a
        # tilted cylinder, and that approximation's flat top shrinks to a point
        # as height/radius grows). Pegs are instead drawn as exact 3D cylinder
        # meshes below, which look like true cylinders at any aspect ratio.
        render_Z = z_base_local
        facecolors = np.tile(plt.get_cmap("viridis")(0.0), sub_X.shape + (1,))
        peg_meshes = [
            _tilted_cylinder_mesh(
                rs, ry, terrain.peg_radius, terrain.peg_height, terrain.slope_angle, z_offset
            )
            for rs, ry in _peg_centers_in_window(terrain, x_min, x_max, y_min, y_max)
        ]
        title_type = "Galton Board"
    else:
        local_tile = min(getattr(terrain, "tile_size_cm", 0.20), 2 * win_size)
        normal_bump = _tiled_grain_bump_field(
            sub_x,
            sub_y,
            d50_cm=local_d50,
            slope_angle=terrain.slope_angle,
            tile_size_cm=local_tile,
            seed=123,
        )
        z_micro_local = normal_bump / np.cos(rad)  # same normal->vertical conversion as generate_terrain
        sub_Z = np.maximum(0.0, z_base_local + z_micro_local)
        render_Z = sub_Z
        facecolors = plt.get_cmap("viridis")(mcolors.Normalize()(z_micro_local))
        title_type = "Sandpaper Surface"

    footprint_dist_sq = (sub_X - ball_x) ** 2 + (sub_Y - ball_y) ** 2
    contact_mask = footprint_dist_sq <= (ball_radius**2)
    z_contact = np.max(sub_Z[contact_mask]) if np.any(contact_mask) else np.max(sub_Z)

    nx, ny, nz = terrain.get_normal(ball_x, ball_y)
    ball_center_x = ball_x + ball_radius * nx
    ball_center_y = ball_y + ball_radius * ny
    ball_center_z = z_contact + ball_radius * nz

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sphere_x = ball_center_x + ball_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = ball_center_y + ball_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = ball_center_z + ball_radius * np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    ax.plot_surface(
        sub_X,
        sub_Y,
        render_Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    peg_top_color = plt.get_cmap("viridis")(1.0)
    for (tube_x, tube_y, tube_z), (cap_x, cap_y, cap_z) in peg_meshes:
        ax.plot_surface(tube_x, tube_y, tube_z, color=peg_top_color, alpha=0.95, linewidth=0)
        ax.plot_surface(cap_x, cap_y, cap_z, color=peg_top_color, alpha=0.95, linewidth=0)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color="crimson", alpha=0.95, edgecolor="darkred", lw=0.3)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(ball_center_z - win_size, ball_center_z + win_size)
    ax.view_init(elev=25, azim=-60)

    ax.set_xlabel("X (cm)", fontweight="bold")
    ax.set_ylabel("Y (cm)", fontweight="bold")
    ax.set_zlabel("Z (cm)", fontweight="bold")
    ax.set_title(f"{title_type} Contact Closeup (R = {ball_radius * 10.0:.1f} mm)", fontweight="bold", pad=15)

    plt.tight_layout()
    return fig


def plot_terrain_3d(terrain: Terrain, *, quiver_skip: int = 1000) -> Figure:
    """Generates 3D surface plot over the entire surface with sparse surface normal vectors."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="viridis", alpha=0.75, rstride=2, cstride=2)

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

    ax.set_xlim(terrain.x_bounds)
    ax.set_ylim(terrain.y_bounds)

    ax.set_title("3D Terrain Surface with Sparse Normal Vectors", fontweight="bold", pad=12)
    ax.set_xlabel("X Position (cm)", fontweight="bold")
    ax.set_ylabel("Y Position (cm)", fontweight="bold")
    ax.set_zlabel("Z Height (cm)", fontweight="bold")
    ax.view_init(elev=35, azim=-50)
    plt.tight_layout()
    return fig


def plot_trajectories_3d(terrain: Terrain, trajectories: list[Trajectory]) -> Figure:
    """Renders simulated trajectories across the entire 3D terrain surface."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="viridis", alpha=0.5)

    colors = cm.viridis(np.linspace(0, 1, max(len(trajectories), 1)))
    for k, traj in enumerate(trajectories):
        x_arr = traj.x if isinstance(traj, Trajectory) else traj[:, 1]
        y_arr = traj.y if isinstance(traj, Trajectory) else traj[:, 2]
        z_arr = traj.z if isinstance(traj, Trajectory) else np.zeros_like(x_arr)
        ax.plot(x_arr, y_arr, z_arr, "-", color=colors[k], linewidth=1.1, zorder=1)

    ax.set_xlim(terrain.x_bounds)
    ax.set_ylim(terrain.y_bounds)

    ax.set_xlabel("X Position (cm)", fontweight="bold")
    ax.set_ylabel("Y Position (cm)", fontweight="bold")
    ax.set_zlabel("Z Height (cm)", fontweight="bold")
    ax.set_title("3D Trajectories Ensemble", fontweight="bold", pad=12)
    ax.view_init(elev=30, azim=-45)
    plt.tight_layout()
    return fig


def plot_experiment_vs_sim_distribution(
    y_exp: Any,
    y_sim: Any,
    x_slice: float | Terrain | None = None,
    title: str | None = None,
) -> Figure:
    """Flexible distribution / trajectory comparator with data-driven axis limits."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if isinstance(x_slice, (int, float)):
        valid_exp = np.asarray(y_exp)[~np.isnan(y_exp)]
        valid_sim = np.asarray(y_sim)[~np.isnan(y_sim)]

        ax.hist(valid_exp, bins=12, density=True, alpha=0.5, color="#1e88e5", label="Exp")
        ax.hist(valid_sim, bins=12, density=True, alpha=0.5, color="#ffb300", label="Sim")

        if len(valid_exp) > 3:
            g_e = np.linspace(min(valid_exp), max(valid_exp), 150)
            ax.plot(g_e, gaussian_kde(valid_exp)(g_e), color="#0d47a1", lw=2)
        if len(valid_sim) > 3:
            g_s = np.linspace(min(valid_sim), max(valid_sim), 150)
            ax.plot(g_s, gaussian_kde(valid_sim)(g_s), color="#ff6f00", lw=2)

        ax.set_xlim(*_auto_limits(valid_exp, valid_sim, fallback=(0.0, 1.0)))

        ax.set_title(title or f"Histogram at X = {x_slice:.1f} cm", fontweight="bold", pad=12)
        ax.set_xlabel("Y Position (cm)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)
    else:

        def _extract_xy(data):
            if len(data) == 0:
                return [], []
            if hasattr(data[0], "x") and hasattr(data[0], "y"):
                return [np.asarray(r.x) for r in data], [np.asarray(r.y) for r in data]
            data_arr = [np.asarray(d) for d in data]
            return [d[:, 1] if d.ndim == 2 else d for d in data_arr], [
                d[:, 2] if d.ndim == 2 else d for d in data_arr
            ]

        exp_xs, exp_ys = _extract_xy(y_exp)
        sim_xs, sim_ys = _extract_xy(y_sim)

        for idx, (x, y) in enumerate(zip(sim_xs, sim_ys)):
            ax.plot(x, y, color="#1f77b4", alpha=0.25, lw=1.2, label="Simulation" if idx == 0 else "")

        for idx, (x, y) in enumerate(zip(exp_xs, exp_ys)):
            ax.plot(x, y, color="#d62728", alpha=0.85, lw=2.0, label="Experiment" if idx == 0 else "")

        all_x = [np.asarray(a) for a in (sim_xs + exp_xs)]
        all_y = [np.asarray(a) for a in (sim_ys + exp_ys)]
        ax.set_xlim(*_auto_limits(*all_x, fallback=(0.0, 25.0)))
        ax.set_ylim(*_auto_limits(*all_y, fallback=(0.0, 23.0)))

        ax.set_xlabel("Downslope Position $x$ (cm)", fontweight="bold")
        ax.set_ylabel("Lateral Position $y$ (cm)", fontweight="bold")
        ax.set_title(title or "Experiment vs. Simulation Trajectory Distribution", fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)

        info_text = f"Exp Runs: {len(exp_xs)}\nSim Runs: {len(sim_xs)}\n"
        _add_info_box(ax, info_text, loc="upper left")
        ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_trajectories_and_three_slices(
    exp_trajs: list,
    sim_trajs: list,
    x_slices: list[float],
    is_dual: bool = False,
    interface_y: float = 11.5,
    terrain: Terrain | None = None,
    ball_radius: float | None = None,
) -> Figure:
    """Multi-panel plot: 90° CW rotated trajectories and 3 cross-section histograms.

    The main panel's lateral (Y) axis is fixed to the terrain/domain width
    (not auto-scaled to the data) so it always reads as "the whole board".
    All three slice histograms share one x-range, taken from the last
    (deepest) slice -- the widest spread -- rather than each auto-scaling on
    its own, so the same physical Y-range is directly comparable slice to
    slice. Only the last slice gets a Gaussian fit overlay.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1.0])

    ax_traj = fig.add_subplot(gs[:, 0])

    all_traj_x: list[np.ndarray] = []

    for t in exp_trajs:
        x_arr = t[:, 1] if isinstance(t, np.ndarray) else t.x
        y_arr = t[:, 2] if isinstance(t, np.ndarray) else t.y
        all_traj_x.append(np.asarray(x_arr))
        ax_traj.plot(y_arr, x_arr, color="#1e88e5", alpha=0.25, lw=1.0)

    for t in sim_trajs:
        x_arr = t[:, 1] if isinstance(t, np.ndarray) else t.x
        y_arr = t[:, 2] if isinstance(t, np.ndarray) else t.y
        all_traj_x.append(np.asarray(x_arr))
        ax_traj.plot(y_arr, x_arr, color="#ffb300", alpha=0.20, lw=0.8)

    colors = ["#e53935", "#8e24aa", "#43a047"]
    for x_s, c in zip(x_slices, colors):
        ax_traj.axhline(x_s, color=c, linestyle="--", lw=1.8, label=f"Slice X={x_s:.1f} cm")

    if is_dual:
        ax_traj.axvline(
            interface_y, color="black", linestyle="-.", lw=2.0, label=f"Interface (Y={interface_y:.1f} cm)"
        )

    # Lateral (Y) axis on the main panel is fixed to the board/domain width
    # (terrain.y_bounds, or (0, 23) if no terrain given) rather than auto-scaled
    # to the trajectory data -- this panel should always read as "the whole board".
    y_fallback = terrain.y_bounds if terrain is not None else (0.0, 23.0)
    x_fallback = terrain.x_bounds if terrain is not None else (0.0, 25.0)
    x_ref = list(x_slices) + ([interface_y] if is_dual else [])
    ax_traj.set_xlim(*y_fallback)
    ax_traj.set_ylim(*_auto_limits(*all_traj_x, x_ref, fallback=x_fallback))

    ax_traj.set_title(_format_run_title(terrain, ball_radius), fontweight="bold", pad=12)
    ax_traj.set_xlabel("Y Position (cm) [Lateral]", fontweight="bold")
    ax_traj.set_ylabel("X Position (cm) [Downslope]", fontweight="bold")
    ax_traj.invert_yaxis()
    ax_traj.grid(True, linestyle=":", alpha=0.5)
    ax_traj.legend(loc="upper right", fontsize=10)

    # Gather all three slices' data first so the shared x-limit (from the last
    # slice) is known before any histogram is drawn.
    slice_data = []
    for x_s in x_slices:
        y_exp = trajectories_at_x_slice(exp_trajs, x_s)
        y_sim = trajectories_at_x_slice(sim_trajs, x_s)
        slice_data.append((y_exp[~np.isnan(y_exp)], y_sim[~np.isnan(y_sim)]))

    valid_e_last, valid_s_last = slice_data[-1]
    shared_xlim = _auto_limits(valid_e_last, valid_s_last, fallback=y_fallback)

    for idx, (x_s, c) in enumerate(zip(x_slices, colors)):
        ax_slice = fig.add_subplot(gs[idx, 1])
        valid_e, valid_s = slice_data[idx]
        is_last = idx == len(x_slices) - 1

        if len(valid_e) > 0:
            ax_slice.hist(valid_e, bins=12, density=True, alpha=0.4, color="#1e88e5", label="Exp")
        if len(valid_s) > 0:
            ax_slice.hist(valid_s, bins=12, density=True, alpha=0.4, color="#ffb300", label="Sim")

        if is_last:
            for valid, color, label in [(valid_e, "#0d47a1", "Exp"), (valid_s, "#ff6f00", "Sim")]:
                if len(valid) > 2 and np.std(valid) > 1e-9:
                    mu, sigma = norm.fit(valid)
                    grid = np.linspace(*shared_xlim, 200)
                    ax_slice.plot(
                        grid,
                        norm.pdf(grid, mu, sigma),
                        color=color,
                        lw=2.0,
                        label=rf"{label} fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}",
                    )

        if is_dual:
            ax_slice.axvline(interface_y, color="black", linestyle="-.", lw=1.2)

        ax_slice.set_title(f"Slice at X = {x_s:.1f} cm", color=c, fontweight="bold")
        ax_slice.set_xlabel("Y Position (cm)", fontweight="bold")
        ax_slice.set_ylabel("Density", fontweight="bold")
        ax_slice.set_xlim(*shared_xlim)
        ax_slice.grid(True, linestyle=":", alpha=0.5)
        if len(valid_e) > 0 or len(valid_s) > 0:
            ax_slice.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_variance_over_time(time_axis: np.ndarray, variance: np.ndarray) -> Figure:
    """Plots lateral (Y) position variance across the ensemble as a function of time."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(time_axis, variance, color="#1e88e5", lw=1.8)
    ax.set_title("Ensemble Lateral Variance Over Time", fontweight="bold", pad=12)
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Var(Y) (cm$^2$)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_trajectory_2d(
    record: Any,
    terrain: Any = None,
    title: str = "Ball Trajectory Path",
) -> Figure:
    """Plots 2D trajectory enforced on the full terrain surface."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x, y = record.x, record.y
    vx, vy = record.vx, record.vy
    speed = np.hypot(vx, vy)

    sc = ax.scatter(x, y, c=speed, cmap="viridis", s=14, alpha=0.85, label="Trajectory")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Speed $v$ (cm/s)", fontweight="bold")

    if terrain is not None and hasattr(terrain, "x_bounds") and hasattr(terrain, "y_bounds"):
        ax.set_xlim(terrain.x_bounds)
        ax.set_ylim(terrain.y_bounds)
    else:
        ax.set_xlim(*_auto_limits(x, fallback=(0.0, 25.0)))
        ax.set_ylim(*_auto_limits(y, fallback=(0.0, 23.0)))

    ax.set_xlabel("Downslope Position $x$ (cm)", fontweight="bold")
    ax.set_ylabel("Lateral Position $y$ (cm)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    slope_angle = getattr(record, "metadata", {}).get("slope_angle", getattr(terrain, "slope_angle", 30.0))
    info_text = (
        f"Restitution $e$: 0.70\n"
        f"Slope Angle $\\theta$: {slope_angle:.1f}$^\\circ$\n"
        f"Spawn $(x_0, y_0)$: ({x[0]:.2f}, {y[0]:.2f}) cm\n"
        f"End $(x_f, y_f)$: ({x[-1]:.2f}, {y[-1]:.2f}) cm\n"
        f"Max Speed: {np.max(speed):.1f} cm/s"
    )
    _add_info_box(ax, info_text, loc="upper left")

    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    return fig


def plot_trajectories_comparison(
    records: Sequence[Any],
    labels: Sequence[str],
    terrain: Any = None,
    title: str = "Trajectory Ensemble Comparison",
) -> Figure:
    """Compares trajectory ensembles across the full terrain surface bounds."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for idx, (rec, label) in enumerate(zip(records, labels)):
        color = colors[idx % len(colors)]
        all_x.append(np.asarray(rec.x))
        all_y.append(np.asarray(rec.y))
        ax.plot(rec.x, rec.y, label=label, color=color, lw=2.0, alpha=0.85)

    if terrain is not None and hasattr(terrain, "x_bounds") and hasattr(terrain, "y_bounds"):
        ax.set_xlim(terrain.x_bounds)
        ax.set_ylim(terrain.y_bounds)
    else:
        ax.set_xlim(*_auto_limits(*all_x, fallback=(0.0, 25.0)))
        ax.set_ylim(*_auto_limits(*all_y, fallback=(0.0, 23.0)))

    ax.set_xlabel("Downslope Position $x$ (cm)", fontweight="bold")
    ax.set_ylabel("Lateral Position $y$ (cm)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    info_text = f"Ensembles: {len(records)}\nRestitution $e$: 0.70\n"
    _add_info_box(ax, info_text, loc="upper left")

    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    return fig
