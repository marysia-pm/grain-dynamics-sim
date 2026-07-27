"""Plotting helpers.

Every function here builds and returns a ``matplotlib.figure.Figure``
instead of calling ``plt.show()`` directly, so callers (scripts, the CLI,
notebooks, tests) can choose whether to display, save, or inspect the
result. This also makes the functions safe to call in headless/CI
environments where no display is available.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from rough_slope_sim.analysis import GritFitResult, RoughnessSweepResult, exponential_decay_model, linear_model
from rough_slope_sim.simulation import Trajectory
from rough_slope_sim.terrain import Terrain


def plot_terrain_3d(terrain: Terrain, *, quiver_skip: int = 100) -> Figure:
    """3D surface plot of the terrain, with surface-normal vectors overlaid."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="summer", alpha=0.8)

    dz_dx, dz_dy = terrain.dz_dx, terrain.dz_dy
    normals_x, normals_y, normals_z = -dz_dx, -dz_dy, np.ones_like(terrain.Z)
    magnitude = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2)
    nx, ny, nz = normals_x / magnitude, normals_y / magnitude, normals_z / magnitude

    s = quiver_skip
    ax.quiver(
        terrain.X[::s, ::s], terrain.Y[::s, ::s], terrain.Z[::s, ::s],
        nx[::s, ::s], ny[::s, ::s], nz[::s, ::s],
        length=0.5, color="red", alpha=1, zorder=1,
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Height")
    ax.set_zlim(0, terrain.y.max())
    ax.set_title(f"3D Terrain (alpha={terrain.config.alpha})")
    return fig


def plot_trajectories_3d(terrain: Terrain, trajectories: list[Trajectory]) -> Figure:
    """3D plot of the terrain with every ball trajectory overlaid."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(terrain.X, terrain.Y, terrain.Z, cmap="summer", alpha=0.5)

    colors = cm.inferno(np.linspace(0, 1, max(len(trajectories), 1)))
    for k, traj in enumerate(trajectories):
        ax.plot(traj.x, traj.y, traj.z, "-", color=colors[k], linewidth=2, label=f"{k}-trajectory", zorder=1)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Height")
    ax.set_zlim(0, terrain.y.max())
    ax.set_title(f"3D Ball Simulation on Terrain (alpha={terrain.config.alpha})")
    ax.view_init(elev=30, azim=-45)
    return fig


def plot_energies(traj: Trajectory, mass: float, g: float, ball_radius: float) -> Figure:
    """Potential, kinetic and total mechanical energy over time for one
    trajectory."""
    potential = mass * g * (traj.z + ball_radius)
    kinetic = 0.5 * mass * (traj.vx**2 + traj.vy**2 + traj.vz**2)
    total = potential + kinetic

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(traj.time, potential, label="Potential Energy")
    ax.plot(traj.time, kinetic, label="Kinetic Energy")
    ax.plot(traj.time, total, label="Total Energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs. Time")
    ax.legend()
    return fig


def plot_final_position_distribution(
    final_y: np.ndarray, final_z: np.ndarray, roughness_transition_y: float
) -> Figure:
    """Scatter + marginal histograms of final (y, z) ball positions, as in
    the original notebook's ``scatter_hist`` figure."""
    fig = plt.figure(figsize=(6, 6))
    axs = fig.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        width_ratios=(4, 1), height_ratios=(1, 4),
    )
    ax_scatter, ax_histx, ax_histy = axs["scatter"], axs["histx"], axs["histy"]

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    valid = ~(np.isnan(final_y) | np.isnan(final_z))
    y, z = final_y[valid], final_z[valid]

    ax_scatter.scatter(y, z)
    ax_scatter.axvline(x=roughness_transition_y, linestyle="--", linewidth=3, alpha=0.6, label="roughness transition")
    ax_scatter.set_xlabel("Y Position")
    ax_scatter.set_ylabel("Z Position")
    ax_scatter.set_xlim(0, 5)
    ax_scatter.set_ylim(0, 5)

    binwidth = 0.25
    bins = np.arange(0, 5 + binwidth, binwidth)
    ax_histx.hist(y, bins=bins, density=True)
    ax_histx.axvline(x=roughness_transition_y, linestyle="--", linewidth=3, alpha=0.6)
    ax_histx.set_title("Ball Position Density Distribution at X=X_max")
    ax_histy.hist(z, bins=bins, orientation="horizontal", density=True)

    return fig


def plot_variance_over_time(time_points: np.ndarray, variance: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(time_points, variance, label="Variance of Y-Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Variance of Y-Position ($\sigma_y^2$)")
    ax.set_title("Variance of Ball's Y-Position Over Time Across All Simulations")
    ax.grid(True)
    ax.legend()
    return fig


def plot_count_below_threshold(time_points: np.ndarray, counts: np.ndarray, threshold: float) -> Figure:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(time_points, counts)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Number of Y <= {threshold}")
    ax.set_title("Number of Balls Staying on the Rough Part of the Surface")
    return fig


def plot_grit_calibration(fit: GritFitResult) -> tuple[Figure, Figure, Figure]:
    """The three grit/Sa calibration figures from the original notebook:
    particle size vs Sa (linear fit), particle size vs grit (exponential
    fit), and the derived Sa vs grit curve."""
    a_l, b_l = fit.linear_params
    a, b, c = fit.exp_params

    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(fit.sa, fit.mean_particle_size)
    ax1.plot(fit.sa, linear_model(fit.sa, a_l, b_l), color="C1", linestyle="--")
    ax1.set_xlabel("Sa")
    ax1.set_ylabel("Mean Particle Size (micrometers)")
    ax1.set_title("Mean Particle Size vs Sa")
    ax1.text(
        0.05, 0.95, f"A = {a_l:.2f}\nB = {b_l:.4f}", transform=ax1.transAxes,
        fontsize=10, verticalalignment="top", bbox=dict(facecolor="none", edgecolor="C0", boxstyle="round,pad=1"),
    )

    grit_fit = np.linspace(0, fit.grit.max(), 100)
    particle_size_fit = exponential_decay_model(grit_fit, a, b, c)

    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(grit_fit, particle_size_fit)
    ax2.errorbar(
        fit.grit, fit.mean_particle_size, yerr=fit.mean_particle_size_stdev,
        marker="o", markersize=8, linestyle="none", color="C4",
    )
    ax2.set_xlabel("Grit (P-grade)")
    ax2.set_ylabel("Mean Particle Size (micrometers)")
    ax2.set_title("Mean Particle Size vs Grit (Exponential Fit)")
    ax2.text(
        0.60, 0.95, f"A = {a:.2f}\nB = {b:.4f}\nC = {c:.2f}", transform=ax2.transAxes,
        fontsize=10, verticalalignment="top", bbox=dict(facecolor="none", edgecolor="C0", boxstyle="round,pad=1"),
    )

    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111)
    ax3.plot(grit_fit, (particle_size_fit - b_l) / a_l, zorder=1)
    ax3.set_xlabel("Grit (P-grade)")
    ax3.set_ylabel("Sa (micrometers)")
    ax3.set_title("Sa vs Grit (Exponential Fit)")

    return fig1, fig2, fig3


def plot_roughness_sweep(result: RoughnessSweepResult) -> Figure:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(result.sa_values, result.diffusion_coefficients, marker="o", linestyle="-")
    ax.set_xlabel("Arithmetical Average Roughness (Sa)")
    ax.set_ylabel("Diffusion Coefficient (Variance of Final Y-Positions)")
    ax.set_title("Diffusion Coefficient vs. Terrain Roughness")
    ax.grid(True)
    return fig
