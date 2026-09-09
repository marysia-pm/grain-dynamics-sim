"""Script with Gaussian PDF fitting, overlay plotting, and JSON/CSV parameter export."""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch

from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import run_ensemble_parallel
from rough_slope_sim.terrain import generate_terrain


def calculate_free_fall_velocity(height_mm: float = 0.5, g: float = 981.0) -> float:
    return -float(np.sqrt(2.0 * g * (height_mm / 10.0)))


def extract_y_at_x(trajectories, target_x: float) -> np.ndarray:
    """Interpolates transverse Y positions across all trajectories at a fixed downslope X plane."""
    y_vals = []
    for t in trajectories:
        pos = t.positions if hasattr(t, "positions") else np.asarray(t)
        if len(pos) > 1 and pos[0, 0] <= target_x <= pos[-1, 0]:
            y_interp = np.interp(target_x, pos[:, 0], pos[:, 1])
            y_vals.append(y_interp)
    return np.array(y_vals)


def main():
    # 1. Configuration Dataclasses
    terrain_cfg = TerrainConfig(ramp_length=29.0, slope_angle=30.0, length_y=23.0)
    phys_cfg = PhysicsConfig(dt=0.001, total_time=3.0, gravity=981.0, slope_angle=30.0, friction=0.0)
    ball_cfg = BallConfig(radius=0.125, mass=1.0)

    v_z_impact = calculate_free_fall_velocity(height_mm=0.5, g=phys_cfg.gravity)

    cfg_zero = EnsembleConfig(
        k_max=150, start_x=0.1, start_y=11.5, vx0=0.0, vy0=0.0, vz0=0.0, x_jitter_std=0.05, y_jitter_max=2.0
    )
    cfg_impact = EnsembleConfig(
        k_max=150,
        start_x=0.1,
        start_y=11.5,
        vx0=0.0,
        vy0=0.0,
        vz0=v_z_impact,
        x_jitter_std=0.05,
        y_jitter_max=2.0,
    )

    # 2. Build Terrain Surface
    print("[+] Generating terrain profile...")
    terrain = generate_terrain(
        slope_angle=terrain_cfg.slope_angle,
        p_value_left=120.0,
        p_value_right=120.0,
        is_dual=False,
        use_gpu=torch.cuda.is_available(),
    )

    # 3. Simulations
    print("[+] Simulating Run 1 (v_z0 = 0)...")
    num_runs = getattr(cfg_zero, "num_runs", 10)
    ball_cfgs = [ball_cfg] * num_runs
    trajs_zero = run_ensemble_parallel(terrain, ball_cfgs, phys_cfg, cfg_zero)

    print(f"[+] Simulating Run 2 (v_z0 = {v_z_impact:.2f} cm/s)...")
    trajs_impact = run_ensemble_parallel(terrain, ball_cfg, phys_cfg, cfg_impact)

    # 4. Figure & Grid Layout
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1.0], hspace=0.35, wspace=0.25)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_slice1 = fig.add_subplot(gs[0, 1])
    ax_slice2 = fig.add_subplot(gs[1, 1])
    ax_slice3 = fig.add_subplot(gs[2, 1])

    color_run1 = "#9ecae1"  # Light Blue
    color_run2 = "#ffe082"  # Gold / Yellow
    c_slice1 = "#d95f02"  # Red
    c_slice2 = "purple"  # Purple
    c_slice3 = "forestgreen"  # Green

    # --- MAIN PANEL ---
    for t in trajs_zero:
        pos = t.positions if hasattr(t, "positions") else np.asarray(t)
        ax_main.plot(pos[:, 1], pos[:, 0], color=color_run1, alpha=0.35, linewidth=0.8)

    for t in trajs_impact:
        pos = t.positions if hasattr(t, "positions") else np.asarray(t)
        ax_main.plot(pos[:, 1], pos[:, 0], color=color_run2, alpha=0.35, linewidth=0.8)

    slice_x_vals = [5.0, 10.0, 15.0]
    ax_main.axhline(slice_x_vals[0], color=c_slice1, linestyle="--", linewidth=1.5, label="Slice X=5.0 cm")
    ax_main.axhline(slice_x_vals[1], color=c_slice2, linestyle="--", linewidth=1.5, label="Slice X=10.0 cm")
    ax_main.axhline(slice_x_vals[2], color=c_slice3, linestyle="--", linewidth=1.5, label="Slice X=15.0 cm")

    ax_main.set_title("Trajectories (Rotated 90° CW: Downslope Downwards)", fontsize=11, fontweight="bold")
    ax_main.set_xlabel("Y Position (cm) [Lateral]", fontsize=10)
    ax_main.set_ylabel("X Position (cm) [Downslope]", fontsize=10)
    ax_main.set_ylim(27.0, -1.0)
    ax_main.set_xlim(7.0, 17.0)
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_main.grid(True, linestyle=":", alpha=0.5)

    # --- RIGHT PANELS & GAUSSIAN FITTING ---
    slice_axes = [ax_slice1, ax_slice2, ax_slice3]
    slice_colors = [c_slice1, c_slice2, c_slice3]
    fit_records = []

    for x_val, ax, color in zip(slice_x_vals, slice_axes, slice_colors):
        y1 = extract_y_at_x(trajs_zero, x_val)
        y2 = extract_y_at_x(trajs_impact, x_val)

        # Plot binned histograms
        if len(y1) > 0:
            ax.hist(y1, bins=15, density=True, color="#56B4E9", alpha=0.5, edgecolor="none")
            mu1, std1 = norm.fit(y1)
            y_axis1 = np.linspace(y1.min() - 0.5, y1.max() + 0.5, 200)
            ax.plot(
                y_axis1,
                norm.pdf(y_axis1, mu1, std1),
                color="#0072B2",
                linewidth=2.0,
                label=rf"Run 1: $\mu$={mu1:.2f}, $\sigma$={std1:.2f}",
            )
            fit_records.append(
                {
                    "slice_x_cm": x_val,
                    "run": "run_1_v0",
                    "mean_y_cm": float(mu1),
                    "std_y_cm": float(std1),
                    "n_samples": int(len(y1)),
                }
            )

        if len(y2) > 0:
            ax.hist(y2, bins=15, density=True, color="#F0E442", alpha=0.5, edgecolor="none")
            mu2, std2 = norm.fit(y2)
            y_axis2 = np.linspace(y2.min() - 0.5, y2.max() + 0.5, 200)
            ax.plot(
                y_axis2,
                norm.pdf(y_axis2, mu2, std2),
                color="#E69F00",
                linewidth=2.0,
                label=rf"Run 2: $\mu$={mu2:.2f}, $\sigma$={std2:.2f}",
            )
            fit_records.append(
                {
                    "slice_x_cm": x_val,
                    "run": "run_2_impact",
                    "mean_y_cm": float(mu2),
                    "std_y_cm": float(std2),
                    "n_samples": int(len(y2)),
                }
            )

        ax.set_title(f"Slice at X = {x_val:.1f} cm", color=color, fontweight="bold", fontsize=10)
        ax.set_xlabel("Y Position (cm)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("trajectories_gaussian_fits.png", dpi=300)
    plt.show()

    # 5. Export Fits to JSON & CSV Text Files
    with open("gaussian_fits.json", "w") as f:
        json.dump(fit_records, f, indent=4)

    with open("gaussian_fits.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_x_cm", "run", "mean_y_cm", "std_y_cm", "n_samples"])
        writer.writeheader()
        writer.writerows(fit_records)

    print("\n[+] Gaussian fit parameters successfully exported to:")
    print("    - gaussian_fits.json")
    print("    - gaussian_fits.csv")


if __name__ == "__main__":
    main()
