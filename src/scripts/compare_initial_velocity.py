"""Script with Gaussian PDF fitting, overlay plotting, and parameter export aligned with experimental metadata."""

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Ensure project root / src directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path_entry in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from rough_slope_sim.config import (
    BallConfig,
    EnsembleConfig,
    PhysicsConfig,
    SimConfig,
    TerrainConfig,
)
from rough_slope_sim.simulation import run_ensemble_parallel
from rough_slope_sim.terrain import generate_terrain


def calculate_free_fall_velocity(height_mm: float = 0.5, g: float = 981.0) -> float:
    return -float(np.sqrt(2.0 * g * (height_mm / 10.0)))


def extract_trajectory_coords(t) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(t, "x") and hasattr(t, "y"):
        return np.asarray(t.x, dtype=float), np.asarray(t.y, dtype=float)
    elif hasattr(t, "positions"):
        pos = np.asarray(t.positions, dtype=float)
        if pos.ndim == 2 and pos.shape[1] >= 3:
            return pos[:, 1], pos[:, 2]
        return pos[:, 0], pos[:, 1]
    else:
        pos = np.asarray(t, dtype=float)
        if pos.ndim == 2 and pos.shape[1] >= 3:
            return pos[:, 1], pos[:, 2]
        elif pos.ndim == 2 and pos.shape[1] == 2:
            return pos[:, 0], pos[:, 1]
        return np.array([]), np.array([])


def extract_y_at_x(trajectories, target_x: float) -> np.ndarray:
    y_vals = []
    for t in trajectories:
        x_arr, y_arr = extract_trajectory_coords(t)
        if len(x_arr) > 1 and np.min(x_arr) <= target_x <= np.max(x_arr):
            sort_idx = np.argsort(x_arr)
            y_interp = float(np.interp(target_x, x_arr[sort_idx], y_arr[sort_idx]))
            if not np.isnan(y_interp):
                y_vals.append(y_interp)
    return np.array(y_vals, dtype=float)


def main():
    # 1. Dataclasses
    terrain_cfg = TerrainConfig(
        ramp_length=29.0,
        slope_angle=30.0,
        length_y=23.0,
        grit_rough=120.0,
        grit_smooth=120.0,
        roughness_transition_y=100,
        seed=42,
    )
    phys_cfg = PhysicsConfig(gravity=981.0)
    sim_cfg = SimConfig(dt=0.001, t_max=3.0)

    v_z_impact = calculate_free_fall_velocity(height_mm=0.5, g=phys_cfg.gravity)

    ensemble_cfg = EnsembleConfig(
        k_max=150,
        start_x=0.1,
        start_y=11.5,
        x_jitter_std=0.05,
        y_jitter_max=0.5,
        seed=42,
    )

    ball_cfg_zero = BallConfig(
        x0=0.1,
        y0=100,
        radius=0.125,
        mass=1.0,
        vx0=0.0,
        vy0=0.0,
        vz0=0.0,
    )
    ball_cfg_impact = BallConfig(
        x0=0.1,
        y0=100,
        radius=0.125,
        mass=1.0,
        vx0=0.0,
        vy0=0.0,
        vz0=v_z_impact,
    )

    # 2. Build Terrain & Run
    print("[+] Generating terrain profile...")
    terrain = generate_terrain(terrain_cfg)

    print("[+] Simulating Run 1 (v_z0 = 0)...")
    trajs_zero = run_ensemble_parallel(
        terrain=terrain,
        ball_cfg=ball_cfg_zero,
        phys_cfg=phys_cfg,
        sim_cfg=sim_cfg,
        ensemble_cfg=ensemble_cfg,
        show_progress=True,
        desc="  ├── Run 1 (v_z0 = 0)",
    )

    print(f"[+] Simulating Run 2 (v_z0 = {v_z_impact:.2f} cm/s)...")
    trajs_impact = run_ensemble_parallel(
        terrain=terrain,
        ball_cfg=ball_cfg_impact,
        phys_cfg=phys_cfg,
        sim_cfg=sim_cfg,
        ensemble_cfg=ensemble_cfg,
        show_progress=True,
        desc=f"  ├── Run 2 (v_z0 = {v_z_impact:.2f})",
    )

    trajs_zero = [t for t in trajs_zero if len(extract_trajectory_coords(t)[0]) > 0]
    trajs_impact = [t for t in trajs_impact if len(extract_trajectory_coords(t)[0]) > 0]

    # 3. Figure Layout & Plotting
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1.0], hspace=0.35, wspace=0.25)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_slice1 = fig.add_subplot(gs[0, 1])
    ax_slice2 = fig.add_subplot(gs[1, 1])
    ax_slice3 = fig.add_subplot(gs[2, 1])

    color_run1 = "#9ecae1"
    color_run2 = "#ffe082"
    c_slice1 = "#d95f02"
    c_slice2 = "purple"
    c_slice3 = "forestgreen"

    all_x_vals: list[np.ndarray] = []
    all_y_vals: list[np.ndarray] = []

    for t in trajs_zero:
        x_arr, y_arr = extract_trajectory_coords(t)
        all_x_vals.append(x_arr)
        all_y_vals.append(y_arr)
        ax_main.plot(y_arr, x_arr, color=color_run1, alpha=0.35, linewidth=0.8)

    for t in trajs_impact:
        x_arr, y_arr = extract_trajectory_coords(t)
        all_x_vals.append(x_arr)
        all_y_vals.append(y_arr)
        ax_main.plot(y_arr, x_arr, color=color_run2, alpha=0.35, linewidth=0.8)

    slice_x_vals = [5.0, 10.0, 15.0]
    ax_main.axhline(slice_x_vals[0], color=c_slice1, linestyle="--", linewidth=1.5, label="Slice X=5.0 cm")
    ax_main.axhline(slice_x_vals[1], color=c_slice2, linestyle="--", linewidth=1.5, label="Slice X=10.0 cm")
    ax_main.axhline(slice_x_vals[2], color=c_slice3, linestyle="--", linewidth=1.5, label="Slice X=15.0 cm")

    ax_main.set_title("Trajectories (Rotated 90° CW: Downslope Downwards)", fontsize=11, fontweight="bold")
    ax_main.set_xlabel("Y Position (cm) [Lateral]", fontsize=10)
    ax_main.set_ylabel("X Position (cm) [Downslope]", fontsize=10)

    # Auto-scale to the actual data (plus the slice reference lines, so those stay
    # visible) instead of a hardcoded (0, 23)/(27, -1) -- a narrow lateral spread
    # used to render as a thin sliver in a mostly-empty plot.
    def _auto_limits(*groups, pad_frac=0.15, min_pad=0.5, fallback=(0.0, 1.0)):
        chunks = [np.asarray(g, dtype=float) for g in groups if len(g)]
        chunks = [c[np.isfinite(c)] for c in chunks]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return fallback
        vals = np.concatenate(chunks)
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * pad_frac, min_pad)
        return lo - pad, hi + pad

    y_lo, y_hi = _auto_limits(*all_x_vals, slice_x_vals, fallback=(0.0, 27.0))
    x_lo, x_hi = _auto_limits(*all_y_vals, fallback=(0.0, 23.0))
    ax_main.set_ylim(y_hi, y_lo)  # inverted: downslope increases downward
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_main.grid(True, linestyle=":", alpha=0.5)

    # 4. Slices & Fits
    slice_axes = [ax_slice1, ax_slice2, ax_slice3]
    slice_colors = [c_slice1, c_slice2, c_slice3]
    fit_records = []

    for x_val, ax, color in zip(slice_x_vals, slice_axes, slice_colors):
        y1 = extract_y_at_x(trajs_zero, x_val)
        y2 = extract_y_at_x(trajs_impact, x_val)

        y1 = y1[~np.isnan(y1)]
        y2 = y2[~np.isnan(y2)]

        if len(y1) > 0:
            ax.hist(y1, bins=15, density=True, color="#56B4E9", alpha=0.5, edgecolor="none")
            mu1, std1 = norm.fit(y1) if len(y1) > 1 and np.std(y1) > 1e-6 else (float(np.mean(y1)), 0.001)
            y_axis1 = np.linspace(mu1 - 3 * max(std1, 0.5), mu1 + 3 * max(std1, 0.5), 200)
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
            mu2, std2 = norm.fit(y2) if len(y2) > 1 and np.std(y2) > 1e-6 else (float(np.mean(y2)), 0.001)
            y_axis2 = np.linspace(mu2 - 3 * max(std2, 0.5), mu2 + 3 * max(std2, 0.5), 200)
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

    # 5. Export
    with open("gaussian_fits.json", "w") as f:
        json.dump(fit_records, f, indent=4)

    with open("gaussian_fits.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_x_cm", "run", "mean_y_cm", "std_y_cm", "n_samples"])
        writer.writeheader()
        writer.writerows(fit_records)

    print("\n[+] Exported fits to 'gaussian_fits.json' and 'gaussian_fits.csv'.")


if __name__ == "__main__":
    main()
