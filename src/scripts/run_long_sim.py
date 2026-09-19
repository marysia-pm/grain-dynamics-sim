"""Script for 500x50 cm large plane dual-grit simulation with trajectory slices and histograms."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def extract_coords(t) -> tuple[np.ndarray, np.ndarray]:
    """Extracts (x, y) arrays safely across different trajectory structures."""
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
    """Interpolates transverse Y positions across trajectories at a fixed downslope X plane."""
    y_vals = []
    for t in trajectories:
        x_arr, y_arr = extract_coords(t)
        if len(x_arr) > 1 and np.min(x_arr) <= target_x <= np.max(x_arr):
            sort_idx = np.argsort(x_arr)
            y_interp = float(np.interp(target_x, x_arr[sort_idx], y_arr[sort_idx]))
            if not np.isnan(y_interp):
                y_vals.append(y_interp)
    return np.array(y_vals, dtype=float)


def main():
    try:
        plt.style.use("https://raw.githubusercontent.com/turczyneq/softmatter-style/main/softmatter.mplstyle")
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    plane_length = 90.0
    plane_width = 30.0
    seam_y = 15.0

    # 1. Configurations using offset values
    terrain_cfg = TerrainConfig(
        ramp_length=plane_length,
        slope_angle=30.0,
        length_y=plane_width,
        resolution_x=450,
        resolution_y=150,
        grit_rough=80.0,
        grit_smooth=120.0,
        roughness_transition_y=seam_y,
        seed=42,
    )

    phys_cfg = PhysicsConfig(gravity=981.0)
    sim_cfg = SimConfig(dt=0.0005, t_max=5.0, num_workers=4)

    ball_cfg = BallConfig(
        x0=0.5,
        y0=seam_y,
        radius=0.125,
        mass=1.0,
    )

    ensemble_cfg = EnsembleConfig(
        k_max=300,
        start_x=0.1,
        start_y=seam_y,
        x_jitter_std=0.1,
        y_jitter_max=0.1,
        seed=42,
    )

    # 2. Build Terrain & Run
    print(f"[+] Generating large plane terrain ({plane_length:.0f} cm x {plane_width:.0f} cm)...")
    terrain = generate_terrain(terrain_cfg)

    print(f"[+] Simulating ensemble ({ensemble_cfg.k_max} trajectories)...")
    trajectories = run_ensemble_parallel(
        terrain=terrain,
        ball_cfg=ball_cfg,
        phys_cfg=phys_cfg,
        sim_cfg=sim_cfg,
        ensemble_cfg=ensemble_cfg,
        show_progress=True,
        desc="  ├── Large Plane Simulation",
    )

    trajectories = [t for t in trajectories if len(extract_coords(t)[0]) > 0]

    # 3. Multi-panel Figure Layout
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1.0], hspace=0.45, wspace=0.25)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_slice1 = fig.add_subplot(gs[0, 1])
    ax_slice2 = fig.add_subplot(gs[1, 1])
    ax_slice3 = fig.add_subplot(gs[2, 1])

    c_slice1 = "#d95f02"  # Red
    c_slice2 = "purple"
    c_slice3 = "forestgreen"

    # --- MAIN PANEL ---
    all_x_vals: list[np.ndarray] = []
    all_y_vals: list[np.ndarray] = []
    for t in trajectories:
        x_arr, y_arr = extract_coords(t)
        all_x_vals.append(x_arr)
        all_y_vals.append(y_arr)
        color = "#1e88e5"
        ax_main.plot(y_arr, x_arr, color=color, alpha=0.35, linewidth=0.8)

    slice_x_vals = [0.1, 10.0, 80.0]
    ax_main.axhline(
        slice_x_vals[0],
        color=c_slice1,
        linestyle="--",
        linewidth=1.5,
        label=f"Slice X={slice_x_vals[0]:.0f} cm",
    )
    ax_main.axhline(
        slice_x_vals[1],
        color=c_slice2,
        linestyle="--",
        linewidth=1.5,
        label=f"Slice X={slice_x_vals[1]:.0f} cm",
    )
    ax_main.axhline(
        slice_x_vals[2],
        color=c_slice3,
        linestyle="--",
        linewidth=1.5,
        label=f"Slice X={slice_x_vals[2]:.0f} cm",
    )

    ax_main.axvline(
        seam_y, color="black", linestyle="-.", linewidth=1.8, label=f"Seam Interface (Y = {seam_y:.0f} cm)"
    )

    ax_main.set_ylabel("Downslope Position x (cm)", fontweight="bold", fontsize=10)
    ax_main.set_xlabel("Lateral Position y (cm)", fontweight="bold", fontsize=10)

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

    # Auto-scale to the actual data (plus the slice/seam reference lines, so those
    # stay visible) instead of the full (0, plane_width)/(550, 0) domain.
    x_lo, x_hi = _auto_limits(*all_y_vals, [seam_y], fallback=(0.0, plane_width))
    y_lo, y_hi = _auto_limits(*all_x_vals, slice_x_vals, fallback=(0.0, 550.0))
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylim(y_hi, y_lo)  # downslope points downwards
    ax_main.set_title(
        "Large Surface Simulation (80P vs 240P)",
        fontweight="bold",
        pad=12,
        fontsize=15,
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # --- RIGHT SLICE HISTOGRAM PANELS ---
    slice_axes = [ax_slice1, ax_slice2, ax_slice3]
    slice_colors = [c_slice1, c_slice2, c_slice3]

    for x_val, ax, color in zip(slice_x_vals, slice_axes, slice_colors):
        y_slice = extract_y_at_x(trajectories, x_val)
        y_slice = y_slice[~np.isnan(y_slice)]

        slice_lo, slice_hi = _auto_limits(y_slice, [seam_y], fallback=(5.0, plane_width - 5.0))
        if len(y_slice) > 0:
            bins = np.linspace(slice_lo, slice_hi, 41)
            ax.hist(y_slice, bins=bins, density=True, color="#56B4E9", alpha=0.65, edgecolor="none")

        ax.axvline(seam_y, color="black", linestyle="-.", linewidth=1.5)
        ax.set_title(f"Slice at X = {x_val:.0f} cm", color=color, fontweight="bold", fontsize=13)
        ax.set_xlabel("Lateral Position y (cm)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(slice_lo, slice_hi)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("large_plane_trajectories.png", dpi=300)
    print("[+] Saved visualization to 'large_plane_trajectories_80_Vs_120.png'.")


if __name__ == "__main__":
    main()
