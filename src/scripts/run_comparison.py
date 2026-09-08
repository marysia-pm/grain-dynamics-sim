"""Complete runner script with centralized angle configuration and dynamic terrain projection."""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from rough_slope_sim import (
    BallConfig,
    EnsembleConfig,
    PhysicsConfig,
    TerrainConfig,
    generate_terrain,
    run_ensemble_parallel,
)
from rough_slope_sim.analysis import (
    calculate_diffusion_coefficient,
    compare_distributions,
    particle_size_from_grit,
    trajectories_at_x_slice,
)
from rough_slope_sim.plotting import (
    plot_ball_surface_closeup,
    plot_experiment_vs_sim_distribution,
    plot_terrain_3d,
    plot_trajectories_3d,
    plot_trajectories_and_three_slices,
)

# 1 px = 0.01185 cm. This is a fixed calibration constant for the camera setup
# used to record experiments; it is NOT a general unit-detection mechanism.
PX_TO_CM = 0.01185

X_SLICES = [5.0, 10.0, 15.0]
CLOSEUP_X, CLOSEUP_Y = 5.0, 15.0  # where to render the ball/surface close-up


@dataclass
class SurfaceConfig:
    folder_name: str
    grit_left: float
    grit_right: float
    start_y: float
    is_dual: bool


def grit_to_amplitude(grit: float) -> float:
    """Converts grit (P-value) to peak-to-trough surface amplitude (particle diameter in cm)."""
    p_size_um = particle_size_from_grit(grit)
    return float(p_size_um / 10000.0)


def parse_folder(folder_name: str, y_mid: float = 11.5) -> SurfaceConfig:
    """Parses folder naming conventions to identify surface grit parameters and layout."""
    dual_match = re.search(r"(\d+)[\\/](\d+)P_(mid|tilted(\d*))", folder_name, re.IGNORECASE)
    if dual_match:
        g_l, g_r = float(dual_match.group(1)), float(dual_match.group(2))
        pos_type, tilt_str = dual_match.group(3).lower(), dual_match.group(4)
        if pos_type == "mid":
            start_y = y_mid
        else:
            tilt = int(tilt_str) if tilt_str else 1
            shift = -1.0 if g_l < g_r else 1.0
            start_y = y_mid + (shift * tilt * 1.0)
        return SurfaceConfig(folder_name, g_l, g_r, start_y, is_dual=(g_l != g_r))

    single_match = re.search(r"(\d+)P", folder_name, re.IGNORECASE)
    if single_match:
        g = float(single_match.group(1))
        return SurfaceConfig(folder_name, g, g, y_mid, is_dual=False)

    return SurfaceConfig(folder_name, 2000.0, 2000.0, y_mid, is_dual=False)


def _guess_scale(max_v: float) -> float:
    """Heuristic unit guess based on the magnitude of raw track coordinates.

    NOTE: this is a magnitude threshold, not a real unit check -- it assumes
    trajectories always span roughly the same physical range. If a future
    dataset has a different camera FOV or ramp size, these thresholds (290,
    29) may misclassify units silently. Worth replacing with an explicit
    per-folder unit tag if this ever bites you.
    """
    if max_v > 290.0:
        return PX_TO_CM
    if max_v > 29.0:
        return 0.1
    return 1.0


def load_trajectories(folder: Path, ramp_length_proj_cm: float) -> list[np.ndarray]:
    """Loads experimental trajectories and aligns release point to X ~ 0 cm."""
    trajs = []
    column_renames = {"particle": "ball_id", "track_id": "ball_id", "pos_x": "x", "pos_y": "y"}

    for txt in glob.glob(os.path.join(folder, "*.txt")):
        try:
            df = pd.read_csv(txt, sep=r"\s+|,|;", engine="python")
            df.columns = [column_renames.get(c.lower().strip(), c.lower().strip()) for c in df.columns]

            if not all(c in df.columns for c in ["ball_id", "frame", "x", "y"]):
                continue

            max_v = max(df["x"].max(), df["y"].max())
            scale = _guess_scale(max_v)

            for _, group in df.groupby("ball_id"):
                group = group.sort_values("frame")
                t = group[["frame", "x", "y"]].to_numpy(dtype=np.float64)
                if len(t) > 0:
                    t[:, 1] *= scale
                    t[:, 2] *= scale
                    trajs.append(t)
        except Exception:
            continue

    if not trajs:
        return []

    start_x_mean = float(np.mean([t[0, 1] for t in trajs]))
    end_x_mean = float(np.mean([t[-1, 1] for t in trajs]))

    if start_x_mean > end_x_mean or start_x_mean > (ramp_length_proj_cm / 2.0):
        for t in trajs:
            t[:, 1] = ramp_length_proj_cm - t[:, 1]

    return trajs


def sample_hybrid_y0(
    y_min: float,
    y_max: float,
    num_samples: int,
    seed: int = 42,
    blend_factor: float = 0.5,
) -> np.ndarray:
    """Samples Y0 using a hybrid distribution (50% Uniform + 50% Truncated Gaussian)."""
    rng = np.random.default_rng(seed)

    u_samples = rng.uniform(y_min, y_max, size=num_samples)

    y_mid = (y_min + y_max) / 2.0
    y_std = (y_max - y_min) / 4.0
    g_samples = rng.normal(y_mid, y_std, size=num_samples)
    g_samples = np.clip(g_samples, y_min, y_max)

    return blend_factor * u_samples + (1.0 - blend_factor) * g_samples


def process_folder(
    exp_folder: Path,
    out_root: Path,
    slope_angle_deg: float,
    ramp_length_cm: float,
    x_max_proj: float,
) -> dict | None:
    """Runs the full experiment-vs-simulation comparison for one folder. Returns a
    summary record dict, or None if the folder had no usable trajectories."""
    folder_name = exp_folder.name
    cfg = parse_folder(folder_name)

    surface_info = (
        f"Dual Grit ({cfg.grit_left:.0f}P / {cfg.grit_right:.0f}P)"
        if cfg.is_dual
        else f"Single Grit ({cfg.grit_left:.0f}P)"
    )

    tqdm.write(f"\n[+] Processing Folder : {folder_name}")
    tqdm.write(f"  ├── Surface Config : {surface_info}")

    sub_out = out_root / folder_name.replace("\\", "_").replace("/", "_")
    sub_out.mkdir(parents=True, exist_ok=True)

    exp_trajs = [t for t in load_trajectories(exp_folder, ramp_length_proj_cm=x_max_proj) if len(t) > 0]
    if not exp_trajs:
        tqdm.write("  └── [!] No valid trajectories found. Skipping.")
        return None

    exp_y_starts = [t[0, 2] for t in exp_trajs if len(t) > 0]
    y_min, y_max = float(np.min(exp_y_starts)), float(np.max(exp_y_starts))

    num_sim_balls = max(len(exp_trajs), 150)
    sim_y0_vals = sample_hybrid_y0(y_min, y_max, num_samples=num_sim_balls)
    sampled_initial_states = [(0.1, float(y0), 0.0, 0.0) for y0 in sim_y0_vals]

    tqdm.write(f"  ├── Loaded Tracks  : {len(exp_trajs)} trajectories")
    tqdm.write(
        f"  ├── Initial Config : Fixed X0 = 0.1 cm, V0 = 0.0 | Hybrid Y0 in [{y_min:.2f}, {y_max:.2f}] cm"
    )

    t_cfg = TerrainConfig(
        ramp_length=ramp_length_cm,
        slope_angle=slope_angle_deg,
        roughness_amplitude_rough=grit_to_amplitude(cfg.grit_left),
        roughness_amplitude_smooth=grit_to_amplitude(cfg.grit_right),
        roughness_transition_y=11.5,
        seed=42,
    )
    terrain = generate_terrain(t_cfg)

    tqdm.write(f"  ├── Rough side     : {terrain.p_value_rough:.0f}P")
    tqdm.write(f"  ├── Smooth side    : {terrain.p_value_smooth:.0f}P")
    tqdm.write(f"  ├── Overall surface: {terrain.p_value_mean:.0f}P")

    ball_cfg = BallConfig(radius=0.125)
    physics_cfg = PhysicsConfig(slope_angle=slope_angle_deg)
    e_cfg = EnsembleConfig(k_max=num_sim_balls, seed=42)

    sim_trajs = run_ensemble_parallel(
        terrain,
        ball_cfg,
        physics_cfg,
        e_cfg,
        initial_states=sampled_initial_states,
        show_progress=True,
        desc=f"  ├── Simulating ({num_sim_balls} balls)",
    )
    sim_trajs = [t for t in sim_trajs if len(t.x) > 0]

    tqdm.write("  └── Generating plots...")

    fig_closeup = plot_ball_surface_closeup(
        terrain=terrain,
        ball_radius=ball_cfg.radius,
        ball_x=CLOSEUP_X,
        ball_y=CLOSEUP_Y,
        window_factor=3.0,
    )
    fig_closeup.savefig(sub_out / "00_ball_surface_closeup.png", dpi=300, bbox_inches="tight")
    plt.close(fig_closeup)

    fig_terrain = plot_terrain_3d(terrain, quiver_skip=28)
    fig_terrain.savefig(sub_out / "01_terrain_3d_normals.png", dpi=150, bbox_inches="tight")
    plt.close(fig_terrain)

    fig_traj3d = plot_trajectories_3d(terrain, sim_trajs)
    fig_traj3d.savefig(sub_out / "02_terrain_trajectories_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig_traj3d)

    y_exp_15 = trajectories_at_x_slice(exp_trajs, 15.0)
    y_sim_15 = trajectories_at_x_slice(sim_trajs, 15.0)

    fig_hist = plot_experiment_vs_sim_distribution(y_exp_15, y_sim_15, x_slice=15.0)
    fig_hist.savefig(sub_out / "03_histogram_15cm.png", dpi=150, bbox_inches="tight")
    plt.close(fig_hist)

    fig_slices = plot_trajectories_and_three_slices(
        exp_trajs, sim_trajs, X_SLICES, is_dual=cfg.is_dual, interface_y=11.5
    )
    fig_slices.savefig(sub_out / "04_trajectories_and_3slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig_slices)

    d_exp = calculate_diffusion_coefficient(y_exp_15, x_slice=15.0)
    d_sim = calculate_diffusion_coefficient(y_sim_15, x_slice=15.0)
    metrics = compare_distributions(y_exp_15, y_sim_15)

    return {
        "folder": folder_name,
        "is_dual": cfg.is_dual,
        "slope_angle_deg": slope_angle_deg,
        "x_max_proj_cm": x_max_proj,
        "exp_start_y_min": y_min,
        "exp_start_y_max": y_max,
        "D_exp_cm2_s": d_exp,
        "D_sim_cm2_s": d_sim,
        "wasserstein_cm": metrics["wasserstein_distance"],
        "ks_stat": metrics["ks_statistic"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-compare experiment vs. simulation for each surface.")
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("../grain-dynamics-analysis/processing_results"),
        help="Directory containing one subfolder per experiment.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--slope-angle", type=float, default=30.0, help="Incline angle in degrees")
    parser.add_argument("--ramp-length", type=float, default=29.0, help="Physical ramp length (cm)")
    args = parser.parse_args()

    t_cfg_template = TerrainConfig(ramp_length=args.ramp_length, slope_angle=args.slope_angle)
    x_max_proj = t_cfg_template.length_x

    if not args.exp_root.exists():
        print(f"[X] Error: Directory '{args.exp_root}' not found.")
        return

    exp_folders = [f for f in sorted(args.exp_root.iterdir()) if f.is_dir()]
    print("\n==================================================")
    print(f" Incline Angle: {args.slope_angle}° | Projected Length X_max = {x_max_proj:.2f} cm")
    print(f" Processing {len(exp_folders)} experiment folders.")
    print("==================================================\n")

    summary_records = []
    for exp_folder in tqdm(exp_folders, desc="Processing Experiments"):
        record = process_folder(
            exp_folder,
            out_root=args.out_dir,
            slope_angle_deg=args.slope_angle,
            ramp_length_cm=args.ramp_length,
            x_max_proj=x_max_proj,
        )
        if record is not None:
            summary_records.append(record)

    if summary_records:
        out_csv = args.out_dir / "batch_summary_metrics.csv"
        pd.DataFrame(summary_records).to_csv(out_csv, index=False)
        print(f"\n[✓] Complete! Results exported to: {out_csv}\n")


if __name__ == "__main__":
    main()
