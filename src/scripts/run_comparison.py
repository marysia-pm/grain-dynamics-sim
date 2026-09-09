"""Complete runner script with calibrated grit amplitude scaling and X-axis track zero-alignment."""

from __future__ import annotations

import argparse
import glob
import os
import re
import torch
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from rough_slope_sim import (
    Terrain,
    BallConfig,
    EnsembleConfig,
    PhysicsConfig,
    TerrainConfig,
    generate_terrain,
    generate_terrain_gpu,
    run_ensemble_parallel,
)
from rough_slope_sim.analysis import (
    calculate_diffusion_coefficient,
    compare_distributions,
    trajectories_at_x_slice,
)
from rough_slope_sim.plotting import (
    plot_ball_surface_closeup,
    plot_experiment_vs_sim_distribution,
    plot_terrain_3d,
    plot_trajectories_3d,
    plot_trajectories_and_three_slices,
)
from rough_slope_sim.terrain import nanovea_d50_from_grit

PX_TO_CM = 0.01185
X_SLICES = [5.0, 10.0, 15.0]
CLOSEUP_X, CLOSEUP_Y = 5.0, 15.0


@dataclass
class SurfaceConfig:
    folder_name: str
    grit_left: float
    grit_right: float
    start_y: float
    is_dual: bool


def grit_to_amplitude(grit: float) -> float:
    """Converts grit (P-value) to grain amplitude d50 (in cm) via Nanovea profilometer lookup."""
    return nanovea_d50_from_grit(grit)


def parse_folder(folder_name: str, y_mid: float = 11.5) -> SurfaceConfig:
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
    if max_v > 290.0:
        return PX_TO_CM
    if max_v > 29.0:
        return 0.1
    return 1.0


<<<<<<< HEAD
def load_trajectories(folder: Path, ramp_length_proj_cm: float, target_x0: float = 0.1) -> list[np.ndarray]:
    """Loads experimental trajectories, fixes orientation, and aligns release points to target_x0."""
=======
def load_trajectories(
    folder: Path, ramp_length_proj_cm: float, target_x_start: float = 0.1
) -> list[np.ndarray]:
    """Loads experimental trajectories and aligns release point to X ~ target_x_start cm."""
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
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

    # Orient trajectory direction so downslope is positive X
    if start_x_mean > end_x_mean or start_x_mean > (ramp_length_proj_cm / 2.0):
        for t in trajs:
            t[:, 1] = ramp_length_proj_cm - t[:, 1]

<<<<<<< HEAD
    # Align experimental tracks along X so every trajectory release point starts at target_x0
    for t in trajs:
        x_start = t[0, 1]
        t[:, 1] = t[:, 1] - x_start + target_x0
=======
    # Shift each trajectory so its initial release point starts at target_x_start (0.1 cm)
    for t in trajs:
        t[:, 1] = (t[:, 1] - t[0, 1]) + target_x_start
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822

    return trajs


def sample_hybrid_y0(
    y_min: float,
    y_max: float,
    num_samples: int,
    seed: int = 42,
    blend_factor: float = 0.5,
) -> np.ndarray:
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
<<<<<<< HEAD
=======
    """Runs the full experiment-vs-simulation comparison for one experimental folder."""
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
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

<<<<<<< HEAD
    exp_trajs = [
        t for t in load_trajectories(exp_folder, ramp_length_proj_cm=x_max_proj, target_x0=0.1) if len(t) > 0
=======
    # 1. Load experimental trajectories zero-aligned to X0 = 0.1 cm
    target_x0 = 0.1
    exp_trajs = [
        t for t in load_trajectories(exp_folder, ramp_length_proj_cm=x_max_proj, target_x_start=target_x0)
        if len(t) > 0
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
    ]
    if not exp_trajs:
        tqdm.write("  └── [!] No valid trajectories found. Skipping.")
        return None

    # 2. Extract initial lateral positions (Y)
    exp_y_starts = [t[0, 2] for t in exp_trajs if len(t) > 0]
    y_min, y_max = float(np.min(exp_y_starts)), float(np.max(exp_y_starts))

    # 3. Generate 3D Terrain directly on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, Z, dz_dx, dz_dy = generate_terrain_gpu(
        x_min=0.0,
        x_max=x_max_proj,
        y_min=0.0,
        y_max=23.0,  # Full 23 cm plane width
        dx=0.02,
        dy=0.02,
        slope_angle_deg=slope_angle_deg,
        p_value_left=cfg.grit_left,
        p_value_right=cfg.grit_right,
        is_dual=cfg.is_dual,
        device=device,
    )

    terrain = Terrain(
        x=X,
        y=Y,
        z=Z,
        dz_dx=dz_dx,
        dz_dy=dz_dy,
        slope_angle=slope_angle_deg,
        p_value_rough=cfg.grit_left,
        p_value_smooth=cfg.grit_right,
    )

    # 4. Configure initial states for simulation ensemble
    num_sim_balls = max(len(exp_trajs), 150)
    sim_y0_vals = sample_hybrid_y0(y_min, y_max, num_samples=num_sim_balls)
    sampled_initial_states = [(target_x0, float(y0), 0.0, 0.0) for y0 in sim_y0_vals]

<<<<<<< HEAD
    tqdm.write(f"  ├── Loaded Tracks  : {len(exp_trajs)} trajectories (X release aligned to 0.1 cm)")
=======
    tqdm.write(f"  ├── Loaded Tracks  : {len(exp_trajs)} experimental trajectories")
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
    tqdm.write(
        f"  ├── Initial Config : Fixed X0 = {target_x0} cm, V0 = 0.0 | Hybrid Y0 in [{y_min:.2f}, {y_max:.2f}] cm"
    )

<<<<<<< HEAD
    # Construct TerrainConfig passing explicit grit values along with amplitudes
    t_cfg = TerrainConfig(
        ramp_length=ramp_length_cm,
        slope_angle=slope_angle_deg,
        roughness_amplitude_rough=grit_to_amplitude(cfg.grit_left),
        roughness_amplitude_smooth=grit_to_amplitude(cfg.grit_right),
        grit_rough=cfg.grit_left,
        grit_smooth=cfg.grit_right,
        roughness_transition_y=11.5,
        seed=42,
    )
    terrain = generate_terrain(t_cfg)

    tqdm.write(f"  ├── Rough side     : {terrain.p_value_rough:.0f}P")
    tqdm.write(f"  ├── Smooth side    : {terrain.p_value_smooth:.0f}P")
    tqdm.write(f"  ├── Overall surface: {terrain.p_value_mean:.0f}P")

    ball_cfg = BallConfig(radius=0.125, x0=0.1)
    physics_cfg = PhysicsConfig(gravity=981.0, slope_angle=slope_angle_deg)
    e_cfg = EnsembleConfig(k_max=num_sim_balls, seed=42)
=======
    ball_cfg = BallConfig()
    phys_cfg = PhysicsConfig(slope_angle=slope_angle_deg)
    ens_cfg = EnsembleConfig(k_max=num_sim_balls)
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822

    # 5. Run GPU-accelerated ensemble simulation
    sim_trajs = run_ensemble_parallel(
        terrain=terrain,
        ball=ball_cfg,
        physics=phys_cfg,
        ensemble=ens_cfg,
        initial_states=sampled_initial_states,
        show_progress=True,
        use_gpu=True,
        desc="  ├── Simulating ensemble",
    )

    # 6. Calculate diffusion coefficients for both experiment and simulation
    exp_diff = calculate_diffusion_coefficients(exp_trajs, fps=60.0, t_min=0.05, t_max=0.70)
    sim_diff = calculate_diffusion_coefficients(sim_trajs, fps=60.0, t_min=0.05, t_max=0.70)

    tqdm.write("  ├── Diffusion Coefficients (Dy):")
    tqdm.write(
        f"  │   ├── Exp Dy = {exp_diff['D_y']:.4f} cm²/s (Ds = {exp_diff['D_s']:.4f} cm²/cm, R² = {exp_diff['r_squared']:.3f})"
    )
    tqdm.write(
        f"  │   └── Sim Dy = {sim_diff['D_y']:.4f} cm²/s (Ds = {sim_diff['D_s']:.4f} cm²/cm, R² = {sim_diff['r_squared']:.3f})"
    )

    # 7. Render and save all plots (Softmatter style, full domain plane, no tight_layout)
    x_slices = [5.0, 15.0, 25.0]

    # Plot 1: Full 23x29 cm Trajectories & Cross-Section Slices
    fig_trajs = plot_trajectories_and_three_slices(
        exp_trajs=exp_trajs,
        sim_trajs=sim_trajs,
        x_slices=x_slices,
        is_dual=cfg.is_dual,
        interface_y=11.5,
    )
    fig_trajs.savefig(sub_out / "trajectories_comparison.png", dpi=300)
    plt.close(fig_trajs)

    # Plot 2: Full 3D Terrain
    fig_terrain = plot_terrain_3d(terrain)
    fig_terrain.savefig(sub_out / "terrain_3d.png", dpi=300)
    plt.close(fig_terrain)

    # Plot 3: 3D Trajectories over Surface
    fig_traj3d = plot_trajectories_3d(terrain, sim_trajs)
    fig_traj3d.savefig(sub_out / "trajectories_3d.png", dpi=300)
    plt.close(fig_traj3d)

    # Plot 4: Contact Close-Up (Ball rendered on top of grains)
    fig_closeup = plot_ball_surface_closeup(
        terrain=terrain,
        ball_radius=ball_cfg.radius,
        ball_x=10.0,
        ball_y=11.5 if cfg.is_dual else 10.0,
    )
    fig_closeup.savefig(sub_out / "ball_surface_closeup.png", dpi=300)
    plt.close(fig_closeup)

    # Plot 5: Ensemble Lateral Variance Growth
    fig_var = plot_variance_over_time(sim_diff["t_grid"], sim_diff["var_y_t"])
    fig_var.savefig(sub_out / "variance_over_time.png", dpi=300)
    plt.close(fig_var)

    tqdm.write("  └── [✓] Analysis & figures generated successfully.")

    return {
        "folder": folder_name,
        "grit_left": cfg.grit_left,
        "grit_right": cfg.grit_right,
        "is_dual": cfg.is_dual,
        "exp_diff": exp_diff,
        "sim_diff": sim_diff,
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
