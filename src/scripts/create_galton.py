"""Runner script for Galton board terrain simulations with 5-image output generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rough_slope_sim import (
    BallConfig,
    EnsembleConfig,
    PhysicsConfig,
    SimConfig,
    TerrainConfig,
    generate_terrain,
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

X_SLICES = [5.0, 10.0, 15.0]
CLOSEUP_X, CLOSEUP_Y = 5.0, 11.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Galton Board Simulation and Export 5 Key Plots")
    parser.add_argument("--out-dir", type=Path, default=Path("output/galton"), help="Output directory")
    parser.add_argument("--num-balls", type=int, default=150, help="Number of simulated balls")
    parser.add_argument("--slope-angle", type=float, default=20.0, help="Slope angle (degrees)")
    parser.add_argument("--ramp-length", type=float, default=29.0, help="Physical ramp length (cm)")
    parser.add_argument("--peg-radius", type=float, default=0.45, help="Peg radius (cm)")
    parser.add_argument("--peg-height", type=float, default=2.0, help="Peg height (cm)")
    parser.add_argument("--peg-dx", type=float, default=2.0, help="Peg X spacing (cm)")
    parser.add_argument("--peg-dy", type=float, default=2.0, help="Peg Y spacing (cm)")
    parser.add_argument(
        "--peg-shape", type=str, default="cylinder", choices=["paraboloid", "gaussian", "cylinder"]
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n==================================================")
    print(f" Galton Board Simulation | Angle: {args.slope_angle}°")
    print(f" Output Directory: {out_dir}")
    print("==================================================\n")

    # 1. Instantiate Terrain Config
    t_cfg = TerrainConfig(
        surface_type="galton",
        ramp_length=args.ramp_length,
        slope_angle=args.slope_angle,
        peg_radius=args.peg_radius,
        peg_height=args.peg_height,
        peg_dx=args.peg_dx,
        peg_dy=args.peg_dy,
        peg_shape=args.peg_shape,
        seed=42,
    )
    terrain = generate_terrain(t_cfg)

    # 2. Setup Initial Ball States
    rng = np.random.default_rng(42)
    y_min, y_max = 11.5, 11.5
    y0_vals = rng.uniform(y_min, y_max, size=args.num_balls)
    initial_states = [(0.1, float(y0), 0.0, 0.0) for y0 in y0_vals]

    ball_cfg = BallConfig(radius=0.125, x0=0.1)
    physics_cfg = PhysicsConfig(gravity=981.0)
    e_cfg = EnsembleConfig(k_max=args.num_balls, x_jitter_std=0, y_jitter_max=0, seed=42)
    print(e_cfg)
    # 3. Parallel Simulation Run
    sim_trajs = run_ensemble_parallel(
        terrain,
        ball_cfg,
        physics_cfg,
        sim_cfg=SimConfig(),
        ensemble_cfg=e_cfg,
        initial_states=initial_states,
        show_progress=True,
        desc=f"Simulating Galton Board ({args.num_balls} balls)",
    )
    sim_trajs = [t for t in sim_trajs if len(t.x) > 0]

    print("\nGenerating 5 Output Plots...")

    # Image 1: Ball-Surface Contact Close-up
    fig_closeup = plot_ball_surface_closeup(
        terrain=terrain,
        ball_radius=ball_cfg.radius,
        ball_x=CLOSEUP_X,
        ball_y=CLOSEUP_Y,
        window_factor=3.0,
    )
    fig_closeup.savefig(out_dir / "00_ball_surface_closeup.png", dpi=300, bbox_inches="tight")
    plt.close(fig_closeup)

    # Image 2: 3D Terrain with Surface Normals
    fig_terrain = plot_terrain_3d(terrain, quiver_skip=28)
    fig_terrain.savefig(out_dir / "01_terrain_3d_normals.png", dpi=150, bbox_inches="tight")
    plt.close(fig_terrain)

    # Image 3: 3D Trajectory Ensemble
    fig_traj3d = plot_trajectories_3d(terrain, sim_trajs)
    fig_traj3d.savefig(out_dir / "02_terrain_trajectories_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig_traj3d)

    # Image 4: Density Distribution Histogram at X = 15 cm
    y_sim_15 = trajectories_at_x_slice(sim_trajs, 15.0)
    fig_hist = plot_experiment_vs_sim_distribution(
        [], y_sim_15, x_slice=15.0, title="Galton Board Density (X = 15.0 cm)"
    )
    fig_hist.savefig(out_dir / "03_histogram_15cm.png", dpi=150, bbox_inches="tight")
    plt.close(fig_hist)

    # Image 5: 90° CW Rotated Trajectories and 3-Slice Profiles
    fig_slices = plot_trajectories_and_three_slices(
        [], sim_trajs, X_SLICES, is_dual=False, terrain=terrain, ball_radius=ball_cfg.radius
    )
    fig_slices.savefig(out_dir / "04_trajectories_and_3slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig_slices)

    diff_coeff = calculate_diffusion_coefficient(y_sim_15, x_slice=15.0)
    print(f"[✓] Completed successfully! Saved 5 plots to '{out_dir}'.")
    print(f"    └── Lateral Diffusion Coefficient D_sim = {diff_coeff:.4f} cm²/s\n")


if __name__ == "__main__":
    main()
