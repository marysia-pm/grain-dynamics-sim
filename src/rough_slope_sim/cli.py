"""CLI entry point for executing simulations and generating analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rough_slope_sim import analysis, plotting
from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import run_ensemble_parallel
from rough_slope_sim.terrain import generate_terrain


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def run_pipeline(out_dir: Path, seed: int | None) -> None:
    terrain_config = TerrainConfig(roughness_transition_y=2.5, seed=seed if seed is not None else 42)
    ball, physics = BallConfig(), PhysicsConfig()
    physics.slope_angle = terrain_config.slope_angle
    ensemble = EnsembleConfig(seed=seed if seed is not None else 42, k_max=200)

    print("Generating terrain...")
    terrain = generate_terrain(terrain_config)
    _save(plotting.plot_terrain_3d(terrain), out_dir, "01_terrain")

    print(f"Running parallel ensemble ({ensemble.k_max} runs)...")
    trajectories = run_ensemble_parallel(terrain, ball, physics, ensemble)
    _save(plotting.plot_trajectories_3d(terrain, trajectories), out_dir, "02_trajectories")

    variance = analysis.variance_over_time(trajectories, attr="y")
    time_axis = physics.dt * np.arange(len(variance))
    _save(plotting.plot_variance_over_time(time_axis, variance), out_dir, "03_variance_over_time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rough slope simulation pipeline.")
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_pipeline(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
