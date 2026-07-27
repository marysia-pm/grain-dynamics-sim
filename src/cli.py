"""Command-line entry point.

Runs the full pipeline that the original notebook ran top-to-bottom:

1. generate a terrain and plot it,
2. run an ensemble of ball trajectories down the slope and plot them,
3. plot energy, final-position distribution, and variance/diffusion
   diagnostics,
4. fit the grit/roughness calibration curves,
5. sweep terrain roughness and plot the resulting diffusion coefficient.

Figures are written as PNG files to an output directory (default
``./output``) since ``plt.show()`` blocking calls don't make sense in a
script. Pass ``--show`` to also open interactive windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rough_slope_sim import analysis, plotting
from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import run_ensemble
from rough_slope_sim.terrain import generate_terrain


def _save(fig, out_dir: Path, name: str, show: bool) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_pipeline(out_dir: Path, show: bool, seed: int | None) -> None:
    terrain_config = TerrainConfig(roughness_transition_y=2.5, seed=seed)
    ball = BallConfig()
    physics = PhysicsConfig()
    ensemble = EnsembleConfig(seed=seed)

    print("Generating terrain...")
    terrain = generate_terrain(terrain_config)
    _save(plotting.plot_terrain_3d(terrain), out_dir, "01_terrain", show)

    print(f"Running ensemble of {ensemble.k_max} ball simulations...")
    trajectories = run_ensemble(terrain, ball, physics, ensemble)
    _save(plotting.plot_trajectories_3d(terrain, trajectories), out_dir, "02_trajectories", show)

    longest = max(trajectories, key=lambda t: len(t.time))
    _save(
        plotting.plot_energies(longest, ball.mass, physics.g, ball.radius),
        out_dir, "03_energy", show,
    )

    final_y, final_z = analysis.final_positions(trajectories)
    _save(
        plotting.plot_final_position_distribution(final_y, final_z, terrain_config.roughness_transition_y),
        out_dir, "04_final_position_distribution", show,
    )

    variance = analysis.variance_over_time(trajectories, attr="y")
    time_axis = np.arange(0, physics.total_time, physics.dt)[: len(variance)]
    _save(plotting.plot_variance_over_time(time_axis, variance), out_dir, "05_variance_over_time", show)

    counts = analysis.count_below_threshold_over_time(
        trajectories, threshold=terrain_config.roughness_transition_y, attr="y"
    )
    _save(
        plotting.plot_count_below_threshold(time_axis, counts, terrain_config.roughness_transition_y),
        out_dir, "06_count_on_rough_side", show,
    )

    print("Fitting grit/roughness calibration curves...")
    fit = analysis.fit_grit_calibration()
    for i, fig in enumerate(plotting.plot_grit_calibration(fit), start=1):
        _save(fig, out_dir, f"07_grit_calibration_{i}", show)

    print("Running roughness sweep (this runs several ensembles, may take a while)...")
    roughness_amplitudes = np.linspace(0.001, 0.1, 10)
    sweep_terrain_config = TerrainConfig(seed=seed)
    result = analysis.roughness_sweep(roughness_amplitudes, sweep_terrain_config, ball, physics, ensemble)
    _save(plotting.plot_roughness_sweep(result), out_dir, "08_roughness_sweep", show)

    print(f"\nDone. Figures written to {out_dir.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rough-slope-sim",
        description="Simulate a ball rolling down a rough inclined surface.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("output"),
        help="Directory to write output figures to (default: ./output)",
    )
    parser.add_argument("--show", action="store_true", help="Also display figures interactively")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    run_pipeline(args.out_dir, args.show, args.seed)


if __name__ == "__main__":
    main()
