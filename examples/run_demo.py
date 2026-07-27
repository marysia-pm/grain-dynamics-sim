"""Minimal end-to-end example.

Run with:

    uv run python examples/run_demo.py
"""

from pathlib import Path

from rough_slope_sim.analysis import final_positions
from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.plotting import plot_terrain_3d, plot_trajectories_3d
from rough_slope_sim.simulation import run_ensemble
from rough_slope_sim.terrain import generate_terrain


def main() -> None:
    out_dir = Path("examples_output")
    out_dir.mkdir(exist_ok=True)

    terrain = generate_terrain(TerrainConfig(alpha=30, resolution=300, seed=0))
    fig_terrain = plot_terrain_3d(terrain)
    fig_terrain.savefig(out_dir / "terrain.png", dpi=150, bbox_inches="tight")

    trajectories = run_ensemble(
        terrain,
        BallConfig(radius=0.5, mass=1.0),
        PhysicsConfig(total_time=10.0),
        EnsembleConfig(k_max=8, seed=0),
    )
    fig_traj = plot_trajectories_3d(terrain, trajectories)
    fig_traj.savefig(out_dir / "trajectories.png", dpi=150, bbox_inches="tight")

    final_y, final_z = final_positions(trajectories)
    n_reached_end = sum(t.reached_end for t in trajectories)
    print(f"{n_reached_end}/{len(trajectories)} balls reached the bottom of the slope")
    print(f"Final y positions: {final_y}")
    print(f"Final z positions: {final_z}")
    print(f"Figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
