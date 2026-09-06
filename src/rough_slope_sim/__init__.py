"""Public API re-exports for the rough_slope_sim package."""

from rough_slope_sim.analysis import compare_distributions, trajectories_at_x_slice
from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import Trajectory, run_ensemble_parallel, simulate_single_ball
from rough_slope_sim.terrain import Terrain, generate_terrain

__all__ = [
    "BallConfig",
    "PhysicsConfig",
    "TerrainConfig",
    "EnsembleConfig",
    "Terrain",
    "generate_terrain",
    "Trajectory",
    "run_ensemble_parallel",
    "simulate_single_ball",
    "trajectories_at_x_slice",
    "compare_distributions",
]

__version__ = "0.2.0"
