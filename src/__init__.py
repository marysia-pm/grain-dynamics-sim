"""rough_slope_sim

A small physics-simulation package modelling a ball rolling down a rough,
inclined surface, originally developed as ad-hoc Colab notebook code for a
thesis project and refactored here into a clean, tested, installable package.

Public API re-exports the most commonly used pieces so that

    from rough_slope_sim import TerrainConfig, generate_terrain, ...

works without digging through submodules.
"""

from rough_slope_sim.config import BallConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.terrain import Terrain, generate_terrain
from rough_slope_sim.simulation import Trajectory, run_ensemble, simulate_single_ball

__all__ = [
    "BallConfig",
    "PhysicsConfig",
    "TerrainConfig",
    "Terrain",
    "generate_terrain",
    "Trajectory",
    "run_ensemble",
    "simulate_single_ball",
]

__version__ = "0.1.0"
