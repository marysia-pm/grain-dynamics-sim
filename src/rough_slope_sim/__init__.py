"""Public API re-exports for the rough_slope_sim package."""

from rough_slope_sim.analysis import (
    calculate_diffusion_coefficient,
    compare_distributions,
    compute_surface_center_stats,
    particle_size_from_grit,
    trajectories_at_x_slice,
    variance_over_time,
)
from rough_slope_sim.config import (
    BallConfig,
    EnsembleConfig,
    PhysicsConfig,
    SimConfig,
    TerrainConfig,
)
from rough_slope_sim.plotting import (
    plot_calibrated_sandpaper_panel,
    plot_terrain_3d,
)
from rough_slope_sim.simulation import Trajectory, run_ensemble_parallel, simulate_single_ball
from rough_slope_sim.terrain import (
    NANOVEA_DATA,
    Terrain,
    estimate_grit,
    generate_calibrated_sandpaper,
    generate_galton_board,
    generate_terrain,
    nanovea_sa_from_d50,
    compute_rolling_ball_envelope,
)

__all__ = [
    "BallConfig",
    "PhysicsConfig",
    "TerrainConfig",
    "EnsembleConfig",
    "SimConfig",
    "Terrain",
    "generate_terrain",
    "generate_galton_board",
    "generate_calibrated_sandpaper",
    "estimate_grit",
    "nanovea_sa_from_d50",
    "compute_rolling_ball_envelope",
    "NANOVEA_DATA",
    "Trajectory",
    "run_ensemble_parallel",
    "simulate_single_ball",
    "trajectories_at_x_slice",
    "compare_distributions",
    "compute_surface_center_stats",
    "calculate_diffusion_coefficient",
    "particle_size_from_grit",
    "variance_over_time",
    "plot_calibrated_sandpaper_panel",
    "plot_terrain_3d",
]

__version__ = "0.3.0"
