import numpy as np

from rough_slope_sim.analysis import (
    fit_grit_calibration,
    roughness_sweep,
    stack_trajectories,
    variance_over_time,
)
from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import Trajectory, run_ensemble
from rough_slope_sim.terrain import generate_terrain


def test_fit_grit_calibration_runs_and_returns_reasonable_params():
    fit = fit_grit_calibration()
    a_l, b_l = fit.linear_params
    a, b, c = fit.exp_params

    # Particle size should decrease with increasing grit -> B should be positive.
    assert b > 0
    # Sanity: predicted particle size at grit=120 should be close to the
    # measured value (127.0 micrometers).
    from rough_slope_sim.analysis import exponential_decay_model

    predicted = exponential_decay_model(np.array([120.0]), a, b, c)[0]
    assert abs(predicted - 127.0) < 20


def test_stack_and_variance_over_time_ignore_short_trajectories():
    t1 = Trajectory(
        time=np.array([0.0, 0.1, 0.2]),
        x=np.zeros(3), y=np.array([1.0, 2.0, 3.0]), z=np.zeros(3),
        vx=np.zeros(3), vy=np.zeros(3), vz=np.zeros(3), reached_end=False,
        final_position=(0.0, 3.0, 0.0),
    )
    t2 = Trajectory(
        time=np.array([0.0, 0.1]),
        x=np.zeros(2), y=np.array([1.5, 2.5]), z=np.zeros(2),
        vx=np.zeros(2), vy=np.zeros(2), vz=np.zeros(2), reached_end=False,
        final_position=(0.0, 2.5, 0.0),
    )

    stacked = stack_trajectories([t1, t2], attr="y")
    assert stacked.shape == (2, 3)
    assert np.isnan(stacked[1, 2])

    variance = variance_over_time([t1, t2], attr="y")
    assert variance.shape == (3,)
    assert not np.isnan(variance[0])
    assert not np.isnan(variance[1])
    # Only one live trajectory at the last time step -> variance undefined.
    assert np.isnan(variance[2])


def test_roughness_sweep_runs_without_the_original_undefined_variable_bug():
    """Regression test for the notebook's ``valid_final_x_positions`` /
    ``valid_final_y_positions`` typo, which made the roughness sweep crash."""
    terrain_config = TerrainConfig(resolution=40, seed=0)
    ball = BallConfig()
    physics = PhysicsConfig(total_time=1.0, dt=0.02)
    ensemble = EnsembleConfig(k_max=3, seed=1)

    result = roughness_sweep(np.array([0.01, 0.05]), terrain_config, ball, physics, ensemble)

    assert result.sa_values.shape == (2,)
    assert result.diffusion_coefficients.shape == (2,)
