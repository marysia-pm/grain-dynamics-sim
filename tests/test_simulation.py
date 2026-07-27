import numpy as np

from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import run_ensemble, simulate_single_ball
from rough_slope_sim.terrain import generate_terrain


def _flat_smooth_terrain(alpha: float = 30.0, resolution: int = 60, seed: int = 0):
    config = TerrainConfig(
        alpha=alpha, resolution=resolution,
        roughness_amplitude_rough=0.0, roughness_amplitude_smooth=0.0,
        seed=seed,
    )
    return generate_terrain(config)


def test_ball_starts_on_the_surface():
    terrain = _flat_smooth_terrain()
    ball = BallConfig()
    physics = PhysicsConfig(total_time=0.5)

    traj = simulate_single_ball(terrain, ball, physics, start_x=0.1, start_y=2.5)

    expected_z0 = terrain.get_height(0.1, 2.5) + ball.radius
    assert np.isclose(traj.z[0], expected_z0, atol=1e-6)


def test_ball_rolls_downhill_on_smooth_ramp():
    """On a smooth, inclined ramp the ball should accelerate down the slope
    (x should increase over time, roughly following a constant-acceleration
    profile)."""
    terrain = _flat_smooth_terrain(alpha=30.0)
    ball = BallConfig()
    physics = PhysicsConfig(total_time=1.0, dt=0.01)

    traj = simulate_single_ball(terrain, ball, physics, start_x=0.1, start_y=2.5)

    assert len(traj.x) > 1
    # Monotonically non-decreasing downslope position.
    assert np.all(np.diff(traj.x) >= -1e-9)
    assert traj.x[-1] > traj.x[0]
    # Downslope velocity should be increasing (accelerating down the ramp).
    assert traj.vx[-1] > traj.vx[0]


def test_ball_does_not_penetrate_surface():
    terrain = _flat_smooth_terrain(alpha=30.0)
    ball = BallConfig()
    physics = PhysicsConfig(total_time=2.0, dt=0.01)

    traj = simulate_single_ball(terrain, ball, physics, start_x=0.1, start_y=2.5)

    heights = np.array([terrain.get_height(x, y) for x, y in zip(traj.x, traj.y)])
    # Ball centre should never sink meaningfully below terrain height + radius.
    assert np.all(traj.z >= heights + ball.radius - 1e-6)


def test_ball_on_flat_zero_incline_does_not_move_horizontally():
    terrain = _flat_smooth_terrain(alpha=0.0)
    ball = BallConfig()
    physics = PhysicsConfig(total_time=0.5, dt=0.01)

    traj = simulate_single_ball(terrain, ball, physics, start_x=2.0, start_y=2.5)

    assert np.allclose(traj.x, traj.x[0], atol=1e-9)
    assert np.allclose(traj.y, traj.y[0], atol=1e-9)


def test_run_ensemble_returns_k_max_trajectories():
    terrain = _flat_smooth_terrain()
    ball = BallConfig()
    physics = PhysicsConfig(total_time=1.0)
    ensemble = EnsembleConfig(k_max=5, seed=42)

    trajectories = run_ensemble(terrain, ball, physics, ensemble)

    assert len(trajectories) == 5
    for traj in trajectories:
        assert len(traj.x) > 0


def test_trajectory_padding():
    terrain = _flat_smooth_terrain()
    ball = BallConfig()
    physics = PhysicsConfig(total_time=0.2, dt=0.01)
    traj = simulate_single_ball(terrain, ball, physics, start_x=0.1, start_y=2.5)

    padded = traj.padded(100)
    assert len(padded.x) == 100
    assert np.isnan(padded.x[-1])
