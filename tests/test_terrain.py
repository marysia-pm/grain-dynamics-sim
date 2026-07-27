import numpy as np

from rough_slope_sim.config import TerrainConfig
from rough_slope_sim.terrain import generate_terrain


def test_terrain_shapes():
    config = TerrainConfig(resolution=50, seed=0)
    terrain = generate_terrain(config)

    assert terrain.X.shape == (50, 50)
    assert terrain.Y.shape == (50, 50)
    assert terrain.Z.shape == (50, 50)
    assert terrain.dz_dx.shape == (50, 50)
    assert terrain.dz_dy.shape == (50, 50)
    assert len(terrain.x) == 50
    assert len(terrain.y) == 50


def test_terrain_slopes_downhill_in_x():
    """Height should decrease, on average, as x increases (ramp slopes
    downhill), since roughness noise has zero mean."""
    config = TerrainConfig(resolution=80, roughness_amplitude_rough=0.0, roughness_amplitude_smooth=0.0, seed=1)
    terrain = generate_terrain(config)

    mean_height_low_x = terrain.Z[:, 0].mean()
    mean_height_high_x = terrain.Z[:, -1].mean()
    assert mean_height_high_x < mean_height_low_x


def test_get_height_matches_grid_near_grid_points():
    config = TerrainConfig(resolution=200, seed=2)
    terrain = generate_terrain(config)

    # Interpolated height at an exact grid point should match the stored
    # grid value closely.
    ix, iy = 50, 60
    x0, y0 = terrain.x[ix], terrain.y[iy]
    interpolated = terrain.get_height(x0, y0)
    assert np.isclose(interpolated, terrain.Z[iy, ix], atol=1e-6)


def test_normal_vector_is_unit_length():
    config = TerrainConfig(resolution=100, seed=3)
    terrain = generate_terrain(config)

    nx, ny, nz = terrain.get_normal_vector(2.5, 2.5)
    magnitude = (nx**2 + ny**2 + nz**2) ** 0.5
    assert np.isclose(magnitude, 1.0, atol=1e-6)


def test_flat_terrain_normal_is_vertical():
    """With zero roughness and a flat (alpha=0) ramp, the surface normal
    should point straight up."""
    config = TerrainConfig(
        alpha=0.0, resolution=50,
        roughness_amplitude_rough=0.0, roughness_amplitude_smooth=0.0,
        seed=4,
    )
    terrain = generate_terrain(config)
    nx, ny, nz = terrain.get_normal_vector(2.0, 2.0)
    assert np.isclose(nx, 0.0, atol=1e-6)
    assert np.isclose(ny, 0.0, atol=1e-6)
    assert np.isclose(nz, 1.0, atol=1e-6)


def test_sa_is_zero_for_flat_terrain():
    config = TerrainConfig(
        resolution=50, roughness_amplitude_rough=0.0, roughness_amplitude_smooth=0.0, seed=5,
    )
    terrain = generate_terrain(config)
    assert terrain.Sa == 0.0
