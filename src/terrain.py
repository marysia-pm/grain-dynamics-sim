"""Terrain generation.

Builds a rough inclined plane and exposes smooth interpolators for its
height and surface-normal field. In the original notebook these lived as
three separate module-level globals (``terrain_height_interpolator``,
``dz_dx_interpolator``, ``dz_dy_interpolator``) that had to be manually
re-initialised with a global function before use. Here they are simply
attributes of a :class:`Terrain` object, so multiple terrains (e.g. for a
roughness sweep) can safely coexist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from rough_slope_sim.config import TerrainConfig


@dataclass
class Terrain:
    """A generated rough-inclined-plane terrain plus its interpolators."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    Z: NDArray[np.float64]
    dz_dx: NDArray[np.float64]
    dz_dy: NDArray[np.float64]
    Sa: float
    config: TerrainConfig

    _height_interp: RectBivariateSpline = field(init=False, repr=False)
    _dzdx_interp: RectBivariateSpline = field(init=False, repr=False)
    _dzdy_interp: RectBivariateSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._height_interp = RectBivariateSpline(self.y, self.x, self.Z)
        self._dzdx_interp = RectBivariateSpline(self.y, self.x, self.dz_dx)
        self._dzdy_interp = RectBivariateSpline(self.y, self.x, self.dz_dy)

    @property
    def x_bounds(self) -> tuple[float, float]:
        return float(self.x.min()), float(self.x.max())

    @property
    def y_bounds(self) -> tuple[float, float]:
        return float(self.y.min()), float(self.y.max())

    def get_height(self, x_position: float, y_position: float) -> float:
        """Interpolated terrain height Z at (x, y)."""
        return float(self._height_interp(y_position, x_position)[0, 0])

    def get_normal_vector(self, x_position: float, y_position: float) -> tuple[float, float, float]:
        """Unit outward surface normal at (x, y).

        For a surface z = f(x, y) the (un-normalised) normal is
        (-dz/dx, -dz/dy, 1).
        """
        dzdx = float(self._dzdx_interp(y_position, x_position)[0, 0])
        dzdy = float(self._dzdy_interp(y_position, x_position)[0, 0])

        nx, ny, nz = -dzdx, -dzdy, 1.0
        magnitude = float(np.sqrt(nx**2 + ny**2 + nz**2))
        return nx / magnitude, ny / magnitude, nz / magnitude


def generate_terrain(config: TerrainConfig) -> Terrain:
    """Generate a rough inclined-plane terrain from a :class:`TerrainConfig`.

    The surface is an inclined ramp (angle ``alpha``) with additive,
    spatially correlated random roughness. The roughness amplitude switches
    between ``roughness_amplitude_rough`` and ``roughness_amplitude_smooth``
    at ``y = roughness_transition_y``, letting a simulation model a rough
    "traction" zone at the top of the slope transitioning into a smooth zone
    further down.
    """
    rng = np.random.default_rng(config.seed)

    x_range, y_range = config.x_range, config.y_range
    x = np.linspace(x_range[0], x_range[1], config.resolution)
    y = np.linspace(y_range[0], y_range[1], config.resolution)
    X, Y = np.meshgrid(x, y)

    roughness_transition_y = config.roughness_transition_y
    if roughness_transition_y is None:
        roughness_transition_y = (y_range[0] + y_range[1]) / 2

    # Inclined plane (ramp height decreases with x).
    alpha_rad = 2 * np.pi * config.alpha / 360
    H = 0.1 + x_range[1] * np.sin(alpha_rad)
    ramp = H - X * np.sin(alpha_rad)

    # Spatially correlated random roughness with a position-dependent
    # amplitude (rough near the top of the slope, smooth further down).
    noise = rng.standard_normal((config.resolution, config.resolution))
    noise = gaussian_filter(noise, sigma=config.correlation_length)
    noise -= np.mean(noise)
    noise /= np.std(noise)

    roughness_amplitude_map = np.where(
        Y > roughness_transition_y,
        config.roughness_amplitude_smooth,
        config.roughness_amplitude_rough,
    )
    noise *= roughness_amplitude_map

    Z = ramp + noise

    # Arithmetical average roughness of the surface (Sa).
    sa = 1e4 * np.mean(np.abs(noise - np.mean(noise))) / 25

    dz_dy, dz_dx = np.gradient(Z, y, x)

    return Terrain(x=x, y=y, X=X, Y=Y, Z=Z, dz_dx=dz_dx, dz_dy=dz_dy, Sa=float(sa), config=config)
