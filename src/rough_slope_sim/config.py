"""Configuration dataclasses for terrain, ball, physics, and simulation execution."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TerrainConfig:
    """Geometry, roughness, and peg parameters for slope terrains."""

    surface_type: str = "sandpaper"  # "flat", "sandpaper" or "galton"
    ramp_length: float = 29.0  # physical ramp length (cm)
    slope_angle: float = 30.0  # incline angle in degrees
    length_y: float = 23.0  # ramp width (cm)
    resolution_x: int = 2500  # grid points along X (downslope)
    resolution_y: int = 2500  # grid points along Y (lateral)
    z_offset: float | None = None  # baseline plane height (cm); None = auto-scaled to ramp_length

    # Sandpaper parameters
    roughness_amplitude_rough: float = 0.02  # d50 particle diameter, rough side (cm)
    roughness_amplitude_smooth: float = 0.001  # d50 particle diameter, smooth side (cm)
    grit_rough: float | None = None
    grit_smooth: float | None = None
    roughness_transition_y: float = 11.5
    max_grains: int = 1_500_000
    tile_size_cm: float = 0.20

    # Galton board peg parameters
    peg_radius: float = 0.30
    peg_height: float = 0.20
    peg_dx: float = 2.0
    peg_dy: float = 1.5
    peg_staggered: bool = True
    peg_x_start: float = 5.0
    peg_x_end: float | None = None
    peg_shape: str = "cylinder"

    seed: int = 42

    @property
    def length_x(self) -> float:
        """Horizontal projected length: length_x = ramp_length * cos(slope_angle)."""
        return float(self.ramp_length * np.cos(np.radians(self.slope_angle)))

    @property
    def x_range(self) -> tuple[float, float]:
        return (0.0, self.length_x)

    @property
    def y_range(self) -> tuple[float, float]:
        return (0.0, self.length_y)


@dataclass
class BallConfig:
    """Configuration for the rolling/bouncing sphere."""

    x0: float = 0.1
    y0: float = 11.5
    z0: float | None = None
    vx0: float = 0.0
    vy0: float = 0.0
    vz0: float = 0.0
    radius: float = 0.125  # cm
    mass: float = 1.0  # g
    restitution: float = 0.7
    friction_mu: float = 0.0


@dataclass
class PhysicsConfig:
    """Global physical constants. Gravity always points along -z."""

    gravity: float = 981.0  # cm/s^2

    @property
    def g(self) -> float:
        return self.gravity


@dataclass
class SimConfig:
    """Simulation time-stepping and execution options."""

    dt: float = 5e-4
    t_max: float = 5.0
    save_interval: int = 10
    num_workers: int = 4
    seed: int | None = 42


@dataclass
class EnsembleConfig:
    """Ensemble sizing and initial conditions generator."""

    k_max: int = 150
    start_x: float = 0.1
    start_y: float = 11.5
    x_jitter_std: float = 0.05
    y_jitter_max: float = 2.0
    seed: int = 42
