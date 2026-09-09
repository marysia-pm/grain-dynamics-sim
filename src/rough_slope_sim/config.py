"""Configuration dataclasses for terrain, ball, physics, and ensemble execution."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TerrainConfig:
    """Geometry and roughness parameters for the inclined sandpaper surface."""

    ramp_length: float = 29.0
    slope_angle: float = 30.0
    length_y: float = 23.0
    resolution: int = 1000
    roughness_amplitude_rough: float = 0.01
    roughness_amplitude_smooth: float = 0.001
    grit_rough: float | None = None
    grit_smooth: float | None = None
    roughness_transition_y: float = 11.5
    seed: int = 42

    @property
    def length_x(self) -> float:
        return float(self.ramp_length * np.cos(np.radians(self.slope_angle)))


@dataclass
@dataclass
class BallConfig:
    """Configuration parameters for sphere properties and initial positions."""

    radius: float = 0.125  # cm
    mass: float = 1.0  # g
    x0: float = 0.1
    y0: float = 11.5
    y0_std: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0


@dataclass
class PhysicsConfig:
    """Global physical constants and contact interaction parameters."""

    dt: float = 0.001
    total_time: float = 3.0
    gravity: float = 981.0  # cm/s^2
    slope_angle: float = 30.0  # Incline angle in degrees
    friction: float = 0.0

    @property
    def g(self) -> float:
        """Alias property for backwards compatibility."""
        return self.gravity


@dataclass
class EnsembleConfig:
    """Settings for running parallel ensemble simulation runs."""

    k_max: int = 150
    start_x: float = 0.1
    start_y: float = 11.5
    vx0: float = 0.0
    vy0: float = 0.0
    vz0: float = 0.0
    x_jitter_std: float = 0.05
    y_jitter_max: float = 2.0
    seed: int = 42
