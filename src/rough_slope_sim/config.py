"""Configuration dataclasses for terrain, ball, physics, and ensemble execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TerrainConfig:
    """Geometry and roughness parameters for the inclined sandpaper surface.

    NOTE: `slope_angle` here and `PhysicsConfig.slope_angle` are two separate
    fields that must be kept equal by the caller -- see OPEN_QUESTIONS.md for
    why merging them isn't a safe drop-in change yet.
    """

    ramp_length: float = 29.0  # Physical ramp length (cm)
    slope_angle: float = 30.0  # Incline angle in degrees
    length_y: float = 23.0  # Ramp width (cm)
    resolution: int = 2000  # Grid resolution for surface sampling
    correlation_length: float = 1.0  # Spatial correlation length
    roughness_amplitude_rough: float = 0.01  # d50 particle diameter, rough side (cm)
    roughness_amplitude_smooth: float = 0.001  # d50 particle diameter, smooth side (cm)
    roughness_transition_y: float = 11.5
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
    radius: float = 0.125  # cm
    mass: float = 1.0  # g


@dataclass
class PhysicsConfig:
    dt: float = 0.001
    total_time: float = 3.0
    gravity: float = 981.0  # cm/s^2
    slope_angle: float = 30.0  # degrees -- keep in sync with TerrainConfig.slope_angle
    friction: float = 0.0


@dataclass
class EnsembleConfig:
    k_max: int = 150
    start_x: float = 0.1
    start_y: float = 11.5
    x_jitter_std: float = 0.05
    y_jitter_max: float = 2.0
    seed: int = 42
