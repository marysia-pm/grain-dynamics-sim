"""Configuration dataclasses for terrain, ball, physics, and simulation execution.

This is the single source of truth for these configs — simulation.py imports
BallConfig/PhysicsConfig/SimConfig from here rather than redefining its own,
so there's no risk of the two drifting out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TerrainConfig:
    """Geometry and roughness parameters for the inclined sandpaper surface.

    X-Y is the horizontal ground plane, Z is up. The incline is baked into
    the generated height field as an additive planar tilt; it is NOT applied
    to gravity anywhere in the physics.
    """

    ramp_length: float = 29.0  # physical ramp length (cm)
    slope_angle: float = 30.0  # incline angle in degrees
    length_y: float = 23.0  # ramp width (cm)
    resolution: int = 1500  # grid points per axis
    roughness_amplitude_rough: float = 0.02  # d50 particle diameter, rough side (cm)
    roughness_amplitude_smooth: float = 0.001  # d50 particle diameter, smooth side (cm)
    grit_rough: float | None = None
    grit_smooth: float | None = None
    roughness_transition_y: float = 11.5
    max_grains: int = 1_500_000  # cap on individually-placed grains per side; finer grits fall back to correlated noise above this
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
    z0: float | None = None  # None -> spawn resting on the surface
    vx0: float = 0.0
    vy0: float = 0.0
    vz0: float = 0.0
    radius: float = 0.125  # cm
    mass: float = 1.0  # g
    restitution: float = 0.3  # 0 = fully inelastic, 1 = perfectly elastic bounce
    friction_mu: float = 0.0  # Coulomb friction coefficient while in contact


@dataclass
class PhysicsConfig:
    """Global physical constants. Gravity always points along -z."""

    gravity: float = 981.0  # cm/s^2

    @property
    def g(self) -> float:
        """Alias property for backwards compatibility."""
        return self.gravity


@dataclass
class SimConfig:
    """Simulation time-stepping and execution options."""

    dt: float = 5e-4
    t_max: float = 2.0
    save_interval: int = 10
    num_workers: int = 4
    seed: int | None = 42


@dataclass
class EnsembleConfig:
    """Sizes an ensemble and/or jitters initial positions when explicit
    `initial_states` aren't supplied to run_ensemble_parallel. Simulation
    timing lives in SimConfig, not here."""

    k_max: int = 150
    start_x: float = 0.1
    start_y: float = 11.5
    x_jitter_std: float = 0.05
    y_jitter_max: float = 2.0
    seed: int = 42
