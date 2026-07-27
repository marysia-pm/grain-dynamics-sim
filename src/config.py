"""Configuration objects.

The original notebook defined every simulation parameter as a bare module
level global. That works in a notebook cell but makes the code impossible to
reuse or test in isolation (functions silently reached out to globals such as
``roughness_transition_y`` or ``ball_radius``). Here every parameter lives on
an explicit, typed, immutable dataclass that gets passed around instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainConfig:
    """Parameters describing the generated rough inclined surface."""

    alpha: float = 30.0
    """Inclination of the plane, in degrees."""

    resolution: int = 1000
    """Number of grid points along each axis."""

    x_range: tuple[float, float] = (0.0, 5.0)
    y_range: tuple[float, float] = (0.0, 5.0)

    roughness_amplitude_smooth: float = 0.0
    """Roughness amplitude applied where y > roughness_transition_y."""

    roughness_amplitude_rough: float = 0.02
    """Roughness amplitude applied where y <= roughness_transition_y."""

    roughness_transition_y: float | None = 2.5
    """Y coordinate where the surface switches from rough to smooth.
    ``None`` defaults to the midpoint of ``y_range``."""

    correlation_length: float = 15.0
    """Gaussian-filter sigma controlling the spatial correlation of the
    surface noise (in grid points)."""

    seed: int | None = None
    """Optional seed for the random noise field, for reproducibility."""


@dataclass(frozen=True)
class BallConfig:
    """Physical properties of the rolling ball."""

    radius: float = 0.5
    mass: float = 1.0


@dataclass(frozen=True)
class PhysicsConfig:
    """Integrator / physics parameters."""

    g: float = 9.81
    dt: float = 0.01
    denser_steps_number: int = 100
    """Number of sub-steps used per ``dt`` interval for the collision
    integrator (keeps the ball from tunnelling through the surface)."""
    total_time: float = 30.0
    restitution_coeff: float = 0.8
    """0 = fully inelastic (no bounce), 1 = perfectly elastic bounce."""


@dataclass(frozen=True)
class EnsembleConfig:
    """Parameters controlling a batch ("ensemble") of ball simulations that
    share the same terrain but start from slightly randomised positions."""

    k_max: int = 500
    start_x: float = 0.1
    start_y: float = 2.45
    x_jitter_std: float = 0.01
    """Std-dev of the (Gaussian) random deviation applied to start_x."""
    y_jitter_max: float = 0.05
    """Max magnitude of the (uniform) random deviation applied to start_y."""
    seed: int | None = None
