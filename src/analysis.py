"""Post-processing and analysis helpers.

Includes the grit-size / surface-roughness curve fitting from the original
notebook, and a "roughness sweep" that regenerates the terrain at several
roughness amplitudes and tracks how the spread (variance) of the ball's
final downslope position changes with surface roughness (``Sa``).

Note: the original notebook's roughness-sweep loop referenced an undefined
variable ``valid_final_x_positions`` (it should have been
``valid_final_y_positions``), which meant that section of the notebook could
never actually run to completion. That bug is fixed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig, TerrainConfig
from rough_slope_sim.simulation import Trajectory, run_ensemble
from rough_slope_sim.terrain import generate_terrain


def stack_trajectories(trajectories: list[Trajectory], attr: str = "y") -> NDArray[np.float64]:
    """Stack a list of (possibly ragged) trajectories into a 2D array of
    shape (n_trajectories, max_len), padded with NaN.

    ``attr`` selects which trajectory field to stack (``"x"``, ``"y"``,
    ``"z"``, ``"vx"``, ``"vy"`` or ``"vz"``).
    """
    max_len = max(len(getattr(t, attr)) for t in trajectories)
    padded = [t.padded(max_len) for t in trajectories]
    return np.array([getattr(t, attr) for t in padded])


def variance_over_time(trajectories: list[Trajectory], attr: str = "y") -> NDArray[np.float64]:
    """Variance of a trajectory attribute across the ensemble at each time
    step, ignoring simulations that have already ended (NaN). Time steps
    with fewer than two live simulations return NaN."""
    stacked = stack_trajectories(trajectories, attr)
    result = np.full(stacked.shape[1], np.nan)
    for i, column in enumerate(stacked.T):
        valid = column[~np.isnan(column)]
        if len(valid) >= 2:
            result[i] = np.var(valid)
    return result


def count_below_threshold_over_time(
    trajectories: list[Trajectory], threshold: float, attr: str = "y"
) -> NDArray[np.float64]:
    """Number of still-active simulations with ``attr <= threshold`` at each
    time step (NaN where fewer than two simulations remain active)."""
    stacked = stack_trajectories(trajectories, attr)
    result = np.full(stacked.shape[1], np.nan)
    for i, column in enumerate(stacked.T):
        valid = column[~np.isnan(column)]
        if len(valid) >= 2:
            result[i] = np.sum(valid <= threshold)
    return result


def final_positions(trajectories: list[Trajectory]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Final (y, z) position of every trajectory that reached the far edge
    of the terrain (NaN for trajectories that didn't)."""
    final_y = np.full(len(trajectories), np.nan)
    final_z = np.full(len(trajectories), np.nan)
    for i, traj in enumerate(trajectories):
        if traj.reached_end:
            _, y, z = traj.final_position
            final_y[i] = y
            final_z[i] = z
    return final_y, final_z


# --- Grit size vs. surface roughness curve fits -----------------------------

def linear_model(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    return a * x + b


def exponential_decay_model(x: NDArray[np.float64], a: float, b: float, c: float) -> NDArray[np.float64]:
    return a * np.exp(-b * x) + c


@dataclass
class GritFitResult:
    """Fit of mean particle size against sandpaper grit (P-grade), and the
    linear relation between mean particle size and Sa used to translate
    between the two."""

    grit: NDArray[np.float64]
    mean_particle_size: NDArray[np.float64]
    mean_particle_size_stdev: NDArray[np.float64]
    sa: NDArray[np.float64]
    linear_params: tuple[float, float]  # (A, B) of particle_size = A*Sa + B
    exp_params: tuple[float, float, float]  # (A, B, C) of particle_size = A*exp(-B*grit)+C

    def equivalent_grit_for_sa(self, sa_value: float) -> float:
        """Invert the exponential fit to estimate the sandpaper grit whose
        mean particle size matches the particle size predicted (via the
        linear Sa fit) for a given measured ``Sa``."""
        a_l, b_l = self.linear_params
        a, b, c = self.exp_params
        predicted_particle_size = a_l * sa_value + b_l
        return float(-np.log((predicted_particle_size - c) / a) / b)


# Reference measurements from the original thesis notebook (sandpaper grit
# calibration data): grit (P-grade), mean particle size and its std-dev in
# micrometers, and measured Sa.
DEFAULT_GRIT = np.array([120, 180, 320, 800, 2000])
DEFAULT_MEAN_PARTICLE_SIZE = np.array([127.0, 105.6, 67.18, 28.16, 21.27])
DEFAULT_MEAN_PARTICLE_SIZE_STDEV = np.array([48.7, 35.43, 22.62, 8.58, 8.35])
DEFAULT_SA = np.array([42.37, 27.28, 17.92, 6.273, 3.639])


def fit_grit_calibration(
    grit: NDArray[np.float64] = DEFAULT_GRIT,
    mean_particle_size: NDArray[np.float64] = DEFAULT_MEAN_PARTICLE_SIZE,
    mean_particle_size_stdev: NDArray[np.float64] = DEFAULT_MEAN_PARTICLE_SIZE_STDEV,
    sa: NDArray[np.float64] = DEFAULT_SA,
) -> GritFitResult:
    """Fit (1) mean particle size vs. Sa (linear) and (2) mean particle size
    vs. grit (exponential decay), mirroring the calibration curves used in
    the original notebook to translate abrasive grit sizes into an
    equivalent surface roughness Sa."""
    linear_p0 = (mean_particle_size[0] - mean_particle_size[-1], mean_particle_size[-1])
    linear_params, _ = curve_fit(linear_model, sa, mean_particle_size, p0=linear_p0)

    c_init = mean_particle_size[-1]
    a_init = mean_particle_size[0] - c_init
    b_init = 0.004
    exp_params, _ = curve_fit(
        exponential_decay_model, grit, mean_particle_size,
        p0=(a_init, b_init, c_init), sigma=mean_particle_size_stdev,
    )

    return GritFitResult(
        grit=grit,
        mean_particle_size=mean_particle_size,
        mean_particle_size_stdev=mean_particle_size_stdev,
        sa=sa,
        linear_params=tuple(linear_params),
        exp_params=tuple(exp_params),
    )


# --- Roughness sweep ---------------------------------------------------------

@dataclass
class RoughnessSweepResult:
    roughness_amplitudes: NDArray[np.float64]
    sa_values: NDArray[np.float64]
    diffusion_coefficients: NDArray[np.float64]


def roughness_sweep(
    roughness_amplitudes: NDArray[np.float64],
    terrain_config: TerrainConfig,
    ball: BallConfig,
    physics: PhysicsConfig,
    ensemble: EnsembleConfig,
) -> RoughnessSweepResult:
    """Regenerate the terrain at each roughness amplitude, run an ensemble
    of ball simulations on it, and compute the variance of the final
    downslope (y) position as a proxy "diffusion coefficient" versus the
    resulting measured Sa.

    This mirrors the original notebook's final analysis loop, with its
    ``valid_final_x_positions`` -> ``valid_final_y_positions`` typo fixed.
    """
    sa_values = []
    diffusion_coefficients = []

    for amplitude in roughness_amplitudes:
        config = TerrainConfig(
            alpha=terrain_config.alpha,
            resolution=terrain_config.resolution,
            x_range=terrain_config.x_range,
            y_range=terrain_config.y_range,
            roughness_amplitude_smooth=0.0,
            roughness_amplitude_rough=float(amplitude),
            roughness_transition_y=terrain_config.y_range[1],  # uniform roughness
            correlation_length=terrain_config.correlation_length,
            seed=terrain_config.seed,
        )
        terrain = generate_terrain(config)
        sa_values.append(terrain.Sa)

        trajectories = run_ensemble(terrain, ball, physics, ensemble)
        final_y, _ = final_positions(trajectories)
        valid_final_y = final_y[~np.isnan(final_y)]

        if len(valid_final_y) >= 2:
            diffusion_coefficients.append(float(np.var(valid_final_y)))
        else:
            diffusion_coefficients.append(np.nan)

    return RoughnessSweepResult(
        roughness_amplitudes=np.asarray(roughness_amplitudes, dtype=float),
        sa_values=np.array(sa_values),
        diffusion_coefficients=np.array(diffusion_coefficients),
    )
