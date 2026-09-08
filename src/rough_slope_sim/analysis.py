"""Analysis, statistical metrics, spatial slicing, and diffusion calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp, wasserstein_distance

from .simulation import Trajectory


def _xy_arrays(t: Trajectory | NDArray) -> tuple[NDArray, NDArray]:
    """Extracts (x, y) arrays from either a Trajectory or a raw [frame, x, y] array."""
    if isinstance(t, Trajectory):
        return t.x, t.y
    return t[:, 1], t[:, 2]


def trajectories_at_x_slice(
    trajectories: list[Trajectory] | list[NDArray], x_slice: float
) -> NDArray[np.float64]:
    """Interpolates lateral Y positions across all trajectories at a fixed downslope X plane."""
    y_interp_list = []
    for t in trajectories:
        x_arr, y_arr = _xy_arrays(t)
        if x_arr is None or len(x_arr) < 2:
            continue

        # Check monotonic condition to avoid unnecessary sorting
        if not np.all(x_arr[:-1] <= x_arr[1:]):
            sort_idx = np.argsort(x_arr)
            x_sorted, y_sorted = x_arr[sort_idx], y_arr[sort_idx]
        else:
            x_sorted, y_sorted = x_arr, y_arr

        if x_sorted[0] <= x_slice <= x_sorted[-1]:
            y_interp_list.append(np.interp(x_slice, x_sorted, y_sorted))

    return np.array(y_interp_list, dtype=np.float64)


def variance_over_time(trajectories: list[Trajectory], attr: str = "y") -> NDArray[np.float64]:
    """Cross-ensemble variance of `attr` at each simulation time step."""
    arrays = [getattr(t, attr) for t in trajectories if len(getattr(t, attr)) > 0]
    if not arrays:
        return np.array([])

    lens = [len(a) for a in arrays]
    max_len = max(lens)
    padded = np.full((len(arrays), max_len), np.nan, dtype=np.float64)

    for i, a in enumerate(arrays):
        padded[i, : len(a)] = a

    with np.errstate(invalid="ignore"):
        return np.nanvar(padded, axis=0)


def compare_distributions(y_exp: NDArray[np.float64], y_sim: NDArray[np.float64]) -> dict[str, float]:
    """Calculates statistical distance metrics between experimental and simulated distributions."""
    valid_exp = y_exp[~np.isnan(y_exp)]
    valid_sim = y_sim[~np.isnan(y_sim)]

    if len(valid_exp) < 2 or len(valid_sim) < 2:
        return {
            "wasserstein_distance": np.nan,
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "variance_ratio": np.nan,
        }

    emd = wasserstein_distance(valid_exp, valid_sim)
    ks_stat, p_value = ks_2samp(valid_exp, valid_sim)
    var_sim = np.var(valid_sim)
    var_ratio = float(np.var(valid_exp) / var_sim) if var_sim > 0 else np.nan

    return {
        "wasserstein_distance": float(emd),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(p_value),
        "variance_ratio": var_ratio,
    }


def calculate_diffusion_coefficient(
    y_positions: NDArray[np.float64], x_slice: float, mean_vx: float = 30.0
) -> float:
    """Estimates lateral diffusion coefficient D = Var(Y) / (2 * t) where t = x / vx."""
    valid_y = y_positions[~np.isnan(y_positions)]
    if len(valid_y) < 2 or mean_vx <= 0:
        return np.nan
    t_avg = x_slice / mean_vx
    var_y = float(np.var(valid_y))
    return var_y / (2.0 * t_avg)


def compute_surface_center_stats(
    trajectories: list[Trajectory] | list[NDArray], x_eval: float = 15.0
) -> dict[str, float]:
    """Calculates center metrics (mean and median Y) for dual-grit surface evaluations."""
    y_vals = trajectories_at_x_slice(trajectories, x_eval)
    valid_y = y_vals[~np.isnan(y_vals)]
    if len(valid_y) == 0:
        return {"mean_y": np.nan, "median_y": np.nan, "std_y": np.nan}
    return {
        "mean_y": float(np.mean(valid_y)),
        "median_y": float(np.median(valid_y)),
        "std_y": float(np.std(valid_y)),
    }


def particle_size_from_grit(grit: float) -> float:
    """Calculates particle size in micrometers from grit number (P-value)."""
    return 173.014 * np.exp(-0.00408466 * grit) + 21.3533


def get_experimental_start_position(
    exp_trajectories: list[np.ndarray],
) -> tuple[float, float]:
    """Extracts the average starting (x, y) coordinates across all experimental trajectories.

    Assumes each trajectory in `exp_trajectories` has shape (N, 3+) where:
      - col 1: x position (cm)
      - col 2: y position (cm)
    """
    if not exp_trajectories:
        return 0.0, 0.0

    x_starts = [t[0, 1] if isinstance(t, np.ndarray) else t.x[0] for t in exp_trajectories]
    y_starts = [t[0, 2] if isinstance(t, np.ndarray) else t.y[0] for t in exp_trajectories]

    return float(np.mean(x_starts)), float(np.mean(y_starts))
