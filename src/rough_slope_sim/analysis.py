"""Analysis, statistical metrics, spatial slicing, and diffusion calculations."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp, wasserstein_distance

from rough_slope_sim.simulation import Trajectory


def trajectories_at_x_slice(
    trajectories: list[Trajectory] | list[NDArray], x_slice: float
) -> NDArray[np.float64]:
    """Interpolates lateral Y positions across all trajectories at a fixed downslope X plane."""
    y_interp_list = []
    for t in trajectories:
        x_arr = t.x if isinstance(t, Trajectory) else t[:, 1]
        y_arr = t.y if isinstance(t, Trajectory) else t[:, 2]

        if x_arr is None or len(x_arr) < 2:
            continue

        sort_idx = np.argsort(x_arr)
        x_sorted, y_sorted = x_arr[sort_idx], y_arr[sort_idx]

        if x_sorted[0] <= x_slice <= x_sorted[-1]:
            y_interp = np.interp(x_slice, x_sorted, y_sorted)
            y_interp_list.append(y_interp)

    return np.array(y_interp_list)


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
    var_ratio = float(np.var(valid_exp) / np.var(valid_sim)) if np.var(valid_sim) > 0 else np.nan

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


def particle_size_from_sa(sa: float) -> float:
    """Calculates particle size from surface average roughness Sa."""
    return 2.87515 * sa + 13.7870
