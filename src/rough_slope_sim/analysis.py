"""Analysis, statistical metrics, spatial slicing, and diffusion calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp, wasserstein_distance

from .simulation import Trajectory
from .terrain import nanovea_d50_from_grit


def _xy_arrays(t: Trajectory | NDArray) -> tuple[NDArray, NDArray]:
    if isinstance(t, Trajectory):
        return t.x, t.y
    return t[:, 1], t[:, 2]


def _txy_arrays(t: Trajectory | NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Like _xy_arrays, but also returns the time-like axis (real seconds for a
    simulated Trajectory; the raw 'frame' column for an experimental ndarray --
    treat that as an arbitrary time-like index, not seconds, unless you know
    the capture frame rate)."""
    if isinstance(t, Trajectory):
        return t.t, t.x, t.y
    return t[:, 0], t[:, 1], t[:, 2]


def count_on_side_over_time(
    trajectories: list[Trajectory] | list[NDArray], interface_y: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Tracks, at each common time step, how many trajectories are currently on
    the rough side (y < interface_y) vs the smooth side (y >= interface_y).

    Once a trajectory ends (the ball left the domain), it stops counting toward
    either side from that point on -- so a drop in total active count reflects
    balls exiting the ramp, separate from balls crossing the interface, which
    shows up as one side's count falling while the other's rises: the flux of
    particles between the rough and smooth regions.

    Returns (time_axis, n_rough, n_smooth, n_active).
    """
    triples = [_txy_arrays(t) for t in trajectories if len(_txy_arrays(t)[0]) > 0]
    if not triples:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty

    max_len = max(len(ts) for ts, _, _ in triples)
    time_axis = max((ts for ts, _, _ in triples), key=len)

    padded_y = np.full((len(triples), max_len), np.nan, dtype=np.float64)
    for i, (_, _, y_arr) in enumerate(triples):
        padded_y[i, : len(y_arr)] = y_arr

    with np.errstate(invalid="ignore"):
        n_rough = np.sum(padded_y < interface_y, axis=0).astype(np.float64)
        n_smooth = np.sum(padded_y >= interface_y, axis=0).astype(np.float64)
    n_active = np.sum(~np.isnan(padded_y), axis=0).astype(np.float64)

    return time_axis, n_rough, n_smooth, n_active


def variance_over_time_by_side(
    trajectories: list[Trajectory] | list[NDArray], interface_y: float, min_count: int = 10
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Variance of Y-position over time, computed separately among whichever
    balls are CURRENTLY on the rough side vs the smooth side at each time step
    (regrouped dynamically every step, since a ball's relevant diffusivity is
    set by whatever texture it's presently rolling over, not where it started).

    A side's variance is reported as NaN wherever fewer than `min_count` balls
    are currently on it -- this is what keeps the tail of the curve from going
    chaotic once only a handful of balls remain (small-N variance estimates on
    a shrinking, non-random survivor set are not trustworthy).

    Returns (time_axis, var_rough, var_smooth, n_rough, n_smooth).
    """
    triples = [_txy_arrays(t) for t in trajectories if len(_txy_arrays(t)[0]) > 0]
    if not triples:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    max_len = max(len(ts) for ts, _, _ in triples)
    time_axis = max((ts for ts, _, _ in triples), key=len)

    padded_y = np.full((len(triples), max_len), np.nan, dtype=np.float64)
    for i, (_, _, y_arr) in enumerate(triples):
        padded_y[i, : len(y_arr)] = y_arr

    with np.errstate(invalid="ignore"):
        rough_mask = (
            padded_y < interface_y
        )  # NaN comparisons are False, so exited balls drop out of both masks
        smooth_mask = padded_y >= interface_y

    n_rough = rough_mask.sum(axis=0).astype(np.float64)
    n_smooth = smooth_mask.sum(axis=0).astype(np.float64)

    var_rough = np.full(max_len, np.nan, dtype=np.float64)
    var_smooth = np.full(max_len, np.nan, dtype=np.float64)
    for j in range(max_len):
        if n_rough[j] >= min_count:
            var_rough[j] = np.var(padded_y[rough_mask[:, j], j])
        if n_smooth[j] >= min_count:
            var_smooth[j] = np.var(padded_y[smooth_mask[:, j], j])

    return time_axis, var_rough, var_smooth, n_rough, n_smooth


def estimate_diffusion_from_variance_slope(
    time_axis: NDArray[np.float64], variance: NDArray[np.float64], t_range: tuple[float, float] | None = None
) -> float:
    """Fits Var(y) = 2*D*t + c by linear regression over the finite portion of
    the curve (optionally restricted to `t_range`), returning D. This is the
    standard 1D-diffusion relation.

    IMPORTANT CAVEAT when applied to variance_over_time_by_side's per-side
    curves: this gives an "apparent" D, not each side's true local diffusivity,
    because the side populations are not stationary samples. The rough side's
    population is progressively censored as its highest-spread members cross
    to the smooth side, which suppresses its apparent variance growth; the
    smooth side's population is continuously diluted by freshly-arrived,
    not-yet-spread members, which suppresses its apparent growth rate too
    (both were confirmed empirically, ~20-45% underestimates on synthetic
    ground truth). Use these numbers to compare *whether* one side spreads
    faster than the other (the direction is preserved), not as precise D
    values. For a bias-free absolute comparison, run two separate uniform-grit
    simulations (rough-only and smooth-only) and compare their
    calculate_diffusion_coefficient results directly -- each population is
    homogeneous throughout, so no such bias arises.
    """
    t = np.asarray(time_axis, dtype=np.float64)
    v = np.asarray(variance, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(v)
    if t_range is not None:
        valid &= (t >= t_range[0]) & (t <= t_range[1])
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(t[valid], v[valid], 1)
    return float(slope / 2.0)


def local_diffusion_coefficient_from_increments(
    trajectories: list[Trajectory] | list[NDArray],
    interface_y: float,
    lag_steps: int = 1,
) -> dict[str, float]:
    """Estimates the LOCAL, position-dependent diffusion coefficient on each
    side of `interface_y` via the Kramers-Moyal second moment:

        D(y) = lim_{dt->0} < (y(t+dt) - y(t))^2 | y(t) = y > / (2*dt)

    Each displacement is binned by the side of its own STARTING position
    y(t), not by long-time population history -- this is what makes it the
    right tool for a spatially-varying D(y): estimate_diffusion_from_variance_slope
    applied to variance_over_time_by_side's curves is biased low on both
    sides (confirmed 20-75% on synthetic ground truth) because the
    population currently on a side is a mix of different histories, not a
    homogeneous ensemble that started there. Conditioning each increment on
    its own starting point sidesteps that entirely. Confirmed on synthetic
    ground truth: <1% error at lag_steps=1, growing slowly with lag as more
    increments start to straddle a crossing (13% error by lag_steps=50 in
    testing) -- keep lag_steps as small as your saved trajectory resolution
    allows.

    Returns {"D_rough", "D_smooth", "n_rough", "n_smooth"}.
    """
    triples = [_txy_arrays(t) for t in trajectories if len(_txy_arrays(t)[0]) > lag_steps]
    if not triples:
        return {"D_rough": np.nan, "D_smooth": np.nan, "n_rough": 0, "n_smooth": 0}

    ts0 = triples[0][0]
    dt = float(ts0[lag_steps] - ts0[0]) if len(ts0) > lag_steps else np.nan
    if not (dt > 0):
        return {"D_rough": np.nan, "D_smooth": np.nan, "n_rough": 0, "n_smooth": 0}

    y0_all, dy_all = [], []
    for _, _, y_arr in triples:
        y0_all.append(y_arr[:-lag_steps])
        dy_all.append(y_arr[lag_steps:] - y_arr[:-lag_steps])

    y0 = np.concatenate(y0_all)
    dy = np.concatenate(dy_all)

    rough_mask = y0 < interface_y
    smooth_mask = y0 >= interface_y
    n_rough, n_smooth = int(rough_mask.sum()), int(smooth_mask.sum())

    D_rough = float(np.mean(dy[rough_mask] ** 2) / (2.0 * lag_steps * dt)) if n_rough > 0 else np.nan
    D_smooth = float(np.mean(dy[smooth_mask] ** 2) / (2.0 * lag_steps * dt)) if n_smooth > 0 else np.nan

    return {"D_rough": D_rough, "D_smooth": D_smooth, "n_rough": n_rough, "n_smooth": n_smooth}


def trajectories_at_x_slice(
    trajectories: list[Trajectory] | list[NDArray], x_slice: float
) -> NDArray[np.float64]:
    y_interp_list = []
    for t in trajectories:
        x_arr, y_arr = _xy_arrays(t)
        if x_arr is None or len(x_arr) < 2:
            continue

        # Fix 3: Find first passage crossing without breaking trajectory time ordering
        crossings = np.where((x_arr[:-1] <= x_slice) & (x_arr[1:] >= x_slice))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            x1, x2 = x_arr[idx], x_arr[idx + 1]
            y1, y2 = y_arr[idx], y_arr[idx + 1]

            if x2 != x1:
                y_interp = y1 + (y2 - y1) * (x_slice - x1) / (x2 - x1)
            else:
                y_interp = y1
            y_interp_list.append(y_interp)

    return np.array(y_interp_list, dtype=np.float64)


def variance_over_time(trajectories: list[Trajectory], attr: str = "y") -> NDArray[np.float64]:
    arrays = [getattr(t, attr) for t in trajectories if len(getattr(t, attr)) > 0]
    if not arrays:
        return np.array([], dtype=np.float64)

    max_len = max(len(a) for a in arrays)
    padded = np.full((len(arrays), max_len), np.nan, dtype=np.float64)

    for i, a in enumerate(arrays):
        padded[i, : len(a)] = a

    with np.errstate(invalid="ignore"):
        return np.nanvar(padded, axis=0)


def compare_distributions(y_exp: NDArray[np.float64], y_sim: NDArray[np.float64]) -> dict[str, float]:
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
    valid_y = y_positions[~np.isnan(y_positions)]
    if len(valid_y) < 2 or mean_vx <= 0:
        return np.nan
    t_avg = x_slice / mean_vx
    var_y = float(np.var(valid_y))
    return var_y / (2.0 * t_avg)


def compute_surface_center_stats(
    trajectories: list[Trajectory] | list[NDArray], x_eval: float = 15.0
) -> dict[str, float]:
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
    """Calculates particle size in micrometers using calibrated Nanovea lookup/fit."""
    d50_cm = nanovea_d50_from_grit(grit)
    return float(d50_cm * 10000.0)


def get_experimental_start_position(
    exp_trajectories: list[np.ndarray],
) -> tuple[float, float]:
    if not exp_trajectories:
        return 0.0, 0.0

    x_starts = [t[0, 1] if isinstance(t, np.ndarray) else t.x[0] for t in exp_trajectories]
    y_starts = [t[0, 2] if isinstance(t, np.ndarray) else t.y[0] for t in exp_trajectories]

    return float(np.mean(x_starts)), float(np.mean(y_starts))
