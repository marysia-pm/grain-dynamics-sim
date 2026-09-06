"""Terrain generation and surface representation for sandpaper micro-geometry using ISO 6344 standards."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .config import TerrainConfig


def p_value_to_d50_cm(p_value: float) -> float:
    """Converts FEPA P-value to average particle diameter d50 (in cm) using ISO 6344 standard fit."""
    if p_value <= 0:
        return 0.02
    # ISO 6344 empirical relation: d50 (um) ~ 12500 * P^(-0.95)
    d50_um = 12500.0 / (float(p_value) ** 0.95)
    return d50_um / 10000.0  # um to cm


def d50_cm_to_p_value(d50_cm: float) -> float:
    """Converts average particle diameter d50 (in cm) to FEPA P-value."""
    d50_um = float(d50_cm) * 10000.0
    if d50_um <= 0:
        return float("inf")
    return float((12500.0 / d50_um) ** (1.0 / 0.95))


class Terrain:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        p_value_mean: float = 0.0,
        p_value_rough: float = 0.0,
        p_value_smooth: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.z = z

        self.p_value_mean = p_value_mean
        self.p_value_rough = p_value_rough
        self.p_value_smooth = p_value_smooth

        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.Z = z

        self.x_bounds = (float(x[0]), float(x[-1]))
        self.y_bounds = (float(y[0]), float(y[-1]))

        self._z_interp = RegularGridInterpolator((x, y), z, bounds_error=False, fill_value=0.0)

        dz_dx, dz_dy = np.gradient(z, x, y)
        self._dz_dx_interp = RegularGridInterpolator((x, y), dz_dx, bounds_error=False, fill_value=0.0)
        self._dz_dy_interp = RegularGridInterpolator((x, y), dz_dy, bounds_error=False, fill_value=0.0)

    def get_elevation(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        vals = self._z_interp(pts)
        return vals[0] if np.isscalar(x) and np.isscalar(y) else vals

    def get_height(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return self.get_elevation(x, y)

    def get_normal(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        zx = self._dz_dx_interp(pts)
        zy = self._dz_dy_interp(pts)

        normals = np.column_stack([-zx, -zy, np.ones_like(zx)])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(norms, 1e-12)

        return normals[0] if np.isscalar(x) and np.isscalar(y) else normals


def generate_terrain(config: TerrainConfig) -> Terrain:
    """Generates an inclined sandpaper surface with micro-grit scaled accurately to FEPA standards."""
    rng = np.random.default_rng(config.seed)

    x = np.linspace(config.x_range[0], config.x_range[1], config.resolution)
    y = np.linspace(config.y_range[0], config.y_range[1], config.resolution)

    nx, ny = len(x), len(y)
    X_grid, Y_grid = np.meshgrid(x, y, indexing="ij")

    # Particle mean diameters d50 (cm) from config amplitudes
    d50_rough = max(config.roughness_amplitude_rough, 1e-5)
    d50_smooth = max(config.roughness_amplitude_smooth, 1e-5)

    # Transition map across Y
    weights = 1.0 / (1.0 + np.exp((Y_grid - config.roughness_transition_y) / 0.15))
    d50_map = weights * d50_rough + (1.0 - weights) * d50_smooth

    # Multi-frequency stochastic micro-surface (prevents spatial grid-aliasing)
    # High-frequency noise frequency scales inversely with grain diameter
    k_rough = 2.0 * np.pi / d50_rough
    k_smooth = 2.0 * np.pi / d50_smooth

    phase1 = rng.uniform(0, 2 * np.pi, size=(nx, ny))
    phase2 = rng.uniform(0, 2 * np.pi, size=(nx, ny))

    noise_field = 0.6 * np.sin(k_rough * X_grid + phase1) * np.cos(
        k_smooth * Y_grid + phase2
    ) + 0.4 * rng.uniform(-1.0, 1.0, size=(nx, ny))

    # Scale micro-roughness profile directly by d50 particle height
    z_micro = np.abs(noise_field) * d50_map

    # Macroscopic incline drop
    z_incline = -X_grid * np.tan(np.radians(config.slope_angle))
    z = z_incline + z_micro

    # Measure actual generated P-values
    mask_rough = Y_grid < config.roughness_transition_y
    mask_smooth = Y_grid >= config.roughness_transition_y

    d50_rough_meas = float(np.mean(d50_map[mask_rough])) if np.any(mask_rough) else d50_rough
    d50_smooth_meas = float(np.mean(d50_map[mask_smooth])) if np.any(mask_smooth) else d50_smooth

    p_rough = d50_cm_to_p_value(d50_rough_meas)
    p_smooth = d50_cm_to_p_value(d50_smooth_meas)
    p_mean = d50_cm_to_p_value((d50_rough_meas + d50_smooth_meas) / 2.0)

    return Terrain(
        x,
        y,
        z,
        p_value_mean=p_mean,
        p_value_rough=p_rough,
        p_value_smooth=p_smooth,
    )
