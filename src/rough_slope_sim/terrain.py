"""Terrain generation and surface roughness models.

Coordinate convention: X-Y is the horizontal ground plane, Z is vertical
(up). The incline is baked into the height field as a pure additive planar
term, -x*tan(slope_angle), applied AFTER the grain texture is built in the
flat (untilted) frame. Grain bump height depends only on horizontal (x, y)
distance from each grain center, so adding a planar tilt afterward does not
shear or stretch the grain footprints -- it just stacks a linear ramp under
them. (Grain heights are tens of microns; the ramp is tens of centimeters,
so this is physically indistinguishable from a true rigid rotation of the
textured surface.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# Calibrated Nanovea profilometer dataset exported for plotting and direct lookup
NANOVEA_DATA: dict[int, dict[str, float]] = {
    80: {"d50_cm": 0.0201, "d50_um": 201.0},
    120: {"d50_cm": 0.0125, "d50_um": 125.0},
    180: {"d50_cm": 0.0082, "d50_um": 82.0},
    240: {"d50_cm": 0.0053, "d50_um": 53.0},
    320: {"d50_cm": 0.0046, "d50_um": 46.0},
    400: {"d50_cm": 0.0035, "d50_um": 35.0},
    600: {"d50_cm": 0.0025, "d50_um": 25.0},
    800: {"d50_cm": 0.0021, "d50_um": 21.0},
    1000: {"d50_cm": 0.0018, "d50_um": 18.0},
    1200: {"d50_cm": 0.0015, "d50_um": 15.0},
    2000: {"d50_cm": 0.0010, "d50_um": 10.0},
}

NANOVEA_GRIT_TABLE = np.array(list(NANOVEA_DATA.keys()), dtype=np.float64)
NANOVEA_D50_CM_TABLE = np.array([v["d50_cm"] for v in NANOVEA_DATA.values()], dtype=np.float64)

_d50_from_grit_interp = interp1d(
    np.log10(NANOVEA_GRIT_TABLE),
    np.log10(NANOVEA_D50_CM_TABLE),
    kind="linear",
    fill_value="extrapolate",
)

_grit_from_d50_interp = interp1d(
    np.log10(NANOVEA_D50_CM_TABLE[::-1]),
    np.log10(NANOVEA_GRIT_TABLE[::-1]),
    kind="linear",
    fill_value="extrapolate",
)


def nanovea_d50_from_grit(grit: float) -> float:
    """Returns calibrated particle diameter d50 (cm) for a given P-grit."""
    log_d50 = _d50_from_grit_interp(np.log10(float(grit)))
    return float(10.0**log_d50)


def nanovea_grit_from_d50(d50_cm: float) -> float:
    """Returns equivalent P-grit for a given particle diameter d50 (cm)."""
    log_grit = _grit_from_d50_interp(np.log10(float(d50_cm)))
    return float(10.0**log_grit)


def estimate_grit(d50_cm: float) -> float:
    """Alias for backwards compatibility with __init__.py imports."""
    return nanovea_grit_from_d50(d50_cm)


def _grain_bump_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    d50: float,
    y_mask: np.ndarray | None = None,
    coverage: float = 0.85,
    height_jitter: tuple[float, float] = (0.8, 1.2),
    max_grains: int = 1_500_000,
    seed: int = 42,
) -> tuple[np.ndarray, bool]:
    """Rasterizes real overlapping hemispherical ("half-sphere") grains over
    (x_grid, y_grid) using a KD-tree for efficiency. Grain height depends
    only on horizontal distance from each grain center -- no slope is
    involved here, so footprints stay perfectly round.

    Returns (bump_height_field, used_real_grains). If the grit is fine
    enough that faithfully placing individual grains would need more than
    `max_grains`, falls back to correlated Gaussian noise matched to the
    same d50 amplitude and correlation length (used_real_grains=False) --
    at that scale a real surface looks like fine continuous texture anyway,
    not discrete visible bumps.
    """
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    if y_mask is None:
        y_mask = np.ones_like(yy, dtype=bool)

    x_len = float(x_grid[-1] - x_grid[0])
    y_len = float(y_grid[-1] - y_grid[0])
    active_area = x_len * y_len * (float(np.count_nonzero(y_mask)) / y_mask.size)

    r_g = d50 / 2.0
    grain_area = np.pi * r_g**2
    n_grains = int(np.clip(coverage * active_area / max(grain_area, 1e-12), 4, max_grains))

    rng = np.random.default_rng(seed)

    if n_grains >= max_grains:
        # Too fine to place individually over this domain -- statistically
        # equivalent correlated noise instead (real fine-grit paper looks
        # like smooth continuous texture at this scale, not discrete bumps).
        dx = float(x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 1.0
        sigma_px = max(0.5 * (d50 / dx), 0.5)
        noise = gaussian_filter(rng.standard_normal(xx.shape), sigma=sigma_px)
        noise *= d50 / (np.std(noise) or 1.0)
        return np.where(y_mask, noise, 0.0), False

    y_idx = np.where(np.any(y_mask, axis=0))[0]
    y_lo = y_grid[y_idx[0]] if len(y_idx) else y_grid[0]
    y_hi = y_grid[y_idx[-1]] if len(y_idx) else y_grid[-1]

    gx = rng.uniform(x_grid[0] - r_g, x_grid[-1] + r_g, size=n_grains)
    gy = rng.uniform(y_lo - r_g, y_hi + r_g, size=n_grains)
    gh = rng.uniform(height_jitter[0], height_jitter[1], size=n_grains) * d50

    tree = cKDTree(np.column_stack([gx, gy]))
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    k = 4
    dists, idxs = tree.query(pts, k=k, distance_upper_bound=r_g, workers=-1)
    if k == 1:
        dists, idxs = dists[:, None], idxs[:, None]

    z_flat = np.zeros(pts.shape[0], dtype=np.float64)
    for j in range(k):
        valid = np.isfinite(dists[:, j])
        dist_sq = dists[valid, j] ** 2
        contrib = np.maximum(0.0, gh[idxs[valid, j]] * (1.0 - dist_sq / (r_g**2)))
        z_flat[valid] = np.maximum(z_flat[valid], contrib)

    return z_flat.reshape(xx.shape) * y_mask, True


class Terrain:
    """3D synthetic rough surface: X-Y ground plane, Z up. h(x, y) already
    includes the incline tilt (applied additively, after grain texturing),
    so a plain vertical gravity vector is enough to drive downslope motion —
    the local surface normal does the rest."""

    def __init__(
        self,
        height_map: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        slope_angle: float,
        roughness_amplitude_rough: float,
        roughness_amplitude_smooth: float,
        grit_rough: float | None = None,
        grit_smooth: float | None = None,
    ):
        self.height_map = height_map
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.slope_angle = slope_angle
        self.roughness_amplitude_rough = roughness_amplitude_rough
        self.roughness_amplitude_smooth = roughness_amplitude_smooth

        self._grit_rough = (
            grit_rough if grit_rough is not None else nanovea_grit_from_d50(roughness_amplitude_rough)
        )
        self._grit_smooth = (
            grit_smooth if grit_smooth is not None else nanovea_grit_from_d50(roughness_amplitude_smooth)
        )

        self.dx = float(x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 1.0
        self.dy = float(y_grid[1] - y_grid[0]) if len(y_grid) > 1 else 1.0
        self.grad_x, self.grad_y = np.gradient(height_map, self.dx, self.dy)

    @property
    def X(self) -> np.ndarray:
        xx, _ = np.meshgrid(self.x_grid, self.y_grid, indexing="ij")
        return xx

    @property
    def Y(self) -> np.ndarray:
        _, yy = np.meshgrid(self.x_grid, self.y_grid, indexing="ij")
        return yy

    @property
    def Z(self) -> np.ndarray:
        return self.height_map

    @property
    def x_bounds(self) -> tuple[float, float]:
        return float(self.x_grid[0]), float(self.x_grid[-1])

    @property
    def y_bounds(self) -> tuple[float, float]:
        return float(self.y_grid[0]), float(self.y_grid[-1])

    @property
    def p_value_rough(self) -> float:
        return float(self._grit_rough)

    @property
    def p_value_smooth(self) -> float:
        return float(self._grit_smooth)

    @property
    def p_value_mean(self) -> float:
        if np.isclose(self.p_value_rough, self.p_value_smooth):
            return self.p_value_rough
        return float((self.p_value_rough + self.p_value_smooth) / 2.0)

    def _grid_coords(self, x: float | np.ndarray, y: float | np.ndarray):
        gx = np.clip((np.asarray(x, dtype=np.float64) - self.x_grid[0]) / self.dx, 0, len(self.x_grid) - 2)
        gy = np.clip((np.asarray(y, dtype=np.float64) - self.y_grid[0]) / self.dy, 0, len(self.y_grid) - 2)
        ix, iy = gx.astype(int), gy.astype(int)
        rx, ry = gx - ix, gy - iy
        return ix, iy, rx, ry

    def get_height(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Bilinear-interpolated surface height at (x, y). Accepts scalars or arrays."""
        ix, iy, rx, ry = self._grid_coords(x, y)
        z00 = self.height_map[ix, iy]
        z10 = self.height_map[ix + 1, iy]
        z01 = self.height_map[ix, iy + 1]
        z11 = self.height_map[ix + 1, iy + 1]
        val = (1 - rx) * (1 - ry) * z00 + rx * (1 - ry) * z10 + (1 - rx) * ry * z01 + rx * ry * z11
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            return float(val)
        return val

    def get_gradient(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Bilinear-interpolated surface gradient at (x, y). Accepts scalars or arrays."""
        ix, iy, rx, ry = self._grid_coords(x, y)
        gx00, gx10 = self.grad_x[ix, iy], self.grad_x[ix + 1, iy]
        gx01, gx11 = self.grad_x[ix, iy + 1], self.grad_x[ix + 1, iy + 1]
        gy00, gy10 = self.grad_y[ix, iy], self.grad_y[ix + 1, iy]
        gy01, gy11 = self.grad_y[ix, iy + 1], self.grad_y[ix + 1, iy + 1]

        gx = (1 - rx) * (1 - ry) * gx00 + rx * (1 - ry) * gx10 + (1 - rx) * ry * gx01 + rx * ry * gx11
        gy = (1 - rx) * (1 - ry) * gy00 + rx * (1 - ry) * gy10 + (1 - rx) * ry * gy01 + rx * ry * gy11
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            return float(gx), float(gy)
        return gx, gy

    def get_normal(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Computes unit normal vector (nx, ny, nz) at (x, y). Accepts scalars or arrays."""
        gx, gy = self.get_gradient(x, y)
        norm = np.sqrt(gx**2 + gy**2 + 1.0)
        nx = -gx / norm
        ny = -gy / norm
        nz = 1.0 / norm
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            return float(nx), float(ny), float(nz)
        return nx, ny, nz


def generate_terrain(cfg: Any) -> Terrain:
    """Generates the terrain height field h(x, y) from real, randomly-placed
    hemispherical grains (falling back to correlated noise only when a grit
    is too fine to place individually across the full domain). The incline
    is added afterward as a pure planar tilt so grain footprints stay round.
    """
    x_len = getattr(cfg, "length_x", 25.0)
    y_len = getattr(cfg, "length_y", 23.0)
    res = getattr(cfg, "resolution", 1000)
    slope_angle = float(getattr(cfg, "slope_angle", 30.0))
    max_grains = int(getattr(cfg, "max_grains", 1_500_000))

    x = np.linspace(0, x_len, res)
    y = np.linspace(0, y_len, res)

    amp_rough = float(getattr(cfg, "roughness_amplitude_rough", 0.01))
    amp_smooth = float(getattr(cfg, "roughness_amplitude_smooth", 0.001))
    y_trans = float(getattr(cfg, "roughness_transition_y", 11.5))

    grit_r = getattr(cfg, "grit_rough", None)
    grit_s = getattr(cfg, "grit_smooth", None)
    seed = int(getattr(cfg, "seed", 42))

    xx, yy = np.meshgrid(x, y, indexing="ij")
    rough_mask = yy < y_trans
    smooth_mask = ~rough_mask

    rough_field, _ = _grain_bump_field(x, y, amp_rough, y_mask=rough_mask, max_grains=max_grains, seed=seed)
    smooth_field, _ = _grain_bump_field(
        x, y, amp_smooth, y_mask=smooth_mask, max_grains=max_grains, seed=seed + 1
    )
    roughness = rough_field + smooth_field

    z_base = -xx * np.tan(np.radians(slope_angle))
    height_map = z_base + roughness

    return Terrain(
        height_map=height_map,
        x_grid=x,
        y_grid=y,
        slope_angle=slope_angle,
        roughness_amplitude_rough=amp_rough,
        roughness_amplitude_smooth=amp_smooth,
        grit_rough=grit_r,
        grit_smooth=grit_s,
    )


def generate_calibrated_sandpaper(
    grit_rough: float = 80.0,
    grit_smooth: float | None = None,
    slope_angle: float = 30.0,
    ramp_length: float = 29.0,
    length_y: float = 23.0,
    resolution: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> Terrain:
    """Generates a calibrated Terrain instance directly from P-grit specifications."""
    if grit_smooth is None:
        grit_smooth = grit_rough

    amp_rough = nanovea_d50_from_grit(grit_rough)
    amp_smooth = nanovea_d50_from_grit(grit_smooth)

    @dataclass
    class DynamicTerrainConfig:
        ramp_length: float = ramp_length
        slope_angle: float = slope_angle
        length_y: float = length_y
        resolution: int = resolution
        roughness_amplitude_rough: float = amp_rough
        roughness_amplitude_smooth: float = amp_smooth
        grit_rough: float = grit_rough
        grit_smooth: float = grit_smooth
        roughness_transition_y: float = length_y / 2.0
        seed: int = seed

        @property
        def length_x(self) -> float:
            return float(self.ramp_length * np.cos(np.radians(self.slope_angle)))

    cfg = DynamicTerrainConfig()
    return generate_terrain(cfg)
