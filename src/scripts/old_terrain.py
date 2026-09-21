"""Terrain generation and surface roughness models.

Combines physical micro-geometry realism (tiled KD-Tree grain placement, random
grain radius/offset distributions, and Sa surface roughness calibration) with
the Terrain class structure and sharp regional transitions. Also supports
procedural Galton board pin layouts and sandpaper micro-grains oriented
normally to the slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

# Calibrated Nanovea profilometer dataset
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


def _tiled_grain_bump_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    d50_cm: float,
    slope_angle: float = 0.0,
    tile_size_cm: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    if d50_cm <= 0.0:
        return np.zeros((len(x_grid), len(y_grid)), dtype=float)

    rad = np.radians(slope_angle)
    cos_theta = np.cos(rad) if slope_angle > 0 else 1.0
    tan_theta = np.tan(rad) if slope_angle > 0 else 0.0
    s_grid = x_grid / cos_theta

    rng = np.random.default_rng(seed)

    ss, yy = np.meshgrid(s_grid, y_grid, indexing="ij")
    nx, ny = ss.shape
    z_bump = np.zeros((nx, ny), dtype=np.float64)

    s_min, s_max = float(s_grid[0]), float(s_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    r_mean = d50_cm / 2.0
    r_std = r_mean * 0.15
    spacing = r_mean * 1.35
    pad = r_mean * 3.0
    z_aspect = 1.30

    target_sa_cm = d50_cm * 0.33

    s_tiles = np.arange(s_min, s_max, tile_size_cm)
    y_tiles = np.arange(y_min, y_max, tile_size_cm)

    # Pre-calculated surface-normal quadratic coefficient for ellipsoidal grains
    A_grain = tan_theta**2 + 1.0 / (z_aspect**2)

    for i, ts_start in enumerate(s_tiles):
        ts_end = min(ts_start + tile_size_cm, s_max)
        ix = np.where(
            (s_grid >= ts_start) & (s_grid <= ts_end if i == len(s_tiles) - 1 else s_grid < ts_end)
        )[0]
        if len(ix) == 0:
            continue

        for j, ty_start in enumerate(y_tiles):
            ty_end = min(ty_start + tile_size_cm, y_max)
            iy = np.where(
                (y_grid >= ty_start) & (y_grid <= ty_end if j == len(y_tiles) - 1 else y_grid < ty_end)
            )[0]
            if len(iy) == 0:
                continue

            sub_S = ss[np.ix_(ix, iy)]
            sub_Y = yy[np.ix_(ix, iy)]
            tile_pts = np.column_stack([sub_S.ravel(), sub_Y.ravel()])

            gs = np.arange(ts_start - pad, ts_end + pad, max(spacing, 1e-6))
            gy = np.arange(ty_start - pad, ty_end + pad, max(spacing, 1e-6))
            if len(gs) == 0 or len(gy) == 0:
                continue

            GS, GY = np.meshgrid(gs, gy)
            jitter = spacing * 0.40
            cs = (GS + rng.uniform(-jitter, jitter, size=GS.shape)).ravel()
            cy = (GY + rng.uniform(-jitter, jitter, size=GY.shape)).ravel()

            n_grains = len(cs)
            r_grains = np.clip(rng.normal(r_mean, r_std, size=n_grains), r_mean * 0.5, r_mean * 1.5)
            z_offsets = rng.normal(0.0, r_mean * 0.40, size=n_grains)

            tree = cKDTree(np.column_stack([cs, cy]))
            k_query = min(6, n_grains)
            dists, indices = tree.query(tile_pts, k=k_query)

            if dists.ndim == 1:
                dists, indices = dists[:, None], indices[:, None]

            z_flat = np.full(len(tile_pts), -1e5, dtype=np.float64)
            S_pts, Y_pts = tile_pts[:, 0], tile_pts[:, 1]

            for k in range(dists.shape[1]):
                idx_k = indices[:, k]
                r_k, z_off_k = r_grains[idx_k], z_offsets[idx_k]
                c_sk, c_yk = cs[idx_k], cy[idx_k]

                ds0 = S_pts - c_sk
                dy0 = Y_pts - c_yk

                # Exact surface-normal quadratic equation for tilted grains
                B_k = -2.0 * ds0 * tan_theta - (2.0 * z_off_k / (z_aspect**2))
                C_k = ds0**2 + dy0**2 - r_k**2 + (z_off_k**2 / (z_aspect**2))

                disc = B_k**2 - 4.0 * A_grain * C_k
                valid = disc >= 0.0

                z_k = np.full_like(S_pts, -1e5)
                if np.any(valid):
                    sqrt_disc = np.sqrt(np.maximum(0.0, disc[valid]))
                    z_k[valid] = (-B_k[valid] + sqrt_disc) / (2.0 * A_grain)

                z_flat = np.maximum(z_flat, z_k)

            z_tile = z_flat.reshape((len(ix), len(iy)))
            z_tile = np.maximum(z_tile, -r_mean * 0.10)
            z_bump[np.ix_(ix, iy)] = z_tile

    z_bump -= z_bump.min()

    current_sa = float(np.mean(np.abs(z_bump - np.mean(z_bump))))
    if current_sa > 0:
        z_bump *= target_sa_cm / current_sa

    return z_bump


def galton_board_bump_field(
    X: np.ndarray,
    Y: np.ndarray,
    slope_angle: float = 30.0,
    peg_radius: float = 0.30,
    peg_height: float = 0.20,
    peg_dx: float = 2.0,
    peg_dy: float = 1.5,
    staggered: bool = True,
    peg_x_start: float = 5.0,
    peg_x_end: float | None = None,
    shape: str = "cone",
) -> np.ndarray:
    """Generates Galton-board peg surface height profile h_n normal to the incline."""
    rad = np.radians(slope_angle)
    cos_th = np.cos(rad)
    tan_th = np.tan(rad)

    x_grid = X[:, 0]
    y_grid = Y[0, :]

    S = X / cos_th  # Local distance along the slope plane

    s_min, s_max = float(S[0, 0]), float(S[-1, 0])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    s_start = peg_x_start
    s_end = s_max if peg_x_end is None else peg_x_end

    rows_s = np.arange(s_start, s_end + 1e-9, peg_dx)
    r_sq = peg_radius**2

    h_n = np.zeros_like(X, dtype=np.float64)

    # Geometry quadratic pre-factors for cone shape normal orientation
    inv_slope = peg_radius / peg_height
    A_cone = tan_th**2 - inv_slope**2

    for i, rs in enumerate(rows_s):
        y_offset = (peg_dy / 2.0) if (staggered and i % 2 == 1) else 0.0
        first_y = y_min + peg_radius + y_offset
        cols_y = np.arange(first_y, y_max - peg_radius + 1e-9, peg_dy)

        for ry in cols_y:
            rx = rs * cos_th

            # Expanded search padding to capture normal peg tilt downhill
            search_pad = (peg_radius + peg_height * tan_th) * cos_th
            ix0 = max(0, int(np.searchsorted(x_grid, rx - search_pad) - 1))
            ix1 = min(len(x_grid), int(np.searchsorted(x_grid, rx + search_pad) + 1))
            iy0 = max(0, int(np.searchsorted(y_grid, ry - peg_radius) - 1))
            iy1 = min(len(y_grid), int(np.searchsorted(y_grid, ry + peg_radius) + 1))

            if ix0 >= ix1 or iy0 >= iy1:
                continue

            sub_S = S[ix0:ix1, iy0:iy1]
            sub_Y = Y[ix0:ix1, iy0:iy1]

            ds0 = sub_S - rs
            dy = sub_Y - ry

            if shape == "cone":
                B = -2.0 * ds0 * tan_th + 2.0 * (peg_radius**2) / peg_height
                C = ds0**2 + dy**2 - peg_radius**2
                disc = B**2 - 4.0 * A_cone * C
                valid = disc >= 0.0

                if not np.any(valid):
                    continue

                profile = np.zeros_like(sub_S)
                sqrt_disc = np.sqrt(np.maximum(0.0, disc[valid]))
                profile[valid] = np.clip((-B[valid] + sqrt_disc) / (2.0 * A_cone), 0.0, peg_height)

            elif shape in ("paraboloid", "gaussian"):
                # Fixed-point iteration for non-conical profiles along surface normal
                profile = np.zeros_like(sub_S)
                for _ in range(4):
                    S_norm = sub_S - profile * tan_th
                    d_sq = (S_norm - rs) ** 2 + dy**2
                    in_peg = d_sq <= r_sq
                    if shape == "paraboloid":
                        profile[in_peg] = peg_height * (1.0 - d_sq[in_peg] / r_sq)
                    elif shape == "gaussian":
                        sigma_sq = (peg_radius / 2.0) ** 2
                        profile[in_peg] = peg_height * np.exp(-d_sq[in_peg] / (2.0 * sigma_sq))
            else:
                raise ValueError(
                    f"Unsupported peg shape: '{shape}'. Use 'paraboloid', 'gaussian', or 'cone'."
                )

            sub_z = h_n[ix0:ix1, iy0:iy1]
            h_n[ix0:ix1, iy0:iy1] = np.maximum(sub_z, profile)

    return h_n


class Terrain:
    """3D synthetic rough surface: X-Y ground plane, Z up."""

    def __init__(
        self,
        height_map: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        slope_angle: float,
        roughness_amplitude_rough: float = 0.0,
        roughness_amplitude_smooth: float = 0.0,
        grit_rough: float | None = None,
        grit_smooth: float | None = None,
        surface_type: str = "sandpaper",
    ):
        self.height_map = height_map
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.slope_angle = slope_angle
        self.roughness_amplitude_rough = roughness_amplitude_rough
        self.roughness_amplitude_smooth = roughness_amplitude_smooth
        self.surface_type = str(surface_type).lower()

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
    def slope_rad(self) -> float:
        return np.radians(self.slope_angle)

    @property
    def x_bounds(self) -> tuple[float, float]:
        return float(self.x_grid[0]), float(self.x_grid[-1])

    @property
    def y_bounds(self) -> tuple[float, float]:
        return float(self.y_grid[0]), float(self.y_grid[-1])

    @property
    def x_range(self) -> tuple[float, float]:
        return self.x_bounds

    @property
    def y_range(self) -> tuple[float, float]:
        return self.y_bounds

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1])

    @property
    def grit_rough(self) -> float:
        return float(self._grit_rough)

    @property
    def grit_smooth(self) -> float:
        return float(self._grit_smooth)

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
    """Generates a Terrain instance based on configuration parameters."""
    ramp_length = float(getattr(cfg, "ramp_length", 29.0))
    slope_angle = float(getattr(cfg, "slope_angle", 30.0))
    rad = np.radians(slope_angle)

    x_len = getattr(cfg, "length_x", None)
    if x_len is None:
        x_len = ramp_length * np.cos(rad)
    else:
        x_len = float(x_len)

    y_len = float(getattr(cfg, "length_y", 23.0))
    res = int(getattr(cfg, "resolution", 1000))
    tile_size_cm = float(getattr(cfg, "tile_size_cm", 0.20))
    surface_type = str(getattr(cfg, "surface_type", "sandpaper")).lower()

    x = np.linspace(0, x_len, res)
    y = np.linspace(0, y_len, res)

    amp_rough = float(getattr(cfg, "roughness_amplitude_rough", 0.01))
    amp_smooth = float(getattr(cfg, "roughness_amplitude_smooth", 0.001))
    y_trans = float(getattr(cfg, "roughness_transition_y", y_len / 2.0))

    grit_r = getattr(cfg, "grit_rough", None)
    grit_s = getattr(cfg, "grit_smooth", None)
    seed = int(getattr(cfg, "seed", 42))

    xx, yy = np.meshgrid(x, y, indexing="ij")

    if surface_type == "galton":
        normal_roughness = galton_board_bump_field(
            xx,
            yy,
            slope_angle=slope_angle,
            peg_radius=float(getattr(cfg, "peg_radius", 0.30)),
            peg_height=float(getattr(cfg, "peg_height", 0.20)),
            peg_dx=float(getattr(cfg, "peg_dx", 2.0)),
            peg_dy=float(getattr(cfg, "peg_dy", 1.5)),
            staggered=bool(getattr(cfg, "peg_staggered", True)),
            peg_x_start=float(getattr(cfg, "peg_x_start", 5.0)),
            peg_x_end=getattr(cfg, "peg_x_end", None),
            shape=str(getattr(cfg, "peg_shape", "cone")),
        )
    else:
        rough_field = _tiled_grain_bump_field(
            x, y, amp_rough, slope_angle=slope_angle, tile_size_cm=tile_size_cm, seed=seed
        )
        if np.isclose(amp_rough, amp_smooth):
            normal_roughness = rough_field
        else:
            smooth_field = _tiled_grain_bump_field(
                x, y, amp_smooth, slope_angle=slope_angle, tile_size_cm=tile_size_cm, seed=seed + 1
            )
            normal_roughness = np.where(yy < y_trans, rough_field, smooth_field)

    z_offset = float(getattr(cfg, "z_offset", 15.0))

    # Baseline inclined plane offset by +15 cm
    z_base = z_offset - xx * np.tan(rad)

    # Convert normal profile h_n to vertical elevation offset Delta Z = h_n / cos(theta)
    delta_Z = normal_roughness / np.cos(rad)

    # Enforce global non-negative height
    height_map = np.maximum(0.0, z_base + delta_Z)

    return Terrain(
        height_map=height_map,
        x_grid=x,
        y_grid=y,
        slope_angle=slope_angle,
        roughness_amplitude_rough=amp_rough,
        roughness_amplitude_smooth=amp_smooth,
        grit_rough=grit_r,
        grit_smooth=grit_s,
        surface_type=surface_type,
    )


def generate_calibrated_sandpaper(
    grit_rough: float = 80.0,
    grit_smooth: float | None = None,
    slope_angle: float = 30.0,
    ramp_length: float = 29.0,
    length_y: float = 23.0,
    resolution: int = 1000,
    tile_size_cm: float = 0.20,
    seed: int = 42,
    **kwargs: Any,
) -> Terrain:
    """Generates a calibrated sandpaper Terrain instance directly from P-grit specifications."""
    if grit_smooth is None:
        grit_smooth = grit_rough

    amp_rough = nanovea_d50_from_grit(grit_rough)
    amp_smooth = nanovea_d50_from_grit(grit_smooth)
    length_x = float(ramp_length * np.cos(np.radians(slope_angle)))

    cfg = SimpleNamespace(
        surface_type="sandpaper",
        ramp_length=ramp_length,
        length_x=length_x,
        length_y=length_y,
        slope_angle=slope_angle,
        resolution=resolution,
        roughness_amplitude_rough=amp_rough,
        roughness_amplitude_smooth=amp_smooth,
        grit_rough=grit_rough,
        grit_smooth=grit_smooth,
        roughness_transition_y=length_y / 2.0,
        tile_size_cm=tile_size_cm,
        seed=seed,
    )
    return generate_terrain(cfg)


def generate_galton_board(
    ramp_length: float = 29.0,
    length_y: float = 23.0,
    slope_angle: float = 30.0,
    resolution: int = 1000,
    peg_radius: float = 0.30,
    peg_height: float = 0.20,
    peg_dx: float = 2.0,
    peg_dy: float = 1.5,
    peg_staggered: bool = True,
    peg_x_start: float = 5.0,
    peg_x_end: float | None = None,
    peg_shape: str = "cone",
    **kwargs: Any,
) -> Terrain:
    """Generates a Galton board peg Terrain instance directly."""
    length_x = float(ramp_length * np.cos(np.radians(slope_angle)))

    cfg = SimpleNamespace(
        surface_type="galton",
        ramp_length=ramp_length,
        length_x=length_x,
        length_y=length_y,
        slope_angle=slope_angle,
        resolution=resolution,
        peg_radius=peg_radius,
        peg_height=peg_height,
        peg_dx=peg_dx,
        peg_dy=peg_dy,
        peg_staggered=peg_staggered,
        peg_x_start=peg_x_start,
        peg_x_end=peg_x_end,
        peg_shape=peg_shape,
        roughness_amplitude_rough=peg_height,
        roughness_amplitude_smooth=peg_height,
    )
    return generate_terrain(cfg)
