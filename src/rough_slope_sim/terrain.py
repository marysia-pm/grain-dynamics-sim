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
from scipy.ndimage import grey_dilation
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


# Real measured Nanovea profilometry Sa (areal roughness) reference points.
# Previously this benchmark data only lived in a standalone exploring_surfaces.py
# script and grain calibration elsewhere used a rough target_sa_cm = 0.33 * d50
# heuristic instead. This table lets every surface calibrate against the actual
# measured Sa, matching the surfaces produced by that exploration script.
_NANOVEA_SA_REFERENCE_D50_CM = np.array([127.0, 105.6, 67.18, 28.16, 21.27], dtype=np.float64) / 10000.0
_NANOVEA_SA_REFERENCE_SA_CM = np.array([42.37, 27.28, 17.92, 6.273, 3.639], dtype=np.float64) / 10000.0

_sa_from_d50_interp = interp1d(
    np.log10(_NANOVEA_SA_REFERENCE_D50_CM),
    np.log10(_NANOVEA_SA_REFERENCE_SA_CM),
    kind="linear",
    fill_value="extrapolate",
)


def nanovea_sa_from_d50(d50_cm: float) -> float:
    """Returns calibrated Sa areal roughness (cm) for a given particle diameter d50 (cm),
    interpolated/extrapolated (log-log) from real Nanovea profilometry measurements."""
    if d50_cm <= 0.0:
        return 0.0
    log_sa = _sa_from_d50_interp(np.log10(float(d50_cm)))
    return float(10.0**log_sa)


def _tiled_grain_bump_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    d50_cm: float,
    slope_angle: float = 0.0,
    tile_size_cm: float = 0.20,
    seed: int = 42,
    k_neighbors: int = 4,
) -> np.ndarray:
    """Procedural grain-bump height field, tiled across (x_grid, y_grid) via local
    KD-Tree grain placement. NOTE: this only resolves grain-scale detail when
    the grid spacing is fine relative to d50_cm -- for a coarse, full-ramp grid
    and small (high-grit) grains, most of the true grain texture aliases away.
    See plot_ball_surface_closeup, which regenerates a dedicated fine patch for
    visualization instead of relying on this function's grid resolution."""
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

    target_sa_cm = nanovea_sa_from_d50(d50_cm)

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
            k_query = min(k_neighbors, n_grains)
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
    shape: str = "cylinder",
) -> np.ndarray:
    """Generates Galton-board peg relief, returned as VERTICAL Z height above the
    inclined base plane.

    Peg axes run along the SURFACE NORMAL of the incline -- i.e. each peg stands
    perpendicular to the ramp (a true 90 degrees between the plane and the
    cylinder's side wall), like a pin pressed straight into the sloped board, not
    like a vertical post. Geometry is solved in closed form: for a vertical ray at
    (x, y), we find the largest axial distance t along the normal at which the ray
    is still inside the peg's tilted circular tube, then convert that to vertical
    height (t / cos(theta), the vertical separation of two planes offset by t
    along the normal)."""
    rad = np.radians(slope_angle)
    cos_th = np.cos(rad)
    sin_th = np.sin(rad)
    tan_th = np.tan(rad)

    x_grid = X[:, 0]
    y_grid = Y[0, :]

    # In-plane downslope coordinate: distance measured along the ramp surface.
    S = X / cos_th

    s_max = float(S[-1, 0])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    s_start = peg_x_start
    s_end = s_max if peg_x_end is None else peg_x_end

    rows_s = np.arange(s_start, s_end + 1e-9, peg_dx)
    r_sq = peg_radius**2

    h_n = np.zeros_like(X, dtype=np.float64)

    # A peg leans downslope by peg_height*sin(theta) in x as it rises, so widen
    # the x search window to cover the tilted tube's full vertical projection.
    search_pad = peg_radius * cos_th + peg_height * sin_th + 2.0 * float(x_grid[1] - x_grid[0])

    for i, rs in enumerate(rows_s):
        rx = rs * cos_th
        ix0 = max(0, int(np.searchsorted(x_grid, rx - search_pad) - 1))
        ix1 = min(len(x_grid), int(np.searchsorted(x_grid, rx + search_pad) + 1))
        if ix0 >= ix1:
            continue

        y_offset = (peg_dy / 2.0) if (staggered and i % 2 == 1) else 0.0
        first_y = y_min + peg_radius + y_offset
        cols_y = np.arange(first_y, y_max - peg_radius + 1e-9, peg_dy)

        for ry in cols_y:
            iy0 = max(0, int(np.searchsorted(y_grid, ry - peg_radius) - 1))
            iy1 = min(len(y_grid), int(np.searchsorted(y_grid, ry + peg_radius) + 1))
            if iy0 >= iy1:
                continue

            sub_S = S[ix0:ix1, iy0:iy1]
            sub_Y = Y[ix0:ix1, iy0:iy1]

            dy = sub_Y - ry
            a = sub_S - rs  # in-plane downslope offset from the peg axis

            half_chord_sq = r_sq - dy**2
            inside_span = half_chord_sq > 0.0
            w = np.sqrt(np.maximum(half_chord_sq, 0.0))

            profile_normal = np.zeros_like(sub_S)

            if shape == "cylinder":
                if tan_th > 1e-12:
                    # Ray is inside the tilted tube for t in [(a-w)/tan, (a+w)/tan].
                    t_lo = (a - w) / tan_th
                    t_hi = (a + w) / tan_th
                    t_max = np.minimum(t_hi, peg_height)
                    t_floor = np.maximum(t_lo, 0.0)
                    hit = inside_span & (t_max >= t_floor)
                    profile_normal[hit] = t_max[hit]
                else:
                    hit = inside_span & (np.abs(a) <= w)
                    profile_normal[hit] = peg_height
            elif shape in ("paraboloid", "gaussian"):
                # Rounded caps: shear the footprint by the axis tilt at the local
                # height, solved by a short fixed-point pass (stable in 2 steps).
                for _ in range(2):
                    d_sq = (a - profile_normal * tan_th) ** 2 + dy**2
                    in_peg = d_sq <= r_sq
                    profile_normal = np.zeros_like(sub_S)
                    if shape == "paraboloid":
                        profile_normal[in_peg] = peg_height * (1.0 - d_sq[in_peg] / r_sq)
                    else:
                        sigma_sq = (peg_radius / 2.0) ** 2
                        profile_normal[in_peg] = peg_height * np.exp(-d_sq[in_peg] / (2.0 * sigma_sq))
            else:
                raise ValueError(
                    f"Unsupported peg shape: '{shape}'. Use 'paraboloid', 'gaussian', or 'cylinder'."
                )

            # Normal-axis distance -> vertical height above the inclined plane.
            profile_vertical = profile_normal / cos_th

            sub_z = h_n[ix0:ix1, iy0:iy1]
            h_n[ix0:ix1, iy0:iy1] = np.maximum(sub_z, profile_vertical)

    return h_n


def compute_rolling_ball_envelope(
    height_map: np.ndarray, dx: float, dy: float, ball_radius: float
) -> np.ndarray:
    """Computes the surface a sphere of `ball_radius` traces while rolling on top
    of `height_map` -- i.e. what the ball can actually feel, as opposed to the
    raw point geometry.

    The terrain's grain/peg edges have genuine vertical tangents (a hemisphere's
    own rim), so the raw height field has unbounded local slope: refining the
    grid resolution never converges to a finite contact slope, it just samples
    closer to an inherent singularity. A finite-size ball physically cannot
    enter crevices narrower than itself, so the physically correct contact
    surface is not the raw terrain but its Minkowski sum with a ball of that
    radius -- exactly a greyscale dilation with a hemispherical structuring
    element (the classic "rolling ball" transform). This caps the effective
    curvature/slope the ball can ever feel to what its own radius allows,
    independent of grid resolution.

    Returns a height field of the same shape as `height_map`: the height of the
    ball's CENTER above the base plane if resting at that (x, y), minus
    `ball_radius` -- i.e. a surface that reproduces correct sphere-on-terrain
    contact when treated as a point-contact height field (matching how
    Terrain.get_height/get_gradient/get_normal are already used everywhere).
    """
    if ball_radius <= 0:
        return height_map.copy()

    rx = max(int(np.ceil(ball_radius / dx)), 1)
    ry = max(int(np.ceil(ball_radius / dy)), 1)
    ii, jj = np.meshgrid(np.arange(-rx, rx + 1), np.arange(-ry, ry + 1), indexing="ij")
    dist_sq = (ii * dx) ** 2 + (jj * dy) ** 2
    within = dist_sq <= ball_radius**2
    structure = np.where(within, np.sqrt(np.maximum(ball_radius**2 - dist_sq, 0.0)), 0.0)

    envelope = grey_dilation(height_map, footprint=within, structure=structure)
    return envelope - ball_radius


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
        z_offset: float = 15.0,
        tile_size_cm: float = 0.20,
        roughness_transition_y: float | None = None,
        peg_radius: float = 0.30,
        peg_height: float = 0.20,
        peg_dx: float = 2.0,
        peg_dy: float = 1.5,
        peg_staggered: bool = True,
        peg_x_start: float = 5.0,
        peg_x_end: float | None = None,
        peg_shape: str = "cylinder",
    ):
        self.height_map = height_map
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.slope_angle = slope_angle
        self.roughness_amplitude_rough = roughness_amplitude_rough
        self.roughness_amplitude_smooth = roughness_amplitude_smooth
        self.surface_type = str(surface_type).lower()
        self.z_offset = float(z_offset)
        self.tile_size_cm = float(tile_size_cm)
        self.roughness_transition_y = (
            float(roughness_transition_y) if roughness_transition_y is not None else float(y_grid[-1]) / 2.0
        )
        self.peg_radius = float(peg_radius)
        self.peg_height = float(peg_height)
        self.peg_dx = float(peg_dx)
        self.peg_dy = float(peg_dy)
        self.peg_staggered = bool(peg_staggered)
        self.peg_x_start = float(peg_x_start)
        self.peg_x_end = peg_x_end
        self.peg_shape = str(peg_shape)

        self._grit_rough = (
            grit_rough if grit_rough is not None else nanovea_grit_from_d50(roughness_amplitude_rough)
        )
        self._grit_smooth = (
            grit_smooth if grit_smooth is not None else nanovea_grit_from_d50(roughness_amplitude_smooth)
        )

        self.dx = float(x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 1.0
        self.dy = float(y_grid[1] - y_grid[0]) if len(y_grid) > 1 else 1.0
        self.grad_x, self.grad_y = np.gradient(height_map, self.dx, self.dy)

        self._contact_cache: dict[float, "Terrain"] = {}

    def get_contact_terrain(self, ball_radius: float) -> "Terrain":
        """Returns a Terrain whose height field is this one's rolling-ball
        contact envelope for a sphere of `ball_radius` (see
        compute_rolling_ball_envelope) -- what a ball can physically feel,
        rather than the raw point geometry. Use this terrain for physics
        (simulate_single_ball / run_ensemble_parallel already do this
        automatically); keep using the original terrain for plotting, so
        visualizations still show the true, sharp grain/peg geometry.
        Results are cached per ball_radius on this instance.
        """
        key = float(ball_radius)
        cached = self._contact_cache.get(key)
        if cached is not None:
            return cached

        contact_height_map = compute_rolling_ball_envelope(self.height_map, self.dx, self.dy, key)
        contact_terrain = Terrain(
            height_map=contact_height_map,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            slope_angle=self.slope_angle,
            roughness_amplitude_rough=self.roughness_amplitude_rough,
            roughness_amplitude_smooth=self.roughness_amplitude_smooth,
            grit_rough=self._grit_rough,
            grit_smooth=self._grit_smooth,
            surface_type=self.surface_type,
            z_offset=self.z_offset,
            tile_size_cm=self.tile_size_cm,
            roughness_transition_y=self.roughness_transition_y,
            peg_radius=self.peg_radius,
            peg_height=self.peg_height,
            peg_dx=self.peg_dx,
            peg_dy=self.peg_dy,
            peg_staggered=self.peg_staggered,
            peg_x_start=self.peg_x_start,
            peg_x_end=self.peg_x_end,
            peg_shape=self.peg_shape,
        )
        self._contact_cache[key] = contact_terrain
        return contact_terrain

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

    # Resolution: prefer explicit per-axis resolution_x/resolution_y (this is what
    # TerrainConfig actually exposes); fall back to a shared 'resolution' attribute
    # (used by the generate_calibrated_sandpaper/generate_galton_board helpers and
    # the CLI), and finally to 1000 if neither is set.
    fallback_res = getattr(cfg, "resolution", None)
    res_x = getattr(cfg, "resolution_x", None)
    res_y = getattr(cfg, "resolution_y", None)
    res_x = int(res_x) if res_x is not None else int(fallback_res) if fallback_res is not None else 1000
    res_y = int(res_y) if res_y is not None else int(fallback_res) if fallback_res is not None else 1000

    tile_size_cm = float(getattr(cfg, "tile_size_cm", 0.20))
    surface_type = str(getattr(cfg, "surface_type", "sandpaper")).lower()

    x = np.linspace(0, x_len, res_x)
    y = np.linspace(0, y_len, res_y)

    amp_rough = float(getattr(cfg, "roughness_amplitude_rough", 0.01))
    amp_smooth = float(getattr(cfg, "roughness_amplitude_smooth", 0.001))
    y_trans = float(getattr(cfg, "roughness_transition_y", y_len / 2.0))

    grit_r = getattr(cfg, "grit_rough", None)
    grit_s = getattr(cfg, "grit_smooth", None)
    seed = int(getattr(cfg, "seed", 42))

    xx, yy = np.meshgrid(x, y, indexing="ij")

    peg_radius = float(getattr(cfg, "peg_radius", 0.30))
    peg_height = float(getattr(cfg, "peg_height", 0.20))
    peg_dx = float(getattr(cfg, "peg_dx", 2.0))
    peg_dy = float(getattr(cfg, "peg_dy", 1.5))
    peg_staggered = bool(getattr(cfg, "peg_staggered", True))
    peg_x_start = float(getattr(cfg, "peg_x_start", 5.0))
    peg_x_end = getattr(cfg, "peg_x_end", None)
    peg_shape = str(getattr(cfg, "peg_shape", "cylinder"))

    if surface_type == "galton":
        normal_roughness = galton_board_bump_field(
            xx,
            yy,
            slope_angle=slope_angle,
            peg_radius=peg_radius,
            peg_height=peg_height,
            peg_dx=peg_dx,
            peg_dy=peg_dy,
            staggered=peg_staggered,
            peg_x_start=peg_x_start,
            peg_x_end=peg_x_end,
            shape=peg_shape,
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

    z_offset_cfg = getattr(cfg, "z_offset", None)
    if z_offset_cfg is None:
        # Default: enough headroom that z_base = z_offset - x*tan(angle) never goes
        # negative across the ramp (which would otherwise get clamped to a flat
        # floor by the np.maximum(0, ...) below). A fixed z_offset=15 only worked
        # by coincidence for ~29cm ramps; long ramps (100s of cm) need to scale.
        margin_cm = 5.0
        z_offset = float(x_len * np.tan(rad) + margin_cm)
    else:
        z_offset = float(z_offset_cfg)

    # Baseline inclined plane, offset so it stays above z=0 across the whole ramp
    z_base = z_offset - xx * np.tan(rad)

    if surface_type == "galton":
        # Pegs stand vertically (straight up, like real hardware) -- already a
        # vertical Z height, no normal->vertical conversion needed.
        delta_Z = normal_roughness
    else:
        # Sandpaper grains are modeled as tilted ellipsoids oriented along the
        # surface normal, so their height needs converting: a point at
        # normal-distance n above the base plane has vertical rise n / cos(theta)
        # (verified via explicit tangent/normal frame derivation).
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
        z_offset=z_offset,
        tile_size_cm=tile_size_cm,
        roughness_transition_y=y_trans,
        peg_radius=peg_radius,
        peg_height=peg_height,
        peg_dx=peg_dx,
        peg_dy=peg_dy,
        peg_staggered=peg_staggered,
        peg_x_start=peg_x_start,
        peg_x_end=peg_x_end,
        peg_shape=peg_shape,
    )


def generate_calibrated_sandpaper(
    grit_rough: float = 80.0,
    grit_smooth: float | None = None,
    slope_angle: float = 30.0,
    ramp_length: float = 29.0,
    length_y: float = 23.0,
    resolution: int = 2500,
    resolution_x: int | None = None,
    resolution_y: int | None = None,
    tile_size_cm: float = 0.20,
    seed: int = 42,
    **kwargs: Any,
) -> Terrain:
    """Generates a calibrated sandpaper Terrain instance directly from P-grit specifications.

    `resolution` sets both grid axes; pass `resolution_x`/`resolution_y` to use a
    different point count per axis (useful for long/narrow ramps).
    """
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
        resolution_x=resolution_x,
        resolution_y=resolution_y,
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
    resolution: int = 2500,
    resolution_x: int | None = None,
    resolution_y: int | None = None,
    peg_radius: float = 0.30,
    peg_height: float = 0.20,
    peg_dx: float = 2.0,
    peg_dy: float = 1.5,
    peg_staggered: bool = True,
    peg_x_start: float = 5.0,
    peg_x_end: float | None = None,
    peg_shape: str = "cylinder",
    **kwargs: Any,
) -> Terrain:
    """Generates a Galton board peg Terrain instance directly.

    `resolution` sets both grid axes; pass `resolution_x`/`resolution_y` to use a
    different point count per axis.
    """
    length_x = float(ramp_length * np.cos(np.radians(slope_angle)))

    cfg = SimpleNamespace(
        surface_type="galton",
        ramp_length=ramp_length,
        length_x=length_x,
        length_y=length_y,
        slope_angle=slope_angle,
        resolution=resolution,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
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
