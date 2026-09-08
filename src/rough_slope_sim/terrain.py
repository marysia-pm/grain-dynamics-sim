"""Terrain generation and surface representation for sandpaper micro-geometry using ISO 6344 standards and Nanovea benchmarks."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from .config import TerrainConfig

# Benchmark NANOVEA Profilometry Reference Data
NANOVEA_DATA = {
    "grit": np.array([120, 180, 320, 800, 2000], dtype=np.float64),
    "diam_um": np.array([127.0, 105.6, 67.18, 28.16, 21.27], dtype=np.float64),
    "Sa_um": np.array([42.37, 27.28, 17.92, 6.273, 3.639], dtype=np.float64),
}

# Log-log power-law interpolation for grit estimation
_log_diams = np.log(NANOVEA_DATA["diam_um"])
_log_grits = np.log(NANOVEA_DATA["grit"])
_grit_interp_func = interp1d(_log_diams, _log_grits, kind="linear", fill_value="extrapolate")


def estimate_grit(mean_diam_um: float) -> int:
    """Calculates estimated grit rating from particle diameter using log-log interpolation."""
    log_d = np.log(mean_diam_um)
    return max(1, int(round(np.exp(_grit_interp_func(log_d)))))


def generate_calibrated_sandpaper(
    target_diam_um: float = 127.0,
    target_Sa_um: float = 42.37,
    patch_size_cm: float | None = 0.20,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    incline_deg: float = 30.0,
    grid_res: int | tuple[int, int] = 500,
    tile_size_cm: float = 0.20,
    seed: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    int,
]:
    """Evaluates micro-grit surface topography in localized spatial tiles to keep the memory footprint
    constant regardless of total surface area.
    """
    rng = np.random.default_rng(seed)

    if x_range is None or y_range is None:
        p_size = patch_size_cm if patch_size_cm is not None else 0.20
        x_range = (0.0, p_size)
        y_range = (0.0, p_size)

    if isinstance(grid_res, int):
        nx, ny = grid_res, grid_res
    else:
        nx, ny = grid_res

    x_min, x_max = x_range
    y_min, y_max = y_range
    width_cm = x_max - x_min
    height_cm = y_max - y_min

    r_mean_cm = (target_diam_um / 2.0) / 10000.0
    r_std_cm = r_mean_cm * 0.15
    spacing = r_mean_cm * 1.35
    pad = r_mean_cm * 3.0
    z_aspect = 1.30

    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    X_m, Y_m = np.meshgrid(x_coords, y_coords, indexing="ij")
    z_local = np.zeros((nx, ny), dtype=np.float64)

    # Divide global grid into tiles
    x_tiles = np.arange(x_min, x_max, tile_size_cm)
    y_tiles = np.arange(y_min, y_max, tile_size_cm)

    for i, tx_start in enumerate(x_tiles):
        tx_end = min(tx_start + tile_size_cm, x_max)
        if i == len(x_tiles) - 1:
            ix = np.where((x_coords >= tx_start) & (x_coords <= tx_end))[0]
        else:
            ix = np.where((x_coords >= tx_start) & (x_coords < tx_end))[0]

        if len(ix) == 0:
            continue

        for j, ty_start in enumerate(y_tiles):
            ty_end = min(ty_start + tile_size_cm, y_max)
            if j == len(y_tiles) - 1:
                iy = np.where((y_coords >= ty_start) & (y_coords <= ty_end))[0]
            else:
                iy = np.where((y_coords >= ty_start) & (y_coords < ty_end))[0]

            if len(iy) == 0:
                continue

            sub_X = X_m[np.ix_(ix, iy)]
            sub_Y = Y_m[np.ix_(ix, iy)]
            tile_pts = np.column_stack([sub_X.ravel(), sub_Y.ravel()])

            # Local seed placement strictly within tile boundaries + padding
            gx = np.arange(tx_start - pad, tx_end + pad, spacing)
            gy = np.arange(ty_start - pad, ty_end + pad, spacing)
            GX, GY = np.meshgrid(gx, gy)

            jitter = spacing * 0.40
            cx = (GX + rng.uniform(-jitter, jitter, size=GX.shape)).ravel()
            cy = (GY + rng.uniform(-jitter, jitter, size=GY.shape)).ravel()

            n_grains = len(cx)
            r_grains = np.clip(
                rng.normal(r_mean_cm, r_std_cm, size=n_grains),
                r_mean_cm * 0.5,
                r_mean_cm * 1.5,
            )
            z_offsets = rng.normal(0.0, r_mean_cm * 0.40, size=n_grains)

            # Localized cKDTree query
            tree = cKDTree(np.column_stack([cx, cy]))
            dists, indices = tree.query(tile_pts, k=4)

            z_flat = np.full(len(tile_pts), -1e5, dtype=np.float64)
            for k in range(4):
                d_k = dists[:, k]
                idx_k = indices[:, k]
                r_k = r_grains[idx_k]
                z_off_k = z_offsets[idx_k]
                diff_k = np.maximum(0.0, r_k**2 - d_k**2)
                z_k = z_off_k + z_aspect * np.sqrt(diff_k)
                np.maximum(z_flat, z_k, out=z_flat)

            z_tile = z_flat.reshape((len(ix), len(iy)))
            z_tile = np.maximum(z_tile, -r_mean_cm * 0.10)
            z_local[np.ix_(ix, iy)] = z_tile

    z_local -= z_local.min()

    # Sa Precision Calibration across the stitched domain
    current_Sa = float(np.mean(np.abs(z_local - np.mean(z_local))) * 10000.0)
    if current_Sa > 0:
        z_local *= target_Sa_um / current_Sa

    Sa_um = float(np.mean(np.abs(z_local - np.mean(z_local))) * 10000.0)
    grit_val = estimate_grit(target_diam_um)

    # 3D Rigid Rotation
    theta = np.radians(incline_deg)
    X_3d = X_m
    Y_3d = Y_m * np.cos(theta) - z_local * np.sin(theta)
    Z_3d = Y_m * np.sin(theta) + z_local * np.cos(theta)

    return (
        X_m,
        Y_m,
        z_local,
        X_3d,
        Y_3d,
        Z_3d,
        max(width_cm, height_cm),
        Sa_um,
        target_diam_um,
        grit_val,
    )


def p_value_to_d50_cm(p_value: float) -> float:
    """Converts FEPA P-value to average particle diameter d50 (in cm) using ISO 6344 standard fit."""
    if p_value <= 0:
        return 0.02
    d50_um = 12500.0 / (float(p_value) ** 0.95)
    return d50_um / 10000.0


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

        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        self.nx = len(x)
        self.ny = len(y)

        dz_dx, dz_dy = np.gradient(z, self.dx, self.dy)
        self.dz_dx = dz_dx
        self.dz_dy = dz_dy

        self.slope_angle = 30.0

    def _bilinear_interp(
        self, grid: np.ndarray, x: float | np.ndarray, y: float | np.ndarray
    ) -> float | np.ndarray:
        """Fast O(1) uniform bilinear interpolation."""
        is_scalar = np.isscalar(x) and np.isscalar(y)
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)

        gx = (x_arr - self.x_bounds[0]) / self.dx
        gy = (y_arr - self.y_bounds[0]) / self.dy

        ix = np.clip(np.floor(gx).astype(int), 0, self.nx - 2)
        iy = np.clip(np.floor(gy).astype(int), 0, self.ny - 2)

        rx = np.clip(gx - ix, 0.0, 1.0)
        ry = np.clip(gy - iy, 0.0, 1.0)

        f00 = grid[ix, iy]
        f10 = grid[ix + 1, iy]
        f01 = grid[ix, iy + 1]
        f11 = grid[ix + 1, iy + 1]

        res = (1.0 - rx) * (1.0 - ry) * f00 + rx * (1.0 - ry) * f10 + (1.0 - rx) * ry * f01 + rx * ry * f11
        return float(res[0]) if is_scalar else res

    def get_elevation(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return self._bilinear_interp(self.z, x, y)

    def get_height(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return self.get_elevation(x, y)

    def get_normal(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        zx = self._bilinear_interp(self.dz_dx, x, y)
        zy = self._bilinear_interp(self.dz_dy, x, y)

        if np.isscalar(zx):
            norm = np.sqrt(zx * zx + zy * zy + 1.0)
            return np.array([-zx / norm, -zy / norm, 1.0 / norm], dtype=np.float64)

        normals = np.column_stack([-zx, -zy, np.ones_like(zx)])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.maximum(norms, 1e-12)


def generate_terrain(config: TerrainConfig) -> Terrain:
    """Generates an inclined sandpaper surface using tiled KDTree micro-geometry
    calibrated directly via generate_calibrated_sandpaper.
    """
    diam_rough_um = max(config.roughness_amplitude_rough * 10000.0, 1.0)
    diam_smooth_um = max(config.roughness_amplitude_smooth * 10000.0, 1.0)

    sa_rough_um = diam_rough_um * 0.33
    sa_smooth_um = diam_smooth_um * 0.33

    seed_rough = config.seed
    seed_smooth = (config.seed + 1000) if config.seed is not None else None

    # Tiled evaluation across rough and smooth domains
    X_m, Y_m, z_rough, _, _, _, _, _, _, grit_rough = generate_calibrated_sandpaper(
        target_diam_um=diam_rough_um,
        target_Sa_um=sa_rough_um,
        x_range=config.x_range,
        y_range=config.y_range,
        grid_res=config.resolution,
        tile_size_cm=0.20,
        incline_deg=0.0,
        seed=seed_rough,
    )

    _, _, z_smooth, _, _, _, _, _, _, grit_smooth = generate_calibrated_sandpaper(
        target_diam_um=diam_smooth_um,
        target_Sa_um=sa_smooth_um,
        x_range=config.x_range,
        y_range=config.y_range,
        grid_res=config.resolution,
        tile_size_cm=0.20,
        incline_deg=0.0,
        seed=seed_smooth,
    )

    # Sigmoidal transition blending
    weights = 1.0 / (1.0 + np.exp((Y_m - config.roughness_transition_y) / 0.15))
    z_micro = weights * z_rough + (1.0 - weights) * z_smooth

    # Macro incline slope
    z_incline = -X_m * np.tan(np.radians(config.slope_angle))
    z = z_incline + z_micro

    x_vec = X_m[:, 0]
    y_vec = Y_m[0, :]

    terrain = Terrain(
        x=x_vec,
        y=y_vec,
        z=z,
        p_value_mean=estimate_grit((diam_rough_um + diam_smooth_um) / 2.0),
        p_value_rough=grit_rough,
        p_value_smooth=grit_smooth,
    )
    terrain.slope_angle = config.slope_angle
    return terrain
