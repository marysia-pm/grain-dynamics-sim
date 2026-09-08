"""3D Terrain Generation Module with GPU (PyTorch CUDA) Acceleration and CPU Interpolation Support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Nanovea Profilometer Calibration Data (d50 in cm, Sa in cm)
NANOVEA_DATA: dict[int, dict[str, float]] = {
    60: {"d50": 0.0269, "sa": 0.0050},
    80: {"d50": 0.0201, "sa": 0.0038},
    120: {"d50": 0.0125, "sa": 0.0024},
    180: {"d50": 0.0082, "sa": 0.0016},
    240: {"d50": 0.0058, "sa": 0.0011},
    320: {"d50": 0.0046, "sa": 0.0009},
}


def nanovea_d50_from_grit(grit: float) -> float:
    """Estimates mean grain diameter d50 (in cm) from P-grit value using empirical Nanovea lookup or power-law fit."""
    int_grit = int(round(grit))
    if int_grit in NANOVEA_DATA:
        return NANOVEA_DATA[int_grit]["d50"]
    return (4750.0 / (grit**0.85)) * 1e-4


def estimate_grit(d50_cm: float) -> float:
    """Estimates P-grit rating from grain diameter d50 (in cm)."""
    for grit, data in NANOVEA_DATA.items():
        if np.isclose(d50_cm, data["d50"], atol=1e-4):
            return float(grit)
    if d50_cm <= 0:
        return 80.0
    return float((0.475 / d50_cm) ** (1.0 / 0.85))


@dataclass
class Terrain:
    """3D Incline terrain container with heightmaps, pre-calculated gradients, and spatial interpolation."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dz_dx: np.ndarray | None = None
    dz_dy: np.ndarray | None = None
    slope_angle: float = 10.0
    p_value_rough: float = 80.0
    p_value_smooth: float = 180.0

    def __post_init__(self):
        # Infer grid resolutions and bounding domains
        if self.x.ndim == 2:
            self.dx = float(abs(self.x[1, 0] - self.x[0, 0]))
            self.dy = float(abs(self.y[0, 1] - self.y[0, 0]))
            self.nx, self.ny = self.x.shape
            self.x_bounds = (float(self.x.min()), float(self.x.max()))
            self.y_bounds = (float(self.y.min()), float(self.y.max()))
        else:
            self.dx = float(abs(self.x[1] - self.x[0])) if len(self.x) > 1 else 0.02
            self.dy = float(abs(self.y[1] - self.y[0])) if len(self.y) > 1 else 0.02
            self.nx, self.ny = len(self.x), len(self.y)
            self.x_bounds = (float(self.x[0]), float(self.x[-1]))
            self.y_bounds = (float(self.y[0]), float(self.y[-1]))

        # Compute CPU fallback surface gradients if missing
        if self.dz_dx is None or self.dz_dy is None:
            self.dz_dx, self.dz_dy = np.gradient(self.z, self.dx, self.dy)

    def get_height(self, x_val: float, y_val: float) -> float:
        """Bilinear interpolation for surface height Z at position (x_val, y_val)."""
        gx = np.clip((x_val - self.x_bounds[0]) / self.dx, 0, self.nx - 2)
        gy = np.clip((y_val - self.y_bounds[0]) / self.dy, 0, self.ny - 2)
        ix, iy = int(gx), int(gy)
        rx, ry = gx - ix, gy - iy

        z00 = self.z[ix, iy]
        z10 = self.z[ix + 1, iy]
        z01 = self.z[ix, iy + 1]
        z11 = self.z[ix + 1, iy + 1]

        return float((1 - rx) * (1 - ry) * z00 + rx * (1 - ry) * z10 + (1 - rx) * ry * z01 + rx * ry * z11)

    def get_gradient(self, x_val: float, y_val: float) -> tuple[float, float]:
        """Bilinear interpolation for surface gradients (dz_dx, dz_dy) at position (x_val, y_val)."""
        gx = np.clip((x_val - self.x_bounds[0]) / self.dx, 0, self.nx - 2)
        gy = np.clip((y_val - self.y_bounds[0]) / self.dy, 0, self.ny - 2)
        ix, iy = int(gx), int(gy)
        rx, ry = gx - ix, gy - iy

        dzdx_00 = self.dz_dx[ix, iy]
        dzdx_10 = self.dz_dx[ix + 1, iy]
        dzdx_01 = self.dz_dx[ix, iy + 1]
        dzdx_11 = self.dz_dx[ix + 1, iy + 1]

        dzdy_00 = self.dz_dy[ix, iy]
        dzdy_10 = self.dz_dy[ix + 1, iy]
        dzdy_01 = self.dz_dy[ix, iy + 1]
        dzdy_11 = self.dz_dy[ix + 1, iy + 1]

        dzdx = (1 - rx) * (1 - ry) * dzdx_00 + rx * (1 - ry) * dzdx_10 + (1 - rx) * ry * dzdx_01 + rx * ry * dzdx_11
        dzdy = (1 - rx) * (1 - ry) * dzdy_00 + rx * (1 - ry) * dzdy_10 + (1 - rx) * ry * dzdy_01 + rx * ry * dzdy_11

        return float(dzdx), float(dzdy)


def generate_terrain_gpu(
    x_min: float = 0.0,
    x_max: float = 29.0,
    y_min: float = 0.0,
    y_max: float = 23.0,
    dx: float = 0.02,
    dy: float = 0.02,
    slope_angle_deg: float = 10.0,
    p_value_left: float = 80.0,
    p_value_right: float = 180.0,
    is_dual: bool = True,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates 3D terrain heightmap and gradients on GPU using PyTorch tensors."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. 2D Meshgrid on GPU
    x_coords = torch.arange(x_min, x_max + dx, dx, device=dev, dtype=torch.float32)
    y_coords = torch.arange(y_min, y_max + dy, dy, device=dev, dtype=torch.float32)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing="ij")

    # 2. Base Incline Plane Z_base = -X * tan(theta)
    theta = np.radians(slope_angle_deg)
    Z_base = -X * float(np.tan(theta))

    # 3. Dual-Grit Micro-Roughness Scale (d50 estimation in cm)
    d50_left = nanovea_d50_from_grit(p_value_left)
    d50_right = nanovea_d50_from_grit(p_value_right) if is_dual else d50_left

    roughness_map = torch.where(Y < (y_max / 2.0), d50_left * 0.5, d50_right * 0.5)

    # 4. Generate Random Noise and Smooth with Gaussian Kernel on GPU
    noise = torch.randn_like(X) * roughness_map

    kernel_size = 7
    sigma = 1.2
    k_1d = torch.exp(
        -0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=dev).float() / sigma) ** 2
    )
    kernel_2d = torch.outer(k_1d, k_1d)
    kernel_2d /= kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    # Fast GPU 2D Reflective Convolution
    padded_noise = torch.nn.functional.pad(
        noise.unsqueeze(0).unsqueeze(0), (kernel_size // 2,) * 4, mode="reflect"
    )
    z_rough = torch.nn.functional.conv2d(padded_noise, kernel_2d).squeeze()

    Z = Z_base + z_rough

    # 5. Partial Derivatives dZ/dX and dZ/dY via GPU Finite Differences
    dZ_dX, dZ_dY = torch.gradient(Z, spacing=(dx, dy))

    return (
        X.cpu().numpy(),
        Y.cpu().numpy(),
        Z.cpu().numpy(),
        dZ_dX.cpu().numpy(),
        dZ_dY.cpu().numpy(),
    )


def generate_terrain(
    x_min: float = 0.0,
    x_max: float = 29.0,
    y_min: float = 0.0,
    y_max: float = 23.0,
    dx: float = 0.02,
    dy: float = 0.02,
    slope_angle_deg: float = 10.0,
    p_value_left: float = 80.0,
    p_value_right: float = 180.0,
    is_dual: bool = True,
    use_gpu: bool = True,
) -> Terrain:
    """Convenience factory function returning a pre-populated Terrain instance."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    X, Y, Z, dz_dx, dz_dy = generate_terrain_gpu(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        dx=dx,
        dy=dy,
        slope_angle_deg=slope_angle_deg,
        p_value_left=p_value_left,
        p_value_right=p_value_right,
        is_dual=is_dual,
        device=device,
    )
    return Terrain(
        x=X,
        y=Y,
        z=Z,
        dz_dx=dz_dx,
        dz_dy=dz_dy,
        slope_angle=slope_angle_deg,
        p_value_rough=p_value_left,
        p_value_smooth=p_value_right,
    )


def generate_calibrated_sandpaper(
    x_min: float = 0.0,
    x_max: float = 29.0,
    y_min: float = 0.0,
    y_max: float = 23.0,
    dx: float = 0.02,
    dy: float = 0.02,
    slope_angle_deg: float = 10.0,
    p_value_left: float = 80.0,
    p_value_right: float = 180.0,
    is_dual: bool = True,
    use_gpu: bool = True,
) -> Terrain:
    """Alias function maintained for backwards compatibility with plotting scripts."""
    return generate_terrain(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        dx=dx,
        dy=dy,
        slope_angle_deg=slope_angle_deg,
        p_value_left=p_value_left,
        p_value_right=p_value_right,
        is_dual=is_dual,
        use_gpu=use_gpu,
    )