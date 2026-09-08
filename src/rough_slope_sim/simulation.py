"""Simulation execution module with 3D physics integration and GPU (PyTorch) / CPU parallel ensemble support."""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .config import BallConfig, EnsembleConfig, PhysicsConfig
from .terrain import Terrain

Vec3 = tuple[float, float, float]


@dataclass
class Trajectory:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    reached_end: bool
    final_position: Vec3


def run_ensemble_gpu(
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    ensemble: EnsembleConfig,
    initial_states: list[tuple[float, float, float, float]],
    device: str = "cuda",
) -> list[Trajectory]:
    """Runs fully vectorized ensemble trajectories on GPU using PyTorch tensors."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    num_balls = len(initial_states)
    dt = physics.dt
    gravity = physics.gravity
    friction = physics.friction
    theta = np.radians(physics.slope_angle)

    a_ext_x = gravity * np.sin(theta)
    a_ext_z = -gravity * np.cos(theta)

    time_points = np.arange(0, physics.total_time, dt)
    n_steps = len(time_points)

    # Convert terrain grids to GPU Tensors
    t_x = torch.tensor(terrain.x, dtype=torch.float32, device=dev)
    t_y = torch.tensor(terrain.y, dtype=torch.float32, device=dev)
    t_z = torch.tensor(terrain.z, dtype=torch.float32, device=dev)
    t_dz_dx = torch.tensor(terrain.dz_dx, dtype=torch.float32, device=dev)
    t_dz_dy = torch.tensor(terrain.dz_dy, dtype=torch.float32, device=dev)

    x_min, x_max = terrain.x_bounds
    y_min, y_max = terrain.y_bounds
    dx, dy = terrain.dx, terrain.dy
    nx, ny = terrain.nx, terrain.ny

    # Initial ball states as GPU Tensors (K particles)
    x0_arr = np.array([s[0] for s in initial_states], dtype=np.float32)
    y0_arr = np.array([s[1] for s in initial_states], dtype=np.float32)
    vx0_arr = np.array([s[2] for s in initial_states], dtype=np.float32)
    vy0_arr = np.array([s[3] for s in initial_states], dtype=np.float32)

    x = torch.tensor(x0_arr, device=dev)
    y = torch.tensor(y0_arr, device=dev)
    vx = torch.tensor(vx0_arr, device=dev)
    vy = torch.tensor(vy0_arr, device=dev)

    # Surface height at initial positions
    gx0 = torch.clamp((x - x_min) / dx, 0, nx - 2).long()
    gy0 = torch.clamp((y - y_min) / dy, 0, ny - 2).long()
    z0 = t_z[gx0, gy0] + ball.radius
    z = z0.clone()
    vz = torch.zeros(num_balls, device=dev)

    # Output history storage [n_steps, num_balls]
    hist_x = torch.zeros((n_steps, num_balls), device=dev)
    hist_y = torch.zeros((n_steps, num_balls), device=dev)
    hist_z = torch.zeros((n_steps, num_balls), device=dev)
    hist_vx = torch.zeros((n_steps, num_balls), device=dev)
    hist_vy = torch.zeros((n_steps, num_balls), device=dev)
    hist_vz = torch.zeros((n_steps, num_balls), device=dev)
    active = torch.ones(num_balls, dtype=torch.bool, device=dev)

    for step in range(n_steps):
        hist_x[step] = x
        hist_y[step] = y
        hist_z[step] = z
        hist_vx[step] = vx
        hist_vy[step] = vy
        hist_vz[step] = vz

        # Boundary check
        in_bounds = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        active = active & in_bounds
        if not active.any():
            break

        # Fast Vectorized Bilinear Interpolation for Surface Heights & Normals
        gx = torch.clamp((x - x_min) / dx, 0, nx - 2)
        gy = torch.clamp((y - y_min) / dy, 0, ny - 2)
        ix = gx.long()
        iy = gy.long()
        rx = torch.clamp(gx - ix, 0.0, 1.0)
        ry = torch.clamp(gy - iy, 0.0, 1.0)

        # Bilinear Z
        z_surf = (
            (1.0 - rx) * (1.0 - ry) * t_z[ix, iy]
            + rx * (1.0 - ry) * t_z[ix + 1, iy]
            + (1.0 - rx) * ry * t_z[ix, iy + 1]
            + rx * ry * t_z[ix + 1, iy + 1]
        ) + ball.radius

        # Bilinear Normals
        dzdx = (
            (1.0 - rx) * (1.0 - ry) * t_dz_dx[ix, iy]
            + rx * (1.0 - ry) * t_dz_dx[ix + 1, iy]
            + (1.0 - rx) * ry * t_dz_dx[ix, iy + 1]
            + rx * ry * t_dz_dx[ix + 1, iy + 1]
        )
        dzdy = (
            (1.0 - rx) * (1.0 - ry) * t_dz_dy[ix, iy]
            + rx * (1.0 - ry) * t_dz_dy[ix + 1, iy]
            + (1.0 - rx) * ry * t_dz_dy[ix, iy + 1]
            + rx * ry * t_dz_dy[ix + 1, iy + 1]
        )

        norm = torch.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
        nx_vec, ny_vec, nz_vec = -dzdx / norm, -dzdy / norm, 1.0 / norm

        in_contact = z <= (z_surf + 1e-4)
        z = torch.where(in_contact, z_surf, z)

        v_norm = vx * nx_vec + vy * ny_vec + vz * nz_vec
        reflect_mask = in_contact & (v_norm < 0)

        vx = torch.where(reflect_mask, vx - v_norm * nx_vec, vx)
        vy = torch.where(reflect_mask, vy - v_norm * ny_vec, vy)
        vz = torch.where(reflect_mask, vz - v_norm * nz_vec, vz)

        dot_a_n = a_ext_x * nx_vec + a_ext_z * nz_vec
        a_tan_x = a_ext_x - dot_a_n * nx_vec
        a_tan_y = -dot_a_n * ny_vec
        a_tan_z = a_ext_z - dot_a_n * nz_vec

        v_speed = torch.sqrt(vx * vx + vy * vy + vz * vz)
        f_cond = in_contact & (v_speed > 1e-6) & (friction > 0)
        f_scale = torch.where(f_cond, -friction * abs(a_ext_z) / v_speed, torch.zeros_like(v_speed))

        ax = torch.where(in_contact, a_tan_x + f_scale * vx, torch.tensor(a_ext_x, device=dev))
        ay = torch.where(in_contact, a_tan_y + f_scale * vy, torch.tensor(0.0, device=dev))
        az = torch.where(in_contact, a_tan_z + f_scale * vz, torch.tensor(a_ext_z, device=dev))

        # Euler step
        vx = vx + ax * dt
        vy = vy + ay * dt
        vz = vz + az * dt

        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt

    # Transfer GPU results back to standard CPU Trajectory objects
    x_cpu = hist_x.cpu().numpy()
    y_cpu = hist_y.cpu().numpy()
    z_cpu = hist_z.cpu().numpy()
    vx_cpu = hist_vx.cpu().numpy()
    vy_cpu = hist_vy.cpu().numpy()
    vz_cpu = hist_vz.cpu().numpy()

    trajectories = []
    for k in range(num_balls):
        trajectories.append(
            Trajectory(
                time=time_points,
                x=x_cpu[:, k],
                y=y_cpu[:, k],
                z=z_cpu[:, k],
                vx=vx_cpu[:, k],
                vy=vy_cpu[:, k],
                vz=vz_cpu[:, k],
                reached_end=bool(x_cpu[-1, k] >= x_max),
                final_position=(float(x_cpu[-1, k]), float(y_cpu[-1, k]), float(z_cpu[-1, k])),
            )
        )
    return trajectories


def simulate_single_ball(
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    start_x: float,
    start_y: float,
    start_vx: float = 0.0,
    start_vy: float = 0.0,
) -> Trajectory:
    """CPU Single ball trajectory fallback."""
    x_min, x_max = terrain.x_bounds
    y_min, y_max = terrain.y_bounds

    z0 = float(terrain.get_height(start_x, start_y)) + ball.radius
    dt = physics.dt
    gravity = physics.gravity
    friction = physics.friction
    theta = np.radians(physics.slope_angle)

    a_ext_x = gravity * np.sin(theta)
    a_ext_z = -gravity * np.cos(theta)

    time_points = np.arange(0, physics.total_time, dt)
    n_max = len(time_points)

    time_buf, x_buf, y_buf = np.empty(n_max), np.empty(n_max), np.empty(n_max)
    z_buf, vx_buf, vy_buf, vz_buf = np.empty(n_max), np.empty(n_max), np.empty(n_max), np.empty(n_max)

    x, y, z = start_x, start_y, z0
    vx, vy, vz = start_vx, start_vy, 0.0

    n = 0
    for t in time_points:
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            break

        time_buf[n], x_buf[n], y_buf[n], z_buf[n] = t, x, y, z
        vx_buf[n], vy_buf[n], vz_buf[n] = vx, vy, vz
        n += 1

        nx_val, ny_val, nz_val = terrain.get_normal(x, y)
        z_surf = float(terrain.get_height(x, y)) + ball.radius

        if z <= z_surf + 1e-4:
            z = z_surf
            v_normal = vx * nx_val + vy * ny_val + vz * nz_val
            if v_normal < 0:
                vx -= v_normal * nx_val
                vy -= v_normal * ny_val
                vz -= v_normal * nz_val

            dot_a_n = a_ext_x * nx_val + a_ext_z * nz_val
            a_tan_x = a_ext_x - dot_a_n * nx_val
            a_tan_y = -dot_a_n * ny_val
            a_tan_z = a_ext_z - dot_a_n * nz_val

            v_speed = np.sqrt(vx * vx + vy * vy + vz * vz)
            if v_speed > 1e-6 and friction > 0:
                f_scale = -friction * abs(a_ext_z) / v_speed
                ax, ay, az = a_tan_x + f_scale * vx, a_tan_y + f_scale * vy, a_tan_z + f_scale * vz
            else:
                ax, ay, az = a_tan_x, a_tan_y, a_tan_z
        else:
            ax, ay, az = a_ext_x, 0.0, a_ext_z

        vx, vy, vz = vx + ax * dt, vy + ay * dt, vz + az * dt
        x, y, z = x + vx * dt, y + vy * dt, z + vz * dt

    return Trajectory(
        time=time_buf[:n], x=x_buf[:n], y=y_buf[:n], z=z_buf[:n],
        vx=vx_buf[:n], vy=vy_buf[:n], vz=vz_buf[:n],
        reached_end=x >= x_max, final_position=(x, y, z)
    )


def _sim_worker(args):
    terrain, ball, physics, start_x, start_y, start_vx, start_vy = args
    return simulate_single_ball(terrain, ball, physics, start_x, start_y, start_vx, start_vy)


def run_ensemble_parallel(
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    ensemble: EnsembleConfig,
    initial_states: list[tuple[float, float, float, float]] | None = None,
    show_progress: bool = True,
    use_gpu: bool = True,
    desc: str = "  ├── Simulating ensemble",
) -> list[Trajectory]:
    """Runs parallel ensemble simulation automatically routing to GPU if available."""
    rng = np.random.default_rng(ensemble.seed)
    states = []

    if initial_states:
        states = initial_states
    else:
        for _ in range(ensemble.k_max):
            dx = rng.normal(0.0, ensemble.x_jitter_std)
            dy = rng.uniform(-ensemble.y_jitter_max, ensemble.y_jitter_max)
            states.append((ensemble.start_x + dx, ensemble.start_y + dy, 0.0, 0.0))
    print("PyTorch Version :", torch.__version__)
    print("CUDA Available  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device Name :", torch.cuda.get_device_name(0))
    else:
        print("Warning: PyTorch is running on CPU only!")
    

    if use_gpu and torch.cuda.is_available():
        return run_ensemble_gpu(terrain, ball, physics, ensemble, states, device="cuda")

    tasks = [(terrain, ball, physics, x0, y0, vx0, vy0) for x0, y0, vx0, vy0 in states]
    max_workers = min(len(tasks), os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        mapped = executor.map(_sim_worker, tasks)
        if show_progress:
            mapped = tqdm(mapped, total=len(tasks), desc=desc, leave=False)
        return list(mapped)