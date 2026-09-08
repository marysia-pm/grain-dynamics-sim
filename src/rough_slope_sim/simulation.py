"""Simulation execution module with 3D physics integration and parallel ensemble support."""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass

import numpy as np
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


def simulate_single_ball(
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    start_x: float,
    start_y: float,
    start_vx: float = 0.0,
    start_vy: float = 0.0,
) -> Trajectory:
    """Simulates a single ball trajectory given initial position and velocity."""
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

    time_buf = np.empty(n_max, dtype=np.float64)
    x_buf = np.empty(n_max, dtype=np.float64)
    y_buf = np.empty(n_max, dtype=np.float64)
    z_buf = np.empty(n_max, dtype=np.float64)
    vx_buf = np.empty(n_max, dtype=np.float64)
    vy_buf = np.empty(n_max, dtype=np.float64)
    vz_buf = np.empty(n_max, dtype=np.float64)

    x, y, z = start_x, start_y, z0
    vx, vy, vz = start_vx, start_vy, 0.0

    n = 0
    for t in time_points:
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            break

        time_buf[n] = t
        x_buf[n], y_buf[n], z_buf[n] = x, y, z
        vx_buf[n], vy_buf[n], vz_buf[n] = vx, vy, vz
        n += 1

        nx, ny, nz = terrain.get_normal(x, y)
        z_surf = float(terrain.get_height(x, y)) + ball.radius

        if z <= z_surf + 1e-4:
            z = z_surf
            v_normal = vx * nx + vy * ny + vz * nz
            if v_normal < 0:
                vx -= v_normal * nx
                vy -= v_normal * ny
                vz -= v_normal * nz

            dot_a_n = a_ext_x * nx + a_ext_z * nz
            a_tan_x = a_ext_x - dot_a_n * nx
            a_tan_y = -dot_a_n * ny
            a_tan_z = a_ext_z - dot_a_n * nz

            v_speed = np.sqrt(vx * vx + vy * vy + vz * vz)
            if v_speed > 1e-6 and friction > 0:
                f_scale = -friction * abs(a_ext_z) / v_speed
                ax = a_tan_x + f_scale * vx
                ay = a_tan_y + f_scale * vy
                az = a_tan_z + f_scale * vz
            else:
                ax, ay, az = a_tan_x, a_tan_y, a_tan_z
        else:
            ax, ay, az = a_ext_x, 0.0, a_ext_z

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        x += vx * dt
        y += vy * dt
        z += vz * dt

        z_new_surf = float(terrain.get_height(x, y)) + ball.radius
        if z < z_new_surf:
            z = z_new_surf

    return Trajectory(
        time=time_buf[:n],
        x=x_buf[:n],
        y=y_buf[:n],
        z=z_buf[:n],
        vx=vx_buf[:n],
        vy=vy_buf[:n],
        vz=vz_buf[:n],
        reached_end=x >= x_max,
        final_position=(x, y, z),
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
    desc: str = "  ├── Simulating ensemble",
) -> list[Trajectory]:
    """Runs a parallel ensemble simulation."""
    rng = np.random.default_rng(ensemble.seed)
    tasks = []

    if initial_states:
        for x0, y0, vx0, vy0 in initial_states:
            tasks.append((terrain, ball, physics, x0, y0, vx0, vy0))
    else:
        for _ in range(ensemble.k_max):
            dx = rng.normal(0.0, ensemble.x_jitter_std)
            dy = rng.uniform(-ensemble.y_jitter_max, ensemble.y_jitter_max)
            tasks.append((terrain, ball, physics, ensemble.start_x + dx, ensemble.start_y + dy, 0.0, 0.0))

    max_workers = min(len(tasks), os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        mapped = executor.map(_sim_worker, tasks)
        if show_progress:
            mapped = tqdm(mapped, total=len(tasks), desc=desc, leave=False)
        trajectories = list(mapped)
    return trajectories
