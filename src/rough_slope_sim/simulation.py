"""Simulation execution module with 3D physics integration and parallel ensemble support."""

from __future__ import annotations

import concurrent.futures
import os
import numpy as np
from tqdm import tqdm

from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig
from rough_slope_sim.terrain import Terrain


class Trajectory:
    def __init__(
        self,
        time: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        reached_end: bool,
        final_position: tuple[float, float, float],
    ):
        self.time = time
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.reached_end = reached_end
        self.final_position = final_position


def _get_terrain_normal(
    terrain: Terrain, x: float, y: float, eps: float = 1e-3
) -> tuple[float, float, float]:
    """Calculates normalized surface normal vector at (x, y) using finite differences."""
    if hasattr(terrain, "get_normal"):
        return terrain.get_normal(x, y)

    z_x1 = terrain.get_height(x + eps, y)
    z_x0 = terrain.get_height(x - eps, y)
    z_y1 = terrain.get_height(x, y + eps)
    z_y0 = terrain.get_height(x, y - eps)

    dz_dx = (z_x1 - z_x0) / (2.0 * eps)
    dz_dy = (z_y1 - z_y0) / (2.0 * eps)

    n = np.array([-dz_dx, -dz_dy, 1.0])
    n /= np.linalg.norm(n)
    return float(n[0]), float(n[1]), float(n[2])


def _step(
    pos: tuple[float, float, float],
    vel: tuple[float, float, float],
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Calculates one time step of the ball's motion on the 3D rough terrain."""
    x, y, z = pos
    vx, vy, vz = vel
    dt = physics.dt

    nx, ny, nz = _get_terrain_normal(terrain, x, y)
    z_surf = terrain.get_height(x, y) + ball.radius

    g_x = getattr(
        physics, "g_eff_x", physics.gravity * np.sin(np.radians(getattr(physics, "slope_angle", 10.0)))
    )
    g_z = -getattr(physics, "gravity", 981.0) * np.cos(np.radians(getattr(physics, "slope_angle", 10.0)))

    a_ext = np.array([g_x, 0.0, g_z])
    n_vec = np.array([nx, ny, nz])
    n_vec /= np.linalg.norm(n_vec)

    v_vec = np.array([vx, vy, vz])

    if z <= z_surf + 1e-4:
        z = z_surf
        v_normal = np.dot(v_vec, n_vec)
        if v_normal < 0:
            v_vec = v_vec - v_normal * n_vec

        a_tangent = a_ext - np.dot(a_ext, n_vec) * n_vec

        v_speed = np.linalg.norm(v_vec)
        mu = getattr(physics, "friction", 0.15)
        if v_speed > 1e-6:
            f_friction = -mu * abs(g_z) * (v_vec / v_speed)
        else:
            f_friction = np.zeros(3)

        a_tot = a_tangent + f_friction
    else:
        a_tot = a_ext

    v_new = v_vec + a_tot * dt
    p_new = np.array([x, y, z]) + v_new * dt

    z_new_surf = terrain.get_height(p_new[0], p_new[1]) + ball.radius
    if p_new[2] < z_new_surf:
        p_new[2] = z_new_surf

    return (float(p_new[0]), float(p_new[1]), float(p_new[2])), (
        float(v_new[0]),
        float(v_new[1]),
        float(v_new[2]),
    )


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

    z0 = terrain.get_height(start_x, start_y) + ball.radius
    pos = (start_x, start_y, z0)
    vel = (start_vx, start_vy, 0.0)
    time_points = np.arange(0, physics.total_time, physics.dt)

    xs, ys, zs, vxs, vys, vzs, used_time = [], [], [], [], [], [], []

    for t in time_points:
        x, y, z = pos
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            break

        used_time.append(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        vxs.append(vel[0])
        vys.append(vel[1])
        vzs.append(vel[2])

        pos, vel = _step(pos, vel, terrain, ball, physics)

    return Trajectory(
        time=np.array(used_time),
        x=np.array(xs),
        y=np.array(ys),
        z=np.array(zs),
        vx=np.array(vxs),
        vy=np.array(vys),
        vz=np.array(vzs),
        reached_end=pos[0] >= x_max,
        final_position=pos,
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
    """Runs parallel ensemble simulation with custom sampled initial states."""
    rng = np.random.default_rng(ensemble.seed)
    tasks = []

    if initial_states and len(initial_states) > 0:
        for x0, y0, vx0, vy0 in initial_states:
            tasks.append((terrain, ball, physics, x0, y0, vx0, vy0))
    else:
        for _ in range(ensemble.k_max):
            dx = rng.normal(0.0, ensemble.x_jitter_std)
            dy = rng.uniform(-ensemble.y_jitter_max, ensemble.y_jitter_max)
            tasks.append((terrain, ball, physics, ensemble.start_x + dx, ensemble.start_y + dy, 0.0, 0.0))

    max_workers = min(len(tasks), os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if show_progress:
            trajectories = list(
                tqdm(
                    executor.map(_sim_worker, tasks),
                    total=len(tasks),
                    desc=desc,
                    leave=False,
                )
            )
        else:
            trajectories = list(executor.map(_sim_worker, tasks))
    return trajectories
