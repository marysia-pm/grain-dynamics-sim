<<<<<<< HEAD
"""Physics Integration and Parallel Ensemble Simulation Engine.

Coordinate convention: X-Y is the horizontal ground plane, Z is vertical
(up). Gravity is always straight down, (0, 0, -g) — the incline is entirely
encoded in the terrain height field (see terrain.py), so it shows up here
only through the local surface normal.

Contact model: instantaneous collision response with restitution, not a
penalty spring. When the ball's surface touches the terrain, its position is
snapped onto the surface along the local normal and the normal component of
velocity is reflected and scaled by `restitution` (0 = fully inelastic,
1 = perfectly elastic). This avoids the stiff-spring stability problems of a
penalty method (no k_n/dt tuning needed) while still giving bouncy behavior.
Coulomb friction (optional, off by default) decelerates the tangential
velocity while in contact.
"""
=======
"""Simulation execution module with 3D physics integration and GPU (PyTorch) / CPU parallel ensemble support."""
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
<<<<<<< HEAD
=======
import torch
from tqdm import tqdm
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822

try:
    import torch
except ImportError:  # torch is only used for optional reproducible seeding
    torch = None


@dataclass
class BallConfig:
    """Configuration parameters for the rolling/bouncing sphere."""

    x0: float = 0.1
    y0: float = 11.5
    z0: float | None = None  # None -> spawn resting on the surface
    vx0: float = 0.0
    vy0: float = 0.0
    vz0: float = 0.0
    radius: float = 0.125  # cm
    mass: float = 1.0  # g
    restitution: float = 0.3  # 0 = inelastic, 1 = perfectly elastic bounce
    friction_mu: float = 0.0  # Coulomb friction coefficient while in contact


@dataclass
class PhysicsConfig:
    """Global physical constants."""

    gravity: float = 981.0  # cm/s^2, magnitude only — direction is always -z

    @property
    def g(self) -> float:
        """Alias property for backwards compatibility."""
        return self.gravity


@dataclass
class SimConfig:
    """Simulation time-stepping, output, and execution options."""

    dt: float = 1e-4
    t_max: float = 5.0
    num_runs: int = 10
    num_workers: int = 4
    seed: int | None = 42
    save_interval: int = 10


@dataclass
class TrajectoryRecord:
    """Storage container for recorded trajectory state histories."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    wx: np.ndarray  # kept for interface compatibility; always zero (no rotation modeled)
    wy: np.ndarray
    wz: np.ndarray
    terrain_z: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


<<<<<<< HEAD
Trajectory = TrajectoryRecord


def run_simulation(
    terrain: Any,
    ball_cfg: BallConfig,
    phys_cfg: PhysicsConfig,
    sim_cfg: SimConfig,
) -> TrajectoryRecord:
    """Integrates 3D motion of a sphere bouncing/sliding over rough terrain
    under plain vertical gravity, using restitution-based collision response.
    """
    dt = float(getattr(sim_cfg, "dt", 1e-4))
    t_max = float(getattr(sim_cfg, "t_max", 5.0))
    save_interval = int(getattr(sim_cfg, "save_interval", 10))

    r = float(ball_cfg.radius)
    e = float(getattr(ball_cfg, "restitution", 0.3))
    mu = float(getattr(ball_cfg, "friction_mu", 0.0))

    x = float(getattr(ball_cfg, "x0", 0.1))
    y = float(getattr(ball_cfg, "y0", 11.5))

    z_rest = float(terrain.get_height(x, y)) + r
    z = float(ball_cfg.z0) if ball_cfg.z0 is not None else z_rest

    vx, vy, vz = float(ball_cfg.vx0), float(ball_cfg.vy0), float(ball_cfg.vz0)

    g = float(getattr(phys_cfg, "gravity", getattr(phys_cfg, "g", 981.0)))
    gravity_vec = np.array([0.0, 0.0, -g], dtype=np.float64)

=======
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
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
    x_min, x_max = terrain.x_bounds
    y_min, y_max = terrain.y_bounds

    num_steps = int(t_max / dt)
    history_len = (num_steps // save_interval) + 1

    t_hist = np.zeros(history_len, dtype=np.float64)
    x_hist = np.zeros(history_len, dtype=np.float64)
    y_hist = np.zeros(history_len, dtype=np.float64)
    z_hist = np.zeros(history_len, dtype=np.float64)
    vx_hist = np.zeros(history_len, dtype=np.float64)
    vy_hist = np.zeros(history_len, dtype=np.float64)
    vz_hist = np.zeros(history_len, dtype=np.float64)
    w_zeros = np.zeros(history_len, dtype=np.float64)
    tz_hist = np.zeros(history_len, dtype=np.float64)

    rec_idx = 0
    t_curr = 0.0

<<<<<<< HEAD
    for step in range(num_steps):
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            break

        if step % save_interval == 0 and rec_idx < history_len:
            t_hist[rec_idx] = t_curr
            x_hist[rec_idx] = x
            y_hist[rec_idx] = y
            z_hist[rec_idx] = z
            vx_hist[rec_idx] = vx
            vy_hist[rec_idx] = vy
            vz_hist[rec_idx] = vz
            tz_hist[rec_idx] = float(terrain.get_height(x, y))
            rec_idx += 1

        # 1. Free-fall / unconstrained integration under plain gravity.
        vx += gravity_vec[0] * dt
        vy += gravity_vec[1] * dt
        vz += gravity_vec[2] * dt

        x += vx * dt
        y += vy * dt
        z += vz * dt

        # 2. Collision detection & response against the (possibly moved-to)
        #    surface, using the surface normal at the new (x, y).
        z_surf = float(terrain.get_height(x, y))
        dz_dx, dz_dy = terrain.get_gradient(x, y)
        n_raw = np.array([-float(dz_dx), -float(dz_dy), 1.0], dtype=np.float64)
        n = n_raw / np.linalg.norm(n_raw)

        # Perpendicular distance from the ball center to the surface,
        # approximated by projecting the vertical gap onto the normal.
        dist_to_surface = (z - z_surf) * n[2]
        penetration = r - dist_to_surface

        if penetration > 0:
            # Snap back onto the surface along the normal.
            x += penetration * n[0]
            y += penetration * n[1]
            z += penetration * n[2]

            v_vec = np.array([vx, vy, vz], dtype=np.float64)
            v_normal = np.dot(v_vec, n)

            if v_normal < 0.0:
                # Reflect the normal component with restitution; leave the
                # tangential component untouched here (friction handles it).
                v_vec = v_vec - (1.0 + e) * v_normal * n

            if mu > 0.0:
                v_n_component = np.dot(v_vec, n) * n
                v_t = v_vec - v_n_component
                speed_t = np.linalg.norm(v_t)
                if speed_t > 1e-9:
                    decel = min(speed_t, mu * g * dt)
                    v_t = v_t - decel * (v_t / speed_t)
                v_vec = v_t + v_n_component

            vx, vy, vz = v_vec

        t_curr += dt

    return TrajectoryRecord(
        t=t_hist[:rec_idx],
        x=x_hist[:rec_idx],
        y=y_hist[:rec_idx],
        z=z_hist[:rec_idx],
        vx=vx_hist[:rec_idx],
        vy=vy_hist[:rec_idx],
        vz=vz_hist[:rec_idx],
        wx=w_zeros[:rec_idx],
        wy=w_zeros[:rec_idx],
        wz=w_zeros[:rec_idx],
        terrain_z=tz_hist[:rec_idx],
        metadata={"slope_angle": getattr(terrain, "slope_angle", 0.0)},
=======
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
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
    )


def simulate_single_ball(
    terrain: Any,
    ball_cfg: BallConfig,
    phys_cfg: PhysicsConfig,
    sim_cfg: SimConfig | None = None,
) -> TrajectoryRecord:
    if sim_cfg is None:
        sim_cfg = SimConfig()
    return run_simulation(terrain, ball_cfg, phys_cfg, sim_cfg)


def _single_run_worker(args: tuple) -> TrajectoryRecord:
    terrain, state, phys_cfg, sim_cfg, seed = args

    if seed is not None:
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)

    if callable(getattr(sim_cfg, "runner", None)):
        return sim_cfg.runner(terrain, state, phys_cfg, sim_cfg)

    if isinstance(state, (tuple, list)):
        b_cfg = BallConfig(
            x0=float(state[0]),
            y0=float(state[1]),
            vx0=float(state[2]) if len(state) > 2 else 0.0,
            vy0=float(state[3]) if len(state) > 3 else 0.0,
        )
    else:
        b_cfg = state

    return run_simulation(terrain, b_cfg, phys_cfg, sim_cfg)


def run_ensemble_parallel(
<<<<<<< HEAD
    terrain: Any,
    states: Any,
    phys_cfg: Any,
    sim_cfg: Any = None,
    workers: int | None = None,
    initial_states: list | None = None,
    show_progress: bool = False,
    desc: str | None = None,
) -> list[TrajectoryRecord]:
    if initial_states is not None:
        states = initial_states

    if sim_cfg is None:
        sim_cfg = SimConfig()

    if not isinstance(states, (list, tuple, np.ndarray)):
        num_runs = int(
            getattr(
                sim_cfg,
                "num_runs",
                getattr(sim_cfg, "ensemble_size", getattr(sim_cfg, "k_max", getattr(sim_cfg, "n_runs", 1))),
            )
        )
        ensemble_states = [states] * num_runs
    else:
        ensemble_states = list(states)

    num_sims = len(ensemble_states)
    if num_sims == 0:
        return []

    if workers is None:
        workers = int(
            getattr(
                sim_cfg,
                "num_workers",
                getattr(sim_cfg, "workers", getattr(sim_cfg, "n_jobs", 4)),
            )
        )
    workers = max(1, min(int(workers), num_sims))

    base_seed = getattr(sim_cfg, "seed", getattr(sim_cfg, "random_seed", None))
    seeds = [base_seed + i for i in range(num_sims)] if base_seed is not None else [None] * num_sims

    task_args = [(terrain, ensemble_states[i], phys_cfg, sim_cfg, seeds[i]) for i in range(num_sims)]

    if workers == 1:
        return [_single_run_worker(args) for args in task_args]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        trajectories = list(executor.map(_single_run_worker, task_args))

    return trajectories
=======
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
>>>>>>> 23c6a278ce2639c520448873a38c765a4a2c7822
