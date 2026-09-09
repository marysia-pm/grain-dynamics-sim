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

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
