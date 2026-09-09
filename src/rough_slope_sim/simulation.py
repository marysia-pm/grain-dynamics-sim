"""Physics Integration and Parallel Ensemble Simulation Engine.

Coordinate convention: X-Y is the horizontal ground plane, Z is vertical
(up). Gravity is always straight down, (0, 0, -g) — the incline is entirely
encoded in the terrain height field (see terrain.py), so it shows up here
only through the local surface normal.

Contact model: instantaneous collision response with restitution, not a
penalty spring. When the ball touches the terrain, its position is snapped
onto the surface along the local normal and the normal component of
velocity is reflected and scaled by `restitution` (0 = fully inelastic,
1 = perfectly elastic). This avoids the stiff-spring stability problems of a
penalty method while still giving bouncy behavior. Coulomb friction
(off by default) decelerates the tangential velocity while in contact.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .config import BallConfig, EnsembleConfig, PhysicsConfig, SimConfig

try:
    import torch
except ImportError:  # torch is only used for optional reproducible seeding
    torch = None


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
    dt = float(getattr(sim_cfg, "dt", 5e-4))
    t_max = float(getattr(sim_cfg, "t_max", 2.0))
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
    gz = -g  # gravity always points straight down in world coordinates

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

        # 1. Free-fall / unconstrained integration under plain vertical gravity.
        vz += gz * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # 2. Collision detection & response against the surface at the new (x, y).
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
    terrain, ball_cfg, phys_cfg, sim_cfg, state, seed = args

    if seed is not None:
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)

    x0, y0 = float(state[0]), float(state[1])
    vx0 = float(state[2]) if len(state) > 2 else 0.0
    vy0 = float(state[3]) if len(state) > 3 else 0.0
    b_cfg = replace(ball_cfg, x0=x0, y0=y0, vx0=vx0, vy0=vy0)

    return run_simulation(terrain, b_cfg, phys_cfg, sim_cfg)


def run_ensemble_parallel(
    terrain: Any,
    ball_cfg: BallConfig,
    phys_cfg: PhysicsConfig,
    sim_cfg: SimConfig | None = None,
    ensemble_cfg: EnsembleConfig | None = None,
    initial_states: list[tuple[float, float, float, float]] | None = None,
    workers: int | None = None,
    show_progress: bool = False,
    desc: str | None = None,
) -> list[TrajectoryRecord]:
    """Runs an ensemble of balls in parallel.

    Initial (x0, y0, vx0, vy0) states come from `initial_states` if given,
    otherwise are jittered from `ensemble_cfg`, otherwise a single run at
    `ball_cfg`'s own initial state. Per-ball radius/mass/restitution/friction
    come from `ball_cfg`; only position/velocity are overridden per run.
    """
    if sim_cfg is None:
        sim_cfg = SimConfig()

    if initial_states is not None:
        states = list(initial_states)
    elif ensemble_cfg is not None:
        rng = np.random.default_rng(ensemble_cfg.seed)
        states = []
        for _ in range(ensemble_cfg.k_max):
            dx = rng.normal(0.0, ensemble_cfg.x_jitter_std)
            dy = rng.uniform(-ensemble_cfg.y_jitter_max, ensemble_cfg.y_jitter_max)
            states.append((ensemble_cfg.start_x + dx, ensemble_cfg.start_y + dy, 0.0, 0.0))
    else:
        states = [(ball_cfg.x0, ball_cfg.y0, ball_cfg.vx0, ball_cfg.vy0)]

    num_sims = len(states)
    if num_sims == 0:
        return []

    if workers is None:
        workers = int(getattr(sim_cfg, "num_workers", 4))
    workers = max(1, min(int(workers), num_sims))

    base_seed = getattr(sim_cfg, "seed", None)
    seeds = [base_seed + i for i in range(num_sims)] if base_seed is not None else [None] * num_sims

    task_args = [(terrain, ball_cfg, phys_cfg, sim_cfg, states[i], seeds[i]) for i in range(num_sims)]

    if workers == 1:
        results = [_single_run_worker(args) for args in task_args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_single_run_worker, task_args)
            if show_progress:
                from tqdm import tqdm

                mapped = tqdm(mapped, total=num_sims, desc=desc or "Simulating", leave=False)
            results = list(mapped)

    return results
