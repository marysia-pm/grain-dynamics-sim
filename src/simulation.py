"""Ball-on-rough-slope physics simulation.

This replaces the original notebook's two near-identical, deeply nested
``for k in range(k_max): for i, t_step in enumerate(time_points): ...`` loops
(one used for a "single simulation" demo, one duplicated almost verbatim for
plotting) with a single, tested integration routine.

Physics model
-------------
While the ball's centre is within ``radius`` of the terrain surface, gravity
is decomposed into a component normal to the surface (which is cancelled,
representing the normal contact force) and a tangential component (which
accelerates the ball along the slope). If the ball is found to have
penetrated the surface, its height is clamped back onto the surface and the
velocity component along the normal is reflected and scaled by
``restitution_coeff`` (0 = fully inelastic, 1 = perfectly elastic).

While the ball is above the surface it is in free fall under gravity alone.

Near the surface, each outer time step ``dt`` is sub-divided into
``denser_steps_number`` finer sub-steps to avoid tunnelling through the
terrain and to resolve the (potentially stiff) contact response accurately;
free-flight steps use a single, cheap Euler step of size ``dt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rough_slope_sim.config import BallConfig, EnsembleConfig, PhysicsConfig
from rough_slope_sim.terrain import Terrain

Vec3 = tuple[float, float, float]


@dataclass
class Trajectory:
    """Time history of a single ball's simulation run."""

    time: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]
    vz: NDArray[np.float64]
    reached_end: bool
    """True if the ball's x-position reached the far (downslope) edge of the
    terrain before running out of simulated time or leaving the y/x bounds."""

    final_position: Vec3
    """The last integrated (x, y, z) position, one step beyond the last
    sample in ``x``/``y``/``z`` (which is what ``reached_end`` is evaluated
    against). Used for end-of-run statistics."""


    def padded(self, length: int) -> "Trajectory":
        """Return a copy padded with NaNs up to ``length`` samples.

        Useful when aggregating trajectories of different lengths (ensemble
        runs stop early once a ball leaves the terrain bounds) into a single
        rectangular array for variance/statistics computations.
        """

        def pad(arr: NDArray[np.float64]) -> NDArray[np.float64]:
            if len(arr) >= length:
                return arr[:length]
            return np.concatenate([arr, np.full(length - len(arr), np.nan)])

        return Trajectory(
            time=pad(self.time),
            x=pad(self.x),
            y=pad(self.y),
            z=pad(self.z),
            vx=pad(self.vx),
            vy=pad(self.vy),
            vz=pad(self.vz),
            reached_end=self.reached_end,
            final_position=self.final_position,
        )


def _euler_substep(
    pos: Vec3,
    vel: Vec3,
    terrain: Terrain,
    mass: float,
    g: float,
    radius: float,
    restitution_coeff: float,
    sub_dt: float,
) -> tuple[Vec3, Vec3]:
    """Advance (pos, vel) by one Euler step of size ``sub_dt``."""
    x, y, z = pos
    vx, vy, vz = vel

    terrain_height = terrain.get_height(x, y)
    on_or_below_surface = z <= terrain_height + radius

    if on_or_below_surface:
        nx, ny, nz = terrain.get_normal_vector(x, y)
        n = np.array([nx, ny, nz])

        f_gravity = np.array([0.0, 0.0, -mass * g])
        f_normal = np.dot(f_gravity, n) * n
        f_tangential = f_gravity - f_normal
        ax, ay, az = f_tangential / mass

        if z < terrain_height + radius:
            # Penetrating the surface: clamp position and reflect the
            # velocity component along the normal (damped by restitution).
            z = terrain_height + radius
            v = np.array([vx, vy, vz])
            v_normal_component = np.dot(v, n)
            if v_normal_component < 0:
                correction = -(1 + restitution_coeff) * v_normal_component * n
                vx, vy, vz = (v + correction).tolist()
    else:
        ax, ay, az = 0.0, 0.0, -g

    vx += ax * sub_dt
    vy += ay * sub_dt
    vz += az * sub_dt

    x += vx * sub_dt
    y += vy * sub_dt
    z += vz * sub_dt

    return (x, y, z), (vx, vy, vz)


def _step(
    pos: Vec3,
    vel: Vec3,
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
) -> tuple[Vec3, Vec3]:
    """Advance (pos, vel) by one outer time step ``physics.dt``.

    Uses ``denser_steps_number`` fine sub-steps whenever the ball is already
    at or below the surface (where the contact response needs resolving
    accurately). When the ball starts the step in free flight, a single
    cheap step is tried first; if that coarse step would land the ball
    inside the surface (it fell far enough during ``dt`` to tunnel past the
    contact boundary), the same interval is redone with fine sub-stepping
    instead, so the recorded trajectory never contains a sample that
    penetrates the terrain by more than a negligible, sub-step-sized amount.
    """
    x, y, z = pos
    terrain_height = terrain.get_height(x, y)
    near_surface = z <= terrain_height + ball.radius

    if near_surface:
        return _substep_loop(pos, vel, terrain, ball, physics, physics.dt, physics.denser_steps_number)

    new_pos, new_vel = _euler_substep(
        pos, vel, terrain, ball.mass, physics.g, ball.radius,
        physics.restitution_coeff, physics.dt,
    )
    new_x, new_y, new_z = new_pos
    new_terrain_height = terrain.get_height(new_x, new_y)
    if new_z <= new_terrain_height + ball.radius:
        # The coarse free-flight step would tunnel past the surface; redo
        # this interval with fine sub-stepping so the collision is caught.
        return _substep_loop(pos, vel, terrain, ball, physics, physics.dt, physics.denser_steps_number)

    return new_pos, new_vel


def _substep_loop(
    pos: Vec3,
    vel: Vec3,
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    dt: float,
    n_substeps: int,
) -> tuple[Vec3, Vec3]:
    """Integrate (pos, vel) across an interval of length ``dt`` using
    ``n_substeps`` fine Euler sub-steps."""
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        pos, vel = _euler_substep(
            pos, vel, terrain, ball.mass, physics.g, ball.radius,
            physics.restitution_coeff, sub_dt,
        )
    return pos, vel


def simulate_single_ball(
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    start_x: float,
    start_y: float,
) -> Trajectory:
    """Simulate one ball starting at rest at (start_x, start_y).

    The simulation stops early if the ball leaves the terrain's x/y bounds,
    or after ``physics.total_time`` seconds elapse.
    """
    x_min, x_max = terrain.x_bounds
    y_min, y_max = terrain.y_bounds

    z0 = terrain.get_height(start_x, start_y) + ball.radius
    pos: Vec3 = (start_x, start_y, z0)
    vel: Vec3 = (0.0, 0.0, 0.0)

    time_points = np.arange(0, physics.total_time, physics.dt)

    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []
    used_time = []

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

    reached_end = pos[0] >= x_max

    return Trajectory(
        time=np.array(used_time),
        x=np.array(xs),
        y=np.array(ys),
        z=np.array(zs),
        vx=np.array(vxs),
        vy=np.array(vys),
        vz=np.array(vzs),
        reached_end=reached_end,
        final_position=pos,
    )


def run_ensemble(
    terrain: Terrain,
    ball: BallConfig,
    physics: PhysicsConfig,
    ensemble: EnsembleConfig,
) -> list[Trajectory]:
    """Run ``ensemble.k_max`` independent simulations from randomly jittered
    starting positions around (start_x, start_y)."""
    rng = np.random.default_rng(ensemble.seed)

    trajectories = []
    for _ in range(ensemble.k_max):
        dx = rng.normal(0.0, ensemble.x_jitter_std)
        dy = rng.uniform(0.0, ensemble.y_jitter_max)
        traj = simulate_single_ball(
            terrain, ball, physics,
            start_x=ensemble.start_x + dx,
            start_y=ensemble.start_y + dy,
        )
        trajectories.append(traj)

    return trajectories
