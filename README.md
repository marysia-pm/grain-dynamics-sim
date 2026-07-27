# grain-dynamics-sim

Physics simulation of a ball falling down a rough, inclined surface, with
tools to analyze how surface roughness affects the spread ("diffusion") of
the ball's trajectories, and to relate abrasive grit size to measured
surface roughness (Sa).


## What it does

1. **Terrain generation** (`rough_slope_sim.terrain`): builds an inclined
   plane with spatially-correlated random roughness. The roughness amplitude
   can switch from "rough" to "smooth" at a given y-coordinate, to model a
   rough traction zone at the top of a slope.
2. **Ball physics** (`rough_slope_sim.simulation`): integrates a ball's
   3D trajectory down the slope, decomposing gravity into tangential
   (surface-following) and normal (contact) components, with a
   restitution-based bounce response when the ball penetrates the surface.
3. **Analysis** (`rough_slope_sim.analysis`): ensemble statistics (variance
   of position over time, final-position distributions), a grit-size /
   Sa calibration curve fit, and a roughness sweep that tracks how a
   "diffusion coefficient" (variance of final position) changes with
   surface roughness.
4. **Plotting** (`rough_slope_sim.plotting`): recreates all of the original
   notebook's figures (3D terrain + normals, 3D trajectories, energy vs.
   time, final-position scatter/histograms, variance over time, grit
   calibration curves, roughness sweep) as reusable functions that return
   `matplotlib` figures.
5. **CLI** (`rough_slope_sim.cli`): runs the whole pipeline end-to-end and
   saves every figure as a PNG.

## What changed vs. the original notebook script

- All bare global variables (`ball_radius`, `roughness_transition_y`,
  `terrain_height_interpolator`, ...) became explicit dataclasses
  (`TerrainConfig`, `BallConfig`, `PhysicsConfig`, `EnsembleConfig`) and
  object attributes (`Terrain.get_height`, `Terrain.get_normal_vector`),
  passed around instead of mutated in place.
- The two nearly-identical, deeply nested simulation loops (one for a
  single demo ball, one duplicated for the ensemble/plotting run) were
  merged into a single, tested integration routine
  (`simulate_single_ball` / `run_ensemble`).
- The confusing "redo the last time step at higher resolution if we
  detect penetration" logic was cleaned up into an explicit, testable rule:
  take a single cheap step during free flight, but if that coarse step
  would tunnel the ball past the surface, redo the same interval with
  `denser_steps_number` fine sub-steps; while already in contact, always
  use fine sub-stepping so the contact response is resolved accurately.
- Fixed a bug in the roughness-sweep analysis where the code referenced an
  undefined variable, `valid_final_x_positions`, instead of the array it
  had actually just computed, `valid_final_y_positions` — this meant that
  section of the original notebook could never run to completion.
- Fixed `roughness_amplitude_smooth=0.0,` / `roughness_amplitude_rough=0.02,`
  accidentally being defined as 1-tuples (trailing commas) at module scope.
- Replaced blocking `plt.show()` calls throughout with functions that
  return `Figure` objects, so plots can be saved, tested headlessly, or
  displayed, as the caller prefers.
- Added type hints, docstrings, and a test suite.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and
environment management.

```bash
# Install uv if you don't have it yet:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install (creates a .venv automatically):
git clone <this-repo-url>
cd rough-slope-sim
uv sync
```

`uv sync` installs runtime dependencies (`numpy`, `scipy`, `matplotlib`) plus
the `dev` dependency group (`pytest`, `pytest-cov`, `ruff`) into a local
`.venv`.

## Usage

### Run the full pipeline via the CLI

```bash
uv run rough-slope-sim --out-dir output --seed 0
```

This generates a terrain, runs an ensemble of ball trajectories, fits the
grit-calibration curves, runs a roughness sweep, and writes every figure as
a PNG into `output/`. Pass `--show` to also open interactive windows.

```bash
uv run rough-slope-sim --help
```

### Use it as a library

```python
from rough_slope_sim import BallConfig, PhysicsConfig, TerrainConfig, generate_terrain
from rough_slope_sim.config import EnsembleConfig
from rough_slope_sim.simulation import run_ensemble
from rough_slope_sim.plotting import plot_trajectories_3d

terrain = generate_terrain(TerrainConfig(alpha=30, resolution=300, seed=0))
trajectories = run_ensemble(
    terrain,
    BallConfig(radius=0.5, mass=1.0),
    PhysicsConfig(total_time=10.0),
    EnsembleConfig(k_max=10, seed=0),
)

fig = plot_trajectories_3d(terrain, trajectories)
fig.savefig("trajectories.png")
```

See `examples/run_demo.py` for a complete runnable example.

## Development

```bash
# Run the test suite
uv run pytest

# With coverage
uv run pytest --cov=rough_slope_sim

# Lint
uv run ruff check .
```

## Project layout

```
rough-slope-sim/
├── pyproject.toml          # uv / package metadata & dependencies
├── src/
│   └── rough_slope_sim/
│       ├── config.py       # TerrainConfig, BallConfig, PhysicsConfig, EnsembleConfig
│       ├── terrain.py      # terrain generation + height/normal interpolation
│       ├── simulation.py   # ball physics integration
│       ├── analysis.py     # ensemble statistics, curve fits, roughness sweep
│       ├── plotting.py     # all figures, as functions returning Figure objects
│       └── cli.py          # `rough-slope-sim` command-line entry point
├── tests/                  # pytest test suite
├── examples/
│   └── run_demo.py         # minimal end-to-end example script
└── thesis_original/
    └── thesis.py           # original, unmodified Colab export, kept for reference
```

## Notes on the physical model

The ball is treated as a point mass with a fixed radius offset above the
terrain surface. While in contact (`z <= terrain_height + radius`), gravity
is split into a component normal to the local surface (balanced by the
contact force) and a tangential component that accelerates the ball along
the slope. If numerical integration causes the ball to dip below the
surface within a step, its height is clamped back to the surface and the
velocity component along the surface normal is reflected and scaled by
`restitution_coeff` (`0` = fully inelastic / no bounce, `1` = perfectly
elastic). Away from the surface, the ball is in free fall under gravity
alone; a coarse free-flight step that would tunnel past the surface within
a single `dt` is automatically redone with fine sub-stepping so the
collision is still resolved accurately.
