# rough_slope_sim

Physics simulation of a ball rolling/bouncing down a rough inclined plane, used to
compare against experimental grain-dynamics footage. Two surface types are
supported: calibrated **sandpaper** (procedurally generated grains, calibrated
against real Nanovea profilometry measurements) and a **Galton board** (regular
peg lattice), on either a short benchtop ramp or a long, narrow track.

## How it works

- **Coordinate convention:** X-Y is the horizontal ground plane, Z is vertical
  (up). Gravity always points straight down; the incline itself is entirely
  encoded in the terrain's height field, so the ball only "feels" the slope
  through the local surface gradient/normal.
- **Terrain:** a baseline inclined plane with a procedural micro-roughness
  (grains or pegs) added on top, sampled onto a grid and bilinearly
  interpolated for height/gradient/normal queries.
- **Contact model:** instantaneous collision response with restitution, not a
  penalty spring. When the ball touches the surface, it's snapped onto it
  along the local normal and the normal velocity component is reflected and
  scaled by the ball's `restitution`. Coulomb friction (off by default)
  decelerates the tangential velocity while in contact.
- **Ensembles:** many balls can be simulated in parallel (`ProcessPoolExecutor`)
  with jittered starting positions, then compared statistically (Wasserstein
  distance, KS test, variance/diffusion coefficient) against experimental
  trajectories extracted from tracking data.

## Installation

```bash
pip install numpy scipy matplotlib pandas tqdm opencv-python
```

`torch` is an optional dependency (used only for reproducible seeding of the
worker processes) — the package works fine without it.

Requires Python 3.10+ (uses `X | Y` union type hints throughout).

## Project structure

```
rough_slope_sim/            # the package
├── __init__.py             # public API re-exports
├── config.py                # dataclasses: TerrainConfig, BallConfig, PhysicsConfig, SimConfig, EnsembleConfig
├── terrain.py                # terrain generation, Nanovea grit/Sa calibration, Terrain class
├── simulation.py             # physics integration + parallel ensemble runner
├── analysis.py                # statistical comparison, spatial slicing, diffusion coefficient
├── plotting.py                # all figure-generation routines
└── cli.py                     # `python -m rough_slope_sim.cli` terrain-generation CLI

# top-level scripts (outside the package)
├── create_galton.py           # run a Galton-board ensemble, export 5 standard plots
├── run_comparison.py          # batch-compare experiment vs. simulation across surface folders
├── run_long_sim.py            # large (600x30 cm) dual-grit plane simulation
├── compare_initial_velocity.py# compare zero vs. non-zero impact velocity ensembles
├── run_angle_analysis.py      # measure incline angle from a photo
├── run_alignment_check.py     # detect rig alignment offsets from a photo
└── exploring_surfaces.py      # exploratory grain-rendering script (superseded — see note below)
```

## Quickstart

### CLI: generate and export a terrain

```bash
# Sandpaper, default 1000x1000 grid
python -m rough_slope_sim.cli --surface sandpaper --grit-rough 80 --output-dir output/sandpaper

# Long/narrow domain: independent X/Y resolution
python -m rough_slope_sim.cli --surface sandpaper \
    --ramp-length 600 --length-y 30 \
    --resolution-x 2400 --resolution-y 150 \
    --output-dir output/long_plane

# Galton board
python -m rough_slope_sim.cli --surface galton --peg-shape cone --output-dir output/galton
```

### Python API: run an ensemble

```python
from rough_slope_sim import (
    TerrainConfig, BallConfig, PhysicsConfig, SimConfig, EnsembleConfig,
    generate_terrain, run_ensemble_parallel, trajectories_at_x_slice,
    calculate_diffusion_coefficient,
)

terrain = generate_terrain(TerrainConfig(
    ramp_length=29.0, slope_angle=30.0, length_y=23.0,
    grit_rough=80.0, grit_smooth=120.0, roughness_transition_y=11.5,
))

trajectories = run_ensemble_parallel(
    terrain,
    BallConfig(radius=0.125, restitution=0.8),
    PhysicsConfig(gravity=981.0),
    sim_cfg=SimConfig(dt=5e-4, t_max=2.0),
    ensemble_cfg=EnsembleConfig(k_max=150, start_x=0.1, start_y=11.5),
    show_progress=True,
)

y_at_15cm = trajectories_at_x_slice(trajectories, x_slice=15.0)
print("Lateral diffusion coefficient:", calculate_diffusion_coefficient(y_at_15cm, x_slice=15.0))
```

### Comparing against experimental tracking data

```bash
python run_comparison.py --exp-root ../grain-dynamics-analysis/processing_results --out-dir output
```

Each experiment subfolder is matched to a simulated ensemble; outputs are five
plots per folder (contact close-up, 3D terrain, 3D trajectories, histogram,
trajectory+slices) plus a `batch_summary_metrics.csv` with Wasserstein
distance, KS statistic, and diffusion coefficients for every surface.

## Configuration reference

All simulation parameters live in `config.py` as dataclasses:

| Class | Purpose |
|---|---|
| `TerrainConfig` | geometry (`ramp_length`, `slope_angle`, `length_y`), grid `resolution_x`/`resolution_y`, roughness (`grit_rough`/`grit_smooth` or explicit amplitudes), Galton peg parameters |
| `BallConfig` | starting position/velocity, `radius`, `mass`, `restitution`, `friction_mu` |
| `PhysicsConfig` | `gravity` |
| `SimConfig` | `dt`, `t_max`, `save_interval`, `num_workers` |
| `EnsembleConfig` | ensemble size and jitter for randomized starting positions |

`TerrainConfig.z_offset` (default: auto-scaled to the ramp's own drop height)
controls the baseline height of the plane before roughness is added — you
normally don't need to touch it, but it's there if you do.

## Known limitations / recent fixes

- **Long-ramp terrain used to go flat.** The baseline plane's height offset
  was a fixed constant that only worked by coincidence for ~29 cm ramps; on
  longer ramps (e.g. the 600 cm plane in `run_long_sim.py`) the terrain was
  silently clamped flat for most of its length. `z_offset` now auto-scales
  with ramp length.
- **`TerrainConfig.resolution_x`/`resolution_y` used to be ignored** — the
  terrain always used a hardcoded 1000×1000 grid regardless of what you set.
  This is now fixed; independent X/Y resolution is respected everywhere
  (`TerrainConfig`, the CLI, and the `generate_calibrated_sandpaper`/
  `generate_galton_board` convenience functions).
- **Sandpaper Sa calibration** now uses real measured Nanovea Sa-vs-diameter
  data instead of a flat `0.33 * d50` heuristic. Note the two Nanovea tables
  in the codebase (the 11-grit d50 table and the 5-point Sa reference table)
  don't fully agree with each other for the same nominal grit numbers — worth
  reconciling against the full profilometry sheet if exact absolute values
  matter for your comparison.
- **`exploring_surfaces.py` is superseded.** Its grain-rendering scheme
  (spacing, jitter, aspect ratio, real Sa calibration) has been folded into
  `terrain.py`'s main grain generator. The script still works standalone but
  duplicates a function name (`generate_calibrated_sandpaper`) with a
  different signature than the package version — safe to archive/delete once
  you've confirmed you don't need it as a reference.
- Contact restitution below 5 cm/s of relative normal velocity is treated as
  fully inelastic (anti-jitter smoothing) regardless of the configured
  `restitution` — worth knowing if you're tuning bounce behavior for
  slow-rolling contacts.
