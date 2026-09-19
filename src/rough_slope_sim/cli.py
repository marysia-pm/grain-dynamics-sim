"""Command-line interface for generating and exporting rough slope terrains."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure root package import directory ('src') is in sys.path
package_root = Path(__file__).resolve().parents[1]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from rough_slope_sim.plotting import (
    plot_ball_surface_closeup,
    plot_calibrated_sandpaper_panel,
    plot_terrain_3d,
)
from rough_slope_sim.terrain import (
    Terrain,
    generate_calibrated_sandpaper,
    generate_galton_board,
)


def save_all_terrain_plots(terrain: Terrain, output_dir: Path, surface_type: str) -> None:
    """Generates and exports all available surface visualization plots to the output directory."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 3D Surface with Normal Vectors
    try:
        fig_3d = plot_terrain_3d(terrain)
        path_3d = output_dir / "terrain_3d_normals.png"
        fig_3d.savefig(path_3d, dpi=300, bbox_inches="tight")
        plt.close(fig_3d)
        print(f"Saved plot image to '{path_3d}'")
    except Exception as err:
        print(f"Could not generate 3D terrain plot: {err}")

    # 2. Contact Close-up Map
    try:
        center_x = (terrain.x_bounds[0] + terrain.x_bounds[1]) / 2.0
        center_y = (terrain.y_bounds[0] + terrain.y_bounds[1]) / 2.0
        fig_closeup = plot_ball_surface_closeup(
            terrain=terrain,
            ball_radius=0.125,
            ball_x=center_x,
            ball_y=center_y,
        )
        path_closeup = output_dir / "ball_surface_closeup.png"
        fig_closeup.savefig(path_closeup, dpi=300, bbox_inches="tight")
        plt.close(fig_closeup)
        print(f"Saved plot image to '{path_closeup}'")
    except Exception as err:
        print(f"Could not generate ball close-up plot: {err}")

    # 3. Calibrated Sandpaper Multi-panel Comparison (Sandpaper surfaces only)
    if surface_type == "sandpaper":
        try:
            fig_panel = plot_calibrated_sandpaper_panel()
            path_panel = output_dir / "sandpaper_benchmark_panel.png"
            fig_panel.savefig(path_panel, dpi=300, bbox_inches="tight")
            plt.close(fig_panel)
            print(f"Saved plot image to '{path_panel}'")
        except Exception as err:
            print(f"Could not generate sandpaper panel plot: {err}")

    plt.close("all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rough Slope Simulation - Terrain Generation CLI")

    # Surface options
    parser.add_argument("--surface", type=str, choices=["sandpaper", "galton"], default="sandpaper")
    parser.add_argument("--ramp-length", type=float, default=29.0)
    parser.add_argument("--length-y", type=float, default=23.0)
    parser.add_argument("--slope-angle", type=float, default=30.0)
    parser.add_argument("--resolution", type=int, default=2500, help="Grid points per axis (both X and Y)")
    parser.add_argument(
        "--resolution-x", type=int, default=None, help="Grid points along X; overrides --resolution for X"
    )
    parser.add_argument(
        "--resolution-y", type=int, default=None, help="Grid points along Y; overrides --resolution for Y"
    )
    parser.add_argument("--seed", type=int, default=42)

    # Sandpaper options
    parser.add_argument("--grit-rough", type=float, default=80.0)
    parser.add_argument("--grit-smooth", type=float, default=None)
    parser.add_argument("--tile-size-cm", type=float, default=0.20)

    # Galton options
    parser.add_argument("--peg-radius", type=float, default=0.30)
    parser.add_argument("--peg-height", type=float, default=0.20)
    parser.add_argument("--peg-dx", type=float, default=2.0)
    parser.add_argument("--peg-dy", type=float, default=1.5)
    parser.add_argument("--peg-staggered", action="store_true", default=True)
    parser.add_argument("--no-peg-staggered", action="store_false", dest="peg_staggered")
    parser.add_argument("--peg-x-start", type=float, default=5.0)
    parser.add_argument("--peg-x-end", type=float, default=None)
    parser.add_argument(
        "--peg-shape", type=str, choices=["paraboloid", "gaussian", "cylinder"], default="cylinder"
    )

    # Output options
    parser.add_argument("-o", "--output", type=str, default=None, help="Save array data (.npz)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to save generated terrain plot images (e.g. output/galton)",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.surface == "galton":
        terrain = generate_galton_board(
            ramp_length=args.ramp_length,
            length_y=args.length_y,
            slope_angle=args.slope_angle,
            peg_radius=args.peg_radius,
            peg_height=args.peg_height,
            peg_dx=args.peg_dx,
            peg_dy=args.peg_dy,
            peg_staggered=args.peg_staggered,
            peg_x_start=args.peg_x_start,
            peg_x_end=args.peg_x_end,
            peg_shape=args.peg_shape,
            resolution=args.resolution,
            resolution_x=args.resolution_x,
            resolution_y=args.resolution_y,
        )
    else:
        terrain = generate_calibrated_sandpaper(
            grit_rough=args.grit_rough,
            grit_smooth=args.grit_smooth,
            slope_angle=args.slope_angle,
            ramp_length=args.ramp_length,
            length_y=args.length_y,
            resolution=args.resolution,
            resolution_x=args.resolution_x,
            resolution_y=args.resolution_y,
            tile_size_cm=args.tile_size_cm,
            seed=args.seed,
        )

    print(f"Generated {args.surface} grid: {terrain.height_map.shape}")

    if args.output:
        np.savez_compressed(
            args.output, height_map=terrain.height_map, x_grid=terrain.x_grid, y_grid=terrain.y_grid
        )
        print(f"Saved binary mesh to '{args.output}'")

    if args.output_dir:
        save_all_terrain_plots(terrain, output_dir=args.output_dir, surface_type=args.surface)


if __name__ == "__main__":
    main()
