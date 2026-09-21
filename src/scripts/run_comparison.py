"""Complete runner script with calibrated grit amplitude scaling and X-axis track zero-alignment."""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from rough_slope_sim import (
    BallConfig,
    EnsembleConfig,
    PhysicsConfig,
    SimConfig,
    TerrainConfig,
    generate_terrain,
    run_ensemble_parallel,
)
from rough_slope_sim.analysis import (
    calculate_diffusion_coefficient,
    compare_distributions,
    count_on_side_over_time,
    estimate_diffusion_from_variance_slope,
    local_diffusion_coefficient_from_increments,
    trajectories_at_x_slice,
    variance_over_time_by_side,
)
from rough_slope_sim.plotting import (
    plot_ball_surface_closeup,
    plot_count_by_side_over_time,
    plot_terrain_3d,
    plot_trajectories_3d,
    plot_trajectories_and_three_slices,
    plot_variance_by_side_over_time,
)
from rough_slope_sim.terrain import nanovea_d50_from_grit

PX_TO_CM = 0.01123
X_SLICES = [5.0, 10.0, 15.0]
CLOSEUP_X, CLOSEUP_Y = 5.0, 15.0


@dataclass
class SurfaceConfig:
    folder_name: str
    grit_left: float
    grit_right: float
    start_y: float
    is_dual: bool


def grit_to_amplitude(grit: float) -> float:
    """Converts grit (P-value) to grain amplitude d50 (in cm) via Nanovea profilometer lookup."""
    return nanovea_d50_from_grit(grit)


def parse_folder(folder_name: str, y_mid: float = 11.5) -> SurfaceConfig:
    dual_match = re.search(r"(\d+)[\\/](\d+)P_(mid|tilted(\d*))", folder_name, re.IGNORECASE)
    if dual_match:
        g_l, g_r = float(dual_match.group(1)), float(dual_match.group(2))
        pos_type, tilt_str = dual_match.group(3).lower(), dual_match.group(4)
        if pos_type == "mid":
            start_y = y_mid
        else:
            tilt = int(tilt_str) if tilt_str else 1
            shift = -1.0 if g_l < g_r else 1.0
            start_y = y_mid + (shift * tilt * 1.0)
        return SurfaceConfig(folder_name, g_l, g_r, start_y, is_dual=(g_l != g_r))

    single_match = re.search(r"(\d+)P", folder_name, re.IGNORECASE)
    if single_match:
        g = float(single_match.group(1))
        return SurfaceConfig(folder_name, g, g, y_mid, is_dual=False)

    return SurfaceConfig(folder_name, 2000.0, 2000.0, y_mid, is_dual=False)


def _guess_scale(max_v: float) -> float:
    if max_v > 290.0:
        return PX_TO_CM
    if max_v > 29.0:
        return 0.1
    return 1.0


def load_trajectories(folder: Path, ramp_length_proj_cm: float, target_x0: float = 0.1) -> list[np.ndarray]:
    """Loads experimental trajectories, fixes orientation, and aligns release points to target_x0."""
    trajs = []
    column_renames = {"particle": "ball_id", "track_id": "ball_id", "pos_x": "x", "pos_y": "y"}

    for txt in glob.glob(os.path.join(folder, "*.txt")):
        try:
            df = pd.read_csv(txt, sep=r"\s+|,|;", engine="python")
            df.columns = [column_renames.get(c.lower().strip(), c.lower().strip()) for c in df.columns]

            if not all(c in df.columns for c in ["ball_id", "frame", "x", "y"]):
                continue

            max_v = max(df["x"].max(), df["y"].max())
            scale = _guess_scale(max_v)

            for _, group in df.groupby("ball_id"):
                group = group.sort_values("frame")
                t = group[["frame", "x", "y"]].to_numpy(dtype=np.float64)
                if len(t) > 0:
                    t[:, 1] *= scale
                    t[:, 2] *= scale
                    trajs.append(t)
        except Exception:
            continue

    if not trajs:
        return []

    start_x_mean = float(np.mean([t[0, 1] for t in trajs]))
    end_x_mean = float(np.mean([t[-1, 1] for t in trajs]))

    if start_x_mean > end_x_mean or start_x_mean > (ramp_length_proj_cm / 2.0):
        for t in trajs:
            t[:, 1] = ramp_length_proj_cm - t[:, 1]

    # Align experimental tracks along X so every trajectory release point starts at target_x0
    for t in trajs:
        x_start = t[0, 1]
        t[:, 1] = t[:, 1] - x_start + target_x0

    return trajs


def sample_hybrid_y0(
    center_y: float,
    std_y: float,
    num_samples: int,
    seed: int = 42,
    blend_factor: float = 0.05,
) -> np.ndarray:
    """
    Samples initial Y positions anchored to the entrainment center (mech_center).
    """
    rng = np.random.default_rng(seed)

    # Gaussian distribution centered ON the entrainment center
    g_samples = rng.normal(loc=center_y, scale=std_y, size=num_samples)

    # Small uniform spread across physical launcher boundaries (+/- 3 std)
    u_min = center_y - 3.0 * std_y
    u_max = center_y + 3.0 * std_y
    u_samples = rng.uniform(u_min, u_max, size=num_samples)

    return blend_factor * u_samples + (1.0 - blend_factor) * g_samples


def load_alignment_params(
    folder_or_file: str | Path, default_seam: float = 11.5, default_mech: float = 11.5
) -> dict[str, float]:
    """Reads experimental alignment metadata (mech_center_cm, seam_cm) from text files."""
    params = {"seam_cm": default_seam, "mech_center_cm": default_mech, "has_seam": True}
    if not folder_or_file:
        return params

    path = Path(folder_or_file)

    # 1. Automatically redirect 'processing_results' to 'annotated_alignment_results'
    if "processing_results" in path.parts:
        parts = list(path.parts)
        idx = parts.index("processing_results")
        parts[idx] = "annotated_alignment_results"
        target_path = Path(*parts)
    else:
        target_path = path

    # 2. Search directory for matching annotation txt files
    candidate_files: list[Path] = []
    if target_path.is_dir():
        candidates = set(target_path.glob("annotated_*.txt")) | set(target_path.glob("*sample_frame.txt"))
        candidate_files = sorted(list(candidates))
    elif target_path.exists():
        candidate_files = [target_path]

    if not candidate_files:
        print(f"[ALIGN LOG] ⚠️ Metadata file not found in '{target_path}'. Using nominal defaults.")
        return params

    # 3. Read alignment parameters
    for txt_path in candidate_files:
        try:
            df = pd.read_csv(txt_path, sep=r"\s+|\t|,", engine="python")

            if "has_seam" in df.columns:
                params["has_seam"] = str(df["has_seam"].iloc[0]).strip().lower() in ["true", "1"]

            if params["has_seam"] and "seam_cm" in df.columns and pd.notna(df["seam_cm"].iloc[0]):
                params["seam_cm"] = float(df["seam_cm"].iloc[0])

            if "mech_center_cm" in df.columns and pd.notna(df["mech_center_cm"].iloc[0]):
                params["mech_center_cm"] = float(df["mech_center_cm"].iloc[0])

            print(
                f"[ALIGN LOG] ✅ Loaded alignment from '{txt_path.name}': "
                f"Release Y = {params['mech_center_cm']:.3f} cm | Seam Y = {params['seam_cm']:.3f} cm"
            )
            return params
        except Exception as e:
            print(f"[ALIGN LOG] ❌ Error parsing '{txt_path.name}': {e}. Using defaults.")

    return params


def process_folder(
    exp_folder: Path,
    out_root: Path,
    slope_angle_deg: float,
    ramp_length_cm: float,
    x_max_proj: float,
) -> dict | None:
    folder_name = exp_folder.name
    cfg = parse_folder(folder_name)

    # 1. Read measured seam position from metadata annotation file (defaults to 11.5 cm)
    params = load_alignment_params(exp_folder)
    has_seam = params["has_seam"]
    seam_y = params["seam_cm"]
    mech_center = params["mech_center_cm"]
    print(has_seam, seam_y, mech_center)

    surface_info = (
        f"Dual Grit ({cfg.grit_left:.0f}P / {cfg.grit_right:.0f}P)"
        if cfg.is_dual
        else f"Single Grit ({cfg.grit_left:.0f}P)"
    )

    tqdm.write(f"\n[+] Processing Folder : {folder_name}")
    tqdm.write(f"  ├── Surface Config : {surface_info}")
    if has_seam:
        tqdm.write(f"  ├── Seam Position  : Y = {seam_y:.3f} cm")
    tqdm.write(f"  ├── Entrainment Center  : Y = {mech_center:.3f} cm")

    sub_out = out_root / folder_name.replace("\\", "_").replace("/", "_")
    sub_out.mkdir(parents=True, exist_ok=True)

    exp_trajs = [
        t for t in load_trajectories(exp_folder, ramp_length_proj_cm=x_max_proj, target_x0=0.1) if len(t) > 0
    ]
    if not exp_trajs:
        tqdm.write("  └── [!] No valid trajectories found. Skipping.")
        return None

    exp_y_starts = np.array([t[0, 2] for t in exp_trajs if len(t) > 0])
    exp_mean_y = float(np.mean(exp_y_starts))
    std_y = float(np.std(exp_y_starts)) if len(exp_y_starts) > 1 else 0.10

    # Sanity check: If mech_center deviates by > 0.5 cm from actual tracked tracks,
    # use empirical experimental center to prevent metadata misdetection issues.
    if abs(mech_center - exp_mean_y) > 0.5:
        tqdm.write(
            f"  ├── [!] WARNING: mech_center ({mech_center:.3f} cm) disagrees with "
            f"exp tracks mean ({exp_mean_y:.3f} cm). Using exp mean for sim center."
        )
        sim_center_y = exp_mean_y
    else:
        sim_center_y = mech_center

    num_sim_balls = max(len(exp_trajs), 150)
    sim_y0_vals = sample_hybrid_y0(
        center_y=sim_center_y,
        std_y=std_y,
        num_samples=num_sim_balls,
        blend_factor=0.05,
    )
    sampled_initial_states = [(0.1, float(y0), 0.0, 0.0) for y0 in sim_y0_vals]
    tqdm.write(f"  ├── Loaded Tracks  : {len(exp_trajs)} trajectories (X release aligned to 0.1 cm)")
    tqdm.write(f"  ├── Initial Config : Fixed X0 = 0.1 cm, V0 = 0.0")

    # 2. Construct TerrainConfig with dynamic seam boundary
    t_cfg = TerrainConfig(
        ramp_length=ramp_length_cm,
        slope_angle=slope_angle_deg,
        roughness_amplitude_rough=grit_to_amplitude(cfg.grit_left),
        roughness_amplitude_smooth=grit_to_amplitude(cfg.grit_right),
        grit_rough=cfg.grit_left,
        grit_smooth=cfg.grit_right,
        roughness_transition_y=seam_y,
        seed=42,
    )
    terrain = generate_terrain(t_cfg)

    tqdm.write(f"  ├── Rough side     : {terrain.p_value_rough:.0f}P")
    tqdm.write(f"  ├── Smooth side    : {terrain.p_value_smooth:.0f}P")
    tqdm.write(f"  ├── Overall surface: {terrain.p_value_mean:.0f}P")

    ball_cfg = BallConfig(radius=0.125, x0=0.1)
    physics_cfg = PhysicsConfig(gravity=981.0)
    e_cfg = EnsembleConfig(k_max=num_sim_balls, seed=42)

    # 3. Explicitly pass sim_cfg and ensemble_cfg as keyword arguments
    sim_trajs = run_ensemble_parallel(
        terrain,
        ball_cfg,
        physics_cfg,
        sim_cfg=SimConfig(),
        ensemble_cfg=e_cfg,
        initial_states=sampled_initial_states,
        show_progress=True,
        desc=f"  ├── Simulating ({num_sim_balls} balls)",
    )
    sim_trajs = [t for t in sim_trajs if len(t.x) > 0]

    tqdm.write("  └── Generating plots...")

    fig_closeup = plot_ball_surface_closeup(
        terrain=terrain,
        ball_radius=ball_cfg.radius,
        ball_x=CLOSEUP_X,
        ball_y=CLOSEUP_Y,
        window_factor=3.0,
    )
    fig_closeup.savefig(sub_out / "00_ball_surface_closeup.png", dpi=300, bbox_inches="tight")
    plt.close(fig_closeup)

    fig_terrain = plot_terrain_3d(terrain, quiver_skip=28)
    fig_terrain.savefig(sub_out / "01_terrain_3d_normals.png", dpi=150, bbox_inches="tight")
    plt.close(fig_terrain)

    fig_traj3d = plot_trajectories_3d(terrain, sim_trajs)
    fig_traj3d.savefig(sub_out / "02_terrain_trajectories_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig_traj3d)

    y_exp_15 = trajectories_at_x_slice(exp_trajs, 15.0)
    y_sim_15 = trajectories_at_x_slice(sim_trajs, 15.0)

    # 4. Render trajectory slice plot using the exact interface y-location
    fig_slices = plot_trajectories_and_three_slices(
        exp_trajs,
        sim_trajs,
        X_SLICES,
        is_dual=cfg.is_dual,
        interface_y=seam_y,
        terrain=terrain,
        ball_radius=ball_cfg.radius,
    )
    fig_slices.savefig(sub_out / "04_trajectories_and_3slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig_slices)

    d_exp = calculate_diffusion_coefficient(y_exp_15, x_slice=15.0)
    d_sim = calculate_diffusion_coefficient(y_sim_15, x_slice=15.0)
    metrics = compare_distributions(y_exp_15, y_sim_15)

    d_rough_apparent = np.nan
    d_smooth_apparent = np.nan
    d_rough_local = np.nan
    d_smooth_local = np.nan
    if cfg.is_dual:
        # These plots/metrics only make sense for a dual-grit surface with an
        # actual rough/smooth interface. Flux plot: does one side's ball count
        # fall while the other's rises (net migration across the interface)?
        # Experimental trajectories are included too, but on their own panel
        # with a frame-based x-axis, since their time-like column is a raw
        # frame index (no capture frame rate is known anywhere in this
        # codebase, so it can't be honestly converted to seconds).
        t_axis, n_rough, n_smooth, n_active = count_on_side_over_time(sim_trajs, interface_y=seam_y)
        exp_flux_kwargs = {}
        if len(exp_trajs) > 0:
            t_axis_e, n_rough_e, n_smooth_e, n_active_e = count_on_side_over_time(
                exp_trajs, interface_y=seam_y
            )
            exp_flux_kwargs = dict(
                exp_time_axis=t_axis_e,
                exp_n_rough=n_rough_e,
                exp_n_smooth=n_smooth_e,
                exp_n_active=n_active_e,
            )
        fig_flux = plot_count_by_side_over_time(
            t_axis, n_rough, n_smooth, n_active, interface_y=seam_y, **exp_flux_kwargs
        )
        fig_flux.savefig(sub_out / "05_flux_by_side.png", dpi=150, bbox_inches="tight")
        plt.close(fig_flux)

        # local_diffusion_coefficient_from_increments is the primary, accurate
        # metric (Kramers-Moyal second moment, conditioned on each increment's
        # own starting position -- confirmed <1% error on synthetic ground
        # truth at lag_steps=1). Use this number for actual comparisons.
        local_D = local_diffusion_coefficient_from_increments(sim_trajs, interface_y=seam_y, lag_steps=1)
        d_rough_local = local_D["D_rough"]
        d_smooth_local = local_D["D_smooth"]

        # variance_over_time_by_side + estimate_diffusion_from_variance_slope
        # give an "apparent" D from long-time population variance -- kept here
        # only for the visual plot (does one side's spread visibly grow
        # faster), NOT as a quantitative value: it's biased low on both sides
        # (see that function's docstring) because the current-side population
        # mixes different crossing histories, unlike the per-increment method
        # above which conditions on each increment's own starting position.
        t_axis2, var_r, var_s, n_r2, n_s2 = variance_over_time_by_side(sim_trajs, interface_y=seam_y)
        d_rough_apparent = estimate_diffusion_from_variance_slope(t_axis2, var_r)
        d_smooth_apparent = estimate_diffusion_from_variance_slope(t_axis2, var_s)
        exp_var_kwargs = {}
        if len(exp_trajs) > 0:
            t_axis2_e, var_r_e, var_s_e, n_r2_e, n_s2_e = variance_over_time_by_side(
                exp_trajs, interface_y=seam_y
            )
            exp_var_kwargs = dict(exp_time_axis=t_axis2_e, exp_var_rough=var_r_e, exp_var_smooth=var_s_e)
        fig_var_side = plot_variance_by_side_over_time(
            t_axis2, var_r, var_s, d_rough_apparent, d_smooth_apparent, **exp_var_kwargs
        )
        fig_var_side.savefig(sub_out / "06_variance_by_side.png", dpi=150, bbox_inches="tight")
        plt.close(fig_var_side)

        tqdm.write(
            f"  ├── D_rough={d_rough_local:.4f}, D_smooth={d_smooth_local:.4f} cm\u00b2/s "
            f"(Kramers-Moyal, lag_steps=1)"
        )

    return {
        "folder": folder_name,
        "is_dual": cfg.is_dual,
        "seam_y_cm": seam_y,
        "slope_angle_deg": slope_angle_deg,
        "x_max_proj_cm": x_max_proj,
        "D_exp_cm2_s": d_exp,
        "D_sim_cm2_s": d_sim,
        "D_rough_cm2_s": d_rough_local,
        "D_smooth_cm2_s": d_smooth_local,
        "D_rough_apparent_cm2_s": d_rough_apparent,
        "D_smooth_apparent_cm2_s": d_smooth_apparent,
        "wasserstein_cm": metrics["wasserstein_distance"],
        "ks_stat": metrics["ks_statistic"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-compare experiment vs. simulation for each surface.")
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("../grain-dynamics-analysis/processing_results"),
        help="Directory containing one subfolder per experiment.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--slope-angle", type=float, default=30.0, help="Incline angle in degrees")
    parser.add_argument("--ramp-length", type=float, default=29.0, help="Physical ramp length (cm)")
    args = parser.parse_args()

    t_cfg_template = TerrainConfig(ramp_length=args.ramp_length, slope_angle=args.slope_angle)
    x_max_proj = t_cfg_template.length_x

    if not args.exp_root.exists():
        print(f"[X] Error: Directory '{args.exp_root}' not found.")
        return

    exp_folders = [f for f in sorted(args.exp_root.iterdir()) if f.is_dir()]
    print("\n==================================================")
    print(f" Incline Angle: {args.slope_angle}° | Projected Length X_max = {x_max_proj:.2f} cm")
    print(f" Processing {len(exp_folders)} experiment folders.")
    print("==================================================\n")

    summary_records = []
    for exp_folder in tqdm(exp_folders, desc="Processing Experiments"):
        record = process_folder(
            exp_folder,
            out_root=args.out_dir,
            slope_angle_deg=args.slope_angle,
            ramp_length_cm=args.ramp_length,
            x_max_proj=x_max_proj,
        )
        if record is not None:
            summary_records.append(record)

    if summary_records:
        out_csv = args.out_dir / "batch_summary_metrics.csv"
        pd.DataFrame(summary_records).to_csv(out_csv, index=False)
        print(f"\n[✓] Complete! Results exported to: {out_csv}\n")


if __name__ == "__main__":
    main()
