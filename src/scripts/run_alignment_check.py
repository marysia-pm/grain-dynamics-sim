from pathlib import Path

import cv2
import matplotlib

# Use non-interactive backend for headless/batch processing
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze_and_visualize_alignment(
    image_path: Path,
    board_width_cm: float = 23.0,
    output_path: Path | None = None,
):
    """Detects alignment lines on incline frames and saves a labeled overlay image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    px_per_cm = w / board_width_cm

    # 1. Geometric Frame Center
    frame_center_x = w / 2.0
    frame_center_cm = frame_center_x / px_per_cm

    # 2. Detect Surface Seam Line
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid_strip = gray[int(h * 0.2) : int(h * 0.8), :]
    grad_x = np.abs(cv2.Sobel(mid_strip, cv2.CV_64F, 1, 0, ksize=3))
    profile = grad_x.mean(axis=0)

    search_range = range(int(w * 0.35), int(w * 0.65))
    seam_x = search_range[np.argmax(profile[search_range])]
    seam_cm = seam_x / px_per_cm

    # 3. Detect White Entrainer Center
    top_strip = gray[0 : int(h * 0.03), :]
    bright_cols = np.where(top_strip.max(axis=0) > 120)[0]
    bright_cols_center = [c for c in bright_cols if c > int(w * 0.2)]

    if not bright_cols_center:
        raise ValueError("Could not detect white entrainer mechanism in top section.")

    white_left = bright_cols_center[0]
    white_right = bright_cols_center[-1]
    mech_center_x = (white_left + white_right) / 2.0
    mech_center_cm = mech_center_x / px_per_cm

    # Offsets
    offset_seam_mm = (seam_cm - frame_center_cm) * 10.0
    offset_mech_seam_mm = (mech_center_cm - seam_cm) * 10.0

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
    ax.imshow(img_rgb)

    # Reference lines
    ax.axvline(
        x=frame_center_x,
        color="cyan",
        linestyle="--",
        linewidth=1.8,
        label=f"Frame Center: {frame_center_cm:.2f} cm ({frame_center_x:.0f} px)",
    )
    ax.axvline(
        x=seam_x,
        color="red",
        linestyle="-",
        linewidth=1.8,
        label=f"Surface Seam: {seam_cm:.2f} cm ({seam_x:.0f} px)",
    )
    ax.axvline(
        x=mech_center_x,
        color="lime",
        linestyle=":",
        linewidth=2.0,
        label=f"Entrainer Center: {mech_center_cm:.2f} cm ({mech_center_x:.0f} px)",
    )

    # Highlight entrainer width at top
    ax.plot(
        [white_left, white_right],
        [15, 15],
        color="lime",
        linewidth=4,
        solid_capstyle="round",
    )

    # Offset connection arrow when shifted
    if abs(offset_mech_seam_mm) > 1.0:
        y_arrow = int(h * 0.12)
        ax.annotate(
            "",
            xy=(seam_x, y_arrow),
            xytext=(mech_center_x, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="yellow", lw=2, mutation_scale=15),
        )
        ax.text(
            (seam_x + mech_center_x) / 2.0,
            y_arrow - 15,
            f"Offset: {offset_mech_seam_mm:+.1f} mm",
            color="yellow",
            fontsize=15,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.7),
        )

    # Info box
    parent_folder = image_path.parent.name
    info_text = (
        f"Folder: {parent_folder}\n"
        f"Board Scale: {px_per_cm:.2f} px/cm\n"
        f"Seam Shift from Frame Center: {offset_seam_mm:+.1f} mm\n"
        f"Entrainer Offset from Seam: {offset_mech_seam_mm:+.1f} mm"
    )

    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8, edgecolor="white"),
        color="white",
    )

    ax.set_title(f"Alignment Analysis: {image_path.name}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)  # Prevent RAM buildup during batch runs


def process_dataset_directory(
    root_dir: str | Path,
    output_dir_name: str = "annotated_alignment_results",
):
    """Recursively processes all .jpg images across subdirectories in root_dir."""
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        print(f"Error: Directory does not exist: {root_path}")
        return

    output_base = root_path.parent / output_dir_name

    # Recursively collect all .jpg and .jpeg files
    jpg_files = sorted([p for p in root_path.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg"]])

    print(f"Found {len(jpg_files)} JPEG images in {root_path.name} subdirectories.\n")

    success_count = 0
    for idx, img_path in enumerate(jpg_files, 1):
        # Mirror relative folder structure in the output directory
        relative_subpath = img_path.relative_to(root_path)
        save_path = output_base / relative_subpath.parent / f"annotated_{img_path.stem}.png"

        try:
            analyze_and_visualize_alignment(image_path=img_path, output_path=save_path)
            print(f"[{idx}/{len(jpg_files)}] Processed: {relative_subpath}")
            success_count += 1
        except Exception as e:
            print(f"[{idx}/{len(jpg_files)}] FAILED: {relative_subpath} | {e}")

    print(f"\nBatch complete: Successfully processed {success_count}/{len(jpg_files)} images.")
    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    target_directory = "../grain-dynamics-analysis/processing_results"
    process_dataset_directory(target_directory)
