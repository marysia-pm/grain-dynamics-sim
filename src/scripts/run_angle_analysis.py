from pathlib import Path
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def measure_and_draw_incline_angle(
    image_path: str | Path,
    output_path: str | Path | None = "measured_incline_result.png",
    show_plot: bool = True,
):
    """Measures the angle between an inclined surface and the ground for a single image."""
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 1. Edge Detection restricted to the incline region (left 65% of image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[:, : int(w * 0.65)] = 255

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.bitwise_and(edges, mask)

    # 2. Line Detection via Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=15)

    incline_pts_x, incline_pts_y = [], []
    if lines is not None:
        for line in lines:
            # Flatten to 1D array to guarantee x1, y1, x2, y2 unpacking works
            x1, y1, x2, y2 = line.ravel()
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue

            # In image coordinates, dy < 0 for an ascending line to the right
            angle_deg = np.degrees(np.arctan2(-dy, dx))

            # Filter for expected incline angles (15° to 55°)
            if 15.0 <= angle_deg <= 55.0:
                incline_pts_x.extend([x1, x2])
                incline_pts_y.extend([y1, y2])

    if len(incline_pts_x) < 4:
        raise ValueError("Could not reliably detect the inclined surface line.")

    # 3. Linear Regression Fit (y = m * x + b)
    m, b = np.polyfit(incline_pts_x, incline_pts_y, 1)

    # Incline angle relative to horizontal ground
    alpha_deg = np.degrees(np.arctan(-m))

    # Define key geometry points
    x_apex = int(w * 0.50)  # Hinge/pivot location
    y_apex = int(m * x_apex + b)

    x_start = 0  # Frame edge endpoint
    y_start = int(m * x_start + b)

    x_ground_left = int(w * 0.15)  # Ground reference line left end
    y_ground = y_apex

    # 4. Visualization & Annotation
    fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
    ax.imshow(img_rgb)

    # Ground Baseline (Horizontal Reference)
    ax.plot(
        [x_ground_left, x_apex],
        [y_ground, y_apex],
        color="cyan",
        linestyle="--",
        linewidth=2.0,
        label="Horizontal Baseline (Ground)",
    )

    # Inclined Surface Line
    ax.plot(
        [x_start, x_apex],
        [y_start, y_apex],
        color="lime",
        linestyle="-",
        linewidth=2.5,
        label=f"Inclined Surface ({alpha_deg:.1f}°)",
    )

    # Angle Arc Indicator
    arc_radius = int(w * 0.18)
    arc = patches.Arc(
        (x_apex, y_apex),
        width=arc_radius,
        height=arc_radius,
        angle=0,
        theta1=180 - alpha_deg,
        theta2=180,
        color="yellow",
        linewidth=2.2,
    )
    ax.add_patch(arc)

    # Angle Text Tag
    text_x = x_apex - int(arc_radius * 0.7)
    text_y = y_apex - int(arc_radius * 0.25)
    ax.text(
        text_x,
        text_y,
        f"α = {alpha_deg:.1f}°",
        fontsize=20,
        fontweight="bold",
        color="yellow",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="black",
            alpha=0.8,
            edgecolor="yellow",
        ),
    )

    # Summary Overlay Box
    info_text = f"Incline Angle (α): {alpha_deg:.2f}°"
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="black",
            alpha=0.8,
            edgecolor="white",
        ),
        color="white",
    )

    ax.set_title(
        f"Ground Angle Measurement: {img_path.name}",
        fontsize=20,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=20)
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved annotated image to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Path to your single image
    target_image = "angle_measurement.jpg"

    measure_and_draw_incline_angle(
        image_path=target_image,
        output_path="measured_incline_result.png",
        show_plot=True,
    )
