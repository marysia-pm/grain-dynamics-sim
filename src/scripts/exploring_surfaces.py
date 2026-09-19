import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

# Benchmark NANOVEA Profilometry Reference Data
NANOVEA_DATA = {
    "grit": np.array([120, 180, 320, 800, 2000]),
    "diam_um": np.array([127.0, 105.6, 67.18, 28.16, 21.27]),
    "Sa_um": np.array([42.37, 27.28, 17.92, 6.273, 3.639]),
}

# Log-log power-law interpolation for grit estimation
_log_diams = np.log(NANOVEA_DATA["diam_um"])
_log_grits = np.log(NANOVEA_DATA["grit"])
_grit_interp_func = interp1d(_log_diams, _log_grits, kind="linear", fill_value="extrapolate")


def estimate_grit(mean_diam_um: float) -> int:
    """Calculates estimated grit rating from particle diameter."""
    log_d = np.log(mean_diam_um)
    return max(1, int(round(np.exp(_grit_interp_func(log_d)))))


def generate_calibrated_sandpaper(
    target_diam_um: float = 127.0,
    target_Sa_um: float = 42.37,
    patch_size_cm: float = 0.20,
    incline_deg: float = 30.0,
    grid_res: int = 500,
):
    r_mean_cm = (target_diam_um / 2.0) / 10000.0
    r_std_cm = r_mean_cm * 0.15

    # 1. Padded Seed Placement (Ensures complete half-spheres on all outer boundaries)
    spacing = r_mean_cm * 1.35
    pad = r_mean_cm * 3.0

    gx = np.arange(-pad, patch_size_cm + pad, spacing)
    gy = np.arange(-pad, patch_size_cm + pad, spacing)
    GX, GY = np.meshgrid(gx, gy)

    jitter = spacing * 0.40
    cx = (GX + np.random.uniform(-jitter, jitter, size=GX.shape)).ravel()
    cy = (GY + np.random.uniform(-jitter, jitter, size=GY.shape)).ravel()

    n_grains = len(cx)
    r_grains = np.random.normal(r_mean_cm, r_std_cm, size=n_grains)
    r_grains = np.clip(r_grains, r_mean_cm * 0.5, r_mean_cm * 1.5)

    z_aspect = 1.30
    z_offsets = np.random.normal(0.0, r_mean_cm * 0.40, size=n_grains)

    # 2. Grid Coordinates
    x_coords = np.linspace(0, patch_size_cm, grid_res)
    y_coords = np.linspace(0, patch_size_cm, grid_res)
    X_m, Y_m = np.meshgrid(x_coords, y_coords)
    grid_pts = np.column_stack([X_m.ravel(), Y_m.ravel()])

    # 3. KDTree Distance Query
    tree = cKDTree(np.column_stack([cx, cy]))
    dists, indices = tree.query(grid_pts, k=4)

    # 4. Analytical Semi-Elliptical Surface Function
    z_flat = np.full(len(grid_pts), -1e5)
    for k in range(4):
        d_k = dists[:, k]
        idx_k = indices[:, k]
        r_k = r_grains[idx_k]
        z_off_k = z_offsets[idx_k]
        diff_k = np.maximum(0.0, r_k**2 - d_k**2)
        z_k = z_off_k + z_aspect * np.sqrt(diff_k)
        z_flat = np.maximum(z_flat, z_k)

    z_local = z_flat.reshape((grid_res, grid_res))

    # Resin binder base leveling
    resin_floor = -r_mean_cm * 0.10
    z_local = np.maximum(z_local, resin_floor)
    z_local = z_local - z_local.min()

    # Precision Calibration to match target Sa
    current_Sa = np.mean(np.abs(z_local - np.mean(z_local))) * 10000.0
    if current_Sa > 0:
        calib_factor = target_Sa_um / current_Sa
        z_local = z_local * calib_factor

    Sa_um = np.mean(np.abs(z_local - np.mean(z_local))) * 10000.0
    grit_val = estimate_grit(target_diam_um)

    # 5. 3D Rigid Rotation
    theta = np.radians(incline_deg)
    X_3d = X_m
    Y_3d = Y_m * np.cos(theta) - z_local * np.sin(theta)
    Z_3d = Y_m * np.sin(theta) + z_local * np.cos(theta)

    return (
        X_m,
        Y_m,
        z_local,
        X_3d,
        Y_3d,
        Z_3d,
        patch_size_cm,
        Sa_um,
        target_diam_um,
        grit_val,
    )


# --- Generate 2x5 Comparison Panel ---
try:
    plt.style.use("https://raw.githubusercontent.com/turczyneq/softmatter-style/main/softmatter.mplstyle")
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

fig = plt.figure(figsize=(22, 9))

# GridSpec controls exact vertical (hspace) and horizontal (wspace) separation
gs = fig.add_gridspec(2, 5, hspace=0.5, wspace=0.15)
cmap_name = "terrain"

for col, target_grit in enumerate(NANOVEA_DATA["grit"]):
    target_diam_um = NANOVEA_DATA["diam_um"][col]
    target_Sa_um = NANOVEA_DATA["Sa_um"][col]

    (
        X_m,
        Y_m,
        z_local,
        X_3d,
        Y_3d,
        Z_3d,
        patch_size_cm,
        Sa_um,
        diam_um,
        grit,
    ) = generate_calibrated_sandpaper(
        target_diam_um=target_diam_um,
        target_Sa_um=target_Sa_um,
        patch_size_cm=0.20,
        grid_res=500,
    )

    # --- Row 1: 2D Topographic Height Map ---
    ax1 = fig.add_subplot(gs[0, col])
    im = ax1.imshow(
        z_local * 10.0,
        extent=[0, patch_size_cm * 10, 0, patch_size_cm * 10],
        origin="lower",
        cmap=cmap_name,
    )
    ax1.set_title(
        f"Simulated P{grit} (Ref: P{target_grit})\n Diam: {diam_um:.1f} µm\n$S_a$: {Sa_um:.2f} µm",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    if col == 0:
        ax1.set_ylabel("y (mm)", fontweight="bold")
    ax1.set_xlabel("x (mm)", fontweight="bold")

    # --- Row 2: 3D Surface Topography ---
    ax2 = fig.add_subplot(gs[1, col], projection="3d")
    ds = 4  # Downsample step for clean line rendering

    # Downsample geometry and local height synchronously
    X_ds = X_3d[::ds, ::ds]
    Y_ds = Y_3d[::ds, ::ds]
    Z_ds = Z_3d[::ds, ::ds]
    z_local_ds = z_local[::ds, ::ds]

    norm = mcolors.Normalize(vmin=z_local_ds.min(), vmax=z_local_ds.max())
    facecolors = plt.get_cmap(cmap_name)(norm(z_local_ds))

    ax2.plot_surface(
        X_ds * 10,
        Y_ds * 10,
        Z_ds * 10,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # Negative pad pulls the title down closer to the 3D surface
    ax2.set_title(f"3D Topography (P{grit})", fontsize=14, fontweight="bold", pad=-8)

    if col == 0:
        ax2.set_zlabel("Z (mm)", fontweight="bold")
    ax2.set_xlabel("X (mm)", fontweight="bold")
    ax2.set_ylabel("Y (mm)", fontweight="bold")

    ax2.set_box_aspect((1, 1, 0.5))
    ax2.view_init(elev=25, azim=-60)

output_filename = "calibrated_sandpaper_2x5_fixed_scale.png"
plt.savefig(output_filename, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f'Simulation complete. Figure saved directly as "{output_filename}".')
