"""
Generate a random wavefront and simulate a Shack-Hartmann image.

Part 1:
- Randomly sample coefficients for the first N non-piston Zernike terms.
- Build a wavefront phase map on the configured grid.

Part 2:
- Use local wavefront slopes per lenslet to synthesize focal-spot shifts.
- Render a Shack-Hartmann-like spot image on the configured grid.

All physical sizes are interpreted in meters.
"""

from __future__ import annotations

import argparse
from math import comb, floor, sqrt
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def j_to_mn(j: int) -> Tuple[int, int]:
    n = int(np.ceil((-3.0 + np.sqrt(9.0 + 8.0 * j)) / 2.0))
    m = int(2 * j - n * (n + 2))
    return m, n


def radial_polynomial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    if (n - m_abs) % 2 == 1:
        return np.zeros_like(rho)

    r = np.zeros_like(rho)
    upper = (n - m_abs) // 2
    for s in range(upper + 1):
        c = (
            ((-1) ** s)
            * comb(n - s, s)
            * comb(n - 2 * s, (n - m_abs) // 2 - s)
        )
        r = r + c * (rho ** (n - 2 * s))
    return r


def zernike_mode(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    m, n = j_to_mn(j)
    m_abs = abs(m)
    r = radial_polynomial(n=n, m_abs=m_abs, rho=rho)

    if m == 0:
        angular = np.ones_like(phi)
        norm = sqrt(n + 1)
    elif m > 0:
        angular = np.cos(m * phi)
        norm = sqrt(2 * (n + 1))
    else:
        angular = np.sin(m_abs * phi)
        norm = sqrt(2 * (n + 1))

    return norm * r * angular


def build_wavefront_phase(
    cfg: Dict,
    coeff_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    src = cfg["source"]
    mla = cfg["microlens_array"]
    grid = cfg["wavefront_grid"]
    runtime = cfg.get("runtime", {})
    wf_rand = cfg.get("wavefront_random", {})

    wavelength_m = float(src["wavelength_m"])
    seed = int(runtime.get("random_seed", 42))

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])

    used_w = float(mla["computed"]["used_area_m"][0])
    used_h = float(mla["computed"]["used_area_m"][1])

    if abs(nx * dx - used_w) > 1e-12 or abs(ny * dy - used_h) > 1e-12:
        raise ValueError(
            "Grid shape/spacing is inconsistent with used microlens area. "
            "Please keep nx*dx and ny*dy equal to used_area_m."
        )

    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    y = (np.arange(ny) - ny / 2 + 0.5) * dy
    xx, yy = np.meshgrid(x, y)

    radius = min(used_w, used_h) / 2.0
    rho = np.sqrt(xx**2 + yy**2) / radius
    phi = np.arctan2(yy, xx)
    pupil_mask = rho <= 1.0

    n_modes = int(wf_rand.get("n_modes", 5))
    if n_modes < 1:
        raise ValueError("wavefront_random.n_modes must be >= 1")

    coeff_std_waves = float(wf_rand.get("coeff_std_waves", 0.08))
    effective_coeff_std_waves = coeff_std_waves * float(coeff_scale)

    rng = np.random.default_rng(seed)
    coeff_std_m = effective_coeff_std_waves * wavelength_m
    coeffs = rng.normal(loc=0.0, scale=coeff_std_m, size=n_modes)

    opd_m = np.zeros((ny, nx), dtype=np.float64)
    for idx, j in enumerate(range(1, n_modes + 1)):
        mode = zernike_mode(j=j, rho=rho, phi=phi)
        mode[~pupil_mask] = 0.0
        opd_m += coeffs[idx] * mode

    opd_m[~pupil_mask] = 0.0
    phase_rad = (2.0 * np.pi / wavelength_m) * opd_m
    phase_rad[~pupil_mask] = 0.0

    coeff_map = {f"j{j}": float(c) for j, c in zip(range(1, n_modes + 1), coeffs)}
    return phase_rad, opd_m, pupil_mask, xx, coeff_map


def simulate_sh_image(
    cfg: Dict,
    opd_m: np.ndarray,
    pupil_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    src = cfg["source"]
    mla = cfg["microlens_array"]
    grid = cfg["wavefront_grid"]
    prop = cfg.get("propagation", {})
    det = cfg.get("detection", {})

    wavelength_m = float(src["wavelength_m"])
    source_intensity = float(src.get("intensity_au", 1.0))
    min_lenslet_fill = float(det.get("min_lenslet_fill", 0.6))

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])
    s = int(grid["samples_per_lenslet"])

    lenslet_count_x = int(mla["computed"]["lenslet_count_x"])
    lenslet_count_y = int(mla["computed"]["lenslet_count_y"])

    if nx != lenslet_count_x * s or ny != lenslet_count_y * s:
        raise ValueError(
            "Grid is not lenslet-aligned: nx/ny must equal lenslet_count * samples_per_lenslet."
        )

    f = mla.get("lenslet_focal_length_m", None)
    if f is None:
        f = prop.get("distance_lenslet_to_cmos_m", None)
    if f is None:
        # Default assumption for a "typical" lenslet when config leaves it empty.
        f = 1.0e-2
    focal_length_m = float(f)

    pitch_m = float(mla["pitch_m"])
    fill_factor = float(mla.get("fill_factor", 1.0))
    clear_aperture_m = pitch_m * np.sqrt(max(fill_factor, 1e-9))

    airy_radius_m = 1.22 * wavelength_m * focal_length_m / clear_aperture_m
    sigma_m = airy_radius_m / 2.355
    sigma_px = max(sigma_m / dx, 0.8)

    d_w_dy, d_w_dx = np.gradient(opd_m, dy, dx)

    image = np.zeros((ny, nx), dtype=np.float32)

    max_shift_px = 0.0
    mean_shift_acc = 0.0
    shift_count = 0

    for ly in range(lenslet_count_y):
        y0 = ly * s
        y1 = (ly + 1) * s
        for lx in range(lenslet_count_x):
            x0 = lx * s
            x1 = (lx + 1) * s

            patch_mask = pupil_mask[y0:y1, x0:x1]
            illum = float(np.mean(patch_mask))
            if illum <= max(1e-6, min_lenslet_fill):
                continue

            sx = float(np.mean(d_w_dx[y0:y1, x0:x1][patch_mask]))
            sy = float(np.mean(d_w_dy[y0:y1, x0:x1][patch_mask]))

            shift_x_px = (focal_length_m * sx) / dx
            shift_y_px = (focal_length_m * sy) / dy

            shift_mag_px = float(np.hypot(shift_x_px, shift_y_px))
            max_shift_px = max(max_shift_px, shift_mag_px)
            mean_shift_acc += shift_mag_px
            shift_count += 1

            cx = x0 + 0.5 * (s - 1) + shift_x_px
            cy = y0 + 0.5 * (s - 1) + shift_y_px

            r = int(floor(4.0 * sigma_px)) + 1
            x_min = max(0, int(floor(cx)) - r)
            x_max = min(nx - 1, int(floor(cx)) + r)
            y_min = max(0, int(floor(cy)) - r)
            y_max = min(ny - 1, int(floor(cy)) + r)

            # Spot center may be outside sensor for large aberrations.
            # If clipped window has no overlap with sensor, skip this spot.
            if x_min > x_max or y_min > y_max:
                continue

            xx = np.arange(x_min, x_max + 1, dtype=np.float64)
            yy = np.arange(y_min, y_max + 1, dtype=np.float64)
            gx, gy = np.meshgrid(xx, yy)

            spot = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * sigma_px**2))
            spot_sum = float(np.sum(spot)) + 1e-12
            spot = spot / spot_sum

            image[y_min:y_max + 1, x_min:x_max + 1] += (
                source_intensity * illum * spot
            ).astype(np.float32)

    mean_shift_px = mean_shift_acc / shift_count if shift_count > 0 else 0.0

    meta = {
        "focal_length_m": focal_length_m,
        "sigma_px": float(sigma_px),
        "airy_radius_m": float(airy_radius_m),
        "max_shift_px": float(max_shift_px),
        "mean_shift_px": float(mean_shift_px),
        "shift_count": float(shift_count),
    }
    return image, meta


def plot_results(
    cfg: Dict,
    phase_rad: np.ndarray,
    sh_image: np.ndarray,
    coeff_map: Dict[str, float],
    spot_meta: Dict[str, float],
    output_png: Path,
) -> None:
    mla = cfg["microlens_array"]
    used_w = float(mla["computed"]["used_area_m"][0])
    used_h = float(mla["computed"]["used_area_m"][1])

    extent = (-used_w / 2.0, used_w / 2.0, -used_h / 2.0, used_h / 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    lim = np.nanmax(np.abs(phase_rad))
    im0 = axes[0].imshow(
        phase_rad,
        origin="lower",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        extent=extent,
    )
    axes[0].set_title("Random Wavefront Phase (rad)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        sh_image,
        origin="lower",
        cmap="inferno",
        extent=extent,
    )
    pitch = float(mla["pitch_m"])
    lenslet_count_x = int(mla["computed"]["lenslet_count_x"])
    lenslet_count_y = int(mla["computed"]["lenslet_count_y"])
    x_centers = -used_w / 2.0 + (np.arange(lenslet_count_x) + 0.5) * pitch
    y_centers = -used_h / 2.0 + (np.arange(lenslet_count_y) + 0.5) * pitch
    cx, cy = np.meshgrid(x_centers, y_centers)

    axes[1].scatter(
        cx.ravel(),
        cy.ravel(),
        s=6,
        c="cyan",
        marker="+",
        linewidths=0.35,
        alpha=0.85,
        label="Lenslet centers",
    )
    axes[1].set_title("Simulated Shack-Hartmann Image")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].legend(loc="upper right", fontsize=7, framealpha=0.75)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    coeff_text = ", ".join([f"{k}={v:.3e} m" for k, v in coeff_map.items()])
    meta_text = (
        f"f={spot_meta['focal_length_m']:.3e} m, "
        f"spot_sigma={spot_meta['sigma_px']:.2f} px, "
        f"mean_shift={spot_meta['mean_shift_px']:.2f} px, "
        f"max_shift={spot_meta['max_shift_px']:.2f} px"
    )
    fig.suptitle(coeff_text + "\n" + meta_text, fontsize=10)

    fig.savefig(output_png, dpi=200, pad_inches=0.05)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml in current directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wavefront_and_sh_image.png",
        help="Output figure filename.",
    )
    parser.add_argument(
        "--coeff-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to random Zernike coefficient std.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    out_path = Path(args.output).resolve()

    cfg = load_config(cfg_path)

    phase_rad, opd_m, pupil_mask, _, coeff_map = build_wavefront_phase(
        cfg,
        coeff_scale=float(args.coeff_scale),
    )
    sh_image, spot_meta = simulate_sh_image(cfg, opd_m=opd_m, pupil_mask=pupil_mask)

    print(f"Zernike coeff scale: {float(args.coeff_scale):.3f}")
    print(f"Mean spot shift: {spot_meta['mean_shift_px']:.3f} px")
    print(f"Max spot shift: {spot_meta['max_shift_px']:.3f} px")

    plot_results(
        cfg=cfg,
        phase_rad=phase_rad,
        sh_image=sh_image,
        coeff_map=coeff_map,
        spot_meta=spot_meta,
        output_png=out_path,
    )


if __name__ == "__main__":
    main()


