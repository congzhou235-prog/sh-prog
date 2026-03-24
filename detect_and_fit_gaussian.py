"""
Detect lenslet spot shifts and fit Zernike coefficients using
2D Gaussian centroid fitting on each lenslet spot.

Pipeline:
1) Generate random wavefront (ground truth coefficients a_n).
2) Simulate SH image using per-lenslet FFT model.
3) Detect lenslet centroids by subpixel 2D Gaussian fitting.
4) Build sensitivity matrix by per-mode calibration simulation.
5) Solve least squares to estimate coefficients and compare to ground truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import simulate_wavefront_and_sh_image as sim



def lenslet_grid_info(cfg: Dict) -> Tuple[int, int, int]:
    grid = cfg["wavefront_grid"]
    mla = cfg["microlens_array"]

    s = int(grid["samples_per_lenslet"])
    lx = int(mla["computed"]["lenslet_count_x"])
    ly = int(mla["computed"]["lenslet_count_y"])

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    if nx != lx * s or ny != ly * s:
        raise ValueError("Grid is not lenslet-aligned in config.")

    return s, lx, ly



def compute_fill_map(pupil_mask: np.ndarray, s: int, lx: int, ly: int) -> np.ndarray:
    fill = np.zeros((ly, lx), dtype=np.float64)
    for j in range(ly):
        y0 = j * s
        y1 = (j + 1) * s
        for i in range(lx):
            x0 = i * s
            x1 = (i + 1) * s
            fill[j, i] = float(np.mean(pupil_mask[y0:y1, x0:x1]))
    return fill



def _fit_gaussian_centroid(
    patch: np.ndarray,
    gx_local: np.ndarray,
    gy_local: np.ndarray,
    fit_threshold_relative: float,
    max_nfev: int,
    sigma_min_px: float,
    sigma_max_px: float,
) -> Tuple[float, float, bool]:
    """
    Fit a separable 2D Gaussian with constant background:
    I(x,y) = b + a * exp(-0.5 * [((x-x0)/sx)^2 + ((y-y0)/sy)^2])

    Returns (x0, y0, success).
    """

    s_y, s_x = patch.shape
    if s_x != s_y:
        return 0.0, 0.0, False

    peak = float(np.max(patch))
    if peak <= 1e-12:
        return 0.0, 0.0, False

    if fit_threshold_relative > 0.0:
        fit_mask = patch >= (fit_threshold_relative * peak)
    else:
        fit_mask = np.ones_like(patch, dtype=bool)

    if int(np.count_nonzero(fit_mask)) < 9:
        fit_mask = patch > 0.0
    if int(np.count_nonzero(fit_mask)) < 9:
        return 0.0, 0.0, False

    w = np.where(fit_mask, patch, 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return 0.0, 0.0, False

    x0_init = float(np.sum(w * gx_local) / w_sum)
    y0_init = float(np.sum(w * gy_local) / w_sum)

    sx_init = float(np.sqrt(np.sum(w * (gx_local - x0_init) ** 2) / w_sum + 1e-12))
    sy_init = float(np.sqrt(np.sum(w * (gy_local - y0_init) ** 2) / w_sum + 1e-12))

    sx_init = float(np.clip(sx_init, sigma_min_px, sigma_max_px))
    sy_init = float(np.clip(sy_init, sigma_min_px, sigma_max_px))

    b_init = float(np.percentile(patch, 10))
    b_init = max(0.0, min(b_init, peak))
    a_init = max(peak - b_init, 1e-9)

    x_data = gx_local[fit_mask].astype(np.float64)
    y_data = gy_local[fit_mask].astype(np.float64)
    z_data = patch[fit_mask].astype(np.float64)

    p0 = np.array([b_init, a_init, x0_init, y0_init, sx_init, sy_init], dtype=np.float64)

    low = np.array([0.0, 0.0, 0.0, 0.0, sigma_min_px, sigma_min_px], dtype=np.float64)
    high = np.array(
        [
            max(peak * 2.0, 1e-9),
            max(peak * 10.0, 1e-9),
            float(s_x - 1),
            float(s_y - 1),
            sigma_max_px,
            sigma_max_px,
        ],
        dtype=np.float64,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        b, a, x0, y0, sx, sy = params
        sx = max(float(sx), 1e-9)
        sy = max(float(sy), 1e-9)
        model = b + a * np.exp(
            -0.5 * (((x_data - x0) / sx) ** 2 + ((y_data - y0) / sy) ** 2)
        )
        return model - z_data

    try:
        result = least_squares(
            residuals,
            x0=p0,
            bounds=(low, high),
            method="trf",
            max_nfev=max_nfev,
        )
    except Exception:
        return 0.0, 0.0, False

    if not np.all(np.isfinite(result.x)):
        return 0.0, 0.0, False

    x_fit = float(result.x[2])
    y_fit = float(result.x[3])
    if not (0.0 <= x_fit <= (s_x - 1) and 0.0 <= y_fit <= (s_y - 1)):
        return 0.0, 0.0, False

    # Accept near-converged solutions too; "status > 0" is strict,
    # while low-res patches may terminate on max_nfev with a good center.
    return x_fit, y_fit, True



def detect_lenslet_centroids(
    image: np.ndarray,
    s: int,
    lx: int,
    ly: int,
    pre_threshold_relative: float,
    background_subtraction: bool,
    gaussian_fit_threshold_relative: float,
    gaussian_fit_max_nfev: int,
    gaussian_fit_sigma_min_px: float,
    gaussian_fit_sigma_max_px: float,
    gaussian_fit_fallback_centroid: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
    - centroids_local: shape (ly, lx, 2), local cell coordinates in pixels
    - valid: shape (ly, lx), whether centroid was detected
    - fitted_by_gaussian: shape (ly, lx), whether Gaussian fitting succeeded
    """

    gx_local, gy_local = np.meshgrid(
        np.arange(s, dtype=np.float64), np.arange(s, dtype=np.float64)
    )

    centroids = np.full((ly, lx, 2), np.nan, dtype=np.float64)
    valid = np.zeros((ly, lx), dtype=bool)
    fitted_by_gaussian = np.zeros((ly, lx), dtype=bool)

    for j in range(ly):
        y0 = j * s
        y1 = (j + 1) * s
        for i in range(lx):
            x0 = i * s
            x1 = (i + 1) * s

            patch = image[y0:y1, x0:x1].astype(np.float64)

            if background_subtraction:
                patch = patch - np.min(patch)

            patch = np.clip(patch, a_min=0.0, a_max=None)

            if pre_threshold_relative > 0.0:
                threshold = pre_threshold_relative * float(np.max(patch))
                patch = np.where(patch >= threshold, patch, 0.0)

            cx, cy, ok = _fit_gaussian_centroid(
                patch=patch,
                gx_local=gx_local,
                gy_local=gy_local,
                fit_threshold_relative=gaussian_fit_threshold_relative,
                max_nfev=gaussian_fit_max_nfev,
                sigma_min_px=gaussian_fit_sigma_min_px,
                sigma_max_px=gaussian_fit_sigma_max_px,
            )

            gaussian_ok = bool(ok)

            if not ok and gaussian_fit_fallback_centroid:
                w_sum = float(np.sum(patch))
                if w_sum > 1e-12:
                    cx = float(np.sum(patch * gx_local) / w_sum)
                    cy = float(np.sum(patch * gy_local) / w_sum)
                    ok = True

            if not ok:
                continue

            centroids[j, i, 0] = cx
            centroids[j, i, 1] = cy
            valid[j, i] = True
            fitted_by_gaussian[j, i] = bool(gaussian_ok)

    return centroids, valid, fitted_by_gaussian



def build_mode_maps(cfg: Dict, n_modes: int) -> Tuple[List[np.ndarray], np.ndarray]:
    mla = cfg["microlens_array"]
    grid = cfg["wavefront_grid"]

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])

    used_w = float(mla["computed"]["used_area_m"][0])
    used_h = float(mla["computed"]["used_area_m"][1])

    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    y = (np.arange(ny) - ny / 2 + 0.5) * dy
    xx, yy = np.meshgrid(x, y)

    radius = min(used_w, used_h) / 2.0
    rho = np.sqrt(xx**2 + yy**2) / radius
    phi = np.arctan2(yy, xx)
    pupil_mask = rho <= 1.0

    modes: List[np.ndarray] = []
    for j in range(1, n_modes + 1):
        mode = sim.zernike_mode(j=j, rho=rho, phi=phi)
        mode = mode.astype(np.float64)
        mode[~pupil_mask] = 0.0
        modes.append(mode)

    return modes, pupil_mask



def build_sensitivity_matrix(
    cfg: Dict,
    mode_maps: List[np.ndarray],
    pupil_mask: np.ndarray,
    centroids_ref: np.ndarray,
    valid_base: np.ndarray,
    calib_delta_m: float,
    s: int,
    lx: int,
    ly: int,
    pre_threshold_relative: float,
    background_subtraction: bool,
    gaussian_fit_threshold_relative: float,
    gaussian_fit_max_nfev: int,
    gaussian_fit_sigma_min_px: float,
    gaussian_fit_sigma_max_px: float,
    gaussian_fit_fallback_centroid: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - A: sensitivity matrix, shape (2*K, n_modes)
    - final_valid: lenslet validity mask used in fitting
    """

    n_modes = len(mode_maps)
    mode_shift_maps: List[np.ndarray] = []
    mode_valid_maps: List[np.ndarray] = []

    for mode in mode_maps:
        opd_mode = calib_delta_m * mode
        image_mode, _ = sim.simulate_sh_image(cfg, opd_mode, pupil_mask)

        cent_mode, valid_mode, _ = detect_lenslet_centroids(
            image=image_mode,
            s=s,
            lx=lx,
            ly=ly,
            pre_threshold_relative=pre_threshold_relative,
            background_subtraction=background_subtraction,
            gaussian_fit_threshold_relative=gaussian_fit_threshold_relative,
            gaussian_fit_max_nfev=gaussian_fit_max_nfev,
            gaussian_fit_sigma_min_px=gaussian_fit_sigma_min_px,
            gaussian_fit_sigma_max_px=gaussian_fit_sigma_max_px,
            gaussian_fit_fallback_centroid=gaussian_fit_fallback_centroid,
        )

        shift_mode = cent_mode - centroids_ref
        mode_shift_maps.append(shift_mode)
        mode_valid_maps.append(valid_mode)

    final_valid = valid_base.copy()
    for v in mode_valid_maps:
        final_valid &= v

    k = int(np.sum(final_valid))
    if k == 0:
        raise RuntimeError("No valid lenslets left after calibration runs.")

    A = np.zeros((2 * k, n_modes), dtype=np.float64)
    for idx in range(n_modes):
        shift = mode_shift_maps[idx]
        col = np.concatenate(
            (shift[:, :, 0][final_valid], shift[:, :, 1][final_valid])
        )
        A[:, idx] = col / calib_delta_m

    return A, final_valid



def plot_fit_summary(
    cfg: Dict,
    sh_image: np.ndarray,
    true_coeff_m: np.ndarray,
    est_coeff_m: np.ndarray,
    output_png: Path,
) -> None:
    src = cfg["source"]
    mla = cfg["microlens_array"]

    wavelength_m = float(src["wavelength_m"])
    used_w = float(mla["computed"]["used_area_m"][0])
    used_h = float(mla["computed"]["used_area_m"][1])
    extent = (-used_w / 2.0, used_w / 2.0, -used_h / 2.0, used_h / 2.0)

    n = len(true_coeff_m)
    idx = np.arange(1, n + 1)

    true_waves = true_coeff_m / wavelength_m
    est_waves = est_coeff_m / wavelength_m

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im = axes[0].imshow(sh_image, origin="lower", cmap="inferno", extent=extent)
    axes[0].set_title("Detected SH Image")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    width = 0.38
    axes[1].bar(idx - width / 2, true_waves, width=width, label="True a_n")
    axes[1].bar(idx + width / 2, est_waves, width=width, label="Fitted a_n")
    axes[1].set_title("Zernike Coefficients (in waves)")
    axes[1].set_xlabel("Mode index j")
    axes[1].set_ylabel("Coefficient (waves)")
    axes[1].set_xticks(idx)
    axes[1].legend()

    fig.savefig(output_png, dpi=200, pad_inches=0.05)
    plt.show()



def make_run_dir(output_root: Path, prefix: str) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{prefix}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{prefix}_{stamp}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--coeff-scale",
        type=float,
        default=4.0,
        help="Scale for random Zernike coefficient generation.",
    )
    parser.add_argument(
        "--calib-delta-m",
        type=float,
        default=1.0e-7,
        help="Calibration amplitude (meters) for each single-mode sensitivity run.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs",
        help="Directory where each run gets a timestamped output folder.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="detect_fit_gaussian",
        help="Prefix for each run folder.",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default="summary.png",
        help="Output figure filename inside the run folder.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default="results.npz",
        help="Output npz filename inside the run folder.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="coeff_compare.csv",
        help="Output CSV filename inside the run folder.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    run_dir = make_run_dir(Path(args.output_root), prefix=str(args.run_prefix))
    out_fig = run_dir / Path(args.output_fig).name
    out_npz = run_dir / Path(args.output_npz).name
    out_csv = run_dir / Path(args.output_csv).name

    cfg = sim.load_config(cfg_path)
    print(f"Run directory: {run_dir}")

    phase_rad, opd_true, pupil_mask, _, coeff_map = sim.build_wavefront_phase(
        cfg, coeff_scale=float(args.coeff_scale)
    )

    n_modes = len(coeff_map)
    true_coeff_m = np.array(
        [coeff_map[f"j{j}"] for j in range(1, n_modes + 1)], dtype=np.float64
    )

    sh_image, sh_meta = sim.simulate_sh_image(cfg, opd_true, pupil_mask)
    ref_image, _ = sim.simulate_sh_image(cfg, np.zeros_like(opd_true), pupil_mask)

    s, lx, ly = lenslet_grid_info(cfg)

    det = cfg.get("detection", {})
    background_subtraction = bool(det.get("background_subtraction", True))
    min_lenslet_fill = float(det.get("min_lenslet_fill", 0.6))

    pre_threshold_relative = float(det.get("gaussian_prethreshold_relative", 0.0))
    gaussian_fit_threshold_relative = float(
        det.get("gaussian_fit_threshold_relative", 0.2)
    )
    gaussian_fit_max_nfev = int(det.get("gaussian_fit_max_nfev", 80))
    gaussian_fit_sigma_min_px = float(det.get("gaussian_fit_sigma_min_px", 0.8))
    gaussian_fit_sigma_max_px = float(
        det.get("gaussian_fit_sigma_max_px", max(2.0, s / 2.0))
    )
    gaussian_fit_fallback_centroid = bool(
        det.get("gaussian_fit_fallback_centroid", True)
    )

    fill_map = compute_fill_map(pupil_mask, s=s, lx=lx, ly=ly)
    valid_fill = fill_map > max(min_lenslet_fill, 1e-6)

    cent_ref, valid_ref, gauss_ref = detect_lenslet_centroids(
        image=ref_image,
        s=s,
        lx=lx,
        ly=ly,
        pre_threshold_relative=pre_threshold_relative,
        background_subtraction=background_subtraction,
        gaussian_fit_threshold_relative=gaussian_fit_threshold_relative,
        gaussian_fit_max_nfev=gaussian_fit_max_nfev,
        gaussian_fit_sigma_min_px=gaussian_fit_sigma_min_px,
        gaussian_fit_sigma_max_px=gaussian_fit_sigma_max_px,
        gaussian_fit_fallback_centroid=gaussian_fit_fallback_centroid,
    )
    cent_meas, valid_meas, gauss_meas = detect_lenslet_centroids(
        image=sh_image,
        s=s,
        lx=lx,
        ly=ly,
        pre_threshold_relative=pre_threshold_relative,
        background_subtraction=background_subtraction,
        gaussian_fit_threshold_relative=gaussian_fit_threshold_relative,
        gaussian_fit_max_nfev=gaussian_fit_max_nfev,
        gaussian_fit_sigma_min_px=gaussian_fit_sigma_min_px,
        gaussian_fit_sigma_max_px=gaussian_fit_sigma_max_px,
        gaussian_fit_fallback_centroid=gaussian_fit_fallback_centroid,
    )

    valid_base = valid_fill & valid_ref & valid_meas

    mode_maps, pupil_mask_check = build_mode_maps(cfg, n_modes=n_modes)
    if not np.array_equal(pupil_mask, pupil_mask_check):
        raise RuntimeError("Internal pupil mask mismatch.")

    A, valid_final = build_sensitivity_matrix(
        cfg=cfg,
        mode_maps=mode_maps,
        pupil_mask=pupil_mask,
        centroids_ref=cent_ref,
        valid_base=valid_base,
        calib_delta_m=float(args.calib_delta_m),
        s=s,
        lx=lx,
        ly=ly,
        pre_threshold_relative=pre_threshold_relative,
        background_subtraction=background_subtraction,
        gaussian_fit_threshold_relative=gaussian_fit_threshold_relative,
        gaussian_fit_max_nfev=gaussian_fit_max_nfev,
        gaussian_fit_sigma_min_px=gaussian_fit_sigma_min_px,
        gaussian_fit_sigma_max_px=gaussian_fit_sigma_max_px,
        gaussian_fit_fallback_centroid=gaussian_fit_fallback_centroid,
    )

    shifts_meas = cent_meas - cent_ref
    b = np.concatenate(
        (shifts_meas[:, :, 0][valid_final], shifts_meas[:, :, 1][valid_final])
    )

    est_coeff_m, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    residual_px = b - A @ est_coeff_m
    rmse_px = float(np.sqrt(np.mean(residual_px**2)))

    coeff_rmse_m = float(np.sqrt(np.mean((est_coeff_m - true_coeff_m) ** 2)))

    print("=== Detection + Fitting Summary ===")
    print(f"Model: {sh_meta.get('model', 'unknown')}")
    print("Centroid method: gaussian_2d_fit")
    print(f"Number of fitted modes: {n_modes}")
    print(f"Valid lenslets used: {int(np.sum(valid_final))}")
    print(f"Residual RMSE (px): {rmse_px:.6f}")
    print(f"Coefficient RMSE (m): {coeff_rmse_m:.6e}")

    ref_valid_count = int(np.sum(valid_ref))
    meas_valid_count = int(np.sum(valid_meas))
    if ref_valid_count > 0:
        print(
            "Gaussian fit usage (ref): "
            f"{int(np.sum(gauss_ref & valid_ref))}/{ref_valid_count}"
        )
    if meas_valid_count > 0:
        print(
            "Gaussian fit usage (meas): "
            f"{int(np.sum(gauss_meas & valid_meas))}/{meas_valid_count}"
        )

    print("")
    print("j | true_a_n (m) | fitted_a_n (m) | error (m)")
    for j in range(1, n_modes + 1):
        t = true_coeff_m[j - 1]
        e = est_coeff_m[j - 1]
        print(f"{j:>1} | {t:+.6e} | {e:+.6e} | {(e - t):+.6e}")

    np.savez_compressed(
        out_npz,
        phase_rad=phase_rad,
        opd_true_m=opd_true,
        sh_image=sh_image,
        true_coeff_m=true_coeff_m,
        est_coeff_m=est_coeff_m,
        residual_px=residual_px,
        valid_final=valid_final,
        A=A,
        b=b,
    )
    print(f"Saved results: {out_npz}")

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("j,true_a_n_m,fitted_a_n_m,error_m\n")
        for j in range(1, n_modes + 1):
            t = true_coeff_m[j - 1]
            e = est_coeff_m[j - 1]
            f.write(f"{j},{t:.12e},{e:.12e},{(e - t):.12e}\n")
    print(f"Saved CSV: {out_csv}")

    plot_fit_summary(
        cfg=cfg,
        sh_image=sh_image,
        true_coeff_m=true_coeff_m,
        est_coeff_m=est_coeff_m,
        output_png=out_fig,
    )
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    main()
