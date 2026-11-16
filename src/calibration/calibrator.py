"""Data-driven mathematical threshold calibration for Posture (Sitting vs Standing)
and Activity (Active Typing / Working vs Idle).
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import src.config  # noqa: F401 - ensures MPLCONFIGDIR=/tmp/mpl before matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


@dataclass
class ZonePostureThreshold:
    """Calibrated Sitting-to-Standing threshold for a camera perspective zone."""
    zone_name: str
    x_min_px: float
    x_max_px: float
    threshold_px: float
    threshold_norm: float
    enter_standing_px: float
    exit_to_sitting_px: float
    seated_median_px: float
    standing_median_px: float
    separability_eta: float
    sample_count: int


@dataclass
class CalibratedThresholds:
    """Complete data-driven calibration profile exported to JSON."""
    frame_width: int
    frame_height: int
    fps: float
    wrist_window_frames: int
    min_dwell_frames: int
    # Global posture metrics
    global_posture_threshold_px: float
    global_posture_threshold_norm: float
    global_posture_separability_eta: float
    use_perspective_zones: bool
    perspective_split_x_px: Optional[float]
    posture_zones: Dict[str, ZonePostureThreshold] = field(default_factory=dict)
    # Wrist velocity metrics
    wrist_velocity_threshold_px_s: float = 0.0
    wrist_velocity_threshold_px_frame: float = 0.0
    wrist_velocity_threshold_norm_s: float = 0.0
    enter_working_vel_px_s: float = 0.0
    exit_to_idle_vel_px_s: float = 0.0
    idle_mode_peak_px_s: float = 0.0
    active_mode_peak_px_s: float = 0.0
    velocity_separability_eta: float = 0.0
    # Extended 12-motion category thresholds
    walking_horiz_vel_threshold_px_s: float = 68.0
    idle_confirm_seconds: float = 5.0
    bilateral_typing_ratio_threshold: float = 0.35
    phone_ear_dist_norm_threshold: float = 0.58
    drinking_face_dist_norm_threshold: float = 0.48
    slouch_spine_angle_deg_threshold: float = 15.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "CalibratedThresholds":
        with open(path, "r") as f:
            raw = json.load(f)
        zones_raw = raw.pop("posture_zones", {})
        zones = {k: ZonePostureThreshold(**v) for k, v in zones_raw.items()}
        return cls(**raw, posture_zones=zones)


def otsu_threshold_1d(values: np.ndarray, num_bins: int = 256) -> Tuple[float, float]:
    """Compute exact 1D Otsu threshold maximizing between-class variance.

    Returns
    -------
    (threshold, eta)
        threshold : optimal split point in data units
        eta : normalized separability metric sigma_B^2 / sigma_T^2 in [0, 1]
    """
    clean = values[np.isfinite(values)]
    if len(clean) < 4:
        val = float(np.median(clean)) if len(clean) > 0 else 0.0
        return val, 0.0

    vmin, vmax = float(np.min(clean)), float(np.max(clean))
    if vmax - vmin < 1e-6:
        return vmin, 0.0

    hist, bin_edges = np.histogram(clean, bins=num_bins, range=(vmin, vmax))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    probs = hist.astype(np.float64) / float(np.sum(hist))

    omega0 = np.cumsum(probs)
    omega1 = 1.0 - omega0
    mu_cum = np.cumsum(probs * bin_centers)
    mu_total = mu_cum[-1]

    total_var = float(np.sum(probs * (bin_centers - mu_total) ** 2))
    if total_var <= 1e-12:
        return float(np.median(clean)), 0.0

    valid = (omega0 > 1e-6) & (omega1 > 1e-6)
    sigma_b2 = np.zeros_like(omega0)
    mu0 = np.where(valid, mu_cum / np.maximum(omega0, 1e-12), 0.0)
    mu1 = np.where(valid, (mu_total - mu_cum) / np.maximum(omega1, 1e-12), 0.0)
    sigma_b2[valid] = omega0[valid] * omega1[valid] * (mu0[valid] - mu1[valid]) ** 2

    max_sigma = float(np.max(sigma_b2))
    plateau_mask = sigma_b2 >= max_sigma * (1.0 - 1e-9)
    best_thresh = float(np.mean(bin_centers[plateau_mask]))
    eta = float(min(max(max_sigma / total_var, 0.0), 1.0))
    return best_thresh, eta


def kde_valley_or_otsu_1d(
    values: np.ndarray,
    num_grid: int = 512,
    prefer_upper_valley: bool = False,
) -> Tuple[float, float, float, float]:
    """Locate bimodal KDE valley between primary modes, falling back to Otsu if unimodal.

    Returns (threshold, lower_peak, upper_peak, otsu_eta).
    """
    clean = values[np.isfinite(values)]
    otsu_t, eta = otsu_threshold_1d(clean)
    if len(clean) < 30 or (np.max(clean) - np.min(clean)) < 1e-5:
        low_med = float(np.median(clean[clean <= otsu_t])) if np.any(clean <= otsu_t) else otsu_t
        high_med = float(np.median(clean[clean > otsu_t])) if np.any(clean > otsu_t) else otsu_t
        return otsu_t, low_med, high_med, eta

    q01, q99 = np.percentile(clean, [1.0, 99.0])
    if q99 - q01 < 1e-5:
        q01, q99 = float(np.min(clean)), float(np.max(clean))

    grid = np.linspace(q01, q99, num_grid)
    try:
        kde = gaussian_kde(clean, bw_method="scott")
        density = kde(grid)
        peaks, _ = find_peaks(density, prominence=0.06 * np.max(density), distance=max(num_grid // 5, 20))
        if len(peaks) >= 2:
            p_low = int(peaks[0])
            upper_candidates = peaks[1:]
            p_high = int(upper_candidates[np.argmax(density[upper_candidates])])
            if p_high > p_low + 4:
                if prefer_upper_valley:
                    valleys, _ = find_peaks(-density[p_low : p_high + 1])
                    valleys = p_low + valleys
                    low_cap = 0.35 * min(float(density[p_low]), float(density[p_high]))
                    valid_v = [int(v) for v in valleys if float(density[v]) <= low_cap]
                    if valid_v:
                        valley_idx = valid_v[-1]
                        return float(grid[valley_idx]), float(grid[p_low]), float(grid[p_high]), eta

                margin = max(int(0.15 * (p_high - p_low)), 1)
                s_lo = p_low + margin
                s_hi = max(p_high - margin, s_lo + 1)
                valley_rel = int(np.argmin(density[s_lo : s_hi + 1]))
                valley_idx = s_lo + valley_rel
                return float(grid[valley_idx]), float(grid[p_low]), float(grid[p_high]), eta
    except Exception:
        pass

    low_vals = clean[clean <= otsu_t]
    high_vals = clean[clean > otsu_t]
    low_med = float(np.median(low_vals)) if len(low_vals) > 0 else otsu_t
    high_med = float(np.median(high_vals)) if len(high_vals) > 0 else otsu_t
    return otsu_t, low_med, high_med, eta


def _calibrate_single_zone_posture(
    dist_px: np.ndarray,
    frame_height: float,
    zone_name: str,
    x_min_px: float,
    x_max_px: float,
    hysteresis_frac: float = 0.04,
) -> ZonePostureThreshold:
    """Calibrate Sitting vs Standing shoulder-to-bottom threshold for a spatial zone."""
    clean = dist_px[np.isfinite(dist_px)]
    n = len(clean)
    if n == 0:
        default_t = 0.30 * frame_height
        return ZonePostureThreshold(
            zone_name=zone_name,
            x_min_px=x_min_px,
            x_max_px=x_max_px,
            threshold_px=round(default_t, 2),
            threshold_norm=round(default_t / frame_height, 4),
            enter_standing_px=round(default_t * (1.0 + hysteresis_frac), 2),
            exit_to_sitting_px=round(default_t * (1.0 - hysteresis_frac), 2),
            seated_median_px=round(default_t * 0.7, 2),
            standing_median_px=round(default_t * 1.3, 2),
            separability_eta=0.0,
            sample_count=0,
        )

    thresh_val, low_peak, high_peak, eta = kde_valley_or_otsu_1d(clean, prefer_upper_valley=True)
    p10, p50, p95 = np.percentile(clean, [10.0, 50.0, 95.0])
    spread_95_10 = float(p95 - p10)

    # Guard against unimodal zone (e.g. an employee who stayed seated 100% of the clip):
    # require meaningful vertical excursion (> 5.5% of frame height) and bimodal separation
    min_excursion = 0.055 * frame_height
    if spread_95_10 < min_excursion or (high_peak - low_peak) < min_excursion:
        mad = float(np.median(np.abs(clean - p50)))
        thresh_val = float(p50 + max(4.5 * mad, 0.065 * frame_height))
        seated_med = float(p50)
        standing_med = float(thresh_val + 0.06 * frame_height)
    else:
        # Ensure threshold sits cleanly above dense seated core (at least seated_mode + 3.5% frame_height)
        seated_vals = clean[clean <= thresh_val]
        standing_vals = clean[clean > thresh_val]
        seated_med = float(np.median(seated_vals)) if len(seated_vals) > 0 else float(low_peak)
        standing_med = float(np.median(standing_vals)) if len(standing_vals) > 0 else float(high_peak)
        min_safe_thresh = seated_med + 0.035 * frame_height
        if thresh_val < min_safe_thresh and standing_med > min_safe_thresh:
            thresh_val = 0.5 * (min_safe_thresh + standing_med)

    delta = thresh_val * hysteresis_frac
    return ZonePostureThreshold(
        zone_name=zone_name,
        x_min_px=round(float(x_min_px), 1),
        x_max_px=round(float(x_max_px), 1),
        threshold_px=round(float(thresh_val), 2),
        threshold_norm=round(float(thresh_val / max(frame_height, 1.0)), 4),
        enter_standing_px=round(float(thresh_val + delta), 2),
        exit_to_sitting_px=round(float(thresh_val - delta), 2),
        seated_median_px=round(seated_med, 2),
        standing_median_px=round(standing_med, 2),
        separability_eta=round(eta, 4),
        sample_count=int(n),
    )


def calibrate_thresholds_from_features(
    feature_df: pl.DataFrame,
    wrist_window_frames: int = 30,
    min_dwell_frames: int = 15,
    enable_perspective_zones: bool = True,
) -> CalibratedThresholds:
    """Analyze telemetry features and compute data-backed posture & wrist velocity thresholds."""
    present = feature_df.filter(pl.col("presence") == 1)
    if present.is_empty():
        raise ValueError("Cannot calibrate thresholds: no 'presence == 1' rows found in telemetry.")

    frame_w = int(present["frame_width"].drop_nulls().median() or 1920)
    frame_h = int(present["frame_height"].drop_nulls().median() or 1080)
    fps = float(present["fps"].drop_nulls().median() or 30.0)

    # --- Action 1: Posture Calibration (Shoulder-to-Bottom Vertical Distance) ---
    dist_all = present["shoulder_dist_bottom_px"].drop_nulls().to_numpy()
    sh_x_all = present["avg_shoulder_x"].drop_nulls().to_numpy()

    global_zone = _calibrate_single_zone_posture(
        dist_px=dist_all,
        frame_height=float(frame_h),
        zone_name="global",
        x_min_px=0.0,
        x_max_px=float(frame_w),
    )

    zones: Dict[str, ZonePostureThreshold] = {"global": global_zone}
    use_zones = False
    split_x: Optional[float] = None

    if enable_perspective_zones and len(sh_x_all) >= 200 and len(sh_x_all) == len(dist_all):
        cand_split_x, eta_x = otsu_threshold_1d(sh_x_all)
        left_mask = sh_x_all <= cand_split_x
        right_mask = sh_x_all > cand_split_x

        if eta_x >= 0.50 and int(np.sum(left_mask)) >= 100 and int(np.sum(right_mask)) >= 100:
            # Check if the two camera zones genuinely sit at different vertical baselines
            left_p25 = float(np.percentile(dist_all[left_mask], 25.0))
            right_p25 = float(np.percentile(dist_all[right_mask], 25.0))
            if abs(left_p25 - right_p25) >= 0.06 * frame_h:
                use_zones = True
                split_x = round(float(cand_split_x), 1)
                # Label the higher-in-frame zone 'background' and lower-in-frame zone 'foreground'
                left_name = "background" if left_p25 > right_p25 else "foreground"
                right_name = "foreground" if left_name == "background" else "background"

                zones[left_name] = _calibrate_single_zone_posture(
                    dist_px=dist_all[left_mask],
                    frame_height=float(frame_h),
                    zone_name=left_name,
                    x_min_px=0.0,
                    x_max_px=split_x,
                )
                zones[right_name] = _calibrate_single_zone_posture(
                    dist_px=dist_all[right_mask],
                    frame_height=float(frame_h),
                    zone_name=right_name,
                    x_min_px=split_x,
                    x_max_px=float(frame_w),
                )

    # --- Action 2: Activity Calibration (30-Frame Rolling Euclidean Wrist Velocity) ---
    # Calibrate Idle vs Active Typing specifically on seated desk-work observations
    # so whole-body walking speeds (100-350 px/s) do not skew the micro-movement threshold.
    vel_col = present["rolling_30f_wrist_vel_px_s"].to_numpy()
    if use_zones and split_x is not None and len(sh_x_all) == len(present):
        seated_mask = np.zeros(len(present), dtype=bool)
        for z_name, z_cfg in zones.items():
            if z_name == "global":
                continue
            in_z = (sh_x_all >= z_cfg.x_min_px) & (sh_x_all <= z_cfg.x_max_px)
            seated_mask |= in_z & (dist_all <= z_cfg.threshold_px)
    else:
        seated_mask = dist_all <= global_zone.threshold_px

    seated_vel_mask = seated_mask & np.isfinite(vel_col) & (vel_col >= 0.0)
    if "torso_horiz_vel_px_s" in present.columns:
        hv_col = np.nan_to_num(present["torso_horiz_vel_px_s"].to_numpy(), nan=0.0)
        seated_vel_mask &= hv_col < 45.0
    seated_vel = vel_col[seated_vel_mask]
    vel_pool = seated_vel if len(seated_vel) >= 30 else vel_col[np.isfinite(vel_col) & (vel_col >= 0.0)]

    # Trim extreme track-entry / walking transient outliers while preserving synthetic typing ranges
    if len(vel_pool) >= 50:
        med_v = float(np.median(vel_pool))
        p90_v = float(np.percentile(vel_pool, 90.0))
        if med_v < 30.0:
            upper_cap = max(p90_v * 1.5, 45.0)
            trimmed = vel_pool[vel_pool <= upper_cap]
            if len(trimmed) >= 30:
                vel_pool = trimmed

    if len(vel_pool) < 10:
        vel_thresh_px_s = 12.0
        idle_peak_px_s = 5.0
        active_peak_px_s = 22.0
        vel_eta = 0.0
    else:
        # Calibrate in log10(v + 1) space where Idle drift vs Active Typing micro-movements separate cleanly
        log_vel = np.log10(vel_pool + 1.0)
        log_thresh, log_low, log_high, vel_eta = kde_valley_or_otsu_1d(log_vel)
        vel_thresh_px_s = float(10.0 ** log_thresh - 1.0)
        idle_peak_px_s = float(10.0 ** log_low - 1.0)
        active_peak_px_s = float(10.0 ** log_high - 1.0)

        # Ensure threshold stays physically above sensor micro-jitter floor (~2.5 px/s)
        vel_thresh_px_s = max(vel_thresh_px_s, 2.5)

    enter_vel = vel_thresh_px_s * 1.08
    exit_vel = vel_thresh_px_s * 0.92

    return CalibratedThresholds(
        frame_width=frame_w,
        frame_height=frame_h,
        fps=round(fps, 2),
        wrist_window_frames=wrist_window_frames,
        min_dwell_frames=min_dwell_frames,
        global_posture_threshold_px=global_zone.threshold_px,
        global_posture_threshold_norm=global_zone.threshold_norm,
        global_posture_separability_eta=global_zone.separability_eta,
        use_perspective_zones=use_zones,
        perspective_split_x_px=split_x,
        posture_zones=zones,
        wrist_velocity_threshold_px_s=round(vel_thresh_px_s, 2),
        wrist_velocity_threshold_px_frame=round(vel_thresh_px_s / max(fps, 1.0), 4),
        wrist_velocity_threshold_norm_s=round(vel_thresh_px_s / max(float(frame_h), 1.0), 5),
        enter_working_vel_px_s=round(enter_vel, 2),
        exit_to_idle_vel_px_s=round(exit_vel, 2),
        idle_mode_peak_px_s=round(idle_peak_px_s, 2),
        active_mode_peak_px_s=round(active_peak_px_s, 2),
        velocity_separability_eta=round(vel_eta, 4),
        walking_horiz_vel_threshold_px_s=round(max(0.036 * float(frame_w), 45.0), 2),
        idle_confirm_seconds=5.0,
        bilateral_typing_ratio_threshold=0.35,
        phone_ear_dist_norm_threshold=0.58,
        drinking_face_dist_norm_threshold=0.48,
        slouch_spine_angle_deg_threshold=15.0,
    )


def generate_calibration_plots(
    classified_df: pl.DataFrame,
    thresholds: CalibratedThresholds,
    plots_dir: Path,
) -> Dict[str, Path]:
    """Render posture and wrist-velocity calibration plots for Milestone 2 validation."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    present = classified_df.filter(pl.col("presence") == 1)

    posture_plot_path = plots_dir / "milestone2_posture_calibration.png"
    velocity_plot_path = plots_dir / "milestone2_wrist_velocity_calibration.png"

    # --- Plot 1: Posture Calibration ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    if thresholds.use_perspective_zones and "camera_zone" in present.columns:
        zone_colors = {"foreground": "#2980b9", "background": "#8e44ad"}
        for z_name in ("foreground", "background"):
            if z_name not in thresholds.posture_zones:
                continue
            z_cfg = thresholds.posture_zones[z_name]
            sub_d = present.filter(pl.col("camera_zone") == z_name)["shoulder_dist_bottom_px"].drop_nulls().to_numpy()
            if len(sub_d) > 0:
                ax1.hist(
                    sub_d,
                    bins=45,
                    alpha=0.45,
                    color=zone_colors.get(z_name, "#3498db"),
                    label=f"{z_name.capitalize()} Zone (n={len(sub_d)})",
                    density=True,
                )
                ax1.axvline(
                    z_cfg.threshold_px,
                    color=zone_colors.get(z_name, "#3498db"),
                    linestyle="--",
                    linewidth=2.2,
                    label=f"{z_name.capitalize()} Threshold: {z_cfg.threshold_px:.1f} px ({z_cfg.threshold_norm:.2f}H)",
                )
    else:
        dist_vals = present["shoulder_dist_bottom_px"].drop_nulls().to_numpy()
        ax1.hist(dist_vals, bins=50, alpha=0.6, color="#2980b9", density=True, label="All Tracked Frames")
        ax1.axvline(
            thresholds.global_posture_threshold_px,
            color="#e74c3c",
            linestyle="--",
            linewidth=2.2,
            label=f"Calibrated Threshold: {thresholds.global_posture_threshold_px:.1f} px",
        )

    ax1.set_title("Posture Calibration: Shoulder-to-Bottom Vertical Distance")
    ax1.set_xlabel("Vertical Distance from Shoulders to Bottom Frame Edge (px)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Spatial Camera Plane (X vs Shoulder-to-Bottom Distance colored by Posture)
    sit_df = present.filter(pl.col("posture_state") == "Sitting")
    stand_df = present.filter(pl.col("posture_state") == "Standing")
    walk_df = present.filter(pl.col("posture_state") == "Walking")

    if not sit_df.is_empty():
        ax2.scatter(
            sit_df["avg_shoulder_x"].to_numpy(),
            sit_df["shoulder_dist_bottom_px"].to_numpy(),
            s=8,
            alpha=0.35,
            color="#27ae60",
            label=f"Sitting (n={len(sit_df)})",
        )
    if not stand_df.is_empty():
        ax2.scatter(
            stand_df["avg_shoulder_x"].to_numpy(),
            stand_df["shoulder_dist_bottom_px"].to_numpy(),
            s=12,
            alpha=0.55,
            color="#e67e22",
            label=f"Standing (n={len(stand_df)})",
        )
    if not walk_df.is_empty():
        ax2.scatter(
            walk_df["avg_shoulder_x"].to_numpy(),
            walk_df["shoulder_dist_bottom_px"].to_numpy(),
            s=14,
            alpha=0.65,
            color="#8e44ad",
            label=f"Walking (n={len(walk_df)})",
        )

    if thresholds.use_perspective_zones and thresholds.perspective_split_x_px is not None:
        ax2.axvline(thresholds.perspective_split_x_px, color="#555555", linestyle=":", linewidth=1.5, label="Zone Split X")
        for z_name, z_cfg in thresholds.posture_zones.items():
            if z_name == "global":
                continue
            ax2.hlines(
                z_cfg.threshold_px,
                xmin=z_cfg.x_min_px,
                xmax=z_cfg.x_max_px,
                 colors="#c0392b",
                linestyles="--",
                linewidth=2.2,
            )
    else:
        ax2.axhline(thresholds.global_posture_threshold_px, color="#c0392b", linestyle="--", linewidth=2.2)

    ax2.set_title("Sitting vs. Standing Decision Boundary Across Camera Perspective")
    ax2.set_xlabel("Horizontal Workstation Position Avg_Shoulder_X (px)")
    ax2.set_ylabel("Shoulder-to-Bottom Distance H - Avg_Shoulder_Y (px)")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(posture_plot_path, dpi=150)
    plt.close(fig)

    # --- Plot 2: Wrist Velocity Calibration ---
    fig, (vx1, vx2) = plt.subplots(1, 2, figsize=(14, 5.5))

    vel_vals = present["rolling_30f_wrist_vel_px_s"].drop_nulls().to_numpy()
    vel_clip = np.clip(vel_vals, 0.0, np.percentile(vel_vals, 99.0) if len(vel_vals) > 10 else 150.0)

    log_bins = np.logspace(np.log10(1.0), np.log10(max(float(np.max(vel_clip)) + 5.0, 50.0)), 55)
    vx1.hist(vel_clip + 1.0, bins=log_bins, color="#34495e", alpha=0.65, edgecolor="white")
    vx1.set_xscale("log")

    t_v = thresholds.wrist_velocity_threshold_px_s
    vx1.axvline(t_v + 1.0, color="#e74c3c", linestyle="--", linewidth=2.3, label=f"Speed Threshold: {t_v:.2f} px/s ({thresholds.wrist_velocity_threshold_px_frame:.2f} px/f)")
    vx1.axvspan(
        thresholds.exit_to_idle_vel_px_s + 1.0,
        thresholds.enter_working_vel_px_s + 1.0,
        color="#e74c3c",
        alpha=0.18,
        label=f"Hysteresis Band [{thresholds.exit_to_idle_vel_px_s:.1f}, {thresholds.enter_working_vel_px_s:.1f}] px/s",
    )
    vx1.set_title(f"Rolling {thresholds.wrist_window_frames}-Frame Euclidean Wrist Speed Distribution")
    vx1.set_xlabel("Rolling Wrist Velocity + 1 (px/s, log scale)")
    vx1.set_ylabel("Frame Count")
    vx1.legend(fontsize=8.5)
    vx1.grid(True, which="both", linestyle="--", alpha=0.35)

    # Panel 2: Temporal trace for top longest-duration workstation track(s)
    track_counts = (
        present.group_by("track_id")
        .len()
        .sort("len", descending=True)
    )
    top_tracks = track_counts["track_id"].head(2).to_list()
    palette = ["#2980b9", "#16a085"]
    for idx, tid in enumerate(top_tracks):
        t_sub = present.filter(pl.col("track_id") == tid).sort("frame")
        vx2.plot(
            t_sub["timestamp_sec"].to_numpy(),
            t_sub["rolling_30f_wrist_vel_px_s"].to_numpy(),
            color=palette[idx % len(palette)],
            linewidth=1.3,
            alpha=0.85,
            label=f"Emp #{tid} Rolling 30f Wrist Speed",
        )

    vx2.axhline(t_v, color="#e74c3c", linestyle="--", linewidth=2.0, label=f"Active Typing Cutoff ({t_v:.1f} px/s)")
    vx2.fill_between(
        vx2.get_xlim(),
        0,
        t_v,
        color="#95a5a6",
        alpha=0.12,
        label="Idle Zone",
    )
    vx2.set_ylim(0, min(max(t_v * 6.0, 60.0), 180.0))
    vx2.set_title("Continuous 30-Frame Wrist Velocity Timeline (Active Typing vs. Idle)")
    vx2.set_xlabel("Video Timestamp (seconds)")
    vx2.set_ylabel("Rolling 30f Wrist Velocity (px/s)")
    vx2.legend(fontsize=8.5, loc="upper right")
    vx2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(velocity_plot_path, dpi=150)
    plt.close(fig)

    return {
        "posture_calibration_plot": posture_plot_path,
        "wrist_velocity_calibration_plot": velocity_plot_path,
    }
