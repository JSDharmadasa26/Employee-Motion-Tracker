"""Finalized mathematical rules engine with Schmitt-trigger hysteresis, temporal dwell
filtering, and high-accuracy 6-class motion categorization:
  1. Sitting
  2. Standing
  3. Walking
  4. Away
  5. Idle
  6. Working
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.calibration.calibrator import (
    CalibratedThresholds,
    calibrate_thresholds_from_features,
    generate_calibration_plots,
)
from src.calibration.features import compute_temporal_features, load_and_extract_features
from src.config import PLOTS_OUTPUT_DIR, TELEMETRY_OUTPUT_DIR, ensure_directories

SIX_MOTION_CLASSES = ("Sitting", "Standing", "Walking", "Away", "Idle", "Working")


def _suppress_short_runs(states: List[str], min_dwell_frames: int) -> List[str]:
    """Eliminate contiguous state runs shorter than `min_dwell_frames`."""
    n = len(states)
    out = list(states)
    dwell = max(int(min_dwell_frames), 1)
    if dwell <= 1 or n <= dwell:
        return out

    changed = True
    while changed:
        changed = False
        runs: List[Tuple[str, int, int]] = []
        run_start = 0
        for i in range(1, n + 1):
            if i == n or out[i] != out[run_start]:
                runs.append((out[run_start], run_start, i))
                run_start = i

        if len(runs) <= 1:
            break

        shortest_idx = -1
        shortest_len = dwell
        for r_idx, (_, s, e) in enumerate(runs):
            r_len = e - s
            if r_len < shortest_len:
                shortest_len = r_len
                shortest_idx = r_idx

        if shortest_idx != -1:
            _, s, e = runs[shortest_idx]
            replacement = runs[shortest_idx - 1][0] if shortest_idx > 0 else runs[shortest_idx + 1][0]
            for k in range(s, e):
                out[k] = replacement
            changed = True
    return out


def apply_hysteresis_and_dwell(
    values: np.ndarray,
    low_thresh: float,
    high_thresh: float,
    low_label: str,
    high_label: str,
    min_dwell_frames: int = 15,
) -> Tuple[List[str], int]:
    """Classify a 1D temporal signal with Schmitt-trigger hysteresis and minimum-dwell suppression."""
    n = len(values)
    if n == 0:
        return [], 0

    mid_thresh = 0.5 * (low_thresh + high_thresh)
    clean_vals = np.where(np.isfinite(values), values, mid_thresh)

    raw_states = [high_label if v > mid_thresh else low_label for v in clean_vals]
    raw_flips = sum(1 for i in range(1, n) if raw_states[i] != raw_states[i - 1])

    init_window = clean_vals[: min(max(min_dwell_frames, 1), n)]
    curr_state = high_label if float(np.median(init_window)) >= mid_thresh else low_label

    states = [curr_state] * n
    for i, val in enumerate(clean_vals):
        if curr_state == low_label and val >= high_thresh:
            curr_state = high_label
        elif curr_state == high_label and val <= low_thresh:
            curr_state = low_label
        states[i] = curr_state

    states = _suppress_short_runs(states, min_dwell_frames)

    final_flips = sum(1 for i in range(1, n) if states[i] != states[i - 1])
    suppressed = max(0, raw_flips - final_flips)
    return states, suppressed


def _dilate_boolean_mask(mask: np.ndarray, half_width: int) -> np.ndarray:
    """Dilate True pulses by +/- `half_width` frames."""
    n = len(mask)
    if n == 0 or half_width <= 0:
        return mask.copy()
    out = np.zeros(n, dtype=bool)
    for idx in np.flatnonzero(mask):
        out[max(0, idx - half_width) : min(n, idx + half_width + 1)] = True
    return out


def _filter_min_true_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only contiguous True runs of length >= `min_len`."""
    n = len(mask)
    if n == 0 or min_len <= 1:
        return mask.copy()
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _enforce_idle_confirmation(
    work_states: List[str],
    idle_confirm_frames: int,
    explicit_idle_mask: Optional[np.ndarray] = None,
) -> List[str]:
    """Require prolonged low-velocity duration (>= idle_confirm_frames, e.g. >5s) before flipping
    from Working to Idle, unless `explicit_idle_mask` (e.g. desk-to-face drinking transition) is active.
    """
    n = len(work_states)
    if n == 0 or idle_confirm_frames <= 1:
        return list(work_states)
    if all(s == "Idle" for s in work_states):
        return list(work_states)

    out = list(work_states)
    runs: List[Tuple[str, int, int]] = []
    run_start = 0
    for i in range(1, n + 1):
        if i == n or out[i] != out[run_start]:
            runs.append((out[run_start], run_start, i))
            run_start = i

    for r_idx, (lbl, s, e) in enumerate(runs):
        if lbl == "Idle" and (e - s) < idle_confirm_frames:
            has_explicit_idle = (
                explicit_idle_mask is not None and bool(np.any(explicit_idle_mask[s:e]))
            )
            if not has_explicit_idle and r_idx > 0 and runs[r_idx - 1][0] == "Working":
                for k in range(s, e):
                    out[k] = "Working"
    return out


class BehaviorRulesEngine:
    """Classifies telemetry into the 6 core classes: Sitting, Standing, Walking, Away, Idle, Working."""

    def __init__(self, thresholds: CalibratedThresholds) -> None:
        self.thresholds = thresholds

    def classify_dataframe(self, feature_df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict]:
        """Classify every row of an enriched feature dataframe into Posture and Activity."""
        if feature_df.is_empty():
            return feature_df, {}

        required_defaults = {
            "torso_horiz_vel_px_s": 0.0,
            "torso_horiz_span_24f_px": 0.0,
            "rolling_30f_left_wrist_vel_px_s": np.nan,
            "rolling_30f_right_wrist_vel_px_s": np.nan,
            "min_wrist_to_ear_norm": np.nan,
            "ear_wrist_rel_std_norm": 1.0,
            "min_wrist_to_face_norm": np.nan,
            "face_wrist_vert_ptp_norm": 0.0,
            "head_pitch_ratio": np.nan,
            "head_forward_sym_norm": np.nan,
            "head_gaze_std_norm": np.nan,
        }
        missing_cols = [
            pl.lit(val).alias(c) for c, val in required_defaults.items() if c not in feature_df.columns
        ]
        if missing_cols:
            feature_df = feature_df.with_columns(missing_cols)

        df_indexed = feature_df.with_row_index("_row_id")
        n_rows = len(df_indexed)

        presence_state = ["Away"] * n_rows
        camera_zone = ["-"] * n_rows
        posture_state = ["Away"] * n_rows
        activity_state = ["Away"] * n_rows
        motion_state = ["Away"] * n_rows

        posture_flickers_suppressed = 0
        activity_flickers_suppressed = 0

        present_df = df_indexed.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
            ["video_name", "track_id", "frame"]
        )

        for _, group in present_df.group_by(["video_name", "track_id"], maintain_order=True):
            r_ids = group["_row_id"].to_numpy()
            m = len(r_ids)
            fps_med = float(group["fps"].drop_nulls().median() or self.thresholds.fps or 30.0)

            sh_x = group["avg_shoulder_x"].to_numpy()
            dist_bot = group["shoulder_dist_bottom_px"].to_numpy()
            horiz_vel = np.nan_to_num(group["torso_horiz_vel_px_s"].to_numpy(), nan=0.0)
            horiz_span = np.nan_to_num(group["torso_horiz_span_24f_px"].to_numpy(), nan=0.0)

            wrist_vel = group["rolling_30f_wrist_vel_px_s"].to_numpy()
            ear_dist = group["min_wrist_to_ear_norm"].to_numpy()
            ear_std = np.nan_to_num(group["ear_wrist_rel_std_norm"].to_numpy(), nan=1.0)
            face_dist = group["min_wrist_to_face_norm"].to_numpy()
            face_ptp = np.nan_to_num(group["face_wrist_vert_ptp_norm"].to_numpy(), nan=0.0)

            pitch_r = group["head_pitch_ratio"].to_numpy()
            fwd_sym = group["head_forward_sym_norm"].to_numpy()
            gaze_std = group["head_gaze_std_norm"].to_numpy()

            # Select perspective zone consistently per track based on median seated/workstation X
            if (
                self.thresholds.use_perspective_zones
                and self.thresholds.perspective_split_x_px is not None
                and np.any(np.isfinite(sh_x))
            ):
                stationary_x = sh_x[np.isfinite(sh_x) & (horiz_vel < 45.0)]
                med_x = float(np.median(stationary_x)) if len(stationary_x) >= 10 else float(np.nanmedian(sh_x))
                matched_zone = "global"
                for z_name, z_cfg in self.thresholds.posture_zones.items():
                    if z_name == "global":
                        continue
                    if z_cfg.x_min_px <= med_x <= z_cfg.x_max_px:
                        matched_zone = z_name
                        break
            else:
                matched_zone = "global"

            z_cfg = self.thresholds.posture_zones.get(
                matched_zone, self.thresholds.posture_zones["global"]
            )

            # --- 1. Spatial Posture: Sitting vs Standing vs Walking ---
            # Account for close-camera foreground perspective: when an upright person steps right next to
            # the foreground lens, their torso spine_length_px expands >1.5x above seated workstation baseline.
            dist_bot_eff = dist_bot.copy()
            if "spine_length_px" in group.columns:
                spine_arr = group["spine_length_px"].to_numpy()
                l_hc = np.nan_to_num(group["left_hip_conf"].to_numpy(), nan=0.0) if "left_hip_conf" in group.columns else np.ones(m)
                r_hc = np.nan_to_num(group["right_hip_conf"].to_numpy(), nan=0.0) if "right_hip_conf" in group.columns else np.ones(m)
                valid_hips = (l_hc >= 0.30) & (r_hc >= 0.30) & np.isfinite(spine_arr)
                seated_sp = spine_arr[valid_hips & np.isfinite(dist_bot) & (dist_bot <= z_cfg.exit_to_sitting_px)]
                if len(seated_sp) >= 20:
                    seated_sp_med = float(np.median(seated_sp))
                    closeup_upright = valid_hips & (spine_arr >= 1.50 * seated_sp_med)
                    dist_bot_eff = np.where(
                        closeup_upright,
                        np.maximum(dist_bot, z_cfg.enter_standing_px + 25.0),
                        dist_bot,
                    )

            p_base, p_sup = apply_hysteresis_and_dwell(
                values=dist_bot_eff,
                low_thresh=z_cfg.exit_to_sitting_px,
                high_thresh=z_cfg.enter_standing_px,
                low_label="Sitting",
                high_label="Standing",
                min_dwell_frames=self.thresholds.min_dwell_frames,
            )
            posture_flickers_suppressed += p_sup

            walk_hi = self.thresholds.walking_horiz_vel_threshold_px_s
            walk_lo = 0.80 * walk_hi
            w_vel_labels, _ = apply_hysteresis_and_dwell(
                values=horiz_vel,
                low_thresh=walk_lo,
                high_thresh=walk_hi,
                low_label="Stationary",
                high_label="Walking",
                min_dwell_frames=min(8, max(self.thresholds.min_dwell_frames // 2, 3)),
            )

            # Upright Standing locomotion uses moderate horizontal span (catches doorway & room walking),
            # whereas Seated posture only flips to Walking under large whole-room traversal (blocks chair scoots!)
            standing_walk_span = min(45.0, max(1.8 * float(m), 25.0))
            foreground_walk_span = min(165.0, max(3.5 * float(m), 90.0))
            raw_posture = [
                "Walking"
                if (
                    (p_base[i] == "Standing" and w_vel_labels[i] == "Walking" and horiz_span[i] >= standing_walk_span)
                    or (horiz_vel[i] >= 145.0 and horiz_span[i] >= foreground_walk_span)
                )
                else p_base[i]
                for i in range(m)
            ]
            p_final = _suppress_short_runs(
                raw_posture, min(8, max(self.thresholds.min_dwell_frames // 2, 3))
            )

            # --- 2. Workstation Activity: Working vs Idle ---
            not_walking = np.array([p != "Walking" for p in p_final], dtype=bool)

            # Working W1 & W2: Bilateral typing/mouse OR unilateral rhythmic dominant-wrist writing
            a_wrist, a_sup = apply_hysteresis_and_dwell(
                values=wrist_vel,
                low_thresh=self.thresholds.exit_to_idle_vel_px_s,
                high_thresh=self.thresholds.enter_working_vel_px_s,
                low_label="Idle",
                high_label="Working",
                min_dwell_frames=self.thresholds.min_dwell_frames,
            )
            activity_flickers_suppressed += a_sup

            # Working W3: One wrist static in close proximity to ear/head for extended duration (Phone Call)
            phone_cand = (
                not_walking
                & np.isfinite(ear_dist)
                & (ear_dist <= self.thresholds.phone_ear_dist_norm_threshold)
                & (ear_std <= 0.09)
            )
            phone_min_frames = min(max(int(round(1.2 * fps_med)), 15), m)
            phone_mask = _filter_min_true_runs(phone_cand, min_len=phone_min_frames)

            # Idle I2: Wrist repeatedly transitions between desk plane and face keypoint (Drinking)
            drink_sip_peak = (
                not_walking
                & (~phone_mask)
                & np.isfinite(face_dist)
                & (face_dist <= self.thresholds.drinking_face_dist_norm_threshold)
                & (face_ptp >= 0.55)
            )
            drink_mask = _dilate_boolean_mask(drink_sip_peak, half_width=10) & not_walking & (~phone_mask)
            drink_mask = _filter_min_true_runs(drink_mask, min_len=min(8, m))

            # Working W4: Wrists idle while nose/eye keypoints tilt downward or remain strictly locked forward
            focus_cand = np.zeros(m, dtype=bool)
            for i in range(m):
                if (
                    not_walking[i]
                    and not phone_mask[i]
                    and not drink_mask[i]
                    and a_wrist[i] == "Idle"
                    and np.isfinite(pitch_r[i])
                ):
                    tilted_down = pitch_r[i] >= 0.22 and np.nan_to_num(gaze_std[i], nan=0.0) <= 0.09
                    locked_forward = (
                        pitch_r[i] >= 0.17
                        and np.isfinite(fwd_sym[i])
                        and fwd_sym[i] <= 0.14
                        and np.isfinite(gaze_std[i])
                        and gaze_std[i] <= 0.05
                    )
                    if tilted_down or locked_forward:
                        focus_cand[i] = True
            focus_mask = _filter_min_true_runs(focus_cand, min_len=min(self.thresholds.min_dwell_frames, m))

            raw_work_states: List[str] = []
            for i in range(m):
                if drink_mask[i]:
                    raw_work_states.append("Idle")
                elif a_wrist[i] == "Working" or phone_mask[i] or focus_mask[i]:
                    raw_work_states.append("Working")
                else:
                    raw_work_states.append("Idle")

            # Idle I1: Require prolonged (>5s) zero/near-zero velocity before flipping Working -> Idle
            idle_confirm_frames = max(int(round(self.thresholds.idle_confirm_seconds * fps_med)), 1)
            confirmed_work = _enforce_idle_confirmation(
                raw_work_states,
                idle_confirm_frames=idle_confirm_frames,
                explicit_idle_mask=drink_mask,
            )
            confirmed_work = _suppress_short_runs(confirmed_work, min(10, self.thresholds.min_dwell_frames))

            for i, rid in enumerate(r_ids):
                r_int = int(rid)
                p_lbl = p_final[i]
                presence_state[r_int] = "Present"
                camera_zone[r_int] = matched_zone
                posture_state[r_int] = p_lbl

                if p_lbl == "Walking":
                    activity_state[r_int] = "Walking"
                    motion_state[r_int] = "Walking"
                else:
                    a_lbl = confirmed_work[i]
                    activity_state[r_int] = a_lbl
                    motion_state[r_int] = f"{p_lbl} | {a_lbl}"

        classified = (
            df_indexed.with_columns(
                [
                    pl.Series("presence_state", presence_state),
                    pl.Series("camera_zone", camera_zone),
                    pl.Series("posture_state", posture_state),
                    pl.Series("activity_state", activity_state),
                    pl.Series("motion_state", motion_state),
                ]
            )
            .sort(["video_name", "frame", "track_id"], nulls_last=True)
            .drop("_row_id")
        )

        def _dist_dict(col_name: str) -> Dict[str, int]:
            rows = classified.group_by(col_name).len().sort("len", descending=True).to_dicts()
            return {str(d[col_name]): int(d["len"]) for d in rows}

        p_dist = _dist_dict("posture_state")
        a_dist = _dist_dict("activity_state")
        six_class_counts = {
            "Sitting": p_dist.get("Sitting", 0),
            "Standing": p_dist.get("Standing", 0),
            "Walking": p_dist.get("Walking", 0),
            "Away": p_dist.get("Away", 0),
            "Working": a_dist.get("Working", 0),
            "Idle": a_dist.get("Idle", 0),
        }

        metrics = {
            "total_rows": int(len(classified)),
            "unique_videos": int(classified["video_name"].n_unique()),
            "unique_tracks": int(classified.filter(pl.col("presence") == 1)["track_id"].n_unique()),
            "posture_distribution_rows": p_dist,
            "activity_distribution_rows": a_dist,
            "motion_state_distribution_rows": _dist_dict("motion_state"),
            "six_class_summary": six_class_counts,
            "posture_flickers_suppressed": int(posture_flickers_suppressed),
            "activity_flickers_suppressed": int(activity_flickers_suppressed),
        }
        return classified, metrics


def generate_state_timeline_plot(
    classified_df: pl.DataFrame,
    plots_dir: Path,
) -> Path:
    """Render state timeline chart for each tracked employee across the 6 classes."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "milestone2_state_timelines.png"

    present = classified_df.filter(pl.col("presence") == 1)
    if present.is_empty():
        return out_path

    top_tracks = (
        present.group_by("track_id")
        .len()
        .sort("len", descending=True)
        .head(4)["track_id"]
        .to_list()
    )

    n_tracks = len(top_tracks)
    fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 3.2 * max(n_tracks, 1)), sharex=False)
    if n_tracks == 1:
        axes = [axes]

    posture_map = {"Sitting": 0.0, "Standing": 0.5, "Walking": 1.0}
    activity_map = {"Idle": 0.0, "Working": 1.0, "Walking": 0.5}

    for ax, tid in zip(axes, top_tracks):
        sub = present.filter(pl.col("track_id") == tid).sort("frame")
        t_sec = sub["timestamp_sec"].to_numpy()
        p_vals = np.array([posture_map.get(s, 0.0) for s in sub["posture_state"].to_list()])
        a_vals = np.array([activity_map.get(s, 0.0) for s in sub["activity_state"].to_list()])
        zone = sub["camera_zone"][0]

        ax.step(t_sec, p_vals + 1.35, where="post", color="#8e44ad", linewidth=2.0, label="Posture (Sitting / Standing / Walking)")
        ax.step(t_sec, a_vals, where="post", color="#27ae60", linewidth=2.0, label="Activity (Idle / Working)")
        ax.set_yticks([0.0, 1.0, 1.35, 1.85, 2.35])
        ax.set_yticklabels(["Idle", "Working", "Sitting", "Standing", "Walking"])
        ax.set_ylim(-0.25, 2.65)
        ax.set_title(f"Emp #{tid} ({zone.capitalize()} Zone, {len(sub)} frames) — 6-Class Stabilized Timeline")
        ax.set_xlabel("Timestamp (seconds)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_motion_categories_plot(
    classified_df: pl.DataFrame,
    plots_dir: Path,
) -> Path:
    """Render 2-panel 6-class summary chart: Spatial Posture/Presence vs Workstation Activity."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "motion_categories_breakdown.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    total_rows = max(len(classified_df), 1)

    # Panel 1: Spatial States (Sitting, Standing, Walking, Away)
    p_counts = {str(d["posture_state"]): int(d["len"]) for d in classified_df.group_by("posture_state").len().to_dicts()}
    p_cats = ["Sitting", "Standing", "Walking", "Away"]
    p_vals = [p_counts.get(c, 0) for c in p_cats]
    p_colors = ["#27ae60", "#2980b9", "#8e44ad", "#95a5a6"]
    bars1 = ax1.barh(np.arange(len(p_cats)), p_vals, color=p_colors, alpha=0.88)
    ax1.set_yticks(np.arange(len(p_cats)))
    ax1.set_yticklabels(p_cats, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_title("Spatial Track (Sitting / Standing / Walking / Away)", fontweight="bold")
    ax1.set_xlabel("Telemetry Rows (Frames)")
    ax1.set_xlim(0, max(max(p_vals, default=1) * 1.25, 10))
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars1, p_vals):
        ax1.text(bar.get_width() + max(p_vals, default=1) * 0.02, bar.get_y() + bar.get_height() / 2.0, f"{v} ({100.0 * v / total_rows:.1f}%)", va="center", fontsize=9)

    # Panel 2: Activity Track (Working, Idle, Walking, Away)
    a_counts = {str(d["activity_state"]): int(d["len"]) for d in classified_df.group_by("activity_state").len().to_dicts()}
    a_cats = ["Working", "Idle", "Walking", "Away"]
    a_vals = [a_counts.get(c, 0) for c in a_cats]
    a_colors = ["#16a085", "#e67e22", "#8e44ad", "#95a5a6"]
    bars2 = ax2.barh(np.arange(len(a_cats)), a_vals, color=a_colors, alpha=0.88)
    ax2.set_yticks(np.arange(len(a_cats)))
    ax2.set_yticklabels(a_cats, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_title("Activity Track (Working / Idle / Walking / Away)", fontweight="bold")
    ax2.set_xlabel("Telemetry Rows (Frames)")
    ax2.set_xlim(0, max(max(a_vals, default=1) * 1.25, 10))
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars2, a_vals):
        ax2.text(bar.get_width() + max(a_vals, default=1) * 0.02, bar.get_y() + bar.get_height() / 2.0, f"{v} ({100.0 * v / total_rows:.1f}%)", va="center", fontsize=9)

    plt.suptitle("6-Class Employee Motion Summary (Sitting, Standing, Walking, Away, Idle, Working)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_milestone2_pipeline(
    input_csv_path: Path = TELEMETRY_OUTPUT_DIR / "batch_landmarks.csv",
    output_telemetry_dir: Path = TELEMETRY_OUTPUT_DIR,
    output_plots_dir: Path = PLOTS_OUTPUT_DIR,
    wrist_window_frames: int = 30,
    min_dwell_frames: int = 15,
    thresholds_override_path: Optional[Path] = None,
    enable_perspective_zones: bool = True,
) -> Dict:
    """End-to-end 6-class calibration, classification, and diagnostic plot pipeline."""
    ensure_directories()
    output_telemetry_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    feature_df = load_and_extract_features(input_csv_path, wrist_window_frames=wrist_window_frames)

    if thresholds_override_path is not None and thresholds_override_path.exists():
        thresholds = CalibratedThresholds.load_json(thresholds_override_path)
    else:
        thresholds = calibrate_thresholds_from_features(
            feature_df=feature_df,
            wrist_window_frames=wrist_window_frames,
            min_dwell_frames=min_dwell_frames,
            enable_perspective_zones=enable_perspective_zones,
        )

    thresholds_path = output_telemetry_dir / "calibrated_thresholds.json"
    thresholds.save_json(thresholds_path)

    engine = BehaviorRulesEngine(thresholds=thresholds)
    classified_df, classification_metrics = engine.classify_dataframe(feature_df)

    temporal_csv_path = output_telemetry_dir / "temporal_features.csv"
    classified_df.write_csv(temporal_csv_path, float_precision=4)

    per_video_paths: List[str] = []
    for vid_tuple, v_df in classified_df.group_by("video_name", maintain_order=True):
        vid_name = str(vid_tuple[0])
        vid_stem = Path(vid_name).stem
        v_out = output_telemetry_dir / f"{vid_stem}_classified.csv"
        v_df.write_csv(v_out, float_precision=4)
        per_video_paths.append(str(v_out))

    plot_paths = generate_calibration_plots(
        classified_df=classified_df,
        thresholds=thresholds,
        plots_dir=output_plots_dir,
    )
    timeline_plot_path = generate_state_timeline_plot(
        classified_df=classified_df,
        plots_dir=output_plots_dir,
    )
    categories_plot_path = generate_motion_categories_plot(
        classified_df=classified_df,
        plots_dir=output_plots_dir,
    )

    summary = {
        "input_csv_path": str(input_csv_path),
        "temporal_features_csv_path": str(temporal_csv_path),
        "per_video_classified_csvs": per_video_paths,
        "calibrated_thresholds_json_path": str(thresholds_path),
        "plots": {
            "posture_calibration": str(plot_paths["posture_calibration_plot"]),
            "wrist_velocity_calibration": str(plot_paths["wrist_velocity_calibration_plot"]),
            "state_timelines": str(timeline_plot_path),
            "motion_categories_breakdown": str(categories_plot_path),
        },
        "calibrated_thresholds": thresholds.to_dict(),
        "classification_metrics": classification_metrics,
    }

    summary_path = output_telemetry_dir / "milestone2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
