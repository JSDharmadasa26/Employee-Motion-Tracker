"""Mathematical feature engineering for 13-keypoint posture, kinematics, keypoint relationships,
and ergonomic angle analysis across all 12 motion categories.
"""

from pathlib import Path
from typing import Union
import numpy as np
import polars as pl

from src.config import TARGET_LANDMARKS


def _rolling_mean_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of length `window` in O(N) via cumulative sum."""
    m = len(arr)
    if m == 0:
        return np.array([], dtype=np.float64)
    w = max(int(window), 1)
    clean = np.nan_to_num(arr, nan=0.0)
    csum = np.cumsum(np.insert(clean, 0, 0.0))
    ar = np.arange(1, m + 1)
    start_pos = np.maximum(0, ar - w)
    counts = np.maximum(ar - start_pos, 1)
    return (csum[ar] - csum[start_pos]) / counts


def _rolling_2d_std(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling 2D spatial standard deviation sqrt(var(x) + var(y)) in O(N)."""
    m = len(x)
    if m == 0:
        return np.array([], dtype=np.float64)
    w = max(int(window), 2)
    cx = np.nan_to_num(x, nan=0.0)
    cy = np.nan_to_num(y, nan=0.0)

    ar = np.arange(1, m + 1)
    s = np.maximum(0, ar - w)
    cnt = np.maximum(ar - s, 1).astype(np.float64)

    sx = np.cumsum(np.insert(cx, 0, 0.0))
    sx2 = np.cumsum(np.insert(cx * cx, 0, 0.0))
    sy = np.cumsum(np.insert(cy, 0, 0.0))
    sy2 = np.cumsum(np.insert(cy * cy, 0, 0.0))

    mx = (sx[ar] - sx[s]) / cnt
    my = (sy[ar] - sy[s]) / cnt
    vx = np.maximum((sx2[ar] - sx2[s]) / cnt - mx * mx, 0.0)
    vy = np.maximum((sy2[ar] - sy2[s]) / cnt - my * my, 0.0)
    return np.sqrt(vx + vy)


def _sliding_ptp_1d(arr: np.ndarray, half_window: int) -> np.ndarray:
    """Centered sliding peak-to-peak range (max - min) for cyclical transition detection."""
    m = len(arr)
    if m == 0:
        return np.array([], dtype=np.float64)
    hw = max(int(half_window), 1)
    out = np.zeros(m, dtype=np.float64)
    for i in range(m):
        lo = max(0, i - hw)
        hi = min(m, i + hw + 1)
        seg = arr[lo:hi]
        valid = seg[np.isfinite(seg)]
        if len(valid) >= 2:
            out[i] = float(np.max(valid) - np.min(valid))
    return out


def compute_temporal_features(
    df: pl.DataFrame,
    wrist_window_frames: int = 30,
    max_frame_gap: int = 3,
    min_wrist_conf: float = 0.10,
) -> pl.DataFrame:
    """Compute spatial posture, wrist velocity, head/ear/face relationships, and spine/stretch ergonomics.

    Parameters
    ----------
    df : pl.DataFrame
        Milestone 1 landmark telemetry dataframe (supports full 13-landmark schema and 6-landmark fallback).
    wrist_window_frames : int
        Rolling window length in frames for wrist velocity smoothing (default: 30).
    max_frame_gap : int
        Maximum frame step allowed before treating as a track discontinuity (default: 3).
    min_wrist_conf : float
        Minimum confidence required on consecutive frames to accumulate wrist step.

    Returns
    -------
    pl.DataFrame
        Dataframe enriched with geometric, kinematic, relational, and angular features,
        ordered deterministically by (video_name, frame, track_id).
    """
    if df.is_empty():
        return df

    # Ensure all 13 target landmark columns exist (graceful fallback if 6-joint CSV is supplied)
    missing_exprs = []
    for j_name in TARGET_LANDMARKS:
        for suf, default_val in (("x", None), ("y", None), ("conf", 0.0)):
            col_name = f"{j_name}_{suf}"
            if col_name not in df.columns:
                missing_exprs.append(pl.lit(default_val, dtype=pl.Float64).alias(col_name))
    if missing_exprs:
        df = df.with_columns(missing_exprs)

    float_cols = [
        "timestamp_sec",
        "fps",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_conf",
    ] + [f"{j_name}_{suf}" for j_name in TARGET_LANDMARKS for suf in ("x", "y", "conf")]

    cast_exprs = []
    for col in float_cols:
        if col in df.columns:
            if df.schema[col] in (pl.Utf8, pl.String):
                cast_exprs.append(
                    pl.when(pl.col(col).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(col))
                    .cast(pl.Float64, strict=False)
                    .alias(col)
                )
            else:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))

    for int_col in ("frame", "frame_width", "frame_height", "presence", "track_id"):
        if int_col in df.columns:
            if df.schema[int_col] in (pl.Utf8, pl.String):
                cast_exprs.append(
                    pl.when(pl.col(int_col).str.strip_chars() == "")
                    .then(None)
                    .otherwise(pl.col(int_col))
                    .cast(pl.Int64, strict=False)
                    .alias(int_col)
                )
            else:
                cast_exprs.append(pl.col(int_col).cast(pl.Int64, strict=False).alias(int_col))

    df_clean = df.with_columns(cast_exprs).with_row_index("_orig_row_idx")

    # 1. Row-wise spatial posture & spine angle geometry
    avg_sh_x = (pl.col("left_shoulder_x") + pl.col("right_shoulder_x")) / 2.0
    avg_sh_y = (pl.col("left_shoulder_y") + pl.col("right_shoulder_y")) / 2.0
    avg_hip_x = (pl.col("left_hip_x") + pl.col("right_hip_x")) / 2.0
    avg_hip_y = (pl.col("left_hip_y") + pl.col("right_hip_y")) / 2.0

    frame_h_f = pl.max_horizontal(pl.col("frame_height").cast(pl.Float64), pl.lit(1.0))
    sh_dist_bot_px = pl.col("frame_height").cast(pl.Float64) - avg_sh_y
    sh_dist_bot_norm = sh_dist_bot_px / frame_h_f
    hip_dist_bot_px = pl.col("frame_height").cast(pl.Float64) - avg_hip_y

    spine_dx = (avg_sh_x - avg_hip_x).abs()
    spine_dy = avg_hip_y - avg_sh_y
    spine_len_px = ((avg_hip_x - avg_sh_x) ** 2 + (avg_hip_y - avg_sh_y) ** 2).sqrt()
    spine_len_norm = spine_len_px / frame_h_f

    sh_width_px = (
        (pl.col("left_shoulder_x") - pl.col("right_shoulder_x")) ** 2
        + (pl.col("left_shoulder_y") - pl.col("right_shoulder_y")) ** 2
    ).sqrt()

    df_geom = df_clean.with_columns(
        [
            pl.when(pl.col("presence") == 1).then(avg_sh_x).otherwise(None).alias("avg_shoulder_x"),
            pl.when(pl.col("presence") == 1).then(avg_sh_y).otherwise(None).alias("avg_shoulder_y"),
            pl.when(pl.col("presence") == 1).then(avg_hip_x).otherwise(None).alias("avg_hip_x"),
            pl.when(pl.col("presence") == 1).then(avg_hip_y).otherwise(None).alias("avg_hip_y"),
            pl.when(pl.col("presence") == 1).then(sh_dist_bot_px).otherwise(None).alias("shoulder_dist_bottom_px"),
            pl.when(pl.col("presence") == 1).then(sh_dist_bot_norm).otherwise(None).alias("shoulder_dist_bottom_norm"),
            pl.when(pl.col("presence") == 1).then(hip_dist_bot_px).otherwise(None).alias("hip_dist_bottom_px"),
            pl.when(pl.col("presence") == 1).then(spine_len_px).otherwise(None).alias("spine_length_px"),
            pl.when(pl.col("presence") == 1).then(spine_len_norm).otherwise(None).alias("spine_length_norm"),
            pl.when(pl.col("presence") == 1).then(sh_width_px).otherwise(None).alias("shoulder_width_px"),
            pl.when(pl.col("presence") == 1)
            .then((spine_dx / pl.max_horizontal(spine_dy, pl.lit(1.0))).arctan() * (180.0 / np.pi))
            .otherwise(None)
            .alias("spine_angle_deg"),
        ]
    )

    # 2. Per-track temporal velocity, bilateral wrist variance, head/ear/face relationships, & ergonomic ratios
    n_rows = len(df_geom)
    lw_vel_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    rw_vel_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_lw_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_rw_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_px_frame = np.full(n_rows, np.nan, dtype=np.float64)
    roll_wrist_norm_s = np.full(n_rows, np.nan, dtype=np.float64)

    torso_horiz_vel_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    torso_horiz_span_24f_px = np.full(n_rows, np.nan, dtype=np.float64)
    bilateral_wrist_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    dom_wrist_rel_std_norm = np.full(n_rows, np.nan, dtype=np.float64)
    min_wrist_to_ear_norm = np.full(n_rows, np.nan, dtype=np.float64)
    ear_wrist_rel_std_norm = np.full(n_rows, np.nan, dtype=np.float64)
    min_wrist_to_face_norm = np.full(n_rows, np.nan, dtype=np.float64)
    face_wrist_vert_ptp_norm = np.full(n_rows, np.nan, dtype=np.float64)
    head_pitch_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    head_forward_sym_norm = np.full(n_rows, np.nan, dtype=np.float64)
    head_gaze_std_norm = np.full(n_rows, np.nan, dtype=np.float64)
    torso_compression_ratio = np.full(n_rows, np.nan, dtype=np.float64)
    both_wrists_above_sh = np.zeros(n_rows, dtype=np.int64)

    present_df = df_geom.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
        ["video_name", "track_id", "frame"]
    )

    window = max(int(wrist_window_frames), 1)

    for _, group in present_df.group_by(["video_name", "track_id"], maintain_order=True):
        idxs = group["_orig_row_idx"].to_numpy()
        frames = group["frame"].to_numpy()
        fps_arr = np.nan_to_num(group["fps"].to_numpy(), nan=30.0)
        fps_arr = np.where(fps_arr > 0, fps_arr, 30.0)
        h_arr = np.maximum(group["frame_height"].to_numpy().astype(np.float64), 1.0)

        sh_x = np.nan_to_num(group["avg_shoulder_x"].to_numpy(), nan=0.0)
        sh_y = np.nan_to_num(group["avg_shoulder_y"].to_numpy(), nan=0.0)
        hip_x = np.nan_to_num(group["avg_hip_x"].to_numpy(), nan=sh_x)
        hip_y = np.nan_to_num(group["avg_hip_y"].to_numpy(), nan=sh_y + 0.22 * h_arr)
        l_hc = np.nan_to_num(group["left_hip_conf"].to_numpy(), nan=0.0)
        r_hc = np.nan_to_num(group["right_hip_conf"].to_numpy(), nan=0.0)
        has_hips = (l_hc >= 0.12) & (r_hc >= 0.12)

        l_sh_y = np.nan_to_num(group["left_shoulder_y"].to_numpy(), nan=sh_y)
        r_sh_y = np.nan_to_num(group["right_shoulder_y"].to_numpy(), nan=sh_y)
        sh_w = np.nan_to_num(group["shoulder_width_px"].to_numpy(), nan=0.0)
        # Robust reference scale S (shoulder width with safe floor)
        scale_s = np.maximum(np.maximum(sh_w, 0.10 * h_arr), 35.0)

        # Torso horizontal speed for Walking / Transitioning
        torso_x = np.where(has_hips, 0.5 * (sh_x + hip_x), sh_x)
        m = len(idxs)
        horiz_v = np.zeros(m, dtype=np.float64)
        if m > 1:
            walk_win = min(15, m - 1)
            for k in range(m):
                k0 = max(0, k - walk_win)
                f_span = frames[k] - frames[k0]
                if 1 <= f_span <= walk_win * max_frame_gap:
                    horiz_v[k] = (abs(torso_x[k] - torso_x[k0]) / float(f_span)) * fps_arr[k]
                elif k > 0 and 1 <= (frames[k] - frames[k - 1]) <= max_frame_gap:
                    horiz_v[k] = abs(torso_x[k] - torso_x[k - 1]) * fps_arr[k]
            if m > 1 and horiz_v[0] == 0.0:
                horiz_v[0] = horiz_v[1]
        horiz_span_24 = _sliding_ptp_1d(torso_x, half_window=12)

        # Wrists
        lwx = np.nan_to_num(group["left_wrist_x"].to_numpy(), nan=0.0)
        lwy = np.nan_to_num(group["left_wrist_y"].to_numpy(), nan=0.0)
        lwc = np.nan_to_num(group["left_wrist_conf"].to_numpy(), nan=0.0)

        rwx = np.nan_to_num(group["right_wrist_x"].to_numpy(), nan=0.0)
        rwy = np.nan_to_num(group["right_wrist_y"].to_numpy(), nan=0.0)
        rwc = np.nan_to_num(group["right_wrist_conf"].to_numpy(), nan=0.0)

        # Compute torso-stabilized wrist trajectories when walking, raw wrist trajectories when at desk
        # (At desk sh_x, sh_y are virtually static so stabilized and raw speeds match within <0.5 px/s,
        # while subtracting whole-body translation prevents walking from masquerading as wrist typing)
        walk_blend = np.clip((horiz_v - 35.0) / 35.0, 0.0, 1.0)
        inst_lv = np.zeros(m, dtype=np.float64)
        inst_rv = np.zeros(m, dtype=np.float64)

        if m > 1:
            df_step = np.diff(frames)
            valid_gap = (df_step >= 1) & (df_step <= max_frame_gap)
            valid_l = valid_gap & (lwc[1:] >= min_wrist_conf) & (lwc[:-1] >= min_wrist_conf)
            valid_r = valid_gap & (rwc[1:] >= min_wrist_conf) & (rwc[:-1] >= min_wrist_conf)

            dl_raw = np.hypot(np.diff(lwx), np.diff(lwy))
            dr_raw = np.hypot(np.diff(rwx), np.diff(rwy))

            dl_rel = np.hypot(np.diff(lwx - sh_x), np.diff(lwy - sh_y))
            dr_rel = np.hypot(np.diff(rwx - sh_x), np.diff(rwy - sh_y))

            wb = walk_blend[1:]
            dl = (1.0 - wb) * dl_raw + wb * dl_rel
            dr = (1.0 - wb) * dr_raw + wb * dr_rel

            inst_lv[1:] = np.where(valid_l, (dl / np.maximum(df_step, 1)) * fps_arr[1:], 0.0)
            inst_rv[1:] = np.where(valid_r, (dr / np.maximum(df_step, 1)) * fps_arr[1:], 0.0)

            inst_lv[0] = inst_lv[1]
            inst_rv[0] = inst_rv[1]

        inst_dom = np.maximum(inst_lv, inst_rv)
        rl = _rolling_mean_1d(inst_lv, window)
        rr = _rolling_mean_1d(inst_rv, window)
        rd = _rolling_mean_1d(inst_dom, window)

        # Bilateral wrist activity ratio & localized torso-relative amplitude
        bi_ratio = np.minimum(rl, rr) / np.maximum(np.maximum(rl, rr), 1e-3)
        lw_rel_std = _rolling_2d_std(lwx - sh_x, lwy - sh_y, window) / scale_s
        rw_rel_std = _rolling_2d_std(rwx - sh_x, rwy - sh_y, window) / scale_s
        dom_rel_std = np.where(rl >= rr, lw_rel_std, rw_rel_std)

        # Head / Ear / Nose landmarks for Secondary Behaviors
        nx = np.nan_to_num(group["nose_x"].to_numpy(), nan=0.0)
        ny = np.nan_to_num(group["nose_y"].to_numpy(), nan=0.0)
        nc = np.nan_to_num(group["nose_conf"].to_numpy(), nan=0.0)

        lex = np.nan_to_num(group["left_eye_x"].to_numpy(), nan=0.0)
        ley = np.nan_to_num(group["left_eye_y"].to_numpy(), nan=0.0)
        lec = np.nan_to_num(group["left_eye_conf"].to_numpy(), nan=0.0)

        rex = np.nan_to_num(group["right_eye_x"].to_numpy(), nan=0.0)
        rey = np.nan_to_num(group["right_eye_y"].to_numpy(), nan=0.0)
        rec = np.nan_to_num(group["right_eye_conf"].to_numpy(), nan=0.0)

        leax = np.nan_to_num(group["left_ear_x"].to_numpy(), nan=0.0)
        leay = np.nan_to_num(group["left_ear_y"].to_numpy(), nan=0.0)
        leac = np.nan_to_num(group["left_ear_conf"].to_numpy(), nan=0.0)

        reax = np.nan_to_num(group["right_ear_x"].to_numpy(), nan=0.0)
        reay = np.nan_to_num(group["right_ear_y"].to_numpy(), nan=0.0)
        reac = np.nan_to_num(group["right_ear_conf"].to_numpy(), nan=0.0)

        # Wrist-to-ear proximity (Phone Call)
        inf_arr = np.full(m, 99.0, dtype=np.float64)
        d_lw_lear = np.where((lwc >= 0.12) & (leac >= 0.12), np.hypot(lwx - leax, lwy - leay) / scale_s, inf_arr)
        d_lw_rear = np.where((lwc >= 0.12) & (reac >= 0.12), np.hypot(lwx - reax, lwy - reay) / scale_s, inf_arr)
        d_rw_rear = np.where((rwc >= 0.12) & (reac >= 0.12), np.hypot(rwx - reax, rwy - reay) / scale_s, inf_arr)
        d_rw_lear = np.where((rwc >= 0.12) & (leac >= 0.12), np.hypot(rwx - leax, rwy - leay) / scale_s, inf_arr)

        # Fallback if ear occluded from side angle but nose/eye visible: ear is ~0.35*S lateral from nose
        d_lw_head = np.minimum(
            np.minimum(d_lw_lear, d_lw_rear),
            np.where((lwc >= 0.12) & (nc >= 0.15), 1.15 * np.hypot(lwx - nx, lwy - ny) / scale_s, inf_arr),
        )
        d_rw_head = np.minimum(
            np.minimum(d_rw_rear, d_rw_lear),
            np.where((rwc >= 0.12) & (nc >= 0.15), 1.15 * np.hypot(rwx - nx, rwy - ny) / scale_s, inf_arr),
        )
        min_ear_d = np.minimum(d_lw_head, d_rw_head)
        lw_short_std = _rolling_2d_std(lwx - sh_x, lwy - sh_y, max(window // 2, 12)) / scale_s
        rw_short_std = _rolling_2d_std(rwx - sh_x, rwy - sh_y, max(window // 2, 12)) / scale_s
        ear_w_std = np.where(d_lw_head <= d_rw_head, lw_short_std, rw_short_std)

        # Wrist-to-face/mouth proximity + vertical desk-to-face travel (Drinking)
        mouth_x = nx
        mouth_y = ny + 0.16 * scale_s
        d_lw_face = np.where((lwc >= 0.12) & (nc >= 0.15), np.hypot(lwx - mouth_x, lwy - mouth_y) / scale_s, inf_arr)
        d_rw_face = np.where((rwc >= 0.12) & (nc >= 0.15), np.hypot(rwx - mouth_x, rwy - mouth_y) / scale_s, inf_arr)
        min_face_d = np.minimum(d_lw_face, d_rw_face)

        lw_y_rel = (lwy - sh_y) / scale_s
        rw_y_rel = (rwy - sh_y) / scale_s
        lw_ptp = _sliding_ptp_1d(lw_y_rel, half_window=45)
        rw_ptp = _sliding_ptp_1d(rw_y_rel, half_window=45)
        face_vert_ptp = np.where(d_lw_face <= d_rw_face, lw_ptp, rw_ptp)

        # Head pitch & forward gaze lock (Reading / Screen Focus)
        has_eyes = (lec >= 0.15) & (rec >= 0.15)
        has_ears = (leac >= 0.15) & (reac >= 0.15)
        eye_mid_x = np.where(has_eyes, 0.5 * (lex + rex), sh_x)
        eye_mid_y = np.where(has_eyes, 0.5 * (ley + rey), np.where(has_ears, 0.5 * (leay + reay), ny - 0.12 * scale_s))

        pitch_r = np.where(nc >= 0.18, (ny - eye_mid_y) / scale_s, np.nan)
        fwd_sym = np.where(nc >= 0.18, np.abs(nx - eye_mid_x) / scale_s, np.nan)
        gaze_std = np.where(nc >= 0.18, _rolling_2d_std(nx - sh_x, ny - sh_y, window) / scale_s, np.nan)

        # Torso upright compression ratio (for forward slouch detection alongside spine_angle_deg)
        sp_len = group["spine_length_px"].to_numpy()
        valid_sp = sp_len[np.isfinite(sp_len) & has_hips & (horiz_v < 35.0)]
        if len(valid_sp) >= 10:
            upright_ref = float(np.percentile(valid_sp, 75.0))
        elif np.any(np.isfinite(sp_len)):
            upright_ref = float(np.nanmedian(sp_len))
        else:
            upright_ref = 1.0
        comp_ratio = np.where(has_hips & np.isfinite(sp_len), sp_len / max(upright_ref, 1.0), 1.0)

        # Stretching: both wrists simultaneously elevated above shoulder coordinates
        stretch_flag = (
            (lwc >= 0.15)
            & (rwc >= 0.15)
            & (lwy < l_sh_y - 0.04 * scale_s)
            & (rwy < r_sh_y - 0.04 * scale_s)
        ).astype(np.int64)

        lw_vel_px_s[idxs] = inst_lv
        rw_vel_px_s[idxs] = inst_rv
        roll_lw_px_s[idxs] = rl
        roll_rw_px_s[idxs] = rr
        roll_wrist_px_s[idxs] = rd
        roll_wrist_px_frame[idxs] = rd / fps_arr
        roll_wrist_norm_s[idxs] = rd / h_arr

        torso_horiz_vel_px_s[idxs] = horiz_v
        torso_horiz_span_24f_px[idxs] = horiz_span_24
        bilateral_wrist_ratio[idxs] = bi_ratio
        dom_wrist_rel_std_norm[idxs] = dom_rel_std
        min_wrist_to_ear_norm[idxs] = np.where(min_ear_d < 50.0, min_ear_d, np.nan)
        ear_wrist_rel_std_norm[idxs] = ear_w_std
        min_wrist_to_face_norm[idxs] = np.where(min_face_d < 50.0, min_face_d, np.nan)
        face_wrist_vert_ptp_norm[idxs] = face_vert_ptp
        head_pitch_ratio[idxs] = pitch_r
        head_forward_sym_norm[idxs] = fwd_sym
        head_gaze_std_norm[idxs] = gaze_std
        torso_compression_ratio[idxs] = comp_ratio
        both_wrists_above_sh[idxs] = stretch_flag

    enriched = (
        df_geom.with_columns(
            [
                pl.Series("left_wrist_vel_px_s", lw_vel_px_s).fill_nan(None),
                pl.Series("right_wrist_vel_px_s", rw_vel_px_s).fill_nan(None),
                pl.Series("rolling_30f_left_wrist_vel_px_s", roll_lw_px_s).fill_nan(None),
                pl.Series("rolling_30f_right_wrist_vel_px_s", roll_rw_px_s).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_px_s", roll_wrist_px_s).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_px_frame", roll_wrist_px_frame).fill_nan(None),
                pl.Series("rolling_30f_wrist_vel_norm_s", roll_wrist_norm_s).fill_nan(None),
                pl.Series("torso_horiz_vel_px_s", torso_horiz_vel_px_s).fill_nan(None),
                pl.Series("torso_horiz_span_24f_px", torso_horiz_span_24f_px).fill_nan(None),
                pl.Series("bilateral_wrist_ratio", bilateral_wrist_ratio).fill_nan(None),
                pl.Series("dom_wrist_rel_std_norm", dom_wrist_rel_std_norm).fill_nan(None),
                pl.Series("min_wrist_to_ear_norm", min_wrist_to_ear_norm).fill_nan(None),
                pl.Series("ear_wrist_rel_std_norm", ear_wrist_rel_std_norm).fill_nan(None),
                pl.Series("min_wrist_to_face_norm", min_wrist_to_face_norm).fill_nan(None),
                pl.Series("face_wrist_vert_ptp_norm", face_wrist_vert_ptp_norm).fill_nan(None),
                pl.Series("head_pitch_ratio", head_pitch_ratio).fill_nan(None),
                pl.Series("head_forward_sym_norm", head_forward_sym_norm).fill_nan(None),
                pl.Series("head_gaze_std_norm", head_gaze_std_norm).fill_nan(None),
                pl.Series("torso_compression_ratio", torso_compression_ratio).fill_nan(None),
                pl.Series("both_wrists_above_shoulders", both_wrists_above_sh),
            ]
        )
        .sort(["video_name", "frame", "track_id"], nulls_last=True)
        .drop("_orig_row_idx")
    )
    return enriched


def load_and_extract_features(
    csv_path: Union[str, Path],
    wrist_window_frames: int = 30,
) -> pl.DataFrame:
    """Load a Milestone 1 landmark CSV and compute all mathematical features."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark CSV not found: {path}")
    raw_df = pl.read_csv(path, infer_schema_length=5000)
    return compute_temporal_features(raw_df, wrist_window_frames=wrist_window_frames)
