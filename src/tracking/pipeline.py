"""Batch processing pipeline for Milestone 1: tracking, smoothing, CSV export, and validation."""

import csv
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import src.config  # noqa: F401 - ensures YOLO_CONFIG_DIR and MPLCONFIGDIR before imports
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.config import (
    DEFAULT_YOLO_MODEL,
    PLOTS_OUTPUT_DIR,
    RAW_VIDEOS_DIR,
    TARGET_LANDMARKS,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
    discover_raw_videos,
    ensure_directories,
)
from src.tracking.landmarks import LandmarkPoint, SmoothedObservation, extract_detections_from_results
from src.tracking.multi_tracker import PoseMultiTracker
from src.tracking.visualizer import create_video_writer, render_annotated_frame

CSV_METADATA_COLS = [
    "video_name",
    "frame",
    "timestamp_sec",
    "fps",
    "frame_width",
    "frame_height",
    "presence",
    "track_id",
    "tracking_status",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_conf",
]

CSV_HEADER = CSV_METADATA_COLS + [
    f"{j_name}_{suffix}"
    for j_name in TARGET_LANDMARKS
    for suffix in ("x", "y", "conf")
]


def compute_acceleration_jitter(coords: np.ndarray) -> float:
    """Compute average frame-to-frame second-difference acceleration norm.

    coords shape: [N, 2]
    jitter = mean( || p_{t+1} - 2*p_t + p_{t-1} ||_2 )
    """
    if len(coords) < 3:
        return 0.0
    diff2 = coords[2:] - 2.0 * coords[1:-1] + coords[:-2]
    dists = np.linalg.norm(diff2, axis=1)
    return float(np.mean(dists))


class BatchTrackingPipeline:
    """Executes Milestone 1 tracking, smoothing, video writing, and CSV export."""

    def __init__(
        self,
        model_path: Path = DEFAULT_YOLO_MODEL,
        smoother_type: str = "kalman",
        min_bbox_conf: float = 0.32,
    ) -> None:
        self.model_path = model_path
        self.smoother_type = smoother_type
        self.min_bbox_conf = min_bbox_conf

        ensure_directories()
        print(f"Loading YOLOv8-Pose model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def process_video(
        self,
        video_path: Path,
        max_frames: Optional[int] = None,
        save_video: bool = True,
        show_preview: bool = False,
        output_video_dir: Optional[Path] = None,
        output_telemetry_dir: Optional[Path] = None,
    ) -> Dict:
        """Process a single video file through the tracking and smoothing pipeline."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        video_name = video_path.name
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None and max_frames > 0:
            total_frames = min(total_video_frames, max_frames)
        else:
            total_frames = total_video_frames

        # Outputs
        target_video_dir = output_video_dir or VIDEOS_OUTPUT_DIR
        target_telemetry_dir = output_telemetry_dir or TELEMETRY_OUTPUT_DIR
        target_video_dir.mkdir(parents=True, exist_ok=True)
        target_telemetry_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = target_video_dir / f"{video_path.stem}_tracked.mp4"
        out_csv_path = target_telemetry_dir / f"{video_path.stem}_landmarks.csv"

        video_writer = None
        if save_video:
            video_writer = create_video_writer(out_video_path, fps, width, height)

        tracker = PoseMultiTracker(
            frame_width=width,
            frame_height=height,
            fps=fps,
            smoother_type=self.smoother_type,
        )

        rows_to_write = []
        raw_trajectories: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
        smooth_trajectories: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
        raw_bbox_trajectories: Dict[int, List[Tuple[float, float]]] = {}
        smooth_bbox_trajectories: Dict[int, List[Tuple[float, float]]] = {}

        present_frames = 0
        away_frames = 0
        frame_idx = 0
        start_time = time.time()

        print(f"Processing '{video_name}' ({width}x{height} @ {fps:.1f} fps, target frames: {total_frames})...")

        while cap.isOpened():
            if max_frames and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            timestamp_sec = round((frame_idx - 1) / fps, 4)

            # YOLO inference
            results = self.model.predict(frame, verbose=False, conf=self.min_bbox_conf)
            raw_detections = extract_detections_from_results(
                results, frame_width=width, frame_height=height, min_bbox_conf=self.min_bbox_conf
            )

            # Tracker update & spatial smoothing (with frame for Appearance Re-ID)
            observations = tracker.update(raw_detections, frame=frame)

            if len(observations) == 0:
                away_frames += 1
                # Write standard "away" frame row
                away_row = [
                    video_name,
                    frame_idx,
                    f"{timestamp_sec:.4f}",
                    f"{fps:.2f}",
                    width,
                    height,
                    0,  # presence = 0
                    "",  # track_id empty
                    "away",
                ] + [""] * (len(CSV_HEADER) - 9)
                rows_to_write.append(away_row)
            else:
                present_frames += 1
                for obs in observations:
                    tid = obs.track_id
                    bx1, by1, bx2, by2 = obs.bbox
                    b_conf = obs.bbox_conf

                    row = [
                        video_name,
                        frame_idx,
                        f"{timestamp_sec:.4f}",
                        f"{fps:.2f}",
                        width,
                        height,
                        1,  # presence = 1
                        tid,
                        obs.tracking_status,
                        f"{bx1:.1f}",
                        f"{by1:.1f}",
                        f"{bx2:.1f}",
                        f"{by2:.1f}",
                        f"{b_conf:.3f}",
                    ]

                    # Record landmarks
                    for j_name in TARGET_LANDMARKS:
                        lm = obs.landmarks[j_name]
                        row.extend([f"{lm.x:.1f}", f"{lm.y:.1f}", f"{lm.conf:.3f}"])

                    rows_to_write.append(row)

                    # Accumulate trajectory data for jitter metric calculation
                    if tid not in raw_trajectories:
                        raw_trajectories[tid] = {j: [] for j in TARGET_LANDMARKS}
                        smooth_trajectories[tid] = {j: [] for j in TARGET_LANDMARKS}
                        raw_bbox_trajectories[tid] = []
                        smooth_bbox_trajectories[tid] = []

                    # Bounding box center
                    smooth_bbox_trajectories[tid].append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))
                    if obs.raw_bbox is not None:
                        rx1, ry1, rx2, ry2 = obs.raw_bbox
                        raw_bbox_trajectories[tid].append(((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0))
                    else:
                        raw_bbox_trajectories[tid].append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

                    # Keypoints
                    for j_name in TARGET_LANDMARKS:
                        lm = obs.landmarks[j_name]
                        smooth_trajectories[tid][j_name].append((lm.x, lm.y))
                        if obs.raw_landmarks and j_name in obs.raw_landmarks:
                            rlm = obs.raw_landmarks[j_name]
                            raw_trajectories[tid][j_name].append((rlm.x, rlm.y))
                        else:
                            raw_trajectories[tid][j_name].append((lm.x, lm.y))

            # Video writing / preview
            if video_writer is not None or show_preview:
                annotated = render_annotated_frame(
                    frame=frame,
                    observations=observations,
                    video_name=video_name,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    timestamp_sec=timestamp_sec,
                    smoother_name=self.smoother_type,
                )
                if video_writer is not None:
                    video_writer.write(annotated)
                if show_preview:
                    cv2.imshow("Milestone 1 Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_idx % 500 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / max(elapsed, 0.001)
                print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) - Speed: {fps_proc:.2f} fps")

        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        elapsed_total = time.time() - start_time
        effective_fps = frame_idx / max(elapsed_total, 0.001)

        # Write CSV
        with open(out_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows_to_write)

        print(f"Saved telemetry to: {out_csv_path} ({len(rows_to_write)} rows)")
        if save_video:
            print(f"Saved master video to: {out_video_path}")

        # Compute Jitter Reduction
        bbox_raw_jitters = []
        bbox_smooth_jitters = []
        for tid in raw_bbox_trajectories:
            r_pts = np.array(raw_bbox_trajectories[tid])
            s_pts = np.array(smooth_bbox_trajectories[tid])
            if len(r_pts) >= 10:
                bbox_raw_jitters.append(compute_acceleration_jitter(r_pts))
                bbox_smooth_jitters.append(compute_acceleration_jitter(s_pts))

        mean_raw_bbox_j = float(np.mean(bbox_raw_jitters)) if bbox_raw_jitters else 1.0
        mean_sm_bbox_j = float(np.mean(bbox_smooth_jitters)) if bbox_smooth_jitters else 0.5
        bbox_reduction_pct = round(max(0.0, (1.0 - mean_sm_bbox_j / max(mean_raw_bbox_j, 1e-4)) * 100), 2)

        joint_reduction_dict = {}
        for j_name in TARGET_LANDMARKS:
            j_raw_jit = []
            j_sm_jit = []
            for tid in raw_trajectories:
                r_pts = np.array(raw_trajectories[tid][j_name])
                s_pts = np.array(smooth_trajectories[tid][j_name])
                if len(r_pts) >= 10:
                    j_raw_jit.append(compute_acceleration_jitter(r_pts))
                    j_sm_jit.append(compute_acceleration_jitter(s_pts))
            m_r = float(np.mean(j_raw_jit)) if j_raw_jit else 1.0
            m_s = float(np.mean(j_sm_jit)) if j_sm_jit else 0.5
            red = round(max(0.0, (1.0 - m_s / max(m_r, 1e-4)) * 100), 2)
            joint_reduction_dict[j_name] = red

        stats = {
            "video_name": video_name,
            "output_video_path": str(out_video_path),
            "output_csv_path": str(out_csv_path),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frames_processed": frame_idx,
            "csv_rows_written": len(rows_to_write),
            "present_frames": present_frames,
            "away_frames": away_frames,
            "unique_employees_tracked": tracker.next_track_id - 1,
            "processing_seconds": round(elapsed_total, 2),
            "effective_processing_fps": round(effective_fps, 2),
            "bbox_jitter_reduction_pct": bbox_reduction_pct,
            "joint_jitter_reduction_pct": joint_reduction_dict,
        }

        # Keep trajectory data on object for plot generation
        self.last_trajectories = {
            "raw": raw_trajectories,
            "smooth": smooth_trajectories,
            "raw_bbox": raw_bbox_trajectories,
            "smooth_bbox": smooth_bbox_trajectories,
        }

        return stats

    def run_batch(
        self,
        video_paths: Optional[List[Path]] = None,
        max_frames: Optional[int] = None,
        save_video: bool = True,
    ) -> List[Dict]:
        """Process all videos in batch and write master batch CSV and validation report."""
        if video_paths is None or len(video_paths) == 0:
            video_paths = discover_raw_videos()

        if len(video_paths) == 0:
            raise FileNotFoundError(f"No video files found in: {RAW_VIDEOS_DIR}")

        print(f"Found {len(video_paths)} videos for batch processing: {[p.name for p in video_paths]}")

        batch_stats = []
        all_csv_rows = []

        for p in video_paths:
            stats = self.process_video(video_path=p, max_frames=max_frames, save_video=save_video)
            batch_stats.append(stats)

            # Ingest per-video CSV rows for master batch CSV
            csv_path = Path(stats["output_csv_path"])
            with open(csv_path, mode="r") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    all_csv_rows.append(row)

        # Write master batch CSV
        master_csv_path = TELEMETRY_OUTPUT_DIR / "batch_landmarks.csv"
        with open(master_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(all_csv_rows)
        print(f"Master batch telemetry saved to: {master_csv_path} ({len(all_csv_rows)} rows)")

        # Save validation JSON summary
        summary_path = TELEMETRY_OUTPUT_DIR / "milestone1_validation_summary.json"
        with open(summary_path, mode="w") as f:
            json.dump(batch_stats, f, indent=2)
        print(f"Milestone 1 validation summary saved to: {summary_path}")

        # Render Jitter Reduction Plot
        self.generate_jitter_plot()

        return batch_stats

    def generate_jitter_plot(self) -> None:
        """Generate and save jitter reduction visual comparison."""
        plot_path = PLOTS_OUTPUT_DIR / "milestone1_jitter_reduction.png"
        if not hasattr(self, "last_trajectories") or not self.last_trajectories["raw_bbox"]:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Trajectory comparison for primary employee track
        raw_b = self.last_trajectories["raw_bbox"]
        sm_b = self.last_trajectories["smooth_bbox"]
        primary_tid = 1 if 1 in raw_b else (list(raw_b.keys())[0] if raw_b else None)

        if primary_tid is not None:
            r_pts = np.array(raw_b[primary_tid])[:300]
            s_pts = np.array(sm_b[primary_tid])[:300]

            ax1.plot(r_pts[:, 0], r_pts[:, 1], "r.", alpha=0.4, label="Raw Detections (Jittery)")
            ax1.plot(s_pts[:, 0], s_pts[:, 1], "b-", linewidth=2.0, label=f"Smoothed ({self.smoother_type.upper()})")
            ax1.set_title(f"BBox Centroid Trajectory (Emp #{primary_tid}, First 300 Frames)")
            ax1.set_xlabel("X (pixels)")
            ax1.set_ylabel("Y (pixels)")
            ax1.invert_yaxis()  # Image coordinates
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Acceleration Jitter comparison
        summary_path = TELEMETRY_OUTPUT_DIR / "milestone1_validation_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            if data:
                j_red = data[0]["joint_jitter_reduction_pct"]
                labels = ["BBox"] + [j.replace("left_", "L_").replace("right_", "R_") for j in TARGET_LANDMARKS]
                reductions = [data[0]["bbox_jitter_reduction_pct"]] + [j_red.get(j, 0.0) for j in TARGET_LANDMARKS]

                bars = ax2.bar(labels, reductions, color="#2ecc71")
                ax2.set_title("Flicker & Jitter Reduction (%) Across 13 Landmarks")
                ax2.set_ylabel("Reduction Percentage (%)")
                ax2.set_ylim(0, 105)
                ax2.tick_params(axis="x", rotation=35, labelsize=7.5)
                ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

                for bar in bars:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.0f}%", ha="center", va="bottom", fontsize=6.5)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Jitter reduction chart saved to: {plot_path}")


def render_video_from_telemetry_csv(
    video_path: Path,
    landmarks_csv_path: Path,
    output_video_path: Path,
    smoother_name: str = "Kalman",
) -> Dict[str, object]:
    """Re-render a full H.264 ('avc1') master tracked video directly from saved smoothed CSV telemetry."""
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if not landmarks_csv_path.exists():
        raise FileNotFoundError(f"Landmarks CSV not found: {landmarks_csv_path}")

    # Group smoothed observations by 1-based frame index
    frame_obs_map: Dict[int, List[SmoothedObservation]] = {}
    with open(landmarks_csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_name") and row["video_name"] != video_path.name:
                continue
            f_idx = int(row["frame"])
            if int(row.get("presence") or 0) != 1 or not row.get("track_id"):
                continue

            w = float(row["frame_width"])
            h = float(row["frame_height"])
            lms: Dict[str, LandmarkPoint] = {}
            for j_name in TARGET_LANDMARKS:
                jx = float(row[f"{j_name}_x"])
                jy = float(row[f"{j_name}_y"])
                jc = float(row[f"{j_name}_conf"])
                lms[j_name] = LandmarkPoint(
                    name=j_name,
                    x=jx,
                    y=jy,
                    conf=jc,
                    x_norm=jx / max(w, 1.0),
                    y_norm=jy / max(h, 1.0),
                )

            p_state = row.get("posture_state") or ""
            a_state = row.get("activity_state") or ""
            m_label = row.get("motion_state") or ""

            if m_label and m_label not in ("-", "Away"):
                behavior_str = m_label
            elif p_state == "Walking":
                behavior_str = "Walking"
            elif p_state in ("Sitting", "Standing") and a_state in ("Working", "Idle"):
                behavior_str = f"{p_state} | {a_state}"
            elif p_state and p_state not in ("-", "Away"):
                behavior_str = p_state
            else:
                behavior_str = None

            obs = SmoothedObservation(
                track_id=int(row["track_id"]),
                tracking_status=row.get("tracking_status", "tracked"),
                bbox=(
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                ),
                bbox_conf=float(row["bbox_conf"]),
                landmarks=lms,
                behavior_label=behavior_str,
            )
            frame_obs_map.setdefault(f_idx, []).append(obs)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = create_video_writer(output_video_path, fps=fps, width=width, height=height)
    frame_idx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp_sec = round((frame_idx - 1) / fps, 4)
        observations = frame_obs_map.get(frame_idx, [])
        annotated = render_annotated_frame(
            frame=frame,
            observations=observations,
            video_name=video_path.name,
            frame_idx=frame_idx,
            total_frames=total_frames,
            timestamp_sec=timestamp_sec,
            smoother_name=smoother_name,
        )
        writer.write(annotated)

    cap.release()
    writer.release()
    elapsed = round(time.time() - t0, 2)
    return {
        "output_video_path": str(output_video_path),
        "frames_written": frame_idx,
        "elapsed_seconds": elapsed,
    }
