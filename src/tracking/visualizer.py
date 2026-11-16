"""Video overlay, H.264 video writer, and skeletal rendering for employee motion tracking."""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np

from src.config import SKELETON_EDGES
from src.tracking.landmarks import SmoothedObservation


def create_video_writer(
    output_path: Union[str, Path],
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an MP4 VideoWriter preferring H.264 ('avc1') for macOS/QuickTime/VS Code compatibility."""
    out_str = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_str, fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Failed to initialize VideoWriter for {out_str}")

# Visual palette (BGR)
COLOR_BOX_PRIMARY = (255, 140, 0)      # Deep Sky Blue
COLOR_BOX_COASTED = (128, 128, 128)    # Gray
COLOR_SKELETON_BONE = (0, 255, 255)    # Yellow bone connections
COLOR_JOINT_HIGH_CONF = (0, 255, 0)    # Green (visible joint)
COLOR_JOINT_OCCLUDED = (0, 165, 255)   # Amber/Orange (occluded/desk)
COLOR_JOINT_LOW_CONF = (0, 0, 255)     # Red (low confidence/coasted)


def draw_hud(
    frame: np.ndarray,
    video_name: str,
    frame_idx: int,
    total_frames: int,
    timestamp_sec: float,
    active_employees: int,
    smoother_name: str,
) -> None:
    """Draw a clean semi-transparent HUD banner with tracking status."""
    overlay = frame.copy()
    hud_h, hud_w = 95, 420
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 20, 20), -1)
    # Blend with original for glassmorphism effect
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (100, 100, 100), 1)

    cv2.putText(
        frame,
        f"Video: {video_name} | {smoother_name.upper()} Smoother",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time: {timestamp_sec:.2f}s | Frame {frame_idx}/{total_frames}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    status_txt = (
        f"Employees Tracked: {active_employees}"
        if active_employees > 0
        else "Employees Tracked: 0 (Away)"
    )
    cv2.putText(
        frame,
        status_txt,
        (25, 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 128) if active_employees > 0 else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )


def draw_tracked_employee(
    frame: np.ndarray,
    obs: SmoothedObservation,
) -> None:
    """Draw smoothed bounding box, persistent employee ID, and skeleton joints."""
    x1, y1, x2, y2 = [int(v) for v in obs.bbox]
    is_coasted = obs.tracking_status == "coasted"
    box_color = COLOR_BOX_COASTED if is_coasted else COLOR_BOX_PRIMARY

    # Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Employee ID badge
    badge_text = f"Emp #{obs.track_id}" + (" (coasting)" if is_coasted else "")
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    badge_y1 = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y1 + th + 8), box_color, -1)
    cv2.putText(
        frame,
        badge_text,
        (x1 + 5, badge_y1 + th + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Live classified motion category pill (when available)
    if obs.behavior_label:
        frame_h, frame_w = frame.shape[:2]
        (btw, bth), _ = cv2.getTextSize(obs.behavior_label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        pill_x1 = max(4, min(x1, frame_w - btw - 14))
        pill_y1 = min(max(y1 + 4, badge_y1 + th + 10), frame_h - bth - 10)
        cv2.rectangle(
            frame,
            (pill_x1, pill_y1),
            (pill_x1 + btw + 10, pill_y1 + bth + 8),
            (25, 25, 25),
            -1,
        )
        cv2.rectangle(
            frame,
            (pill_x1, pill_y1),
            (pill_x1 + btw + 10, pill_y1 + bth + 8),
            box_color,
            1,
        )
        cv2.putText(
            frame,
            obs.behavior_label,
            (pill_x1 + 5, pill_y1 + bth + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 255, 240),
            1,
            cv2.LINE_AA,
        )

    # Draw Torso & Arm Skeleton Lines
    lms = obs.landmarks
    for start_joint, end_joint in SKELETON_EDGES:
        if start_joint in lms and end_joint in lms:
            p1 = lms[start_joint]
            p2 = lms[end_joint]
            # Draw line if both endpoints have reasonable confidence
            if p1.conf > 0.05 and p2.conf > 0.05:
                pt1 = (int(p1.x), int(p1.y))
                pt2 = (int(p2.x), int(p2.y))
                cv2.line(frame, pt1, pt2, COLOR_SKELETON_BONE, 2, cv2.LINE_AA)

    # Draw Individual Target Joints
    for joint_name, lm in lms.items():
        if lm.conf < 0.05 and (lm.x == 0 and lm.y == 0):
            continue
        pt = (int(lm.x), int(lm.y))

        # Color-code by confidence: >0.5 green, 0.15-0.5 amber (desk occlusion), <0.15 red
        if lm.conf >= 0.50:
            j_color = COLOR_JOINT_HIGH_CONF
            r = 5
        elif lm.conf >= 0.15:
            j_color = COLOR_JOINT_OCCLUDED
            r = 4
        else:
            j_color = COLOR_JOINT_LOW_CONF
            r = 3

        cv2.circle(frame, pt, r, j_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r + 1, (0, 0, 0), 1, cv2.LINE_AA)


def render_annotated_frame(
    frame: np.ndarray,
    observations: List[SmoothedObservation],
    video_name: str,
    frame_idx: int,
    total_frames: int,
    timestamp_sec: float,
    smoother_name: str = "Kalman",
) -> np.ndarray:
    """Render full frame overlay with HUD and all tracked employees."""
    annotated = frame.copy()
    for obs in observations:
        draw_tracked_employee(annotated, obs)

    draw_hud(
        annotated,
        video_name=video_name,
        frame_idx=frame_idx,
        total_frames=total_frames,
        timestamp_sec=timestamp_sec,
        active_employees=len(observations),
        smoother_name=smoother_name,
    )
    return annotated
