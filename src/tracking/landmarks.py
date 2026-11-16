"""Keypoint and bounding box extraction data structures with pose-aware NMS deduplication."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import KEYPOINT_INDICES, TARGET_LANDMARKS


@dataclass
class LandmarkPoint:
    """Single landmark point in pixel and normalized coordinates."""
    name: str
    x: float
    y: float
    conf: float
    x_norm: float = 0.0
    y_norm: float = 0.0


@dataclass
class DetectionRaw:
    """Raw bounding box and keypoints detected by YOLOv8-Pose in one person instance."""
    bbox: Tuple[float, float, float, float]
    bbox_conf: float
    class_id: int
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)


@dataclass
class SmoothedObservation:
    """Smoothed bounding box, target landmarks, and optional live behavior tags for an active track."""
    track_id: int
    tracking_status: str  # "tracked" or "coasted"
    bbox: Tuple[float, float, float, float]
    bbox_conf: float
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)
    raw_bbox: Optional[Tuple[float, float, float, float]] = None
    raw_landmarks: Optional[Dict[str, LandmarkPoint]] = None
    behavior_label: Optional[str] = None


def _box_iou_and_containment(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Return (IoU, max_intersection_over_smaller_area) between two boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 1e-6)
    area_b = max((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]), 1e-6)
    iou = inter / max(area_a + area_b - inter, 1e-6)
    contain = inter / min(area_a, area_b)
    return iou, contain


def is_plausible_human_pose(
    bbox: Tuple[float, float, float, float],
    landmarks: Dict[str, LandmarkPoint],
    frame_height: int = 1080,
) -> bool:
    """Reject degenerate non-human or bottom-border fragment detections using skeletal anatomy."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 35.0 or bh < 85.0:
        return False

    # Reject tiny bottom-edge fragments (e.g. top of head/bag barely peeking at bottom of frame)
    if frame_height > 200 and y1 > 0.86 * frame_height:
        return False

    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    if ls is None or rs is None:
        return True

    if ls.conf < 0.20 and rs.conf < 0.20:
        return False

    if frame_height > 200:
        sh_y = 0.5 * (ls.y + rs.y)
        if sh_y > frame_height - 25.0:
            return False

    nose = landmarks.get("nose")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    if nose and lh and rh and nose.conf > 0.35 and lh.conf > 0.25 and rh.conf > 0.25:
        hip_y = 0.5 * (lh.y + rh.y)
        if nose.y > hip_y + 15.0:
            return False

    return True


def deduplicate_detections(detections: List[DetectionRaw]) -> List[DetectionRaw]:
    """Suppress duplicate boxes/skeletons generated on the same person near frame borders."""
    if len(detections) <= 1:
        return detections

    ordered = sorted(detections, key=lambda d: d.bbox_conf, reverse=True)
    kept: List[DetectionRaw] = []

    for cand in ordered:
        is_dup = False
        c_ls = cand.landmarks.get("left_shoulder")
        c_rs = cand.landmarks.get("right_shoulder")
        c_has_sh = c_ls is not None and c_rs is not None and (c_ls.conf > 0.25 or c_rs.conf > 0.25)
        c_sh_x = 0.5 * (c_ls.x + c_rs.x) if c_has_sh else 0.5 * (cand.bbox[0] + cand.bbox[2])
        c_sh_y = 0.5 * (c_ls.y + c_rs.y) if c_has_sh else 0.5 * (cand.bbox[1] + cand.bbox[3])

        for acc in kept:
            iou, contain = _box_iou_and_containment(cand.bbox, acc.bbox)
            if iou >= 0.42 or contain >= 0.72:
                is_dup = True
                break

            a_ls = acc.landmarks.get("left_shoulder")
            a_rs = acc.landmarks.get("right_shoulder")
            a_has_sh = a_ls is not None and a_rs is not None and (a_ls.conf > 0.25 or a_rs.conf > 0.25)
            if c_has_sh and a_has_sh:
                a_sh_x = 0.5 * (a_ls.x + a_rs.x)
                a_sh_y = 0.5 * (a_ls.y + a_rs.y)
                if float(np.hypot(c_sh_x - a_sh_x, c_sh_y - a_sh_y)) < 85.0:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(cand)

    return kept


def extract_detections_from_results(
    results,
    frame_width: int,
    frame_height: int,
    min_bbox_conf: float = 0.32,
) -> List[DetectionRaw]:
    """Extract deduplicated person bounding boxes and all 13 target landmarks from Ultralytics Results."""
    detections: List[DetectionRaw] = []
    if results is None or len(results) == 0:
        return detections

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return detections

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    has_kpts = res.keypoints is not None and len(res.keypoints.data) > 0
    kpts_data = res.keypoints.data.cpu().numpy() if has_kpts else None

    for i in range(len(boxes)):
        bbox_conf = float(confs[i])
        if bbox_conf < min_bbox_conf:
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        cls_id = int(clss[i])

        landmarks_dict: Dict[str, LandmarkPoint] = {}
        if kpts_data is not None and i < len(kpts_data):
            inst_kpts = kpts_data[i]
            for name in TARGET_LANDMARKS:
                idx = KEYPOINT_INDICES[name]
                if idx < len(inst_kpts):
                    kx = float(inst_kpts[idx, 0])
                    ky = float(inst_kpts[idx, 1])
                    kc = float(inst_kpts[idx, 2])
                    kx_norm = min(max(kx / frame_width, 0.0), 1.0) if frame_width > 0 else 0.0
                    ky_norm = min(max(ky / frame_height, 0.0), 1.0) if frame_height > 0 else 0.0
                    landmarks_dict[name] = LandmarkPoint(
                        name=name, x=kx, y=ky, conf=kc, x_norm=kx_norm, y_norm=ky_norm
                    )
                else:
                    landmarks_dict[name] = LandmarkPoint(name=name, x=0.0, y=0.0, conf=0.0)
        else:
            for name in TARGET_LANDMARKS:
                landmarks_dict[name] = LandmarkPoint(name=name, x=0.0, y=0.0, conf=0.0)

        if not is_plausible_human_pose((x1, y1, x2, y2), landmarks_dict, frame_height=frame_height):
            continue

        detections.append(
            DetectionRaw(
                bbox=(x1, y1, x2, y2),
                bbox_conf=bbox_conf,
                class_id=cls_id,
                landmarks=landmarks_dict,
            )
        )

    return deduplicate_detections(detections)
