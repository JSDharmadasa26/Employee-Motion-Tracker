"""Curates a high-precision 17-keypoint YOLOv8-Pose dataset with hard-negative background frames,
desk-barrier occlusion supervision, and fast-motion blur augmentation.
"""

from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from src.config import DEFAULT_INPUT_VIDEO, FINE_TUNE_DATASET_DIR

# Mapping from COCO 17-keypoint index -> MediaPipe 33-landmark index
COCO17_TO_MEDIAPIPE = [
    0,   # 0: nose
    2,   # 1: left_eye
    5,   # 2: right_eye
    7,   # 3: left_ear
    8,   # 4: right_ear
    11,  # 5: left_shoulder
    12,  # 6: right_shoulder
    13,  # 7: left_elbow
    14,  # 8: right_elbow
    15,  # 9: left_wrist
    16,  # 10: right_wrist
    23,  # 11: left_hip
    24,  # 12: right_hip
    25,  # 13: left_knee
    26,  # 14: right_knee
    27,  # 15: left_ankle
    28,  # 16: right_ankle
]


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _extract_tiled_mediapipe_poses(
    frame: np.ndarray,
    mp_pose,
    yolo_teacher: Optional[YOLO] = None,
) -> List[Dict]:
    """Detect all genuine humans in a 1920x1080 frame using overlapping high-res MediaPipe tiles."""
    h, w = frame.shape[:2]
    tiles = [
        (0, 0, int(w * 0.55), h),                 # Left desk zone
        (int(w * 0.30), int(h * 0.10), int(w * 0.85), h),  # Center doorway / aisle zone
        (int(w * 0.55), int(h * 0.15), w, h),     # Right desk zone
        (0, 0, w, h),                             # Full frame fallback
    ]

    candidates: List[Dict] = []
    for tx1, ty1, tx2, ty2 in tiles:
        crop = frame[ty1:ty2, tx1:tx2]
        ch, cw = crop.shape[:2]
        res = mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            continue

        lms = res.pose_landmarks.landmark
        kpts = np.zeros((17, 3), dtype=np.float64)
        for coco_idx, mp_idx in enumerate(COCO17_TO_MEDIAPIPE):
            lm = lms[mp_idx]
            gx = tx1 + lm.x * cw
            gy = ty1 + lm.y * ch
            vis = float(lm.visibility)
            in_bounds = (0.0 <= gx <= w) and (0.0 <= gy <= h)
            if in_bounds and vis >= 0.55:
                v_flag = 2.0
            elif in_bounds and vis >= 0.12:
                v_flag = 1.0  # Occluded behind desk/barrier, still supervised
            else:
                gx, gy, v_flag = 0.0, 0.0, 0.0
            kpts[coco_idx] = [np.clip(gx, 0.0, w), np.clip(gy, 0.0, h), v_flag]

        # Anatomical plausibility gate to eliminate non-human artifacts
        ls_vis = lms[11].visibility
        rs_vis = lms[12].visibility
        head_vis = max(lms[0].visibility, lms[7].visibility, lms[8].visibility)
        if ls_vis < 0.60 or rs_vis < 0.60 or head_vis < 0.55:
            continue

        sh_mid_x = 0.5 * (kpts[5, 0] + kpts[6, 0])
        sh_mid_y = 0.5 * (kpts[5, 1] + kpts[6, 1])
        head_y = kpts[0, 1] if kpts[0, 2] > 0 else min(kpts[3, 1], kpts[4, 1])
        sh_width = float(np.hypot(kpts[5, 0] - kpts[6, 0], kpts[5, 1] - kpts[6, 1]))

        # Head must be above shoulders and shoulder width must be human-sized
        if head_y >= sh_mid_y + 15.0 or sh_width < 25.0 or sh_width > 420.0:
            continue
        if kpts[11, 2] > 0 and kpts[12, 2] > 0:
            hip_mid_y = 0.5 * (kpts[11, 1] + kpts[12, 1])
            if hip_mid_y <= sh_mid_y + 25.0:
                continue

        # Compute tight padded bounding box from valid keypoints
        valid_pts = kpts[kpts[:, 2] > 0][:, :2]
        if len(valid_pts) < 6:
            continue

        pad_x = max(sh_width * 0.42, 28.0)
        pad_top = max(sh_width * 0.55, 35.0)
        pad_bot = max(sh_width * 0.35, 25.0)

        bx1 = float(np.clip(np.min(valid_pts[:, 0]) - pad_x, 0.0, w))
        by1 = float(np.clip(np.min(valid_pts[:, 1]) - pad_top, 0.0, h))
        bx2 = float(np.clip(np.max(valid_pts[:, 0]) + pad_x, 0.0, w))
        by2 = float(np.clip(np.max(valid_pts[:, 1]) + pad_bot, 0.0, h))

        score = float((ls_vis + rs_vis + head_vis) / 3.0)
        candidates.append(
            {
                "bbox": (bx1, by1, bx2, by2),
                "sh_mid": (sh_mid_x, sh_mid_y),
                "kpts": kpts,
                "score": score,
            }
        )

    # Non-maximum suppression across overlapping tiles
    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept: List[Dict] = []
    for cand in candidates:
        duplicate = False
        for existing in kept:
            sh_dist = float(
                np.hypot(
                    cand["sh_mid"][0] - existing["sh_mid"][0],
                    cand["sh_mid"][1] - existing["sh_mid"][1],
                )
            )
            if sh_dist < 130.0 or _box_iou(cand["bbox"], existing["bbox"]) > 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)

    # Fuse high-confidence desk-occluded wrist coordinates from v1 YOLO when matched to a confirmed human
    if yolo_teacher is not None and kept:
        y_res = yolo_teacher.predict(frame, verbose=False, conf=0.35)[0]
        if y_res.boxes is not None and y_res.keypoints is not None and len(y_res.boxes) > 0:
            y_boxes = y_res.boxes.xyxy.cpu().numpy()
            y_kpts = y_res.keypoints.data.cpu().numpy()
            for person in kept:
                for yi in range(len(y_boxes)):
                    ybox = tuple(float(v) for v in y_boxes[yi])
                    if _box_iou(person["bbox"], ybox) >= 0.30:
                        for w_idx in (9, 10):  # left_wrist, right_wrist
                            if y_kpts[yi, w_idx, 2] >= 0.85 and person["kpts"][w_idx, 2] < 2.0:
                                person["kpts"][w_idx, 0] = float(np.clip(y_kpts[yi, w_idx, 0], 0.0, w))
                                person["kpts"][w_idx, 1] = float(np.clip(y_kpts[yi, w_idx, 1], 0.0, h))
                                person["kpts"][w_idx, 2] = 2.0
                        break

    return kept


def _format_yolo_pose_label(instances: List[Dict], img_w: int, img_h: int) -> str:
    """Serialize instances into YOLOv8-Pose normalized label lines (class 0: person, 17x3 keypoints)."""
    lines: List[str] = []
    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        cx = np.clip(((x1 + x2) / 2.0) / img_w, 0.0, 1.0)
        cy = np.clip(((y1 + y2) / 2.0) / img_h, 0.0, 1.0)
        bw = np.clip((x2 - x1) / img_w, 0.001, 1.0)
        bh = np.clip((y2 - y1) / img_h, 0.001, 1.0)

        parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
        for k_idx in range(17):
            kx, ky, kv = inst["kpts"][k_idx]
            if kv <= 0.0:
                parts.append("0.000000 0.000000 0")
            else:
                nx = np.clip(kx / img_w, 0.0, 1.0)
                ny = np.clip(ky / img_h, 0.0, 1.0)
                parts.append(f"{nx:.6f} {ny:.6f} {int(round(kv))}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _apply_barrier_occlusion(frame: np.ndarray, instances: List[Dict], rng: np.random.Generator) -> Tuple[np.ndarray, List[Dict]]:
    """Simulate a desk/partition barrier across lower torso & hips while preserving ground-truth keypoints."""
    aug = frame.copy()
    new_insts: List[Dict] = []
    h, w = aug.shape[:2]

    for inst in instances:
        kpts = inst["kpts"].copy()
        sh_y = 0.5 * (kpts[5, 1] + kpts[6, 1])
        x1, _, x2, y2 = inst["bbox"]

        # Place synthetic barrier starting between mid-chest and waist
        barrier_top = int(np.clip(sh_y + rng.uniform(55.0, 115.0), 0, h - 20))
        barrier_bot = int(np.clip(y2, barrier_top + 20, h))
        bx1 = int(np.clip(x1 - 20, 0, w - 10))
        bx2 = int(np.clip(x2 + 20, bx1 + 10, w))

        # Wood/office desk color tone with subtle texture noise
        wood_bgr = np.array([rng.integers(25, 65), rng.integers(55, 95), rng.integers(95, 145)], dtype=np.uint8)
        patch = np.full((barrier_bot - barrier_top, bx2 - bx1, 3), wood_bgr, dtype=np.uint8)
        noise = rng.integers(-10, 11, size=patch.shape, dtype=np.int16)
        aug[barrier_top:barrier_bot, bx1:bx2] = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Mark joints covered by the synthetic barrier as v=1 (occluded but supervised!)
        for k_idx in range(17):
            kx, ky, kv = kpts[k_idx]
            if kv > 0 and bx1 <= kx <= bx2 and barrier_top <= ky <= barrier_bot:
                kpts[k_idx, 2] = 1.0

        new_insts.append({"bbox": inst["bbox"], "sh_mid": inst["sh_mid"], "kpts": kpts, "score": inst["score"]})
    return aug, new_insts


def _apply_fast_motion_blur(frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply directional horizontal motion blur simulating rapid walking/turning."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / float(kernel_size)
    return cv2.filter2D(frame, -1, kernel)


def build_finetune_dataset(
    video_path: Path = DEFAULT_INPUT_VIDEO,
    dataset_dir: Path = FINE_TUNE_DATASET_DIR,
    yolo_teacher_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Build train/val YOLOv8-Pose dataset covering hard negatives, fast motion, and desk barriers."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video for dataset curation: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Curate frame indices across all critical regimes:
    # 1. Hard negatives (empty room at start 0..36 and end 5450..5870)
    empty_indices = list(range(0, 36, 2)) + list(range(5460, min(total_frames, 5870), 20))
    # 2. Fast entry, rear-view door closing, walking transitions (dense sampling)
    fast_indices = (
        list(range(40, 240, 4))
        + list(range(2170, 2560, 8))
        + list(range(5040, 5435, 10))
    )
    # 3. Seated desk-obstructed work postures across the full session
    desk_indices = list(range(250, 2160, 28)) + list(range(2570, 5030, 28))

    sampled_indices = sorted(set(i for i in (empty_indices + fast_indices + desk_indices) if 0 <= i < total_frames))

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.45,
    )
    yolo_teacher = YOLO(str(yolo_teacher_path)) if (yolo_teacher_path and yolo_teacher_path.exists()) else None
    rng = np.random.default_rng(42)

    train_count = 0
    val_count = 0
    hard_neg_count = 0
    aug_barrier_count = 0
    aug_blur_count = 0

    for idx_order, f_idx in enumerate(sampled_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Known empty-room ranges are enforced as pure hard negatives
        if f_idx < 36 or f_idx >= 5460:
            instances: List[Dict] = []
        else:
            instances = _extract_tiled_mediapipe_poses(frame, mp_pose=mp_pose, yolo_teacher=yolo_teacher)
            # Skip ambiguous non-empty frames where teacher missed a person so we never inject false negatives
            if not instances:
                continue

        is_val = (idx_order % 5 == 0)
        split = "val" if is_val else "train"
        stem = f"frame_{f_idx:05d}"

        # Save resized 640px wide training image (maintaining aspect ratio -> 640x360) for fast I/O & training
        target_w, target_h = 640, 360
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        label_txt = _format_yolo_pose_label(instances, img_w=img_w, img_h=img_h)

        cv2.imwrite(str(dataset_dir / "images" / split / f"{stem}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (dataset_dir / "labels" / split / f"{stem}.txt").write_text(label_txt)

        if len(instances) == 0:
            hard_neg_count += 1
        if is_val:
            val_count += 1
        else:
            train_count += 1

        # Add barrier-occlusion and fast-motion-blur augmentations on training positives
        if not is_val and len(instances) > 0 and (idx_order % 3 == 0):
            bar_frame, bar_insts = _apply_barrier_occlusion(frame, instances, rng=rng)
            bar_resized = cv2.resize(bar_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            bar_lbl = _format_yolo_pose_label(bar_insts, img_w=img_w, img_h=img_h)
            cv2.imwrite(
                str(dataset_dir / "images" / "train" / f"{stem}_barrier.jpg"),
                bar_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            (dataset_dir / "labels" / "train" / f"{stem}_barrier.txt").write_text(bar_lbl)
            train_count += 1
            aug_barrier_count += 1

            blur_frame = _apply_fast_motion_blur(frame, kernel_size=15)
            blur_resized = cv2.resize(blur_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                str(dataset_dir / "images" / "train" / f"{stem}_fastblur.jpg"),
                blur_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            (dataset_dir / "labels" / "train" / f"{stem}_fastblur.txt").write_text(label_txt)
            train_count += 1
            aug_blur_count += 1

    cap.release()
    mp_pose.close()

    yaml_path = dataset_dir / "dataset.yaml"
    yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
"""
    yaml_path.write_text(yaml_content)

    return {
        "dataset_yaml": str(yaml_path),
        "train_images": train_count,
        "val_images": val_count,
        "hard_negative_images": hard_neg_count,
        "barrier_augmented_images": aug_barrier_count,
        "motion_blur_augmented_images": aug_blur_count,
    }
