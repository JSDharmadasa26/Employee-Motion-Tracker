"""Multi-person tracking with velocity-predicted Hungarian matching, barrier-occlusion memory,
and torso HSV+Lab Appearance Re-Identification (Re-ID) gallery for persistent Employee IDs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import TARGET_LANDMARKS
from src.tracking.landmarks import DetectionRaw, LandmarkPoint, SmoothedObservation
from src.tracking.smoother import (
    KalmanBBoxSmoother,
    KalmanPoint2DSmoother,
    RollingAverageSmoother,
)


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    box_a_area = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union_area = box_a_area + box_b_area - inter_area
    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def extract_torso_appearance_embedding(
    frame_bgr: Optional[np.ndarray],
    det: DetectionRaw,
) -> Optional[np.ndarray]:
    """Extract L2-normalized illumination-resilient HSV + CIE-Lab torso color histogram.

    Uses only upper-torso clothing pixels (shoulders to hips) — zero facial biometrics stored.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = det.bbox
    lms = det.landmarks

    ls = lms.get("left_shoulder")
    rs = lms.get("right_shoulder")
    lh = lms.get("left_hip")
    rh = lms.get("right_hip")

    if (
        ls is not None
        and rs is not None
        and lh is not None
        and rh is not None
        and ls.conf > 0.30
        and rs.conf > 0.30
        and lh.conf > 0.25
        and rh.conf > 0.25
    ):
        tx1 = int(max(0, min(ls.x, rs.x, lh.x, rh.x) - 8))
        tx2 = int(min(w, max(ls.x, rs.x, lh.x, rh.x) + 8))
        ty1 = int(max(0, min(ls.y, rs.y)))
        ty2 = int(min(h, max(lh.y, rh.y)))
    else:
        bw, bh = x2 - x1, y2 - y1
        tx1 = int(max(0, x1 + 0.22 * bw))
        tx2 = int(min(w, x2 - 0.22 * bw))
        ty1 = int(max(0, y1 + 0.22 * bh))
        ty2 = int(min(h, y1 + 0.68 * bh))

    if tx2 - tx1 < 14 or ty2 - ty1 < 14:
        return None

    patch = cv2.resize(frame_bgr[ty1:ty2, tx1:tx2], (32, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

    h_hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()
    l_hist = cv2.calcHist([lab], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).flatten()

    vec = np.concatenate([h_hist * 1.25, l_hist]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return None
    return vec / norm


@dataclass
class EmployeeGalleryProfile:
    """Persistent appearance & spatial memory for an identified employee across occlusions."""
    track_id: int
    ema_embedding: Optional[np.ndarray] = None
    exemplars: List[np.ndarray] = field(default_factory=list)
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_seen_step: int = 0
    total_hits: int = 0

    def update_appearance(
        self,
        emb: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float],
        step: int,
        conf: float,
    ) -> None:
        self.last_bbox = bbox
        self.last_seen_step = step
        self.total_hits += 1
        if emb is None or conf < 0.50:
            return

        if self.ema_embedding is None:
            self.ema_embedding = emb.copy()
            self.exemplars.append(emb.copy())
            return

        merged = 0.90 * self.ema_embedding + 0.10 * emb
        nrm = float(np.linalg.norm(merged))
        if nrm > 1e-6:
            self.ema_embedding = merged / nrm

        # Store up to 5 diverse lighting/angle exemplars (e.g. doorway light vs desk shade)
        if len(self.exemplars) < 5:
            max_ex_sim = max(float(np.dot(emb, ex)) for ex in self.exemplars)
            if max_ex_sim < 0.92:
                self.exemplars.append(emb.copy())

    def appearance_similarity(self, emb: Optional[np.ndarray]) -> float:
        if emb is None or self.ema_embedding is None:
            return 0.0
        sims = [float(np.dot(emb, self.ema_embedding))]
        for ex in self.exemplars:
            sims.append(float(np.dot(emb, ex)))
        return max(sims)


class SinglePersonTrack:
    """Represents a single persistent employee track across time."""

    def __init__(
        self,
        track_id: int,
        initial_detection: DetectionRaw,
        frame_width: int,
        frame_height: int,
        smoother_type: str = "kalman",
        dt: float = 1.0 / 30.0,
    ) -> None:
        self.track_id = track_id
        self.smoother_type = smoother_type
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dt = dt

        self.time_since_update: int = 0
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 1

        self.last_bbox = initial_detection.bbox
        self.last_bbox_conf = initial_detection.bbox_conf
        self.last_landmarks = initial_detection.landmarks
        self._just_reseeded: bool = False

        if smoother_type == "kalman":
            self.bbox_smoother = KalmanBBoxSmoother(dt=dt)
            self.point_smoothers: Dict[str, KalmanPoint2DSmoother] = {
                name: KalmanPoint2DSmoother(dt=dt) for name in TARGET_LANDMARKS
            }
            self.bbox_smoother.update(*initial_detection.bbox, conf=initial_detection.bbox_conf)
            for name, lm in initial_detection.landmarks.items():
                if name in self.point_smoothers:
                    self.point_smoothers[name].update(lm.x, lm.y, lm.conf)
        else:
            self.rolling_smoother = RollingAverageSmoother(window_size=5)
            self.rolling_smoother.update_bbox(*initial_detection.bbox, conf=initial_detection.bbox_conf)
            for name, lm in initial_detection.landmarks.items():
                self.rolling_smoother.update_point(name, lm.x, lm.y, lm.conf)

    def get_predicted_matching_box(self) -> Tuple[float, float, float, float]:
        """Return kinematic predicted box (when recently updated) or last known box."""
        if self.smoother_type == "kalman" and self.time_since_update <= 6:
            pred = self.bbox_smoother.predict_bbox()
            if pred is not None:
                return pred
        return self.last_bbox

    def update(self, detection: DetectionRaw) -> None:
        was_long_occluded = self.time_since_update > 6
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.age += 1

        self.last_bbox = detection.bbox
        self.last_bbox_conf = detection.bbox_conf
        self.last_landmarks = detection.landmarks

        # If re-acquired after a multi-frame occlusion gap, cleanly re-seed Kalman state at new pose
        if was_long_occluded:
            if self.smoother_type == "kalman":
                self.bbox_smoother.reset()
                self.bbox_smoother.update(*detection.bbox, conf=detection.bbox_conf)
                for name, sm in self.point_smoothers.items():
                    sm.reset()
                    if name in detection.landmarks:
                        lm = detection.landmarks[name]
                        sm.update(lm.x, lm.y, lm.conf)
            else:
                self.rolling_smoother.reset()
                self.rolling_smoother.update_bbox(*detection.bbox, conf=detection.bbox_conf)
                for name, lm in detection.landmarks.items():
                    self.rolling_smoother.update_point(name, lm.x, lm.y, lm.conf)

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.hit_streak = 0
        self.age += 1

    def get_smoothed_observation(self, is_coasted: bool = False) -> SmoothedObservation:
        status = "coasted" if is_coasted else "tracked"
        orig_raw_bbox = self.last_bbox

        if self.smoother_type == "kalman":
            if not is_coasted:
                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = self.bbox_smoother.update(
                    *self.last_bbox, conf=self.last_bbox_conf
                )
            else:
                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = self.bbox_smoother.update(
                    *self.last_bbox, conf=0.0
                )

            smooth_x1 = max(0.0, min(smooth_x1, self.frame_width))
            smooth_y1 = max(0.0, min(smooth_y1, self.frame_height))
            smooth_x2 = max(0.0, min(smooth_x2, self.frame_width))
            smooth_y2 = max(0.0, min(smooth_y2, self.frame_height))
            smoothed_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
            self.last_bbox = smoothed_bbox

            smoothed_landmarks: Dict[str, LandmarkPoint] = {}
            for name in TARGET_LANDMARKS:
                smoother = self.point_smoothers[name]
                if name in self.last_landmarks and not is_coasted:
                    raw_lm = self.last_landmarks[name]
                    sx, sy, sc = smoother.update(raw_lm.x, raw_lm.y, raw_lm.conf)
                else:
                    sx, sy, sc = smoother.update(0.0, 0.0, 0.0)

                sx = max(0.0, min(sx, self.frame_width))
                sy = max(0.0, min(sy, self.frame_height))
                nx = sx / self.frame_width if self.frame_width > 0 else 0.0
                ny = sy / self.frame_height if self.frame_height > 0 else 0.0
                smoothed_landmarks[name] = LandmarkPoint(
                    name=name, x=sx, y=sy, conf=sc, x_norm=nx, y_norm=ny
                )
        else:
            if not is_coasted:
                sb = self.rolling_smoother.update_bbox(*self.last_bbox, conf=self.last_bbox_conf)
            else:
                sb = self.last_bbox

            smoothed_bbox = (
                max(0.0, min(sb[0], self.frame_width)),
                max(0.0, min(sb[1], self.frame_height)),
                max(0.0, min(sb[2], self.frame_width)),
                max(0.0, min(sb[3], self.frame_height)),
            )
            smoothed_landmarks = {}
            for name in TARGET_LANDMARKS:
                if name in self.last_landmarks:
                    raw_lm = self.last_landmarks[name]
                    sx, sy, sc = self.rolling_smoother.update_point(name, raw_lm.x, raw_lm.y, raw_lm.conf)
                else:
                    sx, sy, sc = 0.0, 0.0, 0.0
                sx = max(0.0, min(sx, self.frame_width))
                sy = max(0.0, min(sy, self.frame_height))
                nx = sx / self.frame_width if self.frame_width > 0 else 0.0
                ny = sy / self.frame_height if self.frame_height > 0 else 0.0
                smoothed_landmarks[name] = LandmarkPoint(
                    name=name, x=sx, y=sy, conf=sc, x_norm=nx, y_norm=ny
                )

        return SmoothedObservation(
            track_id=self.track_id,
            tracking_status=status,
            bbox=smoothed_bbox,
            bbox_conf=self.last_bbox_conf,
            landmarks=smoothed_landmarks,
            raw_bbox=orig_raw_bbox,
            raw_landmarks=self.last_landmarks,
        )


class PoseMultiTracker:
    """Multi-person tracker with active kinematic Hungarian matching + Appearance Re-ID Gallery."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 30.0,
        smoother_type: str = "kalman",
        max_age_frames: int = 40,
        max_coast_render_frames: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.18,
        new_track_min_conf: float = 0.42,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.smoother_type = smoother_type
        self.max_age_frames = max_age_frames
        self.max_coast_render_frames = max_coast_render_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.new_track_min_conf = new_track_min_conf

        self.step_idx: int = 0
        self.next_track_id: int = 1
        self.active_tracks: List[SinglePersonTrack] = []
        self.gallery: Dict[int, EmployeeGalleryProfile] = {}

    def _is_isolated_detection(self, det_idx: int, detections: List[DetectionRaw]) -> bool:
        """Avoid contaminating appearance gallery when two people strongly overlap."""
        box_i = detections[det_idx].bbox
        for j, other in enumerate(detections):
            if j != det_idx and compute_iou(box_i, other.bbox) > 0.15:
                return False
        return True

    def _match_gallery_reid(
        self,
        det: DetectionRaw,
        emb: Optional[np.ndarray],
        excluded_ids: set,
    ) -> Optional[int]:
        """Match a newly re-appearing detection against inactive employees in the Re-ID Gallery."""
        if not self.gallery:
            return None

        diag_len = max(float(np.hypot(self.frame_width, self.frame_height)), 1.0)
        d_cx = 0.5 * (det.bbox[0] + det.bbox[2])
        d_cy = 0.5 * (det.bbox[1] + det.bbox[3])

        best_tid: Optional[int] = None
        best_score = -1.0

        for tid, prof in self.gallery.items():
            if tid in excluded_ids:
                continue
            # Require established gallery identity (at least 5 prior hits)
            if prof.total_hits < 5:
                continue

            p_cx = 0.5 * (prof.last_bbox[0] + prof.last_bbox[2])
            p_cy = 0.5 * (prof.last_bbox[1] + prof.last_bbox[3])
            dist_px = float(np.hypot(d_cx - p_cx, d_cy - p_cy))
            frames_absent = max(self.step_idx - prof.last_seen_step, 1)

            app_sim = prof.appearance_similarity(emb)
            spatial_prox = float(np.exp(-dist_px / (0.45 * diag_len)))
            recency_bonus = 0.06 * float(np.exp(-frames_absent / 240.0))

            if emb is not None and prof.ema_embedding is not None:
                # Strong torso color match (e.g. re-standing from behind desk or returning to doorway)
                if app_sim >= 0.76 or (frames_absent <= 90 and dist_px <= 360.0 and app_sim >= 0.62):
                    score = 0.82 * app_sim + 0.18 * spatial_prox + recency_bonus
                    if score > best_score:
                        best_score = score
                        best_tid = tid
            else:
                # Spatial-only fallback when no RGB frame was supplied
                if frames_absent <= 45 and dist_px <= 260.0:
                    score = spatial_prox + recency_bonus
                    if score > best_score:
                        best_score = score
                        best_tid = tid

        return best_tid

    def update(
        self,
        detections: List[DetectionRaw],
        frame: Optional[np.ndarray] = None,
    ) -> List[SmoothedObservation]:
        self.step_idx += 1
        num_tracks = len(self.active_tracks)
        num_dets = len(detections)

        det_embeddings: List[Optional[np.ndarray]] = [
            extract_torso_appearance_embedding(frame, det) for det in detections
        ]

        matched_track_indices = set()
        matched_det_indices = set()

        # --- Stage 1: Match detections against currently active tracks ---
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.full((num_tracks, num_dets), 10.0, dtype=np.float32)
            diag_len = max(float(np.hypot(self.frame_width, self.frame_height)), 1.0)
            pred_boxes = [t.get_predicted_matching_box() for t in self.active_tracks]

            for t_idx, track in enumerate(self.active_tracks):
                t_box = pred_boxes[t_idx]
                t_cx = 0.5 * (t_box[0] + t_box[2])
                t_cy = 0.5 * (t_box[1] + t_box[3])
                prof = self.gallery.get(track.track_id)

                for d_idx, det in enumerate(detections):
                    d_box = det.bbox
                    d_cx = 0.5 * (d_box[0] + d_box[2])
                    d_cy = 0.5 * (d_box[1] + d_box[3])

                    iou = compute_iou(t_box, d_box)
                    dist_norm = float(np.hypot(t_cx - d_cx, t_cy - d_cy)) / diag_len
                    geom_cost = (1.0 - iou) * 0.60 + dist_norm * 0.40

                    if prof is not None and det_embeddings[d_idx] is not None and prof.ema_embedding is not None:
                        app_sim = prof.appearance_similarity(det_embeddings[d_idx])
                        cost_matrix[t_idx, d_idx] = 0.65 * geom_cost + 0.35 * (1.0 - app_sim)
                    else:
                        cost_matrix[t_idx, d_idx] = geom_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                track = self.active_tracks[r]
                det = detections[c]
                t_box = pred_boxes[r]
                iou = compute_iou(t_box, det.bbox)
                t_cx = 0.5 * (t_box[0] + t_box[2])
                t_cy = 0.5 * (t_box[1] + t_box[3])
                d_cx = 0.5 * (det.bbox[0] + det.bbox[2])
                d_cy = 0.5 * (det.bbox[1] + det.bbox[3])
                center_px = float(np.hypot(t_cx - d_cx, t_cy - d_cy))

                prof = self.gallery.get(track.track_id)
                app_sim = prof.appearance_similarity(det_embeddings[c]) if prof else 0.0

                # Accept kinematic proximity or strong appearance continuity across brief gap
                max_shift = 280.0 + min(track.time_since_update * 8.0, 140.0)
                if iou >= self.iou_threshold or center_px < max_shift or (app_sim >= 0.78 and center_px < 520.0):
                    track.update(det)
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
                    if track.track_id not in self.gallery:
                        self.gallery[track.track_id] = EmployeeGalleryProfile(track_id=track.track_id)
                    emb_to_store = det_embeddings[c] if self._is_isolated_detection(c, detections) else None
                    self.gallery[track.track_id].update_appearance(
                        emb_to_store, det.bbox, self.step_idx, det.bbox_conf
                    )

        # Mark unmatched active tracks as missed
        for r in range(num_tracks):
            if r not in matched_track_indices:
                self.active_tracks[r].mark_missed()

        # IDs currently occupied in this frame (either matched right now or very recently active)
        occupied_ids = {
            self.active_tracks[r].track_id for r in matched_track_indices
        } | {
            t.track_id for idx, t in enumerate(self.active_tracks)
            if idx not in matched_track_indices and t.time_since_update <= 15
        }

        # --- Stage 2: Re-ID Gallery lookup or new Employee ID creation for unmatched detections ---
        for c in range(num_dets):
            if c in matched_det_indices:
                continue
            det = detections[c]
            emb = det_embeddings[c]

            reid_tid = self._match_gallery_reid(det, emb, excluded_ids=occupied_ids)
            if reid_tid is not None:
                # Re-attach to existing archived employee ID!
                occupied_ids.add(reid_tid)
                # Remove any stale ghost instance of reid_tid in active_tracks if present
                self.active_tracks = [t for t in self.active_tracks if t.track_id != reid_tid]
                revived_track = SinglePersonTrack(
                    track_id=reid_tid,
                    initial_detection=det,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    smoother_type=self.smoother_type,
                    dt=1.0 / max(self.fps, 1.0),
                )
                revived_track.hits = self.gallery[reid_tid].total_hits + 1
                self.active_tracks.append(revived_track)
                emb_to_store = emb if self._is_isolated_detection(c, detections) else None
                self.gallery[reid_tid].update_appearance(emb_to_store, det.bbox, self.step_idx, det.bbox_conf)
            elif det.bbox_conf >= self.new_track_min_conf:
                new_tid = self.next_track_id
                self.next_track_id += 1
                occupied_ids.add(new_tid)
                new_track = SinglePersonTrack(
                    track_id=new_tid,
                    initial_detection=det,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    smoother_type=self.smoother_type,
                    dt=1.0 / max(self.fps, 1.0),
                )
                self.active_tracks.append(new_track)
                self.gallery[new_tid] = EmployeeGalleryProfile(track_id=new_tid)
                emb_to_store = emb if self._is_isolated_detection(c, detections) else None
                self.gallery[new_tid].update_appearance(emb_to_store, det.bbox, self.step_idx, det.bbox_conf)

        # --- Stage 3: Prune expired active tracks & emit visible observations ---
        surviving_tracks: List[SinglePersonTrack] = []
        observations: List[SmoothedObservation] = []

        for track in self.active_tracks:
            # Keep tentative tracks alive for up to 5 frames so a 1-frame doorway miss doesn't drop ID #1!
            max_allowed_miss = 5 if track.hits < self.min_hits else self.max_age_frames
            if track.time_since_update <= max_allowed_miss:
                surviving_tracks.append(track)

                is_coasting = track.time_since_update > 0
                if not is_coasting:
                    observations.append(track.get_smoothed_observation(is_coasted=False))
                elif track.hits >= self.min_hits and track.time_since_update <= self.max_coast_render_frames:
                    observations.append(track.get_smoothed_observation(is_coasted=True))

        self.active_tracks = surviving_tracks
        observations.sort(key=lambda o: o.track_id)
        return observations
