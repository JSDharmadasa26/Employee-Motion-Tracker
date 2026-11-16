"""Unit and integration tests for Milestone 1: Spatial Accuracy & Keypoint Validation."""

import json
from pathlib import Path
import unittest
import numpy as np

from src.config import (
    DEFAULT_INPUT_VIDEO,
    DEFAULT_YOLO_MODEL,
    TARGET_LANDMARKS,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
)
from src.tracking.landmarks import DetectionRaw, LandmarkPoint
from src.tracking.multi_tracker import PoseMultiTracker, compute_iou
from src.tracking.pipeline import CSV_HEADER, BatchTrackingPipeline, compute_acceleration_jitter
from src.tracking.smoother import (
    KalmanBBoxSmoother,
    KalmanPoint2DSmoother,
    RollingAverageSmoother,
)


class TestMilestone1Tracking(unittest.TestCase):
    """Test suite for coordinate smoothing, landmark extraction, and multi-tracker stability."""

    def test_compute_iou(self):
        box_a = (100.0, 100.0, 200.0, 200.0)
        box_b = (100.0, 100.0, 200.0, 200.0)
        self.assertAlmostEqual(compute_iou(box_a, box_b), 1.0, places=3)

        box_c = (200.0, 200.0, 300.0, 300.0)
        self.assertAlmostEqual(compute_iou(box_a, box_c), 0.0, places=3)

    def test_kalman_point_smoother_jitter_reduction(self):
        """Verify that Kalman filter significantly dampens high-frequency artificial jitter."""
        np.random.seed(42)
        smoother = KalmanPoint2DSmoother(dt=1.0 / 30.0)

        # Baseline ground truth linear trajectory
        t = np.linspace(0, 100, 100)
        true_x = 200.0 + 0.5 * t
        true_y = 300.0 + 0.2 * t

        # Add heavy sensor jitter (+/- 15px noise)
        noisy_x = true_x + np.random.normal(0, 8.0, size=len(t))
        noisy_y = true_y + np.random.normal(0, 8.0, size=len(t))

        smooth_pts = []
        for x, y in zip(noisy_x, noisy_y):
            sx, sy, sc = smoother.update(x, y, conf=0.9)
            smooth_pts.append((sx, sy))

        raw_coords = np.column_stack([noisy_x, noisy_y])
        sm_coords = np.array(smooth_pts)

        raw_jitter = compute_acceleration_jitter(raw_coords)
        sm_jitter = compute_acceleration_jitter(sm_coords)

        # Expect > 50% jitter suppression
        reduction_pct = (1.0 - sm_jitter / raw_jitter) * 100
        self.assertGreater(reduction_pct, 50.0)

    def test_desk_occlusion_coasting(self):
        """Verify smoother behaves gracefully when confidence drops during desk occlusion."""
        smoother = KalmanPoint2DSmoother(dt=1.0 / 30.0)
        # Prime with 10 visible frames
        for i in range(10):
            smoother.update(500.0 + i, 800.0, conf=0.85)

        # Desk occlusion: confidence drops to 0.0 for 5 frames
        occluded_points = []
        for _ in range(5):
            sx, sy, sc = smoother.update(0.0, 0.0, conf=0.0)
            occluded_points.append((sx, sy, sc))

        # Check coordinates did not snap to (0, 0)
        for sx, sy, sc in occluded_points:
            self.assertGreater(sx, 400.0)
            self.assertGreater(sy, 700.0)

    def test_kalman_bbox_smoother(self):
        smoother = KalmanBBoxSmoother(dt=1.0 / 30.0)
        raw_boxes = [
            (100.0 + i + (2.0 if i % 2 == 0 else -2.0), 200.0, 300.0 + i, 500.0)
            for i in range(20)
        ]
        smoothed_boxes = []
        for box in raw_boxes:
            smoothed_boxes.append(smoother.update(*box, conf=0.9))

        self.assertEqual(len(smoothed_boxes), 20)
        # Final smoothed width should remain near ~200
        w = smoothed_boxes[-1][2] - smoothed_boxes[-1][0]
        self.assertAlmostEqual(w, 200.0, delta=10.0)

    def test_rolling_average_smoother(self):
        smoother = RollingAverageSmoother(window_size=5)
        for i in range(10):
            smoother.update_point("left_wrist", 100.0 + i, 200.0, 0.9)
        sx, sy, sc = smoother.update_point("left_wrist", 110.0, 200.0, 0.9)
        self.assertAlmostEqual(sy, 200.0, places=2)
        self.assertGreater(sx, 100.0)

    def test_multi_tracker_association(self):
        """Test that PoseMultiTracker maintains persistent track IDs across frames."""
        tracker = PoseMultiTracker(frame_width=1920, frame_height=1080, fps=30.0)

        def make_dummy_det(x1, y1, x2, y2):
            lms = {
                name: LandmarkPoint(name=name, x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0, conf=0.9)
                for name in TARGET_LANDMARKS
            }
            return DetectionRaw(
                bbox=(x1, y1, x2, y2),
                bbox_conf=0.9,
                class_id=1,
                landmarks=lms,
            )

        # Frame 1: Person A and Person B
        det_a1 = make_dummy_det(100.0, 100.0, 200.0, 300.0)
        det_b1 = make_dummy_det(500.0, 100.0, 600.0, 300.0)
        obs1 = tracker.update([det_a1, det_b1])

        # Frame 2: Slight movement
        det_a2 = make_dummy_det(102.0, 100.0, 202.0, 300.0)
        det_b2 = make_dummy_det(501.0, 100.0, 601.0, 300.0)
        obs2 = tracker.update([det_a2, det_b2])

        self.assertEqual(len(obs2), 2)
        # Track IDs should match between frames
        ids1 = {o.track_id for o in obs1}
        ids2 = {o.track_id for o in obs2}
        self.assertEqual(ids1, ids2)

    def test_persistent_reid_across_long_occlusion(self):
        """Verify EmployeeGalleryProfile restores Emp #1 after >60 missed frames while assigning Emp #2 to a visitor."""
        tracker = PoseMultiTracker(frame_width=1280, frame_height=720, fps=30.0, max_age_frames=20)

        def make_colored_frame_and_det(x1: float, y1: float, x2: float, y2: float, bgr_color: tuple):
            frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            frame[iy1:iy2, ix1:ix2] = bgr_color
            cx = 0.5 * (x1 + x2)
            lms = {
                name: LandmarkPoint(name=name, x=cx, y=y1 + 60.0, conf=0.92)
                for name in TARGET_LANDMARKS
            }
            lms["left_shoulder"] = LandmarkPoint("left_shoulder", x1 + 25.0, y1 + 50.0, 0.95)
            lms["right_shoulder"] = LandmarkPoint("right_shoulder", x2 - 25.0, y1 + 50.0, 0.95)
            lms["left_hip"] = LandmarkPoint("left_hip", x1 + 30.0, y1 + 170.0, 0.90)
            lms["right_hip"] = LandmarkPoint("right_hip", x2 - 30.0, y1 + 170.0, 0.90)
            det = DetectionRaw(bbox=(x1, y1, x2, y2), bbox_conf=0.92, class_id=0, landmarks=lms)
            return frame, det

        # Phase 1: Emp #1 (light-blue polo: BGR=(235, 165, 75)) seen for 10 frames
        emp1_ids = set()
        for _ in range(10):
            f_bgr, det1 = make_colored_frame_and_det(300.0, 250.0, 480.0, 620.0, (235, 165, 75))
            obs = tracker.update([det1], frame=f_bgr)
            emp1_ids.update(o.track_id for o in obs)
        self.assertEqual(emp1_ids, {1})

        # Phase 2: Complete desk/room occlusion for 60 frames (exceeds max_age_frames=20!)
        empty_frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        for _ in range(60):
            tracker.update([], frame=empty_frame)

        # Phase 3: Emp #1 re-appears at a shifted position + simultaneous Visitor in dark red/navy shirt
        for _ in range(5):
            frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
            _, det_emp1 = make_colored_frame_and_det(620.0, 220.0, 800.0, 650.0, (235, 165, 75))
            _, det_vis = make_colored_frame_and_det(920.0, 220.0, 1100.0, 650.0, (35, 30, 160))
            frame[220:650, 620:800] = (235, 165, 75)
            frame[220:650, 920:1100] = (35, 30, 160)
            obs = tracker.update([det_emp1, det_vis], frame=frame)

        by_x = sorted(obs, key=lambda o: o.bbox[0])
        self.assertEqual(len(by_x), 2)
        self.assertEqual(by_x[0].track_id, 1, "Re-appearing primary employee must keep Emp #1!")
        self.assertEqual(by_x[1].track_id, 2, "Distinct simultaneous visitor must receive Emp #2!")

    def test_13_landmark_schema_and_adaptive_zero_lag_kalman(self):
        """Verify 13-keypoint CSV header (53 cols) and innovation-adaptive Kalman tracking during fast walking."""
        self.assertEqual(len(TARGET_LANDMARKS), 13)
        self.assertEqual(len(CSV_HEADER), 14 + 13 * 3)

        # Fast movement step: 20 px/frame (600 px/s) across doorway
        smoother = KalmanPoint2DSmoother(dt=1.0 / 30.0)
        lags = []
        for f in range(30):
            true_x = 1100.0 + 20.0 * f
            true_y = 450.0
            sx, sy, _ = smoother.update(true_x, true_y, conf=0.92)
            if f >= 5:
                lags.append(abs(true_x - sx))

        # Innovation-adaptive Kalman filter must stay tightly locked (< 6 px lag even at 600 px/s!)
        self.assertLess(float(np.mean(lags)), 6.0)

    def test_pipeline_smoke_run(self):
        """Smoke test processing a short slice of Input_1.mp4 into an isolated temp directory."""
        import tempfile

        if not DEFAULT_INPUT_VIDEO.exists() or not DEFAULT_YOLO_MODEL.exists():
            self.skipTest("Input video or model weights not present.")

        pipeline = BatchTrackingPipeline(
            model_path=DEFAULT_YOLO_MODEL,
            smoother_type="kalman",
            min_bbox_conf=0.32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            stats = pipeline.process_video(
                video_path=DEFAULT_INPUT_VIDEO,
                max_frames=45,
                save_video=True,
                show_preview=False,
                output_video_dir=tmp_path / "videos",
                output_telemetry_dir=tmp_path / "telemetry",
            )

            self.assertEqual(stats["frames_processed"], 45)
            self.assertTrue(Path(stats["output_csv_path"]).exists())
            self.assertTrue(Path(stats["output_video_path"]).exists())
            self.assertGreater(stats["csv_rows_written"], 0)


if __name__ == "__main__":
    unittest.main()
