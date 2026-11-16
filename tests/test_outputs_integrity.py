"""Workspace file-structure and outputs/ folder integrity validation tests."""

import json
from pathlib import Path
import unittest
import cv2
import polars as pl

from src.config import (
    DEFAULT_INPUT_VIDEO,
    DEFAULT_YOLO_MODEL,
    MODELS_DIR,
    PLOTS_OUTPUT_DIR,
    RAW_VIDEOS_DIR,
    REPORTS_OUTPUT_DIR,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
    ensure_directories,
)


class TestOutputsAndStructureIntegrity(unittest.TestCase):
    """Verify project folder hierarchy and integrity of all generated artifacts in outputs/."""

    def test_directory_structure_exists(self):
        ensure_directories()
        for d in (
            MODELS_DIR,
            RAW_VIDEOS_DIR,
            VIDEOS_OUTPUT_DIR,
            TELEMETRY_OUTPUT_DIR,
            PLOTS_OUTPUT_DIR,
            REPORTS_OUTPUT_DIR,
        ):
            self.assertTrue(d.exists() and d.is_dir(), f"Missing directory: {d}")
        self.assertTrue(DEFAULT_YOLO_MODEL.exists(), "Missing YOLOv8-Pose weights")

    def test_tracked_videos_playable_and_complete(self):
        """Ensure every MP4 in outputs/videos/ uses playable H.264 codec and matches source frame count."""
        tracked_videos = sorted(VIDEOS_OUTPUT_DIR.glob("*_tracked.mp4"))
        self.assertGreater(len(tracked_videos), 0, "No tracked MP4 found in outputs/videos/")

        for tv in tracked_videos:
            cap = cv2.VideoCapture(str(tv))
            self.assertTrue(cap.isOpened(), f"Cannot open output video: {tv}")

            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).lower()

            # Verify playable H.264 stream (not unplayable FMP4)
            self.assertIn(fourcc_str.strip(), {"h264", "avc1"}, f"Unexpected codec '{fourcc_str}' in {tv.name}")
            self.assertGreater(fps, 0.0)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

            # Compare frame count against matching raw video if present
            raw_stem = tv.stem.replace("_tracked", "")
            raw_candidates = list(RAW_VIDEOS_DIR.glob(f"{raw_stem}.*"))
            if raw_candidates:
                raw_cap = cv2.VideoCapture(str(raw_candidates[0]))
                raw_frames = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raw_cap.release()
                self.assertEqual(
                    frames,
                    raw_frames,
                    f"{tv.name} has {frames} frames, expected {raw_frames} from {raw_candidates[0].name}",
                )

            # Decode first, middle, and last frame to verify zero stream corruption
            for check_idx in (0, frames // 2, max(frames - 1, 0)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
                ok, frame = cap.read()
                self.assertTrue(ok and frame is not None, f"Failed decoding frame {check_idx} in {tv.name}")
                self.assertEqual(frame.shape, (h, w, 3))

            cap.release()

    def test_telemetry_csvs_and_jsons_valid(self):
        """Ensure all CSVs and JSONs in outputs/telemetry/ parse cleanly without corruption or missing paths."""
        expected_csvs = [
            "Input_1_landmarks.csv",
            "batch_landmarks.csv",
            "Input_1_classified.csv",
            "temporal_features.csv",
        ]
        for name in expected_csvs:
            p = TELEMETRY_OUTPUT_DIR / name
            self.assertTrue(p.exists(), f"Missing telemetry CSV: {p}")
            df = pl.read_csv(p, infer_schema_length=15000)
            self.assertGreater(len(df), 0, f"Empty CSV: {p}")
            # Ensure numeric coordinate columns did not degrade to strings
            self.assertIn(df.schema["left_shoulder_y"], (pl.Float64, pl.Float32))
            self.assertIn(df.schema["right_wrist_x"], (pl.Float64, pl.Float32))
            # Ensure zero duplicate (video_name, frame, track_id) keys
            uniq_len = len(df.unique(subset=["video_name", "frame", "track_id"]))
            self.assertEqual(len(df), uniq_len, f"Duplicate keys found in {name}")

        # Verify strict 6-class taxonomy columns and allowed values in classified telemetry
        classified_df = pl.read_csv(TELEMETRY_OUTPUT_DIR / "temporal_features.csv", infer_schema_length=15000)
        for cat_col in ("presence_state", "posture_state", "activity_state", "motion_state"):
            self.assertIn(cat_col, classified_df.columns)

        allowed_postures = {"Sitting", "Standing", "Walking", "Away"}
        allowed_activities = {"Working", "Idle", "Walking", "Away"}
        self.assertTrue(set(classified_df["posture_state"].unique().to_list()).issubset(allowed_postures))
        self.assertTrue(set(classified_df["activity_state"].unique().to_list()).issubset(allowed_activities))

        # Validate JSON summaries & referenced paths
        m1_json = json.loads((TELEMETRY_OUTPUT_DIR / "milestone1_validation_summary.json").read_text())
        self.assertIsInstance(m1_json, list)
        for item in m1_json:
            self.assertTrue(Path(item["output_video_path"]).exists())
            self.assertTrue(Path(item["output_csv_path"]).exists())

        m2_json = json.loads((TELEMETRY_OUTPUT_DIR / "milestone2_summary.json").read_text())
        self.assertTrue(Path(m2_json["temporal_features_csv_path"]).exists())
        self.assertTrue(Path(m2_json["calibrated_thresholds_json_path"]).exists())
        for plot_p in m2_json["plots"].values():
            self.assertTrue(Path(plot_p).exists(), f"Missing plot referenced in summary: {plot_p}")

    def test_plots_valid_png_headers(self):
        """Ensure all PNG charts in outputs/plots/ have valid PNG headers and non-zero size."""
        pngs = sorted(PLOTS_OUTPUT_DIR.glob("*.png"))
        self.assertGreaterEqual(len(pngs), 4)
        for png in pngs:
            self.assertGreater(png.stat().st_size, 1024, f"Plot file suspiciously small: {png}")
            with open(png, "rb") as f:
                sig = f.read(8)
            self.assertEqual(sig, b"\x89PNG\r\n\x1a\n", f"Invalid PNG header in {png}")


if __name__ == "__main__":
    unittest.main()
