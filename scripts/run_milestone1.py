"""CLI runner for Milestone 1: Spatial Accuracy & Keypoint Validation."""

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_INPUT_VIDEO,
    DEFAULT_YOLO_MODEL,
    RAW_VIDEOS_DIR,
    discover_raw_videos,
)
from src.tracking.pipeline import BatchTrackingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Milestone 1: YOLOv8-Pose tracking with confidence-weighted smoothing."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a single test video. If omitted, all videos in data/raw_videos/ are processed in batch.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_YOLO_MODEL),
        help="Path to YOLOv8-Pose model weights.",
    )
    parser.add_argument(
        "--smoother",
        type=str,
        choices=["kalman", "rolling"],
        default="kalman",
        help="Smoothing algorithm to use (default: kalman).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap on number of frames to process per video (useful for quick validation/smoke tests).",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.22,
        help="Confidence threshold for bounding box detection.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing the annotated output MP4 video (faster, telemetry CSV only).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV live tracking preview window.",
    )

    args = parser.parse_args()

    pipeline = BatchTrackingPipeline(
        model_path=Path(args.model),
        smoother_type=args.smoother,
        min_bbox_conf=args.conf_thres,
    )

    if args.video:
        target_video = Path(args.video)
        if not target_video.exists():
            print(f"Error: Video file not found: {target_video}")
            sys.exit(1)
        video_list = [target_video]
    else:
        video_list = discover_raw_videos(RAW_VIDEOS_DIR)
        if not video_list:
            if DEFAULT_INPUT_VIDEO.exists():
                video_list = [DEFAULT_INPUT_VIDEO]
            else:
                print(f"Error: No videos found in {RAW_VIDEOS_DIR}")
                sys.exit(1)

    print(f"\n=======================================================")
    print(f"  Milestone 1: Spatial Accuracy & Keypoint Validation")
    print(f"  Videos to process: {len(video_list)}")
    print(f"  Smoother: {args.smoother.upper()}")
    print(f"  Max frames per video: {args.max_frames or 'Full video'}")
    print(f"=======================================================\n")

    stats = pipeline.run_batch(
        video_paths=video_list,
        max_frames=args.max_frames,
        save_video=not args.no_video,
    )

    print("\nBatch Processing Completed Successfully!")
    for s in stats:
        print(f"\n- Video: {s['video_name']}")
        print(f"  Processed Frames: {s['frames_processed']}")
        print(f"  Speed: {s['effective_processing_fps']} FPS")
        print(f"  BBox Jitter Reduction: {s['bbox_jitter_reduction_pct']}%")
        print(f"  CSV: {s['output_csv_path']}")
        print(f"  Video: {s['output_video_path']}")


if __name__ == "__main__":
    main()
