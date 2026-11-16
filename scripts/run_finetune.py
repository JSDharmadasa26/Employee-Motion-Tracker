"""CLI script to fine-tune YOLOv8n-Pose on office keypoints, desk barriers, and fast movements."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_INPUT_VIDEO
from src.training.fine_tune import run_pose_model_finetuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n-Pose for accurate 17-keypoint office tracking.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of fine-tuning epochs (default: 25).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: 640).")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size (default: 16).")
    parser.add_argument("--video", type=str, default=str(DEFAULT_INPUT_VIDEO), help="Source office footage.")
    args = parser.parse_args()

    run_pose_model_finetuning(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        video_path=Path(args.video),
    )


if __name__ == "__main__":
    main()
