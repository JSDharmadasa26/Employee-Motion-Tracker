"""Quick local visual validation script for the fine-tuned YOLOv8-Pose model."""

from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

# Ensure project root is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_INPUT_VIDEO, DEFAULT_YOLO_MODEL


def main() -> None:
    model = YOLO(str(DEFAULT_YOLO_MODEL))
    cap = cv2.VideoCapture(str(DEFAULT_INPUT_VIDEO))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {DEFAULT_INPUT_VIDEO}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Local YOLOv8 Tracking Test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
