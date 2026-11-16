"""Render the master H.264 tracked video with persistent Employee Re-ID and live 6-class
motion badges, plus save visual verification grids covering key frames.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # noqa: F401
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DEFAULT_INPUT_VIDEO,
    PLOTS_OUTPUT_DIR,
    TELEMETRY_OUTPUT_DIR,
    VIDEOS_OUTPUT_DIR,
)
from src.tracking.pipeline import render_video_from_telemetry_csv


def save_verification_grid(
    video_path: Path,
    out_plot_path: Path,
    frame_numbers=(199, 872, 2302, 5623, 5663, 5837),
    suptitle: str = "Persistent Employee Re-ID (Emp #1 / Emp #2) & 6-Class Motion Categorization Verification",
) -> None:
    """Extract target 1-based frames from rendered MP4 and save a crisp 3x2 visual verification grid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open rendered video: {video_path}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2))
    axes_flat = axes.flatten()

    descriptions = {
        1: "Frame 1 (Empty Workstation — Away)",
        49: "Frame 49 (Doorway Entry — Emp #1 Walking)",
        75: "Frame 75 (Fast Walking — Zero Kalman Lag)",
        103: "Frame 103 (Mid-Stride — Emp #1 Walking)",
        170: "Frame 170 (Rear-View Door Closing — Emp #1)",
        199: "Frame 199 (Emp #1 Walking In — Persistent ID)",
        872: "Frame 872 (Emp #1 Seated Chair Adjust — Sitting | Working)",
        1200: "Frame 1200 (Desk Occlusion — Emp #1 Sitting | Working)",
        2302: "Frame 2302 (Simultaneous Emp #1 & Visitor Emp #2)",
        5623: "Frame 5623 (Post-Desk Occlusion Recovery — Still Emp #1!)",
        5663: "Frame 5663 (Emp #1 Walking to Exit — Persistent ID)",
        5837: "Frame 5837 (Emp #1 Doorway Exit — Clean Active Badge)",
    }

    for ax, f_num in zip(axes_flat, frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f_num - 1, 0))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        ax.set_title(descriptions.get(f_num, f"Frame {f_num}"), fontsize=10.5, fontweight="bold")
        ax.axis("off")

    cap.release()
    plt.suptitle(suptitle, fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved visual verification grid to: {out_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render master H.264 tracked video from classified CSV.")
    parser.add_argument("--video", type=str, default=str(DEFAULT_INPUT_VIDEO))
    parser.add_argument(
        "--csv",
        type=str,
        default=str(TELEMETRY_OUTPUT_DIR / "Input_1_classified.csv"),
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=str(VIDEOS_OUTPUT_DIR / "Input_1_tracked.mp4"),
    )
    args = parser.parse_args()

    res = render_video_from_telemetry_csv(
        video_path=Path(args.video),
        landmarks_csv_path=Path(args.csv),
        output_video_path=Path(args.output_video),
        smoother_name="Kalman",
    )
    print(
        f"Rendered H.264 video: {res['output_video_path']} "
        f"({res['frames_written']} frames in {res['elapsed_seconds']}s)"
    )

    save_verification_grid(
        video_path=Path(args.output_video),
        out_plot_path=PLOTS_OUTPUT_DIR / "reid_6class_verification.png",
        frame_numbers=(199, 872, 2302, 5623, 5663, 5837),
    )
    save_verification_grid(
        video_path=Path(args.output_video),
        out_plot_path=PLOTS_OUTPUT_DIR / "finetune_visual_verification.png",
        frame_numbers=(1, 49, 199, 872, 2302, 5663),
        suptitle="Fine-Tuned YOLOv8n-Pose + Persistent Re-ID + 6-Class Categorization",
    )


if __name__ == "__main__":
    main()
