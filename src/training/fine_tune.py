"""Fine-tunes YOLOv8n-Pose for high-accuracy 17-keypoint detection, desk-barrier occlusion
recovery, hard-negative false-positive suppression, and rapid movement responsiveness.
"""

import csv
import json
from pathlib import Path
import shutil
import time
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.config import (
    BACKUP_V1_YOLO_MODEL,
    DEFAULT_INPUT_VIDEO,
    DEFAULT_YOLO_MODEL,
    FINE_TUNE_DATASET_DIR,
    PLOTS_OUTPUT_DIR,
    TELEMETRY_OUTPUT_DIR,
    ensure_directories,
)
from src.training.dataset_builder import build_finetune_dataset


def run_pose_model_finetuning(
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    video_path: Path = DEFAULT_INPUT_VIDEO,
) -> Dict[str, object]:
    """Build multi-teacher pose dataset and fine-tune YOLOv8n-Pose weights locally."""
    ensure_directories()

    # Preserve original v1 weights safely before fine-tuning
    if DEFAULT_YOLO_MODEL.exists() and not BACKUP_V1_YOLO_MODEL.exists():
        shutil.copy2(DEFAULT_YOLO_MODEL, BACKUP_V1_YOLO_MODEL)

    base_weights = BACKUP_V1_YOLO_MODEL if BACKUP_V1_YOLO_MODEL.exists() else DEFAULT_YOLO_MODEL
    print(f"[1/3] Curating 17-keypoint dataset with barrier-occlusion & fast-motion samples from {video_path.name}...")
    ds_stats = build_finetune_dataset(
        video_path=video_path,
        dataset_dir=FINE_TUNE_DATASET_DIR,
        yolo_teacher_path=base_weights,
    )
    print(f"      Dataset ready: {ds_stats}")

    runs_dir = Path("/tmp/yolo_office_pose_finetune")
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    print(f"[2/3] Fine-tuning YOLOv8n-Pose for {epochs} epochs (imgsz={imgsz}, batch={batch})...")
    t0 = time.time()
    model = YOLO(str(base_weights))
    model.train(
        data=str(ds_stats["dataset_yaml"]),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device="cpu",
        project=str(runs_dir),
        name="pose_v2",
        exist_ok=True,
        pose=14.0,
        kobj=2.0,
        box=7.5,
        cls=1.0,
        erasing=0.30,
        mosaic=0.8,
        fliplr=0.5,
        scale=0.35,
        translate=0.10,
        plots=False,
        verbose=True,
    )
    train_seconds = round(time.time() - t0, 2)

    run_out_dir = runs_dir / "pose_v2"
    best_pt = run_out_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_out_dir / "weights" / "last.pt"

    # Deploy updated fine-tuned model to models/best_yolov8n_pose.pt
    shutil.copy2(best_pt, DEFAULT_YOLO_MODEL)
    model_size_mb = round(DEFAULT_YOLO_MODEL.stat().st_size / (1024 * 1024), 2)

    # Parse Ultralytics results.csv for quantitative mAP & loss curves
    results_csv = run_out_dir / "results.csv"
    history = []
    if results_csv.exists():
        with open(results_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): float(v.strip()) for k, v in row.items() if v.strip()}
                history.append(clean_row)

    curves_plot_path = PLOTS_OUTPUT_DIR / "model_finetune_curves.png"
    if history:
        ep = [int(r.get("epoch", i + 1)) for i, r in enumerate(history)]
        pose_loss = [r.get("train/pose_loss", 0.0) for r in history]
        box_loss = [r.get("train/box_loss", 0.0) for r in history]
        box_map50 = [r.get("metrics/mAP50(B)", 0.0) for r in history]
        pose_map50 = [r.get("metrics/mAP50(P)", 0.0) for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(ep, pose_loss, "o-", color="#8e44ad", linewidth=2.0, markersize=4, label="Train Pose Loss")
        ax1.plot(ep, box_loss, "s-", color="#2980b9", linewidth=2.0, markersize=4, label="Train Box Loss")
        ax1.set_title("YOLOv8n-Pose Fine-Tuning Loss Convergence")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        ax2.plot(ep, [m * 100 for m in box_map50], "o-", color="#27ae60", linewidth=2.0, markersize=4, label="Box mAP50 (%)")
        ax2.plot(ep, [m * 100 for m in pose_map50], "s-", color="#e67e22", linewidth=2.0, markersize=4, label="Keypoint Pose mAP50 (%)")
        ax2.set_title("Validation Accuracy (17 Keypoints + Barrier Occlusion)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP50 (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(curves_plot_path, dpi=150)
        plt.close(fig)

    best_box_map50 = max((r.get("metrics/mAP50(B)", 0.0) for r in history), default=0.0)
    best_pose_map50 = max((r.get("metrics/mAP50(P)", 0.0) for r in history), default=0.0)

    metrics = {
        "updated_model_path": str(DEFAULT_YOLO_MODEL),
        "backup_v1_model_path": str(BACKUP_V1_YOLO_MODEL),
        "model_size_mb": model_size_mb,
        "epochs": epochs,
        "training_seconds": train_seconds,
        "dataset": ds_stats,
        "best_box_mAP50": round(best_box_map50, 4),
        "best_pose_mAP50": round(best_pose_map50, 4),
        "curves_plot_path": str(curves_plot_path),
    }
    metrics_path = TELEMETRY_OUTPUT_DIR / "finetune_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    shutil.rmtree(runs_dir, ignore_errors=True)

    print(
        f"[3/3] Fine-tuning complete in {train_seconds}s! "
        f"Box mAP50={best_box_map50:.4f}, Pose mAP50={best_pose_map50:.4f}, "
        f"Weights saved to {DEFAULT_YOLO_MODEL} ({model_size_mb} MB)"
    )
    return metrics
