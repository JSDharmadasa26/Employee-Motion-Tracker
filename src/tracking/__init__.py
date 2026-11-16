"""Milestone 1: Spatial Accuracy & Keypoint Validation package."""

from src.tracking.landmarks import (
    DetectionRaw,
    LandmarkPoint,
    SmoothedObservation,
    extract_detections_from_results,
)
from src.tracking.multi_tracker import PoseMultiTracker, SinglePersonTrack
from src.tracking.pipeline import BatchTrackingPipeline, render_video_from_telemetry_csv
from src.tracking.smoother import (
    KalmanBBoxSmoother,
    KalmanPoint2DSmoother,
    RollingAverageSmoother,
)
from src.tracking.visualizer import create_video_writer, render_annotated_frame

__all__ = [
    "KalmanPoint2DSmoother",
    "KalmanBBoxSmoother",
    "RollingAverageSmoother",
    "LandmarkPoint",
    "DetectionRaw",
    "SmoothedObservation",
    "extract_detections_from_results",
    "SinglePersonTrack",
    "PoseMultiTracker",
    "create_video_writer",
    "render_annotated_frame",
    "render_video_from_telemetry_csv",
    "BatchTrackingPipeline",
]
