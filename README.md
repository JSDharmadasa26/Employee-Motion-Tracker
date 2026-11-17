# Employee Motion Tracker

Privacy-preserving computer vision & business intelligence pipeline that extracts 13 skeletal landmarks from office footage using a fine-tuned **YOLOv8n-Pose** model, maintains persistent Employee IDs across occlusions via torso appearance Re-ID, classifies behavior into **6 calibrated motion states** (`Sitting`, `Standing`, `Walking`, `Away`, `Idle`, `Working`), and surfaces interactive operational & ergonomic KPIs in a dark-themed **Streamlit + Plotly Executive Dashboard**.

![Executive Dashboard Snapshot](outputs/reports/executive_dashboard_snapshot.png)

## Key Capabilities

- **Fine-Tuned YOLOv8n-Pose (`models/best_yolov8n_pose.pt`)**: `0.9940` Box mAP50 and `0.9938` Pose mAP50 at `~32.7 FPS` on local CPU, resilient to solid desk occlusions and fast walking.
- **Zero-Lag Directional Kalman Smoother & Persistent Re-ID**: `54%` bounding-box jitter reduction plus privacy-safe HSV + CIE-Lab torso clothing gallery (`Emp #1`, `Emp #2`) with zero facial biometrics stored.
- **Data-Calibrated 6-Class Rules Engine**: 1D KDE valley & Otsu threshold calibration with Schmitt-trigger hysteresis and dwell filtering for `Sitting`, `Standing`, `Walking`, `Away`, `Idle`, and `Working`.
- **Interactive Streamlit + Plotly BI Dashboard (`app.py`)**: Spatial Posture & Utilization Donut charts, KPI sparkline cards, Ergonomic Gantt timelines, 2D Wrist Velocity heatmaps, and automated CSV/PNG/PDF executive exports.

## Project Structure

```text
employee-motion-tracker/
├── .streamlit/
│   └── config.toml                 # Dark glassmorphic Streamlit theme configuration
├── models/
│   └── best_yolov8n_pose.pt        # Fine-tuned YOLOv8n-Pose weights (COCO 17 keypoints)
├── data/
│   └── raw_videos/                 # Standardized input test videos (ignored in git for privacy)
├── outputs/
│   ├── videos/                     # H.264 (avc1) annotated tracking videos
│   ├── telemetry/                  # temporal_features.csv & calibrated_thresholds.json
│   ├── plots/                      # Jitter, posture & velocity calibration charts
│   └── reports/                    # Executive KPI CSV/JSON/PNG & 2-page PDF reports
├── src/
│   ├── config.py                   # Centralized paths & COCO 17-keypoint index map
│   ├── training/                   # Pose dataset curation & YOLOv8n-Pose fine-tuning
│   ├── tracking/                   # Milestone 1: Batch tracking, Re-ID gallery & Kalman smoothing
│   ├── calibration/                # Milestone 2: Posture/velocity calibration & 6-class rules engine
│   └── dashboard/                  # Milestone 3: Polars KPI engine, Plotly charts & PDF/PNG reporter
├── scripts/
│   ├── run_milestone1.py           # CLI runner for Milestone 1 tracking & Re-ID
│   ├── run_milestone2.py           # CLI runner for Milestone 2 calibration & 6-class labeling
│   ├── render_classified_video.py  # Render H.264 annotated video with live 6-class badges
│   ├── run_dashboard.py            # Launch Streamlit Executive Dashboard locally
│   └── generate_report.py          # Compile standalone CSV/JSON/PNG/PDF executive reports
├── tests/
│   ├── test_milestone1_tracking.py     # Tracking, Kalman & Re-ID unit/smoke tests
│   ├── test_milestone2_calibration.py  # Calibration & 6-class rules engine tests
│   ├── test_milestone3_dashboard.py    # Dashboard KPIs, Plotly figures & AppTest suite
│   └── test_outputs_integrity.py       # Structural & output file integrity validation
├── app.py                          # Main Streamlit Executive Dashboard application
├── requirements.txt
└── .gitignore
```

## Quick Start

1. **Activate virtual environment & install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Milestone 1 & 2 (Tracking, Re-ID & 6-Class Calibration)**:
   ```bash
   python scripts/run_milestone1.py --smoother kalman
   python scripts/run_milestone2.py
   python scripts/render_classified_video.py
   ```

3. **Launch Milestone 3 Interactive Executive Dashboard (`http://localhost:8501`)**:
   ```bash
   python scripts/run_dashboard.py
   ```

4. **Generate Standalone Executive Reports (`outputs/reports/`)**:
   ```bash
   python scripts/generate_report.py
   ```

5. **Run Automated Test Suite (24 Tests)**:
   ```bash
   python -m unittest discover -s tests -p "test_*.py" -v
   ```
