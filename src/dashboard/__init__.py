"""Milestone 3: Executive Dashboard & Reporting package."""

from src.dashboard.data_loader import (
    build_employee_summary_table,
    build_velocity_heatmap_data,
    compute_dashboard_kpis,
    extract_state_segments,
    filter_telemetry,
    load_calibrated_thresholds,
    load_telemetry_dataframe,
)

__all__ = [
    "load_telemetry_dataframe",
    "load_calibrated_thresholds",
    "filter_telemetry",
    "compute_dashboard_kpis",
    "extract_state_segments",
    "build_velocity_heatmap_data",
    "build_employee_summary_table",
]
