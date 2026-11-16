"""Unit and integration tests for Milestone 3: Executive Dashboard & Reporting."""

from pathlib import Path
import tempfile
import unittest
import polars as pl

import src.config  # noqa: F401
from src.config import PROJECT_ROOT, TELEMETRY_OUTPUT_DIR
from src.dashboard.data_loader import (
    build_employee_summary_table,
    build_velocity_heatmap_data,
    compute_dashboard_kpis,
    extract_state_segments,
    filter_telemetry,
    load_calibrated_thresholds,
    load_telemetry_dataframe,
)
from src.dashboard.plots import (
    build_activity_heatmap_figure,
    build_ergonomic_timeline_figure,
    build_kpi_sparkline_figure,
    build_posture_donut_figure,
    build_posture_elevation_figure,
    build_utilization_donut_figure,
)
from src.dashboard.reporter import generate_executive_reports


class TestMilestone3Dashboard(unittest.TestCase):
    """Verify Milestone 3 telemetry ingestion, KPI percentages, Plotly figures, exports, and Streamlit app."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_telemetry_dataframe(TELEMETRY_OUTPUT_DIR / "temporal_features.csv")
        cls.thresholds = load_calibrated_thresholds()

    def test_data_ingest_and_exact_kpi_percentages(self):
        """Verify KPI aggregation matches verified 6-class telemetry percentages."""
        kpis = compute_dashboard_kpis(self.df)
        self.assertEqual(kpis["total_rows"], 6434)
        self.assertEqual(kpis["unique_frames"], 5874)
        self.assertEqual(kpis["unique_employees"], 2)
        self.assertEqual(kpis["employee_ids"], [1, 2])

        # Posture share sum ~ 100% and matches target values
        self.assertAlmostEqual(kpis["sitting_pct"], 63.0, places=1)
        self.assertAlmostEqual(kpis["standing_pct"], 19.6, places=1)
        self.assertAlmostEqual(kpis["walking_pct"], 13.9, places=1)
        self.assertAlmostEqual(kpis["away_pct"], 3.6, places=1)

        posture_sum = (
            kpis["sitting_pct"] + kpis["standing_pct"] + kpis["walking_pct"] + kpis["away_pct"]
        )
        self.assertAlmostEqual(posture_sum, 100.0, delta=0.3)

        # Activity KPIs match target values
        self.assertAlmostEqual(kpis["working_pct"], 71.7, places=1)
        self.assertAlmostEqual(kpis["idle_pct"], 10.8, places=1)
        self.assertGreater(len(kpis["working_sparkline"]), 10)
        self.assertEqual(len(kpis["sparkline_x"]), len(kpis["working_sparkline"]))

    def test_employee_and_time_window_filtering(self):
        """Verify per-employee filtering builds complete 1-row-per-frame timelines including Away intervals."""
        emp1_df = filter_telemetry(self.df, employee_id=1)
        self.assertEqual(len(emp1_df), 5874)
        k1 = compute_dashboard_kpis(emp1_df)
        self.assertAlmostEqual(
            k1["sitting_pct"] + k1["standing_pct"] + k1["walking_pct"] + k1["away_pct"],
            100.0,
            delta=0.3,
        )

        emp2_df = filter_telemetry(self.df, employee_id=2)
        self.assertEqual(len(emp2_df), 5874)
        k2 = compute_dashboard_kpis(emp2_df)
        self.assertGreater(k2["away_pct"], 85.0)  # Visitor present ~560 / 5874 frames

        # Sub-window filter [68s, 85s] where Emp #2 is actively conversing at the desk
        emp2_win = filter_telemetry(self.df, employee_id=2, time_range=(68.0, 85.0))
        k2_win = compute_dashboard_kpis(emp2_win)
        self.assertLess(k2_win["away_pct"], 3.0)

    def test_plotly_figures_construct_cleanly(self):
        """Ensure all Plotly figures build valid traces without exceptions."""
        kpis = compute_dashboard_kpis(self.df)

        f_posture = build_posture_donut_figure(kpis)
        self.assertEqual(len(f_posture.data), 1)

        f_util4 = build_utilization_donut_figure(kpis, three_way_only=False)
        self.assertEqual(len(f_util4.data), 1)

        f_util3 = build_utilization_donut_figure(kpis, three_way_only=True)
        self.assertEqual(len(f_util3.data), 1)

        f_spark = build_kpi_sparkline_figure(kpis["sparkline_x"], kpis["working_sparkline"])
        self.assertEqual(len(f_spark.data), 1)

        segs = extract_state_segments(self.df, state_col="posture_state")
        self.assertGreater(len(segs), 5)

        f_tl = build_ergonomic_timeline_figure(self.df)
        self.assertGreater(len(f_tl.data), 5)

        f_el = build_posture_elevation_figure(self.df, thresholds=self.thresholds)
        self.assertGreaterEqual(len(f_el.data), 2)

        hm = build_velocity_heatmap_data(self.df, bin_seconds=2.0)
        self.assertEqual(len(hm["y_labels"]), len(hm["z_matrix"]))
        f_hm = build_activity_heatmap_figure(self.df, bin_seconds=2.0)
        self.assertEqual(len(f_hm.data), 1)

    def test_report_generation_in_isolated_tempdir(self):
        """Verify CSV, JSON, PNG snapshot, and PDF executive report generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = generate_executive_reports(
                csv_path=TELEMETRY_OUTPUT_DIR / "temporal_features.csv",
                output_dir=Path(tmp_dir),
            )
            for p_str in out.values():
                p = Path(p_str)
                self.assertTrue(p.exists(), f"Missing generated report file: {p}")
                self.assertGreater(p.stat().st_size, 200)

            # Check PDF magic bytes (%PDF) and PNG magic bytes
            self.assertTrue(Path(out["executive_pdf_path"]).read_bytes().startswith(b"%PDF"))
            self.assertEqual(Path(out["dashboard_png_path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            tbl = pl.read_csv(out["summary_csv_path"])
            self.assertEqual(len(tbl), 3)  # Aggregate + Emp #1 + Emp #2

    def test_streamlit_app_headless_render(self):
        """Execute `app.py` via Streamlit's native `AppTest` harness and verify zero runtime exceptions."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Streamlit app raised exception: {at.exception}")


if __name__ == "__main__":
    unittest.main()
