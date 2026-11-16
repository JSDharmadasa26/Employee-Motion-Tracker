"""Milestone 3: Interactive Executive Dashboard & Reporting Application (Streamlit + Plotly).

Launch locally with:
    ./venv/bin/streamlit run app.py
or:
    ./venv/bin/python scripts/run_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config  # noqa: F401
import polars as pl
import streamlit as st

from src.config import REPORTS_OUTPUT_DIR, TELEMETRY_OUTPUT_DIR, VIDEOS_OUTPUT_DIR
from src.dashboard.data_loader import (
    build_employee_summary_table,
    compute_dashboard_kpis,
    filter_telemetry,
    format_duration_mmss,
    get_available_employees,
    get_available_videos,
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

st.set_page_config(
    page_title="Employee Motion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_DARK_CSS = """
<style>
    /* Overall carbon-black canvas */
    .stApp {
        background: radial-gradient(circle at 78% 18%, rgba(45, 85, 165, 0.12) 0%, rgba(11, 14, 20, 1) 48%), #0B0E14;
        color: #F0F6FC;
    }
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }
    section[data-testid="stSidebar"] {
        background-color: #10141D;
        border-right: 1px solid #1E2636;
    }

    /* Glassmorphic bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(22, 27, 38, 0.92) 0%, rgba(17, 21, 30, 0.96) 100%);
        border: 1px solid rgba(56, 70, 97, 0.48) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        padding: 0.35rem 0.55rem;
    }

    /* Dashboard Header */
    .dash-title {
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .dash-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-bottom: 1.25rem;
    }
    .meta-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(71, 85, 105, 0.45);
        border-radius: 999px;
        padding: 0.2rem 0.72rem;
        margin-right: 0.45rem;
        font-size: 0.80rem;
        color: #CBD5E1;
    }

    /* Mockup-matched KPI header & big stat layout */
    .kpi-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0.5rem 0.1rem 0.5rem;
    }
    .kpi-label {
        font-size: 1.22rem;
        font-weight: 500;
        color: #9CA8BA;
    }
    .kpi-pills {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .pill-muted {
        width: 24px;
        height: 7px;
        border-radius: 999px;
        background: #333C4D;
        display: inline-block;
    }
    .pill-teal {
        width: 24px;
        height: 7px;
        border-radius: 999px;
        background: #2DD4BF;
        box-shadow: 0 0 10px rgba(45, 212, 191, 0.45);
        display: inline-block;
    }
    .kpi-mid-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.15rem 0.5rem 0.0rem 0.5rem;
    }
    .kpi-big-value {
        font-size: 3.15rem;
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -0.025em;
        color: #F8FAFC;
    }
    .kpi-sub-dur {
        font-size: 0.85rem;
        color: #64748B;
        padding-left: 0.55rem;
        margin-top: 0.1rem;
    }
    .kpi-bar-track {
        width: 92px;
        height: 7px;
        border-radius: 999px;
        background: #252C3B;
        overflow: hidden;
    }
    .kpi-bar-fill-blue {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3B82F6, #60A5FA);
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.6);
    }
    .kpi-bar-fill-purple {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #7C3AED, #C084FC);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    }
    .section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        color: #E2E8F0;
        margin: 0.25rem 0 0.15rem 0.25rem;
    }
    .section-caption {
        font-size: 0.84rem;
        color: #94A3B8;
        margin: 0 0 0.5rem 0.25rem;
    }
</style>
"""
st.markdown(CUSTOM_DARK_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_load_telemetry(path_str: str) -> pl.DataFrame:
    return load_telemetry_dataframe(Path(path_str))


@st.cache_data(show_spinner=False)
def _cached_load_thresholds() -> dict:
    return load_calibrated_thresholds()


def main() -> None:
    default_csv = TELEMETRY_OUTPUT_DIR / "temporal_features.csv"
    if not default_csv.exists():
        st.error(f"Telemetry file not found at `{default_csv}`. Run Milestone 1 & 2 scripts first.")
        return

    raw_df = _cached_load_telemetry(str(default_csv))
    thresholds = _cached_load_thresholds()

    # --- Sidebar Filters & Controls ---
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        videos = get_available_videos(raw_df)
        selected_video = st.selectbox("Video Stream", options=videos, index=0)

        emp_ids = get_available_employees(raw_df, video_name=selected_video)
        emp_options = ["All Employees (Overview)"] + [f"Emp #{eid}" for eid in emp_ids]
        selected_emp_label = st.selectbox("Employee Filter", options=emp_options, index=0)
        selected_emp_id = (
            None
            if selected_emp_label.startswith("All")
            else int(selected_emp_label.replace("Emp #", "").strip())
        )

        vid_df = raw_df.filter(pl.col("video_name") == selected_video)
        t_min_all = float(vid_df["timestamp_sec"].min() or 0.0)
        t_max_all = float(vid_df["timestamp_sec"].max() or 1.0)

        selected_time = st.slider(
            "Timeline Window (Seconds)",
            min_value=round(t_min_all, 1),
            max_value=round(t_max_all, 1),
            value=(round(t_min_all, 1), round(t_max_all, 1)),
            step=1.0,
        )

        donut_view = st.radio(
            "Primary Donut Chart Mode",
            options=[
                "Spatial Posture (Sitting / Standing / Walking / Away)",
                "Utilization (Working / Idle / Walking / Away)",
                "Workstation Utilization (Working / Idle / Away)",
            ],
            index=0,
        )

        bin_sec = st.select_slider(
            "Velocity Heatmap Bin Size (s)",
            options=[1.0, 2.0, 5.0, 10.0],
            value=2.0,
        )

        st.divider()
        st.caption(
            "Privacy-Preserving Architecture: Metrics are derived strictly from mathematical "
            "keypoint telemetry without storing facial biometrics."
        )

    # Apply reactive filters
    filtered_df = filter_telemetry(
        raw_df,
        video_name=selected_video,
        employee_id=selected_emp_id,
        time_range=selected_time,
    )
    kpis = compute_dashboard_kpis(filtered_df)

    # --- Top Header ---
    scope_tag = "All Employees" if selected_emp_id is None else f"Emp #{selected_emp_id}"
    fg_cutoff = thresholds.get("posture_zones", {}).get("foreground", {}).get("threshold_px", 457.9)
    vel_cutoff = thresholds.get("wrist_velocity_threshold_px_s", 32.0)

    st.markdown('<div class="dash-title">Employee Motion Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dash-subtitle">
            <span class="meta-pill">📹 {selected_video}</span>
            <span class="meta-pill">👤 {scope_tag}</span>
            <span class="meta-pill">⏱️ {format_duration_mmss(kpis['video_duration_sec'])} ({kpis['unique_frames']:,} frames)</span>
            <span class="meta-pill">📐 Posture Cutoff: {fg_cutoff:.1f} px</span>
            <span class="meta-pill">⚡ Wrist Cutoff: {vel_cutoff:.1f} px/s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Row 1: Mockup-Matched Hero Grid (Left Donut Card + Right Stacked Working/Idle Sparkline Cards) ---
    col_donut, col_kpis = st.columns([1.48, 1.0], gap="medium")

    with col_donut:
        with st.container(border=True):
            if donut_view.startswith("Spatial Posture"):
                fig_donut = build_posture_donut_figure(kpis, height=420)
            elif donut_view.startswith("Workstation Utilization"):
                fig_donut = build_utilization_donut_figure(kpis, three_way_only=True, height=420)
            else:
                fig_donut = build_utilization_donut_figure(kpis, three_way_only=False, height=420)

            st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})

    with col_kpis:
        # Card 1: Working % + Sparkline
        work_bar_w = min(max(kpis["working_pct"], 12.0), 100.0)
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-top-row">
                    <span class="kpi-label">Working</span>
                    <span class="kpi-pills">
                        <span class="pill-muted"></span>
                        <span class="pill-teal"></span>
                    </span>
                </div>
                <div class="kpi-mid-row">
                    <span class="kpi-big-value">{kpis['working_pct']:.1f}%</span>
                    <div class="kpi-bar-track">
                        <div class="kpi-bar-fill-blue" style="width: {work_bar_w:.0f}%;"></div>
                    </div>
                </div>
                <div class="kpi-sub-dur">Active Work Time: {format_duration_mmss(kpis['working_sec'])} • Mean Wrist: {kpis['mean_wrist_vel_px_s']:.1f} px/s</div>
                """,
                unsafe_allow_html=True,
            )
            fig_sp_work = build_kpi_sparkline_figure(
                kpis["sparkline_x"],
                kpis["working_sparkline"],
                line_color="#4F7DF3",
                fill_rgba="rgba(79, 125, 243, 0.15)",
                metric_name="Working",
                height=72,
            )
            st.plotly_chart(fig_sp_work, width="stretch", config={"displayModeBar": False})

        # Card 2: Idle % + Sparkline
        idle_bar_w = min(max(kpis["idle_pct"] * 2.2, 16.0), 100.0)
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="kpi-top-row">
                    <span class="kpi-label">Idle</span>
                    <span class="kpi-pills">
                        <span class="pill-teal"></span>
                        <span class="pill-muted"></span>
                    </span>
                </div>
                <div class="kpi-mid-row">
                    <span class="kpi-big-value">{kpis['idle_pct']:.1f}%</span>
                    <div class="kpi-bar-track">
                        <div class="kpi-bar-fill-purple" style="width: {idle_bar_w:.0f}%;"></div>
                    </div>
                </div>
                <div class="kpi-sub-dur">Idle / Break Time: {format_duration_mmss(kpis['idle_sec'])} • Away: {kpis['away_pct']:.1f}% ({format_duration_mmss(kpis['posture_breakdown']['Away']['seconds'])})</div>
                """,
                unsafe_allow_html=True,
            )
            fig_sp_idle = build_kpi_sparkline_figure(
                kpis["sparkline_x"],
                kpis["idle_sparkline"],
                line_color="#2CA89A",
                fill_rgba="rgba(44, 168, 154, 0.15)",
                metric_name="Idle",
                height=72,
            )
            st.plotly_chart(fig_sp_idle, width="stretch", config={"displayModeBar": False})

    st.write("")

    # --- Row 2: Ergonomic Timeline & Dual Utilization Breakdown ---
    with st.container(border=True):
        tab_gantt, tab_elev, tab_dual = st.tabs(
            [
                "🕒 Ergonomic Posture & Activity Timeline",
                "📏 Vertical Shoulder Elevation (Sitting vs. Standing)",
                "🍩 Posture vs. Utilization Side-by-Side",
            ]
        )
        with tab_gantt:
            st.markdown(
                '<div class="section-caption">Chronological state ribbons tracking Sitting, Standing, Walking, Away, Working, and Idle intervals across the video.</div>',
                unsafe_allow_html=True,
            )
            fig_tl = build_ergonomic_timeline_figure(filtered_df, include_activity_ribbon=True, height=305)
            st.plotly_chart(fig_tl, width="stretch")

        with tab_elev:
            st.markdown(
                '<div class="section-caption">Continuous vertical distance from shoulders to bottom frame boundary (px) vs. calibrated Standing threshold.</div>',
                unsafe_allow_html=True,
            )
            fig_el = build_posture_elevation_figure(filtered_df, thresholds=thresholds, height=295)
            st.plotly_chart(fig_el, width="stretch")

        with tab_dual:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-heading">Spatial Posture Share</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    build_posture_donut_figure(kpis, height=360),
                    width="stretch",
                    key="dual_posture_donut",
                )
            with c2:
                st.markdown('<div class="section-heading">Workstation Utilization (Working vs. Idle vs. Away)</div>', unsafe_allow_html=True)
                st.plotly_chart(
                    build_utilization_donut_figure(kpis, three_way_only=False, height=360),
                    width="stretch",
                    key="dual_util_donut",
                )

    st.write("")

    # --- Row 3: Wrist Velocity Activity Heatmap ---
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">🔥 Wrist Velocity Activity Heatmap</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Peak periods of continuous micro-movement typing/mouse velocity (px/s) across Left, Right, and Dominant wrists.</div>',
            unsafe_allow_html=True,
        )
        fig_hm = build_activity_heatmap_figure(
            filtered_df,
            bin_seconds=float(bin_sec),
            velocity_threshold_px_s=float(vel_cutoff),
            height=305,
        )
        st.plotly_chart(fig_hm, width="stretch")

    st.write("")

    # --- Row 4: Executive Employee Summary Table, Annotated Video Preview & Report Downloads ---
    col_tbl, col_export = st.columns([1.65, 1.0], gap="medium")

    summary_table = build_employee_summary_table(vid_df)

    with col_tbl:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">📋 Per-Employee Executive Telemetry Summary</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                summary_table.to_pandas(),
                width="stretch",
                hide_index=True,
            )

    with col_export:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">📥 Executive Reports & Exports</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-caption">Download presentation-ready KPI summaries and executive PDF/PNG reports.</div>',
                unsafe_allow_html=True,
            )

            csv_bytes = summary_table.write_csv().encode("utf-8")
            st.download_button(
                label="⬇️ Download Executive KPI Summary (.CSV)",
                data=csv_bytes,
                file_name="executive_kpi_summary.csv",
                mime="text/csv",
                width="stretch",
            )

            pdf_path = REPORTS_OUTPUT_DIR / "executive_motion_report.pdf"
            png_path = REPORTS_OUTPUT_DIR / "executive_dashboard_snapshot.png"
            if not pdf_path.exists() or not png_path.exists():
                generate_executive_reports()

            if pdf_path.exists():
                st.download_button(
                    label="📄 Download 2-Page Executive Report (.PDF)",
                    data=pdf_path.read_bytes(),
                    file_name="executive_motion_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            if png_path.exists():
                st.download_button(
                    label="🖼️ Download Dashboard Snapshot (.PNG)",
                    data=png_path.read_bytes(),
                    file_name="executive_dashboard_snapshot.png",
                    mime="image/png",
                    width="stretch",
                )

    tracked_mp4 = VIDEOS_OUTPUT_DIR / f"{Path(selected_video).stem}_tracked.mp4"
    if tracked_mp4.exists():
        with st.expander("🎬 Preview Annotated Master Tracked Video (H.264)", expanded=False):
            st.video(str(tracked_mp4))


if __name__ == "__main__":
    main()
