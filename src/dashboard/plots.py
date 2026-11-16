"""Interactive dark-glassmorphic Plotly visualization builders for Milestone 3:
  1. Spatial Posture Donut & Workstation Utilization Donut (matching executive dark UI)
  2. Glowing KPI Sparklines (Working % & Idle % trends)
  3. Interactive Ergonomic Posture & Activity Gantt Timeline + Shoulder Elevation Curve
  4. 2D Rolling Wrist Velocity Activity Heatmap
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.data_loader import (
    ACTIVITY_ORDER,
    POSTURE_ORDER,
    build_velocity_heatmap_data,
    extract_state_segments,
    format_duration_mmss,
)

# Color palette engineered to match the target dark executive dashboard UI
POSTURE_COLORS: Dict[str, str] = {
    "Sitting": "#23478E",
    "Standing": "#5D3FD3",
    "Walking": "#1F6E9C",
    "Away": "#2CA89A",
}

UTILIZATION_COLORS: Dict[str, str] = {
    "Working": "#3B82F6",
    "Idle": "#8B5CF6",
    "Walking": "#1F6E9C",
    "Away": "#2CA89A",
}

CARD_BG = "rgba(0,0,0,0)"
DARK_SLICE_BORDER = "#131720"
TEXT_PRIMARY = "#F0F6FC"
TEXT_MUTED = "#94A3B8"
GRID_SUBTLE = "rgba(148, 163, 184, 0.12)"


def build_posture_donut_figure(
    kpis: Dict[str, Any],
    height: int = 410,
) -> go.Figure:
    """Build the Spatial Posture & Presence Donut Chart matching the executive dark UI mockup."""
    pb = kpis.get("posture_breakdown", {})
    labels = list(POSTURE_ORDER)
    counts = [int(pb.get(lbl, {}).get("count", 0)) for lbl in labels]
    secs = [float(pb.get(lbl, {}).get("seconds", 0.0)) for lbl in labels]
    pcts = [float(pb.get(lbl, {}).get("pct", 0.0)) for lbl in labels]
    colors = [POSTURE_COLORS[lbl] for lbl in labels]

    # Guard empty state
    if sum(counts) == 0:
        counts = [0, 0, 0, 1]
        pcts = [0.0, 0.0, 0.0, 100.0]

    callout_text = [
        f"{lbl}<br>{pct:.1f}%" if cnt > 0 else ""
        for lbl, pct, cnt in zip(labels, pcts, counts)
    ]
    hover_text = [
        f"<b>{lbl}</b><br>Share: {pct:.1f}%<br>Duration: {format_duration_mmss(sec)} ({cnt} frames)"
        for lbl, pct, sec, cnt in zip(labels, pcts, secs, counts)
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.56,
                sort=False,
                direction="clockwise",
                rotation=0,
                marker=dict(
                    colors=colors,
                    line=dict(color=DARK_SLICE_BORDER, width=3),
                ),
                text=callout_text,
                textinfo="text",
                textposition="outside",
                textfont=dict(size=14, color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
                hovertext=hover_text,
                hoverinfo="text",
                pull=[0.0, 0.0, 0.0, 0.015],
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=35, r=35, t=35, b=55),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#CBD5E1"),
            itemsizing="constant",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_utilization_donut_figure(
    kpis: Dict[str, Any],
    three_way_only: bool = False,
    height: int = 410,
) -> go.Figure:
    """Build the Workstation Utilization Donut Chart (Working vs. Idle vs. Walking/Away)."""
    if three_way_only:
        u3 = kpis.get("utilization_3way", {})
        labels = ["Working", "Idle", "Away"]
        counts = [int(u3.get(lbl, {}).get("count", 0)) for lbl in labels]
        secs = [float(u3.get(lbl, {}).get("seconds", 0.0)) for lbl in labels]
        pcts = [float(u3.get(lbl, {}).get("share_ex_walk_pct", 0.0)) for lbl in labels]
    else:
        ab = kpis.get("activity_breakdown", {})
        labels = list(ACTIVITY_ORDER)
        counts = [int(ab.get(lbl, {}).get("count", 0)) for lbl in labels]
        secs = [float(ab.get(lbl, {}).get("seconds", 0.0)) for lbl in labels]
        pcts = [float(ab.get(lbl, {}).get("pct", 0.0)) for lbl in labels]

    colors = [UTILIZATION_COLORS[lbl] for lbl in labels]
    if sum(counts) == 0:
        counts[-1] = 1
        pcts[-1] = 100.0

    callout_text = [
        f"{lbl}<br>{pct:.1f}%" if cnt > 0 else ""
        for lbl, pct, cnt in zip(labels, pcts, counts)
    ]
    hover_text = [
        f"<b>{lbl}</b><br>Share: {pct:.1f}%<br>Duration: {format_duration_mmss(sec)} ({cnt} frames)"
        for lbl, pct, sec, cnt in zip(labels, pcts, secs, counts)
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.56,
                sort=False,
                direction="clockwise",
                rotation=0,
                marker=dict(colors=colors, line=dict(color=DARK_SLICE_BORDER, width=3)),
                text=callout_text,
                textinfo="text",
                textposition="outside",
                textfont=dict(size=14, color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
                hovertext=hover_text,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=35, r=35, t=35, b=55),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_kpi_sparkline_figure(
    x_sec: List[float],
    y_pct: List[float],
    line_color: str = "#4F7DF3",
    fill_rgba: str = "rgba(79, 125, 243, 0.14)",
    metric_name: str = "Working",
    height: int = 68,
) -> go.Figure:
    """Build a compact frameless neon sparkline for executive KPI cards."""
    xs = list(x_sec) if x_sec else [0.0, 1.0]
    ys = list(y_pct) if y_pct else [0.0, 0.0]
    y_min = min(ys)
    y_max = max(ys)
    span = max(y_max - y_min, 6.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=3.0, shape="spline", smoothing=0.75),
            fill="tozeroy",
            fillcolor=fill_rgba,
            hovertemplate=f"<b>{metric_name}</b>: %{{y:.1f}}% at %{{x:.1f}}s<extra></extra>",
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=2, r=2, t=4, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(
            visible=False,
            fixedrange=True,
            range=[max(0.0, y_min - 0.35 * span), min(102.0, y_max + 0.30 * span)],
        ),
    )
    return fig


def build_ergonomic_timeline_figure(
    df: pl.DataFrame,
    include_activity_ribbon: bool = True,
    height: int = 300,
) -> go.Figure:
    """Build an interactive Gantt / state-ribbon timeline tracking Posture (Sitting vs Standing vs Walking vs Away)
    and Workstation Activity across elapsed video seconds.
    """
    posture_segs = extract_state_segments(df, state_col="posture_state")
    activity_segs = extract_state_segments(df, state_col="activity_state") if include_activity_ribbon else []

    fig = go.Figure()
    added_legend_states = set()

    def _add_segments(segs: List[Dict[str, Any]], lane_suffix: str, color_map: Dict[str, str]) -> None:
        for seg in segs:
            st = seg["state"]
            lane_name = f"{seg['track_label']} • {lane_suffix}"
            color = color_map.get(st, "#475569")
            show_leg = st not in added_legend_states
            if show_leg:
                added_legend_states.add(st)

            fig.add_trace(
                go.Bar(
                    x=[seg["duration_sec"]],
                    y=[lane_name],
                    base=[seg["start_sec"]],
                    orientation="h",
                    name=st,
                    legendgroup=st,
                    showlegend=show_leg,
                    marker=dict(
                        color=color,
                        line=dict(width=0),
                    ),
                    hovertemplate=(
                        f"<b>{lane_name}</b><br>"
                        f"State: <b>{st}</b> ({seg['motion_detail']})<br>"
                        f"Window: {seg['start_sec']:.1f}s – {seg['end_sec']:.1f}s "
                        f"({seg['duration_sec']:.1f}s)<br>"
                        f"Frames: {seg['start_frame']} – {seg['end_frame']}<extra></extra>"
                    ),
                )
            )

    combined_colors = {**UTILIZATION_COLORS, **POSTURE_COLORS}
    if include_activity_ribbon and activity_segs:
        _add_segments(activity_segs, "Activity", combined_colors)
    _add_segments(posture_segs, "Posture", combined_colors)

    fig.update_layout(
        barmode="overlay",
        height=height,
        margin=dict(l=15, r=20, t=25, b=45),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="rgba(15, 20, 30, 0.55)",
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(
            title="Video Timeline (Seconds)",
            gridcolor=GRID_SUBTLE,
            zeroline=False,
            color=TEXT_MUTED,
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(0,0,0,0)",
            color=TEXT_PRIMARY,
            tickfont=dict(size=12.5),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=12, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_posture_elevation_figure(
    df: pl.DataFrame,
    thresholds: Optional[Dict[str, Any]] = None,
    height: int = 270,
) -> go.Figure:
    """Build a continuous ergonomic shoulder-elevation line chart vs calibrated Sitting/Standing cutoff."""
    fig = go.Figure()
    present = df.filter((pl.col("presence") == 1) & pl.col("track_id").is_not_null()).sort(
        ["track_id", "frame"]
    )

    palette = ["#38BDF8", "#A78BFA", "#34D399", "#FBBF24"]
    if not present.is_empty() and "shoulder_dist_bottom_px" in present.columns:
        for idx, (tid_tuple, grp) in enumerate(present.group_by(["track_id"], maintain_order=True)):
            tid = tid_tuple[0] if isinstance(tid_tuple, tuple) else tid_tuple
            col = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=grp["timestamp_sec"].to_list(),
                    y=grp["shoulder_dist_bottom_px"].to_list(),
                    mode="lines",
                    name=f"Emp #{tid} Shoulder Height (H - Y)",
                    line=dict(color=col, width=2.0),
                    customdata=grp["posture_state"].to_list(),
                    hovertemplate=(
                        f"<b>Emp #{tid}</b> • %{{customdata}}<br>"
                        "Time: %{x:.2f}s<br>Height from Bottom: %{y:.1f} px<extra></extra>"
                    ),
                )
            )

    thresh_px = None
    if thresholds:
        fg = thresholds.get("posture_zones", {}).get("foreground", {})
        thresh_px = fg.get("threshold_px") or thresholds.get("global_posture_threshold_px")

    if thresh_px is not None:
        fig.add_hline(
            y=float(thresh_px),
            line_dash="dash",
            line_color="#F43F5E",
            line_width=1.8,
            annotation_text=f"Sitting ↔ Standing Cutoff ({float(thresh_px):.1f} px)",
            annotation_position="top right",
            annotation_font=dict(color="#FDA4AF", size=11.5),
        )

    fig.update_layout(
        height=height,
        margin=dict(l=15, r=20, t=25, b=40),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="rgba(15, 20, 30, 0.55)",
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(title="Video Timeline (Seconds)", gridcolor=GRID_SUBTLE, color=TEXT_MUTED),
        yaxis=dict(title="Vertical Distance (px)", gridcolor=GRID_SUBTLE, color=TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def build_activity_heatmap_figure(
    df: pl.DataFrame,
    bin_seconds: float = 2.0,
    velocity_threshold_px_s: Optional[float] = None,
    height: int = 295,
) -> go.Figure:
    """Build an interactive 2D Wrist Velocity Activity Heatmap highlighting active work bursts."""
    hm = build_velocity_heatmap_data(df, bin_seconds=bin_seconds)

    z_arr = np.array(hm["z_matrix"], dtype=np.float64)
    z_cap = float(np.percentile(z_arr[z_arr > 0], 96.0)) if np.any(z_arr > 0) else 120.0
    z_max = max(z_cap, (velocity_threshold_px_s or 32.0) * 2.5, 60.0)

    colorscale = [
        [0.00, "#0D131F"],
        [0.16, "#162447"],
        [0.35, "#1E4E8C"],
        [0.60, "#0284C7"],
        [0.82, "#22D3EE"],
        [1.00, "#A5F3FC"],
    ]

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=hm["x_centers"],
                y=hm["y_labels"],
                z=hm["z_matrix"],
                text=hm["hover_text"],
                hoverinfo="text",
                colorscale=colorscale,
                zmin=0.0,
                zmax=round(z_max, 1),
                xgap=1.5,
                ygap=3.0,
                colorbar=dict(
                    title=dict(text="px/s", font=dict(color=TEXT_MUTED, size=11)),
                    tickfont=dict(color=TEXT_MUTED, size=11),
                    thickness=12,
                    len=0.88,
                ),
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=15, r=20, t=20, b=45),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="rgba(13, 19, 31, 0.85)",
        font=dict(color=TEXT_PRIMARY, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(
            title=f"Video Timeline (Seconds, {hm.get('bin_seconds', bin_seconds):.1f}s Bins)",
            gridcolor="rgba(0,0,0,0)",
            color=TEXT_MUTED,
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(0,0,0,0)",
            color=TEXT_PRIMARY,
            tickfont=dict(size=12),
        ),
    )
    return fig
