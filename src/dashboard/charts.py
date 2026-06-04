"""
src/dashboard/charts.py
-----------------------
All Plotly figure builders.  Every chart uses _layout() for a consistent theme.

Palette
-------
  Positive SHAP / up:   #10b981  emerald
  Negative SHAP / down: #f43f5e  rose
  Compare A:            #818cf8  indigo-2
  Compare B:            #fb7185  rose-2
  Score tiers: emerald → green → amber → orange → red
"""
from __future__ import annotations

import plotly.graph_objects as go
from src.dashboard.styles import score_color


# ── Base layout ─────────────────────────────────────────────────────────────────

def _layout(**kw) -> dict:
    """Shared Plotly layout — dark, minimal, Inter font."""
    base = dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="Inter, sans-serif", color="#9898a0", size=12),
        title_font    = dict(family="Inter, sans-serif", size=14, color="#e4e4e7"),
        showlegend    = False,
        xaxis = dict(
            gridcolor     = "#2d2d32",
            zerolinecolor = "#3a3a40",
            linecolor     = "#2d2d32",
            tickfont      = dict(size=11, color="#71717a"),
        ),
        yaxis = dict(
            gridcolor     = "#2d2d32",
            zerolinecolor = "#3a3a40",
            linecolor     = "#2d2d32",
            tickfont      = dict(size=11, color="#71717a"),
        ),
    )
    base.update(kw)
    return base


# ── Chart builders ──────────────────────────────────────────────────────────────

def plot_shap_waterfall(hex_data: dict, title: str = "") -> go.Figure:
    """Horizontal bar chart — top SHAP feature contributions."""
    top_shap = hex_data.get("top_shap", [])
    if not top_shap:
        return go.Figure()

    feats = [
        d["feature"].replace("_", " ").replace("poi ", "").replace("census ", "")
        for d in top_shap
    ]
    vals  = [d["shap_value"] for d in top_shap]
    cols  = ["#10b981" if v > 0 else "#f43f5e" for v in vals]

    # Sort ascending so largest absolute value is at top when rendered
    pairs  = sorted(zip(vals, feats, cols), key=lambda x: x[0])
    vals   = [p[0] for p in pairs]
    feats  = [p[1] for p in pairs]
    cols   = [p[2] for p in pairs]

    pred   = hex_data.get("predicted_score", 0)
    actual = hex_data.get("walk_score", 0)

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation="h",
        marker=dict(color=cols, line_width=0, opacity=0.9),
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color="#9898a0", family="Inter"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title   = title or f"SHAP Attribution  ·  Predicted {pred:.1f}  ·  Actual {actual:.0f}",
        height  = 370,
        margin  = dict(l=185, r=80, t=48, b=12),
        xaxis   = dict(
            title         = dict(text="SHAP value (Walk Score pts)", font=dict(size=11, color="#52525a")),
            gridcolor     = "#2d2d32",
            zerolinecolor = "#52525a",
            zerolinewidth = 1.5,
            tickfont      = dict(size=10, color="#71717a"),
        ),
        yaxis   = dict(
            gridcolor = "rgba(0,0,0,0)",
            tickfont  = dict(size=11, color="#a1a1aa"),
        ),
    ))
    return fig


def plot_score_gauge(score: float, label: str = "") -> go.Figure:
    """Clean gauge indicator for a single walk-score."""
    c = score_color(score)

    # Very subtle tinted arcs — avoids the harsh dark blocks
    def _tint(hex_c: str, a: float) -> str:
        r = int(hex_c[1:3], 16)
        g = int(hex_c[3:5], 16)
        b = int(hex_c[5:7], 16)
        return f"rgba({r},{g},{b},{a})"

    tier_colors = [
        (0,   25,  "#ef4444"),
        (25,  50,  "#f97316"),
        (50,  70,  "#f59e0b"),
        (70,  90,  "#22c55e"),
        (90,  100, "#10b981"),
    ]

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": label, "font": {"size": 12, "color": "#52525a", "family": "Inter"}},
        gauge = {
            "axis": {
                "range":    [0, 100],
                "tickwidth": 0,
                "tickcolor": "#2d2d32",
                "tickfont": {"size": 9, "color": "#52525a"},
                "tickvals": [0, 25, 50, 70, 90, 100],
            },
            "bar": {"color": c, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [lo, hi], "color": _tint(tc, 0.10)}
                for lo, hi, tc in tier_colors
            ],
            "threshold": {
                "line": {"color": c, "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        number = {
            "font": {"size": 36, "color": c, "family": "Inter", "weight": "700"},
            "suffix": "",
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        height = 200,
        margin = dict(l=24, r=24, t=32, b=4),
    )
    return fig


def plot_comparison_bars(
    compare_data: dict,
    name_a: str = "A",
    name_b: str = "B",
) -> go.Figure:
    """Grouped bar chart comparing SHAP contributions for two locations."""
    top_diffs = compare_data.get("top_diffs", [])
    if not top_diffs:
        return go.Figure()

    features = [d["feature"].replace("_", " ")[:26] for d in top_diffs[:10]]
    shap_a   = [d["shap_a"] for d in top_diffs[:10]]
    shap_b   = [d["shap_b"] for d in top_diffs[:10]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name_a[:25], x=features, y=shap_a,
        marker=dict(color="#818cf8", opacity=0.9, line_width=0),
        hovertemplate="<b>%{x}</b><br>SHAP: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=name_b[:25], x=features, y=shap_b,
        marker=dict(color="#fb7185", opacity=0.9, line_width=0),
        hovertemplate="<b>%{x}</b><br>SHAP: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title      = "SHAP Contributions — Side by Side",
        height     = 370,
        margin     = dict(l=10, r=10, t=48, b=90),
        barmode    = "group",
        bargap     = 0.25,
        bargroupgap= 0.05,
        showlegend = True,
        legend     = dict(
            orientation  = "h",
            y            = -0.30,
            x            = 0.5,
            xanchor      = "center",
            bgcolor      = "rgba(0,0,0,0)",
            font         = dict(size=11, color="#9898a0"),
            itemsizing   = "constant",
        ),
        xaxis = dict(
            tickangle = -30,
            tickfont  = dict(size=10, color="#71717a"),
            gridcolor = "#2d2d32",
            linecolor = "#2d2d32",
        ),
        yaxis = dict(
            title     = dict(text="SHAP value", font=dict(size=11, color="#52525a")),
            gridcolor = "#2d2d32",
            linecolor = "#2d2d32",
            tickfont  = dict(size=10, color="#71717a"),
        ),
    ))
    return fig


def plot_score_distribution(summary: dict) -> go.Figure:
    """Bar chart of Walk Score distribution across all hexes."""
    dist   = summary.get("score_distribution", {})
    bins   = list(dist.keys())
    counts = list(dist.values())

    def _bc(b: str) -> str:
        v = int(b.split("-")[0])
        if v < 25:  return "#ef4444"
        if v < 50:  return "#f97316"
        if v < 70:  return "#f59e0b"
        return "#22c55e"

    fig = go.Figure(go.Bar(
        x=bins, y=counts,
        marker=dict(
            color=[_bc(b) for b in bins],
            opacity=0.85,
            line_width=0,
        ),
        text=counts,
        textposition="outside",
        textfont=dict(size=10, color="#71717a"),
        hovertemplate="<b>%{x}</b><br>Hexes: %{y:,}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title  = "Walk Score Distribution",
        height = 300,
        margin = dict(l=10, r=16, t=48, b=40),
        xaxis  = dict(
            title     = dict(text="Score range", font=dict(size=11, color="#52525a")),
            tickfont  = dict(size=10, color="#71717a"),
            gridcolor = "rgba(0,0,0,0)",
            linecolor = "#2d2d32",
        ),
        yaxis  = dict(
            title     = dict(text="Hexes", font=dict(size=11, color="#52525a")),
            gridcolor = "#2d2d32",
            linecolor = "#2d2d32",
            tickfont  = dict(size=10, color="#71717a"),
        ),
    ))
    return fig


def plot_top_features(summary: dict) -> go.Figure:
    """Horizontal bar chart of top global features by mean |SHAP|."""
    features = summary.get("top_features", [])
    if not features:
        return go.Figure()

    names   = [f["feature"].replace("_", " ")[:30] for f in features]
    values  = [f["mean_abs_shap"] for f in features]
    colours = ["#10b981" if f["mean_shap"] > 0 else "#f43f5e" for f in features]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker=dict(color=colours, opacity=0.85, line_width=0),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#71717a"),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title  = "Top Features — Global Importance (mean |SHAP|)",
        height = 320,
        margin = dict(l=195, r=75, t=48, b=12),
        xaxis  = dict(
            title     = dict(text="Mean |SHAP|", font=dict(size=11, color="#52525a")),
            gridcolor = "#2d2d32",
            linecolor = "#2d2d32",
            tickfont  = dict(size=10, color="#71717a"),
        ),
        yaxis  = dict(
            gridcolor = "rgba(0,0,0,0)",
            tickfont  = dict(size=11, color="#a1a1aa"),
        ),
    ))
    return fig
