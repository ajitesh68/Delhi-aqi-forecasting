"""Plotly figures."""

import numpy as np
import plotly.graph_objects as go

from src.aqi import get_category
from src.config import AQI_COLORS
from src.styles import MUTED, SERIES, TEXT

CATEGORY_BANDS = [
    (0, 50, "Good"), (50, 100, "Satisfactory"), (100, 200, "Moderate"),
    (200, 300, "Poor"), (300, 400, "Very Poor"), (400, 500, "Severe"),
]


def _band_shapes(ymax, opacity=0.09):
    shapes = []
    for lo, hi, label in CATEGORY_BANDS:
        if lo > ymax:
            break
        shapes.append(dict(
            type="rect", xref="paper", yref="y", x0=0, x1=1,
            y0=lo, y1=min(hi, ymax * 1.05), line=dict(width=0),
            fillcolor=AQI_COLORS[label], opacity=opacity, layer="below"))
    return shapes


def _empty(message="Not enough data yet"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color=MUTED, size=13), xref="paper", yref="paper",
                       x=0.5, y=0.5)
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=260)
    return fig


def diurnal_chart(profile, best_window=None, title="Average AQI by hour of day"):
    """Hour-of-day profile with an inter-quartile ribbon."""
    if profile is None or len(profile) == 0:
        return _empty()
    d = profile.sort_values("hour")
    ymax = float(d["p75"].max() if "p75" in d.columns else d["mean"].max()) * 1.15

    fig = go.Figure()
    if "p25" in d.columns and "p75" in d.columns:
        fig.add_trace(go.Scatter(
            x=list(d["hour"]) + list(d["hour"])[::-1],
            y=list(d["p75"]) + list(d["p25"])[::-1],
            fill="toself", fillcolor="rgba(56,189,248,0.13)",
            line=dict(width=0), hoverinfo="skip", name="25-75 percentile"))
    fig.add_trace(go.Scatter(
        x=d["hour"], y=d["mean"], mode="lines+markers", name="Mean AQI",
        line=dict(color=SERIES[0], width=2.5), marker=dict(size=6),
        hovertemplate="%{x}:00 - AQI %{y:.0f}<extra></extra>"))

    if best_window:
        fig.add_vrect(x0=best_window[0] - 0.5, x1=best_window[1] + 0.5,
                      fillcolor="rgba(34,197,94,0.16)", line_width=0,
                      annotation_text="cleanest", annotation_position="top left",
                      annotation_font=dict(color="#4ADE80", size=11))

    fig.update_layout(
        title=title, height=320, shapes=_band_shapes(ymax),
        xaxis=dict(title="Hour of day", tickmode="array",
                   tickvals=list(range(0, 24, 3)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)]),
        yaxis=dict(title="AQI", range=[0, ymax]), showlegend=False)
    return fig


def station_ranking_chart(ranking, highlight=None, title="Stations right now"):
    if ranking is None or len(ranking) == 0:
        return _empty()
    d = ranking.sort_values("mean", ascending=True).tail(24)
    colors = [AQI_COLORS.get(get_category(v), MUTED) for v in d["mean"]]
    widths = [0.9 if s == highlight else 0.72 for s in d["station"]]
    names = [str(s).split(",")[0] for s in d["station"]]

    fig = go.Figure(go.Bar(
        x=d["mean"], y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)), width=widths,
        text=[f"{v:.0f}" for v in d["mean"]], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}<br>AQI %{x:.0f}<extra></extra>"))
    fig.update_layout(title=title, height=max(300, 23 * len(d)),
                      xaxis=dict(title="AQI"), yaxis=dict(title=""),
                      margin=dict(l=4, r=42))
    return fig


def _estimated_split(g, value):
    """Split a series into its observed part and its interpolated part."""
    est = g["estimated"].fillna(False).to_numpy(dtype=bool)
    bridge = est | np.r_[est[1:], False] | np.r_[False, est[:-1]]
    solid = g[value].where(~est)
    dashed = g[value].where(bridge)
    return solid, dashed


def timeseries_chart(df, value="aqi", by=None, title="AQI over time", height=340):
    if df is None or len(df) == 0 or value not in df.columns:
        return _empty()
    d = df.dropna(subset=[value]).copy()
    if len(d) == 0:
        return _empty()
    xcol = "datetime" if "datetime" in d.columns else d.columns[0]
    ymax = float(d[value].max()) * 1.1
    marked = "estimated" in d.columns and bool(d["estimated"].any())

    fig = go.Figure()
    if by and by in d.columns and d[by].nunique() > 1:
        for i, (name, g) in enumerate(d.groupby(by)):
            g = g.sort_values(xcol)
            short = str(name).split(",")[0]
            colour = SERIES[i % len(SERIES)]
            y = g[value]
            if marked:
                y, dashed = _estimated_split(g, value)
                fig.add_trace(go.Scatter(
                    x=g[xcol], y=dashed, mode="lines", name=short,
                    line=dict(width=1.4, color=colour, dash="dot"),
                    opacity=0.55, showlegend=False, connectgaps=False,
                    hovertemplate="%{y:.0f} estimated<extra>" + short + "</extra>"))
            fig.add_trace(go.Scatter(
                x=g[xcol], y=y, mode="lines", name=short,
                line=dict(width=1.7, color=colour), connectgaps=False,
                hovertemplate="%{y:.0f}<extra>" + short + "</extra>"))
    else:
        d = d.sort_values(xcol)
        y = d[value]
        if marked:
            y, dashed = _estimated_split(d, value)
            fig.add_trace(go.Scatter(
                x=d[xcol], y=dashed, mode="lines", name="Estimated",
                line=dict(width=1.5, color=MUTED, dash="dot"),
                connectgaps=False,
                hovertemplate="%{x|%d %b}<br>AQI %{y:.0f}"
                              "<br><i>estimated, no station data</i><extra></extra>"))
        fig.add_trace(go.Scatter(
            x=d[xcol], y=y, mode="lines", name="AQI",
            line=dict(width=1.9, color=SERIES[0]), connectgaps=False,
            hovertemplate="%{x|%d %b %H:%M}<br>AQI %{y:.0f}<extra></extra>"))
        fig.update_layout(showlegend=marked, legend=dict(
            orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0))

    fig.update_layout(title=title, height=height, shapes=_band_shapes(ymax),
                      yaxis=dict(title="AQI", range=[0, ymax]), xaxis=dict(title=""),
                      hovermode="x unified")
    return fig


def forecast_chart(hours, values, observed=None, title="Next 24 hours"):
    """Forecast line, optionally with the observed run-up before it."""
    if hours is None or len(hours) == 0:
        return _empty()
    fig = go.Figure()
    tops = [float(np.max(values))]
    if observed is not None and len(observed) > 0:
        fig.add_trace(go.Scatter(
            x=observed["datetime"], y=observed["aqi"], mode="lines",
            name="Observed", line=dict(color=MUTED, width=1.8),
            hovertemplate="%{x|%d %b %H:%M}<br>observed %{y:.0f}<extra></extra>"))
        tops.append(float(observed["aqi"].max()))
    colors = [AQI_COLORS.get(get_category(v), MUTED) for v in values]
    fig.add_trace(go.Scatter(
        x=hours, y=values, mode="lines+markers", name="Forecast",
        line=dict(color=SERIES[0], width=2.6, dash="dot"),
        marker=dict(size=7, color=colors),
        hovertemplate="%{x|%d %b %H:%M}<br>forecast %{y:.0f}<extra></extra>"))
    ymax = max(tops) * 1.15
    fig.update_layout(title=title, height=340, shapes=_band_shapes(ymax),
                      yaxis=dict(title="AQI", range=[0, ymax]), xaxis=dict(title=""),
                      hovermode="x unified")
    return fig


def exposure_chart(breakdown, title="Where your daily dose comes from"):
    if not breakdown:
        return _empty()
    labels, values = list(breakdown.keys()), list(breakdown.values())
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=[SERIES[0], SERIES[1], SERIES[3]][:len(labels)]),
        text=[f"{v:.0f}" for v in values], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}<br>%{x:.0f} ug-hours<extra></extra>"))
    fig.update_layout(title=title, height=230,
                      xaxis=dict(title="PM2.5 concentration x hours"),
                      yaxis=dict(title=""), margin=dict(l=4, r=46))
    return fig
