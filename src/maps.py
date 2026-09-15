"""Pydeck maps over Delhi-NCR monitoring stations.

The spatial view is the payoff of using real station data: a model grid
gives every corner of the city the same number, whereas the ground
network routinely shows a 150+ AQI spread across it.
"""

import pandas as pd
import pydeck as pdk

from src.aqi import get_category
from src.config import AQI_RGB, DELHI_CENTER

TOOLTIP = {
    "html": "<b>{short_name}</b><br/>"
            "AQI <b>{aqi_text}</b> &nbsp;{category}<br/>"
            "PM2.5 {pm25_text} &nbsp;&middot;&nbsp; PM10 {pm10_text}<br/>"
            "<span style='opacity:.65'>{updated}</span>",
    "style": {"backgroundColor": "#1E293B", "color": "#E2E8F0",
              "fontSize": "12px", "borderRadius": "8px", "padding": "8px"},
}


def _prepare(stations):
    """Attach the display fields the deck tooltip and layers need."""
    d = stations.dropna(subset=["lat", "lon"]).copy()
    if len(d) == 0:
        return d

    d["short_name"] = d["station"].astype(str).str.split(",").str[0]
    aqi = pd.to_numeric(d.get("aqi"), errors="coerce")
    d["aqi_value"] = aqi
    d["category"] = [get_category(v) if pd.notna(v) else "Unknown" for v in aqi]
    d["color"] = [AQI_RGB.get(c, AQI_RGB["Unknown"]) for c in d["category"]]
    d["aqi_text"] = [f"{v:.0f}" if pd.notna(v) else "no data" for v in aqi]

    for col, label in (("pm2_5", "pm25_text"), ("pm10", "pm10_text")):
        vals = pd.to_numeric(d.get(col), errors="coerce")
        d[label] = [f"{v:.0f}" if pd.notna(v) else "--" for v in vals]

    if "datetime" in d.columns:
        stamps = pd.to_datetime(d["datetime"], errors="coerce")
        d["updated"] = stamps.dt.strftime("%d %b, %H:%M").fillna("")
    else:
        d["updated"] = ""

    # Radius tracks severity so hotspots read before the colour does.
    d["radius"] = (aqi.fillna(60).clip(30, 500) * 1.7 + 420).astype(float)
    d["weight"] = aqi.fillna(0).clip(0, 500).astype(float)
    return d


def station_map(stations, zoom=9.6, height=520, style="dark"):
    """Scatter map of live station AQI."""
    d = _prepare(stations)
    if len(d) == 0:
        return None

    layer = pdk.Layer(
        "ScatterplotLayer", data=d,
        get_position=["lon", "lat"], get_fill_color="color",
        get_radius="radius", radius_min_pixels=7, radius_max_pixels=44,
        pickable=True, opacity=0.78, stroked=True,
        get_line_color=[226, 232, 240, 70], line_width_min_pixels=1,
    )
    return pdk.Deck(
        layers=[layer], map_style=_style(style), tooltip=TOOLTIP,
        initial_view_state=pdk.ViewState(
            latitude=float(d["lat"].mean()), longitude=float(d["lon"].mean()),
            zoom=zoom, pitch=0, height=height),
    )


def heatmap(stations, zoom=9.4, height=520, style="dark"):
    """Interpolated AQI surface plus the station points that produced it.

    The surface is a visual interpolation between monitors, not a
    measurement; the points stay on top so the real data is never hidden.
    """
    d = _prepare(stations)
    d = d[d["aqi_value"].notna()]
    if len(d) == 0:
        return None

    heat = pdk.Layer(
        "HeatmapLayer", data=d,
        get_position=["lon", "lat"], get_weight="weight",
        radius_pixels=78, intensity=1.0, threshold=0.04, opacity=0.55,
        color_range=[[34, 197, 94], [132, 204, 22], [251, 191, 36],
                     [249, 115, 22], [239, 68, 68], [139, 26, 58]],
    )
    points = pdk.Layer(
        "ScatterplotLayer", data=d,
        get_position=["lon", "lat"], get_fill_color=[226, 232, 240, 190],
        get_radius=320, radius_min_pixels=3, radius_max_pixels=7, pickable=True,
    )
    return pdk.Deck(
        layers=[heat, points], map_style=_style(style), tooltip=TOOLTIP,
        initial_view_state=pdk.ViewState(
            latitude=DELHI_CENTER["lat"], longitude=DELHI_CENTER["lon"],
            zoom=zoom, pitch=0, height=height),
    )


def column_map(stations, zoom=9.3, height=520, style="dark"):
    """3D columns: height and colour both encode AQI."""
    d = _prepare(stations)
    d = d[d["aqi_value"].notna()]
    if len(d) == 0:
        return None
    d["elevation"] = d["aqi_value"] * 26

    layer = pdk.Layer(
        "ColumnLayer", data=d,
        get_position=["lon", "lat"], get_elevation="elevation",
        elevation_scale=1, radius=560, get_fill_color="color",
        pickable=True, auto_highlight=True, opacity=0.86,
    )
    return pdk.Deck(
        layers=[layer], map_style=_style(style), tooltip=TOOLTIP,
        initial_view_state=pdk.ViewState(
            latitude=DELHI_CENTER["lat"], longitude=DELHI_CENTER["lon"],
            zoom=zoom, pitch=48, bearing=12, height=height),
    )


def _style(name):
    # Carto basemaps need no token, unlike Mapbox styles.
    return {
        "dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    }.get(name, "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json")
