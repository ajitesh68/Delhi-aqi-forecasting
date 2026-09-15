"""The app's only data entry point.

Streamlit talks to this module and nothing else. It sits on top of three
sources with very different characteristics:

  data.gov.in   truly live, one snapshot per station, no history
  OpenAQ        ~19 months of hourly history, several days behind
  Open-Meteo    weather only, observed and forecast

Reads come from the parquet store wherever possible so a page load never
depends on an API being up or within quota.
"""

import pandas as pd
import streamlit as st

from src.aqi import add_rolling_aqi, calculate_aqi, get_category
from src.config import (FORECAST_STATIONS, NCR_CITIES, OPENMETEO_ARCHIVE,
                        OPENMETEO_FORECAST, POLLUTANTS)
from src.sources import store
from src.sources.cpcb_live import fetch_live

# data.gov.in publishes min/max/avg per station-pollutant alongside the
# reading time. CPCB computes the published AQI from a 24-hour average, and
# `avg_value` is that aggregate, so it is used directly rather than being
# treated as an instantaneous reading.
AVG_IS_24H = True

LIVE_TTL = 1800
HISTORY_TTL = 900


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def live_stations(cities=("Delhi",)):
    """Current AQI at every reporting station in `cities`.

    Defaults to Delhi alone. The demo data.gov.in key returns 10 records
    per call, so each extra city costs another few seconds of paging on a
    cold cache; the wider NCR sweep is opt-in rather than the default.

    Returns (frame, meta). One row per station with pollutant levels, the
    CPCB AQI, its category and the dominant pollutant.
    """
    df, meta = fetch_live(list(cities) if cities else NCR_CITIES)
    if len(df) == 0:
        return df, meta

    latest = (df.sort_values("datetime")
                .groupby("station", as_index=False)
                .tail(1)
                .reset_index(drop=True))

    rows = []
    for _, row in latest.iterrows():
        averages = {p: row[p] for p in POLLUTANTS
                    if p in row.index and pd.notna(row[p])}
        result = calculate_aqi(averages)
        rows.append({
            "aqi": result["aqi"],
            "category": result["category"],
            "dominant": result["dominant"],
            "dominant_label": result["dominant_label"],
            "aqi_reason": result["reason"],
            "n_pollutants": len(averages),
        })
    latest = pd.concat([latest, pd.DataFrame(rows, index=latest.index)], axis=1)

    meta["with_aqi"] = int(latest["aqi"].notna().sum())
    meta["total_stations"] = len(latest)
    return latest.sort_values("station").reset_index(drop=True), meta


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def history(stations=None, days=None, with_aqi=True):
    """Observed hourly history from the parquet store.

    This is the source for every historical chart in the app. Nothing here
    is model output.
    """
    start = None
    if days:
        start = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

    # Station names differ between feeds -- the live data.gov.in sweep says
    # "Anand Vihar, Delhi - DPCC" where the OpenAQ archive says
    # "Anand Vihar, New Delhi - DPCC" -- so resolve by locality, not equality.
    if stations:
        stored = store.load()
        available = stored["station"].dropna().unique()
        stations = [have for have in available
                    if any(_same(want, have) for want in stations)] or None
    df = store.load(stations=stations, start=start)
    if len(df) == 0 or not with_aqi:
        return df
    return add_rolling_aqi(df)


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def store_status():
    """What the store currently holds, for honest UI labelling."""
    df = store.load()
    if len(df) == 0:
        return {"empty": True, "rows": 0, "stations": 0, "first": None,
                "last": None, "lag_hours": None}
    last = df["datetime"].max()
    return {
        "empty": False,
        "rows": int(len(df)),
        "stations": int(df["station"].nunique()),
        "first": df["datetime"].min(),
        "last": last,
        "lag_hours": round((pd.Timestamp.now() - last).total_seconds() / 3600, 1),
        "station_list": sorted(df["station"].dropna().unique().tolist()),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def weather(lat, lon, days_back=14, forecast_hours=24):
    """Observed and forecast weather from Open-Meteo.

    Open-Meteo is kept for weather only. Its air-quality product is a
    coarse chemistry-transport model that correlates just 0.40 with the
    ground stations here, which is why pollutants come from CPCB instead.
    """
    import requests

    now = pd.Timestamp.now()
    fields = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
    frames = []

    try:
        archive = requests.get(OPENMETEO_ARCHIVE, params={
            "latitude": lat, "longitude": lon,
            "start_date": (now - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"),
            "end_date": (now - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "hourly": fields, "timezone": "Asia/Kolkata",
        }, timeout=30).json()
        frames.append(_weather_frame(archive))
    except Exception:
        pass

    try:
        ahead = requests.get(OPENMETEO_FORECAST, params={
            "latitude": lat, "longitude": lon,
            "past_days": 2, "forecast_days": max(1, forecast_hours // 24 + 1),
            "hourly": fields, "timezone": "Asia/Kolkata",
        }, timeout=30).json()
        frames.append(_weather_frame(ahead))
    except Exception:
        pass

    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return (out.dropna(subset=["datetime"])
               .drop_duplicates("datetime", keep="last")
               .sort_values("datetime").reset_index(drop=True))


def _weather_frame(payload):
    hourly = (payload or {}).get("hourly")
    if not hourly:
        return None
    return pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temp_c": hourly.get("temperature_2m"),
        "humidity": hourly.get("relative_humidity_2m"),
        "pressure_mb": hourly.get("surface_pressure"),
        "windspeed_kph": hourly.get("wind_speed_10m"),
        "wind_dir": hourly.get("wind_direction_10m"),
    })


def city_summary(stations):
    """Headline numbers across the live station network."""
    if stations is None or len(stations) == 0:
        return None
    valid = stations.dropna(subset=["aqi"])
    if len(valid) == 0:
        return None
    best = valid.loc[valid["aqi"].idxmin()]
    worst = valid.loc[valid["aqi"].idxmax()]
    median = float(valid["aqi"].median())
    return {
        "median_aqi": round(median),
        "median_category": get_category(median),
        "best_station": best["station"],
        "best_aqi": int(best["aqi"]),
        "worst_station": worst["station"],
        "worst_aqi": int(worst["aqi"]),
        "spread": int(worst["aqi"] - best["aqi"]),
        "n_reporting": int(len(valid)),
        "n_total": int(len(stations)),
        "updated": stations["datetime"].max(),
    }


def forecast_station_options(status=None):
    """Stations that have enough stored history to forecast from."""
    status = status or store_status()
    available = set(status.get("station_list") or [])
    if not available:
        return []
    matched = [s for s in FORECAST_STATIONS
               if any(_same(s, a) for a in available)]
    resolved = []
    for want in matched:
        for have in available:
            if _same(want, have):
                resolved.append(have)
                break
    return resolved or sorted(available)


def _same(a, b):
    """Match station names across sources ('Delhi' vs 'New Delhi')."""
    return str(a).split(",")[0].strip().lower() == str(b).split(",")[0].strip().lower()


@st.cache_data(ttl=1800, show_spinner=False)
def station_weather(station):
    """Observed and forecast weather for one station.

    The observed half fills the model's input window; the forecast half is
    what the model uses to predict a change rather than a continuation.
    """
    import os

    from src.config import STORE_DIR

    path = os.path.join(STORE_DIR, "weather_hourly.parquet")
    observed = pd.DataFrame()
    if os.path.exists(path):
        stored = pd.read_parquet(path)
        observed = stored[stored["station"] == station].drop(columns=["station"])

    coords = store.load(stations=[station])
    if len(coords) == 0:
        return observed, pd.DataFrame()
    lat = float(coords["lat"].dropna().iloc[0])
    lon = float(coords["lon"].dropna().iloc[0])

    ahead = weather(lat, lon, days_back=10, forecast_hours=48)
    if len(ahead):
        ahead = ahead.rename(columns={"windspeed_kph": "wind_kph"})
    return observed, ahead


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_24h(station):
    """Next 24 hours for one station, with the persistence baseline beside it.

    Returns None when no model is trained, the station was not in the
    training set, or there is not enough recent history to fill the input
    window honestly. Every one of those is a reason to show nothing rather
    than something invented.
    """
    from src import forecast as fc

    if not fc.available():
        return None

    meta = fc.load_meta()
    match = next((s for s in meta["stations"] if _same(s, station)), None)
    if match is None:
        return None

    history = history_raw(match)
    if len(history) < 24:
        return None

    observed_weather, ahead = station_weather(match)
    result = fc.predict(history, match, meta=meta,
                        weather_history=observed_weather,
                        weather_forecast=ahead)
    if result is None:
        return None

    frame = fc.forecast_aqi(history, result["pm2_5"], result["hours"])
    frame["persistence"] = fc.persistence_baseline(history)
    result["frame"] = frame
    result["scorecard"] = fc.load_scorecard()
    result["station_matched"] = match
    result["last_observed"] = history["datetime"].max()
    return result


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def history_raw(station):
    """Unrolled hourly readings for one station, for model input."""
    return store.load(stations=[station]).sort_values("datetime")
