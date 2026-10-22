import os
import re
from typing import Any

import pandas as pd
import requests
import streamlit as st

from src.aqi import get_category
from src.config import (
    CPCB_STORE,
    FORECAST_STATIONS,
    OPENMETEO_ARCHIVE,
    OPENMETEO_FORECAST,
    POLLUTANT_LABELS,
    RECENT_STORE,
    STORE_DIR,
)
from src.sources import store
from src.sources.cpcb_live import fetch_live

LIVE_TTL = 60
HISTORY_TTL = 300


def is_matching_station(station_a: Any, station_b: Any) -> bool:
    """
    Fuzzy-matches station names by locality (e.g. 'Anand Vihar, Delhi' == 'Anand Vihar, New Delhi - DPCC').
    Normalizes punctuation and casing.
    """
    clean_a = " ".join(re.findall(r"\w+", str(station_a).split(",")[0].lower()))
    clean_b = " ".join(re.findall(r"\w+", str(station_b).split(",")[0].lower()))
    return clean_a == clean_b or clean_a in clean_b or clean_b in clean_a


# Backward-compatibility alias
_same = is_matching_station


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def live_stations() -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetches the latest reading for each monitoring station from the live feed.
    Enriches each record with AQI category and formatted dominant pollutant labels.
    """
    df, meta = fetch_live()
    if len(df) == 0:
        return df, meta

    latest = (
        df.sort_values("datetime")
        .groupby("station", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    categories = []
    dominant_labels = []

    for _, row in latest.iterrows():
        aqi_val = row.get("aqi")
        if pd.notna(aqi_val):
            categories.append(get_category(int(aqi_val)))
        else:
            categories.append("Unknown")

        dominant_raw = str(row.get("dominant", "pm2_5")).lower().strip()
        dominant_key = "pm2_5" if "pm2" in dominant_raw else dominant_raw
        dominant_labels.append(POLLUTANT_LABELS.get(dominant_key, dominant_key.upper()))

    latest["category"] = categories
    latest["dominant_label"] = dominant_labels
    latest["aqi_reason"] = ""

    meta["with_aqi"] = int(latest["aqi"].notna().sum())
    meta["total_stations"] = len(latest)
    return latest.sort_values("station").reset_index(drop=True), meta


def city_summary(stations: pd.DataFrame | None) -> dict[str, Any] | None:
    """Calculates city-wide high-level statistics (cleanest, worst, spread, median AQI)."""
    if stations is None or len(stations) == 0:
        return None

    valid_stations = stations.dropna(subset=["aqi"])
    if len(valid_stations) == 0:
        return None

    best = valid_stations.loc[valid_stations["aqi"].idxmin()]
    worst = valid_stations.loc[valid_stations["aqi"].idxmax()]
    median = float(valid_stations["aqi"].median())

    return {
        "median_aqi": round(median),
        "median_category": get_category(median),
        "best_station": best["station"],
        "best_aqi": int(best["aqi"]),
        "worst_station": worst["station"],
        "worst_aqi": int(worst["aqi"]),
        "spread": int(worst["aqi"] - best["aqi"]),
        "n_reporting": int(len(valid_stations)),
        "n_total": int(len(stations)),
        "updated": stations["datetime"].max(),
    }


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def history(stations: list[str] | None = None, days: int | None = None,
            with_aqi: bool = True, collected_only: bool = False) -> pd.DataFrame:
    """
    Loads historical hourly air quality observations.
    When `collected_only=True`, reads solely from the rolling recent store.
    """
    start_date = None
    if days:
        start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

    primary_path = RECENT_STORE if collected_only else CPCB_STORE
    recent_path = None if collected_only else RECENT_STORE

    # Match user station requests with stored station names
    if stations:
        stored = store.load(primary_path, recent=recent_path)
        available_stations = stored["station"].dropna().unique()
        stations = [have for have in available_stations if any(is_matching_station(want, have) for want in stations)] or None

    df = store.load(primary_path, stations=stations, start=start_date, recent=recent_path)
    if len(df) == 0:
        return df

    # Ensure aqi and aqi_instant columns exist
    if "aqi" not in df.columns:
        df["aqi"] = df["pm2_5"] if "pm2_5" in df.columns else None
    else:
        df["aqi"] = df["aqi"].fillna(df["pm2_5"] if "pm2_5" in df.columns else None)

    df["aqi_instant"] = df["aqi"]
    return df


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def store_status() -> dict[str, Any]:
    """Returns metadata about the historical and rolling data store."""
    df = store.load()
    collected_rows = len(store.load(RECENT_STORE, recent=None))

    if len(df) == 0:
        return {
            "empty": True,
            "rows": 0,
            "collected_rows": 0,
            "stations": 0,
            "first": None,
            "last": None,
            "lag_hours": None,
            "station_list": [],
        }

    last_datetime = df["datetime"].max()
    lag = round((pd.Timestamp.now() - last_datetime).total_seconds() / 3600.0, 1)

    return {
        "empty": False,
        "rows": int(len(df)),
        "collected_rows": int(collected_rows),
        "stations": int(df["station"].nunique()),
        "first": df["datetime"].min(),
        "last": last_datetime,
        "lag_hours": lag,
        "station_list": sorted(df["station"].dropna().unique().tolist()),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def weather(lat: float, lon: float, days_back: int = 14, forecast_hours: int = 24) -> pd.DataFrame:
    """Fetches continuous weather timeline (past observed + next forecast) from Open-Meteo."""
    now = pd.Timestamp.now()
    weather_fields = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
    frames = []

    # Historical weather
    try:
        archive_resp = requests.get(
            OPENMETEO_ARCHIVE,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": (now - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"),
                "end_date": (now - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "hourly": weather_fields,
                "timezone": "Asia/Kolkata",
            },
            timeout=15,
        ).json()
        frames.append(_weather_frame(archive_resp))
    except Exception:
        pass

    # Ahead forecast weather
    try:
        ahead_resp = requests.get(
            OPENMETEO_FORECAST,
            params={
                "latitude": lat,
                "longitude": lon,
                "past_days": 2,
                "forecast_days": max(1, forecast_hours // 24 + 1),
                "hourly": weather_fields,
                "timezone": "Asia/Kolkata",
            },
            timeout=15,
        ).json()
        frames.append(_weather_frame(ahead_resp))
    except Exception:
        pass

    valid_frames = [f for f in frames if f is not None and len(f)]
    if not valid_frames:
        return pd.DataFrame()

    out = pd.concat(valid_frames, ignore_index=True)
    return (
        out.dropna(subset=["datetime"])
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def _weather_frame(payload: dict[str, Any] | None) -> pd.DataFrame | None:
    """Formats Open-Meteo hourly response into standardized DataFrame columns."""
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


def forecast_station_options(status: dict[str, Any] | None = None) -> list[str]:
    """Returns candidate stations eligible for neural network 24h forecasting."""
    current_status = status or store_status()
    available_stations = set(current_status.get("station_list") or [])
    if not available_stations:
        return []

    matched = [s for s in FORECAST_STATIONS if any(is_matching_station(s, a) for a in available_stations)]
    resolved = []
    for want in matched:
        for have in available_stations:
            if is_matching_station(want, have):
                resolved.append(have)
                break
    return resolved or sorted(available_stations)


@st.cache_data(ttl=1800, show_spinner=False)
def station_weather(station: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieves observed weather history and upcoming 48-hour weather forecast for a station."""
    weather_parquet = os.path.join(STORE_DIR, "weather_hourly.parquet")
    observed_weather = pd.DataFrame()
    if os.path.exists(weather_parquet):
        stored = pd.read_parquet(weather_parquet)
        observed_weather = stored[stored["station"] == station].drop(columns=["station"])

    coords = store.load(stations=[station])
    if len(coords) == 0:
        return observed_weather, pd.DataFrame()

    lat = float(coords["lat"].dropna().iloc[0])
    lon = float(coords["lon"].dropna().iloc[0])

    ahead = weather(lat, lon, days_back=10, forecast_hours=48)
    if len(ahead):
        ahead = ahead.rename(columns={"windspeed_kph": "wind_kph"})
    return observed_weather, ahead


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_24h(station: str) -> dict[str, Any] | None:
    """
    Executes the 24-hour neural forecast for the specified station.
    Feeds past 168 hours history and ahead weather forecast into the Conv1D-LSTM model.
    """
    from src import forecast as fc

    if not fc.available():
        return None

    meta = fc.load_meta()
    matching_station = next((s for s in meta["stations"] if is_matching_station(s, station)), None)
    if matching_station is None:
        return None

    history_df = history_raw(matching_station)
    if len(history_df) < 24:
        return None

    observed_weather, ahead_weather = station_weather(matching_station)
    result = fc.predict(
        history_df,
        matching_station,
        meta=meta,
        weather_history=observed_weather,
        weather_forecast=ahead_weather,
    )
    if result is None:
        return None

    forecast_frame = fc.forecast_aqi(history_df, result["pm2_5"], result["hours"])
    forecast_frame["persistence"] = fc.persistence_baseline(history_df)

    result["frame"] = forecast_frame
    result["scorecard"] = fc.load_scorecard()
    result["station_matched"] = matching_station
    result["last_observed"] = history_df["datetime"].max()
    return result


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def history_raw(station: str) -> pd.DataFrame:
    """Loads unbroken raw hourly history for a single station."""
    return store.load(stations=[station]).sort_values("datetime")
