import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from src.config import CACHE_DIR, DELHI_BOUNDS, POLLUTANTS, waqi_key

CACHE_PATH = os.path.join(CACHE_DIR, "waqi_last_sweep.json")
HEADERS = {"User-Agent": "delhi-aqi-dashboard/1.0", "Accept": "application/json"}
CACHE_MAX_AGE_MIN = 15

POLLUTANT_MAP = {
    "pm25": "pm2_5",
    "pm10": "pm10",
    "no2": "no2",
    "so2": "so2",
    "o3": "o3",
    "co": "co",
    "nh3": "nh3",
}


def _safe_float(value: Any) -> float | None:
    """Converts a value to float, returning None if negative or non-numeric."""
    try:
        val = float(value)
        return None if val < 0 else val
    except (TypeError, ValueError):
        return None


def _get_waqi(url: str) -> dict[str, Any]:
    """Sends a GET request to the WAQI API and parses the JSON response."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.load(response)
    except Exception:
        return {}


def fetch_records() -> list[dict[str, Any]]:
    """
    Fetches live air quality records for all Delhi stations using concurrent worker threads.
    First retrieves station UIDs within Delhi geographic bounds, then fetches each station's feed.
    """
    api_key = waqi_key()
    if not api_key:
        return []

    bounds = DELHI_BOUNDS
    bounds_url = (
        f"https://api.waqi.info/v2/map/bounds?"
        f"latlng={bounds['lat_min']},{bounds['lon_min']},{bounds['lat_max']},{bounds['lon_max']}"
        f"&networks=all&token={api_key}"
    )

    bounds_data = _get_waqi(bounds_url)
    if bounds_data.get("status") != "ok":
        return []

    station_uids = [station["uid"] for station in bounds_data.get("data", []) if "uid" in station]
    if not station_uids:
        return []

    def fetch_station_feed(uid: int) -> dict[str, Any]:
        return _get_waqi(f"https://api.waqi.info/feed/@{uid}/?token={api_key}")

    records = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_uid = {executor.submit(fetch_station_feed, uid): uid for uid in station_uids}
        for future in as_completed(future_to_uid):
            result = future.result()
            if result.get("status") == "ok":
                records.append(result.get("data", {}))

    return records


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Transforms raw WAQI JSON station records into a standardized pandas DataFrame."""
    rows = []

    for record in records:
        city_info = record.get("city", {})
        station_name = city_info.get("name", "Unknown")

        # Exclude non-ground sensor tests and calibrations
        if "NASA" in station_name or "Calib" in station_name:
            continue

        time_info = record.get("time", {})
        timestamp = pd.to_datetime(time_info.get("iso"), errors="coerce")
        if pd.isna(timestamp):
            continue
        timestamp = timestamp.floor("h")

        geo_coords = city_info.get("geo", [None, None])
        lat, lon = (geo_coords[0], geo_coords[1]) if len(geo_coords) == 2 else (None, None)

        dominant_raw = str(record.get("dominentpol", "pm25")).lower().strip()
        dominant_key = "pm2_5" if "pm2" in dominant_raw else dominant_raw

        row = {
            "datetime": timestamp,
            "station": station_name.split(",")[0].strip(),
            "city": "Delhi",
            "lat": _safe_float(lat),
            "lon": _safe_float(lon),
            "source": "waqi",
            "is_observed": True,
            "aqi": _safe_float(record.get("aqi")),
            "dominant": dominant_key,
        }

        # Extract individual pollutant sub-indices (IAQI)
        iaqi = record.get("iaqi", {})
        for api_param, canonical_name in POLLUTANT_MAP.items():
            row[canonical_name] = _safe_float(iaqi.get(api_param, {}).get("v"))

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Ensure all canonical pollutant columns exist
    for pollutant in POLLUTANTS:
        if pollutant not in df.columns:
            df[pollutant] = None

    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


def _read_cache() -> tuple[list[dict[str, Any]], float | None]:
    """Reads previously cached station records and calculates cache age in minutes."""
    if not os.path.exists(CACHE_PATH):
        return [], None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        records = payload.get("records", [])
        cached_at = pd.to_datetime(payload.get("cached_at"), errors="coerce")
        if pd.isna(cached_at):
            return records, None
        age_minutes = (pd.Timestamp.now() - cached_at).total_seconds() / 60.0
        return records, age_minutes
    except Exception:
        return [], None


def _write_cache(records: list[dict[str, Any]]) -> None:
    """Saves fetched station records to disk cache with current timestamp."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump({"records": records, "cached_at": pd.Timestamp.now().isoformat()}, file)


def fetch_live(use_cache_on_error: bool = True, max_age_min: float = CACHE_MAX_AGE_MIN) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Primary entry point for live air quality data.
    Checks disk cache first. If stale or missing, queries WAQI API concurrently.
    Falls back to disk cache on network or rate limit errors.
    """
    metadata = {
        "stale": False,
        "error": None,
        "fetched_at": pd.Timestamp.now(),
        "from_cache": False,
        "cache_age_min": 0.0,
    }

    cached_records, age_minutes = _read_cache()
    if cached_records and age_minutes is not None and age_minutes <= max_age_min:
        df = records_to_frame(cached_records)
        metadata.update({
            "from_cache": True,
            "cache_age_min": round(age_minutes, 1),
            "stations": int(df["station"].nunique()) if len(df) else 0,
            "last_update": df["datetime"].max() if len(df) else None,
        })
        return df, metadata

    live_records = fetch_records()
    if live_records:
        _write_cache(live_records)
    elif use_cache_on_error and cached_records:
        live_records = cached_records
        metadata["stale"] = True
        metadata["error"] = "Live API unreachable, served from disk cache."

    if not live_records:
        metadata.update({"stations": 0, "last_update": None})
        return pd.DataFrame(), metadata

    df = records_to_frame(live_records)
    metadata["stations"] = int(df["station"].nunique()) if len(df) else 0
    metadata["last_update"] = df["datetime"].max() if len(df) else None
    return df, metadata
