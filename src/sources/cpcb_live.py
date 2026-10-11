import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.config import CACHE_DIR, DELHI_BOUNDS, POLLUTANTS, waqi_key

CACHE_PATH = os.path.join(CACHE_DIR, "waqi_last_sweep.json")
HEADERS = {"User-Agent": "delhi-aqi-dashboard/1.0", "Accept": "application/json"}
CACHE_MAX_AGE_MIN = 15

def _to_float(value):
    try:
        v = float(value)
        return None if v < 0 else v
    except (TypeError, ValueError):
        return None

def _get_waqi(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.load(resp)
    except Exception:
        return {}

def fetch_records():
    key = waqi_key()
    if not key:
        return []

    b = DELHI_BOUNDS
    bounds_url = (
        f"https://api.waqi.info/v2/map/bounds?"
        f"latlng={b['lat_min']},{b['lon_min']},{b['lat_max']},{b['lon_max']}"
        f"&networks=all&token={key}"
    )
    bounds_data = _get_waqi(bounds_url)
    if bounds_data.get("status") != "ok":
        return []

    uids = [x["uid"] for x in bounds_data.get("data", []) if "uid" in x]

    records = []
    def fetch_station(uid):
        return _get_waqi(f"https://api.waqi.info/feed/@{uid}/?token={key}")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_station, uid): uid for uid in uids}
        for future in as_completed(futures):
            res = future.result()
            if res.get("status") == "ok":
                records.append(res.get("data", {}))

    return records

def records_to_frame(records):
    rows = []
    pollutant_map = {
        "pm25": "pm2_5", "pm10": "pm10", "no2": "no2",
        "so2": "so2", "o3": "o3", "co": "co", "nh3": "nh3",
    }

    for rec in records:
        city_info = rec.get("city", {})
        name = city_info.get("name", "Unknown")

        if "NASA" in name or "Calib" in name:
            continue

        time_info = rec.get("time", {})
        stamp = pd.to_datetime(time_info.get("iso"), errors="coerce")
        if pd.isna(stamp):
            continue
        stamp = stamp.floor("h")

        geo = city_info.get("geo", [None, None])
        lat, lon = (geo[0], geo[1]) if len(geo) == 2 else (None, None)

        dom = str(rec.get("dominentpol", "pm25")).lower().strip()
        dominant_key = "pm2_5" if "pm2" in dom else dom

        row = {
            "datetime": stamp,
            "station": name.split(",")[0].strip(),
            "city": "Delhi",
            "lat": _to_float(lat),
            "lon": _to_float(lon),
            "source": "waqi",
            "is_observed": True,
            "aqi": _to_float(rec.get("aqi")),
            "dominant": dominant_key,
        }

        iaqi = rec.get("iaqi", {})
        for waqi_key, canonical in pollutant_map.items():
            row[canonical] = _to_float(iaqi.get(waqi_key, {}).get("v"))

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    for p in POLLUTANTS:
        if p not in df.columns:
            df[p] = None

    return df.sort_values(["station", "datetime"]).reset_index(drop=True)

def _read_cache():
    if not os.path.exists(CACHE_PATH):
        return [], None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload.get("records", [])
        cached_at = pd.to_datetime(payload.get("cached_at"), errors="coerce")
        if pd.isna(cached_at):
            return records, None
        age_min = (pd.Timestamp.now() - cached_at).total_seconds() / 60
        return records, age_min
    except Exception:
        return [], None

def _write_cache(records):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"records": records, "cached_at": pd.Timestamp.now().isoformat()}, fh)

def fetch_live(use_cache_on_error=True, max_age_min=CACHE_MAX_AGE_MIN):
    meta = {
        "stale": False, "error": None, "fetched_at": pd.Timestamp.now(),
        "from_cache": False, "cache_age_min": 0,
    }

    cached, age_min = _read_cache()
    if cached and age_min is not None and age_min <= max_age_min:
        df = records_to_frame(cached)
        meta.update({
            "from_cache": True, "cache_age_min": round(age_min, 1),
            "stations": int(df["station"].nunique()) if len(df) else 0,
            "last_update": df["datetime"].max() if len(df) else None,
        })
        return df, meta

    records = fetch_records()
    if records:
        _write_cache(records)
    elif use_cache_on_error and cached:
        records = cached
        meta["stale"] = True
        meta["error"] = "API error, used disk cache"

    if not records:
        meta.update({"stations": 0, "last_update": None})
        return pd.DataFrame(), meta

    df = records_to_frame(records)
    meta["stations"] = int(df["station"].nunique()) if len(df) else 0
    meta["last_update"] = df["datetime"].max() if len(df) else None
    return df, meta
