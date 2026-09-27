"""Real-time CPCB observations from WAQI API."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.config import (CACHE_DIR, CO_MGM3_RANGE, CO_UGM3_RANGE,
                        DELHI_BOUNDS, waqi_key)

CACHE_PATH = os.path.join(CACHE_DIR, "waqi_last_sweep.json")
HEADERS = {"User-Agent": "delhi-aqi-dashboard/1.0", "Accept": "application/json"}

class RateLimited(Exception):
    pass

def page_size():
    return 100

def _get_waqi(url):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
                if data.get("status") == "error" and "quota" in str(data.get("data", "")).lower():
                    raise RateLimited("WAQI quota exceeded")
                return data
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1)
    return {}

def fetch_records(cities=None, max_pages=None):
    """Fetch all stations within Delhi bounds from WAQI."""
    key = waqi_key()
    if not key:
        print("No WAQI API key provided.")
        return []
        
    b = DELHI_BOUNDS
    bounds_url = f"https://api.waqi.info/v2/map/bounds?latlng={b['lat_min']},{b['lon_min']},{b['lat_max']},{b['lon_max']}&networks=all&token={key}"
    bounds_data = _get_waqi(bounds_url)
    
    if bounds_data.get("status") != "ok":
        return []
        
    uids = [x["uid"] for x in bounds_data.get("data", [])]
    
    records = []
    
    def fetch_station(uid):
        url = f"https://api.waqi.info/feed/@{uid}/?token={key}"
        return _get_waqi(url)
        
    # Fetch all stations in parallel for speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_station, uid): uid for uid in uids}
        for future in as_completed(futures):
            res = future.result()
            if res.get("status") == "ok":
                records.append(res.get("data", {}))
                
    return records


def _to_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else v

def records_to_frame(records):
    """Convert WAQI feed records to DataFrame."""
    rows = []
    for rec in records:
        city_info = rec.get("city", {})
        name = city_info.get("name", "Unknown")
        
        # All stations from bounds query are within Delhi NCR area
        # Skip only non-standard entries (e.g. NASA calibration sensors)
        if "NASA" in name or "Calib" in name:
            continue
            
        iaqi = rec.get("iaqi", {})
        time_info = rec.get("time", {})
        
        stamp = pd.to_datetime(time_info.get("iso"), errors="coerce")
        if pd.isna(stamp):
            continue
            
        # Floor to hour as in original
        stamp = stamp.floor("h")
        
        geo = city_info.get("geo", [None, None])
        lat, lon = None, None
        if len(geo) == 2:
            lat, lon = geo
            
        row = {
            "datetime": stamp,
            "station": name.split(",")[0].strip(),
            "city": "Delhi",
            "lat": _to_float(lat),
            "lon": _to_float(lon),
            "source": "waqi",
            "is_observed": True,
        }
        
        # Map WAQI pollutants to our canonical names
        # WAQI keys: pm25, pm10, o3, no2, so2, co, nh3
        mapping = {
            "pm25": "pm2_5", "pm10": "pm10", "no2": "no2",
            "so2": "so2", "o3": "o3", "co": "co", "nh3": "nh3"
        }
        
        for waqi_key, canonical in mapping.items():
            val = iaqi.get(waqi_key, {}).get("v")
            row[canonical] = _to_float(val)
            
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
        
    from src.config import POLLUTANTS
    for pollutant in POLLUTANTS:
        if pollutant not in df.columns:
            df[pollutant] = None
            
    df.attrs["co_unit"] = "mg/m3" # WAQI CO is usually in mg/m3
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


CACHE_MAX_AGE_MIN = 30

def fetch_live(cities=None, use_cache_on_error=True, max_age_min=CACHE_MAX_AGE_MIN):
    """Live sweep, disk cache first."""
    meta = {"stale": False, "error": None, "fetched_at": pd.Timestamp.now(),
            "from_cache": False}

    cached, age_min = _read_cache_with_age()
    if cached and age_min is not None and age_min <= max_age_min:
        df = records_to_frame(cached)
        meta.update({"from_cache": True, "cache_age_min": round(age_min, 1),
                     "co_unit": df.attrs.get("co_unit", "unknown"),
                     "stations": int(df["station"].nunique()) if len(df) else 0,
                     "last_update": df["datetime"].max() if len(df) else None})
        return df, meta

    try:
        records = fetch_records(cities)
        if records:
            _write_cache(records)
        elif use_cache_on_error:
            records = _read_cache()
            meta["stale"] = True
            meta["error"] = "API returned no records"
    except (RateLimited, urllib.error.URLError, TimeoutError, OSError) as exc:
        if not use_cache_on_error:
            raise
        records = _read_cache()
        meta["stale"] = True
        meta["error"] = str(exc)
        
    if not records:
        meta.update({"co_unit": "unknown", "stations": 0, "last_update": None})
        return pd.DataFrame(), meta

    df = records_to_frame(records)
    meta["co_unit"] = df.attrs.get("co_unit", "unknown")
    meta["stations"] = int(df["station"].nunique()) if len(df) else 0
    meta["last_update"] = df["datetime"].max() if len(df) else None
    return df, meta


def _write_cache(records):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"records": records, "cached_at": pd.Timestamp.now().isoformat()}, fh)


def _read_cache():
    return _read_cache_with_age()[0]


def _read_cache_with_age():
    if not os.path.exists(CACHE_PATH):
        return [], None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return [], None
    records = payload.get("records", [])
    cached_at = pd.to_datetime(payload.get("cached_at"), errors="coerce")
    if pd.isna(cached_at):
        return records, None
    return records, (pd.Timestamp.now() - cached_at).total_seconds() / 60
