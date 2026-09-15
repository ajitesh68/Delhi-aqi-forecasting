"""Real-time CPCB observations from data.gov.in.

This is the *only* genuinely live source available without a registered
key. The public demo key caps every response at 10 records regardless of
the `limit` parameter, so the sweep pages with `offset` and throttles.

On HTTP 429 the last successful sweep is served from disk rather than
failing the page, and the caller is told the data is stale.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src.config import (CACHE_DIR, CO_MGM3_RANGE, CO_UGM3_RANGE, DATAGOV_PAGE_SIZE,
                        DATAGOV_RESOURCE, DATAGOV_THROTTLE, DATAGOV_POLLUTANTS,
                        NCR_CITIES, POLLUTANTS, datagov_key)

ENDPOINT = f"https://api.data.gov.in/resource/{DATAGOV_RESOURCE}"
CACHE_PATH = os.path.join(CACHE_DIR, "datagov_last_sweep.json")
HEADERS = {"User-Agent": "delhi-aqi-dashboard/1.0", "Accept": "application/json"}

MISSING = {"NA", "N/A", "", "-", "null", "None", None}


class RateLimited(Exception):
    pass


def _request(params, timeout=45, retries=2):
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            if attempt == retries:
                raise RateLimited("data.gov.in rate limit reached") from exc
            time.sleep(2 ** attempt * 3)
    raise RateLimited("data.gov.in rate limit reached")


def fetch_records(cities=None, max_pages=60):
    """Page through data.gov.in for the given cities. Returns raw records."""
    cities = list(cities or NCR_CITIES)
    key = datagov_key()
    records, pages = [], 0

    for city in cities:
        offset = 0
        while pages < max_pages:
            payload = _request({
                "api-key": key, "format": "json",
                "limit": DATAGOV_PAGE_SIZE, "offset": offset,
                "filters[city]": city,
            })
            pages += 1
            batch = payload.get("records", [])
            if not batch:
                break
            records.extend(batch)
            offset += len(batch)
            if len(batch) < DATAGOV_PAGE_SIZE or offset >= int(payload.get("total", 0)):
                break
            time.sleep(DATAGOV_THROTTLE)
    return records


def _to_float(value):
    if value in MISSING:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else v


def resolve_co_unit(series):
    """Infer whether a CO series is mg/m3 or ug/m3 from its magnitude.

    Neither source can be trusted to declare this correctly: CPCB publishes
    mg/m3, OpenAQ relabels the same numbers 'ppb', and data.gov.in values
    sit between the two plausible scales. Guessing wrong introduces a
    1000x error, so an unresolvable series is reported as unknown and
    excluded from AQI rather than silently converted.
    """
    vals = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(vals) == 0:
        return "unknown", None
    median = float(vals.median())
    if CO_MGM3_RANGE[0] <= median <= CO_MGM3_RANGE[1]:
        return "mg/m3", median
    if CO_UGM3_RANGE[0] <= median <= CO_UGM3_RANGE[1]:
        return "ug/m3", median
    return "unknown", median


def records_to_frame(records):
    """Pivot the long pollutant-per-row feed into one row per station-hour."""
    rows = {}
    for rec in records:
        canonical = DATAGOV_POLLUTANTS.get(rec.get("pollutant_id"))
        if canonical is None:
            continue
        station = rec.get("station")
        stamp = pd.to_datetime(rec.get("last_update"), dayfirst=True, errors="coerce")
        if station is None or pd.isna(stamp):
            continue
        stamp = stamp.floor("h")
        key = (station, stamp)
        row = rows.setdefault(key, {
            "datetime": stamp, "station": station, "city": rec.get("city"),
            "lat": _to_float(rec.get("latitude")), "lon": _to_float(rec.get("longitude")),
            "source": "datagov", "is_observed": True,
        })
        row[canonical] = _to_float(rec.get("avg_value"))

    df = pd.DataFrame(list(rows.values()))
    if len(df) == 0:
        return df
    for pollutant in POLLUTANTS:
        if pollutant not in df.columns:
            df[pollutant] = None

    unit, median = resolve_co_unit(df["co"])
    if unit == "ug/m3":
        df["co"] = df["co"] / 1000.0          # normalise to mg/m3 for AQI
    elif unit == "unknown":
        df["co"] = None                        # refuse to guess
    df.attrs["co_unit"] = unit
    df.attrs["co_median_raw"] = median
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


CACHE_MAX_AGE_MIN = 30


def fetch_live(cities=None, use_cache_on_error=True, max_age_min=CACHE_MAX_AGE_MIN):
    """Live CPCB sweep, disk cache first.

    A full Delhi sweep is ~31 paged calls against a key that returns 10
    records at a time, so it costs the better part of a minute. The feed
    only updates hourly, so a recent cache is served straight from disk
    and the network is left alone entirely.

    Returns (frame, meta). `meta['from_cache']` marks a disk-served sweep;
    `meta['stale']` means the network failed and the cache was the
    fallback rather than the fast path.
    """
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
