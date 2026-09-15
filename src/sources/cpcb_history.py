"""Historical hourly CPCB observations from OpenAQ v3.

OpenAQ republishes CPCB/DPCC station data with roughly 19 months of
hourly depth, which is what makes retraining on real ground-station
readings possible. Two caveats drive the design:

  * Ingestion lags several days and can stall outright, so this is a
    history source only. Live values come from data.gov.in.
  * The API rate-limits aggressively, so every call is throttled and
    429s back off rather than failing the download.

Declared units are not trustworthy: the same station exposes a CO sensor
labelled 'ppb' whose values are plainly mg/m3. Units are therefore
resolved from magnitude, and anything unresolvable is dropped instead of
being converted on a guess.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src.config import (CO_MGM3_RANGE, CO_UGM3_RANGE, DELHI_BOUNDS, OPENAQ_BASE,
                        OPENAQ_POLLUTANTS, OPENAQ_THROTTLE, PHYSICAL_MAX, openaq_key)

PAGE_SIZE = 1000
SENTINELS = (-999.0, -9999.0, -99.0)


def _headers():
    return {"X-API-Key": openaq_key(), "Accept": "application/json",
            "User-Agent": "delhi-aqi-dashboard/1.0"}


def _get(path, params=None, retries=4):
    url = f"{OPENAQ_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    return {}


def discover_stations(bounds=None, min_last="2026-01-01", providers=("CPCB",)):
    """CPCB stations inside the Delhi-NCR box that still report."""
    b = bounds or DELHI_BOUNDS
    lat = (b["lat_min"] + b["lat_max"]) / 2
    lon = (b["lon_min"] + b["lon_max"]) / 2
    payload = _get("/locations", {
        "coordinates": f"{lat},{lon}", "radius": 25000, "limit": 200,
    })
    stations = []
    for loc in payload.get("results", []):
        provider = (loc.get("provider") or {}).get("name")
        if providers and provider not in providers:
            continue
        last = ((loc.get("datetimeLast") or {}).get("local") or "")
        if min_last and last[:10] < min_last:
            continue
        sensors = {}
        for sensor in loc.get("sensors", []):
            name = (sensor.get("parameter") or {}).get("name", "").lower()
            canonical = OPENAQ_POLLUTANTS.get(name)
            if canonical is None:
                continue
            # A station can expose several sensors for one pollutant, most of
            # them retired. Keep the highest id, which is the current one.
            prev = sensors.get(canonical)
            if prev is None or sensor["id"] > prev["id"]:
                sensors[canonical] = {"id": sensor["id"],
                                      "units": (sensor.get("parameter") or {}).get("units")}
        if not sensors:
            continue
        coords = loc.get("coordinates") or {}
        stations.append({
            "location_id": loc["id"], "name": loc.get("name"),
            "lat": coords.get("latitude"), "lon": coords.get("longitude"),
            "last": last, "sensors": sensors,
        })
    return sorted(stations, key=lambda s: s["name"] or "")


def fetch_sensor_hours(sensor_id, date_from, date_to=None, max_pages=40):
    """All hourly readings for one sensor. Returns [(timestamp, value)]."""
    out, page = [], 1
    while page <= max_pages:
        params = {"datetime_from": date_from, "limit": PAGE_SIZE, "page": page}
        if date_to:
            params["datetime_to"] = date_to
        payload = _get(f"/sensors/{sensor_id}/hours", params)
        results = payload.get("results", [])
        if not results:
            break
        for row in results:
            stamp = ((row.get("period") or {}).get("datetimeFrom") or {}).get("local")
            value = row.get("value")
            if stamp and value is not None:
                out.append((stamp, value))
        if len(results) < PAGE_SIZE:
            break
        page += 1
        time.sleep(OPENAQ_THROTTLE)
    return out


def _clean(values, pollutant):
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    s = s.mask(s.isin(SENTINELS))
    s = s.mask(s < 0)
    if pollutant == "co":
        unit = resolve_co_unit(s)
        if unit == "ug/m3":
            s = s / 1000.0
        elif unit == "unknown":
            s[:] = pd.NA
    ceiling = PHYSICAL_MAX.get(pollutant)
    if ceiling is not None:
        s = s.mask(s > ceiling)
    return s


def resolve_co_unit(series):
    vals = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(vals) == 0:
        return "unknown"
    median = float(vals.median())
    if CO_MGM3_RANGE[0] <= median <= CO_MGM3_RANGE[1]:
        return "mg/m3"
    if CO_UGM3_RANGE[0] <= median <= CO_UGM3_RANGE[1]:
        return "ug/m3"
    return "unknown"


def fetch_station_history(station, date_from, date_to=None, pollutants=None, log=print):
    """Wide hourly frame for one station across all its pollutants."""
    wanted = pollutants or list(station["sensors"].keys())
    frames = []
    for pollutant in wanted:
        sensor = station["sensors"].get(pollutant)
        if sensor is None:
            continue
        rows = fetch_sensor_hours(sensor["id"], date_from, date_to)
        if not rows:
            log(f"      {pollutant:<6} no data")
            continue
        df = pd.DataFrame(rows, columns=["datetime", pollutant])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df["datetime"] = df["datetime"].dt.floor("h")
        df[pollutant] = _clean(df[pollutant], pollutant)
        df = df.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
        kept = int(df[pollutant].notna().sum())
        log(f"      {pollutant:<6} {kept:>6} hours  "
            f"{df['datetime'].min():%Y-%m-%d} -> {df['datetime'].max():%Y-%m-%d}")
        frames.append(df[["datetime", pollutant]])
        time.sleep(OPENAQ_THROTTLE)

    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="datetime", how="outer")
    merged["station"] = station["name"]
    merged["lat"] = station["lat"]
    merged["lon"] = station["lon"]
    merged["city"] = "Delhi"
    merged["source"] = "openaq"
    merged["is_observed"] = True
    return merged.sort_values("datetime").reset_index(drop=True)
