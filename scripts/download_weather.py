"""Fetch hourly weather history for every station in the store."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests

from src.config import OPENMETEO_ARCHIVE, STORE_DIR
from src.sources import store

WEATHER_PATH = os.path.join(STORE_DIR, "weather_hourly.parquet")

FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
    "precipitation",
]

COLUMNS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure_mb",
    "wind_speed_10m": "wind_kph",
    "wind_direction_10m": "wind_dir",
    "boundary_layer_height": "blh_m",
    "precipitation": "precip_mm",
}


def fetch(lat, lon, start, end, retries=3):
    params = {
        "latitude": round(float(lat), 4), "longitude": round(float(lon), 4),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(FIELDS), "timezone": "Asia/Kolkata",
    }
    for attempt in range(retries):
        try:
            r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=90)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            hourly = r.json().get("hourly")
            if not hourly:
                return None
            frame = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"])})
            for field, column in COLUMNS.items():
                frame[column] = hourly.get(field)
            return frame
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="refetch stations already present")
    args = ap.parse_args()

    readings = store.load()
    if len(readings) == 0:
        print("Store is empty -- run download_cpcb_history.py first.")
        return 1

    coords = (readings.dropna(subset=["lat", "lon"])
                      .groupby("station")[["lat", "lon"]].first())
    start = readings["datetime"].min().normalize()
    end = min(readings["datetime"].max().normalize(),
              pd.Timestamp.now().normalize() - pd.Timedelta(days=6))

    existing = pd.DataFrame()
    if os.path.exists(WEATHER_PATH) and not args.force:
        existing = pd.read_parquet(WEATHER_PATH)
        print(f"{len(existing):,} weather rows already stored "
              f"for {existing['station'].nunique()} stations")

    done = set(existing["station"].unique()) if len(existing) else set()
    print(f"{start:%Y-%m-%d} to {end:%Y-%m-%d}, {len(coords)} stations\n")

    frames = [existing] if len(existing) else []
    for station, row in coords.iterrows():
        if station in done:
            print(f"  {station.split(',')[0]:<28} already stored")
            continue
        print(f"  {station.split(',')[0]:<28} ", end="", flush=True)
        frame = fetch(row["lat"], row["lon"], start, end)
        if frame is None or len(frame) == 0:
            print("no data")
            continue
        frame["station"] = station
        frames.append(frame)
        print(f"{len(frame):,} hours")
        time.sleep(1.0)

    if not frames:
        print("\nNothing fetched.")
        return 1

    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(["station", "datetime"], keep="last")
             .sort_values(["station", "datetime"])
             .reset_index(drop=True))
    os.makedirs(STORE_DIR, exist_ok=True)
    out.to_parquet(WEATHER_PATH, index=False)

    print(f"\n{len(out):,} rows across {out['station'].nunique()} stations "
          f"-> {WEATHER_PATH}")
    coverage = out.groupby("station")["wind_kph"].apply(lambda s: s.notna().mean())
    print(f"wind coverage: {coverage.min():.0%} to {coverage.max():.0%}")
    if "blh_m" in out.columns:
        blh = out["blh_m"].notna().mean()
        print(f"boundary layer height available for {blh:.0%} of hours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
