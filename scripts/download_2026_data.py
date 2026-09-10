import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.ground_data import fetch_cpcb_data, fetch_openmeteo_air_quality_fallback
from src.fire_data import get_fire_counts_range

LOCATIONS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.316},
    "Connaught Place": {"lat": 28.6315, "lon": 77.2167},
    "Dwarka": {"lat": 28.5921, "lon": 77.0460},
    "IGI Airport": {"lat": 28.5562, "lon": 77.1000},
    "Okhla Phase III": {"lat": 28.5308, "lon": 77.2713},
    "Rohini": {"lat": 28.7495, "lon": 77.0565},
}

START_DATE = "2026-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(lat, lon, start_date=None, end_date=None):
    sd = start_date or START_DATE
    ed = end_date or END_DATE

    five_days_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    frames = []

    try:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": sd,
            "end_date": min(ed, five_days_ago),
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(WEATHER_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        df.rename(columns={
            "time": "datetime",
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "humidity",
            "surface_pressure": "pressure_mb",
            "wind_speed_10m": "windspeed_kph",
        }, inplace=True)
        frames.append(df)
    except Exception as e:
        print(f"  [Weather] Archive API error: {e}")

    try:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": five_days_ago,
            "end_date": ed,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        df.rename(columns={
            "time": "datetime",
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "humidity",
            "surface_pressure": "pressure_mb",
            "wind_speed_10m": "windspeed_kph",
        }, inplace=True)
        frames.append(df)
    except Exception as e:
        print(f"  [Weather] Forecast API error: {e}")

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.drop_duplicates(subset="datetime", keep="last")
    result.sort_values("datetime", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def download_all():
    all_frames = []

    print("=" * 60)
    print("Fetching fire count data (Punjab/Haryana region)...")
    print("=" * 60)
    fire_df = get_fire_counts_range(START_DATE, END_DATE)
    fire_df["date"] = pd.to_datetime(fire_df["date"]).dt.date

    for name, coords in LOCATIONS.items():
        print(f"\n{'=' * 60}")
        print(f"Downloading {name}...")
        print(f"{'=' * 60}")

        print(f"  [Weather] Fetching from Open-Meteo...")
        weather = fetch_weather(coords["lat"], coords["lon"])
        if weather is None:
            print(f"  [ERROR] Weather data failed for {name}, skipping...")
            continue
        print(f"  [Weather] Got {len(weather)} hourly rows")

        print(f"  [AQ] Trying CPCB ground data via OpenAQ...")
        aq_df = fetch_cpcb_data(name, START_DATE, END_DATE)

        if aq_df is None or len(aq_df) < 100:
            print(f"  [AQ] OpenAQ insufficient, falling back to Open-Meteo satellite data...")
            aq_df = fetch_openmeteo_air_quality_fallback(
                coords["lat"], coords["lon"], START_DATE, END_DATE
            )

        if aq_df is None:
            print(f"  [ERROR] No AQ data for {name}, skipping...")
            continue

        aq_df["datetime"] = pd.to_datetime(aq_df["datetime"]).dt.floor("h")
        print(f"  [AQ] Got {len(aq_df)} hourly rows")

        weather["datetime"] = pd.to_datetime(weather["datetime"]).dt.floor("h")
        merged = weather.merge(aq_df, on="datetime", how="inner")

        merged["date_only"] = pd.to_datetime(merged["datetime"]).dt.date
        merged = merged.merge(
            fire_df, left_on="date_only", right_on="date", how="left"
        )
        merged["fire_count"] = merged["fire_count"].fillna(0)
        merged.drop(columns=["date_only", "date"], inplace=True)

        merged["location"] = name
        merged["lat"] = coords["lat"]
        merged["lon"] = coords["lon"]

        dt = pd.to_datetime(merged["datetime"])
        merged["date_ist"] = dt.dt.strftime("%d/%m/%Y")
        merged["time_ist"] = dt.dt.hour.astype(str) + ":00"

        merged = merged[[
            "date_ist", "time_ist", "location", "lat", "lon",
            "temp_c", "humidity", "pressure_mb", "windspeed_kph",
            "pm2_5", "pm10", "co", "no2", "fire_count",
        ]]

        all_frames.append(merged)
        print(f"  [DONE] {name}: {len(merged)} rows")

    if not all_frames:
        print("\n[FATAL] No data downloaded for any location!")
        return

    result = pd.concat(all_frames, ignore_index=True)

    # Save output
    output_path = os.path.join(BASE_DIR, "data", "raw", "delhi-weather-aqi-2026.csv")
    result.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(result)}")
    print(f"Locations: {result['location'].nunique()}")
    print(f"Date range: {result['date_ist'].iloc[0]} to {result['date_ist'].iloc[-1]}")
    print(f"Columns: {list(result.columns)}")
    print(f"Fire count stats: mean={result['fire_count'].mean():.1f}, max={result['fire_count'].max():.1f}")


if __name__ == "__main__":
    download_all()
