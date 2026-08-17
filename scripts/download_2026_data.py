import requests
import pandas as pd
from datetime import datetime

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
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(WEATHER_URL, params=params)
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
    return df


def fetch_air_quality(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,carbon_monoxide",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df.rename(columns={
        "time": "datetime",
        "nitrogen_dioxide": "no2",
        "carbon_monoxide": "co",
    }, inplace=True)
    return df


def download_all():
    all_frames = []

    for name, coords in LOCATIONS.items():
        print(f"Downloading {name}...")

        weather = fetch_weather(coords["lat"], coords["lon"])
        air = fetch_air_quality(coords["lat"], coords["lon"])

        merged = weather.merge(air, on="datetime", how="inner")
        merged["location"] = name
        merged["lat"] = coords["lat"]
        merged["lon"] = coords["lon"]

        dt = pd.to_datetime(merged["datetime"])
        merged["date_ist"] = dt.dt.strftime("%d/%m/%Y")
        merged["time_ist"] = dt.dt.strftime("%-H:00")

        merged = merged[[
            "date_ist", "time_ist", "location", "lat", "lon",
            "temp_c", "humidity", "pressure_mb", "windspeed_kph",
            "pm2_5", "pm10", "co", "no2",
        ]]

        all_frames.append(merged)
        print(f"  {name}: {len(merged)} rows")

    result = pd.concat(all_frames, ignore_index=True)
    output_path = "data/raw/delhi-weather-aqi-2026.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total rows: {len(result)}")
    print(f"Date range: {result['date_ist'].iloc[0]} to {result['date_ist'].iloc[-1]}")


if __name__ == "__main__":
    download_all()
