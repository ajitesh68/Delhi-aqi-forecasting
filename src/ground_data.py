import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

DELHI_STATIONS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.316, "openaq_id": None},
    "Connaught Place": {"lat": 28.6315, "lon": 77.2167, "openaq_id": None},
    "Dwarka": {"lat": 28.5921, "lon": 77.0460, "openaq_id": None},
    "IGI Airport": {"lat": 28.5562, "lon": 77.1000, "openaq_id": None},
    "Okhla Phase III": {"lat": 28.5308, "lon": 77.2713, "openaq_id": None},
    "Rohini": {"lat": 28.7495, "lon": 77.0565, "openaq_id": None},
}

POLLUTANT_MAP = {
    "pm25": "pm2_5",
    "pm10": "pm10",
    "no2": "no2",
    "co": "co",
}


def _get_api_key():
    key = os.environ.get("OPENAQ_API_KEY", "")
    if not key:
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAQ_API_KEY"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    return key


def _get_headers():
    key = _get_api_key()
    headers = {"Accept": "application/json"}
    if key:
        headers["X-API-Key"] = key
    return headers


def find_nearest_station(lat, lon, radius_m=10000):
    try:
        resp = requests.get(
            f"{OPENAQ_BASE_URL}/locations",
            headers=_get_headers(),
            params={
                "coordinates": f"{lat},{lon}",
                "radius": radius_m,
                "limit": 5,
                "order_by": "distance",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        for station in results:
            sensors = station.get("sensors", [])
            has_pm25 = any(
                s.get("parameter", {}).get("name", "").lower() in ["pm25", "pm2.5"]
                for s in sensors
            )
            if has_pm25:
                return {
                    "id": station["id"],
                    "name": station.get("name", "Unknown"),
                    "sensors": {
                        s.get("parameter", {}).get("name", "").lower(): s["id"]
                        for s in sensors
                    },
                }

        return {
            "id": station["id"],
            "name": station.get("name", "Unknown"),
            "sensors": {
                s.get("parameter", {}).get("name", "").lower(): s["id"]
                for s in station.get("sensors", [])
            },
        }
    except Exception as e:
        print(f"  [OpenAQ] Station lookup failed: {e}")
        return None


def fetch_openaq_hourly(sensor_id, date_from, date_to):
    all_data = []
    page = 1
    max_pages = 200

    while page <= max_pages:
        try:
            resp = requests.get(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
                headers=_get_headers(),
                params={
                    "date_from": date_from,
                    "date_to": date_to,
                    "limit": 1000,
                    "page": page,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                period = r.get("period", {})
                dt_start = period.get("datetimeFrom", {}).get("local", "")
                value = r.get("value", None)
                if dt_start and value is not None:
                    all_data.append((dt_start, value))

            # Check if there are more pages
            meta = data.get("meta", {})
            found = meta.get("found", 0)
            if page * 1000 >= found:
                break

            page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [OpenAQ] Sensor {sensor_id} page {page} error: {e}")
            break

    return all_data


def fetch_cpcb_data(location_name, start_date, end_date):
    if location_name not in DELHI_STATIONS:
        print(f"  [OpenAQ] Unknown location: {location_name}")
        return None

    station_info = DELHI_STATIONS[location_name]
    lat, lon = station_info["lat"], station_info["lon"]

    print(f"  [OpenAQ] Finding CPCB station near {location_name}...")
    station = find_nearest_station(lat, lon)

    if station is None:
        print(f"  [OpenAQ] No station found near {location_name}")
        return None

    print(f"  [OpenAQ] Found station: {station['name']} (ID: {station['id']})")
    print(f"  [OpenAQ] Available sensors: {list(station['sensors'].keys())}")

    pollutant_dfs = {}
    for oaq_name, our_name in POLLUTANT_MAP.items():
        sensor_id = station["sensors"].get(oaq_name)
        if sensor_id is None:
            alt_names = {"pm25": ["pm2.5", "pm25"], "co": ["co"], "no2": ["no2"], "pm10": ["pm10"]}
            for alt in alt_names.get(oaq_name, []):
                sensor_id = station["sensors"].get(alt)
                if sensor_id:
                    break

        if sensor_id is None:
            print(f"  [OpenAQ] No sensor for {oaq_name} at {station['name']}")
            continue

        print(f"  [OpenAQ] Fetching {oaq_name} (sensor {sensor_id})...")
        readings = fetch_openaq_hourly(sensor_id, start_date, end_date)

        if readings:
            df = pd.DataFrame(readings, columns=["datetime", our_name])
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
            df["datetime"] = df["datetime"].dt.floor("h")
            df = df.drop_duplicates(subset="datetime", keep="last")
            pollutant_dfs[our_name] = df
            print(f"  [OpenAQ] Got {len(df)} hourly readings for {oaq_name}")
        else:
            print(f"  [OpenAQ] No data for {oaq_name}")

    if not pollutant_dfs:
        print(f"  [OpenAQ] No pollutant data obtained for {location_name}")
        return None

    result = None
    for col_name, df in pollutant_dfs.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on="datetime", how="outer")

    result.sort_values("datetime", inplace=True)
    result.reset_index(drop=True, inplace=True)

    for our_name in POLLUTANT_MAP.values():
        if our_name not in result.columns:
            result[our_name] = float("nan")

    print(f"  [OpenAQ] Final CPCB data for {location_name}: {len(result)} rows")
    return result


def fetch_openmeteo_air_quality_fallback(lat, lon, start_date, end_date):
    try:
        resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
                "timezone": "Asia/Kolkata",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["hourly"]

        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["time"]),
            "pm2_5": data["pm2_5"],
            "pm10": data["pm10"],
            "co": data["carbon_monoxide"],
            "no2": data["nitrogen_dioxide"],
        })
        return df
    except Exception as e:
        print(f"  [Fallback] Open-Meteo AQ also failed: {e}")
        return None
