"""Central configuration: paths, pollutants, units, stations, palette."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
STORE_DIR = os.path.join(DATA_DIR, "store")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

CPCB_STORE = os.path.join(STORE_DIR, "cpcb_hourly.parquet")
RECENT_STORE = os.path.join(STORE_DIR, "cpcb_recent.parquet")
TREND_DAYS = {"day": 1, "week": 7, "month": 30}
RECENT_WINDOW_DAYS = max(TREND_DAYS.values()) + 10

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, STORE_DIR, MODELS_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "nh3"]

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2": "SO₂", "o3": "O₃", "co": "CO", "nh3": "NH₃",
}

DATAGOV_POLLUTANTS = {
    "PM2.5": "pm2_5", "PM10": "pm10", "NO2": "no2",
    "SO2": "so2", "OZONE": "o3", "CO": "co", "NH3": "nh3",
}

OPENAQ_POLLUTANTS = {
    "pm25": "pm2_5", "pm2.5": "pm2_5", "pm10": "pm10", "no2": "no2",
    "so2": "so2", "o3": "o3", "co": "co", "nh3": "nh3",
}

PHYSICAL_MAX = {
    "pm2_5": 1200.0, "pm10": 2000.0, "no2": 800.0,
    "so2": 1200.0, "o3": 800.0, "co": 50.0, "nh3": 1000.0,
}

CO_MGM3_RANGE = (0.05, 12.0)
CO_UGM3_RANGE = (150.0, 20000.0)


NCR_CITIES = ["Delhi", "Noida", "Ghaziabad", "Gurugram", "Faridabad",
              "Greater Noida", "Bahadurgarh", "Sonipat", "Bhiwadi"]

DELHI_BOUNDS = {"lat_min": 28.30, "lat_max": 28.95, "lon_min": 76.75, "lon_max": 77.60}
DELHI_CENTER = {"lat": 28.6139, "lon": 77.2090}

FORECAST_STATIONS = [
    "Anand Vihar, Delhi - DPCC",
    "R K Puram, Delhi - DPCC",
    "Punjabi Bagh, Delhi - DPCC",
    "Rohini, Delhi - DPCC",
    "Dwarka-Sector 8, Delhi - DPCC",
    "Jahangirpuri, Delhi - DPCC",
    "ITO, Delhi - CPCB",
    "Okhla Phase-2, Delhi - DPCC",
    "Mundka, Delhi - DPCC",
    "Nehru Nagar, Delhi - DPCC",
]


AQI_CATEGORIES = [
    (0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
    (201, 300, "Poor"), (301, 400, "Very Poor"), (401, 10**9, "Severe"),
]

AQI_COLORS = {
    "Good": "#22C55E", "Satisfactory": "#84CC16", "Moderate": "#FBBF24",
    "Poor": "#F97316", "Very Poor": "#EF4444", "Severe": "#8B1A3A",
    "Unknown": "#475569",
}

AQI_RGB = {
    "Good": [34, 197, 94], "Satisfactory": [132, 204, 22],
    "Moderate": [251, 191, 36], "Poor": [249, 115, 22],
    "Very Poor": [239, 68, 68], "Severe": [139, 26, 58],
    "Unknown": [71, 85, 105],
}


WINDOW_SIZE = 168
FORECAST_HOURS = 24
MAX_INTERPOLATE_GAP = 3
MAX_INTERNAL_GAP = 6
MAX_IMPUTED_FRAC = 0.10


DATAGOV_RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

DATAGOV_DEMO_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
DATAGOV_PAGE_SIZE = 10
DATAGOV_THROTTLE = 0.4
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_THROTTLE = 1.5

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def load_env():
    """Populate os.environ from .env if present. Safe to call repeatedly."""
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def datagov_key():
    load_env()
    return os.environ.get("DATA_GOV_IN_API_KEY") or DATAGOV_DEMO_KEY


def openaq_key():
    load_env()
    return os.environ.get("OPENAQ_API_KEY", "")
