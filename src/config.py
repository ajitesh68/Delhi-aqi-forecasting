"""Central configuration: paths, constants, stations, and API settings."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORE_DIR = os.path.join(DATA_DIR, "store")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

CPCB_STORE = os.path.join(STORE_DIR, "cpcb_hourly.parquet")
RECENT_STORE = os.path.join(STORE_DIR, "cpcb_recent.parquet")
TREND_DAYS = {"day": 1, "week": 7, "month": 30}
RECENT_WINDOW_DAYS = 40

for _d in (DATA_DIR, STORE_DIR, MODELS_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "nh3"]

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2": "SO₂", "o3": "O₃", "co": "CO", "nh3": "NH₃",
}

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
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 10**9, "Severe"),
]

AQI_COLORS = {
    "Good": "#22C55E",
    "Satisfactory": "#84CC16",
    "Moderate": "#FBBF24",
    "Poor": "#F97316",
    "Very Poor": "#EF4444",
    "Severe": "#8B1A3A",
    "Unknown": "#475569",
}

AQI_RGB = {
    "Good": [34, 197, 94],
    "Satisfactory": [132, 204, 22],
    "Moderate": [251, 191, 36],
    "Poor": [249, 115, 22],
    "Very Poor": [239, 68, 68],
    "Severe": [139, 26, 58],
    "Unknown": [71, 85, 105],
}

WINDOW_SIZE = 168
FORECAST_HOURS = 24
MAX_INTERPOLATE_GAP = 3
MAX_INTERNAL_GAP = 6
MAX_IMPUTED_FRAC = 0.10

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def load_env():
    """Load key-value pairs from .env file."""
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


def waqi_key():
    """Retrieve WAQI API key from environment or .env."""
    load_env()
    return os.environ.get("WAQI_API_KEY", "")
