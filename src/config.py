"""Central configuration: paths, pollutants, units, stations, palette."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
STORE_DIR = os.path.join(DATA_DIR, "store")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# The archive is written once by the backfill and then left alone. Live
# snapshots go to a separate rolling file, trimmed to the last
# RECENT_WINDOW_DAYS, because the collector runs hourly and rewriting a
# 1 MB parquet 24 times a day would add ~700 MB a month to git history.
# Readers see the union of the two; only the small file ever churns.
CPCB_STORE = os.path.join(STORE_DIR, "cpcb_hourly.parquet")
RECENT_STORE = os.path.join(STORE_DIR, "cpcb_recent.parquet")
# Sized by the trend views, not by the model: the 30-day view needs 30 days
# of live hours once the archive falls more than a month behind, which it
# permanently does -- the archive is frozen at its last backfill and only
# this file moves forward. Ten days of margin absorb a multi-day collector
# or feed outage. The 168 hours the model reads fit inside this comfortably.
TREND_DAYS = {"day": 1, "week": 7, "month": 30}
RECENT_WINDOW_DAYS = max(TREND_DAYS.values()) + 10
STATION_REGISTRY = os.path.join(STORE_DIR, "stations.json")

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, STORE_DIR, MODELS_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- pollutants

POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "nh3"]
MODEL_POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2": "SO₂", "o3": "O₃", "co": "CO", "nh3": "NH₃",
}

# data.gov.in pollutant_id -> canonical
DATAGOV_POLLUTANTS = {
    "PM2.5": "pm2_5", "PM10": "pm10", "NO2": "no2",
    "SO2": "so2", "OZONE": "o3", "CO": "co", "NH3": "nh3",
}

# OpenAQ v3 parameter name -> canonical
OPENAQ_POLLUTANTS = {
    "pm25": "pm2_5", "pm2.5": "pm2_5", "pm10": "pm10", "no2": "no2",
    "so2": "so2", "o3": "o3", "co": "co", "nh3": "nh3",
}

# Physical sanity ceilings in ug/m3 (CO in mg/m3). Values above are sensor faults.
# Clipping happens BEFORE log1p; OpenAQ emits -999 sentinels.
PHYSICAL_MAX = {
    "pm2_5": 1200.0, "pm10": 2000.0, "no2": 800.0,
    "so2": 1200.0, "o3": 800.0, "co": 50.0, "nh3": 1000.0,
}

# CO is the one pollutant whose reported unit cannot be trusted:
# CPCB publishes mg/m3; OpenAQ mislabels the same numbers as 'ppb';
# data.gov.in values sit in an ambiguous band. Resolve empirically per station
# from the median, and refuse to guess in between.
CO_MGM3_RANGE = (0.05, 12.0)      # median in this band -> mg/m3
CO_UGM3_RANGE = (150.0, 20000.0)  # median in this band -> ug/m3

# ---------------------------------------------------------------- geography

NCR_CITIES = ["Delhi", "Noida", "Ghaziabad", "Gurugram", "Faridabad",
              "Greater Noida", "Bahadurgarh", "Sonipat", "Bhiwadi"]

DELHI_BOUNDS = {"lat_min": 28.30, "lat_max": 28.95, "lon_min": 76.75, "lon_max": 77.60}
DELHI_CENTER = {"lat": 28.6139, "lon": 77.2090}

# Stations used for forecasting. Chosen for data completeness and spatial
# spread; the live map shows every station the API returns, not just these.
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

# ---------------------------------------------------------------- AQI palette

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

# ---------------------------------------------------------------- model

WINDOW_SIZE = 168        # 7 days of hourly history
FORECAST_HOURS = 24
WINDOW_STRIDE = 3
MAX_INTERPOLATE_GAP = 3  # hours
MAX_INTERNAL_GAP = 6     # reject window if any gap exceeds this
MAX_IMPUTED_FRAC = 0.10  # reject window if more than this fraction imputed

DIWALI_DATES = {
    2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-20",
    2026: "2026-11-08", 2027: "2027-10-29", 2028: "2028-11-15",
}

# ---------------------------------------------------------------- api

DATAGOV_RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

# data.gov.in publishes this demo key in its own API documentation. It is
# not a credential: it is rate-limited to 10 records per request and is
# what lets the app run with no signup. A personal key set as
# DATA_GOV_IN_API_KEY in the environment or .env takes precedence.
DATAGOV_DEMO_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
DATAGOV_PAGE_SIZE = 10    # demo key caps responses at 10 regardless of `limit`
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
