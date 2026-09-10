import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIRE_BBOX = "74,28,77,32"

FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE = "VIIRS_SNPP_NRT"


def _get_map_key():
    key = os.environ.get("NASA_FIRMS_MAP_KEY", "")
    if not key:
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("NASA_FIRMS_MAP_KEY"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    return key


def fetch_firms_nrt(day_range=1, date=None):
    map_key = _get_map_key()
    if not map_key:
        return None

    try:
        url = f"{FIRMS_API_URL}/{map_key}/{FIRMS_SOURCE}/{FIRE_BBOX}/{day_range}"
        if date:
            url += f"/{date}"

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            return 0

        fire_count = len(df)
        return fire_count

    except Exception as e:
        print(f"  [FIRMS] API error: {e}")
        return None


def estimate_fire_count_seasonal(date):
    if isinstance(date, str):
        date = pd.to_datetime(date)

    month = date.month
    day = date.day
    if month == 10:
        if day <= 10:
            base = 30 + (day / 10) * 70
        elif day <= 25:
            base = 100 + ((day - 10) / 15) * 200
        else:
            base = 300 + ((day - 25) / 6) * 200
    elif month == 11:
        if day <= 15:
            base = 500 - (day / 15) * 200
        else:
            base = 300 - ((day - 15) / 15) * 250
    elif month == 4 and day >= 15:
        base = 10 + ((day - 15) / 15) * 40
    elif month == 5:
        if day <= 20:
            base = 50 + (day / 20) * 30
        else:
            base = 80 - ((day - 20) / 11) * 60
    elif month in [12, 1, 2, 3]:
        base = 5
    elif month in [6, 7, 8, 9]:
        base = 2
    else:
        base = 8

    np.random.seed(date.year * 1000 + date.timetuple().tm_yday)
    noise = np.random.normal(0, base * 0.15)
    result = max(0, base + noise)

    return round(result, 1)


def get_fire_count_for_date(date):
    if isinstance(date, str):
        date_str = date
        date_obj = pd.to_datetime(date)
    else:
        date_str = date.strftime("%Y-%m-%d")
        date_obj = date

    days_ago = (datetime.now() - date_obj).days
    if 0 <= days_ago <= 60:
        firms_count = fetch_firms_nrt(day_range=1, date=date_str)
        if firms_count is not None:
            return float(firms_count)

    return estimate_fire_count_seasonal(date_obj)


def get_fire_counts_range(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    map_key = _get_map_key()
    use_api = bool(map_key)

    if use_api:
        print("  [FIRMS] Using NASA API for fire data...")
        current = dates[0]
        end = dates[-1]
        all_api_data = []

        while current <= end:
            batch_end = min(current + timedelta(days=9), end)
            day_range = (batch_end - current).days + 1
            date_str = current.strftime("%Y-%m-%d")

            try:
                url = f"{FIRMS_API_URL}/{map_key}/{FIRMS_SOURCE}/{FIRE_BBOX}/{day_range}/{date_str}"
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()

                if resp.text.strip():
                    df = pd.read_csv(io.StringIO(resp.text))
                    if not df.empty and "acq_date" in df.columns:
                        daily_counts = df.groupby("acq_date").size().reset_index(name="fire_count")
                        all_api_data.append(daily_counts)
            except Exception as e:
                print(f"  [FIRMS] Batch fetch error at {date_str}: {e}")

            current = batch_end + timedelta(days=1)

        if all_api_data:
            api_df = pd.concat(all_api_data, ignore_index=True)
            api_df["date"] = pd.to_datetime(api_df["acq_date"])
            api_lookup = dict(zip(api_df["date"].dt.date, api_df["fire_count"]))

            for d in dates:
                count = api_lookup.get(d.date(), estimate_fire_count_seasonal(d))
                records.append({"date": d, "fire_count": float(count)})
        else:
            # API failed, use seasonal for all
            print("  [FIRMS] API returned no data, using seasonal estimates...")
            for d in dates:
                records.append({"date": d, "fire_count": estimate_fire_count_seasonal(d)})
    else:
        print("  [FIRMS] No MAP_KEY found, using seasonal fire count estimates...")
        for d in dates:
            records.append({"date": d, "fire_count": estimate_fire_count_seasonal(d)})

    result = pd.DataFrame(records)
    print(f"  [FIRMS] Fire data: {len(result)} days, "
          f"avg={result['fire_count'].mean():.1f}, max={result['fire_count'].max():.1f}")
    return result
