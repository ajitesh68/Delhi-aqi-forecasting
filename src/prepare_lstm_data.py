import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]
WEATHER = ["temp_c", "humidity", "pressure_mb", "windspeed_kph"]
FEATURES = POLLUTANTS + WEATHER

DIWALI_DATES = {
    2015: "2015-11-11", 2016: "2016-10-30", 2017: "2017-10-19",
    2018: "2018-11-07", 2019: "2019-10-27", 2020: "2020-11-14",
    2021: "2021-11-04", 2022: "2022-10-24", 2023: "2023-11-12",
    2024: "2024-11-01", 2025: "2025-10-20", 2026: "2026-11-08",
    2027: "2027-10-29",
}

WINDOW_SIZE = 72
FORECAST_HOURS = 72


def load_and_combine():
    frames = []
    for f in sorted(os.listdir(RAW_DIR)):
        if f.startswith("delhi-weather-aqi") and f.endswith(".csv"):
            path = os.path.join(RAW_DIR, f)
            df = pd.read_csv(path)

            drop_cols = ["aqi_index", "condition_text", "description"]
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

            sample_time = str(df["time_ist"].iloc[0])
            if "-" in sample_time:
                df["datetime"] = pd.to_datetime(df["time_ist"])
            else:
                df["datetime"] = pd.to_datetime(
                    df["date_ist"] + " " + df["time_ist"].astype(str),
                    dayfirst=True,
                    format="mixed",
                )

            frames.append(df)
            print(f"Loaded {f}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["location", "datetime"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def add_time_features(df):
    hour = df["datetime"].dt.hour
    month = df["datetime"].dt.month
    dow = df["datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_diwali_feature(df):
    years = df["datetime"].dt.year.unique()
    diwali_timestamps = []
    for y in years:
        if y in DIWALI_DATES:
            diwali_timestamps.append(pd.Timestamp(DIWALI_DATES[y]))

    def calc_days_to_diwali(dt):
        if not diwali_timestamps:
            return 999
        diffs = [abs((dt - d).days) for d in diwali_timestamps]
        nearest_idx = np.argmin(diffs)
        return (diwali_timestamps[nearest_idx] - dt).days

    df["days_to_diwali"] = df["datetime"].apply(calc_days_to_diwali)
    return df


def create_sequences(data, target_data, window_size, forecast_hours):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_hours + 1):
        X.append(data[i : i + window_size])
        y.append(target_data[i + window_size : i + window_size + forecast_hours])
    return np.array(X), np.array(y)


def prepare_location_data(df, location):
    loc_df = df[df["location"] == location].copy()
    loc_df.sort_values("datetime", inplace=True)
    loc_df.reset_index(drop=True, inplace=True)

    feature_cols = FEATURES + [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_weekend", "days_to_diwali",
    ]

    loc_df[FEATURES] = loc_df[FEATURES].interpolate(method="linear")
    loc_df.dropna(subset=FEATURES, inplace=True)

    scaler = MinMaxScaler()
    loc_df[feature_cols] = scaler.fit_transform(loc_df[feature_cols])

    target_scaler = MinMaxScaler()
    loc_df[POLLUTANTS] = target_scaler.fit_transform(loc_df[POLLUTANTS])

    feature_values = loc_df[feature_cols].values
    target_values = loc_df[POLLUTANTS].values

    X, y = create_sequences(feature_values, target_values, WINDOW_SIZE, FORECAST_HOURS)

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, target_scaler


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_and_combine()
    print(f"Combined: {len(df)} rows")

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding Diwali feature...")
    df = add_diwali_feature(df)

    locations = sorted(df["location"].unique())
    print(f"Locations: {locations}")

    for loc in locations:
        print(f"\nPreparing {loc}...")
        X_train, X_test, y_train, y_test, scaler, target_scaler = prepare_location_data(df, loc)

        loc_tag = loc.lower().replace(" ", "_")
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_test.npy"), X_test)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_test.npy"), y_test)
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{loc_tag}_feature_scaler.pkl"))
        joblib.dump(target_scaler, os.path.join(MODELS_DIR, f"{loc_tag}_target_scaler.pkl"))

        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    run()
