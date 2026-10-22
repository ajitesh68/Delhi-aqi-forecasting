import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.aqi import AVERAGING_HOURS, get_category
from src.config import FORECAST_HOURS, MODELS_DIR, POLLUTANTS, WINDOW_SIZE

WEATHER_FEATURES = [
    "temp_c", "humidity", "pressure_mb", "wind_kph",
    "blh_m", "precip_mm", "wind_u", "wind_v",
]

MODEL_DIR = Path(MODELS_DIR) / "cpcb"
MODEL_PATH = MODEL_DIR / "pm25_24h.keras"
META_PATH = MODEL_DIR / "meta.json"
SCORECARD_PATH = MODEL_DIR / "scorecard.json"

MIN_WINDOW_COVERAGE = 0.25      # At least 25% of the 168-hour input window must have real readings
MIN_BASELINE_COVERAGE = 0.15    # At least 15% of the last 24 hours must be observed for persistence baseline

_CACHED_MODEL = None


def available() -> bool:
    """Checks if the trained model weights and metadata files exist."""
    return MODEL_PATH.exists() and META_PATH.exists()


def load_meta() -> dict[str, Any]:
    """Loads model training metadata including station scalers and feature mappings."""
    with open(META_PATH, encoding="utf-8") as file:
        return json.load(file)


def load_scorecard() -> dict[str, Any] | None:
    """Loads holdout test evaluation scorecard comparing model against persistence baseline."""
    if not SCORECARD_PATH.exists():
        return None
    with open(SCORECARD_PATH, encoding="utf-8") as file:
        return json.load(file)


def load_model():
    """
    Loads the trained Keras Conv1D-LSTM model.
    Applies a temporary monkey-patch to GlorotUniform initializer to maintain compatibility
    between different Keras 3 versions without failing on serialized 'input_axes' kwargs.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from tensorflow import keras
        import keras.src.initializers.random_initializers as random_initializers

        original_init = random_initializers.GlorotUniform.__init__

        def compatible_init(self, seed=None, **kwargs):
            kwargs.pop("input_axes", None)
            kwargs.pop("output_axes", None)
            original_init(self, seed=seed, **kwargs)

        random_initializers.GlorotUniform.__init__ = compatible_init
        try:
            _CACHED_MODEL = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        except TypeError:
            _CACHED_MODEL = keras.models.load_model(MODEL_PATH, compile=False)
        finally:
            random_initializers.GlorotUniform.__init__ = original_init

    return _CACHED_MODEL


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Generates cyclical sin/cos encodings for hour-of-day, day-of-week, and day-of-year."""
    hours = index.hour.to_numpy()
    days_of_week = index.dayofweek.to_numpy()
    days_of_year = index.dayofyear.to_numpy()

    return np.column_stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * days_of_week / 7),
        np.cos(2 * np.pi * days_of_week / 7),
        np.sin(2 * np.pi * days_of_year / 365.25),
        np.cos(2 * np.pi * days_of_year / 365.25),
    ]).astype(np.float32)


def _wind_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Decomposes scalar wind speed and direction into orthogonal u (east-west) and v (north-south) vectors."""
    if "wind_dir" not in frame.columns or "wind_kph" not in frame.columns:
        return frame

    radians = np.deg2rad(frame["wind_dir"].to_numpy(dtype=float))
    speed = frame["wind_kph"].to_numpy(dtype=float)
    df = frame.copy()
    df["wind_u"] = speed * np.sin(radians)
    df["wind_v"] = speed * np.cos(radians)
    return df


def _scale_weather(frame: pd.DataFrame, weather_stats: dict[str, Any] | None) -> np.ndarray:
    """Standardizes weather metrics using the training-time mean and standard deviation."""
    scaled_columns = []
    for feature in WEATHER_FEATURES:
        stat_entry = (weather_stats or {}).get(feature)
        if stat_entry is None or feature not in frame.columns:
            scaled_columns.append(np.zeros(len(frame), dtype=np.float32))
            continue

        raw_values = np.nan_to_num(frame[feature].to_numpy(dtype=float), nan=stat_entry["mean"])
        standardized = (raw_values - stat_entry["mean"]) / stat_entry["scale"]
        scaled_columns.append(standardized.astype(np.float32))

    return np.column_stack(scaled_columns)


def build_future(forecast_weather: pd.DataFrame | None, forecast_hours: pd.DatetimeIndex,
                 weather_stats: dict[str, Any] | None) -> np.ndarray:
    """Constructs the (24, 10) future input tensor containing ahead forecast weather and hour encodings."""
    frame = pd.DataFrame({"datetime": forecast_hours})
    if forecast_weather is not None and len(forecast_weather):
        frame = frame.merge(_wind_components(forecast_weather), on="datetime", how="left")

    scaled_weather = _scale_weather(frame, weather_stats)
    hours = forecast_hours.hour.to_numpy()
    cyclical_hours = np.column_stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
    ]).astype(np.float32)

    return np.hstack([scaled_weather, cyclical_hours]).astype(np.float32)


def choose_anchor(history: pd.DataFrame, min_coverage: float = MIN_WINDOW_COVERAGE,
                  min_baseline: float = MIN_BASELINE_COVERAGE) -> pd.Timestamp | None:
    """
    Finds the latest valid timestamp ('anchor') from which to start the 24h forecast.
    Ensures that both the 168-hour window and the last 24-hour baseline period have sufficient observed data.
    """
    clean_history = history.sort_values("datetime").drop_duplicates("datetime")
    if len(clean_history) == 0:
        return None

    pm_columns = [col for col in ("pm2_5", "pm10") if col in clean_history.columns]
    if not pm_columns:
        return None

    observed_mask = clean_history.set_index("datetime")[pm_columns].notna().any(axis=1)
    full_timeline = observed_mask.reindex(
        pd.date_range(observed_mask.index.min(), observed_mask.index.max(), freq="h"),
        fill_value=False
    )

    if len(full_timeline) < WINDOW_SIZE:
        return None

    window_coverage = full_timeline.rolling(WINDOW_SIZE).mean()
    baseline_coverage = full_timeline.rolling(FORECAST_HOURS).mean()
    valid_anchors = full_timeline.index[(window_coverage >= min_coverage) & (baseline_coverage >= min_baseline)]

    return valid_anchors[-1] if len(valid_anchors) else None


def build_window(history: pd.DataFrame, station: str, meta: dict[str, Any],
                 weather_history: pd.DataFrame | None = None,
                 anchor: pd.Timestamp | None = None) -> tuple[np.ndarray, float, float] | None:
    """
    Constructs the (168, 22) input tensor for the past week:
    - 7 standardized log-space pollutants
    - 8 standardized weather features
    - 6 cyclical temporal features
    - 1 observed/interpolated boolean indicator
    """
    station_scalers = meta["scalers"].get(station)
    if station_scalers is None:
        return None

    clean_history = history.sort_values("datetime").drop_duplicates("datetime")
    anchor_timestamp = anchor if anchor is not None else clean_history["datetime"].max()
    hourly_index = pd.date_range(end=anchor_timestamp, periods=WINDOW_SIZE, freq="h")
    window_frame = clean_history.set_index("datetime").reindex(hourly_index)

    if weather_history is not None and len(weather_history):
        weather_df = _wind_components(weather_history).set_index("datetime")
        for col in WEATHER_FEATURES:
            if col in weather_df.columns:
                window_frame[col] = weather_df[col].reindex(hourly_index)

    pm_cols = [col for col in ("pm2_5", "pm10") if col in window_frame.columns]
    observed_series = window_frame[pm_cols].notna().any(axis=1) if pm_cols else pd.Series(False, index=hourly_index)
    if observed_series.mean() < MIN_WINDOW_COVERAGE:
        return None

    pollutant_columns = []
    out_of_range_flags = []

    for pollutant in POLLUTANTS:
        scaler = station_scalers.get(pollutant)
        if scaler is None or pollutant not in window_frame.columns:
            pollutant_columns.append(np.zeros(WINDOW_SIZE, dtype=np.float32))
            continue

        series = (window_frame[pollutant]
                  .interpolate(limit=3, limit_area="inside")
                  .ffill().bfill())

        raw_array = series.to_numpy(dtype=float)
        typical_val = np.expm1(scaler["mean"])
        raw_array = np.clip(np.nan_to_num(raw_array, nan=typical_val), 0, None)

        z_scores = (np.log1p(raw_array) - scaler["mean"]) / scaler["scale"]
        out_of_range_flags.append(np.abs(z_scores) > 3.5)
        pollutant_columns.append(np.clip(z_scores, -6, 6).astype(np.float32))

    weather_scalers = (meta.get("weather_scalers") or {}).get(station)
    feature_matrix = np.hstack([
        np.column_stack(pollutant_columns),
        _scale_weather(window_frame, weather_scalers),
        _time_features(pd.DatetimeIndex(hourly_index)),
        observed_series.to_numpy(dtype=np.float32).reshape(-1, 1),
    ]).astype(np.float32)

    beyond_training_share = float(np.mean(out_of_range_flags)) if out_of_range_flags else 0.0
    return feature_matrix, float(observed_series.mean()), beyond_training_share


def predict(history: pd.DataFrame, station: str, meta: dict[str, Any] | None = None,
            smearing: float = 1.0, weather_history: pd.DataFrame | None = None,
            weather_forecast: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """
    Runs model inference to forecast PM2.5 for the next 24 hours.
    Adds predicted residual departures onto the 24h persistence baseline.
    """
    model_metadata = meta or load_meta()
    anchor = choose_anchor(history)
    if anchor is None:
        return None

    window_data = build_window(history, station, model_metadata, weather_history, anchor)
    if window_data is None:
        return None
    features, fraction_observed, beyond_range = window_data

    station_idx = model_metadata["stations"].get(station)
    if station_idx is None:
        return None

    forecast_hours = pd.date_range(anchor + pd.Timedelta(hours=1), periods=FORECAST_HOURS, freq="h")
    weather_stats = (model_metadata.get("weather_scalers") or {}).get(station)
    future_tensor = build_future(weather_forecast, forecast_hours, weather_stats)

    # Model predicts residual deviation from persistence baseline
    model = load_model()
    scaled_residual = model.predict(
        {
            "sequence": features[None, ...],
            "future": future_tensor[None, ...],
            "station": np.array([[station_idx]], dtype=np.int32),
        },
        verbose=0
    )[0]

    # Persistence baseline (mean of last 24 observed hours in input tensor)
    target_idx = model_metadata["target_index"]
    baseline = float(features[-24:, target_idx].mean())

    target_stats = model_metadata["scalers"][station][model_metadata["target"]]
    predicted_values = np.expm1((scaled_residual + baseline) * target_stats["scale"] + target_stats["mean"]).clip(0) * smearing

    return {
        "hours": forecast_hours,
        "pm2_5": predicted_values,
        "frac_observed": fraction_observed,
        "frac_beyond_training": beyond_range,
        "station": station,
        "has_weather_forecast": weather_forecast is not None and len(weather_forecast) > 0,
        "anchor": anchor,
    }


def forecast_aqi(history: pd.DataFrame, forecast_pm25: np.ndarray, forecast_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Converts predicted PM2.5 hourly concentrations into discrete AQI numbers and health categories.
    Computes rolling 24-hour mean blending observed history with incoming forecast hours.
    """
    rolling_window = AVERAGING_HOURS.get("pm2_5", 24)
    recent_observed = (history.sort_values("datetime")["pm2_5"]
                       .dropna().tail(rolling_window - 1).to_numpy(dtype=float))

    rows = []
    for i, (hour, value) in enumerate(zip(forecast_hours, forecast_pm25)):
        blended_slice = np.concatenate([recent_observed, forecast_pm25[:i + 1]])[-rolling_window:]
        val = float(value)
        aqi_val = round(val)
        category = get_category(aqi_val)

        rows.append({
            "datetime": hour,
            "pm2_5": val,
            "pm2_5_24h": float(blended_slice.mean()) if len(blended_slice) else val,
            "aqi": aqi_val,
            "category": category,
        })

    return pd.DataFrame(rows)


def persistence_baseline(history: pd.DataFrame, hours: int = FORECAST_HOURS) -> np.ndarray | None:
    """Generates the naive persistence baseline: assuming the next 24 hours equal the last 24 observed hours."""
    recent_readings = (history.sort_values("datetime")["pm2_5"]
                       .dropna().tail(24).to_numpy(dtype=float))
    if len(recent_readings) == 0:
        return None
    return np.full(hours, float(recent_readings.mean()))
