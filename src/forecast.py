"""Serving side of the 24-hour forecast.

The model predicts PM2.5 in per-station scaled log space. Turning that
back into something the dashboard can show takes three steps, and the
middle one is easy to miss:

1. inverse the per-station scaler
2. correct for the log-space bias
3. fold the forecast into the trailing 23 observed hours before computing
   an AQI, because CPCB's index is a rolling average and a forecast hour
   on its own is not one

On (2): Huber loss in log space optimises something close to the
conditional median, so expm1 of the mean prediction sits below the mean
concentration. Uncorrected, the model reads low exactly during the
episodes a warning exists for. The smearing factor estimated on training
residuals puts that back.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.aqi import AVERAGING_HOURS, calculate_aqi
from src.config import FORECAST_HOURS, MODELS_DIR, POLLUTANTS, WINDOW_SIZE

WEATHER_FEATURES = ["temp_c", "humidity", "pressure_mb", "wind_kph",
                    "blh_m", "precip_mm", "wind_u", "wind_v"]

MODEL_DIR = Path(MODELS_DIR) / "cpcb"
MODEL_PATH = MODEL_DIR / "pm25_24h.keras"
META_PATH = Path("data/train/meta.json")
SCORECARD_PATH = MODEL_DIR / "scorecard.json"


def available():
    """Whether a trained model and its metadata are both on disk."""
    return MODEL_PATH.exists() and META_PATH.exists()


def load_meta():
    with open(META_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_scorecard():
    if not SCORECARD_PATH.exists():
        return None
    with open(SCORECARD_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_model = None


def load_model():
    global _model
    if _model is None:
        from tensorflow import keras
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def _time_features(index):
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ]).astype(np.float32)


def _wind_components(frame):
    if "wind_dir" not in frame.columns or "wind_kph" not in frame.columns:
        return frame
    radians = np.deg2rad(frame["wind_dir"].to_numpy(dtype=float))
    speed = frame["wind_kph"].to_numpy(dtype=float)
    frame = frame.copy()
    frame["wind_u"] = speed * np.sin(radians)
    frame["wind_v"] = speed * np.cos(radians)
    return frame


def _scale_weather(frame, stats):
    columns = []
    for column in WEATHER_FEATURES:
        entry = (stats or {}).get(column)
        if entry is None or column not in frame.columns:
            columns.append(np.zeros(len(frame), dtype=np.float32))
            continue
        raw = np.nan_to_num(frame[column].to_numpy(dtype=float),
                            nan=entry["mean"])
        columns.append(((raw - entry["mean"]) / entry["scale"]).astype(np.float32))
    return np.column_stack(columns)


def build_future(forecast_weather, hours, weather_stats):
    """The forecast branch: weather over the hours being predicted.

    This is the input persistence cannot have. Without it the model has
    nothing to say that the last 24 hours do not already say.
    """
    frame = pd.DataFrame({"datetime": hours})
    if forecast_weather is not None and len(forecast_weather):
        frame = frame.merge(_wind_components(forecast_weather),
                            on="datetime", how="left")
    scaled = _scale_weather(frame, weather_stats)
    index = pd.DatetimeIndex(hours)
    hour = index.hour.to_numpy()
    times = np.column_stack([np.sin(2 * np.pi * hour / 24),
                             np.cos(2 * np.pi * hour / 24)]).astype(np.float32)
    return np.hstack([scaled, times]).astype(np.float32)


def build_window(history, station, meta, weather_history=None):
    """Last WINDOW_SIZE hours as a model input, plus a completeness score.

    Returns (features, frac_observed, out_of_range_frac) or None when
    there is not enough recent history to fill the window honestly.
    """
    scalers = meta["scalers"].get(station)
    if scalers is None:
        return None

    d = history.sort_values("datetime").drop_duplicates("datetime")
    index = pd.date_range(end=d["datetime"].max(), periods=WINDOW_SIZE, freq="h")
    frame = d.set_index("datetime").reindex(index)

    if weather_history is not None and len(weather_history):
        w = _wind_components(weather_history).set_index("datetime")
        for column in WEATHER_FEATURES:
            if column in w.columns:
                frame[column] = w[column].reindex(index)

    pm = [p for p in ("pm2_5", "pm10") if p in frame.columns]
    observed = frame[pm].notna().any(axis=1) if pm else pd.Series(False, index=index)
    if observed.mean() < 0.5:
        return None

    columns, out_of_range = [], []
    for pollutant in POLLUTANTS:
        stats = scalers.get(pollutant)
        if stats is None or pollutant not in frame.columns:
            columns.append(np.zeros(WINDOW_SIZE, dtype=np.float32))
            continue
        series = frame[pollutant].interpolate(limit=3, limit_area="inside")
        raw = np.clip(np.nan_to_num(series.to_numpy(dtype=float), nan=0.0), 0, None)
        z = (np.log1p(raw) - stats["mean"]) / stats["scale"]
        # Unbounded on purpose: a 700 ug/m3 hour should land near z=3, not
        # be clipped into the ordinary range. +/-6 sigma only catches
        # corrupt data.
        out_of_range.append(np.abs(z) > 3.5)
        columns.append(np.clip(z, -6, 6).astype(np.float32))

    weather_stats = (meta.get("weather_scalers") or {}).get(station)
    features = np.hstack([
        np.column_stack(columns),
        _scale_weather(frame, weather_stats),
        _time_features(pd.DatetimeIndex(index)),
        observed.to_numpy(dtype=np.float32).reshape(-1, 1),
    ]).astype(np.float32)

    beyond = float(np.mean(out_of_range)) if out_of_range else 0.0
    return features, float(observed.mean()), beyond


def predict(history, station, meta=None, smearing=1.0,
            weather_history=None, weather_forecast=None):
    """24 hourly PM2.5 values in ug/m3, starting one hour after `history`.

    Returns None when the window cannot be filled from observed data.
    """
    meta = meta or load_meta()
    built = build_window(history, station, meta, weather_history)
    if built is None:
        return None
    features, frac_observed, beyond = built

    station_idx = meta["stations"].get(station)
    if station_idx is None:
        return None

    last = pd.Timestamp(history["datetime"].max())
    hours = pd.date_range(last + pd.Timedelta(hours=1), periods=FORECAST_HOURS,
                          freq="h")
    weather_stats = (meta.get("weather_scalers") or {}).get(station)
    future = build_future(weather_forecast, hours, weather_stats)

    scaled = load_model().predict(
        {"sequence": features[None, ...],
         "future": future[None, ...],
         "station": np.array([[station_idx]], dtype=np.int32)},
        verbose=0)[0]

    # The model predicts a departure from persistence, so the baseline it
    # was trained against has to be added back before inverting.
    target_idx = meta["target_index"]
    baseline = float(features[-24:, target_idx].mean())

    stats = meta["scalers"][station][meta["target"]]
    values = np.expm1((scaled + baseline) * stats["scale"]
                      + stats["mean"]).clip(0) * smearing

    return {
        "hours": hours,
        "pm2_5": values,
        "frac_observed": frac_observed,
        "frac_beyond_training": beyond,
        "station": station,
        "has_weather_forecast": weather_forecast is not None
                                and len(weather_forecast) > 0,
    }


def forecast_aqi(history, forecast_pm25, forecast_hours):
    """AQI for each forecast hour, on proper CPCB rolling windows.

    Each hour's index is computed from a 24-hour window made of the real
    observed hours before it plus the forecast hours up to it. Applying
    breakpoints to a single predicted hour would produce a number that is
    not an AQI.
    """
    window = AVERAGING_HOURS.get("pm2_5", 24)
    recent = (history.sort_values("datetime")["pm2_5"]
                     .dropna().tail(window - 1).to_numpy(dtype=float))

    rows = []
    for i, (hour, value) in enumerate(zip(forecast_hours, forecast_pm25)):
        mixed = np.concatenate([recent, forecast_pm25[:i + 1]])[-window:]
        # Other pollutants are carried at their recent observed level:
        # the model forecasts PM2.5, and PM2.5 drives the index here.
        result = calculate_aqi({"pm2_5": (float(mixed.mean()), len(mixed))},
                               require_three=False)
        rows.append({"datetime": hour, "pm2_5": float(value),
                     "pm2_5_24h": float(mixed.mean()),
                     "aqi": result["aqi"], "category": result["category"]})
    return pd.DataFrame(rows)


def persistence_baseline(history, hours=FORECAST_HOURS):
    """What tomorrow looks like if it looks like today.

    Shown next to the model so the comparison the scorecard makes is
    visible in the UI too, not just in a JSON file.
    """
    recent = (history.sort_values("datetime")["pm2_5"]
                     .dropna().tail(24).to_numpy(dtype=float))
    if len(recent) == 0:
        return None
    return np.full(hours, float(recent.mean()))
