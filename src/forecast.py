"""Serving side of the 24-hour forecast."""

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
META_PATH = MODEL_DIR / "meta.json"
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
    """The forecast branch: weather over the hours being predicted."""
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


MIN_WINDOW_COVERAGE = 0.25
MIN_BASELINE_COVERAGE = 0.15


def choose_anchor(history, min_coverage=MIN_WINDOW_COVERAGE,
                  min_baseline=MIN_BASELINE_COVERAGE):
    """Latest hour with a usable window of history behind it."""
    d = history.sort_values("datetime").drop_duplicates("datetime")
    if len(d) == 0:
        return None

    pm = [p for p in ("pm2_5", "pm10") if p in d.columns]
    if not pm:
        return None
    observed = d.set_index("datetime")[pm].notna().any(axis=1)
    full = observed.reindex(
        pd.date_range(observed.index.min(), observed.index.max(), freq="h"),
        fill_value=False)
    if len(full) < WINDOW_SIZE:
        return None

    window = full.rolling(WINDOW_SIZE).mean()
    baseline = full.rolling(FORECAST_HOURS).mean()
    usable = full.index[(window >= min_coverage) & (baseline >= min_baseline)]
    return usable[-1] if len(usable) else None


def build_window(history, station, meta, weather_history=None, anchor=None):
    """WINDOW_SIZE hours ending at `anchor`, plus a completeness score."""
    scalers = meta["scalers"].get(station)
    if scalers is None:
        return None

    d = history.sort_values("datetime").drop_duplicates("datetime")
    anchor = anchor if anchor is not None else d["datetime"].max()
    index = pd.date_range(end=anchor, periods=WINDOW_SIZE, freq="h")
    frame = d.set_index("datetime").reindex(index)

    if weather_history is not None and len(weather_history):
        w = _wind_components(weather_history).set_index("datetime")
        for column in WEATHER_FEATURES:
            if column in w.columns:
                frame[column] = w[column].reindex(index)

    pm = [p for p in ("pm2_5", "pm10") if p in frame.columns]
    observed = frame[pm].notna().any(axis=1) if pm else pd.Series(False, index=index)
    if observed.mean() < MIN_WINDOW_COVERAGE:
        return None

    columns, out_of_range = [], []
    for pollutant in POLLUTANTS:
        stats = scalers.get(pollutant)
        if stats is None or pollutant not in frame.columns:
            columns.append(np.zeros(WINDOW_SIZE, dtype=np.float32))
            continue
        series = (frame[pollutant].interpolate(limit=3, limit_area="inside")
                                  .ffill().bfill())
        raw = series.to_numpy(dtype=float)
        typical = np.expm1(stats["mean"])
        raw = np.clip(np.nan_to_num(raw, nan=typical), 0, None)
        z = (np.log1p(raw) - stats["mean"]) / stats["scale"]
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
    """24 hourly PM2.5 values in ug/m3, starting one hour after `history`."""
    meta = meta or load_meta()
    anchor = choose_anchor(history)
    if anchor is None:
        return None
    built = build_window(history, station, meta, weather_history, anchor)
    if built is None:
        return None
    features, frac_observed, beyond = built

    station_idx = meta["stations"].get(station)
    if station_idx is None:
        return None

    hours = pd.date_range(anchor + pd.Timedelta(hours=1),
                          periods=FORECAST_HOURS, freq="h")
    weather_stats = (meta.get("weather_scalers") or {}).get(station)
    future = build_future(weather_forecast, hours, weather_stats)

    scaled = load_model().predict(
        {"sequence": features[None, ...],
         "future": future[None, ...],
         "station": np.array([[station_idx]], dtype=np.int32)},
        verbose=0)[0]

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
        "anchor": anchor,
    }


def forecast_aqi(history, forecast_pm25, forecast_hours):
    """AQI for each forecast hour, on proper CPCB rolling windows."""
    window = AVERAGING_HOURS.get("pm2_5", 24)
    recent = (history.sort_values("datetime")["pm2_5"]
                     .dropna().tail(window - 1).to_numpy(dtype=float))

    rows = []
    for i, (hour, value) in enumerate(zip(forecast_hours, forecast_pm25)):
        mixed = np.concatenate([recent, forecast_pm25[:i + 1]])[-window:]
        result = calculate_aqi({"pm2_5": (float(mixed.mean()), len(mixed))},
                               require_three=False)
        rows.append({"datetime": hour, "pm2_5": float(value),
                     "pm2_5_24h": float(mixed.mean()),
                     "aqi": result["aqi"], "category": result["category"]})
    return pd.DataFrame(rows)


def persistence_baseline(history, hours=FORECAST_HOURS):
    """What tomorrow looks like if it looks like today."""
    recent = (history.sort_values("datetime")["pm2_5"]
                     .dropna().tail(24).to_numpy(dtype=float))
    if len(recent) == 0:
        return None
    return np.full(hours, float(recent.mean()))


SCENARIOS = {
    "As forecast": {},
    "Wind doubles": {"wind_kph": ("x", 2.0)},
    "Wind drops by half": {"wind_kph": ("x", 0.5)},
    "Dead calm": {"wind_kph": ("=", 1.0)},
    "10°C colder": {"temp_c": ("+", -10.0)},
    "Boundary layer halves": {"blh_m": ("x", 0.5)},
    "Rain arrives": {"precip_mm": ("=", 2.0), "humidity": ("=", 90.0)},
}


def apply_scenario(weather_forecast, scenario):
    """Return a copy of the forecast weather with one scenario applied."""
    changes = SCENARIOS.get(scenario) or {}
    if not changes or weather_forecast is None or len(weather_forecast) == 0:
        return weather_forecast

    out = weather_forecast.copy()
    for column, (op, amount) in changes.items():
        if column not in out.columns:
            continue
        values = out[column].to_numpy(dtype=float)
        if op == "x":
            values = values * amount
        elif op == "+":
            values = values + amount
        else:
            values = np.full_like(values, amount)
        out[column] = np.clip(values, 0, None)

    if "wind_kph" in changes and "wind_dir" in out.columns:
        radians = np.deg2rad(out["wind_dir"].to_numpy(dtype=float))
        speed = out["wind_kph"].to_numpy(dtype=float)
        out["wind_u"] = speed * np.sin(radians)
        out["wind_v"] = speed * np.cos(radians)
    return out


def scenario_sweep(history, station, weather_history, weather_forecast,
                   meta=None, scenarios=None):
    """Run every scenario through the model and report the spread."""
    meta = meta or load_meta()
    rows = []
    for name in (scenarios or SCENARIOS):
        result = predict(history, station, meta=meta,
                         weather_history=weather_history,
                         weather_forecast=apply_scenario(weather_forecast, name))
        if result is None:
            continue
        rows.append({"scenario": name,
                     "mean_pm2_5": float(np.mean(result["pm2_5"])),
                     "peak_pm2_5": float(np.max(result["pm2_5"]))})
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["scenario"] == "As forecast", "mean_pm2_5"]
    if len(baseline):
        frame["change_pct"] = (frame["mean_pm2_5"] / float(baseline.iloc[0]) - 1) * 100
    return frame
