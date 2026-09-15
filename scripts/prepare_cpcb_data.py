"""Turn the CPCB store into training sequences.

Three decisions here carry most of the weight, and each is a fix for a
specific way the previous pipeline was wrong:

1. **Log-space scaling, not MinMax.** MinMax anchors on the maximum, the
   least stable statistic in this data. The old scaler was fitted where
   PM2.5 topped out at 370; real CPCB reaches 710, so every severe winter
   hour saturated at 1.0 and became indistinguishable from every other
   severe hour. log1p then standardise leaves a 710 near z=3.

2. **Scalers are fitted on the training period only**, so the winter
   holdout stays genuinely unseen.

3. **Sequences are cut from the continuous series, then assigned to a
   split by target date.** Dropping the holdout rows first would splice
   October onto February and manufacture windows that never happened.

Windows are rejected rather than patched when the data underneath them is
too thin -- a forecast trained on interpolation is a forecast of the
interpolator.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (FORECAST_HOURS, MAX_IMPUTED_FRAC, MAX_INTERNAL_GAP,
                        MAX_INTERPOLATE_GAP, POLLUTANTS, STORE_DIR, WINDOW_SIZE)
from src.sources import store

WEATHER_PATH = Path(STORE_DIR) / "weather_hourly.parquet"

# Weather is what lets the model disagree with persistence. Both see the
# same pollutant history; only the model gets told the wind is about to
# pick up. Direction is decomposed into components because 359 and 1
# degrees are neighbours.
WEATHER_COLS = ["temp_c", "humidity", "pressure_mb", "wind_kph",
                "blh_m", "precip_mm"]
WEATHER_FEATURES = WEATHER_COLS + ["wind_u", "wind_v"]

STRIDE = 3
TARGET = "pm2_5"

# Ceilings for obvious sensor faults, not for real extremes. Delhi has
# genuinely recorded PM10 near 1500, so these sit well above that.
PHYSICAL_MAX = {"pm2_5": 1200.0, "pm10": 2500.0, "no2": 800.0,
                "so2": 1000.0, "o3": 500.0, "co": 60.0, "nh3": 800.0}

# The store holds exactly one severe season: Oct 2025 through Jan 2026.
# Holding all of it out would leave a training set with no severe episode
# in it at all, and then scoring the model on the only severe episode
# there is -- a test the model cannot pass and which measures nothing
# useful. So January is the holdout and Oct-Dec stay in training: the
# model learns what a Delhi inversion does, and is scored on a stretch of
# it that it has never seen.
#
# Split rule is a purged split on the targets. A window is test if its 24
# target hours all fall in the holdout, and train if none of them do. A
# test window's input history may reach back into December, because at
# serving time you always have the previous seven days -- what must not
# overlap is what the model was fitted against. Training windows sitting
# after the holdout are dropped when their inputs reach back into it, so
# nothing is fitted on a January hour.
HOLDOUT_START = pd.Timestamp("2026-01-01")
HOLDOUT_END = pd.Timestamp("2026-02-01")


def _time_features(index):
    """Cyclical encodings so 23:00 and 00:00 are adjacent, not 23 apart."""
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ]).astype(np.float32)


FEATURE_NAMES = ([f"{p}_scaled" for p in POLLUTANTS]
                 + WEATHER_FEATURES
                 + ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "doy_sin", "doy_cos", "is_observed"])

# The forecast branch: what the weather will be doing over the target
# hours. At serving time this comes from Open-Meteo's forecast, which is
# information genuinely available ahead of the fact.
FUTURE_FEATURE_NAMES = WEATHER_FEATURES + ["hour_sin", "hour_cos"]


def load_weather():
    if not WEATHER_PATH.exists():
        return None
    w = pd.read_parquet(WEATHER_PATH)
    radians = np.deg2rad(w["wind_dir"].to_numpy(dtype=float))
    speed = w["wind_kph"].to_numpy(dtype=float)
    w["wind_u"] = speed * np.sin(radians)
    w["wind_v"] = speed * np.cos(radians)
    return w


def fit_weather_scalers(frame, train_mask):
    stats = {}
    for column in WEATHER_FEATURES:
        if column not in frame.columns:
            continue
        values = frame.loc[train_mask, column].dropna()
        if len(values) < 200:
            continue
        std = float(values.std())
        stats[column] = {"mean": float(values.mean()),
                         "scale": std if std > 1e-6 else 1.0}
    return stats


def weather_block(frame, stats):
    columns = []
    for column in WEATHER_FEATURES:
        entry = stats.get(column)
        if entry is None or column not in frame.columns:
            columns.append(np.zeros(len(frame), dtype=np.float32))
            continue
        raw = frame[column].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=entry["mean"])
        columns.append(((raw - entry["mean"]) / entry["scale"]).astype(np.float32))
    return np.column_stack(columns)


def build_station_frame(df, weather=None):
    """Continuous hourly frame with an observation mask and short-gap fill."""
    df = df.sort_values("datetime").drop_duplicates("datetime")
    index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    frame = df.set_index("datetime").reindex(index)
    frame.index.name = "datetime"

    present = [p for p in POLLUTANTS if p in frame.columns]
    # "Observed" means the hour carried a PM reading, which is what the
    # target and the AQI both depend on.
    pm = [p for p in ("pm2_5", "pm10") if p in frame.columns]
    frame["is_observed"] = frame[pm].notna().any(axis=1) if pm else False

    for pollutant in present:
        ceiling = PHYSICAL_MAX.get(pollutant, 1000.0)
        # OpenAQ emits -999 sentinels; log1p of a negative is NaN.
        frame[pollutant] = frame[pollutant].where(
            frame[pollutant].between(0, ceiling))
        frame[pollutant] = frame[pollutant].interpolate(
            method="linear", limit=MAX_INTERPOLATE_GAP, limit_area="inside")

    frame["is_filled"] = ~frame["is_observed"] & frame[pm].notna().any(axis=1)
    frame = frame.reset_index()

    if weather is not None and len(weather):
        columns = ["datetime"] + [c for c in WEATHER_FEATURES if c in weather.columns]
        frame = frame.merge(weather[columns], on="datetime", how="left")
        for column in WEATHER_FEATURES:
            if column in frame.columns:
                # Reanalysis is gapless in practice; this covers the edges.
                frame[column] = frame[column].interpolate(limit=6).ffill().bfill()
    return frame


def fit_scalers(frame, train_mask):
    """Per-pollutant log-space mean and scale, from training rows only."""
    scalers = {}
    for pollutant in POLLUTANTS:
        if pollutant not in frame.columns:
            continue
        values = frame.loc[train_mask & frame["is_observed"], pollutant].dropna()
        if len(values) < 200:
            continue
        logged = np.log1p(values.to_numpy())
        std = float(logged.std())
        scalers[pollutant] = {"mean": float(logged.mean()),
                              "scale": std if std > 1e-6 else 1.0,
                              "n_fit": int(len(values))}
    return scalers


def apply_scalers(frame, scalers, weather_stats=None):
    """Scaled matrix plus the per-hour feature block."""
    columns = []
    for pollutant in POLLUTANTS:
        stats = scalers.get(pollutant)
        if stats is None or pollutant not in frame.columns:
            columns.append(np.zeros(len(frame), dtype=np.float32))
            continue
        raw = frame[pollutant].to_numpy(dtype=float)
        logged = np.log1p(np.clip(np.nan_to_num(raw, nan=0.0), 0, None))
        columns.append(((logged - stats["mean"]) / stats["scale"]).astype(np.float32))

    stack = np.column_stack(columns)
    times = _time_features(pd.DatetimeIndex(frame["datetime"]))
    observed = frame["is_observed"].to_numpy(dtype=np.float32).reshape(-1, 1)
    blocks = [stack]
    if weather_stats:
        blocks.append(weather_block(frame, weather_stats))
    blocks += [times, observed]
    return np.hstack(blocks).astype(np.float32)


def future_block(frame, weather_stats):
    """Weather over the target hours, plus hour-of-day so the model can
    place a wind shift at 3am against one at 3pm."""
    times = _time_features(pd.DatetimeIndex(frame["datetime"]))[:, :2]
    if not weather_stats:
        return times.astype(np.float32)
    return np.hstack([weather_block(frame, weather_stats), times]).astype(np.float32)


def cut_windows(features, future, frame, target_col_idx):
    """Yield (input window, 24h target, target start) for usable windows."""
    observed = frame["is_observed"].to_numpy()
    filled = frame["is_filled"].to_numpy()
    times = pd.DatetimeIndex(frame["datetime"])
    usable = observed | filled
    n = len(frame) - WINDOW_SIZE - FORECAST_HOURS + 1

    for start in range(0, max(0, n), STRIDE):
        w0, w1 = start, start + WINDOW_SIZE
        t0, t1 = w1, w1 + FORECAST_HOURS

        # Every forecast hour must be a real measurement. A model scored
        # against interpolation is scoring the interpolator.
        if not observed[t0:t1].all():
            continue
        if not usable[w0:w1].all():
            # A hole the short-gap fill could not close.
            continue
        if filled[w0:w1].mean() > MAX_IMPUTED_FRAC:
            continue
        if _longest_run(filled[w0:w1]) > MAX_INTERNAL_GAP:
            continue

        yield (features[w0:w1],
               future[t0:t1],
               features[t0:t1, target_col_idx],
               times[w0], times[t0], times[t1 - 1],
               float(observed[w0:w1].mean()))


def _longest_run(mask):
    longest = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--min-windows", type=int, default=200,
                    help="skip a station that yields fewer usable windows")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = store.load()
    if len(raw) == 0:
        print("Store is empty -- run scripts/download_cpcb_history.py first.")
        return 1

    target_idx = POLLUTANTS.index(TARGET)
    stations = sorted(raw["station"].dropna().unique())
    print(f"{len(raw):,} stored rows across {len(stations)} stations\n")

    X_parts, F_parts, y_parts, s_parts = [], [], [], []
    w_parts, t_parts, e_parts, f_parts = [], [], [], []
    scaler_book, weather_book, station_index, report = {}, {}, {}, []

    weather = load_weather()
    if weather is None:
        print("No weather store -- run scripts/download_weather.py. Without it "
              "the model sees only what persistence sees.\n")

    for position, station in enumerate(stations):
        station_weather = None
        if weather is not None:
            station_weather = weather[weather["station"] == station]
        frame = build_station_frame(raw[raw["station"] == station], station_weather)
        in_holdout = frame["datetime"].between(HOLDOUT_START, HOLDOUT_END,
                                               inclusive="left")
        scalers = fit_scalers(frame, ~in_holdout)
        if TARGET not in scalers:
            print(f"  {station.split(',')[0]:<28} skipped -- too little {TARGET}")
            continue

        weather_stats = fit_weather_scalers(frame, ~in_holdout)
        features = apply_scalers(frame, scalers, weather_stats)
        future = future_block(frame, weather_stats)
        windows = list(cut_windows(features, future, frame, target_idx))
        if len(windows) < args.min_windows:
            print(f"  {station.split(',')[0]:<28} skipped -- {len(windows)} windows")
            continue

        X_parts.append(np.stack([w[0] for w in windows]))
        F_parts.append(np.stack([w[1] for w in windows]))
        y_parts.append(np.stack([w[2] for w in windows]))
        w_parts.append(pd.DatetimeIndex([w[3] for w in windows]))
        t_parts.append(pd.DatetimeIndex([w[4] for w in windows]))
        e_parts.append(pd.DatetimeIndex([w[5] for w in windows]))
        f_parts.append(np.array([w[6] for w in windows], dtype=np.float32))
        s_parts.append(np.full(len(windows), len(station_index), dtype=np.int32))

        station_index[station] = len(station_index)
        scaler_book[station] = scalers
        weather_book[station] = weather_stats

        observed = int(frame["is_observed"].sum())
        report.append({
            "station": station,
            "hours": int(len(frame)),
            "observed": observed,
            "coverage": round(observed / len(frame), 3),
            "filled": int(frame["is_filled"].sum()),
            "windows": len(windows),
        })
        print(f"  {station.split(',')[0]:<28} {observed:>6,} observed h  "
              f"{observed / len(frame):>5.0%} coverage  {len(windows):>5,} windows")

    if not X_parts:
        print("\nNo station produced enough usable windows.")
        return 1

    X = np.concatenate(X_parts)
    F = np.concatenate(F_parts)
    y = np.concatenate(y_parts)
    station_ids = np.concatenate(s_parts)
    win_starts = pd.DatetimeIndex(np.concatenate([w.to_numpy() for w in w_parts]))
    starts = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in t_parts]))
    ends = pd.DatetimeIndex(np.concatenate([e.to_numpy() for e in e_parts]))
    frac_observed = np.concatenate(f_parts)

    tgt_start, tgt_end = starts.to_numpy(), ends.to_numpy()
    win_start = win_starts.to_numpy()
    hstart, hend = HOLDOUT_START.to_numpy(), HOLDOUT_END.to_numpy()

    target_in = (tgt_start >= hstart) & (tgt_end < hend)
    target_out = (tgt_end < hstart) | (tgt_start >= hend)
    inputs_clear = win_start >= hend          # only matters after the holdout

    is_holdout = target_in
    is_train = target_out & ((tgt_end < hstart) | inputs_clear)
    dropped = int(len(X) - is_holdout.sum() - is_train.sum())

    np.savez_compressed(
        out / "sequences.npz",
        X_train=X[is_train], F_train=F[is_train], y_train=y[is_train],
        station_train=station_ids[is_train],
        frac_train=frac_observed[is_train],
        start_train=starts[is_train].astype("datetime64[s]").to_numpy(),
        X_test=X[is_holdout], F_test=F[is_holdout], y_test=y[is_holdout],
        station_test=station_ids[is_holdout],
        frac_test=frac_observed[is_holdout],
        start_test=starts[is_holdout].astype("datetime64[s]").to_numpy(),
    )

    meta = {
        "window_size": WINDOW_SIZE,
        "forecast_hours": FORECAST_HOURS,
        "stride": STRIDE,
        "target": TARGET,
        "target_index": target_idx,
        "features": FEATURE_NAMES,
        "future_features": FUTURE_FEATURE_NAMES,
        "weather_scalers": weather_book,
        "has_weather": weather is not None,
        "pollutants": POLLUTANTS,
        "stations": station_index,
        "scalers": scaler_book,
        "holdout": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END)},
        "counts": {"train": int(is_train.sum()),
                   "test": int(is_holdout.sum()),
                   "embargoed": dropped},
        "per_station": report,
    }
    # JSON, not pickle: sklearn pickles have already broken twice in this
    # repo's history across version bumps.
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n{len(X):,} windows total -- "
          f"{(~is_holdout).sum():,} train / {is_holdout.sum():,} winter holdout")
    if is_holdout.sum() == 0:
        print("WARNING: holdout is empty. The store does not yet cover "
              f"{HOLDOUT_START:%b %Y}-{HOLDOUT_END:%b %Y}.")
    print(f"features: {X.shape[2]} past + {F.shape[2]} forecast   "
          f"stations: {len(station_index)}")
    print(f"wrote {out / 'sequences.npz'} and {out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
