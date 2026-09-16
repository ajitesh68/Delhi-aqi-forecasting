"""CPCB National Air Quality Index."""

import numpy as np
import pandas as pd

from src.config import AQI_CATEGORIES, POLLUTANT_LABELS

BREAKPOINTS = {
    "pm2_5": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
              (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    "pm10":  [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
              (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
    "no2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
              (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)],
    "so2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
              (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2400, 401, 500)],
    "o3":    [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
              (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
    "co":    [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
              (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
    "nh3":   [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
              (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)],
}

AVERAGING_HOURS = {
    "pm2_5": 24, "pm10": 24, "no2": 24, "so2": 24, "nh3": 24,
    "co": 8, "o3": 8,
}

PM_POLLUTANTS = ("pm2_5", "pm10")
MIN_SUBINDICES = 3


def sub_index(pollutant, concentration):
    """CPCB sub-index for one pollutant at an already-averaged concentration."""
    if pollutant not in BREAKPOINTS or concentration is None:
        return None
    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(c) or c < 0:
        return None

    bands = BREAKPOINTS[pollutant]
    for lo, hi, ilo, ihi in bands:
        if c <= hi:
            lo = max(lo, 0.0)
            if hi == lo:
                return int(round(ihi))
            return int(round((ihi - ilo) / (hi - lo) * (c - lo) + ilo))
    return 500


def rolling_averages(df, now=None):
    """Collapse an hourly frame into one CPCB-averaged value per pollutant."""
    if df is None or len(df) == 0:
        return {}
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    end = pd.Timestamp(now) if now is not None else d["datetime"].max()

    out = {}
    for pollutant, hours in AVERAGING_HOURS.items():
        if pollutant not in d.columns:
            continue
        window = d[(d["datetime"] > end - pd.Timedelta(hours=hours)) &
                   (d["datetime"] <= end)]
        vals = pd.to_numeric(window[pollutant], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        out[pollutant] = (float(vals.mean()), int(len(vals)))
    return out


def calculate_aqi(averages, min_coverage=0.5, require_three=True):
    """AQI from CPCB rolling averages."""
    subs, coverage = {}, {}
    for pollutant, entry in (averages or {}).items():
        if isinstance(entry, (tuple, list)):
            value, n_hours = entry[0], entry[1]
        else:
            value, n_hours = entry, None
        if n_hours is not None:
            need = AVERAGING_HOURS.get(pollutant, 24) * min_coverage
            if n_hours < need:
                continue
            coverage[pollutant] = n_hours
        s = sub_index(pollutant, value)
        if s is not None:
            subs[pollutant] = s

    if not subs:
        return _invalid("no usable pollutant readings")
    if require_three and len(subs) < MIN_SUBINDICES:
        return _invalid(
            f"only {len(subs)} sub-index(es); CPCB needs {MIN_SUBINDICES}", subs)
    if not any(p in subs for p in PM_POLLUTANTS):
        return _invalid("no PM2.5 or PM10 reading", subs)

    dominant = max(subs, key=subs.get)
    aqi = subs[dominant]
    return {
        "aqi": int(aqi),
        "category": get_category(aqi),
        "dominant": dominant,
        "dominant_label": POLLUTANT_LABELS.get(dominant, dominant),
        "sub_indices": subs,
        "coverage": coverage,
        "valid": True,
        "reason": "",
    }


def _invalid(reason, subs=None):
    return {"aqi": None, "category": "Unknown", "dominant": None,
            "dominant_label": None, "sub_indices": subs or {}, "coverage": {},
            "valid": False, "reason": reason}


def aqi_from_hourly(df, now=None, min_coverage=0.5):
    """Convenience: hourly frame -> AQI result."""
    return calculate_aqi(rolling_averages(df, now=now), min_coverage=min_coverage)


def get_category(aqi):
    if aqi is None or not np.isfinite(aqi):
        return "Unknown"
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Severe"


def health_advisory(category):
    return {
        "Good": "Air is clean. Any outdoor activity is fine.",
        "Satisfactory": "Minor discomfort possible for unusually sensitive people.",
        "Moderate": "Discomfort for people with asthma, lung or heart disease.",
        "Poor": "Discomfort for most on prolonged exposure. Avoid outdoor exertion.",
        "Very Poor": "Respiratory illness on prolonged exposure. Stay indoors.",
        "Severe": "Serious health impact for everyone. Avoid all outdoor activity.",
        "Unknown": "Not enough data to rate air quality right now.",
    }.get(category, "")


def sub_index_series(pollutant, values):
    """Vectorised sub-index over an array of already-averaged concentrations."""
    bands = BREAKPOINTS.get(pollutant)
    if bands is None:
        return pd.Series(np.nan, index=getattr(values, "index", None))
    xs, ys = [], []
    for lo, hi, ilo, ihi in bands:
        xs.extend([lo, hi])
        ys.extend([ilo, ihi])
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    out = pd.Series(np.interp(v.to_numpy(dtype=float), xs, ys), index=v.index)
    out[v.isna() | (v < 0)] = np.nan
    out[v > xs[-1]] = 500.0
    return out


def add_rolling_aqi(df, station_col="station", min_coverage=0.5):
    """Attach CPCB rolling averages, per-pollutant sub-indices and AQI."""
    if df is None or len(df) == 0:
        return df
    out = []
    for station, g in df.groupby(station_col, sort=False):
        g = g.sort_values("datetime").copy()
        g = g.set_index("datetime")
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        full[station_col] = station

        subs = {}
        for pollutant, hours in AVERAGING_HOURS.items():
            if pollutant not in full.columns:
                continue
            roll = full[pollutant].rolling(hours, min_periods=max(1, int(hours * min_coverage)))
            avg = roll.mean()
            full[f"{pollutant}_avg"] = avg
            subs[pollutant] = sub_index_series(pollutant, avg)

        raw = {}
        for pollutant in AVERAGING_HOURS:
            if pollutant in full.columns:
                raw[pollutant] = sub_index_series(pollutant, full[pollutant])
        if raw:
            raw_df = pd.DataFrame(raw)
            has_pm_raw = raw_df[[c for c in ("pm2_5", "pm10")
                                 if c in raw_df]].notna().any(axis=1)
            full["aqi_instant"] = raw_df.max(axis=1).where(has_pm_raw)
        else:
            full["aqi_instant"] = np.nan

        if subs:
            sub_df = pd.DataFrame(subs)
            has_pm = sub_df[[c for c in ("pm2_5", "pm10") if c in sub_df]].notna().any(axis=1)
            enough = sub_df.notna().sum(axis=1) >= MIN_SUBINDICES
            aqi = sub_df.max(axis=1).where(has_pm & enough)
            full["aqi"] = aqi
            usable = sub_df.notna().any(axis=1)
            dominant = pd.Series(pd.NA, index=sub_df.index, dtype="object")
            if usable.any():
                dominant.loc[usable] = sub_df.loc[usable].idxmax(axis=1)
            full["dominant"] = dominant.where(aqi.notna())
            for pollutant in sub_df.columns:
                full[f"{pollutant}_sub"] = sub_df[pollutant]
        else:
            full["aqi"] = np.nan
            full["dominant"] = None

        out.append(full.rename_axis("datetime").reset_index())
    return pd.concat(out, ignore_index=True)
