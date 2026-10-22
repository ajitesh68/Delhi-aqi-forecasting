import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import AQI_CATEGORIES, POLLUTANT_LABELS

# Official CPCB Breakpoints: [(Conc_Lo, Conc_Hi, AQI_Lo, AQI_Hi), ...]
BREAKPOINTS = {
    "pm2_5": [
        (0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
        (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)
    ],
    "pm10": [
        (0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
        (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)
    ],
    "no2": [
        (0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
        (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)
    ],
    "so2": [
        (0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
        (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2400, 401, 500)
    ],
    "o3": [
        (0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
        (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)
    ],
    "co": [
        (0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
        (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)
    ],
    "nh3": [
        (0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
        (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)
    ],
}

# CPCB standard rolling averaging duration in hours
AVERAGING_HOURS = {
    "pm2_5": 24, "pm10": 24, "no2": 24, "so2": 24, "nh3": 24,
    "co": 8, "o3": 8,
}

PM_POLLUTANTS = ("pm2_5", "pm10")
MIN_SUBINDICES = 3
MIN_COVERAGE = 16 / 24  # CPCB requires at least 16 out of 24 hours of data


def sub_index(pollutant: str, concentration: Any) -> int | None:
    """Calculates the CPCB sub-index for a single pollutant at a given concentration."""
    if pollutant not in BREAKPOINTS or concentration is None:
        return None
    try:
        conc = float(concentration)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(conc) or conc < 0:
        return None

    bands = BREAKPOINTS[pollutant]
    for conc_lo, conc_hi, aqi_lo, aqi_hi in bands:
        if conc <= conc_hi:
            conc_lo = max(conc_lo, 0.0)
            if conc_hi == conc_lo:
                return int(round(aqi_hi))
            fraction = (conc - conc_lo) / (conc_hi - conc_lo)
            return int(round(fraction * (aqi_hi - aqi_lo) + aqi_lo))

    return 500


def rolling_averages(df: pd.DataFrame, now: Any = None) -> dict[str, tuple[float, int]]:
    """Collapses an hourly frame into CPCB-compliant rolling window averages per pollutant."""
    if df is None or len(df) == 0:
        return {}

    clean_df = df.copy()
    clean_df["datetime"] = pd.to_datetime(clean_df["datetime"])
    end_time = pd.Timestamp(now) if now is not None else clean_df["datetime"].max()

    output = {}
    for pollutant, window_hours in AVERAGING_HOURS.items():
        if pollutant not in clean_df.columns:
            continue
        start_time = end_time - pd.Timedelta(hours=window_hours)
        window = clean_df[(clean_df["datetime"] > start_time) & (clean_df["datetime"] <= end_time)]
        valid_readings = pd.to_numeric(window[pollutant], errors="coerce").dropna()
        if len(valid_readings) == 0:
            continue
        output[pollutant] = (float(valid_readings.mean()), int(len(valid_readings)))

    return output


def calculate_aqi(averages: dict[str, Any], min_coverage: float = MIN_COVERAGE,
                  require_three: bool = True) -> dict[str, Any]:
    """
    Computes official CPCB AQI from pollutant averages:
    - Minimum 3 sub-indices required (including at least one PM pollutant).
    - Worst pollutant sub-index determines overall AQI.
    """
    sub_indices = {}
    coverage = {}

    for pollutant, entry in (averages or {}).items():
        if isinstance(entry, (tuple, list)):
            value, n_hours = entry[0], entry[1]
        else:
            value, n_hours = entry, None

        if n_hours is not None:
            needed_hours = math.ceil(AVERAGING_HOURS.get(pollutant, 24) * min_coverage)
            if n_hours < needed_hours:
                continue
            coverage[pollutant] = n_hours

        sub_idx = sub_index(pollutant, value)
        if sub_idx is not None:
            sub_indices[pollutant] = sub_idx

    if not sub_indices:
        return _invalid_result("no usable pollutant readings")

    if require_three and len(sub_indices) < MIN_SUBINDICES:
        return _invalid_result(f"only {len(sub_indices)} sub-index(es); CPCB needs {MIN_SUBINDICES}", sub_indices)

    if not any(pm in sub_indices for pm in PM_POLLUTANTS):
        return _invalid_result("no PM2.5 or PM10 reading", sub_indices)

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    final_aqi = sub_indices[dominant_pollutant]

    return {
        "aqi": int(final_aqi),
        "category": get_category(final_aqi),
        "dominant": dominant_pollutant,
        "dominant_label": POLLUTANT_LABELS.get(dominant_pollutant, dominant_pollutant),
        "sub_indices": sub_indices,
        "coverage": coverage,
        "valid": True,
        "reason": "",
    }


def _invalid_result(reason: str, subs: dict[str, int] | None = None) -> dict[str, Any]:
    """Helper returning a standard dictionary representation for invalid/insufficient AQI data."""
    return {
        "aqi": None,
        "category": "Unknown",
        "dominant": None,
        "dominant_label": None,
        "sub_indices": subs or {},
        "coverage": {},
        "valid": False,
        "reason": reason,
    }


def aqi_from_hourly(df: pd.DataFrame, now: Any = None, min_coverage: float = MIN_COVERAGE) -> dict[str, Any]:
    """Computes CPCB AQI from a raw hourly time-series frame."""
    return calculate_aqi(rolling_averages(df, now=now), min_coverage=min_coverage)


def get_category(aqi: float | None) -> str:
    """Returns the CPCB AQI category label (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)."""
    if aqi is None or not np.isfinite(aqi):
        return "Unknown"
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Severe"


def health_advisory(category: str) -> str:
    """Returns clear health impact advisory text for a given AQI category."""
    advisories = {
        "Good": "Air is clean. Any outdoor activity is fine.",
        "Satisfactory": "Minor discomfort possible for unusually sensitive people.",
        "Moderate": "Discomfort for people with asthma, lung or heart disease.",
        "Poor": "Discomfort for most on prolonged exposure. Avoid outdoor exertion.",
        "Very Poor": "Respiratory illness on prolonged exposure. Stay indoors.",
        "Severe": "Serious health impact for everyone. Avoid all outdoor activity.",
        "Unknown": "Not enough data to rate air quality right now.",
    }
    return advisories.get(category, "")


def sub_index_series(pollutant: str, values: pd.Series | np.ndarray) -> pd.Series:
    """Vectorized sub-index calculation using linear piecewise interpolation for pandas series."""
    bands = BREAKPOINTS.get(pollutant)
    if bands is None:
        return pd.Series(np.nan, index=getattr(values, "index", None))

    conc_points = []
    aqi_points = []
    for conc_lo, conc_hi, aqi_lo, aqi_hi in bands:
        conc_points.extend([conc_lo, conc_hi])
        aqi_points.extend([aqi_lo, aqi_hi])

    numeric_vals = pd.to_numeric(pd.Series(values), errors="coerce")
    interpolated = pd.Series(
        np.interp(numeric_vals.to_numpy(dtype=float), conc_points, aqi_points),
        index=numeric_vals.index
    )
    interpolated[numeric_vals.isna() | (numeric_vals < 0)] = np.nan
    interpolated[numeric_vals > conc_points[-1]] = 500.0
    return interpolated


def add_rolling_aqi(df: pd.DataFrame, station_col: str = "station",
                    min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Computes rolling sub-indices and attaches overall CPCB rolling AQI to an hourly time-series frame."""
    if df is None or len(df) == 0:
        return df

    output_frames = []
    for station_name, group in df.groupby(station_col, sort=False):
        sorted_group = group.sort_values("datetime").copy().set_index("datetime")
        full_timeline = sorted_group.reindex(pd.date_range(sorted_group.index.min(), sorted_group.index.max(), freq="h"))
        full_timeline[station_col] = station_name

        sub_dict = {}
        for pollutant, window_hours in AVERAGING_HOURS.items():
            if pollutant not in full_timeline.columns:
                continue
            min_required = max(1, math.ceil(window_hours * min_coverage))
            rolling_mean = full_timeline[pollutant].rolling(window_hours, min_periods=min_required).mean()
            full_timeline[f"{pollutant}_avg"] = rolling_mean
            sub_dict[pollutant] = sub_index_series(pollutant, rolling_mean)

        raw_dict = {}
        for pollutant in AVERAGING_HOURS:
            if pollutant in full_timeline.columns:
                raw_dict[pollutant] = sub_index_series(pollutant, full_timeline[pollutant])

        if raw_dict:
            raw_df = pd.DataFrame(raw_dict)
            has_pm_raw = raw_df[[c for c in ("pm2_5", "pm10") if c in raw_df]].notna().any(axis=1)
            full_timeline["aqi_instant"] = raw_df.max(axis=1).where(has_pm_raw)
        else:
            full_timeline["aqi_instant"] = np.nan

        if sub_dict:
            sub_df = pd.DataFrame(sub_dict)
            has_pm = sub_df[[c for c in ("pm2_5", "pm10") if c in sub_df]].notna().any(axis=1)
            has_enough_pollutants = sub_df.notna().sum(axis=1) >= MIN_SUBINDICES
            computed_aqi = sub_df.max(axis=1).where(has_pm & has_enough_pollutants)

            full_timeline["aqi"] = computed_aqi
            usable_rows = sub_df.notna().any(axis=1)
            dominant_series = pd.Series(pd.NA, index=sub_df.index, dtype="object")
            if usable_rows.any():
                dominant_series.loc[usable_rows] = sub_df.loc[usable_rows].idxmax(axis=1)
            full_timeline["dominant"] = dominant_series.where(computed_aqi.notna())

            for pollutant in sub_df.columns:
                full_timeline[f"{pollutant}_sub"] = sub_df[pollutant]
        else:
            full_timeline["aqi"] = np.nan
            full_timeline["dominant"] = None

        output_frames.append(full_timeline.rename_axis("datetime").reset_index())

    return pd.concat(output_frames, ignore_index=True)
