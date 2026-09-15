"""Historical roll-ups computed from observed CPCB readings.

Every function here reads the parquet store of real station measurements.
No model output is involved: the LSTM exists only to forecast the next 24
hours and contributes nothing to any chart produced by this module.

The store is hourly, because that is the resolution CPCB stations report
at. Daily, weekly and monthly views are aggregations of those real hours,
and each carries `n_observed` so a period built from thin data is visible
rather than silently averaged away.
"""

import numpy as np
import pandas as pd

from src.config import DIWALI_DATES, POLLUTANTS

MIN_HOURS_PER_DAY = 12       # a "daily mean" needs at least half a day
MIN_DAYS_PER_MONTH = 10


def _prep(df):
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    return d.dropna(subset=["datetime"])


def diurnal_profile(df, value="aqi_instant", by="station"):
    """Average value for each hour of the day.

    Defaults to `aqi_instant` rather than `aqi`: the CPCB index is a
    24-hour rolling mean, which by construction erases the diurnal cycle
    this function exists to expose.
    """
    d = _prep(df)
    if value not in d.columns:
        return pd.DataFrame()
    d["hour"] = d["datetime"].dt.hour
    keys = ["hour"] + ([by] if by and by in d.columns else [])
    out = (d.groupby(keys)[value]
             .agg(mean="mean", median="median", p25=lambda s: s.quantile(.25),
                  p75=lambda s: s.quantile(.75), n_observed="count")
             .reset_index())
    return out.sort_values(keys).reset_index(drop=True)


def daily_summary(df, value="aqi", by="station", min_hours=MIN_HOURS_PER_DAY):
    d = _prep(df)
    if value not in d.columns:
        return pd.DataFrame()
    d["date"] = d["datetime"].dt.normalize()
    keys = ["date"] + ([by] if by and by in d.columns else [])
    out = (d.groupby(keys)[value]
             .agg(mean="mean", max="max", min="min", n_observed="count")
             .reset_index())
    out["sparse"] = out["n_observed"] < min_hours
    return out.sort_values(keys).reset_index(drop=True)


def weekly_summary(df, value="aqi", by="station"):
    daily = daily_summary(df, value=value, by=by)
    if len(daily) == 0:
        return daily
    daily = daily[~daily["sparse"]]
    daily["week"] = pd.to_datetime(daily["date"]).dt.to_period("W").dt.start_time
    keys = ["week"] + ([by] if by and by in daily.columns else [])
    return (daily.groupby(keys)
                 .agg(mean=("mean", "mean"), max=("max", "max"),
                      n_days=("mean", "size"))
                 .reset_index()
                 .sort_values(keys).reset_index(drop=True))


def monthly_summary(df, value="aqi", by="station", min_days=MIN_DAYS_PER_MONTH):
    daily = daily_summary(df, value=value, by=by)
    if len(daily) == 0:
        return daily
    daily = daily[~daily["sparse"]]
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").dt.start_time
    keys = ["month"] + ([by] if by and by in daily.columns else [])
    out = (daily.groupby(keys)
                .agg(mean=("mean", "mean"), max=("max", "max"),
                     p90=("mean", lambda s: s.quantile(.9)), n_days=("mean", "size"))
                .reset_index())
    out["sparse"] = out["n_days"] < min_days
    return out.sort_values(keys).reset_index(drop=True)


def day_of_week_effect(df, value="aqi_instant"):
    """Weekday vs weekend contrast, mostly a traffic signal.

    Uses raw hourly readings for the same reason `diurnal_profile` does:
    a 24-hour rolling mean smears one day into the next.
    """
    d = _prep(df)
    if value not in d.columns:
        return pd.DataFrame()
    d["dow"] = d["datetime"].dt.dayofweek
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = (d.groupby("dow")[value]
             .agg(mean="mean", median="median", n_observed="count").reset_index())
    out["day"] = out["dow"].map(dict(enumerate(names)))
    out["is_weekend"] = out["dow"] >= 5
    return out.sort_values("dow").reset_index(drop=True)


def year_over_year(df, value="aqi", by=None):
    """Same calendar window compared across years.

    Restricted to the day-of-year range both years actually cover, so a
    partial year is not compared against a full one.
    """
    d = _prep(df)
    if value not in d.columns or len(d) == 0:
        return pd.DataFrame()
    d["year"] = d["datetime"].dt.year
    d["doy"] = d["datetime"].dt.dayofyear
    years = sorted(d["year"].unique())
    if len(years) < 2:
        return pd.DataFrame()

    spans = d.groupby("year")["doy"].agg(["min", "max"])
    lo, hi = int(spans["min"].max()), int(spans["max"].min())
    if lo >= hi:
        return pd.DataFrame()

    window = d[(d["doy"] >= lo) & (d["doy"] <= hi)].copy()
    window["month"] = window["datetime"].dt.month
    keys = ["year", "month"] + ([by] if by and by in window.columns else [])
    out = (window.groupby(keys)[value]
                 .agg(mean="mean", n_observed="count").reset_index())
    out.attrs["doy_range"] = (lo, hi)
    return out


def station_ranking(df, value="aqi", window_hours=24):
    """Rank stations over the most recent `window_hours` of the frame."""
    d = _prep(df)
    if value not in d.columns or len(d) == 0:
        return pd.DataFrame()
    cutoff = d["datetime"].max() - pd.Timedelta(hours=window_hours)
    recent = d[d["datetime"] > cutoff]
    out = (recent.groupby("station")[value]
                 .agg(mean="mean", max="max", n_observed="count").reset_index())
    out = out[out["n_observed"] >= max(1, window_hours // 4)]
    return out.sort_values("mean").reset_index(drop=True)


def pollutant_mix(df, pollutants=None, window_hours=24):
    """Mean level of each pollutant over a recent window, for the radar chart."""
    d = _prep(df)
    cols = [p for p in (pollutants or POLLUTANTS) if p in d.columns]
    if not cols or len(d) == 0:
        return pd.DataFrame()
    cutoff = d["datetime"].max() - pd.Timedelta(hours=window_hours)
    recent = d[d["datetime"] > cutoff]
    rows = []
    for p in cols:
        vals = pd.to_numeric(recent[p], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        sub = recent.get(f"{p}_sub")
        rows.append({
            "pollutant": p,
            "mean": round(float(vals.mean()), 1),
            "sub_index": round(float(pd.to_numeric(sub, errors="coerce").mean()), 1)
                          if sub is not None and sub.notna().any() else np.nan,
            "n_observed": int(len(vals)),
        })
    return pd.DataFrame(rows)


def seasonal_windows(df, value="aqi"):
    """Named Delhi pollution episodes measured against the real series."""
    d = _prep(df)
    if value not in d.columns or len(d) == 0:
        return pd.DataFrame()
    rows = []
    for year, date in DIWALI_DATES.items():
        centre = pd.Timestamp(date)
        for label, lo, hi in [("Pre-Diwali week", -10, -4), ("Diwali +/- 3d", -3, 3),
                              ("Post-Diwali week", 4, 10)]:
            window = d[(d["datetime"] >= centre + pd.Timedelta(days=lo)) &
                       (d["datetime"] <= centre + pd.Timedelta(days=hi))]
            vals = pd.to_numeric(window[value], errors="coerce").dropna()
            if len(vals) < 24:
                continue
            rows.append({"year": year, "window": label,
                         "mean": round(float(vals.mean())),
                         "max": round(float(vals.max())),
                         "n_observed": int(len(vals))})
    return pd.DataFrame(rows)
