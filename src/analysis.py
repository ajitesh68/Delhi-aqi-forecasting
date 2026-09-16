"""Roll-ups computed from observed CPCB readings."""

import pandas as pd

MIN_HOURS_PER_DAY = 12


def _prep(df):
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    return d.dropna(subset=["datetime"])


def diurnal_profile(df, value="aqi_instant", by="station"):
    """Average value for each hour of the day."""
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


def fill_daily_gaps(daily, by="station", max_run=None):
    """Interpolate absent days, flagging every filled row as estimated."""
    if len(daily) == 0 or "date" not in daily.columns:
        return daily

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["estimated"] = False
    groups = d.groupby(by) if by and by in d.columns else [(None, d)]

    out = []
    for name, grp in groups:
        grp = grp.sort_values("date").set_index("date")
        full = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full)
        missing = grp["mean"].isna()

        if max_run and missing.any():
            runs = (missing != missing.shift()).cumsum()
            too_wide = missing.groupby(runs).transform("sum") > max_run
            missing = missing & ~too_wide

        for col in ("mean", "max", "min"):
            if col in grp.columns:
                grp[col] = grp[col].interpolate(limit_area="inside")

        grp["estimated"] = missing.fillna(False)
        grp["n_observed"] = grp["n_observed"].fillna(0)
        if by and name is not None:
            grp[by] = name
        if "sparse" in grp.columns:
            grp["sparse"] = grp["sparse"].fillna(False).astype(bool)
        out.append(grp.rename_axis("date").reset_index())

    filled = pd.concat(out, ignore_index=True)
    return filled.dropna(subset=["mean"]).reset_index(drop=True)
