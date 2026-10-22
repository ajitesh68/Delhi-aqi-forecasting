from typing import Any

import pandas as pd

MIN_HOURS_PER_DAY = 12


def _clean_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures datetime column is formatted properly and drops rows with missing timestamps."""
    clean_df = df.copy()
    clean_df["datetime"] = pd.to_datetime(clean_df["datetime"])
    return clean_df.dropna(subset=["datetime"])


def diurnal_profile(
    df: pd.DataFrame,
    value: str = "aqi_instant",
    by: str | None = "station",
    value_col: str | None = None,
    by_station: str | None = None,
) -> pd.DataFrame:
    """
    Computes typical hour-of-day statistics (00:00 to 23:00) across all observed days:
    mean, median, 25th percentile, 75th percentile, and observed count.
    Used by the 'When to go out' dashboard tab.
    """
    target_val = value_col if value_col is not None else value
    group_col = by_station if by_station is not None else by

    cleaned = _clean_datetime(df)
    if target_val not in cleaned.columns:
        return pd.DataFrame()

    cleaned["hour"] = cleaned["datetime"].dt.hour
    group_keys = ["hour"] + ([group_col] if group_col and group_col in cleaned.columns else [])

    profile = (
        cleaned.groupby(group_keys)[target_val]
        .agg(
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            n_observed="count",
        )
        .reset_index()
    )
    return profile.sort_values(group_keys).reset_index(drop=True)


def daily_summary(
    df: pd.DataFrame,
    value: str = "aqi",
    by: str | None = "station",
    min_hours: int = MIN_HOURS_PER_DAY,
    value_col: str | None = None,
    by_station: str | None = None,
) -> pd.DataFrame:
    """
    Rolls hourly observations into daily summaries (mean, max, min).
    Marks days with fewer than `min_hours` (default 12) observed hours as 'sparse'.
    """
    target_val = value_col if value_col is not None else value
    group_col = by_station if by_station is not None else by

    cleaned = _clean_datetime(df)
    if target_val not in cleaned.columns:
        return pd.DataFrame()

    cleaned["date"] = cleaned["datetime"].dt.normalize()
    group_keys = ["date"] + ([group_col] if group_col and group_col in cleaned.columns else [])

    summary = (
        cleaned.groupby(group_keys)[target_val]
        .agg(mean="mean", max="max", min="min", n_observed="count")
        .reset_index()
    )
    summary["sparse"] = summary["n_observed"] < min_hours
    return summary.sort_values(group_keys).reset_index(drop=True)


def fill_daily_gaps(
    daily_df: pd.DataFrame,
    by: str | None = "station",
    max_run: int | None = None,
    by_station: str | None = None,
) -> pd.DataFrame:
    """
    Fills gaps in daily time-series by creating a complete date grid.
    Linearly interpolates missing values up to `max_run` consecutive missing days.
    Marks interpolated days with `estimated=True` so charts can render them as dotted lines.
    """
    group_col = by_station if by_station is not None else by
    if len(daily_df) == 0 or "date" not in daily_df.columns:
        return daily_df

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["estimated"] = False

    station_groups = df.groupby(group_col) if group_col and group_col in df.columns else [(None, df)]
    filled_groups = []

    for station_name, group in station_groups:
        sorted_group = group.sort_values("date").set_index("date")
        full_calendar = pd.date_range(sorted_group.index.min(), sorted_group.index.max(), freq="D")
        reindexed = sorted_group.reindex(full_calendar)

        missing_mask = reindexed["mean"].isna()

        # If a gap is longer than max_run days, do not interpolate it
        if max_run and missing_mask.any():
            run_identifiers = (missing_mask != missing_mask.shift()).cumsum()
            gap_lengths = missing_mask.groupby(run_identifiers).transform("sum")
            exceeds_threshold = gap_lengths > max_run
            missing_mask = missing_mask & ~exceeds_threshold

        for col in ("mean", "max", "min"):
            if col in reindexed.columns:
                reindexed[col] = reindexed[col].interpolate(limit_area="inside")

        reindexed["estimated"] = missing_mask.fillna(False)
        reindexed["n_observed"] = reindexed["n_observed"].fillna(0)

        if group_col and station_name is not None:
            reindexed[group_col] = station_name
        if "sparse" in reindexed.columns:
            reindexed["sparse"] = reindexed["sparse"].fillna(False).astype(bool)

        filled_groups.append(reindexed.rename_axis("date").reset_index())

    combined_filled = pd.concat(filled_groups, ignore_index=True)
    return combined_filled.dropna(subset=["mean"]).reset_index(drop=True)
