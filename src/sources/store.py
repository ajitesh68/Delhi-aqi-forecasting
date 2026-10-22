import os
from typing import Any

import pandas as pd

from src.config import CPCB_STORE, POLLUTANTS, RECENT_STORE, RECENT_WINDOW_DAYS, STORE_DIR

SOURCE_PRIORITY = {"openaq": 3, "datagov": 2, "waqi": 2, "interpolated": 1}

META_COLUMNS = ["datetime", "station", "city", "lat", "lon", "source", "is_observed", "aqi"]
COLUMNS = META_COLUMNS + POLLUTANTS


def empty_frame() -> pd.DataFrame:
    """Returns an empty DataFrame conforming to the store schema."""
    df = pd.DataFrame(columns=COLUMNS)
    return _coerce(df)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces strict schema, datatypes, and hourly timestamp rounding on the DataFrame."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(df["datetime"].dtype, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["datetime"] = df["datetime"].dt.floor("h")

    for col in POLLUTANTS + ["lat", "lon", "aqi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("station", "city", "source"):
        df[col] = df[col].astype("string")

    df["is_observed"] = df["is_observed"].fillna(False).astype(bool)
    return df[COLUMNS]


def read_file(path: str) -> pd.DataFrame:
    """Safely reads a parquet file and enforces standard schema."""
    if not os.path.exists(path):
        return empty_frame()
    return _coerce(pd.read_parquet(path))


def load(path: str = CPCB_STORE, stations: list[str] | None = None,
         start: Any = None, end: Any = None, recent: str | None = RECENT_STORE) -> pd.DataFrame:
    """
    Loads historical station data from parquet stores.
    By default unions the archive store with the active rolling store.
    """
    df = read_file(path)
    if recent:
        extra = read_file(recent)
        if len(extra):
            df = merge(df, extra)

    if stations is not None:
        df = df[df["station"].isin(list(stations))]
    if start is not None:
        df = df[df["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["datetime"] <= pd.Timestamp(end)]

    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


def save(df: pd.DataFrame, path: str = CPCB_STORE) -> None:
    """Persists a coerced DataFrame to a parquet file."""
    os.makedirs(STORE_DIR, exist_ok=True)
    _coerce(df).to_parquet(path, index=False)


def merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """
    Combines two DataFrames. For duplicate (station, datetime) records,
    retains the record with the higher source priority.
    """
    existing_coerced = _coerce(existing)
    incoming_coerced = _coerce(incoming)

    if len(incoming_coerced) == 0:
        return existing_coerced
    if len(existing_coerced) == 0:
        return incoming_coerced

    combined = pd.concat([existing_coerced, incoming_coerced], ignore_index=True)
    combined = combined.dropna(subset=["datetime", "station"])

    combined["_rank"] = combined["source"].map(SOURCE_PRIORITY).fillna(0)
    combined = combined.sort_values(["station", "datetime", "_rank"])

    # Retain the highest priority record per station-hour
    grouped = combined.groupby(["station", "datetime"], as_index=False, sort=False)
    out = grouped.agg({
        **{pollutant: "last" for pollutant in POLLUTANTS},
        "city": "last",
        "lat": "last",
        "lon": "last",
        "source": "last",
        "is_observed": "max",
        "aqi": "last",
        "_rank": "max",
    })
    out = out.drop(columns=["_rank"])
    return _coerce(out).sort_values(["station", "datetime"]).reset_index(drop=True)


def canonical_names(incoming: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes incoming station names to match existing station names in the reference store
    based on locality (e.g. 'Anand Vihar, Delhi' -> 'Anand Vihar, New Delhi - DPCC').
    """
    if len(incoming) == 0 or len(reference) == 0:
        return incoming

    def extract_locality(name: Any) -> str:
        return str(name).split(",")[0].strip().lower()

    known_stations = {}
    for name in reference["station"].dropna().unique():
        known_stations.setdefault(extract_locality(name), name)

    name_mapping = {
        name: known_stations[extract_locality(name)]
        for name in incoming["station"].dropna().unique()
        if extract_locality(name) in known_stations and known_stations[extract_locality(name)] != name
    }

    if not name_mapping:
        return incoming

    out = incoming.copy()
    out["station"] = out["station"].replace(name_mapping)
    return out


def append_recent(incoming: pd.DataFrame, path: str = RECENT_STORE,
                  archive: str | None = None, days: int = RECENT_WINDOW_DAYS) -> pd.DataFrame:
    """
    Appends a new hourly sweep to the rolling recent store and trims rows older than `days` (default 40 days).
    Called hourly by the automated GitHub Actions collector.
    """
    existing = load(path, recent=None)
    reference = existing if (archive is None or len(existing) > 0) else load(archive, recent=None)
    if len(reference) == 0:
        reference = incoming

    normalized_incoming = canonical_names(incoming, reference)
    merged = merge(existing, normalized_incoming)

    # Trim rolling window to prevent git repository bloat
    if len(merged) and days:
        cutoff_date = merged["datetime"].max() - pd.Timedelta(days=days)
        merged = merged[merged["datetime"] >= cutoff_date]

    save(merged, path)
    return merged
