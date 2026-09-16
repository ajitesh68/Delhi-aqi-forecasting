"""Parquet store for CPCB hourly observations."""

import os
import pandas as pd

from src.config import (CPCB_STORE, POLLUTANTS, RECENT_STORE,
                        RECENT_WINDOW_DAYS, STORE_DIR)

SOURCE_PRIORITY = {"openaq": 3, "datagov": 2, "interpolated": 1}

META_COLUMNS = ["datetime", "station", "city", "lat", "lon", "source", "is_observed"]
COLUMNS = META_COLUMNS + POLLUTANTS


def empty_frame():
    df = pd.DataFrame(columns=COLUMNS)
    return _coerce(df)


def _coerce(df):
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(df["datetime"].dtype, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["datetime"] = df["datetime"].dt.floor("h")
    for col in POLLUTANTS + ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("station", "city", "source"):
        df[col] = df[col].astype("string")
    df["is_observed"] = df["is_observed"].fillna(False).astype(bool)
    return df[COLUMNS]


def read_file(path):
    """One parquet file, coerced to the store schema. Missing file is empty."""
    if not os.path.exists(path):
        return empty_frame()
    return _coerce(pd.read_parquet(path))


def load(path=CPCB_STORE, stations=None, start=None, end=None, recent=RECENT_STORE):
    """The whole store: archive plus the rolling recent window."""
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


def save(df, path=CPCB_STORE):
    os.makedirs(STORE_DIR, exist_ok=True)
    _coerce(df).to_parquet(path, index=False)


def merge(existing, incoming):
    """Union two frames, keeping the higher-priority source per station-hour."""
    existing, incoming = _coerce(existing), _coerce(incoming)
    if len(incoming) == 0:
        return existing
    if len(existing) == 0:
        combined = incoming
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)

    combined = combined.dropna(subset=["datetime", "station"])
    combined["_rank"] = combined["source"].map(SOURCE_PRIORITY).fillna(0)
    combined = combined.sort_values(["station", "datetime", "_rank"])

    grouped = combined.groupby(["station", "datetime"], as_index=False, sort=False)
    out = grouped.agg({
        **{p: "last" for p in POLLUTANTS},
        "city": "last", "lat": "last", "lon": "last",
        "source": "last", "is_observed": "max", "_rank": "max",
    })
    out = out.drop(columns=["_rank"])
    return _coerce(out).sort_values(["station", "datetime"]).reset_index(drop=True)


def canonical_names(incoming, existing):
    """Rename incoming stations to match ones already stored."""
    if len(incoming) == 0 or len(existing) == 0:
        return incoming

    def locality(name):
        return str(name).split(",")[0].strip().lower()

    known = {}
    for name in existing["station"].dropna().unique():
        known.setdefault(locality(name), name)

    mapping = {name: known[locality(name)]
               for name in incoming["station"].dropna().unique()
               if locality(name) in known and known[locality(name)] != name}
    if not mapping:
        return incoming

    out = incoming.copy()
    out["station"] = out["station"].replace(mapping)
    return out


def append(incoming, path=CPCB_STORE):
    """Merge rows into one store file. Used by the archive backfill."""
    existing = load(path, recent=None)
    merged = merge(existing, canonical_names(incoming, existing))
    save(merged, path)
    return merged


def append_recent(incoming, path=RECENT_STORE, archive=CPCB_STORE,
                  days=RECENT_WINDOW_DAYS):
    """Add an hour to the rolling file and drop whatever fell out of it."""
    existing = load(path, recent=None)
    reference = load(archive, recent=None)
    if len(reference) == 0:
        reference = existing

    merged = merge(existing, canonical_names(incoming, reference))
    if len(merged) and days:
        cutoff = merged["datetime"].max() - pd.Timedelta(days=days)
        merged = merged[merged["datetime"] >= cutoff]

    save(merged, path)
    return merged


def coverage(df, pollutant="pm2_5"):
    """Per-station coverage summary: span, observed hours, gap fraction."""
    rows = []
    for station, g in df.groupby("station"):
        g = g.sort_values("datetime")
        vals = g[pollutant]
        first, last = g["datetime"].min(), g["datetime"].max()
        expected = int((last - first).total_seconds() // 3600) + 1 if pd.notna(first) else 0
        observed = int(vals.notna().sum())
        rows.append({
            "station": station, "first": first, "last": last,
            "expected_hours": expected, "observed_hours": observed,
            "gap_fraction": round(1 - observed / expected, 4) if expected else 1.0,
        })
    return pd.DataFrame(rows).sort_values("gap_fraction").reset_index(drop=True)


def gap_lengths(df, pollutant="pm2_5"):
    """Distribution of consecutive-missing-hour run lengths, per station."""
    runs = []
    for station, g in df.groupby("station"):
        g = g.sort_values("datetime").set_index("datetime")
        full = g[pollutant].reindex(
            pd.date_range(g.index.min(), g.index.max(), freq="h"))
        missing = full.isna().values
        length = 0
        for m in missing:
            if m:
                length += 1
            elif length:
                runs.append({"station": station, "gap_hours": length})
                length = 0
        if length:
            runs.append({"station": station, "gap_hours": length})
    return pd.DataFrame(runs)
