"""Parquet store for CPCB hourly observations.

One row per (station, hour). Every row carries provenance so the UI can
tell the user which hours are real measurements and which were filled.

  source        'openaq' | 'datagov' | 'interpolated'
  is_observed   True only for rows that came from a monitoring station

When both sources cover the same hour, OpenAQ wins: it publishes a proper
hourly aggregate, whereas data.gov.in gives a snapshot of whatever the
station last reported.
"""

import os
import pandas as pd

from src.config import CPCB_STORE, POLLUTANTS, STORE_DIR

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


def load(path=CPCB_STORE, stations=None, start=None, end=None):
    if not os.path.exists(path):
        return empty_frame()
    df = pd.read_parquet(path)
    df = _coerce(df)
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
    """Union two frames, keeping the higher-priority source per station-hour.

    Rows are merged column-wise rather than replaced wholesale: a datagov
    row carrying SO2 is not discarded just because an openaq row for the
    same hour carries PM2.5.
    """
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
    # last() per column takes the highest-ranked non-null value, so a lower
    # priority source still contributes pollutants the winner is missing.
    out = grouped.agg({
        **{p: "last" for p in POLLUTANTS},
        "city": "last", "lat": "last", "lon": "last",
        "source": "last", "is_observed": "max", "_rank": "max",
    })
    out = out.drop(columns=["_rank"])
    return _coerce(out).sort_values(["station", "datetime"]).reset_index(drop=True)


def canonical_names(incoming, existing):
    """Rename incoming stations to match ones already stored.

    The two feeds spell the same monitor differently -- data.gov.in says
    "Anand Vihar, Delhi - DPCC" where the OpenAQ archive says "Anand
    Vihar, New Delhi - DPCC". Stored as written, they become two
    unrelated stations: the live snapshots never close the archive's lag,
    and the forecast never gets a continuous window. Matching on the
    locality prefix keeps one monitor as one station.
    """
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
    existing = load(path)
    merged = merge(existing, canonical_names(incoming, existing))
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
    """Distribution of consecutive-missing-hour run lengths, per station.

    Decides whether gaps are scattered singletons (safe to interpolate) or
    clustered multi-day outages (which force windows to be rejected).
    """
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
