"""Bulk-backfill CPCB hourly history from OpenAQ into the parquet store.

Resumable: stations already covered in the store are skipped unless
--force is passed. Prints a gap report at the end, which decides the
sequence-rejection thresholds used during training.

  python scripts/download_cpcb_history.py --dry-run
  python scripts/download_cpcb_history.py --from 2025-02-01
  python scripts/download_cpcb_history.py --all
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import CPCB_STORE, FORECAST_STATIONS
from src.sources import store
from src.sources.cpcb_history import discover_stations, fetch_station_history


def normalise(name):
    """'Anand Vihar, New Delhi - DPCC' -> 'anand vihar'.

    data.gov.in and OpenAQ spell the same station differently ('Delhi' vs
    'New Delhi'), so match on the locality alone.
    """
    if not name:
        return ""
    head = name.split(",")[0]
    head = re.sub(r"[^a-z0-9 ]", " ", head.lower())
    return re.sub(r"\s+", " ", head).strip()


def select(stations, wanted):
    if wanted is None:
        return stations
    keys = {normalise(w) for w in wanted}
    return [s for s in stations if normalise(s["name"]) in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default="2025-02-01")
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--all", action="store_true", help="every discovered station")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download covered stations")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("Discovering CPCB stations via OpenAQ...")
    stations = discover_stations()
    print(f"  found {len(stations)}")

    targets = select(stations, None if args.all else FORECAST_STATIONS)
    if args.limit:
        targets = targets[: args.limit]

    if not args.all:
        matched = {normalise(s["name"]) for s in targets}
        missing = [w for w in FORECAST_STATIONS if normalise(w) not in matched]
        if missing:
            print(f"  WARNING unmatched forecast stations: {missing}")

    print(f"  downloading {len(targets)} station(s), "
          f"{sum(len(s['sensors']) for s in targets)} sensors, from {args.date_from}")

    if args.dry_run:
        for s in targets:
            units = {p: v["units"] for p, v in s["sensors"].items()}
            print(f"  {s['name'][:46]:<46} {units}")
        return

    existing = store.load()
    covered = set(existing["station"].unique()) if len(existing) else set()

    for i, st in enumerate(targets, 1):
        if not args.force and st["name"] in covered:
            print(f"[{i}/{len(targets)}] {st['name']} -- already in store, skipping")
            continue
        print(f"[{i}/{len(targets)}] {st['name']}")
        try:
            df = fetch_station_history(st, args.date_from, args.date_to,
                                       log=lambda m: print(m, flush=True))
        except Exception as exc:
            print(f"      FAILED: {type(exc).__name__}: {exc}")
            continue
        if len(df) == 0:
            print("      no data")
            continue
        store.append(df)
        print(f"      stored {len(df)} rows", flush=True)

    report()


def report():
    df = store.load()
    if len(df) == 0:
        print("\nstore is empty")
        return
    print(f"\n{'='*70}\nSTORE: {len(df):,} rows, {df['station'].nunique()} stations, "
          f"{df['datetime'].min():%Y-%m-%d} -> {df['datetime'].max():%Y-%m-%d}")

    cov = store.coverage(df)
    print("\nCOVERAGE (pm2_5)")
    print(cov.to_string(index=False))

    gaps = store.gap_lengths(df)
    if len(gaps):
        print(f"\nGAP LENGTHS (pm2_5): {len(gaps)} gaps total")
        buckets = pd.cut(gaps["gap_hours"], [0, 1, 3, 6, 24, 72, 10**6],
                         labels=["1h", "2-3h", "4-6h", "7-24h", "1-3d", ">3d"])
        summary = gaps.groupby(buckets, observed=False).agg(
            gaps=("gap_hours", "size"), hours_lost=("gap_hours", "sum"))
        summary["pct_of_lost_hours"] = (
            100 * summary["hours_lost"] / summary["hours_lost"].sum()).round(1)
        print(summary.to_string())
        short = gaps[gaps["gap_hours"] <= 3]["gap_hours"].sum()
        total = gaps["gap_hours"].sum()
        print(f"\n  interpolatable (<=3h): {short:,} of {total:,} missing hours "
              f"({100*short/total:.1f}%)")


if __name__ == "__main__":
    main()
