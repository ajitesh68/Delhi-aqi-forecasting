"""Append one live CPCB sweep to the parquet store.

Run this hourly. The OpenAQ archive lags several days and occasionally
stalls, so this is what closes the gap between the archive and now: every
run adds the current hour for every reporting station, and after a week
the store has a continuous window ending at the present hour.

  python scripts/collect_snapshot.py
  schtasks /create /tn DelhiAQI /tr "...python.exe scripts\\collect_snapshot.py" /sc hourly
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import NCR_CITIES, RECENT_STORE
from src.sources import store
from src.sources.cpcb_live import fetch_live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=NCR_CITIES)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(msg):
        if not args.quiet:
            print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    df, meta = fetch_live(args.cities, use_cache_on_error=False)
    if len(df) == 0:
        log("no records returned")
        return 1
    if meta.get("stale"):
        log("served from cache, not appending stale rows")
        return 1

    before = store.load(RECENT_STORE, recent=None)
    after = store.append_recent(df)
    span = f"{after['datetime'].min():%d %b} to {after['datetime'].max():%d %b}" \
        if len(after) else "empty"
    log(f"{meta['stations']} stations, {len(df)} station-hours at "
        f"{meta['last_update']} -- rolling file {len(before):,} -> "
        f"{len(after):,} rows ({span})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
