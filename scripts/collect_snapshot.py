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
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import NCR_CITIES, RECENT_STORE
from src.sources import store
from src.sources.cpcb_live import RateLimited, fetch_live

# A missed hour costs almost nothing -- the next run is sixty minutes away
# and the rolling window spans two weeks. A red cross every time the feed
# rate-limits costs a great deal, because it trains you to ignore the
# Actions tab and then a real break goes unnoticed. Transient failures are
# therefore reported and skipped, not raised.
TRANSIENT = (RateLimited, urllib.error.URLError, TimeoutError, OSError)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=NCR_CITIES)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(msg):
        if not args.quiet:
            print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    try:
        df, meta = fetch_live(args.cities, use_cache_on_error=False)
    except TRANSIENT as exc:
        # The demo key is shared globally and CI runners share addresses,
        # so 429 is an ordinary outcome here rather than a fault.
        log(f"skipped -- {type(exc).__name__}: {exc}")
        return 0

    if len(df) == 0:
        log("skipped -- feed returned no records")
        return 0
    if meta.get("stale"):
        log("skipped -- served from cache, refusing to append stale rows")
        return 0

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
