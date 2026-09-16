"""Append one live CPCB sweep to the parquet store."""

import argparse
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import DATAGOV_DEMO_KEY, NCR_CITIES, RECENT_STORE, datagov_key
from src.sources import store
from src.sources.cpcb_live import RateLimited, fetch_live, page_size

TRANSIENT = (RateLimited, urllib.error.URLError, TimeoutError, OSError)

PERMANENT_STATUS = {401, 403, 404}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=NCR_CITIES)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(msg):
        if not args.quiet:
            print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    registered = datagov_key() != DATAGOV_DEMO_KEY
    log(f"{'registered' if registered else 'demo'} key, "
        f"{page_size()} records per call")

    try:
        df, meta = fetch_live(args.cities, use_cache_on_error=False)
    except urllib.error.HTTPError as exc:
        if exc.code in PERMANENT_STATUS:
            log(f"FAILED -- HTTP {exc.code}: the key was rejected or the "
                f"resource moved. Check DATA_GOV_IN_API_KEY.")
            return 1
        log(f"skipped -- HTTP {exc.code}")
        return 0
    except TRANSIENT as exc:
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
