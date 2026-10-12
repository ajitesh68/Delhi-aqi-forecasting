import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import RECENT_STORE, waqi_key
from src.sources import store
from src.sources.cpcb_live import fetch_live

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(msg):
        if not args.quiet:
            print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    if not waqi_key():
        log("FAILED: WAQI_API_KEY is not set.")
        return 1

    try:
        df, meta = fetch_live(use_cache_on_error=False)
    except Exception as exc:
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
