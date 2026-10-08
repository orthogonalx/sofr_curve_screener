#!/usr/bin/env python3
"""
Live ingest.

  If sofr_live.csv is missing → pull [BBG_START_DATE, BBG_END_DATE] at BBG_FREQ.
  Then every DATA_UPDATE_HOURS → pull the latest print and append it.

Reports read that same file (run_report.py does the same append on each run).
"""

from __future__ import annotations

import time
from pathlib import Path

from curve_config import (
    BBG_END_DATE,
    BBG_FREQ,
    BBG_START_DATE,
    DATA_UPDATE_HOURS,
    LIVE_STORE_PATH,
)
from bbg_pull import append_live_bar, pull_history_into_store
from live_store import load_live_store


def _hours_to_seconds(h: float) -> float:
    return max(float(h), 0.01) * 3600.0


def tick_ingest() -> None:
    append_live_bar()


def main() -> None:
    store = Path(LIVE_STORE_PATH)
    existing = load_live_store(store) if store.exists() else None
    if existing is None or existing.empty:
        end = BBG_END_DATE or "today"
        print(
            f"[history] {BBG_START_DATE} → {end}  freq={BBG_FREQ} → {LIVE_STORE_PATH}",
            flush=True,
        )
        pull_history_into_store()
    print(
        f"live: append latest print every {DATA_UPDATE_HOURS}h → {LIVE_STORE_PATH}",
        flush=True,
    )

    ingest_every = _hours_to_seconds(DATA_UPDATE_HOURS)
    next_ingest = time.monotonic()

    while True:
        now = time.monotonic()
        if now >= next_ingest:
            try:
                tick_ingest()
            except Exception as exc:  # noqa: BLE001 — keep loop alive
                print(f"[ingest] ERROR {exc}", flush=True)
            next_ingest = now + ingest_every

        sleep_for = next_ingest - time.monotonic()
        time.sleep(max(sleep_for, 1.0))


if __name__ == "__main__":
    main()
