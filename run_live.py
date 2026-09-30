#!/usr/bin/env python3
"""
Optional Bloomberg ingest only — NOT the prediction path.

  every DATA_UPDATE_HOURS → pull 1 BBG row, append to sofr_live.csv, trim oldest

Predictions + email live in run_prediction.py (reads the store; no ingest).
Replace bbg_pull.fetch_bloomberg_latest with your real Bloomberg call.
"""

from __future__ import annotations

import time

from curve_config import (
    DATA_UPDATE_HOURS,
    LIVE_STORE_PATH,
    MAX_HISTORY_BARS,
)
from bbg_pull import fetch_bloomberg_latest
from live_store import append_snapshot, ensure_seed_history


def _hours_to_seconds(h: float) -> float:
    return max(float(h), 0.01) * 3600.0


def tick_ingest() -> None:
    snap = fetch_bloomberg_latest()
    panel = append_snapshot(snap, path=LIVE_STORE_PATH, max_bars=MAX_HISTORY_BARS)
    print(
        f"[ingest] +1 → {LIVE_STORE_PATH}  n={len(panel)}  "
        f"last={panel.index[-1]}",
        flush=True,
    )


def main() -> None:
    ensure_seed_history(LIVE_STORE_PATH)
    print(
        f"ingest-only: every {DATA_UPDATE_HOURS}h → {LIVE_STORE_PATH} "
        f"(keep last {MAX_HISTORY_BARS}). "
        f"Use run_prediction.py for forecasts/email.",
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
