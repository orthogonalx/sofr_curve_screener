#!/usr/bin/env python3
"""
Live loop:
  every DATA_UPDATE_HOURS  → pull 1 BBG row, append to sofr_live.csv, trim oldest
  every REPORT_EVERY_HOURS → re-run screen + email terse summary

Replace bbg_pull.fetch_bloomberg_latest with your real Bloomberg call.
"""

from __future__ import annotations

import time

from curve_config import (
    DATA_UPDATE_HOURS,
    LIVE_STORE_PATH,
    LIVE_TUNE_WINDOW,
    MAX_HISTORY_BARS,
    REPORT_EVERY_HOURS,
)
from bbg_pull import fetch_bloomberg_latest
from live_store import append_snapshot, ensure_seed_history
from report_email import send_summary_email
from screener_core import emit_summary, run_screen


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


def tick_report() -> None:
    from live_store import load_live_store

    data_raw = load_live_store(LIVE_STORE_PATH)
    if data_raw.empty or len(data_raw) < 50:
        print("[report] skipped — not enough history", flush=True)
        return
    asof = str(data_raw.index[-1])
    summary, _, _ = run_screen(
        data_raw, tune_window=LIVE_TUNE_WINDOW, quiet=True
    )
    emit_summary(summary, asof=asof)
    send_summary_email(summary, asof=asof)


def main() -> None:
    ensure_seed_history(LIVE_STORE_PATH)
    print(
        f"live: ingest every {DATA_UPDATE_HOURS}h → {LIVE_STORE_PATH} "
        f"(keep last {MAX_HISTORY_BARS}); "
        f"report/email every {REPORT_EVERY_HOURS}h",
        flush=True,
    )

    ingest_every = _hours_to_seconds(DATA_UPDATE_HOURS)
    report_every = _hours_to_seconds(REPORT_EVERY_HOURS)
    next_ingest = time.monotonic()
    next_report = time.monotonic()  # report once at start after seed

    while True:
        now = time.monotonic()
        if now >= next_ingest:
            try:
                tick_ingest()
            except Exception as exc:  # noqa: BLE001 — keep loop alive
                print(f"[ingest] ERROR {exc}", flush=True)
            next_ingest = now + ingest_every

        if now >= next_report:
            try:
                tick_report()
            except Exception as exc:  # noqa: BLE001
                print(f"[report] ERROR {exc}", flush=True)
            next_report = now + report_every

        sleep_for = min(next_ingest, next_report) - time.monotonic()
        time.sleep(max(sleep_for, 1.0))


if __name__ == "__main__":
    main()
