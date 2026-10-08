#!/usr/bin/env python3
"""
Residual report — default models (CHOSEN_MODELS).

1) Button / on-demand:
       python3.8 -u run_report.py --once
   → refresh data, run analysis, print ONLY the residual table.

2) Scheduled email every REPORT_EMAIL_EVERY_HOURS (XXX):
       python3.8 -u run_report.py
   → same analysis on a timer; emails the table.

Table columns:
  parametro | fly | residual | y_real | y_pred | regressors | model | periodo
"""

from __future__ import annotations

import argparse
import time

from curve_config import REPORT_EMAIL_EVERY_HOURS
from report import format_report_text, refresh_data, run_report
from report_email import send_summary_email


def main() -> None:
    parser = argparse.ArgumentParser(description="SOFR residual report")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once (button): refresh + table; no email loop",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip BBG pull; use existing panel on disk",
    )
    parser.add_argument(
        "--pull-history",
        action="store_true",
        help="Rebuild LIVE_STORE from BBG_START_DATE/END/FREQ, then stop",
    )
    args = parser.parse_args()
    refresh = not args.no_refresh

    if args.pull_history:
        refresh_data(force_history=True)
        return

    if args.once:
        table = run_report(refresh=refresh)
        # optional email on button press if SMTP configured
        send_summary_email(
            table,
            asof=table.attrs.get("asof", ""),
            subject=f"SOFR residual report {table.attrs.get('asof', '')}",
            body=table.attrs.get("text") or format_report_text(table),
        )
        return

    every = max(float(REPORT_EMAIL_EVERY_HOURS), 0.01) * 3600.0
    print(
        f"report loop: every {REPORT_EMAIL_EVERY_HOURS}h  "
        f"(refresh_on_request={refresh})",
        flush=True,
    )
    while True:
        try:
            table = run_report(refresh=refresh)
            send_summary_email(
                table,
                asof=table.attrs.get("asof", ""),
                subject=f"SOFR residual report {table.attrs.get('asof', '')}",
                body=table.attrs.get("text") or format_report_text(table),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[report] ERROR {exc}", flush=True)
        time.sleep(every)


if __name__ == "__main__":
    main()
