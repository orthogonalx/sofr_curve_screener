#!/usr/bin/env python3
"""Prediction: refresh live data → CHOSEN_MODELS next-bar forecast."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd

import curve_config as _curve_cfg
from curve_config import (
    CHOSEN_MODELS,
    DATA_BAR_MINUTES,
    PREDICTION_EMAIL_EVERY_HOURS,
    REFRESH_DATA_ON_REQUEST,
    TRAIN_SIZE,
)
from data import build_panel, load_panel, print_live_levels, refresh_live_store
from models import (
    PREDICTION_COLS,
    append_zscore_latest,
    format_prediction_table,
    run_chosen_predictions,
    send_summary_email,
)

# Last-run metadata (avoid DataFrame.attrs — missing on pandas 0.25 / DataZone)
LAST_ASOF = ""
LAST_TEXT = ""

SEND_EMAIL_ON_PREDICTION = getattr(_curve_cfg, "SEND_EMAIL_ON_PREDICTION", True)


def run_once(
    *,
    refresh: bool = True,
    force_history: bool = False,
    send_email: Optional[bool] = None,
) -> pd.DataFrame:
    global LAST_ASOF, LAST_TEXT

    if refresh or force_history:
        data_raw, src = refresh_live_store(force_history=force_history)
    else:
        data_raw, src = load_panel()
    print_live_levels(data_raw, src=src)
    _, full, pred = build_panel(data_raw)
    asof = str(data_raw.index[-1])
    print(
        f"prediction  src={src}  n={len(data_raw)}  asof={asof}  "
        f"models={[m.name for m in CHOSEN_MODELS]}",
        flush=True,
    )
    table = format_prediction_table(
        append_zscore_latest(
            run_chosen_predictions(full, pred, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE),
            full,
        )
    )
    cols = [c for c in PREDICTION_COLS if c in table.columns]
    show = table[cols]
    text = (
        f"SOFR prediction  |  asof {asof}  |  k={TRAIN_SIZE}  |  "
        f"next ~{DATA_BAR_MINUTES}m\n\n{show.to_string(index=False)}\n"
    )
    print(text, flush=True)
    csv_path = Path("predictions_latest.csv")
    csv_path.write_text(table.to_csv(index=False))
    LAST_ASOF = asof
    LAST_TEXT = text

    do_email = SEND_EMAIL_ON_PREDICTION if send_email is None else bool(send_email)
    if do_email:
        send_summary_email(
            table,
            asof=asof,
            subject="SOFR prediction %s" % asof,
            body=text,
            attach_paths=[str(csv_path)] if csv_path.exists() else None,
        )
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--pull-history", action="store_true")
    p.add_argument("--no-email", action="store_true", help="Skip email even if EMAIL_ENABLED")
    args = p.parse_args()
    refresh = REFRESH_DATA_ON_REQUEST and not args.no_refresh

    def _tick() -> None:
        run_once(
            refresh=refresh,
            force_history=args.pull_history,
            send_email=False if args.no_email else None,
        )

    if args.once:
        _tick()
        return
    every = max(float(PREDICTION_EMAIL_EVERY_HOURS), 0.01) * 3600.0
    print(f"prediction loop every {PREDICTION_EMAIL_EVERY_HOURS}h", flush=True)
    while True:
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001
            print(f"[prediction] ERROR {exc}", flush=True)
        time.sleep(every)


if __name__ == "__main__":
    main()
