#!/usr/bin/env python3
"""
Prediction mode — locked CHOSEN_MODELS only.

Fit on last k bars → y_hat_now (last bar OOS) + y_hat_next (next bar ~ DATA_BAR_MINUTES).
Dataset is updated outside this script; re-run after each new bar.

  python3.8 -u run_prediction.py --once     # single shot
  python3.8 -u run_prediction.py            # loop; email every PREDICTION_EMAIL_EVERY_HOURS
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from curve_config import (
    CHOSEN_MODELS,
    DATA_BAR_MINUTES,
    PREDICTION_EMAIL_EVERY_HOURS,
    TRAIN_SIZE,
)
from report_email import send_summary_email
from screener_core import (
    append_zscore_latest,
    build_panel,
    load_panel,
    run_chosen_predictions,
)


def _format_prediction_text(table: pd.DataFrame, asof: str) -> str:
    show = table.copy()
    for c in ("nivel", "y_hat_now", "residual", "y_hat_next"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
    cols = [
        c
        for c in [
            "model",
            "structure",
            "features",
            "nivel",
            "y_hat_now",
            "residual",
            "y_hat_next",
            "equation",
            "time",
        ]
        if c in show.columns
    ]
    show = show[cols]
    header = (
        f"SOFR prediction  |  asof {asof}  |  k={TRAIN_SIZE}  |  "
        f"next bar ~{DATA_BAR_MINUTES}m"
    )
    return header + "\n\n" + show.to_string(index=False) + "\n"


def run_once() -> pd.DataFrame:
    data_raw, src = load_panel()
    _, full_data, predicted_data = build_panel(data_raw)
    asof = str(data_raw.index[-1])
    print(
        f"prediction  src={src}  n={len(data_raw)}  asof={asof}  "
        f"models={[m.name for m in CHOSEN_MODELS]}",
        flush=True,
    )
    table = run_chosen_predictions(
        full_data, predicted_data, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE
    )
    table = append_zscore_latest(table, full_data)
    text = _format_prediction_text(table, asof=asof)
    print(text, flush=True)
    Path("predictions_latest.csv").write_text(table.to_csv(index=False))
    table.attrs["asof"] = asof
    table.attrs["text"] = text
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="SOFR locked-model prediction")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one prediction and exit (no email loop)",
    )
    args = parser.parse_args()

    if args.once:
        table = run_once()
        asof = table.attrs.get("asof", "")
        send_summary_email(
            table,
            asof=asof,
            subject=f"SOFR prediction {asof}",
            body=table.attrs.get("text"),
        )
        return

    every = max(float(PREDICTION_EMAIL_EVERY_HOURS), 0.01) * 3600.0
    print(
        f"prediction loop: refresh + email every {PREDICTION_EMAIL_EVERY_HOURS}h "
        f"(dataset updated externally every ~{DATA_BAR_MINUTES}m)",
        flush=True,
    )
    while True:
        try:
            table = run_once()
            asof = table.attrs.get("asof", "")
            send_summary_email(
                table,
                asof=asof,
                subject=f"SOFR prediction {asof}",
                body=table.attrs.get("text"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[prediction] ERROR {exc}", flush=True)
        time.sleep(every)


if __name__ == "__main__":
    main()
