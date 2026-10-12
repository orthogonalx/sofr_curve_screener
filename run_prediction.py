#!/usr/bin/env python3
"""Prediction: refresh live data → CHOSEN_MODELS next-bar forecast."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from curve_config import (
    CHOSEN_MODELS,
    DATA_BAR_MINUTES,
    PREDICTION_EMAIL_EVERY_HOURS,
    REFRESH_DATA_ON_REQUEST,
    TRAIN_SIZE,
)
from data import build_panel, load_panel, refresh_live_store
from models import append_zscore_latest, run_chosen_predictions, send_summary_email


def run_once(*, refresh: bool = True, force_history: bool = False) -> pd.DataFrame:
    if refresh or force_history:
        data_raw, src = refresh_live_store(force_history=force_history)
    else:
        data_raw, src = load_panel()
    _, full, pred = build_panel(data_raw)
    asof = str(data_raw.index[-1])
    print(
        f"prediction  src={src}  n={len(data_raw)}  asof={asof}  "
        f"models={[m.name for m in CHOSEN_MODELS]}",
        flush=True,
    )
    table = append_zscore_latest(
        run_chosen_predictions(full, pred, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE),
        full,
    )
    show = table.copy()
    for c in ("nivel", "y_hat_now", "residual", "y_hat_next"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
    cols = [
        c
        for c in (
            "model",
            "structure",
            "features",
            "nivel",
            "y_hat_now",
            "residual",
            "y_hat_next",
            "equation",
            "time",
        )
        if c in show.columns
    ]
    text = (
        f"SOFR prediction  |  asof {asof}  |  k={TRAIN_SIZE}  |  "
        f"next ~{DATA_BAR_MINUTES}m\n\n{show[cols].to_string(index=False)}\n"
    )
    print(text, flush=True)
    Path("predictions_latest.csv").write_text(table.to_csv(index=False))
    table.attrs["asof"] = asof
    table.attrs["text"] = text
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--pull-history", action="store_true")
    args = p.parse_args()
    refresh = REFRESH_DATA_ON_REQUEST and not args.no_refresh

    def _tick() -> None:
        t = run_once(refresh=refresh, force_history=args.pull_history)
        send_summary_email(
            t,
            asof=t.attrs.get("asof", ""),
            subject=f"SOFR prediction {t.attrs.get('asof', '')}",
            body=t.attrs.get("text"),
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
