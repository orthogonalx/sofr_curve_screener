#!/usr/bin/env python3
"""
Backtest mode — locked CHOSEN_MODELS only.

Walk-forward (fixed k=TRAIN_SIZE) over [BACKTEST_START, BACKTEST_END].

  python3.8 -u run_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from curve_config import (
    BACKTEST_END,
    BACKTEST_START,
    CHOSEN_MODELS,
    TRAIN_SIZE,
)
from rolling_model import instance_predictions, plot_residual_evolution
from screener_core import build_panel, load_panel, run_chosen_backtest


def main() -> None:
    data_raw, src = load_panel()
    _, full_data, predicted_data = build_panel(data_raw)

    print("=" * 72, flush=True)
    print("  SOFR backtest  (CHOSEN_MODELS, fixed k)", flush=True)
    print("=" * 72, flush=True)
    print(f"  src     : {src}", flush=True)
    print(f"  period  : {BACKTEST_START} → {BACKTEST_END or 'end'}", flush=True)
    print(f"  k       : {TRAIN_SIZE}", flush=True)
    print(f"  models  : {[m.name for m in CHOSEN_MODELS]}", flush=True)
    print(flush=True)

    results = run_chosen_backtest(
        full_data,
        predicted_data,
        specs=CHOSEN_MODELS,
        train_size=TRAIN_SIZE,
        start=BACKTEST_START,
        end=BACKTEST_END,
    )

    rows = []
    for name, res in results.items():
        preds = instance_predictions(res)
        preds.to_csv(Path(f"backtest_{name}.csv"))
        o = res.overall
        rows.append(
            {
                "model": name,
                "structure": res.spec.target,
                "features": "+".join(res.spec.features),
                "k": int(o.get("train_size", TRAIN_SIZE)),
                "n_folds": int(o.get("n_folds", 0)),
                "prediction_accuracy": round(
                    float(o.get("prediction_accuracy", float("nan"))), 4
                ),
            }
        )
        plot_residual_evolution(res, outfile=f"backtest_residual_{name}.png")

    summary = pd.DataFrame(rows)
    print("─" * 72, flush=True)
    print("  Backtest accuracy  (prediction_accuracy = mean(pred − actual))", flush=True)
    print("─" * 72, flush=True)
    print(flush=True)
    print(summary.to_string(index=False), flush=True)
    print(flush=True)
    Path("backtest_summary.csv").write_text(summary.to_csv(index=False))
    print("  wrote backtest_summary.csv + backtest_<model>.csv", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
