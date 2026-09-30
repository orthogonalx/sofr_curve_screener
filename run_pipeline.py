#!/usr/bin/env python3
"""
Explore mode — compare REGRESSION_SCREENS feature_sets, pick winners, inspect.

Does NOT email. After you choose, lock them in curve_config.CHOSEN_MODELS and use:
  run_prediction.py  /  run_backtest.py
"""

from pathlib import Path

import pandas as pd

from curve_config import (
    REGRESSION_SCREENS,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    ZSCORE_LOOKBACKS,
    ZSCORE_MODELS,
)
from report_email import format_summary_text
from rolling_model import instance_predictions, run_regressions
from screener_core import load_panel
from structures import build_datasets
from summary import build_summary, format_equation
from zscore_model import run_zscore_models

_W = 72


def _rule(char: str = "─") -> None:
    print(char * _W, flush=True)


def _blank() -> None:
    print(flush=True)


def _section(title: str, note: str = "") -> None:
    _blank()
    _blank()
    _rule("═")
    print(f"  {title}", flush=True)
    if note:
        print(f"  {note}", flush=True)
    _rule("═")
    _blank()


def _explain_folds(n: int, train_size: int) -> None:
    """How 1-step rolling folds map to row indices (0-based iloc on clean series)."""
    _section(
        "Folds  (test=1, step=1)",
        f"Clean series n={n} rows (iloc 0 … {n - 1}). "
        f"Each fold trains on {train_size} rows → predicts the next 1.",
    )
    print(f"    fold 0 :  train [{0} : {train_size}]     →  predict [{train_size}]", flush=True)
    print(
        f"    fold 1 :  train [1 : {train_size + 1}]     →  predict [{train_size + 1}]",
        flush=True,
    )
    print(
        f"    fold k :  train [k : k+{train_size}]   →  predict [k+{train_size}]",
        flush=True,
    )
    last_k = n - train_size - 1
    if last_k >= 0:
        _blank()
        print(
            f"    last fold {last_k} :  train [{last_k} : {last_k + train_size}]  "
            f"→  predict [{last_k + train_size}]  (= last row)",
            flush=True,
        )
    _blank()


def main() -> None:
    _blank()
    _rule("═")
    print("  SOFR curve screener  —  EXPLORE (feature search)", flush=True)
    _rule("═")
    _blank()

    data_raw, src = load_panel()
    print(f"  src        : {src}", flush=True)
    print(f"  n          : {len(data_raw)}", flush=True)
    print(f"  asof       : {data_raw.index[-1]}", flush=True)

    _, full_data, predicted_data = build_datasets(data_raw)
    _blank()
    print(f"  full_data  : {full_data.shape}", flush=True)
    print(f"  predicted  : {list(predicted_data.columns)}", flush=True)
    print(
        f"  walk-fwd   : k={TRAIN_SIZE} (fixed)  test={TEST_SIZE}  step={STEP_SIZE}",
        flush=True,
    )
    print(
        f"  screens    : {[s.name for s in REGRESSION_SCREENS]}",
        flush=True,
    )

    _explain_folds(n=len(full_data), train_size=TRAIN_SIZE)

    results = run_regressions(
        full_data,
        predicted_data,
        screens=REGRESSION_SCREENS,
        train_size=TRAIN_SIZE,
        tune_features=True,
        quiet=False,
    )

    last_rows = []
    for screen in REGRESSION_SCREENS:
        res = results[screen.name]
        preds = instance_predictions(res)
        preds.to_csv(Path(f"predictions_{screen.name}.csv"))
        if preds.empty:
            continue
        last = preds.iloc[-1]
        last_rows.append(
            {
                "model": screen.name,
                "structure": screen.target,
                "features": "+".join(res.spec.features),
                "k": int(res.overall.get("train_size", TRAIN_SIZE)),
                "prediction_accuracy": round(
                    float(res.overall.get("prediction_accuracy", float("nan"))), 4
                ),
                "time": preds.index[-1],
                "y_true": round(float(last["y_true"]), 4),
                "y_pred": round(float(last["y_pred"]), 4),
                "residual": round(float(last["residual"]), 4),
                "equation": format_equation(res, last),
            }
        )

    _section(
        "[2]  Last prediction results  —  regressions",
        "Latest bar only · copy winners into curve_config.CHOSEN_MODELS",
    )
    if last_rows:
        print(pd.DataFrame(last_rows).to_string(index=False), flush=True)
    else:
        print("  (no regression predictions)", flush=True)
    _blank()

    z_results = run_zscore_models(full_data, specs=ZSCORE_MODELS)
    z_cols = [f"z_{w}" for w in ZSCORE_LOOKBACKS]

    z_last_rows = []
    for spec in ZSCORE_MODELS:
        res = z_results[spec.name]
        panel = res.panel.dropna(subset=[f"z_{min(ZSCORE_LOOKBACKS)}"])
        panel.to_csv(Path(f"zscore_{spec.name.replace(':', '_')}.csv"))
        if panel.empty:
            continue
        last = panel.iloc[-1]
        row = {
            "model": spec.param,
            "structure": spec.series,
            "time": panel.index[-1],
            "nivel": round(float(last["value"]), 4),
        }
        for c in z_cols:
            row[c] = (
                round(float(last[c]), 4)
                if c in last.index and pd.notna(last[c])
                else float("nan")
            )
        z_last_rows.append(row)

    _section(
        "[2]  Last prediction results  —  z-scores",
        f"Latest bar only · lookbacks={ZSCORE_LOOKBACKS} · nivel=fly · z_*=(nivel−μ)/σ",
    )
    if z_last_rows:
        print(pd.DataFrame(z_last_rows).to_string(index=False), flush=True)
    else:
        print("  (no z-score predictions)", flush=True)
    _blank()

    summary = build_summary(results, z_results)
    asof = str(data_raw.index[-1])
    Path("summary.csv").write_text(summary.to_csv(index=False))

    _section(
        "[3]  Final results table",
        "Explore snapshot · lock chosen features in CHOSEN_MODELS for prediction/backtest",
    )
    print(format_summary_text(summary, asof=asof), flush=True)
    _blank()
    _rule("═")
    print("  Done.  Next: edit CHOSEN_MODELS → run_prediction.py / run_backtest.py", flush=True)
    _rule("═")
    _blank()


if __name__ == "__main__":
    main()
