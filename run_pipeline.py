#!/usr/bin/env python3
"""Explore: compare REGRESSION_SCREENS feature sets → lock winners in CHOSEN_MODELS."""

from pathlib import Path

import numpy as np
import pandas as pd

from curve_config import (
    REGRESSION_SCREENS,
    TRAIN_SIZE,
    ZSCORE_LOOKBACKS,
    ZSCORE_MODELS,
)
from lib.data import build_panel, load_panel
from lib.models import (
    build_summary,
    equation_from_beta,
    format_summary_text,
    run_regressions,
    run_zscore_models,
)


def _eq(spec, row) -> str:
    feats = list(spec.features)
    beta = np.array(
        [float(row.get("intercept", 0.0))] + [float(row.get(f"beta_{f}", 0.0)) for f in feats]
    )
    return equation_from_beta(spec.target, feats, beta)


def main() -> None:
    data_raw, src = load_panel()
    _, full, pred = build_panel(data_raw)
    asof = str(data_raw.index[-1])
    print(
        f"explore  src={src}  n={len(data_raw)}  asof={asof}  k={TRAIN_SIZE}  "
        f"screens={[s.name for s in REGRESSION_SCREENS]}",
        flush=True,
    )

    results = run_regressions(
        full, pred, screens=REGRESSION_SCREENS, train_size=TRAIN_SIZE, quiet=False
    )

    rows = []
    for screen in REGRESSION_SCREENS:
        if screen.name not in results:
            continue
        res = results[screen.name]
        preds = res.predictions
        preds.to_csv(Path(f"predictions_{screen.name}.csv"))
        if preds.empty:
            continue
        last = preds.iloc[-1]
        o = res.overall
        rows.append(
            {
                "model": screen.name,
                "structure": screen.target,
                "features": "+".join(res.spec.features),
                "k": int(o.get("train_size", TRAIN_SIZE)),
                "mean_error": round(float(o.get("mean_error", float("nan"))), 4),
                "mean_value": round(float(o.get("mean_value", float("nan"))), 4),
                "prediction_accuracy": round(
                    float(o.get("prediction_accuracy", float("nan"))), 4
                ),
                "time": preds.index[-1],
                "y_true": round(float(last["y_true"]), 4),
                "y_pred": round(float(last["y_pred"]), 4),
                "residual": round(float(last["residual"]), 4),
                "equation": _eq(res.spec, last),
            }
        )
    print("\n[last regression bar]", flush=True)
    print(pd.DataFrame(rows).to_string(index=False) if rows else "  (none)", flush=True)

    z_results = run_zscore_models(full, specs=ZSCORE_MODELS)
    z_cols = [f"z_{w}" for w in ZSCORE_LOOKBACKS]
    z_rows = []
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
        z_rows.append(row)
    print("\n[last z-score bar]", flush=True)
    print(pd.DataFrame(z_rows).to_string(index=False) if z_rows else "  (none)", flush=True)

    summary = build_summary(results, z_results)
    Path("summary.csv").write_text(summary.to_csv(index=False))
    print("\n" + format_summary_text(summary, asof=asof), flush=True)
    print("Done. Edit CHOSEN_MODELS → run_backtest / run_prediction", flush=True)


if __name__ == "__main__":
    main()
