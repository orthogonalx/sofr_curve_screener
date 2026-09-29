# =============================================================================
# summary.py  — Latest-observation screen table across all models
# =============================================================================
# Columns: nivel | predicted | model | residual | structure | time
# (Multi-lookback z-scores stay in the per-model Last-5 print, not here.)
# =============================================================================

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from rolling_model import RollingResult
from zscore_model import ZScoreResult


def build_summary(
    reg_results: Dict[str, RollingResult],
    z_results: Dict[str, ZScoreResult],
) -> pd.DataFrame:
    """
    One row per model at the latest available timestamp.

    nivel      — observed level
    predicted  — model fair value (y_pred / rolling mean)
    model      — id (x7, z6, …)
    residual   — actual − predicted (regressions); NaN for z-scores
    """
    rows: List[dict] = []

    for res in reg_results.values():
        preds = res.predictions.dropna(subset=["y_pred"])
        if preds.empty:
            continue
        last = preds.iloc[-1]
        rows.append(
            {
                "time": preds.index[-1],
                "structure": res.spec.target,
                "nivel": float(last["y_true"]),
                "predicted": float(last["y_pred"]),
                "model": res.spec.name,
                "residual": float(last["residual"]),
            }
        )

    for res in z_results.values():
        panel = res.panel.dropna(subset=["fair"])
        if panel.empty:
            continue
        last = panel.iloc[-1]
        # residual column carries z_50 when available (compact single table)
        z50 = last["z_50"] if "z_50" in last.index else float("nan")
        rows.append(
            {
                "time": panel.index[-1],
                "structure": res.spec.series,
                "nivel": float(last["value"]),
                "predicted": float(last["fair"]),
                "model": res.spec.param,
                "residual": float(z50) if pd.notna(z50) else float("nan"),
            }
        )

    cols = ["nivel", "predicted", "model", "residual", "structure", "time"]
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    reg_order = list(reg_results.keys())
    z_order = list(z_results.keys())
    out["_ord"] = out.apply(
        lambda r: (
            reg_order.index(r["model"])
            if r["model"] in reg_order
            else 1000 + z_order.index(f"{r['model']}:{r['structure']}")
            if f"{r['model']}:{r['structure']}" in z_order
            else 2000
        ),
        axis=1,
    )
    return (
        out.sort_values("_ord")
        .drop(columns="_ord")
        .reset_index(drop=True)[cols]
    )
