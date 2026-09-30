# =============================================================================
# summary.py  — Latest-observation screen table across all models
# =============================================================================
# Columns:
#   model | structure | nivel | predicted | residual/zscore | equation | time
# - regressions: predicted = y_pred; residual/zscore = residual; equation filled
# - z-scores:    predicted blank; residual/zscore = z_50; equation blank
# =============================================================================

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from rolling_model import RollingResult
from zscore_model import ZScoreResult


def format_equation(res: RollingResult, row: pd.Series) -> str:
    """Build e.g. 5s7s10s = -0.0914 + -0.0906*2s5s10s from the fold that predicted row."""
    target = res.spec.target
    intercept = float(row.get("intercept", 0.0)) if "intercept" in row.index else 0.0
    pieces = [f"{intercept:.4f}"]
    for feat in res.spec.features:
        key = f"beta_{feat}"
        if key not in row.index or pd.isna(row[key]):
            continue
        b = float(row[key])
        sign = "+" if b >= 0 else "-"
        pieces.append(f"{sign} {abs(b):.4f}*{feat}")
    return f"{target} = " + " ".join(pieces)


def build_summary(
    reg_results: Dict[str, RollingResult],
    z_results: Dict[str, ZScoreResult],
) -> pd.DataFrame:
    """
    One row per model at the latest available timestamp.

    nivel             — observed level (y_true / fly value)
    predicted         — y_pred for regressions; blank for z-scores
    residual/zscore   — residual (regs) or z_50 (z-scores)
    equation          — fitted line used for last pred (regs only)
    """
    rows: List[dict] = []
    col_rz = "residual/zscore"

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
                col_rz: float(last["residual"]),
                "equation": format_equation(res, last),
            }
        )

    for res in z_results.values():
        panel = res.panel.dropna(subset=["fair"])
        if panel.empty:
            continue
        last = panel.iloc[-1]
        z50 = last["z_50"] if "z_50" in last.index else float("nan")
        rows.append(
            {
                "time": panel.index[-1],
                "structure": res.spec.series,
                "nivel": float(last["value"]),  # fly level
                "predicted": float("nan"),      # not used for z-score models
                "model": res.spec.param,
                col_rz: float(z50) if pd.notna(z50) else float("nan"),
                "equation": "",
            }
        )

    cols = [
        "model",
        "structure",
        "nivel",
        "predicted",
        col_rz,
        "equation",
        "time",
    ]
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
