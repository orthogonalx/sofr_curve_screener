# =============================================================================
# screener_core.py  — Shared load / build / prediction / backtest helpers
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from curve_config import (
    CHOSEN_MODELS,
    DATA_PATH,
    LIVE_STORE_PATH,
    LIVE_TUNE_FEATURES,
    REGRESSION_SCREENS,
    TRAIN_SIZE,
    ZSCORE_LOOKBACKS,
    ZSCORE_MODELS,
    RegressionSpec,
)
from data_loader import generate_synthetic_sofr, load_raw
from live_store import load_live_store
from report_email import format_summary_text
from rolling_model import (
    RollingResult,
    fit_predict_next,
    rolling_ols,
    run_regressions,
)
from structures import build_datasets, resolve_feature_name
from summary import build_summary
from zscore_model import ZScoreResult, run_zscore_models


def load_panel() -> Tuple[pd.DataFrame, str]:
    """live store → DATA_PATH file → synthetic."""
    live = Path(LIVE_STORE_PATH)
    if live.exists():
        return load_live_store(live), f"live:{live}"
    path = Path(DATA_PATH)
    if path.exists():
        return load_raw(path), f"file:{path}"
    return generate_synthetic_sofr(n_bars=400, freq="15T"), "synthetic"


def build_panel(
    data_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """raw → (cleaned not returned separately) full_data, predicted_data; also normalize."""
    raw = data_raw.copy()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("data_raw must be indexed by time")
    raw = raw.apply(pd.to_numeric, errors="coerce").ffill()
    cleaned, full_data, predicted_data = build_datasets(raw)
    return cleaned, full_data, predicted_data


def run_screen(
    data_raw: pd.DataFrame,
    *,
    tune_features: bool = LIVE_TUNE_FEATURES,
    tune_window: Optional[bool] = None,  # legacy alias
    quiet: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, RollingResult], Dict[str, ZScoreResult]]:
    """Explore path: REGRESSION_SCREENS (+ optional feature tune) + z-scores."""
    if tune_window is not None:
        tune_features = tune_window
    _, full_data, predicted_data = build_panel(data_raw)
    reg_results = run_regressions(
        full_data,
        predicted_data,
        screens=REGRESSION_SCREENS,
        tune_features=tune_features,
        train_size=TRAIN_SIZE,
        quiet=quiet,
    )
    z_results = run_zscore_models(full_data, specs=ZSCORE_MODELS)
    summary = build_summary(reg_results, z_results)
    return summary, reg_results, z_results


def emit_summary(summary: pd.DataFrame, asof=None) -> str:
    """Print + return the terse table text."""
    text = format_summary_text(summary, asof=asof)
    print(text, flush=True)
    Path("summary.csv").write_text(summary.to_csv(index=False))
    return text


def run_chosen_predictions(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = CHOSEN_MODELS,
    train_size: int = TRAIN_SIZE,
) -> pd.DataFrame:
    """
    Locked-model next-bar table.
    Columns: model, structure, features, k, time, nivel, y_hat_now, residual,
             y_hat_next, equation  (+ z-score rows when included separately).
    """
    rows: List[dict] = []
    for spec in specs:
        rows.append(
            fit_predict_next(
                full_data, predicted_data, spec, train_size=train_size
            )
        )
    return pd.DataFrame(rows)


def append_zscore_latest(
    table: pd.DataFrame,
    full_data: pd.DataFrame,
) -> pd.DataFrame:
    """Append latest z-score rows (nivel + z_*) to a prediction table."""
    z_results = run_zscore_models(full_data, specs=ZSCORE_MODELS)
    z_cols = [f"z_{w}" for w in ZSCORE_LOOKBACKS]
    extra = []
    for res in z_results.values():
        panel = res.panel.dropna(subset=[f"z_{min(ZSCORE_LOOKBACKS)}"])
        if panel.empty:
            continue
        last = panel.iloc[-1]
        row = {
            "model": res.spec.param,
            "structure": res.spec.series,
            "features": "",
            "k": "",
            "time": panel.index[-1],
            "nivel": float(last["value"]),
            "y_hat_now": float("nan"),
            "residual": float(last["z_50"]) if "z_50" in last.index else float("nan"),
            "y_hat_next": float("nan"),
            "equation": "",
        }
        for c in z_cols:
            if c in last.index:
                row[c] = float(last[c]) if pd.notna(last[c]) else float("nan")
        extra.append(row)
    if not extra:
        return table
    return pd.concat([table, pd.DataFrame(extra)], ignore_index=True)


def run_chosen_backtest(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = CHOSEN_MODELS,
    train_size: int = TRAIN_SIZE,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, RollingResult]:
    """Walk-forward on CHOSEN_MODELS over [start, end] (no feature tune)."""
    full = full_data
    pred = predicted_data
    if start is not None:
        full = full.loc[full.index >= pd.Timestamp(start)]
        pred = pred.loc[pred.index >= pd.Timestamp(start)]
    if end is not None:
        full = full.loc[full.index <= pd.Timestamp(end)]
        pred = pred.loc[pred.index <= pd.Timestamp(end)]

    resolved: List[RegressionSpec] = [
        RegressionSpec(
            name=s.name,
            target=s.target,
            features=tuple(resolve_feature_name(f) for f in s.features),
        )
        for s in specs
    ]
    return run_regressions(
        full,
        pred,
        specs=resolved,
        train_size=train_size,
        tune_features=False,
        quiet=True,
    )
