# =============================================================================
# screener_core.py  — Build models + terse summary (shared by one-shot / live)
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from curve_config import LIVE_TUNE_WINDOW, REGRESSIONS, TRAIN_SIZE, ZSCORE_MODELS
from report_email import format_summary_text
from rolling_model import RollingResult, run_regressions
from structures import build_datasets
from summary import build_summary
from zscore_model import ZScoreResult, run_zscore_models


def run_screen(
    data_raw: pd.DataFrame,
    *,
    tune_window: bool = LIVE_TUNE_WINDOW,
    quiet: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, RollingResult], Dict[str, ZScoreResult]]:
    """Structures → regressions + z-scores → summary table."""
    raw = data_raw.copy()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("data_raw must be indexed by time")
    raw = raw.apply(pd.to_numeric, errors="coerce").ffill()

    _, full_data, predicted_data = build_datasets(raw)
    reg_results = run_regressions(
        full_data,
        predicted_data,
        specs=REGRESSIONS,
        tune_window=tune_window,
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
