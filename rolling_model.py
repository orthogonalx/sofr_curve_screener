# =============================================================================
# rolling_model.py  — Walk-forward OLS for curve / fly screening
# =============================================================================
# Fit on TRAIN_SIZE bars → predict next TEST_SIZE bars → roll by STEP_SIZE.
# Produces a prediction for every out-of-sample timestamp.
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from curve_config import (
    ADD_INTERCEPT,
    MIN_TRAIN_OBS,
    REGRESSIONS,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    RegressionSpec,
)

log = logging.getLogger(__name__)


# ── OLS helpers (no sklearn dependency) ───────────────────────────────────────

def _design(X: np.ndarray, add_intercept: bool) -> np.ndarray:
    if add_intercept:
        return np.column_stack([np.ones(len(X)), X])
    return X


def ols_fit(
    X: np.ndarray, y: np.ndarray, add_intercept: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Least squares y ~ X. Returns (beta, train_metrics).
    beta[0] is intercept when add_intercept=True.
    """
    A = _design(X, add_intercept)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ beta
    resid = y - fitted
    metrics = _regression_metrics(y, fitted)
    metrics["n"] = float(len(y))
    return beta, metrics


def ols_predict(X: np.ndarray, beta: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    return _design(X, add_intercept) @ beta


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_true - y_pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    n = len(y_true)
    rmse = float(np.sqrt(sse / max(n, 1)))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1.0 - sse / sst) if sst > 1e-18 else np.nan
    # hit rate: same-direction change vs previous actual (useful for spreads)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RollingResult:
    spec: RegressionSpec
    predictions: pd.DataFrame   # index=time; cols: y_true, y_pred, residual, fold
    fold_metrics: pd.DataFrame  # one row per fold (test-block metrics + coeffs)
    overall: Dict[str, float] = field(default_factory=dict)

    @property
    def y_pred(self) -> pd.Series:
        return self.predictions["y_pred"]


# ── Walk-forward engine ───────────────────────────────────────────────────────

def _aligned_xy(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    spec: RegressionSpec,
) -> Tuple[pd.DataFrame, pd.Series]:
    if spec.target not in predicted_data.columns and spec.target not in full_data.columns:
        raise KeyError(f"Target '{spec.target}' not found in predicted_data/full_data")
    y = (
        predicted_data[spec.target]
        if spec.target in predicted_data.columns
        else full_data[spec.target]
    )
    missing_f = [f for f in spec.features if f not in full_data.columns]
    if missing_f:
        raise KeyError(f"Features missing from full_data: {missing_f}")
    X = full_data.loc[:, list(spec.features)]
    frame = pd.concat([y.rename("y"), X], axis=1).dropna()
    return frame[list(spec.features)], frame["y"]


def rolling_ols(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    spec: RegressionSpec,
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    step_size: int = STEP_SIZE,
    add_intercept: bool = ADD_INTERCEPT,
    min_train_obs: int = MIN_TRAIN_OBS,
) -> RollingResult:
    """
    Walk-forward OLS.

    Window i:
      train = [start, start + train_size)
      test  = [start + train_size, start + train_size + test_size)
      then start += step_size

    Predictions are written for every test timestamp (overlapping windows keep
    the latest prediction if STEP_SIZE < TEST_SIZE).
    """
    X_all, y_all = _aligned_xy(full_data, predicted_data, spec)
    n = len(y_all)
    if n < train_size + 1:
        raise ValueError(
            f"Need at least train_size+1={train_size + 1} clean obs; have {n}"
        )

    pred_y = pd.Series(index=y_all.index, dtype=float)
    pred_fold = pd.Series(index=y_all.index, dtype=float)
    fold_rows: List[dict] = []

    start = 0
    fold_id = 0
    while start + train_size < n:
        train_slice = slice(start, start + train_size)
        test_end = min(start + train_size + test_size, n)
        test_slice = slice(start + train_size, test_end)

        X_tr = X_all.iloc[train_slice].to_numpy(dtype=float)
        y_tr = y_all.iloc[train_slice].to_numpy(dtype=float)
        X_te = X_all.iloc[test_slice].to_numpy(dtype=float)
        y_te = y_all.iloc[test_slice].to_numpy(dtype=float)
        idx_te = y_all.index[test_slice]

        if len(y_tr) < min_train_obs or len(y_te) == 0:
            break

        beta, train_m = ols_fit(X_tr, y_tr, add_intercept=add_intercept)
        y_hat = ols_predict(X_te, beta, add_intercept=add_intercept)
        test_m = _regression_metrics(y_te, y_hat)

        pred_y.loc[idx_te] = y_hat
        pred_fold.loc[idx_te] = fold_id

        row = {
            "fold": fold_id,
            "train_start": y_all.index[train_slice][0],
            "train_end": y_all.index[train_slice][-1],
            "test_start": idx_te[0],
            "test_end": idx_te[-1],
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "train_r2": train_m["r2"],
            "train_rmse": train_m["rmse"],
            "test_r2": test_m["r2"],
            "test_rmse": test_m["rmse"],
            "test_mae": test_m["mae"],
        }
        if add_intercept:
            row["intercept"] = float(beta[0])
            for j, fname in enumerate(spec.features):
                row[f"beta_{fname}"] = float(beta[j + 1])
        else:
            for j, fname in enumerate(spec.features):
                row[f"beta_{fname}"] = float(beta[j])
        fold_rows.append(row)

        fold_id += 1
        start += step_size

    mask = pred_y.notna()
    predictions = pd.DataFrame(
        {
            "y_true": y_all[mask],
            "y_pred": pred_y[mask],
            "residual": y_all[mask] - pred_y[mask],
            "fold": pred_fold[mask].astype(int),
        }
    )
    fold_metrics = pd.DataFrame(fold_rows)
    overall = (
        _regression_metrics(
            predictions["y_true"].to_numpy(),
            predictions["y_pred"].to_numpy(),
        )
        if len(predictions)
        else {}
    )
    overall["n_pred"] = float(len(predictions))
    overall["n_folds"] = float(fold_id)

    log.info(
        "%s | folds=%d | OOS n=%d | R2=%.3f | RMSE=%.5f",
        spec.name,
        fold_id,
        len(predictions),
        overall.get("r2", np.nan),
        overall.get("rmse", np.nan),
    )
    return RollingResult(
        spec=spec,
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall=overall,
    )


def run_regressions(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = REGRESSIONS,
    **kwargs,
) -> Dict[str, RollingResult]:
    """Run every RegressionSpec; returns dict keyed by spec.name."""
    results = {}
    for spec in specs:
        results[spec.name] = rolling_ols(
            full_data, predicted_data, spec, **kwargs
        )
    return results


def predictions_panel(results: Dict[str, RollingResult]) -> pd.DataFrame:
    """
    Wide panel of y_pred columns, one per regression (hourly / 3h timestamps).
    Column names = target (or spec.name if colliding).
    """
    series = []
    for name, res in results.items():
        col = res.spec.target if len(results) == 1 else name
        s = res.predictions["y_pred"].rename(col)
        series.append(s)
    return pd.concat(series, axis=1).sort_index()
