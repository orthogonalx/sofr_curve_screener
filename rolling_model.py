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
    TRAIN_WINDOWS,
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

    # Attach the fitted model (R², intercept, betas) used for each prediction
    model_cols = []
    if "train_r2" in fold_metrics.columns:
        model_cols.append("train_r2")
    model_cols.extend(
        c for c in fold_metrics.columns if c == "intercept" or c.startswith("beta_")
    )
    if model_cols and not fold_metrics.empty:
        model_frame = fold_metrics.set_index("fold")[model_cols].rename(
            columns={"train_r2": "r2"}
        )
        predictions = predictions.join(model_frame, on="fold")

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
    overall["train_size"] = float(train_size)

    log.info(
        "%s | train=%d | folds=%d | OOS n=%d | R2=%.3f | RMSE=%.5f",
        spec.name,
        train_size,
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


def select_train_window(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    spec: RegressionSpec,
    train_windows: Sequence[int] = TRAIN_WINDOWS,
    test_size: int = TEST_SIZE,
    step_size: int = STEP_SIZE,
    **kwargs,
) -> Tuple[int, pd.DataFrame, RollingResult]:
    """
    Compare walk-forward OOS accuracy across candidate train windows.

    Selection rule: lowest overall OOS RMSE (ties → higher R², then larger window).

    Returns
    -------
    best_train_size, comparison_table, best_RollingResult
    """
    X_all, y_all = _aligned_xy(full_data, predicted_data, spec)
    n = len(y_all)

    rows: List[dict] = []
    results: Dict[int, RollingResult] = {}
    for tw in train_windows:
        if tw < MIN_TRAIN_OBS or tw + 1 > n:
            print(f"  skip train={tw}: need train+1 <= n={n}", flush=True)
            continue
        res = rolling_ols(
            full_data,
            predicted_data,
            spec,
            train_size=tw,
            test_size=test_size,
            step_size=step_size,
            **kwargs,
        )
        results[tw] = res
        o = res.overall
        rows.append(
            {
                "train_size": tw,
                "n_folds": int(o.get("n_folds", 0)),
                "n_pred": int(o.get("n_pred", 0)),
                "r2": o.get("r2", np.nan),
                "rmse": o.get("rmse", np.nan),
                "mae": o.get("mae", np.nan),
            }
        )

    if not rows:
        raise ValueError(f"No valid train windows for n={n}; tried {list(train_windows)}")

    comparison = pd.DataFrame(rows).sort_values("train_size").reset_index(drop=True)
    # best = min RMSE; if NaN RMSE, push to the end
    ranked = comparison.sort_values(
        by=["rmse", "r2", "train_size"],
        ascending=[True, False, False],
        na_position="last",
    )
    best_tw = int(ranked.iloc[0]["train_size"])
    return best_tw, comparison, results[best_tw]


def plot_residual_evolution(
    result: RollingResult,
    outfile: str = "residual_evolution.png",
) -> str:
    """One chart: OOS residual path for the chosen model. Saves PNG, returns path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = result.predictions
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.plot(preds.index, preds["residual"], color="C3", lw=1.0)
    ax.set_title(
        f"Residual evolution — {result.spec.name}  "
        f"(train={int(result.overall.get('train_size', TRAIN_SIZE))})"
    )
    ax.set_ylabel("residual (actual − pred)")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def instance_predictions(result: RollingResult) -> pd.DataFrame:
    """
    Main deliverable: one row per OOS timestamp.
    Columns: y_true, y_pred, residual, fold, r2, intercept, beta_<feature>, …
    """
    out = result.predictions.copy()
    out.index.name = "time"
    return out


def run_regressions(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = REGRESSIONS,
    train_size: int = TRAIN_SIZE,
    tune_window: bool = True,
    train_windows: Sequence[int] = TRAIN_WINDOWS,
    **kwargs,
) -> Dict[str, RollingResult]:
    """
    Run every RegressionSpec.

    If tune_window=True, pick the best TRAIN_WINDOWS entry per spec (min OOS RMSE)
    and refit is already done inside select_train_window.
    """
    results: Dict[str, RollingResult] = {}
    for spec in specs:
        if tune_window:
            print(f"\nWindow search — {spec.name}", flush=True)
            best_tw, comparison, best_res = select_train_window(
                full_data,
                predicted_data,
                spec,
                train_windows=train_windows,
                **kwargs,
            )
            print(comparison.to_string(index=False), flush=True)
            print(
                f"  → chosen train_size={best_tw}  "
                f"OOS R2={best_res.overall.get('r2', float('nan')):.4f}  "
                f"RMSE={best_res.overall.get('rmse', float('nan')):.5f}",
                flush=True,
            )
            results[spec.name] = best_res
        else:
            results[spec.name] = rolling_ols(
                full_data,
                predicted_data,
                spec,
                train_size=train_size,
                **kwargs,
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
