# =============================================================================
# rolling_model.py  — Walk-forward OLS for curve / fly screening
# =============================================================================
# Fit on TRAIN_SIZE bars → predict next TEST_SIZE bars → roll by STEP_SIZE.
# Produces a prediction for every out-of-sample timestamp.
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from curve_config import (
    ADD_INTERCEPT,
    MIN_TRAIN_OBS,
    REGRESSION_SCREENS,
    REGRESSIONS,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    RegressionScreen,
    RegressionSpec,
)
from structures import resolve_feature_name

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

    overall: Dict[str, float] = {}
    if len(predictions):
        # Mean absolute *relative* error across folds:
        #   prediction_accuracy = mean( |pred − actual| / |actual| )
        # Floor |actual| so near-zero flies/curves don't explode.
        # Lower / closer to 0 is better (always ≥ 0).
        y_true = predictions["y_true"].to_numpy(dtype=float)
        y_pred = predictions["y_pred"].to_numpy(dtype=float)
        err = np.abs(y_pred - y_true)
        abs_y = np.abs(y_true)
        denom = np.maximum(abs_y, 1e-6)
        overall["mean_error"] = float(np.mean(err))       # mean(|pred − actual|)
        overall["mean_value"] = float(np.mean(abs_y))     # mean(|actual|)
        overall["prediction_accuracy"] = float(np.mean(err / denom))
        # Keep classic pooled metrics for optional diagnostics / plots
        overall.update(
            _regression_metrics(y_true, y_pred)
        )
    overall["n_pred"] = float(len(predictions))
    overall["n_folds"] = float(fold_id)
    overall["train_size"] = float(train_size)

    log.info(
        "%s | train=%d | folds=%d | prediction_accuracy=%.5f",
        spec.name,
        train_size,
        fold_id,
        overall.get("prediction_accuracy", np.nan),
    )
    return RollingResult(
        spec=spec,
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall=overall,
    )


def _spec_from_features(
    screen: RegressionScreen,
    features: Sequence[str],
) -> RegressionSpec:
    resolved = tuple(resolve_feature_name(f) for f in features)
    return RegressionSpec(name=screen.name, target=screen.target, features=resolved)


def _features_label(features: Sequence[str]) -> str:
    return "+".join(features) if features else "(none)"


def select_feature_set(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    screen: RegressionScreen,
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    step_size: int = STEP_SIZE,
    quiet: bool = False,
    **kwargs,
) -> Tuple[Tuple[str, ...], pd.DataFrame, RollingResult]:
    """
    Compare walk-forward prediction accuracy across candidate feature sets.

    prediction_accuracy = mean( |y_pred − y_true| / |y_true| ) over all OOS folds
    (MAPE-style; |y_true| floored at 1e-6). Lowest / closest to 0 wins.

    Returns
    -------
    best_features, comparison_table, best_RollingResult
    """
    rows: List[dict] = []
    results: Dict[str, RollingResult] = {}
    label_to_feats: Dict[str, Tuple[str, ...]] = {}

    for raw_feats in screen.feature_sets:
        spec = _spec_from_features(screen, raw_feats)
        label = _features_label(spec.features)
        try:
            res = rolling_ols(
                full_data,
                predicted_data,
                spec,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
                **kwargs,
            )
        except (KeyError, ValueError) as exc:
            if not quiet:
                print(f"  skip features={label}: {exc}", flush=True)
            continue
        results[label] = res
        label_to_feats[label] = spec.features
        o = res.overall
        rows.append(
            {
                "features": label,
                "n_folds": int(o.get("n_folds", 0)),
                "mean_error": o.get("mean_error", np.nan),
                "mean_value": o.get("mean_value", np.nan),
                "prediction_accuracy": o.get("prediction_accuracy", np.nan),
            }
        )

    if not rows:
        raise ValueError(
            f"No valid feature sets for screen={screen.name} target={screen.target}; "
            f"tried {list(screen.feature_sets)}"
        )

    comparison = pd.DataFrame(rows).reset_index(drop=True)
    ranked = comparison.assign(
        _abs_acc=comparison["prediction_accuracy"].abs(),
        _n_feat=comparison["features"].str.count(r"\+") + 1,
    ).sort_values(
        by=["_abs_acc", "_n_feat"],
        ascending=[True, False],
        na_position="last",
    )
    best_label = str(ranked.iloc[0]["features"])
    return label_to_feats[best_label], comparison, results[best_label]


# Back-compat alias
def select_train_window(*args, **kwargs):
    raise RuntimeError(
        "select_train_window is removed — k is fixed (TRAIN_SIZE). "
        "Use select_feature_set / run_regressions(tune_features=True)."
    )


def plot_residual_evolution(
    result: RollingResult,
    outfile: str = "residual_evolution.png",
) -> Optional[str]:
    """
    One chart: OOS residual path. Saves PNG and returns path.
    If matplotlib is not installed, skips and returns None.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            f"  skip plot ({result.spec.name}): matplotlib not installed",
            flush=True,
        )
        return None

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


def equation_from_beta(
    target: str,
    features: Sequence[str],
    beta: np.ndarray,
    add_intercept: bool = True,
) -> str:
    """Format y = a + b1*x1 + … from a fitted beta vector."""
    if add_intercept:
        intercept = float(beta[0])
        coefs = beta[1:]
    else:
        intercept = 0.0
        coefs = beta
    pieces = [f"{intercept:.4f}"]
    for feat, b in zip(features, coefs):
        b = float(b)
        sign = "+" if b >= 0 else "-"
        pieces.append(f"{sign} {abs(b):.4f}*{feat}")
    return f"{target} = " + " ".join(pieces)


def fit_predict_next(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    spec: RegressionSpec,
    train_size: int = TRAIN_SIZE,
    add_intercept: bool = ADD_INTERCEPT,
) -> dict:
    """
    One-shot prediction for a locked model.

    - y_hat_now: train iloc[n-k-1 : n-1], predict last bar (realized OOS)
    - y_hat_next: train iloc[n-k : n], forecast with X of last bar (next bar ~15m)
    """
    resolved = RegressionSpec(
        name=spec.name,
        target=spec.target,
        features=tuple(resolve_feature_name(f) for f in spec.features),
    )
    X_all, y_all = _aligned_xy(full_data, predicted_data, resolved)
    n = len(y_all)
    k = int(train_size)
    if n < k + 1:
        raise ValueError(
            f"{resolved.name}: need at least k+1={k + 1} clean rows; have {n}"
        )

    # Latest realized OOS: train [n-k-1 : n-1] → predict [n-1]
    tr0, tr1 = n - k - 1, n - 1
    X_tr = X_all.iloc[tr0:tr1].to_numpy(dtype=float)
    y_tr = y_all.iloc[tr0:tr1].to_numpy(dtype=float)
    X_te = X_all.iloc[tr1 : tr1 + 1].to_numpy(dtype=float)
    beta_now, _ = ols_fit(X_tr, y_tr, add_intercept=add_intercept)
    y_hat_now = float(ols_predict(X_te, beta_now, add_intercept=add_intercept)[0])
    y_true_now = float(y_all.iloc[tr1])
    t_now = y_all.index[tr1]

    # Next-bar forecast: train [n-k : n] → predict with X[-1]
    X_tr2 = X_all.iloc[n - k : n].to_numpy(dtype=float)
    y_tr2 = y_all.iloc[n - k : n].to_numpy(dtype=float)
    X_last = X_all.iloc[n - 1 : n].to_numpy(dtype=float)
    beta_next, _ = ols_fit(X_tr2, y_tr2, add_intercept=add_intercept)
    y_hat_next = float(ols_predict(X_last, beta_next, add_intercept=add_intercept)[0])

    row = {
        "model": resolved.name,
        "structure": resolved.target,
        "features": "+".join(resolved.features),
        "k": k,
        "time": t_now,
        "nivel": y_true_now,
        "y_hat_now": y_hat_now,
        "residual": y_true_now - y_hat_now,
        "y_hat_next": y_hat_next,
        "equation": equation_from_beta(
            resolved.target, resolved.features, beta_next, add_intercept=add_intercept
        ),
    }
    if add_intercept:
        row["intercept"] = float(beta_next[0])
        for j, fname in enumerate(resolved.features):
            row[f"beta_{fname}"] = float(beta_next[j + 1])
    else:
        for j, fname in enumerate(resolved.features):
            row[f"beta_{fname}"] = float(beta_next[j])
    return row


def run_regressions(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    screens: Sequence[RegressionScreen] = REGRESSION_SCREENS,
    specs: Optional[Sequence[RegressionSpec]] = None,
    train_size: int = TRAIN_SIZE,
    tune_features: bool = True,
    tune_window: Optional[bool] = None,  # legacy alias → tune_features
    quiet: bool = False,
    **kwargs,
) -> Dict[str, RollingResult]:
    """
    Run every RegressionScreen (or explicit specs).

    If tune_features=True (default), compare screen.feature_sets at fixed k=train_size
    and keep the set with prediction_accuracy closest to 0.
    """
    if tune_window is not None:
        tune_features = tune_window

    results: Dict[str, RollingResult] = {}

    # Legacy path: flat RegressionSpec list, no feature search
    if specs is not None:
        for spec in specs:
            results[spec.name] = rolling_ols(
                full_data,
                predicted_data,
                RegressionSpec(
                    name=spec.name,
                    target=spec.target,
                    features=tuple(resolve_feature_name(f) for f in spec.features),
                ),
                train_size=train_size,
                **kwargs,
            )
        return results

    for screen in screens:
        if tune_features and len(screen.feature_sets) > 0:
            if not quiet:
                print(flush=True)
                print("─" * 72, flush=True)
                print(
                    f"  [1]  Accuracy metrics  —  {screen.name}  "
                    f"(target={screen.target}, k={train_size})",
                    flush=True,
                )
                print(
                    "  Per fold: rel = |pred − actual| / |actual| · "
                    "prediction_accuracy = mean(rel) · "
                    "pick features closest to 0",
                    flush=True,
                )
                print("─" * 72, flush=True)
                print(flush=True)
            best_feats, comparison, best_res = select_feature_set(
                full_data,
                predicted_data,
                screen,
                train_size=train_size,
                quiet=quiet,
                **kwargs,
            )
            if not quiet:
                show = comparison.copy()
                for c in ("mean_error", "mean_value", "prediction_accuracy"):
                    if c in show.columns:
                        show[c] = show[c].round(4)
                print(show.to_string(index=False), flush=True)
                print(flush=True)
                print(
                    f"  →  chosen features = {_features_label(best_feats)}    "
                    f"prediction_accuracy = "
                    f"{best_res.overall.get('prediction_accuracy', float('nan')):.4f}  "
                    f"(mean_error={best_res.overall.get('mean_error', float('nan')):.4f}, "
                    f"mean_value={best_res.overall.get('mean_value', float('nan')):.4f})",
                    flush=True,
                )
                print(flush=True)
            results[screen.name] = best_res
        else:
            # No search: use first feature set
            if not screen.feature_sets:
                raise ValueError(f"Screen {screen.name} has empty feature_sets")
            spec = _spec_from_features(screen, screen.feature_sets[0])
            results[screen.name] = rolling_ols(
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
