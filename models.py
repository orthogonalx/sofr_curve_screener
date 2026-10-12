# =============================================================================
# models.py — Rolling OLS, z-scores, residual backtest, summary + email
# =============================================================================
# Merges: rolling_model, zscore_model, backtest, summary, report_email and the
# model helpers of screener_core (run_chosen_predictions, append_zscore_latest,
# run_chosen_backtest).
# =============================================================================

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from curve_config import (
    ADD_INTERCEPT,
    BACKTEST_END,
    BACKTEST_POSITION_MODE,
    BACKTEST_RESIDUAL_THR_BP,
    BACKTEST_START,
    BACKTEST_Z_LOOKBACK,
    BACKTEST_ZSCORE_THR,
    CHOSEN_MODELS,
    EMAIL_ENABLED,
    MIN_TRAIN_OBS,
    REGRESSION_SCREENS,
    REPORT_FROM,
    REPORT_TO,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    ZSCORE_LOOKBACKS,
    ZSCORE_MODELS,
    ZSCORE_WINDOW,
    RegressionScreen,
    RegressionSpec,
    ZScoreSpec,
)
from data import resolve_feature_name

log = logging.getLogger(__name__)

COL_RZ = "residual/zscore"


# ── OLS helpers (numpy only) ──────────────────────────────────────────────────

def _design(X: np.ndarray, ic: bool) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) if ic else X


def _fit(X: np.ndarray, y: np.ndarray, ic: bool = True) -> np.ndarray:
    return np.linalg.lstsq(_design(X, ic), y, rcond=None)[0]


def _predict(X: np.ndarray, beta: np.ndarray, ic: bool = True) -> np.ndarray:
    return _design(X, ic) @ beta


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(sse / max(len(y), 1))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - sse / sst) if sst > 1e-18 else np.nan,
    }


def _coef_dict(features: Sequence[str], beta: np.ndarray, ic: bool) -> dict:
    out = {"intercept": float(beta[0])} if ic else {}
    off = 1 if ic else 0
    for j, f in enumerate(features):
        out[f"beta_{f}"] = float(beta[j + off])
    return out


def equation_from_beta(
    target: str, features: Sequence[str], beta: np.ndarray, add_intercept: bool = True
) -> str:
    """Format y = a + b1*x1 + … from a beta vector (beta[0] = intercept if used)."""
    coefs = beta[1:] if add_intercept else beta
    pieces = ["%.4f" % (float(beta[0]) if add_intercept else 0.0)]
    for f, b in zip(features, coefs):
        pieces.append("%s %.4f*%s" % ("+" if b >= 0 else "-", abs(float(b)), f))
    return f"{target} = " + " ".join(pieces)


def _equation_from_row(spec: RegressionSpec, row: pd.Series) -> str:
    """Equation of the fold that produced `row` (columns intercept / beta_<f>)."""
    feats = [f for f in spec.features if pd.notna(row.get(f"beta_{f}"))]
    beta = np.array([float(row.get("intercept", 0.0))] + [float(row[f"beta_{f}"]) for f in feats])
    return equation_from_beta(spec.target, feats, beta, True)


def _resolved(spec: RegressionSpec) -> RegressionSpec:
    return RegressionSpec(
        name=spec.name,
        target=spec.target,
        features=tuple(resolve_feature_name(f) for f in spec.features),
    )


def _label(features: Sequence[str]) -> str:
    return "+".join(features) if features else "(none)"


# ── Rolling OLS ───────────────────────────────────────────────────────────────

@dataclass
class RollingResult:
    spec: RegressionSpec
    predictions: pd.DataFrame   # index=time; y_true, y_pred, residual, fold, r2, intercept, beta_*
    fold_metrics: pd.DataFrame  # one row per fold
    overall: Dict[str, float] = field(default_factory=dict)

    @property
    def y_pred(self) -> pd.Series:
        return self.predictions["y_pred"]


def _aligned_xy(
    full_data: pd.DataFrame, predicted_data: pd.DataFrame, spec: RegressionSpec
) -> Tuple[pd.DataFrame, pd.Series]:
    src = predicted_data if spec.target in predicted_data.columns else full_data
    if spec.target not in src.columns:
        raise KeyError(f"Target '{spec.target}' not found in predicted_data/full_data")
    missing = [f for f in spec.features if f not in full_data.columns]
    if missing:
        raise KeyError(f"Features missing from full_data: {missing}")
    feats = list(spec.features)
    frame = pd.concat([src[spec.target].rename("y"), full_data[feats]], axis=1).dropna()
    return frame[feats], frame["y"]


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
    Walk-forward OLS: train [s, s+train) → predict [s+train, s+train+test) → s += step.
    Overlapping test blocks keep the latest prediction.
    """
    X_all, y_all = _aligned_xy(full_data, predicted_data, spec)
    n = len(y_all)
    if n < train_size + 1:
        raise ValueError(f"Need at least train_size+1={train_size + 1} clean obs; have {n}")

    pred_y = pd.Series(index=y_all.index, dtype=float)
    pred_fold = pd.Series(index=y_all.index, dtype=float)
    rows: List[dict] = []
    start = fold = 0
    while start + train_size < n:
        tr = slice(start, start + train_size)
        te = slice(start + train_size, min(start + train_size + test_size, n))
        idx_te = y_all.index[te]
        if train_size < min_train_obs or len(idx_te) == 0:
            break
        X_tr, y_tr = X_all.iloc[tr].to_numpy(float), y_all.iloc[tr].to_numpy(float)
        y_te = y_all.iloc[te].to_numpy(float)

        beta = _fit(X_tr, y_tr, add_intercept)
        y_hat = _predict(X_all.iloc[te].to_numpy(float), beta, add_intercept)
        train_m, test_m = _metrics(y_tr, _predict(X_tr, beta, add_intercept)), _metrics(y_te, y_hat)

        pred_y.loc[idx_te] = y_hat
        pred_fold.loc[idx_te] = fold
        row = {
            "fold": fold,
            "train_start": y_all.index[tr][0],
            "train_end": y_all.index[tr][-1],
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
        row.update(_coef_dict(spec.features, beta, add_intercept))
        rows.append(row)
        fold += 1
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
    fold_metrics = pd.DataFrame(rows)
    if not fold_metrics.empty:
        model_cols = ["train_r2"] + [
            c for c in fold_metrics.columns if c == "intercept" or c.startswith("beta_")
        ]
        predictions = predictions.join(
            fold_metrics.set_index("fold")[model_cols].rename(columns={"train_r2": "r2"}),
            on="fold",
        )

    overall: Dict[str, float] = {}
    if len(predictions):
        # MAPE-style: mean(|pred − actual| / max(|actual|, 1e-6)); closer to 0 is better
        yt = predictions["y_true"].to_numpy(float)
        yp = predictions["y_pred"].to_numpy(float)
        err, abs_y = np.abs(yp - yt), np.abs(yt)
        overall["mean_error"] = float(err.mean())
        overall["mean_value"] = float(abs_y.mean())
        overall["prediction_accuracy"] = float(np.mean(err / np.maximum(abs_y, 1e-6)))
        overall.update(_metrics(yt, yp))
    overall.update(n_pred=float(len(predictions)), n_folds=float(fold), train_size=float(train_size))
    log.info(
        "%s | train=%d | folds=%d | prediction_accuracy=%.5f",
        spec.name, train_size, fold, overall.get("prediction_accuracy", np.nan),
    )
    return RollingResult(spec=spec, predictions=predictions, fold_metrics=fold_metrics, overall=overall)


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
    Compare walk-forward prediction_accuracy across screen.feature_sets
    (lowest wins; ties prefer more features).
    Returns (best_features, comparison_table, best_RollingResult).
    """
    rows: List[dict] = []
    results: Dict[str, RollingResult] = {}
    for raw in screen.feature_sets:
        spec = _resolved(RegressionSpec(target=screen.target, features=tuple(raw), name=screen.name))
        label = _label(spec.features)
        try:
            res = rolling_ols(
                full_data, predicted_data, spec,
                train_size=train_size, test_size=test_size, step_size=step_size, **kwargs,
            )
        except (KeyError, ValueError) as exc:
            if not quiet:
                print(f"  skip features={label}: {exc}", flush=True)
            continue
        results[label] = res
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
    ).sort_values(by=["_abs_acc", "_n_feat"], ascending=[True, False], na_position="last")
    best = str(ranked.iloc[0]["features"])
    return results[best].spec.features, comparison, results[best]


def run_regressions(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    screens: Sequence[RegressionScreen] = REGRESSION_SCREENS,
    specs: Optional[Sequence[RegressionSpec]] = None,
    train_size: int = TRAIN_SIZE,
    tune_features: bool = True,
    quiet: bool = False,
    **kwargs,
) -> Dict[str, RollingResult]:
    """
    Run every RegressionScreen (or explicit `specs`, no feature search).
    With tune_features=True, compare screen.feature_sets at fixed k=train_size and
    keep the set with prediction_accuracy closest to 0.
    """
    results: Dict[str, RollingResult] = {}

    def say(msg: str = "") -> None:
        if not quiet:
            print(msg, flush=True)

    def available(target: str) -> bool:
        return target in predicted_data.columns or target in full_data.columns

    if specs is not None:
        for spec in specs:
            if not available(spec.target):
                say(f"  skip {spec.name}: target '{spec.target}' not in panel")
                continue
            try:
                results[spec.name] = rolling_ols(
                    full_data, predicted_data, _resolved(spec), train_size=train_size, **kwargs
                )
            except (KeyError, ValueError) as exc:
                say(f"  skip {spec.name}: {exc}")
        return results

    for screen in screens:
        if not available(screen.target):
            say(f"  skip screen {screen.name}: target '{screen.target}' not in panel")
            continue
        if not screen.feature_sets:
            raise ValueError(f"Screen {screen.name} has empty feature_sets")
        if not tune_features:  # no search: first feature set
            spec = _resolved(
                RegressionSpec(
                    target=screen.target, features=tuple(screen.feature_sets[0]), name=screen.name
                )
            )
            results[screen.name] = rolling_ols(
                full_data, predicted_data, spec, train_size=train_size, **kwargs
            )
            continue

        say()
        say("─" * 72)
        say(f"  [1]  Accuracy metrics  —  {screen.name}  (target={screen.target}, k={train_size})")
        say(
            "  Per fold: rel = |pred − actual| / |actual| · "
            "prediction_accuracy = mean(rel) · pick features closest to 0"
        )
        say("─" * 72)
        say()
        best_feats, comparison, best = select_feature_set(
            full_data, predicted_data, screen, train_size=train_size, quiet=quiet, **kwargs
        )
        if not quiet:
            show = comparison.copy()
            for c in ("mean_error", "mean_value", "prediction_accuracy"):
                show[c] = show[c].round(4)
            say(show.to_string(index=False))
            say()
            o = best.overall
            nan = float("nan")
            say(
                f"  →  chosen features = {_label(best_feats)}    "
                f"prediction_accuracy = {o.get('prediction_accuracy', nan):.4f}  "
                f"(mean_error={o.get('mean_error', nan):.4f}, "
                f"mean_value={o.get('mean_value', nan):.4f})"
            )
            say()
        results[screen.name] = best
    return results


def fit_predict_next(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    spec: RegressionSpec,
    train_size: int = TRAIN_SIZE,
    add_intercept: bool = ADD_INTERCEPT,
) -> dict:
    """
    One-shot prediction for a locked model.
    - y_hat_now:  train iloc[n-k-1 : n-1], predict last bar (realized OOS)
    - y_hat_next: train iloc[n-k : n], forecast with X of last bar (next bar)
    """
    spec = _resolved(spec)
    X_all, y_all = _aligned_xy(full_data, predicted_data, spec)
    n, k = len(y_all), int(train_size)
    if n < k + 1:
        raise ValueError(f"{spec.name}: need at least k+1={k + 1} clean rows; have {n}")

    X, y = X_all.to_numpy(float), y_all.to_numpy(float)
    x_last = X[n - 1 : n]
    beta_now = _fit(X[n - k - 1 : n - 1], y[n - k - 1 : n - 1], add_intercept)
    beta_next = _fit(X[n - k : n], y[n - k : n], add_intercept)
    y_hat_now = float(_predict(x_last, beta_now, add_intercept)[0])
    row = {
        "model": spec.name,
        "structure": spec.target,
        "features": "+".join(spec.features),
        "k": k,
        "time": y_all.index[n - 1],
        "nivel": float(y[n - 1]),
        "y_hat_now": y_hat_now,
        "residual": float(y[n - 1]) - y_hat_now,
        "y_hat_next": float(_predict(x_last, beta_next, add_intercept)[0]),
        "train_start": y_all.index[n - k - 1],
        "train_end": y_all.index[n - 2],
        "equation": equation_from_beta(spec.target, spec.features, beta_now, add_intercept),
    }
    row.update(_coef_dict(spec.features, beta_now, add_intercept))
    return row


# ── Z-score screen ────────────────────────────────────────────────────────────

def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    """z_t = (x_t − μ)/σ with μ, σ from bars t-W … t-1 (no look-ahead)."""
    r = x.rolling(window, min_periods=window)
    return (x - r.mean().shift(1)) / r.std(ddof=1).shift(1).replace(0.0, np.nan)


@dataclass
class ZScoreResult:
    spec: ZScoreSpec
    panel: pd.DataFrame   # series, model, value, fair, z_<W> …
    lookbacks: List[int]
    fair_window: int


def run_zscore_models(
    full_data: pd.DataFrame,
    specs: Sequence[ZScoreSpec] = ZSCORE_MODELS,
    lookbacks: Sequence[int] = ZSCORE_LOOKBACKS,
    fair_window: int = ZSCORE_WINDOW,
) -> Dict[str, ZScoreResult]:
    """Multi-lookback z-scores (past bars only) + rolling-mean fair value per spec."""
    results: Dict[str, ZScoreResult] = {}
    for spec in specs:
        if spec.series not in full_data.columns:
            raise KeyError(
                f"Z-score series '{spec.series}' not in full_data. "
                f"Have: {list(full_data.columns)}"
            )
        x = full_data[spec.series].astype(float)
        data = {
            "value": x,
            "fair": x.rolling(fair_window, min_periods=fair_window).mean().shift(1),
        }
        for w in lookbacks:
            data[f"z_{w}"] = _rolling_z(x, w)
        panel = pd.DataFrame(data, index=x.index)
        panel.index.name = "time"
        panel.insert(0, "series", spec.series)
        panel.insert(1, "model", spec.param)
        results[spec.name] = ZScoreResult(spec, panel, list(lookbacks), fair_window)
    return results


# ── Backtest: residual mean-reversion on locked models ────────────────────────
# z(residual, past L bars) + |residual| threshold → signal → position → PnL.
#   one_bar: each signal bar is its own 1-period trade (n_trades = n_long + n_short)
#   hold:    stay ±1 while signal persists (n_trades = # regime entries)
#   PnL_t = position_{t-1} * (y_t − y_{t-1})

@dataclass
class StrategyResult:
    spec: RegressionSpec
    path: pd.DataFrame
    summary: dict
    position_mode: str


def _run_strategy(
    preds: pd.DataFrame,
    spec: RegressionSpec,
    residual_thr_bp: float,
    z_thr: float,
    z_lookback: int,
    position_mode: str,
    feature_panel: Optional[pd.DataFrame] = None,
) -> StrategyResult:
    mode = (position_mode or "one_bar").strip().lower()
    if mode not in ("one_bar", "hold"):
        raise ValueError(f"BACKTEST_POSITION_MODE must be 'one_bar' or 'hold', got {mode!r}")

    coef_cols = [c for c in preds.columns if c in ("fold", "r2", "intercept") or c.startswith("beta_")]
    path = preds[["y_true", "y_pred", "residual"] + coef_cols].copy()
    if feature_panel is not None:
        for f in spec.features:
            if f in feature_panel.columns:
                path[f"x_{f}"] = feature_panel[f].reindex(path.index)

    # coefficient check: y_pred == intercept + Σ beta_f * x_f
    if "intercept" in path.columns and all(
        f"beta_{f}" in path.columns and f"x_{f}" in path.columns for f in spec.features
    ):
        chk = path["intercept"].astype(float)
        for f in spec.features:
            chk = chk + path[f"beta_{f}"].astype(float) * path[f"x_{f}"].astype(float)
        path["y_pred_from_coefs"] = chk
        path["y_pred_check_ok"] = (path["y_pred"] - chk).abs() < 1e-6

    res, z = path["residual"], _rolling_z(path["residual"], z_lookback)
    big = res.abs() >= residual_thr_bp
    path["z_resid"] = z
    path["signal"] = pd.Series(
        np.select([(z <= -z_thr) & big, (z >= z_thr) & big], [1.0, -1.0], default=0.0),
        index=path.index,
    )
    # |position| ≤ 1 in both modes; modes differ only in trade accounting
    path["position"] = path["signal"].astype(float).clip(-1.0, 1.0)
    path["position_mode"] = mode

    pos = path["position"]
    path["dy"] = path["y_true"].diff()
    path["position_lag"] = pos.shift(1)
    path["pnl"] = path["position_lag"] * path["dy"]
    path["cum_pnl"] = path["pnl"].fillna(0.0).cumsum()

    n_long, n_short = int((pos == 1).sum()), int((pos == -1).sum())
    entries = int(((pos != 0) & (pos.shift(1).fillna(0) != pos)).sum())
    n_trades = n_long + n_short if mode == "one_bar" else entries
    traded = path["pnl"].dropna()
    nan = float("nan")
    summary = {
        "model": spec.name,
        "structure": spec.target,
        "features": "+".join(spec.features),
        "position_mode": mode,
        "n_bars": int(len(path)),
        "n_trades": n_trades,
        "n_entries": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "n_flat": int((pos == 0).sum()),
        "total_pnl": float(traded.sum()) if len(traded) else nan,
        "mean_pnl": float(traded.mean()) if len(traded) else nan,
        "hit_rate": float((traded > 0).mean()) if len(traded) else nan,
        "max_dd": float((path["cum_pnl"] - path["cum_pnl"].cummax()).min()) if len(path) else nan,
    }
    return StrategyResult(spec=spec, path=path, summary=summary, position_mode=mode)


def run_chosen_backtest(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = CHOSEN_MODELS,
    train_size: int = TRAIN_SIZE,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, RollingResult]:
    """Walk-forward on CHOSEN_MODELS over [start, end] (no feature tune)."""
    full, pred = full_data, predicted_data
    if start is not None:
        t0 = pd.Timestamp(start)
        full, pred = full.loc[full.index >= t0], pred.loc[pred.index >= t0]
    if end is not None:
        t1 = pd.Timestamp(end)
        full, pred = full.loc[full.index <= t1], pred.loc[pred.index <= t1]
    return run_regressions(
        full, pred, specs=[_resolved(s) for s in specs],
        train_size=train_size, tune_features=False, quiet=True,
    )


def backtest_chosen_models(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = CHOSEN_MODELS,
    train_size: int = TRAIN_SIZE,
    start: Optional[str] = BACKTEST_START,
    end: Optional[str] = BACKTEST_END,
    residual_thr_bp: float = BACKTEST_RESIDUAL_THR_BP,
    z_thr: float = BACKTEST_ZSCORE_THR,
    z_lookback: int = BACKTEST_Z_LOOKBACK,
    position_mode: str = BACKTEST_POSITION_MODE,
) -> Dict[str, StrategyResult]:
    """Walk-forward residuals → strategy path for each locked model."""
    wf = run_chosen_backtest(
        full_data, predicted_data, specs=specs, train_size=train_size, start=start, end=end
    )
    out: Dict[str, StrategyResult] = {}
    for name, res in wf.items():
        if res.predictions.empty:
            continue
        preds = res.predictions.copy()
        preds.index.name = "time"
        out[name] = _run_strategy(
            preds, res.spec, residual_thr_bp, z_thr, z_lookback, position_mode, full_data
        )
    return out


_SUMMARY_COLS = [
    "model", "structure", "features", "position_mode", "n_bars", "n_trades", "n_entries",
    "n_long", "n_short", "n_flat", "total_pnl", "mean_pnl", "hit_rate", "max_dd",
]


def summary_table(results: Dict[str, StrategyResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.summary for r in results.values()])[_SUMMARY_COLS]


def save_backtest_plots(result: StrategyResult) -> List[str]:
    """Save cum-PnL and variables (y_true / y_pred [/ single feature]) plots; return paths."""
    name, spec, path = result.spec.name, result.spec, result.path
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  skip plots ({name}): matplotlib not installed", flush=True)
        return []

    written: List[str] = []

    def finish(fig, ax, title: str, ylabel: str, outfile: str) -> None:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outfile, dpi=120)
        plt.close(fig)
        written.append(outfile)

    # 1) cumulative PnL
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.plot(path.index, path["cum_pnl"], color="C0", lw=1.2, label="cum_pnl")
    ax.legend(loc="best")
    finish(
        fig, ax,
        f"Backtest cum PnL — {name}  ({spec.target})  mode={result.position_mode}",
        "cum PnL (bp × position)", f"backtest_pnl_{name}.png",
    )

    # 2) y_true / y_pred (+ feature on twin axis when single-feature)
    feat = spec.features[0] if len(spec.features) == 1 else None
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(path.index, path["y_true"], color="C0", lw=1.0, label=f"y_true ({spec.target})")
    ax.plot(path.index, path["y_pred"], color="C1", lw=1.0, alpha=0.85, label="y_pred")
    handles, labels = ax.get_legend_handles_labels()
    if feat and f"x_{feat}" in path.columns:
        ax2 = ax.twinx()
        ax2.plot(path.index, path[f"x_{feat}"], color="C2", lw=0.9, alpha=0.7, label=f"x ({feat})")
        ax2.set_ylabel(f"feature {feat}", color="C2")
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="best")
    finish(
        fig, ax,
        f"Backtest variables — {name}  {spec.target} ~ {'+'.join(spec.features)}",
        "structure (bp)", f"backtest_vars_{name}.png",
    )
    return written


# ── Locked-model predictions + latest z-scores ────────────────────────────────

def run_chosen_predictions(
    full_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    specs: Sequence[RegressionSpec] = CHOSEN_MODELS,
    train_size: int = TRAIN_SIZE,
) -> pd.DataFrame:
    """Locked-model next-bar table: model, structure, features, k, time, nivel,
    y_hat_now, residual, y_hat_next, equation, intercept, beta_*."""
    return pd.DataFrame(
        [fit_predict_next(full_data, predicted_data, s, train_size=train_size) for s in specs]
    )


def append_zscore_latest(table: pd.DataFrame, full_data: pd.DataFrame) -> pd.DataFrame:
    """Append latest z-score rows (nivel + z_*) to a prediction table."""
    z_cols = [f"z_{w}" for w in ZSCORE_LOOKBACKS]
    nan = float("nan")
    extra: List[dict] = []
    for res in run_zscore_models(full_data, specs=ZSCORE_MODELS).values():
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
            "y_hat_now": nan,
            "residual": float(last["z_50"]) if "z_50" in last.index else nan,
            "y_hat_next": nan,
            "equation": "",
        }
        for c in z_cols:
            if c in last.index:
                row[c] = float(last[c]) if pd.notna(last[c]) else nan
        extra.append(row)
    if not extra:
        return table
    return pd.concat([table, pd.DataFrame(extra)], ignore_index=True)


# ── Latest-observation summary + email ────────────────────────────────────────

_SUMMARY_VIEW = ["model", "structure", "nivel", "predicted", COL_RZ, "equation", "time"]


def build_summary(
    reg_results: Dict[str, RollingResult],
    z_results: Dict[str, ZScoreResult],
) -> pd.DataFrame:
    """
    One row per model at the latest timestamp (regressions first, then z-scores).
    regressions: predicted = y_pred, residual/zscore = residual, equation filled.
    z-scores:    predicted blank, residual/zscore = z_50, equation blank.
    """
    nan = float("nan")
    rows: List[dict] = []
    for res in reg_results.values():
        preds = res.predictions.dropna(subset=["y_pred"])
        if preds.empty:
            continue
        last = preds.iloc[-1]
        rows.append(
            {
                "model": res.spec.name,
                "structure": res.spec.target,
                "nivel": float(last["y_true"]),
                "predicted": float(last["y_pred"]),
                COL_RZ: float(last["residual"]),
                "equation": _equation_from_row(res.spec, last),
                "time": preds.index[-1],
            }
        )
    for res in z_results.values():
        panel = res.panel.dropna(subset=["fair"])
        if panel.empty:
            continue
        last = panel.iloc[-1]
        z50 = last["z_50"] if "z_50" in last.index else nan
        rows.append(
            {
                "model": res.spec.param,
                "structure": res.spec.series,
                "nivel": float(last["value"]),
                "predicted": nan,
                COL_RZ: float(z50) if pd.notna(z50) else nan,
                "equation": "",
                "time": panel.index[-1],
            }
        )
    if not rows:
        return pd.DataFrame(columns=_SUMMARY_VIEW)
    return pd.DataFrame(rows)[_SUMMARY_VIEW]


def format_summary_text(summary: pd.DataFrame, asof: Optional[str] = None) -> str:
    """Plain-text body: as-of header + results table (numbers at 3dp, NaN blank)."""
    show = summary.copy()
    for c in ("nivel", "predicted", COL_RZ):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").apply(
                lambda v: "" if pd.isna(v) else f"{v:.3f}"
            )
    show = show[[c for c in _SUMMARY_VIEW if c in show.columns]]
    header = "SOFR screener — results" + (f"  |  asof {asof}" if asof else "")
    return header + "\n\n" + show.to_string(index=False) + "\n"


def _cfg(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def send_summary_email(
    summary: pd.DataFrame,
    asof: Optional[str] = None,
    subject: Optional[str] = None,
    to: Optional[Sequence[str]] = None,
    body: Optional[str] = None,
) -> bool:
    """
    Send summary. Returns True if sent, False if skipped (disabled / incomplete).
    Credentials from curve_config or env (EMAIL_ENABLED, SMTP_*, REPORT_TO, REPORT_FROM).
    Pass body= to override format_summary_text (e.g. prediction tables).
    """
    enabled = os.environ.get("EMAIL_ENABLED", str(EMAIL_ENABLED)).lower() in {"1", "true", "yes"}
    host = _cfg("SMTP_HOST", SMTP_HOST)
    port = int(_cfg("SMTP_PORT", str(SMTP_PORT)) or 587)
    user = _cfg("SMTP_USER", SMTP_USER)
    password = _cfg("SMTP_PASSWORD", SMTP_PASSWORD)
    mail_from = _cfg("REPORT_FROM", REPORT_FROM) or user
    mail_to: List[str] = list(to) if to else list(REPORT_TO)
    env_to = _cfg("REPORT_TO")
    if env_to:
        mail_to = [x.strip() for x in env_to.split(",") if x.strip()]

    text = body if body is not None else format_summary_text(summary, asof=asof)
    if not enabled or not host or not mail_to or not mail_from:
        if body is None:
            print(text, flush=True)
        print(
            "[email skipped] set EMAIL_ENABLED=true and SMTP_HOST / REPORT_TO "
            "(and credentials) to send",
            flush=True,
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject or f"SOFR screener {asof or ''}".strip()
    msg["From"] = mail_from
    msg["To"] = ", ".join(mail_to)
    msg.attach(MIMEText(text, "plain", "utf-8"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.sendmail(mail_from, mail_to, msg.as_string())
    return True
