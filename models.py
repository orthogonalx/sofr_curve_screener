"""Rolling OLS, z-scores, residual backtest, summary + email."""
from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import curve_config as _curve_cfg
from curve_config import (
    ADD_INTERCEPT, BACKTEST_END, BACKTEST_POSITION_MODE, BACKTEST_RESIDUAL_THR_BP,
    BACKTEST_START, BACKTEST_Z_LOOKBACK, BACKTEST_ZSCORE_THR, CHOSEN_MODELS,
    DATA_BAR_MINUTES, EMAIL_ENABLED, MIN_TRAIN_OBS, REGRESSION_SCREENS, REPORT_FROM,
    REPORT_TO, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER, STEP_SIZE, TEST_SIZE,
    TRAIN_SIZE, ZSCORE_LOOKBACKS, ZSCORE_MODELS, ZSCORE_WINDOW, RegressionScreen,
    RegressionSpec, ZScoreSpec,
)
from data import resolve_feature_name

EMAIL_ATTACH_CSV = getattr(_curve_cfg, "EMAIL_ATTACH_CSV", False)
EMAIL_ATTACH_GRAPHS = getattr(_curve_cfg, "EMAIL_ATTACH_GRAPHS", False)
EMAIL_INLINE_GRAPHS = getattr(_curve_cfg, "EMAIL_INLINE_GRAPHS", True)
SMTP_USE_TLS = getattr(_curve_cfg, "SMTP_USE_TLS", True)
GRAPH_LOOKBACK_DAYS = getattr(_curve_cfg, "GRAPH_LOOKBACK_DAYS", 9)
GRAPH_DIR_EVOLUTION = getattr(_curve_cfg, "GRAPH_DIR_EVOLUTION", "graphs/evolution")
GRAPH_DIR_BACKTEST_PNL = getattr(_curve_cfg, "GRAPH_DIR_BACKTEST_PNL", "graphs/backtest_pnl")
GRAPH_DIR_BACKTEST_VARS = getattr(_curve_cfg, "GRAPH_DIR_BACKTEST_VARS", "graphs/backtest_vars")
log = logging.getLogger(__name__)
COL_RZ = "residual/zscore"
PREDICTION_COLS = [
    "model", "structure", "features", "side", "predicted_level", "time",
    "y_pred_prev", "y_realized_prev", "residuo", "residuo_std", "equation", "n_lookback",
]
_SUMMARY_COLS = [
    "model", "structure", "features", "position_mode", "n_bars", "n_trades", "n_entries",
    "n_long", "n_short", "n_flat", "total_pnl", "mean_pnl", "hit_rate", "max_dd",
]
_SUMMARY_VIEW = ["model", "structure", "nivel", "predicted", COL_RZ, "equation", "time"]
_NAN = float("nan")


def _round3(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v if np.isnan(v) else round(v, 3)


def _side_from_residuo(residuo):
    """realized < predicted → PAY (red); realized > predicted → REC (green)."""
    try:
        r = float(residuo)
    except (TypeError, ValueError):
        return ""
    if np.isnan(r) or abs(r) < 1e-12:
        return ""
    return "PAY" if r < 0 else "REC"


def format_prediction_table(table):
    if table is None or table.empty:
        return table
    out = table.copy()
    if "residuo" not in out.columns and "residual_prev" in out.columns:
        out["residuo"] = out["residual_prev"]
    if "side" not in out.columns and "residuo" in out.columns:
        out["side"] = out["residuo"].map(_side_from_residuo)
    for c in list(out.columns):
        if c in (
            "predicted_level", "y_pred_prev", "y_realized_prev", "residuo",
            "residuo_std", "residual_prev", "intercept",
        ) or str(c).startswith("beta_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    ordered = [c for c in PREDICTION_COLS if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def _design(X, ic):
    return np.column_stack([np.ones(len(X)), X]) if ic else X


def _fit(X, y, ic=True):
    return np.linalg.lstsq(_design(X, ic), y, rcond=None)[0]


def _predict(X, beta, ic=True):
    return _design(X, ic) @ beta


def _metrics(y, yhat):
    resid, sse = y - yhat, float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(sse / max(len(y), 1))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - sse / sst) if sst > 1e-18 else np.nan,
    }


def _coef_dict(features, beta, ic):
    out, off = ({"intercept": float(beta[0])} if ic else {}), (1 if ic else 0)
    for j, f in enumerate(features):
        out["beta_%s" % f] = float(beta[j + off])
    return out


def equation_from_beta(target, features, beta, add_intercept=True):
    coefs = beta[1:] if add_intercept else beta
    pieces = ["%.3f" % (float(beta[0]) if add_intercept else 0.0)]
    for f, b in zip(features, coefs):
        pieces.append("%s %.3f*%s" % ("+" if b >= 0 else "-", abs(float(b)), f))
    return "%s = %s" % (target, " ".join(pieces))


def _equation_from_row(spec, row):
    feats = [f for f in spec.features if pd.notna(row.get("beta_%s" % f))]
    beta = np.array([float(row.get("intercept", 0.0))] + [float(row["beta_%s" % f]) for f in feats])
    return equation_from_beta(spec.target, feats, beta, True)


def _resolved(spec):
    return RegressionSpec(
        name=spec.name, target=spec.target,
        features=tuple(resolve_feature_name(f) for f in spec.features),
    )


def _label(features):
    return "+".join(features) if features else "(none)"


@dataclass
class RollingResult:
    spec: RegressionSpec
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall: Dict[str, float] = field(default_factory=dict)

    @property
    def y_pred(self):
        return self.predictions["y_pred"]


def _aligned_xy(full_data, predicted_data, spec):
    src = predicted_data if spec.target in predicted_data.columns else full_data
    if spec.target not in src.columns:
        raise KeyError("Target '%s' not found in predicted_data/full_data" % spec.target)
    missing = [f for f in spec.features if f not in full_data.columns]
    if missing:
        raise KeyError("Features missing from full_data: %s" % missing)
    feats = list(spec.features)
    frame = pd.concat([src[spec.target].rename("y"), full_data[feats]], axis=1).dropna()
    return frame[feats], frame["y"]


def rolling_ols(
    full_data, predicted_data, spec, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
    step_size=STEP_SIZE, add_intercept=ADD_INTERCEPT, min_train_obs=MIN_TRAIN_OBS,
):
    X_all, y_all = _aligned_xy(full_data, predicted_data, spec)
    n = len(y_all)
    if n < train_size + 1:
        raise ValueError("Need at least train_size+1=%d clean obs; have %d" % (train_size + 1, n))
    pred_y = pd.Series(index=y_all.index, dtype=float)
    pred_fold = pd.Series(index=y_all.index, dtype=float)
    rows, start, fold = [], 0, 0
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
        pred_y.loc[idx_te], pred_fold.loc[idx_te] = y_hat, fold
        row = {
            "fold": fold, "train_start": y_all.index[tr][0], "train_end": y_all.index[tr][-1],
            "test_start": idx_te[0], "test_end": idx_te[-1], "n_train": len(y_tr), "n_test": len(y_te),
            "train_r2": train_m["r2"], "train_rmse": train_m["rmse"],
            "test_r2": test_m["r2"], "test_rmse": test_m["rmse"], "test_mae": test_m["mae"],
        }
        row.update(_coef_dict(spec.features, beta, add_intercept))
        rows.append(row)
        fold += 1
        start += step_size
    mask = pred_y.notna()
    predictions = pd.DataFrame({
        "y_true": y_all[mask], "y_pred": pred_y[mask],
        "residual": y_all[mask] - pred_y[mask], "fold": pred_fold[mask].astype(int),
    })
    fold_metrics = pd.DataFrame(rows)
    if not fold_metrics.empty:
        model_cols = ["train_r2"] + [
            c for c in fold_metrics.columns if c == "intercept" or c.startswith("beta_")
        ]
        predictions = predictions.join(
            fold_metrics.set_index("fold")[model_cols].rename(columns={"train_r2": "r2"}), on="fold",
        )
    overall = {}
    if len(predictions):
        yt, yp = predictions["y_true"].to_numpy(float), predictions["y_pred"].to_numpy(float)
        err, abs_y = np.abs(yp - yt), np.abs(yt)
        overall.update(
            mean_error=float(err.mean()), mean_value=float(abs_y.mean()),
            prediction_accuracy=float(np.mean(err / np.maximum(abs_y, 1e-6))),
            **_metrics(yt, yp),
        )
    overall.update(n_pred=float(len(predictions)), n_folds=float(fold), train_size=float(train_size))
    log.info("%s | train=%d | folds=%d | prediction_accuracy=%.5f",
             spec.name, train_size, fold, overall.get("prediction_accuracy", np.nan))
    return RollingResult(spec=spec, predictions=predictions, fold_metrics=fold_metrics, overall=overall)


def select_feature_set(
    full_data, predicted_data, screen, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
    step_size=STEP_SIZE, quiet=False, **kwargs,
):
    rows, results = [], {}
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
                print("  skip features=%s: %s" % (label, exc), flush=True)
            continue
        results[label] = res
        o = res.overall
        rows.append({
            "features": label, "n_folds": int(o.get("n_folds", 0)),
            "mean_error": o.get("mean_error", np.nan), "mean_value": o.get("mean_value", np.nan),
            "prediction_accuracy": o.get("prediction_accuracy", np.nan),
        })
    if not rows:
        raise ValueError("No valid feature sets for screen=%s target=%s; tried %s" % (
            screen.name, screen.target, list(screen.feature_sets)))
    comparison = pd.DataFrame(rows).reset_index(drop=True)
    ranked = comparison.assign(
        _abs_acc=comparison["prediction_accuracy"].abs(),
        _n_feat=comparison["features"].str.count(r"\+") + 1,
    ).sort_values(by=["_abs_acc", "_n_feat"], ascending=[True, False], na_position="last")
    best = str(ranked.iloc[0]["features"])
    return results[best].spec.features, comparison, results[best]


def run_regressions(
    full_data, predicted_data, screens=REGRESSION_SCREENS, specs=None,
    train_size=TRAIN_SIZE, tune_features=True, quiet=False, **kwargs,
):
    results = {}

    def say(msg=""):
        if not quiet:
            print(msg, flush=True)

    def available(target):
        return target in predicted_data.columns or target in full_data.columns

    if specs is not None:
        for spec in specs:
            if not available(spec.target):
                say("  skip %s: target '%s' not in panel" % (spec.name, spec.target))
                continue
            try:
                results[spec.name] = rolling_ols(
                    full_data, predicted_data, _resolved(spec), train_size=train_size, **kwargs
                )
            except (KeyError, ValueError) as exc:
                say("  skip %s: %s" % (spec.name, exc))
        return results

    for screen in screens:
        if not available(screen.target):
            say("  skip screen %s: target '%s' not in panel" % (screen.name, screen.target))
            continue
        if not screen.feature_sets:
            raise ValueError("Screen %s has empty feature_sets" % screen.name)
        if not tune_features:
            spec = _resolved(RegressionSpec(
                target=screen.target, features=tuple(screen.feature_sets[0]), name=screen.name
            ))
            results[screen.name] = rolling_ols(
                full_data, predicted_data, spec, train_size=train_size, **kwargs
            )
            continue
        say(); say("─" * 72)
        say("  [1]  Accuracy metrics  —  %s  (target=%s, k=%s)" % (
            screen.name, screen.target, train_size))
        say("  Per fold: rel = |pred − actual| / |actual| · "
            "prediction_accuracy = mean(rel) · pick features closest to 0")
        say("─" * 72); say()
        best_feats, comparison, best = select_feature_set(
            full_data, predicted_data, screen, train_size=train_size, quiet=quiet, **kwargs
        )
        if not quiet:
            show = comparison.copy()
            for c in ("mean_error", "mean_value", "prediction_accuracy"):
                show[c] = show[c].round(4)
            say(show.to_string(index=False)); say()
            o = best.overall
            say("  →  chosen features = %s    prediction_accuracy = %.4f  "
                "(mean_error=%.4f, mean_value=%.4f)" % (
                    _label(best_feats), o.get("prediction_accuracy", _NAN),
                    o.get("mean_error", _NAN), o.get("mean_value", _NAN)))
            say()
        results[screen.name] = best
    return results


def _oos_residual_std(X, y, k, add_intercept=True, max_folds=None):
    """Std of 1-step OOS residuals: train [t-k:t) → predict t, over recent folds."""
    n = len(y)
    if n <= k:
        return _NAN
    max_folds = int(max_folds or k)
    t0 = max(k, n - max_folds)
    resids = []
    for t in range(t0, n):
        beta = _fit(X[t - k:t], y[t - k:t], add_intercept)
        yhat = float(_predict(X[t:t + 1], beta, add_intercept)[0])
        resids.append(float(y[t]) - yhat)
    if len(resids) < 2:
        return _NAN
    return _round3(float(np.std(resids, ddof=1)))


def fit_predict_next(
    full_data, predicted_data, spec, train_size=TRAIN_SIZE, add_intercept=ADD_INTERCEPT,
):
    spec = _resolved(spec)
    X_all, y_all = _aligned_xy(full_data, predicted_data, spec)
    n, k = len(y_all), int(train_size)
    if n < k + 1:
        raise ValueError("%s: need at least k+1=%d clean rows; have %d" % (spec.name, k + 1, n))
    X, y = X_all.to_numpy(float), y_all.to_numpy(float)
    x_last, t_prev = X[n - 1:n], y_all.index[n - 1]
    t_next = pd.Timestamp(t_prev) + pd.Timedelta(minutes=int(DATA_BAR_MINUTES))
    beta_now = _fit(X[n - k - 1:n - 1], y[n - k - 1:n - 1], add_intercept)
    beta_next = _fit(X[n - k:n], y[n - k:n], add_intercept)
    y_pred_prev = _round3(_predict(x_last, beta_now, add_intercept)[0])
    y_realized_prev = _round3(y[n - 1])
    residuo = _round3(y_realized_prev - y_pred_prev)
    row = {
        "model": spec.name, "structure": spec.target, "features": "+".join(spec.features),
        "side": _side_from_residuo(residuo),
        "predicted_level": _round3(_predict(x_last, beta_next, add_intercept)[0]),
        "time": t_next, "y_pred_prev": y_pred_prev, "y_realized_prev": y_realized_prev,
        "residuo": residuo,
        "residuo_std": _oos_residual_std(X, y, k, add_intercept, max_folds=k),
        "residual_prev": residuo,  # legacy alias
        "equation": equation_from_beta(spec.target, spec.features, beta_now, add_intercept),
        "n_lookback": k, "train_start": y_all.index[n - k - 1],
        "train_end": y_all.index[n - 2], "asof_prev": t_prev,
    }
    row.update({kk: _round3(vv) for kk, vv in _coef_dict(spec.features, beta_now, add_intercept).items()})
    return row


def _rolling_z(x, window):
    r = x.rolling(window, min_periods=window)
    return (x - r.mean().shift(1)) / r.std(ddof=1).shift(1).replace(0.0, np.nan)


@dataclass
class ZScoreResult:
    spec: ZScoreSpec
    panel: pd.DataFrame
    lookbacks: List[int]
    fair_window: int


def run_zscore_models(
    full_data, specs=ZSCORE_MODELS, lookbacks=ZSCORE_LOOKBACKS, fair_window=ZSCORE_WINDOW,
):
    results = {}
    for spec in specs:
        if spec.series not in full_data.columns:
            raise KeyError("Z-score series '%s' not in full_data. Have: %s" % (
                spec.series, list(full_data.columns)))
        x = full_data[spec.series].astype(float)
        data = {
            "value": x,
            "fair": x.rolling(fair_window, min_periods=fair_window).mean().shift(1),
        }
        for w in lookbacks:
            data["z_%d" % w] = _rolling_z(x, w)
        panel = pd.DataFrame(data, index=x.index)
        panel.index.name = "time"
        panel.insert(0, "series", spec.series)
        panel.insert(1, "model", spec.param)
        results[spec.name] = ZScoreResult(spec, panel, list(lookbacks), fair_window)
    return results


@dataclass
class StrategyResult:
    spec: RegressionSpec
    path: pd.DataFrame
    summary: dict
    position_mode: str


def _run_strategy(preds, spec, residual_thr_bp, z_thr, z_lookback, position_mode,
                  feature_panel=None):
    mode = (position_mode or "one_bar").strip().lower()
    if mode not in ("one_bar", "hold"):
        raise ValueError("BACKTEST_POSITION_MODE must be 'one_bar' or 'hold', got %r" % mode)
    coef_cols = [c for c in preds.columns if c in ("fold", "r2", "intercept") or c.startswith("beta_")]
    path = preds[["y_true", "y_pred", "residual"] + coef_cols].copy()
    if feature_panel is not None:
        for f in spec.features:
            if f in feature_panel.columns:
                path["x_%s" % f] = feature_panel[f].reindex(path.index)
    if "intercept" in path.columns and all(
        "beta_%s" % f in path.columns and "x_%s" % f in path.columns for f in spec.features
    ):
        chk = path["intercept"].astype(float)
        for f in spec.features:
            chk = chk + path["beta_%s" % f].astype(float) * path["x_%s" % f].astype(float)
        path["y_pred_from_coefs"], path["y_pred_check_ok"] = chk, (path["y_pred"] - chk).abs() < 1e-6
    res, z = path["residual"], _rolling_z(path["residual"], z_lookback)
    big = res.abs() >= residual_thr_bp
    path["z_resid"] = z
    path["signal"] = pd.Series(
        np.select([(z <= -z_thr) & big, (z >= z_thr) & big], [1.0, -1.0], default=0.0),
        index=path.index,
    )
    path["position"] = path["signal"].astype(float).clip(-1.0, 1.0)
    path["position_mode"] = mode
    pos = path["position"]
    path["dy"], path["position_lag"] = path["y_true"].diff(), pos.shift(1)
    path["pnl"] = path["position_lag"] * path["dy"]
    path["cum_pnl"] = path["pnl"].fillna(0.0).cumsum()
    n_long, n_short = int((pos == 1).sum()), int((pos == -1).sum())
    entries = int(((pos != 0) & (pos.shift(1).fillna(0) != pos)).sum())
    n_trades = n_long + n_short if mode == "one_bar" else entries
    traded = path["pnl"].dropna()
    summary = {
        "model": spec.name, "structure": spec.target, "features": "+".join(spec.features),
        "position_mode": mode, "n_bars": int(len(path)), "n_trades": n_trades,
        "n_entries": n_trades, "n_long": n_long, "n_short": n_short, "n_flat": int((pos == 0).sum()),
        "total_pnl": float(traded.sum()) if len(traded) else _NAN,
        "mean_pnl": float(traded.mean()) if len(traded) else _NAN,
        "hit_rate": float((traded > 0).mean()) if len(traded) else _NAN,
        "max_dd": float((path["cum_pnl"] - path["cum_pnl"].cummax()).min()) if len(path) else _NAN,
    }
    return StrategyResult(spec=spec, path=path, summary=summary, position_mode=mode)


def run_chosen_backtest(
    full_data, predicted_data, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE, start=None, end=None,
):
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
    full_data, predicted_data, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE,
    start=BACKTEST_START, end=BACKTEST_END, residual_thr_bp=BACKTEST_RESIDUAL_THR_BP,
    z_thr=BACKTEST_ZSCORE_THR, z_lookback=BACKTEST_Z_LOOKBACK,
    position_mode=BACKTEST_POSITION_MODE,
):
    wf = run_chosen_backtest(
        full_data, predicted_data, specs=specs, train_size=train_size, start=start, end=end
    )
    out = {}
    for name, res in wf.items():
        if res.predictions.empty:
            continue
        preds = res.predictions.copy()
        preds.index.name = "time"
        out[name] = _run_strategy(
            preds, res.spec, residual_thr_bp, z_thr, z_lookback, position_mode, full_data
        )
    return out


def summary_table(results):
    return pd.DataFrame([r.summary for r in results.values()])[_SUMMARY_COLS] if results else pd.DataFrame()


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slice_last_days(df, days, bar_minutes=None):
    """
    Last `days` calendar days of rows.
    If the window is sparse/gapped (<2 points) — common with synthetic stores —
    fall back to ~days worth of bars using DATA_BAR_MINUTES.
    """
    if df is None or df.empty:
        return df
    days = float(days)
    end = pd.Timestamp(df.index[-1])
    sl = df.loc[df.index >= end - pd.Timedelta(days=days)]
    if len(sl) >= 2:
        return sl
    bm = float(bar_minutes if bar_minutes is not None else DATA_BAR_MINUTES)
    n_bars = max(int(round(days * (24.0 * 60.0 / max(bm, 1.0)))), 2)
    n_bars = min(n_bars, len(df))
    print(
        "  [graphs] calendar window sparse (n=%d) → using last %d bars (~%g days @ %gm)"
        % (len(sl), n_bars, days, bm),
        flush=True,
    )
    return df.iloc[-n_bars:]


def save_evolution_plots(full_data, days=None, specs=None, out_dir=None):
    """Plot CHOSEN structures over lookback; small PNGs for email inline."""
    days = GRAPH_LOOKBACK_DAYS if days is None else int(days)
    specs = list(specs or CHOSEN_MODELS)
    out_dir = _ensure_dir(out_dir or GRAPH_DIR_EVOLUTION)
    plt = _plt()
    if plt is None:
        print("  skip evolution plots: matplotlib not installed", flush=True)
        return []
    panel = _slice_last_days(full_data, days)
    if panel is None or panel.empty:
        print("  skip evolution plots: empty panel", flush=True)
        return []
    written = []
    targets = []
    for spec in specs:
        t = getattr(spec, "target", None)
        if t and t in panel.columns and t not in targets:
            targets.append(t)
    if not targets:
        print("  skip evolution plots: no target columns in full_data", flush=True)
        return []

    def _save(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # Combined (compact)
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    for t in targets:
        ax.plot(panel.index, panel[t], lw=1.0, label=t)
    ax.set_title("Structures — last ~%d days  (n=%d)" % (days, len(panel)), fontsize=9)
    ax.set_ylabel("bp", fontsize=8); ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=6, ncol=2, frameon=False)
    _save(fig, str(out_dir / "evolution_structures.png"))

    for spec in specs:
        cols = [c for c in ([spec.target] + list(spec.features)) if c in panel.columns]
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 2.0))
        for c in cols:
            ax.plot(panel.index, panel[c], lw=1.0, label=c)
        ax.set_title("%s  (%s)" % (spec.name, spec.target), fontsize=9)
        ax.set_ylabel("bp", fontsize=8); ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6, frameon=False)
        _save(fig, str(out_dir / ("evolution_%s.png" % spec.name)))

    print("  [graphs] evolution → %s  (%d files, lookback~%d days, points=%d)"
          % (out_dir, len(written), days, len(panel)), flush=True)
    return written


def format_prediction_html(table, asof="", graph_cids=None):
    """HTML body: header + colored table + optional inline CID images."""
    cols = [c for c in PREDICTION_COLS if c in table.columns]
    show = table[cols].copy() if cols else table.copy()
    skip_num = {"model", "structure", "features", "equation", "time", "n_lookback", "side"}
    for c in show.columns:
        if c in skip_num:
            continue
        show[c] = pd.to_numeric(show[c], errors="coerce")

    def _cell(col, v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            txt = ""
        elif isinstance(v, (float, np.floating)):
            txt = "%.3f" % float(v)
        else:
            txt = str(v)
        extra = ""
        if col == "side":
            if txt == "PAY":
                extra = "color:#c62828;font-weight:bold;"
            elif txt == "REC":
                extra = "color:#2e7d32;font-weight:bold;"
        return (
            '<td style="padding:6px 8px;border-bottom:1px solid #e6eaf0;font-size:12px;'
            'font-family:Arial,sans-serif;white-space:nowrap;%s">%s</td>' % (extra, txt)
        )

    rows_html = []
    for i, (_, row) in enumerate(show.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#f4f7fb"
        tds = "".join(_cell(c, row[c]) for c in show.columns)
        rows_html.append('<tr style="background:%s;">%s</tr>' % (bg, tds))

    th = "".join(
        '<th style="padding:8px;background:#1f4e79;color:#fff;font-size:12px;'
        'font-family:Arial,sans-serif;text-align:left;border:none;">%s</th>' % c
        for c in show.columns
    )
    table_html = (
        '<table style="border-collapse:collapse;width:100%%;max-width:960px;" cellpadding="0" cellspacing="0">'
        "<thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
        % (th, "".join(rows_html))
    )

    imgs = ""
    if graph_cids:
        blocks = []
        for cid, title in graph_cids:
            blocks.append(
                '<div style="display:inline-block;vertical-align:top;margin:4px 6px 8px 0;">'
                '<div style="font:11px Arial,sans-serif;color:#445;margin-bottom:2px;">%s</div>'
                '<img src="cid:%s" width="320" style="display:block;border:1px solid #dde3ea;border-radius:4px;" />'
                "</div>" % (title, cid)
            )
        imgs = (
            '<div style="margin-top:14px;">'
            '<div style="font:13px Arial,sans-serif;color:#1f4e79;font-weight:bold;margin-bottom:6px;">'
            "Evolution</div>%s</div>" % "".join(blocks)
        )

    return (
        '<div style="font-family:Arial,sans-serif;color:#222;">'
        '<div style="font-size:15px;font-weight:bold;color:#1f4e79;margin-bottom:4px;">SOFR prediction</div>'
        '<div style="font-size:12px;color:#556;margin-bottom:12px;">asof %s &nbsp;|&nbsp; k=%s &nbsp;|&nbsp; next ~%sm</div>'
        "%s%s</div>"
        % (asof, TRAIN_SIZE, DATA_BAR_MINUTES, table_html, imgs)
    )


def save_backtest_plots(result):
    name, spec, path = result.spec.name, result.spec, result.path
    plt = _plt()
    if plt is None:
        print("  skip plots (%s): matplotlib not installed" % name, flush=True)
        return []
    pnl_dir = _ensure_dir(GRAPH_DIR_BACKTEST_PNL)
    vars_dir = _ensure_dir(GRAPH_DIR_BACKTEST_VARS)
    written = []

    def finish(fig, ax, title, ylabel, outfile):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("time")
        ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(outfile, dpi=120)
        plt.close(fig); written.append(outfile)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.plot(path.index, path["cum_pnl"], color="C0", lw=1.2, label="cum_pnl")
    ax.legend(loc="best")
    finish(fig, ax, "Backtest cum PnL — %s  (%s)  mode=%s" % (
        name, spec.target, result.position_mode), "cum PnL (bp × position)",
           str(pnl_dir / ("backtest_pnl_%s.png" % name)))
    feat = spec.features[0] if len(spec.features) == 1 else None
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(path.index, path["y_true"], color="C0", lw=1.0, label="y_true (%s)" % spec.target)
    ax.plot(path.index, path["y_pred"], color="C1", lw=1.0, alpha=0.85, label="y_pred")
    handles, labels = ax.get_legend_handles_labels()
    if feat and "x_%s" % feat in path.columns:
        ax2 = ax.twinx()
        ax2.plot(path.index, path["x_%s" % feat], color="C2", lw=0.9, alpha=0.7, label="x (%s)" % feat)
        ax2.set_ylabel("feature %s" % feat, color="C2")
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="best")
    finish(fig, ax, "Backtest variables — %s  %s ~ %s" % (
        name, spec.target, "+".join(spec.features)), "structure (bp)",
           str(vars_dir / ("backtest_vars_%s.png" % name)))
    return written


def run_chosen_predictions(full_data, predicted_data, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE):
    return format_prediction_table(pd.DataFrame(
        [fit_predict_next(full_data, predicted_data, s, train_size=train_size) for s in specs]
    ))


def append_zscore_latest(table, full_data):
    z_cols, extra = ["z_%d" % w for w in ZSCORE_LOOKBACKS], []
    for res in run_zscore_models(full_data, specs=ZSCORE_MODELS).values():
        panel = res.panel.dropna(subset=["z_%d" % min(ZSCORE_LOOKBACKS)])
        if panel.empty:
            continue
        last, t_prev = panel.iloc[-1], panel.index[-1]
        t_next = pd.Timestamp(t_prev) + pd.Timedelta(minutes=int(DATA_BAR_MINUTES))
        z50 = _round3(last["z_50"]) if "z_50" in last.index else _NAN
        # z>0 → value above fair → REC; z<0 → PAY (same menor/mayor convention on z)
        row = {
            "model": res.spec.param, "structure": res.spec.series, "features": "",
            "side": _side_from_residuo(z50),
            "predicted_level": _NAN, "time": t_next, "y_pred_prev": _NAN,
            "y_realized_prev": _round3(last["value"]),
            "residuo": z50, "residuo_std": _NAN,
            "residual_prev": z50,
            "equation": "", "n_lookback": "", "asof_prev": t_prev,
        }
        for c in z_cols:
            if c in last.index:
                row[c] = _round3(last[c]) if pd.notna(last[c]) else _NAN
        extra.append(row)
    if not extra:
        return format_prediction_table(table)
    return format_prediction_table(pd.concat([table, pd.DataFrame(extra)], ignore_index=True))


def build_summary(reg_results, z_results):
    rows = []
    for res in reg_results.values():
        preds = res.predictions.dropna(subset=["y_pred"])
        if preds.empty:
            continue
        last = preds.iloc[-1]
        rows.append({
            "model": res.spec.name, "structure": res.spec.target,
            "nivel": float(last["y_true"]), "predicted": float(last["y_pred"]),
            COL_RZ: float(last["residual"]), "equation": _equation_from_row(res.spec, last),
            "time": preds.index[-1],
        })
    for res in z_results.values():
        panel = res.panel.dropna(subset=["fair"])
        if panel.empty:
            continue
        last = panel.iloc[-1]
        z50 = last["z_50"] if "z_50" in last.index else _NAN
        rows.append({
            "model": res.spec.param, "structure": res.spec.series,
            "nivel": float(last["value"]), "predicted": _NAN,
            COL_RZ: float(z50) if pd.notna(z50) else _NAN, "equation": "",
            "time": panel.index[-1],
        })
    return pd.DataFrame(rows)[_SUMMARY_VIEW] if rows else pd.DataFrame(columns=_SUMMARY_VIEW)


def format_summary_text(summary, asof=None):
    show = summary.copy()
    for c in ("nivel", "predicted", COL_RZ):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").apply(
                lambda v: "" if pd.isna(v) else "%.3f" % v
            )
    show = show[[c for c in _SUMMARY_VIEW if c in show.columns]]
    header = "SOFR screener — results" + ("  |  asof %s" % asof if asof else "")
    return header + "\n\n" + show.to_string(index=False) + "\n"


def _cfg(name, default=""):
    return os.environ.get(name, default) or default


def send_summary_email(
    summary,
    asof=None,
    subject=None,
    to=None,
    body=None,
    html_body=None,
    inline_images=None,
    attach_paths=None,
):
    """
    Send email. Prefer html_body + inline_images (CID) for visual prediction mails.
    attach_paths only if explicitly requested (CSV/PNG attachments off by default).
    """
    enabled = os.environ.get("EMAIL_ENABLED", str(EMAIL_ENABLED)).lower() in {"1", "true", "yes"}
    host, port = _cfg("SMTP_HOST", SMTP_HOST), int(_cfg("SMTP_PORT", str(SMTP_PORT)) or 587)
    user, password = _cfg("SMTP_USER", SMTP_USER), _cfg("SMTP_PASSWORD", SMTP_PASSWORD)
    mail_from = _cfg("REPORT_FROM", REPORT_FROM) or user
    mail_to = list(to) if to else list(REPORT_TO)
    env_to = _cfg("REPORT_TO")
    if env_to:
        mail_to = [x.strip() for x in env_to.split(",") if x.strip()]
    use_tls = os.environ.get("SMTP_USE_TLS", str(SMTP_USE_TLS)).lower() in {"1", "true", "yes"}
    text = body if body is not None else format_summary_text(summary, asof=asof)
    if not enabled or not host or not mail_to or not mail_from:
        if body is None and html_body is None:
            print(text, flush=True)
        missing = [k for k, ok in (
            ("EMAIL_ENABLED", enabled), ("SMTP_HOST", host),
            ("REPORT_TO", mail_to), ("REPORT_FROM/SMTP_USER", mail_from),
        ) if not ok]
        print("[email skipped] fill %s in curve_config.py (or env), then re-run" % ", ".join(missing),
              flush=True)
        return False

    inline = list(inline_images or [])
    root = MIMEMultipart("related") if (html_body and inline) else MIMEMultipart("mixed")
    root["Subject"] = subject or ("SOFR prediction %s" % (asof or "")).strip()
    root["From"], root["To"] = mail_from, ", ".join(mail_to)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain", "utf-8"))
    if html_body:
        alt.attach(MIMEText(html_body, "html", "utf-8"))
    root.attach(alt)

    for i, p in enumerate(inline):
        fp = Path(p)
        if not fp.exists():
            continue
        img = MIMEImage(fp.read_bytes())
        cid = "graph%d" % i
        img.add_header("Content-ID", "<%s>" % cid)
        img.add_header("Content-Disposition", "inline", filename=fp.name)
        root.attach(img)

    paths = list(attach_paths or [])
    if not paths and not html_body:
        if EMAIL_ATTACH_CSV and Path("predictions_latest.csv").exists():
            paths.append("predictions_latest.csv")
        if EMAIL_ATTACH_GRAPHS:
            evo = Path(GRAPH_DIR_EVOLUTION)
            if evo.exists():
                paths.extend(str(p) for p in sorted(evo.glob("*.png")))
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        part = MIMEApplication(fp.read_bytes(), Name=fp.name)
        part["Content-Disposition"] = 'attachment; filename="%s"' % fp.name
        root.attach(part)

    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            if use_tls:
                smtp.starttls()
            if user:
                smtp.login(user, password)
            smtp.sendmail(mail_from, mail_to, root.as_string())
        print("[email sent] to=%s  subject=%r  inline=%d"
              % (", ".join(mail_to), root["Subject"], len(inline)), flush=True)
        return True
    except Exception as exc:
        print("[email FAILED] %s" % exc, flush=True)
        return False
