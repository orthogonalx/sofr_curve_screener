#!/usr/bin/env python3
"""
Reproduce fold-0 of x7 (or any spec) and dump train/test CSVs for Excel LINEST.

  python -c "from lib.backtest_verify import verify_fold0; verify_fold0()"
  # or run_backtest_verify.ipynb
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from curve_config import (
    ADD_INTERCEPT,
    BACKTEST_END,
    BACKTEST_START,
    TRAIN_SIZE,
    RegressionSpec,
)
from lib.data import build_panel, load_panel
from lib.models import _aligned_xy, _coef_dict, _fit, _predict, _resolved, rolling_ols


def _slice_period(full, pred, start=None, end=None):
    if start is not None and str(start).strip():
        t0 = pd.Timestamp(start)
        full = full.loc[full.index >= t0]
        pred = pred.loc[pred.index >= t0]
    if end is not None and str(end).strip():
        t1 = pd.Timestamp(end)
        full = full.loc[full.index <= t1]
        pred = pred.loc[pred.index <= t1]
    return full, pred


def verify_fold0(
    target: str = "5s7s10s",
    features: Sequence[str] = ("2s5s10s",),
    train_size: int = None,
    start=None,
    end=None,
    out_dir: str = "exports/backtest",
    compare_csv: Optional[str] = "backtest_x7.csv",
) -> dict:
    """
    Rebuild the FIRST walk-forward fold exactly like models.rolling_ols.

    Writes:
      exports/backtest/verify_train_fold0.csv  — 160 rows for Excel LINEST
      exports/backtest/verify_test_fold0.csv   — the 1 OOS bar
      exports/backtest/verify_summary.csv      — coefs + y_pred + residual
    """
    k = int(train_size if train_size is not None else TRAIN_SIZE)
    start = BACKTEST_START if start is None else start
    end = BACKTEST_END if end is None else end

    raw, src = load_panel()
    _, full, pred = build_panel(raw)
    full, pred = _slice_period(full, pred, start, end)
    print("  [verify] src=%s  n_full=%d  period=%s→%s" % (
        src, len(full), start, end or "end"), flush=True)

    spec = _resolved(RegressionSpec(target=target, features=tuple(features), name="verify"))
    X_all, y_all = _aligned_xy(full, pred, spec)
    n = len(y_all)
    print("  [verify] aligned clean rows=%d  train_size=%d" % (n, k), flush=True)
    if n < k + 1:
        raise ValueError("Need ≥%d clean rows; have %d" % (k + 1, n))

    # Fold 0 — identical slices to rolling_ols
    tr = slice(0, k)
    te = slice(k, k + 1)
    X_tr = X_all.iloc[tr].to_numpy(float)
    y_tr = y_all.iloc[tr].to_numpy(float)
    X_te = X_all.iloc[te].to_numpy(float)
    y_te = y_all.iloc[te].to_numpy(float)
    beta = _fit(X_tr, y_tr, ADD_INTERCEPT)
    y_hat = float(_predict(X_te, beta, ADD_INTERCEPT)[0])
    coefs = _coef_dict(spec.features, beta, ADD_INTERCEPT)
    resid = float(y_te[0] - y_hat)

    train = pd.DataFrame({"y_%s" % target: y_all.iloc[tr].to_numpy(float)})
    for i, f in enumerate(spec.features):
        train["x_%s" % f] = X_tr[:, i]
    train.index = y_all.index[tr]
    train.index.name = "time"
    train.insert(0, "i", np.arange(1, k + 1))

    test = pd.DataFrame({
        "time": [y_all.index[te][0]],
        "y_true": [float(y_te[0])],
        "y_pred": [y_hat],
        "residual": [resid],
        **{kk: [vv] for kk, vv in coefs.items()},
        **{"x_%s" % f: [float(X_te[0, i])] for i, f in enumerate(spec.features)},
        "train_start": [y_all.index[tr][0]],
        "train_end": [y_all.index[tr][-1]],
        "n_train": [k],
    })

    # Cross-check vs rolling_ols fold 0 (allow small k for audits)
    res = rolling_ols(full, pred, spec, train_size=k, min_train_obs=min(k, 1))
    if res.predictions.empty:
        raise ValueError("rolling_ols returned no predictions — check train_size / data")
    row0 = res.predictions.iloc[0]
    fold0 = res.fold_metrics.iloc[0]
    match_pred = abs(float(row0["y_pred"]) - y_hat) < 1e-8
    match_beta = abs(float(fold0.get("intercept", 0)) - float(coefs.get("intercept", 0))) < 1e-8

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    train_path = dest / "verify_train_fold0.csv"
    test_path = dest / "verify_test_fold0.csv"
    sum_path = dest / "verify_summary.csv"
    train.to_csv(train_path)
    test.to_csv(test_path, index=False)

    summary = {
        "target": target,
        "features": "+".join(spec.features),
        "train_size": k,
        "aligned_n": n,
        "period_start": str(start),
        "train_start": str(y_all.index[tr][0]),
        "train_end": str(y_all.index[tr][-1]),
        "test_time": str(y_all.index[te][0]),
        "y_true": float(y_te[0]),
        "y_pred": y_hat,
        "residual": resid,
        **coefs,
        "matches_rolling_ols_pred": match_pred,
        "matches_rolling_ols_intercept": match_beta,
        "rolling_ols_y_pred": float(row0["y_pred"]),
        "rolling_ols_residual": float(row0["residual"]),
        "excel_tip": (
            "LINEST on verify_train_fold0.csv: y col = y_%s, x col = x_%s. "
            "Excel LINEST returns [slope, intercept] in that order (slope first!)."
            % (target, spec.features[0])
        ),
    }
    pd.DataFrame([summary]).to_csv(sum_path, index=False)

    print("  [verify] train %s → %s  (n=%d)" % (train.index[0], train.index[-1], k), flush=True)
    print("  [verify] test  %s" % y_all.index[te][0], flush=True)
    print("  [verify] intercept=%.6f  beta=%.6f" % (
        coefs.get("intercept", float("nan")),
        coefs.get("beta_%s" % spec.features[0], float("nan")),
    ), flush=True)
    print("  [verify] y_true=%.6f  y_pred=%.6f  resid=%.6f" % (
        float(y_te[0]), y_hat, resid), flush=True)
    print("  [verify] matches rolling_ols pred=%s intercept=%s" % (match_pred, match_beta), flush=True)
    print("  [verify] wrote %s" % train_path.resolve(), flush=True)
    print("  [verify] wrote %s" % test_path.resolve(), flush=True)
    print("  [verify] wrote %s" % sum_path.resolve(), flush=True)

    # Compare to backtest_x7.csv first row if present
    if compare_csv:
        p = Path(compare_csv)
        if p.exists():
            bt = pd.read_csv(p)
            if "time" in bt.columns:
                bt["time"] = pd.to_datetime(bt["time"])
                bt = bt.set_index("time")
            r0 = bt.iloc[0]
            print("\n  [verify] vs %s row0:" % p.name, flush=True)
            print("    csv time     = %s" % bt.index[0], flush=True)
            print("    verify time  = %s" % y_all.index[te][0], flush=True)
            for col in ("y_true", "y_pred", "residual", "intercept"):
                if col in r0.index:
                    print(
                        "    %-10s  csv=%s  verify=%s  diff=%s"
                        % (
                            col,
                            r0[col],
                            summary.get(col, summary.get(col)),
                            float(r0[col]) - float(summary.get(col, float("nan"))),
                        ),
                        flush=True,
                    )
            bcol = "beta_%s" % spec.features[0]
            if bcol in r0.index:
                print(
                    "    %-10s  csv=%s  verify=%s  diff=%s"
                    % (bcol, r0[bcol], coefs[bcol], float(r0[bcol]) - float(coefs[bcol])),
                    flush=True,
                )
            if pd.Timestamp(bt.index[0]) != pd.Timestamp(y_all.index[te][0]):
                print(
                    "  [verify] WARNING: first CSV time ≠ fold0 test time. "
                    "Often means BACKTEST_START / clean-align mismatch vs how you picked 160 rows.",
                    flush=True,
                )
        else:
            print("  [verify] no %s yet — run: python run_dz.py backtest" % compare_csv, flush=True)

    return {
        "train": train,
        "test": test,
        "summary": summary,
        "paths": (train_path.resolve(), test_path.resolve(), sum_path.resolve()),
    }


def main():
    verify_fold0()


if __name__ == "__main__":
    main()
