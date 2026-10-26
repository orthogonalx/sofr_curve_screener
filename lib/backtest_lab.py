#!/usr/bin/env python3
"""
Minimal residual mean-reversion backtest.

  1. Walk-forward OLS: y ~ features
  2. z = rolling z-score of residual
  3. Enter when |z| >= ENTRY_Z  (long if z<0, short if z>0)
  4. Hold one bar; pnl = pos_{t-1} * Δy

  python -c "from lib.backtest_lab import run_simple_backtest; ..."
  # or run_backtest.ipynb
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from curve_config import DATA_BAR_MINUTES, TRAIN_SIZE, RegressionSpec
from lib.data import build_panel, load_panel
from lib.models import _rolling_z, rolling_ols

PathLike = Union[str, Path]


def load_panel_for_backtest(
    source: str = "live",
    csv_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    source='live' → load_panel + build_panel (SOFR structures + cross assets)
    source='csv'  → read CSV/Excel (DatetimeIndex, any columns)
    """
    src = (source or "live").strip().lower()
    if src in ("csv", "excel", "file", "xlsx"):
        if not csv_path:
            raise ValueError("csv_path required when source=%r" % source)
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, index_col=0)
        else:
            df = pd.read_csv(path, index_col=0)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "time"
        return df.apply(pd.to_numeric, errors="coerce").sort_index()

    raw, tag = load_panel()
    _, full, _ = build_panel(raw)
    print("  [lab] live panel src=%s  n=%d  asof=%s" % (tag, len(full), full.index[-1]), flush=True)
    return full


def run_simple_backtest(
    panel: pd.DataFrame,
    target: str,
    features: Sequence[str],
    entry_z: float = 1.0,
    z_lookback: int = 50,
    train_size: int = None,
    start=None,
    end=None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simple OOS residual z-entry backtest.

    Parameters
    ----------
    panel : DataFrame with DatetimeIndex; must contain target + features
    target, features : column names
    entry_z : enter when |z_resid| >= this (e.g. 1.0 = 1 std)
    z_lookback : bars for residual z-score
    train_size : walk-forward train window (default TRAIN_SIZE)
    start, end : optional date filters

    Returns
    -------
    path : bar-by-bar y_true, y_pred, residual, z, signal, position, pnl, cum_pnl
    summary : fit + strategy scorecard dict
    """
    df = panel.copy()
    if start is not None and str(start).strip():
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None and str(end).strip():
        df = df.loc[df.index <= pd.Timestamp(end)]

    need = [target] + list(features)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError("Missing columns %s — have %s" % (missing, list(df.columns)[:40]))

    k = int(train_size if train_size is not None else TRAIN_SIZE)
    spec = RegressionSpec(target=target, features=tuple(features))
    # predicted_data = same panel (target lives there)
    res = rolling_ols(df, df, spec, train_size=k)
    if res.predictions.empty:
        raise ValueError("No OOS predictions — need more history (train_size=%d)" % k)

    path = res.predictions[["y_true", "y_pred", "residual"]].copy()
    path.index.name = "time"
    z = _rolling_z(path["residual"], int(z_lookback))
    path["z_resid"] = z
    thr = float(entry_z)
    # mean-reversion: buy cheap residual (z<<0), sell rich (z>>0)
    path["signal"] = pd.Series(
        np.select([(z <= -thr), (z >= thr)], [1.0, -1.0], default=0.0),
        index=path.index,
    )
    path["position"] = path["signal"].astype(float)
    path["position_lag"] = path["position"].shift(1)
    path["dy"] = path["y_true"].diff()
    path["pnl"] = path["position_lag"] * path["dy"]
    path["cum_pnl"] = path["pnl"].fillna(0.0).cumsum()

    active = path.loc[path["position_lag"].fillna(0.0) != 0.0, "pnl"].dropna()
    wins = active[active > 0]
    losses = active[active < 0]
    std = float(active.std(ddof=1)) if len(active) > 1 else float("nan")
    mean_pnl = float(active.mean()) if len(active) else float("nan")
    sharpe = float("nan")
    if std == std and abs(std) > 1e-12:
        bars_per_year = max((365.0 * 24.0 * 60.0) / float(DATA_BAR_MINUTES) * (5.0 / 7.0), 1.0)
        sharpe = float((mean_pnl / std) * np.sqrt(bars_per_year))
    gross_loss = float((-losses).sum()) if len(losses) else 0.0
    pf = (
        float(wins.sum() / gross_loss) if gross_loss > 1e-12
        else (float("nan") if len(wins) == 0 else float("inf"))
    )
    if pf == float("inf"):
        pf = float("nan")

    ov = res.overall or {}
    summary = {
        "target": target,
        "features": "+".join(features),
        "entry_z": thr,
        "z_lookback": int(z_lookback),
        "train_size": k,
        "oos_r2": float(ov.get("r2", float("nan"))),
        "oos_rmse": float(ov.get("rmse", float("nan"))),
        "n_folds": int(ov.get("n_folds", 0)),
        "n_bars": int(len(path)),
        "n_trades": int((path["position"] != 0).sum()),
        "total_pnl": float(active.sum()) if len(active) else float("nan"),
        "mean_pnl": mean_pnl,
        "hit_rate": float((active > 0).mean()) if len(active) else float("nan"),
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_dd": float((path["cum_pnl"] - path["cum_pnl"].cummax()).min()) if len(path) else float("nan"),
    }
    print(
        "  [lab] %s ~ %s  entry_z=%.2f  oos_r2=%.3f  trades=%s  pnl=%.3f  hit=%.3f  sharpe=%.2f"
        % (
            target, "+".join(features), thr,
            summary["oos_r2"], summary["n_trades"], summary["total_pnl"],
            summary["hit_rate"], summary["sharpe"],
        ),
        flush=True,
    )
    return path, summary


def save_lab_outputs(
    path: pd.DataFrame,
    summary: Dict,
    out_dir: PathLike = "exports/backtest",
    name: str = None,
) -> List[Path]:
    """Write path CSV + summary CSV under exports/backtest/."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tag = name or ("%s__%s" % (summary.get("target", "y"), summary.get("features", "x")))
    tag = re_sub_safe(tag)
    paths = []
    p_csv = dest / ("path_%s.csv" % tag)
    path.to_csv(p_csv)
    paths.append(p_csv.resolve())
    s_csv = dest / ("summary_%s.csv" % tag)
    pd.DataFrame([summary]).to_csv(s_csv, index=False)
    paths.append(s_csv.resolve())
    # quick cum-pnl plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(path.index, path["cum_pnl"], color="#1f77d0", lw=1.2)
        ax.set_title("cum PnL | entry_z=%s | %s" % (summary.get("entry_z"), tag), fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png = dest / ("cum_pnl_%s.png" % tag)
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(png.resolve())
    except ImportError:
        pass
    for p in paths:
        print("  [lab] wrote %s" % p, flush=True)
    return paths


def re_sub_safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))[:80]
