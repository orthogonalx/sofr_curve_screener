# =============================================================================
# backtest.py  — Residual mean-reversion strategy on locked CHOSEN_MODELS
# =============================================================================
# Pipeline per model:
#   1) Walk-forward OLS → y_true, y_pred, residual (= y_true − y_pred) each OOS bar
#   2) z-score of residual using only PAST bars (lookback L)
#   3) Signal from z + |residual| thresholds (mean-reversion)
#   4) Position mode (curve_config.BACKTEST_POSITION_MODE):
#        "one_bar" — each signal bar is its own 1-period trade (close next bar);
#                    |pos| ∈ {0,1}; n_trades = n_long + n_short
#        "hold"    — stay in ±1 while the signal persists (regime);
#                    n_entries = # regime entries; n_long/n_short = bars in state
#   5) PnL_t = position_{t-1} * (y_t − y_{t-1})
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from curve_config import (
    BACKTEST_END,
    BACKTEST_POSITION_MODE,
    BACKTEST_RESIDUAL_THR_BP,
    BACKTEST_START,
    BACKTEST_Z_LOOKBACK,
    BACKTEST_ZSCORE_THR,
    CHOSEN_MODELS,
    TRAIN_SIZE,
    RegressionSpec,
)
from rolling_model import instance_predictions
from screener_core import run_chosen_backtest


@dataclass
class StrategyResult:
    spec: RegressionSpec
    path: pd.DataFrame
    summary: dict
    position_mode: str


def residual_zscore(residual: pd.Series, lookback: int) -> pd.Series:
    """z_t = (r_t − μ_{t-L:t-1}) / σ_{t-L:t-1}  (past only)."""
    mu = residual.rolling(lookback, min_periods=lookback).mean().shift(1)
    sigma = residual.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
    return (residual - mu) / sigma.replace(0.0, np.nan)


def signal_from_residual(
    residual: pd.Series,
    z: pd.Series,
    residual_thr_bp: float,
    z_thr: float,
) -> pd.Series:
    """
    Raw signal (before position mode):
      +1 if z <= −z_thr and |r| >= thr
      −1 if z >= +z_thr and |r| >= thr
       0 otherwise
    """
    long_mask = (z <= -z_thr) & (residual.abs() >= residual_thr_bp)
    short_mask = (z >= z_thr) & (residual.abs() >= residual_thr_bp)
    return pd.Series(
        np.select([long_mask, short_mask], [1.0, -1.0], default=0.0),
        index=residual.index,
    )


def apply_position_mode(signal: pd.Series, mode: str) -> pd.Series:
    """
    Map signal → position.

    one_bar: each bar's signal is a standalone ±1/0 trade (never stacks beyond ±1).
             Economically: enter at t, exposure for the next bar's Δy, then done;
             if signal is on again at t+1, that is a *new* one-bar trade.

    hold:    same ±1/0 path while signal persists (stay long/short across bars).
             Still never stacks to ±2; "accumulate" = time in the trade, not size.
    """
    mode = (mode or "one_bar").strip().lower()
    if mode not in ("one_bar", "hold"):
        raise ValueError(
            f"BACKTEST_POSITION_MODE must be 'one_bar' or 'hold', got {mode!r}"
        )
    # Path is identical (±1/0 from signal); modes differ in trade accounting.
    # |position| never exceeds 1 in either mode.
    return signal.astype(float).clip(-1.0, 1.0)


def simulate_pnl(y_true: pd.Series, position: pd.Series) -> pd.DataFrame:
    """PnL_t = position_{t-1} * (y_t − y_{t-1})."""
    dy = y_true.diff()
    pos_lag = position.shift(1)
    pnl = pos_lag * dy
    out = pd.DataFrame(
        {
            "y_true": y_true,
            "dy": dy,
            "position": position,
            "position_lag": pos_lag,
            "pnl": pnl,
        },
        index=y_true.index,
    )
    out["cum_pnl"] = pnl.fillna(0.0).cumsum()
    return out


def _trade_stats(position: pd.Series, mode: str) -> dict:
    pos = position.fillna(0.0)
    n_long = int((pos == 1).sum())
    n_short = int((pos == -1).sum())
    n_flat = int((pos == 0).sum())
    # regime entry: move into a non-flat state or flip long↔short
    entries = int(((pos != 0) & (pos.shift(1).fillna(0) != pos)).sum())

    if mode == "one_bar":
        # every signal bar is one trade; closes next period
        n_trades = n_long + n_short
        n_entries = n_trades
    else:
        # hold: one trade per contiguous regime
        n_trades = entries
        n_entries = entries

    return {
        "n_long": n_long,
        "n_short": n_short,
        "n_flat": n_flat,
        "n_entries": n_entries,
        "n_trades": n_trades,
    }


def run_residual_strategy(
    preds: pd.DataFrame,
    spec: RegressionSpec,
    residual_thr_bp: float = BACKTEST_RESIDUAL_THR_BP,
    z_thr: float = BACKTEST_ZSCORE_THR,
    z_lookback: int = BACKTEST_Z_LOOKBACK,
    position_mode: str = BACKTEST_POSITION_MODE,
    feature_panel: Optional[pd.DataFrame] = None,
) -> StrategyResult:
    """Build signal → position (mode) → pnl path + summary."""
    mode = (position_mode or "one_bar").strip().lower()

    base_cols = ["y_true", "y_pred", "residual"]
    coef_cols = [
        c
        for c in preds.columns
        if c in ("fold", "r2", "intercept") or c.startswith("beta_")
    ]
    path = preds[base_cols + coef_cols].copy()

    if feature_panel is not None:
        for feat in spec.features:
            if feat in feature_panel.columns:
                path[f"x_{feat}"] = feature_panel[feat].reindex(path.index)

    if "intercept" in path.columns and all(
        f"beta_{f}" in path.columns and f"x_{f}" in path.columns for f in spec.features
    ):
        y_check = path["intercept"].astype(float)
        for f in spec.features:
            y_check = y_check + path[f"beta_{f}"].astype(float) * path[f"x_{f}"].astype(
                float
            )
        path["y_pred_from_coefs"] = y_check
        path["y_pred_check_ok"] = (path["y_pred"] - path["y_pred_from_coefs"]).abs() < 1e-6

    path["z_resid"] = residual_zscore(path["residual"], z_lookback)
    path["signal"] = signal_from_residual(
        path["residual"], path["z_resid"], residual_thr_bp, z_thr
    )
    path["position"] = apply_position_mode(path["signal"], mode)
    path["position_mode"] = mode

    pnl_frame = simulate_pnl(path["y_true"], path["position"])
    path = path.join(pnl_frame[["dy", "position_lag", "pnl", "cum_pnl"]])

    traded = path["pnl"].dropna()
    stats = _trade_stats(path["position"], mode)
    summary = {
        "model": spec.name,
        "structure": spec.target,
        "features": "+".join(spec.features),
        "position_mode": mode,
        "n_bars": int(len(path)),
        **stats,
        "total_pnl": float(traded.sum()) if len(traded) else float("nan"),
        "mean_pnl": float(traded.mean()) if len(traded) else float("nan"),
        "hit_rate": float((traded > 0).mean()) if len(traded) else float("nan"),
        "max_dd": float((path["cum_pnl"] - path["cum_pnl"].cummax()).min())
        if len(path)
        else float("nan"),
    }
    return StrategyResult(
        spec=spec, path=path, summary=summary, position_mode=mode
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
        full_data,
        predicted_data,
        specs=specs,
        train_size=train_size,
        start=start,
        end=end,
    )
    out: Dict[str, StrategyResult] = {}
    for name, res in wf.items():
        preds = instance_predictions(res)
        if preds.empty:
            continue
        out[name] = run_residual_strategy(
            preds,
            res.spec,
            residual_thr_bp=residual_thr_bp,
            z_thr=z_thr,
            z_lookback=z_lookback,
            position_mode=position_mode,
            feature_panel=full_data,
        )
    return out


def summary_table(results: Dict[str, StrategyResult]) -> pd.DataFrame:
    rows: List[dict] = [r.summary for r in results.values()]
    if not rows:
        return pd.DataFrame()
    cols = [
        "model",
        "structure",
        "features",
        "position_mode",
        "n_bars",
        "n_trades",
        "n_entries",
        "n_long",
        "n_short",
        "n_flat",
        "total_pnl",
        "mean_pnl",
        "hit_rate",
        "max_dd",
    ]
    return pd.DataFrame(rows)[cols]


def _try_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_backtest_pnl(
    result: StrategyResult,
    outfile: Optional[str] = None,
) -> Optional[str]:
    """Cumulative PnL path. Returns saved path or None if matplotlib missing."""
    plt = _try_pyplot()
    if plt is None:
        print(f"  skip pnl plot ({result.spec.name}): matplotlib not installed", flush=True)
        return None

    name = result.spec.name
    outfile = outfile or f"backtest_pnl_{name}.png"
    path = result.path
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.plot(path.index, path["cum_pnl"], color="C0", lw=1.2, label="cum_pnl")
    ax.set_title(
        f"Backtest cum PnL — {name}  ({result.spec.target})  "
        f"mode={result.position_mode}"
    )
    ax.set_ylabel("cum PnL (bp × position)")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def plot_backtest_variables(
    result: StrategyResult,
    outfile: Optional[str] = None,
) -> Optional[str]:
    """
    Plot the two main series: y_true (structure) and y_pred (model fair).
    If a single feature exists as x_<feat>, add it on a twin axis.
    """
    plt = _try_pyplot()
    if plt is None:
        print(
            f"  skip vars plot ({result.spec.name}): matplotlib not installed",
            flush=True,
        )
        return None

    name = result.spec.name
    outfile = outfile or f"backtest_vars_{name}.png"
    path = result.path
    feat = result.spec.features[0] if len(result.spec.features) == 1 else None
    xcol = f"x_{feat}" if feat and f"x_{feat}" in path.columns else None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        path.index,
        path["y_true"],
        color="C0",
        lw=1.0,
        label=f"y_true ({result.spec.target})",
    )
    ax.plot(path.index, path["y_pred"], color="C1", lw=1.0, label="y_pred", alpha=0.85)
    ax.set_ylabel("structure (bp)")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)

    if xcol is not None:
        ax2 = ax.twinx()
        ax2.plot(
            path.index,
            path[xcol],
            color="C2",
            lw=0.9,
            alpha=0.7,
            label=f"x ({feat})",
        )
        ax2.set_ylabel(f"feature {feat}", color="C2")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

    ax.set_title(
        f"Backtest variables — {name}  "
        f"{result.spec.target} ~ {'+'.join(result.spec.features)}"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def save_backtest_plots(result: StrategyResult) -> List[str]:
    """Save PnL + variables plots; return list of written paths."""
    written: List[str] = []
    for fn in (plot_backtest_pnl, plot_backtest_variables):
        path = fn(result)
        if path:
            written.append(path)
    return written
