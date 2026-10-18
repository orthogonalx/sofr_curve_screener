#!/usr/bin/env python3
"""Backtest residual mean-reversion on CHOSEN_MODELS."""

from pathlib import Path

from curve_config import (
    BACKTEST_END,
    BACKTEST_POSITION_MODE,
    BACKTEST_RESIDUAL_THR_BP,
    BACKTEST_START,
    BACKTEST_Z_LOOKBACK,
    BACKTEST_ZSCORE_THR,
    CHOSEN_MODELS,
    TRAIN_SIZE,
)
from data import build_panel, load_panel
from models import backtest_chosen_models, save_backtest_plots, summary_table


def main() -> None:
    mode = (BACKTEST_POSITION_MODE or "one_bar").strip().lower()
    data_raw, src = load_panel()
    _, full, pred = build_panel(data_raw)
    print(
        f"backtest  src={src}  n={len(data_raw)}  asof={data_raw.index[-1]}  "
        f"k={TRAIN_SIZE}  period={BACKTEST_START}→{BACKTEST_END or 'end'}  "
        f"mode={mode}  |r|>={BACKTEST_RESIDUAL_THR_BP}  |z|>={BACKTEST_ZSCORE_THR}  "
        f"L={BACKTEST_Z_LOOKBACK}",
        flush=True,
    )

    results = backtest_chosen_models(
        full,
        pred,
        specs=CHOSEN_MODELS,
        train_size=TRAIN_SIZE,
        start=BACKTEST_START,
        end=BACKTEST_END,
        residual_thr_bp=BACKTEST_RESIDUAL_THR_BP,
        z_thr=BACKTEST_ZSCORE_THR,
        z_lookback=BACKTEST_Z_LOOKBACK,
        position_mode=mode,
    )
    if not results:
        print("  (no paths — check CHOSEN_MODELS / data)", flush=True)
        return

    for name, res in results.items():
        res.path.to_csv(f"backtest_{name}.csv")
        s = res.summary
        print(
            "  %s  fit: oos_r2=%.3f rmse=%.3f  |  strat: trades=%s pnl=%.3f "
            "hit=%.3f sharpe=%.2f pf=%.2f dd=%.3f"
            % (
                name,
                s.get("oos_r2", float("nan")),
                s.get("oos_rmse", float("nan")),
                s.get("n_trades"),
                s.get("total_pnl", float("nan")),
                s.get("hit_rate", float("nan")),
                s.get("sharpe", float("nan")),
                s.get("profit_factor", float("nan")),
                s.get("max_dd", float("nan")),
            ),
            flush=True,
        )
        for p in save_backtest_plots(res):
            print(f"    wrote {p}", flush=True)

    summary = summary_table(results)
    for c in (
        "oos_r2", "oos_rmse", "oos_mae", "total_pnl", "mean_pnl",
        "hit_rate", "profit_factor", "sharpe", "max_dd",
    ):
        if c in summary.columns:
            summary[c] = summary[c].round(4)
    print("\n=== scorecard (fit + strategy) ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    Path("backtest_summary.csv").write_text(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
