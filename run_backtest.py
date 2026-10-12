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
            f"  {name}  trades={s['n_trades']}  "
            f"pnl={s['total_pnl']:.4f}  hit={s['hit_rate']:.3f}  dd={s['max_dd']:.4f}",
            flush=True,
        )
        for p in save_backtest_plots(res):
            print(f"    wrote {p}", flush=True)

    summary = summary_table(results)
    for c in ("total_pnl", "mean_pnl", "hit_rate", "max_dd"):
        if c in summary.columns:
            summary[c] = summary[c].round(4)
    print(summary.to_string(index=False), flush=True)
    Path("backtest_summary.csv").write_text(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
