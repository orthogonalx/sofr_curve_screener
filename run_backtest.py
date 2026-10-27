#!/usr/bin/env python3
"""Backtest residual mean-reversion on CHOSEN_MODELS.

  python run_dz.py backtest
  python run_dz.py backtest --z-thr 0 --residual-thr 0
"""

from __future__ import annotations

import argparse
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
from lib.data import (
    build_panel,
    load_panel,
    refresh_live_store,
    update_live_store_to_now,
)
from lib.models import backtest_chosen_models, save_backtest_plots, summary_table


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Backtest CHOSEN_MODELS")
    p.add_argument(
        "--z-thr", type=float, default=None,
        help="|z| entry threshold (default: BACKTEST_ZSCORE_THR from config)",
    )
    p.add_argument(
        "--residual-thr", type=float, default=None,
        help="|residual| min bp (default: BACKTEST_RESIDUAL_THR_BP)",
    )
    p.add_argument("--start", default=None, help="Backtest start (default: config)")
    p.add_argument("--end", default=None, help="Backtest end (default: config / None)")
    p.add_argument(
        "--mode", default=None,
        help="Position mode: one_bar | hold (default: config)",
    )
    p.add_argument(
        "--pull-history", action="store_true",
        help="Update sofr_live.csv to now (core SOFR tickers; merges into existing store)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Append latest bar only (does NOT fill multi-day gaps)",
    )
    args = p.parse_args(argv)

    z_thr = float(BACKTEST_ZSCORE_THR if args.z_thr is None else args.z_thr)
    r_thr = float(BACKTEST_RESIDUAL_THR_BP if args.residual_thr is None else args.residual_thr)
    start = BACKTEST_START if args.start is None else args.start
    end = BACKTEST_END if args.end is None else args.end
    mode = (args.mode or BACKTEST_POSITION_MODE or "one_bar").strip().lower()

    if args.pull_history:
        print("  [backtest] updating sofr_live.csv to now (core SOFR, merge) …", flush=True)
        try:
            update_live_store_to_now(core_only=True, lookback_days=21)
        except Exception as exc:  # noqa: BLE001
            print("  [backtest] update failed (%s) — using existing store" % exc, flush=True)
        data_raw, src = load_panel()
    elif args.refresh:
        try:
            data_raw, src = refresh_live_store(force_history=False)
        except Exception as exc:  # noqa: BLE001
            print("  [backtest] refresh failed (%s) — using existing store" % exc, flush=True)
            data_raw, src = load_panel()
    else:
        data_raw, src = load_panel()
    _, full, pred = build_panel(data_raw)
    print(
        "backtest  src=%s  n=%d  asof=%s  k=%d  period=%s→%s  "
        "mode=%s  |r|>=%g  |z|>=%g  L=%d"
        % (
            src, len(data_raw), data_raw.index[-1], TRAIN_SIZE,
            start, end or "end", mode, r_thr, z_thr, BACKTEST_Z_LOOKBACK,
        ),
        flush=True,
    )

    results = backtest_chosen_models(
        full,
        pred,
        specs=CHOSEN_MODELS,
        train_size=TRAIN_SIZE,
        start=start,
        end=end,
        residual_thr_bp=r_thr,
        z_thr=z_thr,
        z_lookback=BACKTEST_Z_LOOKBACK,
        position_mode=mode,
    )
    if not results:
        print("  (no paths — check CHOSEN_MODELS / data)", flush=True)
        return

    for name, res in results.items():
        res.path.to_csv("backtest_%s.csv" % name)
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
        for path in save_backtest_plots(res):
            print("    wrote %s" % path, flush=True)

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
