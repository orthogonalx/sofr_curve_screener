#!/usr/bin/env python3
"""
Backtest — residual mean-reversion strategy on CHOSEN_MODELS.

How it works
------------
1. Walk-forward regression (fixed k=TRAIN_SIZE) over [BACKTEST_START, BACKTEST_END]
   → each OOS bar: y_true, y_pred, residual r = y_true − y_pred (bp).

2. Z-score of the residual using only PAST bars (lookback BACKTEST_Z_LOOKBACK):
       z_t = (r_t − mean(r_{t-L:t-1})) / std(r_{t-L:t-1})

3. Signal (mean-reversion):
       long  (+1) if z <= −BACKTEST_ZSCORE_THR  AND  |r| >= BACKTEST_RESIDUAL_THR_BP
       short (−1) if z >= +BACKTEST_ZSCORE_THR  AND  |r| >= BACKTEST_RESIDUAL_THR_BP
       else flat (0)
   Position size never stacks beyond ±1.

4. Position mode (BACKTEST_POSITION_MODE in curve_config):
       "one_bar" (default)
           Each signal bar is its own trade: enter → PnL on next bar's Δy → close.
           If the signal is still on next bar, that opens a *new* one-bar trade.
           ⇒ n_trades = n_long + n_short  (= # signal bars)

       "hold"
           Stay long/short while the signal persists (same as before).
           ⇒ n_entries / n_trades = # times you enter or flip a regime
           ⇒ n_long / n_short = bars spent in each state

5. PnL on bar t (no look-ahead):
       PnL_t = position_{t-1} * (y_t − y_{t-1})

  python3.8 -u run_backtest.py
"""

from __future__ import annotations

from pathlib import Path

from backtest import backtest_chosen_models, save_backtest_plots, summary_table
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
from screener_core import build_panel, load_panel

_W = 72


def _rule(char: str = "─") -> None:
    print(char * _W, flush=True)


def _blank() -> None:
    print(flush=True)


def _section(title: str, note: str = "") -> None:
    _blank()
    _rule("═")
    print(f"  {title}", flush=True)
    if note:
        print(f"  {note}", flush=True)
    _rule("═")
    _blank()


def main() -> None:
    mode = (BACKTEST_POSITION_MODE or "one_bar").strip().lower()

    _section(
        "SOFR backtest  —  residual mean-reversion",
        f"position_mode={mode!r}  ·  PnL = pos[t-1] * Δy[t]  ·  |pos| ≤ 1",
    )

    data_raw, src = load_panel()
    _, full_data, predicted_data = build_panel(data_raw)

    print(f"  src              : {src}", flush=True)
    print(f"  n / asof         : {len(data_raw)}  /  {data_raw.index[-1]}", flush=True)
    print(f"  period           : {BACKTEST_START} → {BACKTEST_END or 'end'}", flush=True)
    print(f"  walk-forward k   : {TRAIN_SIZE}", flush=True)
    print(f"  models           : {[m.name for m in CHOSEN_MODELS]}", flush=True)
    _blank()
    print("  Strategy inputs (curve_config):", flush=True)
    print(f"    residual thr   : |r| >= {BACKTEST_RESIDUAL_THR_BP} bp", flush=True)
    print(f"    z-score thr    : |z| >= {BACKTEST_ZSCORE_THR}", flush=True)
    print(f"    z lookback L   : {BACKTEST_Z_LOOKBACK} past bars", flush=True)
    print(f"    position mode  : {mode}", flush=True)
    _blank()
    print("  Signal rules:", flush=True)
    print("    r = y_true − y_pred", flush=True)
    print(
        f"    long  (+1) if z(r) <= -{BACKTEST_ZSCORE_THR} and |r| >= {BACKTEST_RESIDUAL_THR_BP}",
        flush=True,
    )
    print(
        f"    short (-1) if z(r) >= +{BACKTEST_ZSCORE_THR} and |r| >= {BACKTEST_RESIDUAL_THR_BP}",
        flush=True,
    )
    print("    else flat (0)   ·  never stacks to ±2, ±3, …", flush=True)
    _blank()
    if mode == "one_bar":
        print("  Position mode one_bar (default):", flush=True)
        print("    • each signal bar = one trade", flush=True)
        print("    • exposure lasts one period (PnL on next Δy), then closed", flush=True)
        print("    • next bar can open a new trade if signal still on", flush=True)
        print("    • n_trades = n_long + n_short", flush=True)
    else:
        print("  Position mode hold:", flush=True)
        print("    • stay long/short while signal persists", flush=True)
        print("    • n_trades / n_entries = # regime entries (not bars)", flush=True)
        print("    • n_long / n_short = bars spent in each state", flush=True)
    print("    PnL_t = position_{t-1} * (y_t − y_{t-1})", flush=True)

    _section(
        "[1]  Walk-forward + strategy",
        "Fit rolling OLS → residual → z → signal → position → PnL",
    )

    results = backtest_chosen_models(
        full_data,
        predicted_data,
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
        print("  (no models produced a path — check CHOSEN_MODELS / data)", flush=True)
        return

    for name, res in results.items():
        path = res.path
        out = Path(f"backtest_{name}.csv")
        path.to_csv(out)
        s = res.summary
        print(f"  {name}  ({res.spec.target} ~ {'+'.join(res.spec.features)})", flush=True)
        print(
            f"    mode={s['position_mode']}  bars={s['n_bars']}  "
            f"trades={s['n_trades']}  entries={s['n_entries']}  "
            f"long/short/flat={s['n_long']}/{s['n_short']}/{s['n_flat']}",
            flush=True,
        )
        if mode == "one_bar":
            ok = s["n_trades"] == s["n_long"] + s["n_short"]
            print(
                f"    check n_trades == n_long+n_short: "
                f"{s['n_trades']} == {s['n_long']}+{s['n_short']} → {ok}",
                flush=True,
            )
        print(
            f"    total_pnl={s['total_pnl']:.4f}  mean_pnl={s['mean_pnl']:.4f}  "
            f"hit_rate={s['hit_rate']:.3f}  max_dd={s['max_dd']:.4f}",
            flush=True,
        )
        show_cols = [
            c
            for c in [
                "y_true",
                "y_pred",
                "y_pred_from_coefs",
                "residual",
                "intercept",
                *[f"beta_{f}" for f in res.spec.features],
                *[f"x_{f}" for f in res.spec.features],
                "signal",
                "position",
                "z_resid",
                "pnl",
                "cum_pnl",
            ]
            if c in path.columns
        ]
        show = path[show_cols].tail(5).round(4)
        print("    last 5 bars:", flush=True)
        print(show.to_string().replace("\n", "\n    "), flush=True)
        print(
            "    (Excel: y_pred_from_coefs = intercept + Σ beta_f * x_f)",
            flush=True,
        )
        print(f"    wrote {out}", flush=True)
        for p in save_backtest_plots(res):
            print(f"    wrote {p}", flush=True)
        _blank()

    _section(
        "[2]  Summary table",
        "one_bar: n_trades = n_long+n_short · hold: n_trades = regime entries",
    )
    summary = summary_table(results)
    for c in ("total_pnl", "mean_pnl", "hit_rate", "max_dd"):
        if c in summary.columns:
            summary[c] = summary[c].round(4)
    print(summary.to_string(index=False), flush=True)
    Path("backtest_summary.csv").write_text(summary.to_csv(index=False))
    _blank()
    print("  wrote backtest_summary.csv", flush=True)
    _blank()
    _rule("═")
    print("  Done.", flush=True)
    _rule("═")
    _blank()


if __name__ == "__main__":
    main()
