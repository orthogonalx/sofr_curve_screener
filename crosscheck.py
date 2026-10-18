#!/usr/bin/env python3
"""
Cross-check computed SOFR curves/flies vs Bloomberg listed structures.

BBG tickers (same units as our bp formulae):
  curve  XsYs   → USOS{X:02d}{Y:02d} Curncy     e.g. USOS0510 Curncy
  fly    XsYsZs → UF{X:02d}{Y:02d}{Z:02d} Curncy e.g. UF050710 Curncy

Writes Excel with side-by-side python | bbg columns per structure.

  python run_dz.py crosscheck
  python run_dz.py crosscheck --from-store   # use sofr_live.csv outrights; still pulls BBG structs
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import curve_config as _cfg
from curve_config import (
    BBG_END_DATE,
    BBG_INTERVAL_MINUTES,
    BBG_START_DATE,
    CURVES,
    FLIES,
)
from data import (
    _fetch_bdib_panel,
    bbg_curve_ticker,
    bbg_fly_ticker,
    build_full_data,
    clean_raw_columns,
    curve_ticker_map,
    drop_non_trading_days,
    fly_ticker_map,
    load_live_store,
    load_panel,
)

CROSSCHECK_PATH = getattr(_cfg, "CROSSCHECK_PATH", "sofr_structures_crosscheck.xlsx")
CROSSCHECK_TOL_BP = float(getattr(_cfg, "CROSSCHECK_TOL_BP", 1e-4))


def _side_by_side(python_s: pd.Series, bbg_s: pd.Series, name: str) -> pd.DataFrame:
    """Align on shared timestamps; columns {name}_python | {name}_bbg."""
    py = pd.to_numeric(python_s, errors="coerce").rename("%s_python" % name)
    bb = pd.to_numeric(bbg_s, errors="coerce").rename("%s_bbg" % name)
    both = pd.concat([py, bb], axis=1, join="inner").sort_index()
    both.index.name = "time"
    return both.dropna(how="all")


def _summary_row(name: str, ticker: str, pair: pd.DataFrame, tol: float) -> dict:
    py_col, bb_col = "%s_python" % name, "%s_bbg" % name
    if pair.empty or py_col not in pair or bb_col not in pair:
        return {
            "structure": name, "bbg_ticker": ticker, "n": 0,
            "n_both": 0, "n_match": 0, "pct_match": np.nan,
            "max_abs_diff": np.nan, "mean_abs_diff": np.nan,
            "ok": False,
        }
    sub = pair[[py_col, bb_col]].dropna()
    if sub.empty:
        return {
            "structure": name, "bbg_ticker": ticker, "n": len(pair),
            "n_both": 0, "n_match": 0, "pct_match": np.nan,
            "max_abs_diff": np.nan, "mean_abs_diff": np.nan,
            "ok": False,
        }
    diff = (sub[py_col] - sub[bb_col]).abs()
    n_match = int((diff <= tol).sum())
    n_both = len(sub)
    return {
        "structure": name,
        "bbg_ticker": ticker,
        "n": len(pair),
        "n_both": n_both,
        "n_match": n_match,
        "pct_match": 100.0 * n_match / n_both if n_both else np.nan,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "ok": bool(n_match == n_both and n_both > 0),
    }


def _write_excel(
    curves_df: pd.DataFrame,
    flies_df: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> Path:
    path = Path(path)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            if not curves_df.empty:
                curves_df.to_excel(writer, sheet_name="curves")
            if not flies_df.empty:
                flies_df.to_excel(writer, sheet_name="flies")
            summary.to_excel(writer, sheet_name="summary", index=False)
    except ImportError:
        # Fallback: three CSVs next to the requested path
        base = path.with_suffix("")
        curves_df.to_csv("%s_curves.csv" % base)
        flies_df.to_csv("%s_flies.csv" % base)
        summary.to_csv("%s_summary.csv" % base, index=False)
        print("  [crosscheck] openpyxl missing → wrote CSV trio under %s_*" % base, flush=True)
        return path
    print("  [crosscheck] wrote %s" % path.resolve(), flush=True)
    return path.resolve()


def run_crosscheck(
    *,
    from_store: bool = True,
    start_date=None,
    end_date=None,
    path: Optional[str] = None,
    tol: Optional[float] = None,
    curves: Sequence = None,
    flies: Sequence = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Compare python-built structures vs BBG USOS/UF tickers bar-by-bar.

    from_store=True  → python side from sofr_live.xlsx outrights (build_full_data)
    from_store=False → also refresh outrights from bdib for the window
    """
    tol = CROSSCHECK_TOL_BP if tol is None else float(tol)
    out_path = Path(path or CROSSCHECK_PATH)
    curves = list(curves) if curves is not None else list(CURVES)
    flies = list(flies) if flies is not None else list(FLIES)
    c_map = curve_ticker_map(curves)
    f_map = fly_ticker_map(flies)

    # ── Python side (from outrights) ──────────────────────────────────────────
    if from_store:
        raw, src = load_panel()
        print("  [crosscheck] python outrights from %s  n=%d" % (src, len(raw)), flush=True)
    else:
        print("  [crosscheck] pulling outrights via bdib…", flush=True)
        raw = _fetch_bdib_panel(start_date=start_date, end_date=end_date)
    raw = drop_non_trading_days(raw.apply(pd.to_numeric, errors="coerce"))
    cleaned = clean_raw_columns(raw)
    # Python side = always computed from outrights (ignore listed USOS/UF in store)
    full = build_full_data(
        cleaned, curves=curves, flies=flies, clean=False, force_compute=True,
    )

    # Window for BBG structure pull: overlap with python panel
    start = start_date or (full.index.min() if len(full) else BBG_START_DATE)
    end = end_date or (full.index.max() if len(full) else BBG_END_DATE)

    # ── BBG side (listed curve / fly tickers) ─────────────────────────────────
    struct_secs = list(c_map.values()) + list(f_map.values())
    print(
        "  [crosscheck] pulling %d BBG structure tickers  %s → %s  interval=%sm"
        % (len(struct_secs), start, end or "now", BBG_INTERVAL_MINUTES),
        flush=True,
    )
    for name, tkr in list(c_map.items()) + list(f_map.items()):
        print("           %s → %s" % (name, tkr), flush=True)

    bbg_raw = _fetch_bdib_panel(
        start_date=start, end_date=end, securities=struct_secs,
    )
    bbg_raw = drop_non_trading_days(bbg_raw.apply(pd.to_numeric, errors="coerce"))

    # ── Pair columns ──────────────────────────────────────────────────────────
    curve_parts: List[pd.DataFrame] = []
    fly_parts: List[pd.DataFrame] = []
    rows = []

    for name, ticker in c_map.items():
        if name not in full.columns:
            print("  [crosscheck] skip curve %s (not in python panel)" % name, flush=True)
            rows.append(_summary_row(name, ticker, pd.DataFrame(), tol))
            continue
        if ticker not in bbg_raw.columns:
            print("  [crosscheck] skip curve %s (BBG empty: %s)" % (name, ticker), flush=True)
            rows.append(_summary_row(name, ticker, pd.DataFrame(), tol))
            continue
        pair = _side_by_side(full[name], bbg_raw[ticker], name)
        curve_parts.append(pair)
        rows.append(_summary_row(name, ticker, pair, tol))

    for name, ticker in f_map.items():
        if name not in full.columns:
            print("  [crosscheck] skip fly %s (not in python panel)" % name, flush=True)
            rows.append(_summary_row(name, ticker, pd.DataFrame(), tol))
            continue
        if ticker not in bbg_raw.columns:
            print("  [crosscheck] skip fly %s (BBG empty: %s)" % (name, ticker), flush=True)
            rows.append(_summary_row(name, ticker, pd.DataFrame(), tol))
            continue
        pair = _side_by_side(full[name], bbg_raw[ticker], name)
        fly_parts.append(pair)
        rows.append(_summary_row(name, ticker, pair, tol))

    curves_df = pd.concat(curve_parts, axis=1) if curve_parts else pd.DataFrame()
    flies_df = pd.concat(fly_parts, axis=1) if fly_parts else pd.DataFrame()
    if not curves_df.empty:
        curves_df = curves_df.loc[:, ~curves_df.columns.duplicated()].sort_index()
        curves_df.index.name = "time"
    if not flies_df.empty:
        flies_df = flies_df.loc[:, ~flies_df.columns.duplicated()].sort_index()
        flies_df.index.name = "time"

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["ok", "structure"], ascending=[False, True])

    dest = _write_excel(curves_df, flies_df, summary, out_path)

    # Console rollup
    if not summary.empty:
        n_ok = int(summary["ok"].fillna(False).sum())
        print(
            "  [crosscheck] summary: %d/%d structures exact (tol=%.1e bp)"
            % (n_ok, len(summary), tol),
            flush=True,
        )
        bad = summary.loc[~summary["ok"].fillna(False)]
        if not bad.empty:
            print(bad.to_string(index=False), flush=True)
        else:
            print(summary.to_string(index=False), flush=True)

    return curves_df, flies_df, summary, dest


def main(argv=None):
    p = argparse.ArgumentParser(description="Cross-check python curves/flies vs BBG USOS/UF")
    p.add_argument(
        "--pull-outrights", action="store_true",
        help="Pull outrights from bdib instead of sofr_live.csv (default: live store)",
    )
    p.add_argument("--start", default=None, help="BBG start (YYYYMMDD or ISO)")
    p.add_argument("--end", default=None, help="BBG end (default: now / panel end)")
    p.add_argument("--out", default=None, help="Excel path (default: %s)" % CROSSCHECK_PATH)
    p.add_argument("--tol", type=float, default=None, help="Match tolerance in bp")
    args = p.parse_args(argv)

    run_crosscheck(
        from_store=not args.pull_outrights,
        start_date=args.start,
        end_date=args.end,
        path=args.out,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
