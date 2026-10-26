#!/usr/bin/env python3
"""
Pull a small BBG panel — no full pipeline.

  python run_dz.py pull --tickers USGG5YR,CL1 --start 20240101
  # or run_pull_data.ipynb
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from lib.data import (
    BBG_INTERVAL_MINUTES,
    _fetch_bdib_panel,
    bbg_ticker_for_name,
    clean_raw_columns,
)

PanelPath = Union[str, Path]


def resolve_tickers(tickers: Sequence[str]) -> List[str]:
    """Clean names or full BBG tickers → BBG securities list."""
    out, seen = [], set()
    for raw in tickers:
        s = str(raw).strip()
        if not s:
            continue
        t = bbg_ticker_for_name(s)
        if t is None and re.search(r"\s+(Comdty|Index|Curncy|Equity)\s*$", s, re.I):
            t = s
        if t is None:
            raise ValueError(
                "Unknown ticker %r — add to CROSS_ASSET or pass full BBG "
                "(e.g. 'CL1 Comdty')" % s
            )
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    if not out:
        raise ValueError("No tickers to pull")
    return out


def pull_series(
    tickers: Sequence[str],
    start=None,
    end=None,
    freq_minutes: int = None,
    out_path: Optional[PanelPath] = None,
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Fetch tickers via bdib (bdh daily fallback on NOT_ENTITLED).

    Parameters
    ----------
    tickers : clean names ('CL1') or full BBG ('CL1 Comdty')
    start, end : date-like; empty end → now
    freq_minutes : bar size (default BBG_INTERVAL_MINUTES = 180)
    out_path : if set, write CSV (and .xlsx if openpyxl available)
    ffill : forward-fill sparse bars (different tickers / daily oil vs intradaily)

    Returns cleaned DataFrame (DatetimeIndex, clean column names).
    """
    secs = resolve_tickers(tickers)
    freq = int(freq_minutes if freq_minutes is not None else BBG_INTERVAL_MINUTES)
    start_s = None if start in (None, "") else str(start).strip()
    end_s = None if end in (None, "") else str(end).strip()
    print(
        "  [pull] tickers=%s  start=%s  end=%s  freq=%sm"
        % (secs, start_s or "(default)", end_s or "now", freq),
        flush=True,
    )
    raw = _fetch_bdib_panel(
        start_date=start_s, end_date=end_s, freq=freq, securities=secs,
    )
    panel = clean_raw_columns(raw)
    panel = panel.apply(pd.to_numeric, errors="coerce").sort_index()
    if ffill:
        panel = panel.ffill()
    panel.index.name = "time"
    print(
        "  [pull] n=%d  cols=%s  %s → %s"
        % (len(panel), list(panel.columns), panel.index[0], panel.index[-1]),
        flush=True,
    )
    if out_path:
        _write_panel(panel, out_path)
    return panel


def _write_panel(panel: pd.DataFrame, path: PanelPath) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() not in (".csv", ".xlsx", ".xls"):
        dest = dest.with_suffix(".csv")
    if dest.suffix.lower() == ".csv":
        panel.to_csv(dest)
        print("  [pull] wrote %s" % dest.resolve(), flush=True)
        xlsx = dest.with_suffix(".xlsx")
        try:
            panel.to_excel(xlsx)
            print("  [pull] wrote %s" % xlsx.resolve(), flush=True)
        except ImportError:
            pass
        return dest.resolve()
    try:
        panel.to_excel(dest)
        print("  [pull] wrote %s" % dest.resolve(), flush=True)
    except ImportError:
        csv = dest.with_suffix(".csv")
        panel.to_csv(csv)
        print("  [pull] openpyxl missing → %s" % csv.resolve(), flush=True)
        return csv.resolve()
    return dest.resolve()


def default_out_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return "exports/panels/panel_%s.csv" % stamp


def main(argv=None):
    p = argparse.ArgumentParser(description="Pull BBG series into a small panel CSV")
    p.add_argument(
        "--tickers", required=True,
        help="Comma-separated clean or BBG tickers (USGG5YR,CL1,ESA)",
    )
    p.add_argument("--start", default=None, help="YYYYMMDD or ISO (default: config)")
    p.add_argument("--end", default=None, help="YYYYMMDD or ISO (default: now)")
    p.add_argument("--freq", type=int, default=None, help="Bar minutes (default 180)")
    p.add_argument("--out", default=None, help="CSV/xlsx path (default exports/panels/…)")
    args = p.parse_args(argv)
    tickers = [t.strip() for t in str(args.tickers).split(",") if t.strip()]
    out = args.out or default_out_path()
    pull_series(
        tickers, start=args.start, end=args.end,
        freq_minutes=args.freq, out_path=out,
    )


if __name__ == "__main__":
    main()
