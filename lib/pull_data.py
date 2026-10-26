#!/usr/bin/env python3
"""
Pull a small BBG panel — no full pipeline.

  # intradaily (minutes)
  python run_dz.py pull --tickers USGG5YR,CL1 --start 20240101 --freq 180

  # daily
  python run_dz.py pull --tickers USGG5YR,CL1 --start 20240101 --freq 1d

  # or run_pull_data.ipynb  → set FREQ = "1d"
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from lib.data import (
    BBG_END_DATE,
    BBG_INTERVAL_MINUTES,
    BBG_START_DATE,
    _as_dt,
    _bdh_px_last,
    _bquery,
    _fetch_bdib_panel,
    _interval_minutes,
    bbg_ticker_for_name,
    clean_raw_columns,
)

PanelPath = Union[str, Path]


def _is_daily(freq) -> bool:
    if freq is None:
        return False
    if isinstance(freq, str) and freq.strip().lower() in {"1d", "d", "daily", "1day"}:
        return True
    try:
        return int(freq) >= 24 * 60
    except (TypeError, ValueError):
        return False


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


def _pull_daily(securities, start, end) -> pd.DataFrame:
    """PX_LAST via bdh for each security (true daily bars)."""
    start_ts = _as_dt(BBG_START_DATE if start in (None, "") else start)
    end_ts = _as_dt(BBG_END_DATE if end in (None, "") else end, default_now=True)
    bq = _bquery()
    print(
        "  [pull] daily bdh  %s → %s  n=%d" % (start_ts.date(), end_ts.date(), len(securities)),
        flush=True,
    )
    series_list = []
    for sec in securities:
        try:
            s = _bdh_px_last(bq, sec, start_ts, end_ts)
        except Exception as exc:  # noqa: BLE001
            print("  [bdh] skip %s: %s" % (sec, exc), flush=True)
            continue
        if s is None or s.empty:
            print("  [bdh] empty %s" % sec, flush=True)
            continue
        series_list.append(s)
        print("  [bdh] %s  bars=%d" % (sec, len(s)), flush=True)
    if not series_list:
        raise ValueError("bdh returned no series for %s" % list(securities))
    return pd.concat(series_list, axis=1).sort_index()


def pull_series(
    tickers: Sequence[str],
    start=None,
    end=None,
    freq=None,
    freq_minutes=None,
    out_path: Optional[PanelPath] = None,
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Fetch tickers from BBG.

    Parameters
    ----------
    tickers : clean names ('CL1') or full BBG ('CL1 Comdty')
    start, end : date-like; empty end → now
    freq / freq_minutes :
        - int minutes → intradaily bdib (e.g. 180)
        - \"1d\" / \"daily\" / 1440 → daily bdh PX_LAST
    out_path : if set, write CSV (and .xlsx if openpyxl available)
    ffill : forward-fill sparse bars across the joint index

    Returns cleaned DataFrame (DatetimeIndex, clean column names).
    """
    secs = resolve_tickers(tickers)
    freq_arg = freq if freq is not None else freq_minutes
    start_s = None if start in (None, "") else str(start).strip()
    end_s = None if end in (None, "") else str(end).strip()

    if _is_daily(freq_arg):
        print("  [pull] tickers=%s  mode=daily" % secs, flush=True)
        raw = _pull_daily(secs, start_s, end_s)
    else:
        mins = int(_interval_minutes(freq_arg if freq_arg is not None else BBG_INTERVAL_MINUTES))
        print(
            "  [pull] tickers=%s  start=%s  end=%s  freq=%sm"
            % (secs, start_s or "(default)", end_s or "now", mins),
            flush=True,
        )
        raw = _fetch_bdib_panel(
            start_date=start_s, end_date=end_s, freq=mins, securities=secs,
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
    p.add_argument(
        "--freq", default=None,
        help="Bar size: minutes (180) or daily (1d / daily). Default 180.",
    )
    p.add_argument("--out", default=None, help="CSV/xlsx path (default exports/panels/…)")
    args = p.parse_args(argv)
    tickers = [t.strip() for t in str(args.tickers).split(",") if t.strip()]
    out = args.out or default_out_path()
    freq = args.freq
    if freq is not None and str(freq).isdigit():
        freq = int(freq)
    pull_series(
        tickers, start=args.start, end=args.end,
        freq=freq, out_path=out,
    )


if __name__ == "__main__":
    main()
