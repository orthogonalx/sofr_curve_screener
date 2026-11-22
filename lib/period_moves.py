#!/usr/bin/env python3
"""
Period move tables: change between two timestamps + std of same-horizon moves.

  START = "10/15/26 09:00"
  END   = "10/16/26 12:00"
  → outrights table + structures table

Accepted datetime formats (pandas-friendly), e.g.:
  "10/15/26 9:00" | "10/15/2026 09:00" | "2026-10-15 09:00" | "10/15/26 9am"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import curve_config as _cfg
from lib.data import (
    all_curves,
    all_flies,
    build_panel,
    ensure_series_in_live_store,
    load_panel,
    resolve_live_store_path,
    tickers_for_names,
    update_live_store_to_now,
)

DATA_BAR_MINUTES = float(getattr(_cfg, "DATA_BAR_MINUTES", 180))
MOVE_OUTRIGHTS = list(getattr(_cfg, "MOVE_OUTRIGHTS", ["2", "5", "10", "30"]))
MOVE_STRUCTURES = list(
    getattr(
        _cfg,
        "MOVE_STRUCTURES",
        [
            "ESA", "CO1",
            "2s5s", "5s10s", "10s30s", "2s5s10s", "5s10s30s",
            "USSFCT05", "USSFCT10", "USSFCT30",
        ],
    )
)
REFRESH_ON_RUN = bool(getattr(_cfg, "REFRESH_DATA_ON_REQUEST", True))


def parse_ts(text) -> pd.Timestamp:
    """Parse user datetime; accepts MM/DD/YY HH:MM, ISO, and '9am' style."""
    s = str(text).strip()
    if not s:
        raise ValueError("empty timestamp")
    # normalize am/pm glued to digits: "9am" → "9 am"
    s = (
        s.replace("am", " am").replace("pm", " pm")
        .replace("AM", " AM").replace("PM", " PM")
    )
    s = " ".join(s.split())
    ts = pd.to_datetime(s)
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _nearest_idx(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    if len(index) == 0:
        raise ValueError("empty panel index")
    loc = index.get_indexer([ts], method="nearest")[0]
    if loc < 0:
        raise KeyError("cannot locate %s in panel" % ts)
    return pd.Timestamp(index[loc])


def _horizon_bars(t0: pd.Timestamp, t1: pd.Timestamp, bar_minutes=None) -> int:
    bm = float(bar_minutes if bar_minutes is not None else DATA_BAR_MINUTES)
    hours = abs((t1 - t0).total_seconds()) / 3600.0
    return max(int(round(hours / (bm / 60.0))), 1)


def _horizon_label(t0: pd.Timestamp, t1: pd.Timestamp, n_bars: int) -> str:
    hours = abs((t1 - t0).total_seconds()) / 3600.0
    days = hours / 24.0
    if days < 0.75:
        return "~%.0fh (%d bars)" % (hours, n_bars)
    if days < 1.5:
        return "~1d (%d bars)" % n_bars
    if days < 6:
        return "~%.1fd (%d bars)" % (days, n_bars)
    if days < 9:
        return "~1w (%d bars)" % n_bars
    if days < 25:
        return "~%.0fd (%d bars)" % (days, n_bars)
    if days < 40:
        return "~1m (%d bars)" % n_bars
    return "~%.0fd (%d bars)" % (days, n_bars)


def _level_at(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    t = _nearest_idx(pd.DatetimeIndex(series.dropna().index), ts)
    return t, float(series.loc[t])


def _move_row(
    name: str,
    series: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    n_bars: int,
    scale_to_bp: bool = False,
) -> dict:
    """
    change over [t0,t1]; std of historical same-length moves (diff n_bars).
    scale_to_bp: multiply levels/change by 100 (outrights in percent → bp).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "name": name, "t_start": pd.NaT, "t_end": pd.NaT,
            "level_start": np.nan, "level_end": np.nan,
            "change": np.nan, "std": np.nan, "z": np.nan, "n_hist": 0,
        }
    ts0, v0 = _level_at(s, t0)
    ts1, v1 = _level_at(s, t1)
    mult = 100.0 if scale_to_bp else 1.0
    chg = (v1 - v0) * mult
    hist = (s.diff(n_bars).dropna()) * mult
    std = float(hist.std(ddof=1)) if len(hist) > 2 else np.nan
    z = float(chg / std) if std and abs(std) > 1e-12 else np.nan
    return {
        "name": name,
        "t_start": ts0,
        "t_end": ts1,
        "level_start": v0 * mult,
        "level_end": v1 * mult,
        "change": chg,
        "std": std,
        "z": z,
        "n_hist": int(len(hist)),
    }


def period_move_table(
    panel: pd.DataFrame,
    names: Sequence[str],
    start,
    end,
    *,
    scale_to_bp: bool = False,
    bar_minutes=None,
) -> Tuple[pd.DataFrame, dict]:
    t0, t1 = parse_ts(start), parse_ts(end)
    if t1 < t0:
        t0, t1 = t1, t0
    n_bars = _horizon_bars(t0, t1, bar_minutes=bar_minutes)
    meta = {
        "start_asked": t0,
        "end_asked": t1,
        "n_bars": n_bars,
        "horizon": _horizon_label(t0, t1, n_bars),
        "bar_minutes": float(bar_minutes if bar_minutes is not None else DATA_BAR_MINUTES),
    }
    rows = []
    missing = []
    for name in names:
        if name not in panel.columns:
            missing.append(name)
            continue
        rows.append(_move_row(name, panel[name], t0, t1, n_bars, scale_to_bp=scale_to_bp))
    if missing:
        print("  [moves] missing (skip): %s" % missing, flush=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        # display order
        cols = [
            "name", "level_start", "level_end", "change", "std", "z",
            "t_start", "t_end", "n_hist",
        ]
        df = df[[c for c in cols if c in df.columns]]
    return df, meta


def _structures_with_ussfct(panel: pd.DataFrame, base: Sequence[str]) -> List[str]:
    """MOVE_STRUCTURES plus any USSFCT* columns present in the panel."""
    names = list(base)
    for c in panel.columns:
        cu = str(c).upper()
        if cu.startswith("USSFCT") and c not in names and cu not in {x.upper() for x in names}:
            names.append(str(c))
    return names


def _names_needed_for_moves(outrights, structures):
    """Outrights + structure names + tenor legs so curves/flies can be built."""
    names = list(outrights) + list(structures)
    curve_map = {n: (a, b) for n, a, b in all_curves()}
    fly_map = {n: (a, b, c) for n, a, b, c in all_flies()}
    for n in list(structures):
        if n in curve_map:
            a, b = curve_map[n]
            names.extend([str(a), str(b)])
        elif n in fly_map:
            a, b, c = fly_map[n]
            names.extend([str(a), str(b), str(c)])
    # unique, preserve order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def load_moves_panel_from_csv(csv_path, ffill: bool = True):
    """
    Build a moves-ready panel from a BBG dump (e.g. data_11_22.csv)
    via lib.csv_analysis: clean → rates + curves/flies + spreads + extras.
    """
    from lib import csv_analysis

    raw = csv_analysis.load_csv(csv_path, ffill=False)
    cleaned, report = csv_analysis.clean_panel(raw)
    if ffill:
        cleaned = cleaned.ffill()
    panel = csv_analysis.build_analysis_panel(cleaned)
    print(
        "  [moves] csv src=%s  n=%d  asof=%s  tenors=%d"
        % (
            Path(csv_path).resolve(),
            len(panel),
            report.get("asof"),
            len(report.get("tenors") or []),
        ),
        flush=True,
    )
    return panel, report


def run_period_moves(
    start,
    end,
    *,
    outrights=None,
    structures=None,
    panel=None,
    refresh=None,
    prefer_daily=False,
    csv_path=None,
):
    """
    Build outrights + structures move tables between start and end.
    Outrights reported in bp (percent×100). Structures already in bp / BBG units.

    csv_path: if set (e.g. data_11_22.csv), load offline dump and skip BBG/live store.
    panel: optional pre-built panel (takes precedence).
    refresh: only applies to live-store path (ignored when csv_path is set).
    """
    out_names = list(outrights if outrights is not None else MOVE_OUTRIGHTS)
    st_want = list(structures if structures is not None else MOVE_STRUCTURES)
    do_refresh = REFRESH_ON_RUN if refresh is None else bool(refresh)

    if panel is not None:
        pass
    elif csv_path:
        panel, _ = load_moves_panel_from_csv(csv_path)
        # include extras in structures table when present
        # Always surface ESA (then CO1) at the front of the structures table
        for extra in ("ESA", "CO1"):
            if extra in panel.columns and extra not in st_want:
                st_want.insert(0 if extra == "ESA" else len(st_want), extra)
            elif extra == "ESA" and extra in st_want:
                st_want = ["ESA"] + [x for x in st_want if x != "ESA"]
    else:
        store_path = resolve_live_store_path()
        data_before, src_before = load_panel()
        asof_before = data_before.index[-1] if len(data_before) else None
        print(
            "  [moves] store before: path=%s  src=%s  n=%d  asof=%s"
            % (store_path, src_before, len(data_before), asof_before),
            flush=True,
        )
        if do_refresh:
            need_names = _names_needed_for_moves(out_names, st_want)
            secs = tickers_for_names(need_names)
            print(
                "  [moves] refreshing %d tickers (daily=%s): %s"
                % (len(secs), bool(prefer_daily), secs),
                flush=True,
            )
            if not secs:
                print("  [moves] no resolvable tickers — skip refresh", flush=True)
            else:
                try:
                    update_live_store_to_now(
                        path=store_path,
                        securities=secs,
                        lookback_days=60,
                        prefer_daily=prefer_daily,
                        require_new=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)
                    print(
                        "  [moves] WARNING: refresh failed — continuing with "
                        "existing sofr_live.\n           %s" % err,
                        flush=True,
                    )
                    if "refdata" in err.lower() or "//blp/" in err.lower():
                        print(
                            "  [moves] Bloomberg //blp/refdata is down on this host. "
                            "Period moves will use the last stored asof "
                            "(no new USOSFR bars until BBG is back).",
                            flush=True,
                        )
        ensure_series_in_live_store(list(out_names) + list(st_want))
        data_raw, src = load_panel()
        _, panel, _ = build_panel(data_raw)
        asof = panel.index[-1] if len(panel) else None
        print(
            "  [moves] after: path=%s  src=%s  n=%d  asof=%s"
            % (store_path, src, len(panel), asof),
            flush=True,
        )
        if asof_before is not None and asof is not None and asof <= asof_before:
            print(
                "  [moves] WARNING: asof did not advance (%s → %s). "
                "Check BBG / that git pull brought the latest lib/data.py."
                % (asof_before, asof),
                flush=True,
            )

    st_names = _structures_with_ussfct(panel, st_want)

    out_tbl, meta = period_move_table(panel, out_names, start, end, scale_to_bp=True)
    st_tbl, _ = period_move_table(panel, st_names, start, end, scale_to_bp=False)

    print(
        "  [moves] horizon=%s  asked %s → %s"
        % (meta["horizon"], meta["start_asked"], meta["end_asked"]),
        flush=True,
    )
    return out_tbl, st_tbl, meta, panel


def main(argv=None):
    p = argparse.ArgumentParser(description="Period move tables (outrights + structures)")
    p.add_argument("--start", required=True, help='e.g. "10/15/26 09:00"')
    p.add_argument("--end", required=True, help='e.g. "10/16/26 12:00"')
    p.add_argument(
        "--csv", default=None,
        help="Offline BBG dump (e.g. data_11_22.csv) — skips live store / BBG",
    )
    p.add_argument(
        "--no-refresh", action="store_true",
        help="Skip BBG update; use existing sofr_live.csv only",
    )
    p.add_argument(
        "--daily", action="store_true", default=True,
        help="Prefer daily bdh for the refresh (default; same path as run_pull_data)",
    )
    p.add_argument(
        "--intradaily", action="store_true",
        help="Use bdib intradaily instead of daily bdh",
    )
    args = p.parse_args(argv)
    out_tbl, st_tbl, meta, _ = run_period_moves(
        args.start, args.end,
        csv_path=args.csv,
        refresh=False if args.csv else (not args.no_refresh),
        prefer_daily=not args.intradaily,
    )
    print("\n=== OUTRIGHTS (bp)  horizon=%s ===" % meta["horizon"], flush=True)
    print(out_tbl.round(3).to_string(index=False), flush=True)
    print("\n=== STRUCTURES / SPREADS (bp) ===", flush=True)
    print(st_tbl.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
