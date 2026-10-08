# =============================================================================
# bbg_pull.py  — Live Bloomberg pull via internal bbgapi / bquery
# =============================================================================
# Notebook pattern:
#   import bbgapi
#   bquery.bdp(securities, ['PX_LAST'])                         # latest
#   bquery.bdh(securities=..., fields=['PX_LAST'],
#              start_date='YYYYMMDD', end_date='YYYYMMDD',
#              request_reroute=True)                           # history
#
# Returns the same column names the rest of the pipeline already parses:
#   "USOSFR10 BGN Curncy", "S0490FS 20Y5Y BLC Curncy"
# =============================================================================

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from curve_config import (
    BBG_END_DATE,
    BBG_FIELD,
    BBG_FREQ,
    BBG_FWD_PREFIX,
    BBG_FWD_SUFFIX,
    BBG_PREFIX,
    BBG_REQUEST_REROUTE,
    BBG_START_DATE,
    BBG_SUFFIX,
    CURVES,
    DATA_BAR_MINUTES,
    FLIES,
    FORWARDS,
    LIVE_STORE_PATH,
    MAX_HISTORY_BARS,
)


def outright_ticker(tenor: int) -> str:
    return f"{BBG_PREFIX}{int(tenor)} {BBG_SUFFIX}"


def forward_ticker(start: int, gap: int) -> str:
    return f"{BBG_FWD_PREFIX} {int(start)}Y{int(gap)}Y {BBG_FWD_SUFFIX}"


def securities_universe() -> List[str]:
    """Outrights required by CURVES/FLIES + every configured forward."""
    tenors = set()
    for _, a, b in CURVES:
        tenors.update((a, b))
    for _, a, b, c in FLIES:
        tenors.update((a, b, c))
    names = [outright_ticker(t) for t in sorted(tenors)]
    names += [forward_ticker(s, g) for s, g in FORWARDS]
    return names


def _get_bquery():
    """Load the desk Bloomberg client (`import bbgapi` then `bquery`)."""
    try:
        import bbgapi
    except ImportError as exc:
        raise ImportError(
            "bbgapi is not installed in this Python. "
            "Run the report from the Bloomberg / DataZone environment."
        ) from exc
    bquery = getattr(bbgapi, "bquery", None)
    if bquery is None:
        try:
            import bquery as bquery_mod
        except ImportError as exc:
            raise ImportError(
                "Imported bbgapi but could not find bquery. "
                "Expected `from bbgapi import bquery` or a top-level `bquery`."
            ) from exc
        bquery = bquery_mod
    return bquery


def _as_security_frame(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Normalize bdp/bdh output to columns = security tickers.

    Accepts:
      - index=security, column=field          (typical bdp)
      - columns=security                       (single-field bdh)
      - columns MultiIndex (security, field) or (field, security)
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level_with_field = None
        for lvl in range(df.columns.nlevels):
            vals = {str(v) for v in df.columns.get_level_values(lvl)}
            if field in vals:
                level_with_field = lvl
                break
        if level_with_field is None:
            raise ValueError(f"Field {field!r} not in bdh/bdp columns {df.columns[:5]}")
        df = df.xs(field, axis=1, level=level_with_field)
    elif field in df.columns and df.index.name != "time":
        # bdp: one row per security
        if len(df.columns) == 1 or set(map(str, df.columns)) <= {field, "security"}:
            s = df[field] if field in df.columns else df.iloc[:, 0]
            df = s.to_frame().T
            df.columns = s.index
            return df
        # bdh already wide, field is not a security
    if field in df.columns and df.shape[1] == 1:
        # single security bdp with field column and security in the index name/values
        s = df[field]
        out = s.to_frame().T
        out.columns = s.index
        return out
    return df


def _stamp_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx.name = "time"
    return idx


def fetch_bloomberg_latest(
    asof: Optional[pd.Timestamp] = None,
    securities: Optional[Sequence[str]] = None,
    field: str = BBG_FIELD,
) -> pd.DataFrame:
    """
    One live print: bquery.bdp(securities, [PX_LAST]).

    Returns a 1-row frame, DatetimeIndex named "time", columns = BBG tickers.
    """
    names = list(securities) if securities is not None else securities_universe()
    bquery = _get_bquery()
    kwargs = {}
    if BBG_REQUEST_REROUTE:
        kwargs["request_reroute"] = True
    try:
        raw = bquery.bdp(names, [field], **kwargs)
    except TypeError:
        raw = bquery.bdp(names, [field])
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    wide = _as_security_frame(raw, field)
    if len(wide) != 1:
        # bdp sometimes returns securities as rows; collapse to one snapshot row
        if wide.shape[0] > 1 and wide.shape[1] == 1:
            wide = wide.T
        wide = wide.iloc[[-1]]

    if asof is None:
        asof = pd.Timestamp.now().floor(f"{int(DATA_BAR_MINUTES)}min")
    wide.index = pd.DatetimeIndex([pd.Timestamp(asof)], name="time")
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide


def _yyyymmdd(value: Optional[str], *, default_today: bool = False) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        if default_today:
            return pd.Timestamp.today().strftime("%Y%m%d")
        raise ValueError("empty Bloomberg date")
    return pd.Timestamp(text).strftime("%Y%m%d")


def freq_kwargs(freq: str) -> dict:
    """
    Map BBG_FREQ onto bdh kwargs.

    "1D" / "daily" → periodicity DAILY
    "15min" / "1h" → interval in minutes (intraday bars)
    """
    f = str(freq).strip().lower()
    if f in {"1d", "d", "daily", "1day"}:
        return {"periodicity": "DAILY"}
    if f.endswith("min"):
        return {"interval": int(f[: -len("min")])}
    if f.endswith("h"):
        return {"interval": int(float(f[:-1]) * 60)}
    raise ValueError(f"BBG_FREQ={freq!r} — use '15min', '1h', or '1D'")


def _call_bdh(bquery, kwargs: dict):
    """Call bdh, dropping optional kwargs the installed client does not accept."""
    attempt = dict(kwargs)
    while True:
        try:
            return bquery.bdh(**attempt)
        except TypeError as exc:
            dropped = False
            for key in ("request_reroute", "interval", "periodicity"):
                if key in attempt:
                    attempt.pop(key)
                    print(f"  [bdh] client rejected {key} ({exc}); retrying", flush=True)
                    dropped = True
                    break
            if not dropped:
                raise


def fetch_bloomberg_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: Optional[str] = None,
    securities: Optional[Sequence[str]] = None,
    field: str = BBG_FIELD,
) -> pd.DataFrame:
    """
    History via bquery.bdh for [start, end] at `freq`.

    Defaults: BBG_START_DATE, BBG_END_DATE (blank = today), BBG_FREQ.
    Dates as 'YYYYMMDD' (same as the jumpstart notebook).
    """
    start = _yyyymmdd(BBG_START_DATE if start_date is None else start_date)
    end = _yyyymmdd(
        BBG_END_DATE if end_date is None else end_date,
        default_today=True,
    )
    freq = BBG_FREQ if freq is None else freq
    names = list(securities) if securities is not None else securities_universe()
    bquery = _get_bquery()
    kwargs = {
        "securities": names,
        "fields": [field],
        "start_date": start,
        "end_date": end,
    }
    kwargs.update(freq_kwargs(freq))
    if BBG_REQUEST_REROUTE:
        kwargs["request_reroute"] = True
    print(
        f"  [bdh] {start} → {end}  freq={freq}  n_securities={len(names)}",
        flush=True,
    )
    raw = _call_bdh(bquery, kwargs)
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    wide = _as_security_frame(raw, field)
    wide.index = _stamp_index(wide.index)
    wide = wide.apply(pd.to_numeric, errors="coerce").sort_index()
    wide = _resample_if_finer(wide, freq)
    return wide


def _resample_if_finer(wide: pd.DataFrame, freq: str) -> pd.DataFrame:
    """If the API returned ticks finer than BBG_FREQ, keep the last print in each bar."""
    f = str(freq).strip().lower()
    if f in {"1d", "d", "daily", "1day"} or len(wide) < 3:
        return wide
    rule = f if f.endswith("min") else f.replace("h", "h")
    if f.endswith("h") and not f.endswith("min"):
        rule = f"{int(float(f[:-1]) * 60)}min"
    target = pd.Timedelta(rule).total_seconds()
    spacing = wide.index.to_series().diff().dt.total_seconds().median()
    if pd.isna(spacing) or spacing >= target * 0.9:
        return wide
    out = wide.resample(rule).last().dropna(how="all")
    out.index.name = "time"
    return out


def pull_history_into_store(
    path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """Pull [BBG_START_DATE, BBG_END_DATE] at BBG_FREQ and write LIVE_STORE_PATH."""
    from live_store import save_live_store

    dest = path or LIVE_STORE_PATH
    hist = fetch_bloomberg_history(start_date=start_date, end_date=end_date, freq=freq)
    if hist.empty:
        raise ValueError("Bloomberg history pull returned no rows")
    save_live_store(hist, dest)
    print(f"  [history] wrote {dest}  n={len(hist)}  {hist.index[0]} → {hist.index[-1]}", flush=True)
    return hist


def append_live_bar(max_bars: int = MAX_HISTORY_BARS) -> pd.DataFrame:
    """Pull the latest print and append it onto the existing live store."""
    from live_store import append_snapshot

    snap = fetch_bloomberg_latest()
    panel = append_snapshot(snap, path=LIVE_STORE_PATH, max_bars=max_bars)
    print(
        f"  [live] +1 → {LIVE_STORE_PATH}  n={len(panel)}  last={panel.index[-1]}",
        flush=True,
    )
    return panel
