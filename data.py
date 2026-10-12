"""Data layer: load/live store, structure building, Bloomberg pull, panel loaders."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from curve_config import (
    BBG_END_DATE, BBG_FIELD, BBG_FREQ, BBG_FWD_PREFIX, BBG_FWD_SUFFIX, BBG_PREFIX,
    BBG_REQUEST_REROUTE, BBG_START_DATE, BBG_SUFFIX, CURVES, DATA_BAR_MINUTES,
    DATA_PATH, FLIES, FORWARDS, FWD_CURVES, FWD_FLIES, LIVE_STORE_PATH,
    MAX_HISTORY_BARS, PREDICTED,
)

_TENOR_RE = re.compile(rf"^{re.escape(BBG_PREFIX)}(\d+)\s*{re.escape(BBG_SUFFIX)}$", re.I)
_FWD_RE = re.compile(rf"^{re.escape(BBG_FWD_PREFIX)}\s+(\d+)Y(\d+)Y\s*{re.escape(BBG_FWD_SUFFIX)}$", re.I)
_FWD_CLEAN_RE = re.compile(r"^(\d+)[Yy](\d+)[Yy]$")


# ── Raw loading / synthetic ──────────────────────────────────────────────────

def load_raw(filepath, timestamp_col: str = "time", sheet_name=0, ffill: bool = True) -> pd.DataFrame:
    """Load a SOFR panel (.xlsx/.csv) into a time-indexed numeric DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx or .csv")

    if timestamp_col not in df.columns:
        cands = ["time", "datetime", "date", "Date", "Time", "timestamp"]
        timestamp_col = next((c for c in cands if c in df.columns), df.columns[0])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    df.index.name = "time"
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.ffill() if ffill else df


def generate_synthetic_sofr(tenors: Optional[Sequence[int]] = None, n_bars: int = 400,
                            freq: str = "3h", seed: int = 42) -> pd.DataFrame:
    """Synthetic SOFR panel with Bloomberg-style columns (outrights + all FORWARDS)."""
    tenors = tenors or [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq)
    level = 3.5 + np.cumsum(rng.normal(0, 0.01, n_bars))
    slope = 0.15 + np.cumsum(rng.normal(0, 0.002, n_bars))

    data = {f"{BBG_PREFIX}{t} {BBG_SUFFIX}": level + slope * np.log1p(t) / np.log1p(30)
            + rng.normal(0, 0.005, n_bars) for t in tenors}
    for start, gap in FORWARDS:
        data[f"{BBG_FWD_PREFIX} {start}Y{gap}Y {BBG_FWD_SUFFIX}"] = (
            slope * (start / 30.0) + gap * 0.01 + rng.normal(0, 0.004, n_bars))

    shock = np.zeros(n_bars)  # mild dislocation near the end (35y)
    shock[-20:] = 0.08
    key = f"{BBG_PREFIX}35 {BBG_SUFFIX}"
    if key in data:
        data[key] = data[key] + shock
    df = pd.DataFrame(data, index=idx)
    df.index.name = "time"
    return df


# ── Live store (stack → dedupe → trim) ───────────────────────────────────────

def load_live_store(path=LIVE_STORE_PATH) -> pd.DataFrame:
    path = Path(path)
    return load_raw(path, ffill=True) if path.exists() else pd.DataFrame()


def save_live_store(df: pd.DataFrame, path=LIVE_STORE_PATH) -> Path:
    out = df.sort_index()
    out.index.name = "time"
    out.to_csv(path)
    return Path(path).resolve()


def append_snapshot(snapshot: pd.DataFrame, path=LIVE_STORE_PATH,
                    max_bars: Optional[int] = MAX_HISTORY_BARS) -> pd.DataFrame:
    """Stack snapshot onto the stored panel (last wins on dup time), trim to max_bars."""
    if snapshot.empty:
        raise ValueError("snapshot is empty")
    if not isinstance(snapshot.index, pd.DatetimeIndex):
        raise TypeError("snapshot must have a DatetimeIndex named time")
    snap = snapshot.copy()
    snap.index = pd.DatetimeIndex(snap.index).tz_localize(None)
    snap.index.name = "time"
    snap = snap.apply(pd.to_numeric, errors="coerce")

    existing = load_live_store(path)
    if existing.empty:
        combined = snap
    else:
        combined = pd.concat([existing, snap], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    if max_bars is not None and len(combined) > max_bars:
        combined = combined.iloc[-max_bars:]
    save_live_store(combined, path)
    return combined


def ensure_seed_history(path=LIVE_STORE_PATH, seed_path=None, n_synthetic: int = 400) -> pd.DataFrame:
    """If the live store is missing, seed it from DATA_PATH or synthetic data."""
    if Path(path).exists():
        return load_live_store(path)
    src = Path(seed_path) if seed_path else Path(DATA_PATH)
    hist = load_raw(src) if src.exists() else generate_synthetic_sofr(n_bars=n_synthetic, freq="h")
    save_live_store(hist, path)
    return hist


# ── Structures: raw → full_data → predicted_data ─────────────────────────────

def _parse(col) -> Optional[str]:
    """Raw column → '10' (tenor) or '20y5y' (forward); None if unparseable."""
    s = str(col).strip()
    m = _TENOR_RE.match(s)
    if m:
        return str(int(m.group(1)))
    m = _FWD_RE.match(s) or _FWD_CLEAN_RE.fullmatch(s)
    if m:
        return f"{int(m.group(1))}y{int(m.group(2))}y"
    m = re.fullmatch(r"(\d+)\s*[Yy]?", s)
    return str(int(m.group(1))) if m else None


def resolve_feature_name(name: str) -> str:
    """Feature label ('USOSFR5', '5Y', ...) → full_data column ('5'); else unchanged."""
    s = str(name).strip()
    parsed = _parse(s)
    if parsed is not None:
        return parsed
    m = re.fullmatch(rf"{re.escape(BBG_PREFIX)}(\d+)", s, re.I)
    return str(int(m.group(1))) if m else s


def clean_raw_columns(data_raw: pd.DataFrame) -> pd.DataFrame:
    """Rename BBG columns to '1','2',… / '20y5y',…; drop unparseable columns."""
    rename = {c: _parse(c) for c in data_raw.columns if _parse(c) is not None}
    missing = [c for c in data_raw.columns if c not in rename]
    if missing:
        print(f"  dropping unparsed columns: {missing}", flush=True)
    out = data_raw.rename(columns=rename)[list(rename.values())].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    return out.apply(pd.to_numeric, errors="coerce")


def _structs(raw, specs, avail, key=str):
    """Build (spec_name, series) for specs whose legs are all available; list skipped names.

    Curve/spread = (last - first)*100; fly = (2*mid - first - last)*100 (percent → bp).
    """
    built, skipped = [], []
    for name, *legs in specs:
        if not all(key(x) in avail for x in legs):
            skipped.append(name)
            continue
        c = [raw[key(x)] for x in legs]
        s = (c[1] - c[0]) * 100.0 if len(c) == 2 else (2.0 * c[1] - c[0] - c[2]) * 100.0
        built.append(s.rename(name))
    return built, skipped


def build_full_data(data_raw: pd.DataFrame, curves=CURVES, flies=FLIES,
                    fwd_curves=FWD_CURVES, fwd_flies=FWD_FLIES, clean: bool = True) -> pd.DataFrame:
    """Raw panel → outrights, swap curves/flies, forward levels, fwd curves/flies.

    Structures with missing legs are skipped (with a printed note).
    """
    raw = clean_raw_columns(data_raw) if clean else data_raw.copy()
    tenors = {c for c in raw.columns if str(c).isdigit()}
    fwds = {c for c in raw.columns if _FWD_CLEAN_RE.fullmatch(str(c))}

    cur, sk_c = _structs(raw, curves, tenors)
    fly, sk_f = _structs(raw, flies, tenors)
    if sk_c or sk_f:
        print(f"  skip structures (missing tenors): curves={sk_c or '—'}  flies={sk_f or '—'}  "
              f"have_tenors={sorted(int(t) for t in tenors)}", flush=True)
    fcur, sk_fc = _structs(raw, fwd_curves, fwds)
    ffly, sk_ff = _structs(raw, fwd_flies, fwds)
    if sk_fc or sk_ff:
        print(f"  skip fwd structures (missing forwards): "
              f"curves={sk_fc or '—'}  flies={sk_ff or '—'}", flush=True)

    def _fkey(n):
        m = _FWD_CLEAN_RE.fullmatch(n)
        return (int(m.group(2)), int(m.group(1)))

    pieces = [raw[t].rename(t) for t in sorted(tenors, key=int)] + cur + fly
    pieces += [raw[c].rename(c) for c in sorted(fwds, key=_fkey)] + fcur + ffly
    if not pieces:
        raise ValueError("No structures could be built — check raw tenors/forwards")
    full = pd.concat(pieces, axis=1)
    full.index = raw.index
    return full


def build_predicted_data(full_data: pd.DataFrame, predicted: Sequence[str] = PREDICTED) -> pd.DataFrame:
    """Isolate target columns (copy) from full_data."""
    missing = [c for c in predicted if c not in full_data.columns]
    if missing:
        print(f"  skip predicted targets not in full_data: {missing}", flush=True)
    cols = [c for c in predicted if c in full_data.columns]
    if not cols:
        raise KeyError(f"None of the PREDICTED targets are in full_data. "
                       f"Wanted {list(predicted)}; have {list(full_data.columns)}")
    return full_data.loc[:, cols].copy()


def build_datasets(data_raw: pd.DataFrame, curves=CURVES, flies=FLIES,
                   predicted: Sequence[str] = PREDICTED):
    """raw → (cleaned_raw, full_data, predicted_data)."""
    cleaned = clean_raw_columns(data_raw)
    full = build_full_data(cleaned, curves=curves, flies=flies, clean=False)
    return cleaned, full, build_predicted_data(full, predicted=predicted)


# ── Bloomberg pull (bbgapi bdp / bdh) ────────────────────────────────────────

def _securities() -> List[str]:
    tenors = set()
    for _, *legs in list(CURVES) + list(FLIES):
        tenors.update(legs)
    return ([f"{BBG_PREFIX}{int(t)} {BBG_SUFFIX}" for t in sorted(tenors)]
            + [f"{BBG_FWD_PREFIX} {int(s)}Y{int(g)}Y {BBG_FWD_SUFFIX}" for s, g in FORWARDS])


def _bquery():
    try:
        import bbgapi
    except ImportError as exc:
        raise ImportError("bbgapi is not installed in this Python. "
                          "Run from the Bloomberg / DataZone environment.") from exc
    bq = getattr(bbgapi, "bquery", None)
    if bq is None:
        try:
            import bquery as bq
        except ImportError as exc:
            raise ImportError("Imported bbgapi but could not find bquery.") from exc
    return bq


def _wide(raw, field: str) -> pd.DataFrame:
    """Normalize bdp/bdh output to columns = security tickers."""
    df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        lvl = next((i for i in range(df.columns.nlevels)
                    if field in {str(v) for v in df.columns.get_level_values(i)}), None)
        if lvl is None:
            raise ValueError(f"Field {field!r} not in bdh/bdp columns {df.columns[:5]}")
        df = df.xs(field, axis=1, level=lvl)
    if field in df.columns and (df.shape[1] == 1 or (
            df.index.name != "time" and set(map(str, df.columns)) <= {field, "security"})):
        s = df[field]
        df = s.to_frame().T
        df.columns = s.index
    return df


def freq_kwargs(freq: str) -> dict:
    """BBG_FREQ → bdh kwargs: '1D' → periodicity DAILY; '15min'/'1h' → interval minutes."""
    f = str(freq).strip().lower()
    if f in {"1d", "d", "daily", "1day"}:
        return {"periodicity": "DAILY"}
    if f.endswith("min"):
        return {"interval": int(f[:-3])}
    if f.endswith("h"):
        return {"interval": int(float(f[:-1]) * 60)}
    raise ValueError(f"BBG_FREQ={freq!r} — use '15min', '1h', or '1D'")


def _bdh(bq, kwargs: dict):
    """bdh, dropping optional kwargs the installed client rejects (request_reroute retry)."""
    kw = dict(kwargs)
    while True:
        try:
            return bq.bdh(**kw)
        except TypeError as exc:
            key = next((k for k in ("request_reroute", "interval", "periodicity") if k in kw), None)
            if key is None:
                raise
            kw.pop(key)
            print(f"  [bdh] client rejected {key} ({exc}); retrying", flush=True)


def _fetch_latest(field: str = BBG_FIELD) -> pd.DataFrame:
    """One live print via bdp → 1-row frame indexed by time."""
    names, bq = _securities(), _bquery()
    try:
        raw = bq.bdp(names, [field], **({"request_reroute": True} if BBG_REQUEST_REROUTE else {}))
    except TypeError:
        raw = bq.bdp(names, [field])
    wide = _wide(raw, field)
    if len(wide) != 1:
        if wide.shape[0] > 1 and wide.shape[1] == 1:
            wide = wide.T
        wide = wide.iloc[[-1]]
    asof = pd.Timestamp.now().floor(f"{int(DATA_BAR_MINUTES)}min")
    wide.index = pd.DatetimeIndex([asof], name="time")
    return wide.apply(pd.to_numeric, errors="coerce")


def _fetch_history(start_date=None, end_date=None, freq=None, field: str = BBG_FIELD) -> pd.DataFrame:
    """History via bdh for [start, end] at freq (defaults from curve_config)."""
    start = pd.Timestamp(str(BBG_START_DATE if start_date is None else start_date)).strftime("%Y%m%d")
    end_txt = str(BBG_END_DATE if end_date is None else end_date).strip()
    end = (pd.Timestamp(end_txt) if end_txt else pd.Timestamp.today()).strftime("%Y%m%d")
    freq = BBG_FREQ if freq is None else freq
    names = _securities()
    kwargs = {"securities": names, "fields": [field], "start_date": start, "end_date": end}
    kwargs.update(freq_kwargs(freq))
    if BBG_REQUEST_REROUTE:
        kwargs["request_reroute"] = True
    print(f"  [bdh] {start} → {end}  freq={freq}  n_securities={len(names)}", flush=True)

    wide = _wide(_bdh(_bquery(), kwargs), field)
    idx = pd.DatetimeIndex(pd.to_datetime(wide.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx.name = "time"
    wide.index = idx
    wide = wide.apply(pd.to_numeric, errors="coerce").sort_index()

    # If the API returned bars finer than freq, keep the last print per bar.
    mins = freq_kwargs(freq).get("interval")
    if mins and len(wide) >= 3:
        spacing = wide.index.to_series().diff().dt.total_seconds().median()
        if not pd.isna(spacing) and spacing < mins * 60 * 0.9:
            wide = wide.resample(f"{mins}min").last().dropna(how="all")
            wide.index.name = "time"
    return wide


def pull_history_into_store(path: Optional[str] = None, start_date=None, end_date=None,
                            freq=None) -> pd.DataFrame:
    """Pull [BBG_START_DATE, BBG_END_DATE] at BBG_FREQ and write the live store."""
    dest = path or LIVE_STORE_PATH
    hist = _fetch_history(start_date, end_date, freq)
    if hist.empty:
        raise ValueError("Bloomberg history pull returned no rows")
    save_live_store(hist, dest)
    print(f"  [history] wrote {dest}  n={len(hist)}  {hist.index[0]} → {hist.index[-1]}", flush=True)
    return hist


def append_live_bar(max_bars: int = MAX_HISTORY_BARS) -> pd.DataFrame:
    """Pull the latest print and append it onto the live store."""
    panel = append_snapshot(_fetch_latest(), path=LIVE_STORE_PATH, max_bars=max_bars)
    print(f"  [live] +1 → {LIVE_STORE_PATH}  n={len(panel)}  last={panel.index[-1]}", flush=True)
    return panel


def refresh_live_store(force_history: bool = False) -> Tuple[pd.DataFrame, str]:
    """bdh history if store empty/forced, then bdp latest append; fallback to load_panel."""
    store = Path(LIVE_STORE_PATH)
    try:
        existing = load_live_store(store) if store.exists() else pd.DataFrame()
        if force_history or existing.empty:
            print(f"  [history] {BBG_START_DATE} → {BBG_END_DATE or 'today'}  freq={BBG_FREQ}", flush=True)
            pull_history_into_store()
        return append_live_bar(), f"live:{store}"
    except Exception as exc:  # noqa: BLE001
        print(f"  [refresh] pull failed ({exc}); using existing panel", flush=True)
        if not store.exists():
            ensure_seed_history(store)
        data, src = load_panel()
        return data, f"fallback:{src}"


# ── Panel loaders ────────────────────────────────────────────────────────────

def load_panel() -> Tuple[pd.DataFrame, str]:
    """live store → DATA_PATH file → synthetic."""
    live, path = Path(LIVE_STORE_PATH), Path(DATA_PATH)
    if live.exists():
        return load_live_store(live), f"live:{live}"
    if path.exists():
        return load_raw(path), f"file:{path}"
    return generate_synthetic_sofr(n_bars=400, freq="15T"), "synthetic"


def build_panel(data_raw: pd.DataFrame):
    """raw → (cleaned, full_data, predicted_data)."""
    if not isinstance(data_raw.index, pd.DatetimeIndex):
        raise TypeError("data_raw must be indexed by time")
    return build_datasets(data_raw.apply(pd.to_numeric, errors="coerce").ffill())
