"""Data: load/live store, structures, Bloomberg bdib, panel loaders."""
from __future__ import annotations

import glob
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import curve_config as _cfg
from curve_config import (
    BBG_END_DATE, BBG_EVENT_TYPE, BBG_FWD_PREFIX, BBG_FWD_SUFFIX,
    BBG_INTERVAL_MINUTES, BBG_PREFIX, BBG_START_DATE, BBG_SUFFIX, CURVES,
    DATA_BAR_MINUTES, DATA_PATH, FLIES, FORWARDS, FWD_CURVES, FWD_FLIES,
    LIVE_STORE_PATH, MAX_HISTORY_BARS, PREDICTED,
)

STRUCTURES_STORE_PATH = getattr(_cfg, "STRUCTURES_STORE_PATH", "sofr_structures.csv")
us_holiday_dates = getattr(_cfg, "us_holiday_dates", None)
USE_BBG_SWAP_STRUCTURES = getattr(_cfg, "USE_BBG_SWAP_STRUCTURES", True)
BBG_CURVE_TICKER_PREFIX = getattr(_cfg, "BBG_CURVE_TICKER_PREFIX", "USOS")
BBG_FLY_TICKER_PREFIX = getattr(_cfg, "BBG_FLY_TICKER_PREFIX", "UF")
BBG_STRUCT_SUFFIX = getattr(_cfg, "BBG_STRUCT_SUFFIX", "Curncy")
CROSS_ASSET = list(getattr(_cfg, "CROSS_ASSET", []))
CUSTOM_CURVES = list(getattr(_cfg, "CUSTOM_CURVES", []))
CUSTOM_FLIES = list(getattr(_cfg, "CUSTOM_FLIES", []))


def all_curves():
    return list(CURVES) + list(CUSTOM_CURVES)


def all_flies():
    return list(FLIES) + list(CUSTOM_FLIES)


def drop_non_trading_days(df, holidays=None):
    """
    Drop weekends and US market holidays (no session → no real SOFR prints).
    Holidays include federal days (e.g. 2026-10-12 Columbus Day) + Good Friday.
    """
    if df is None or df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        days = idx.tz_convert(None).normalize()
    else:
        days = idx.normalize()
    if holidays is None:
        if us_holiday_dates is None:
            hol = pd.DatetimeIndex([])
        else:
            hol = us_holiday_dates(
                days.min() - pd.Timedelta(days=7),
                days.max() + pd.Timedelta(days=7),
            )
    else:
        hol = pd.DatetimeIndex(pd.to_datetime(list(holidays))).normalize()
    mask = (idx.dayofweek < 5) & (~days.isin(hol))
    out = df.loc[mask]
    n_drop = len(df) - len(out)
    if n_drop:
        print(
            "  [calendar] dropped %d weekend/holiday bars (%d → %d)"
            % (n_drop, len(df), len(out)),
            flush=True,
        )
    return out

BBGAPI_VERSION = getattr(_cfg, "BBGAPI_VERSION", "2022.09.26-2")
BBGAPI_PATH = getattr(_cfg, "BBGAPI_PATH", "")
_TENOR_RE = re.compile(r"^%s(\d+)\s*%s$" % (re.escape(BBG_PREFIX), re.escape(BBG_SUFFIX)), re.I)
_FWD_RE = re.compile(
    r"^%s\s+(\d+)Y(\d+)Y\s*%s$" % (re.escape(BBG_FWD_PREFIX), re.escape(BBG_FWD_SUFFIX)), re.I
)
_FWD_CLEAN_RE = re.compile(r"^(\d+)[Yy](\d+)[Yy]$")
# Listed swap curve / fly (not outright USOSFR): USOS0510 Curncy, UF050710 Curncy
_BBG_CURVE_RE = re.compile(
    r"^%s(\d{2})(\d{2})\s*%s$" % (re.escape(BBG_CURVE_TICKER_PREFIX), re.escape(BBG_STRUCT_SUFFIX)),
    re.I,
)
_BBG_FLY_RE = re.compile(
    r"^%s(\d{2})(\d{2})(\d{2})\s*%s$"
    % (re.escape(BBG_FLY_TICKER_PREFIX), re.escape(BBG_STRUCT_SUFFIX)),
    re.I,
)
# SOFR vs FF basis: USSFCT05 Curncy → USSFCT05 (any tenor, not only CROSS_ASSET list)
_USSFCT_RE = re.compile(r"^USSFCT(\d+)\s*Curncy$", re.I)
_BQ = None
_DT_FMT = "%Y-%m-%dT%H:%M:%S"
_BBG_LOG_KEYS = ("bbg", "blp", "dz", "datazone", "activity", "kerberos", "ms.")


def bbg_curve_ticker(short, long_):
    return "%s%02d%02d %s" % (BBG_CURVE_TICKER_PREFIX, int(short), int(long_), BBG_STRUCT_SUFFIX)


def bbg_fly_ticker(left, body, right):
    return "%s%02d%02d%02d %s" % (
        BBG_FLY_TICKER_PREFIX, int(left), int(body), int(right), BBG_STRUCT_SUFFIX,
    )


def curve_ticker_map(curves=None):
    return {name: bbg_curve_ticker(a, b) for name, a, b in (curves or all_curves())}


def fly_ticker_map(flies=None):
    return {name: bbg_fly_ticker(a, b, c) for name, a, b, c in (flies or all_flies())}


def _curve_name_by_legs(curves=None):
    return {(int(a), int(b)): name for name, a, b in (curves or all_curves())}


def _fly_name_by_legs(flies=None):
    return {(int(a), int(b), int(c)): name for name, a, b, c in (flies or all_flies())}


def structure_securities(curves=None, flies=None):
    """BBG tickers for every CURVES / FLIES / CUSTOM_* entry."""
    return (
        list(curve_ticker_map(curves or all_curves()).values())
        + list(fly_ticker_map(flies or all_flies()).values())
    )


def load_raw(filepath, timestamp_col="time", sheet_name=0, ffill=True):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError("File not found: %s" % filepath)
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type '%s'. Use .xlsx or .csv" % ext)
    if timestamp_col not in df.columns:
        timestamp_col = next(
            (c for c in ("time", "datetime", "date", "Date", "Time", "timestamp") if c in df.columns),
            df.columns[0],
        )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    df.index.name = "time"
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.ffill() if ffill else df


def generate_synthetic_sofr(tenors=None, n_bars=400, freq="3h", seed=42):
    tenors = tenors or [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq)
    level = 3.5 + np.cumsum(rng.normal(0, 0.01, n_bars))
    slope = 0.15 + np.cumsum(rng.normal(0, 0.002, n_bars))
    data = {
        "%s%d %s" % (BBG_PREFIX, t, BBG_SUFFIX):
        level + slope * np.log1p(t) / np.log1p(30) + rng.normal(0, 0.005, n_bars)
        for t in tenors
    }
    for start, gap in FORWARDS:
        data["%s %dY%dY %s" % (BBG_FWD_PREFIX, start, gap, BBG_FWD_SUFFIX)] = (
            slope * (start / 30.0) + gap * 0.01 + rng.normal(0, 0.004, n_bars)
        )
    key = "%s35 %s" % (BBG_PREFIX, BBG_SUFFIX)
    if key in data:
        data[key] = data[key] + np.concatenate([np.zeros(n_bars - 20), np.full(20, 0.08)])
    df = pd.DataFrame(data, index=idx)
    # Synthetic listed structures (same tickers as live BBG pull), in bp
    for name, a, b in all_curves():
        ca, cb = "%s%d %s" % (BBG_PREFIX, a, BBG_SUFFIX), "%s%d %s" % (BBG_PREFIX, b, BBG_SUFFIX)
        if ca in df.columns and cb in df.columns:
            df[bbg_curve_ticker(a, b)] = (df[cb] - df[ca]) * 100.0
    for name, a, b, c in all_flies():
        cols = ["%s%d %s" % (BBG_PREFIX, t, BBG_SUFFIX) for t in (a, b, c)]
        if all(x in df.columns for x in cols):
            df[bbg_fly_ticker(a, b, c)] = (2.0 * df[cols[1]] - df[cols[0]] - df[cols[2]]) * 100.0
    # Synthetic cross-assets / extras
    for clean, ticker in CROSS_ASSET:
        df[ticker] = 100.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    df.index.name = "time"
    return df


def load_live_store(path=LIVE_STORE_PATH):
    path = Path(path)
    return load_raw(path, ffill=True) if path.exists() else pd.DataFrame()


def save_live_store(df, path=LIVE_STORE_PATH):
    out = drop_non_trading_days(df.sort_index())
    out.index.name = "time"
    out.to_csv(path)
    return Path(path).resolve()


def save_structures_store(full_data, path=None):
    """
    Persist built panel (tenors + curves/flies/fwd structures) next to sofr_live.csv.
    Columns like 2, 5, 2s5s, 5s7s10s, 4y1y_5y1y_6y1y, …
    """
    dest = Path(path or STRUCTURES_STORE_PATH)
    out = full_data.sort_index().copy()
    out.index.name = "time"
    out.to_csv(dest)
    print(
        "  [structures] wrote %s  n=%d  cols=%d  %s → %s"
        % (dest, len(out), out.shape[1], out.index[0], out.index[-1]),
        flush=True,
    )
    return dest.resolve()


def append_snapshot(snapshot, path=LIVE_STORE_PATH, max_bars=MAX_HISTORY_BARS):
    if snapshot.empty:
        raise ValueError("snapshot is empty")
    if not isinstance(snapshot.index, pd.DatetimeIndex):
        raise TypeError("snapshot must have a DatetimeIndex named time")
    snap = snapshot.copy()
    snap.index = pd.DatetimeIndex(snap.index).tz_localize(None)
    snap.index.name = "time"
    snap = snap.apply(pd.to_numeric, errors="coerce")
    path = Path(path)
    existing = load_raw(path, ffill=False) if path.exists() else pd.DataFrame()
    if existing.empty:
        combined = snap
    else:
        combined = pd.concat([existing, snap], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = drop_non_trading_days(
        combined.apply(pd.to_numeric, errors="coerce")
    ).ffill()
    if max_bars is not None and len(combined) > max_bars:
        combined = combined.iloc[-max_bars:]
    save_live_store(combined, path)
    return combined


def ensure_seed_history(path=LIVE_STORE_PATH, seed_path=None, n_synthetic=400):
    if Path(path).exists():
        return load_live_store(path)
    src = Path(seed_path) if seed_path else Path(DATA_PATH)
    hist = load_raw(src) if src.exists() else generate_synthetic_sofr(n_bars=n_synthetic, freq="h")
    save_live_store(hist, path)
    return hist


def _parse(col, curves=None, flies=None):
    s = str(col).strip()
    # Outright USOSFR{N} before listed USOS#### (avoid USOSFR10 matching as curve)
    m = _TENOR_RE.match(s)
    if m:
        return str(int(m.group(1)))
    m = _BBG_CURVE_RE.match(s)
    if m:
        legs = (int(m.group(1)), int(m.group(2)))
        return _curve_name_by_legs(curves).get(legs) or "%ds%ds" % legs
    m = _BBG_FLY_RE.match(s)
    if m:
        legs = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return _fly_name_by_legs(flies).get(legs) or "%ds%ds%ds" % legs
    m = _FWD_RE.match(s) or _FWD_CLEAN_RE.fullmatch(s)
    if m:
        return "%dy%dy" % (int(m.group(1)), int(m.group(2)))
    # Already-cleaned structure names (incl. CUSTOM_*)
    curve_names = {spec[0] for spec in (curves or all_curves())}
    fly_names = {spec[0] for spec in (flies or all_flies())}
    if s in curve_names or s in fly_names:
        return s
    # USSFCT## Curncy (SOFR/FF) — always keep, even if not listed in CROSS_ASSET yet
    m = _USSFCT_RE.match(s)
    if m:
        return "USSFCT%02d" % int(m.group(1)) if len(m.group(1)) <= 2 else "USSFCT%s" % m.group(1)
    if re.fullmatch(r"USSFCT\d+", s, re.I):
        return s.upper()
    # Cross-asset: map BBG ticker → clean name (and accept clean name as-is)
    xa_by_ticker = {str(t).strip().lower(): n for n, t in CROSS_ASSET}
    xa_names = {n for n, _ in CROSS_ASSET}
    if s.lower() in xa_by_ticker:
        return xa_by_ticker[s.lower()]
    if s in xa_names:
        return s
    m = re.fullmatch(r"(\d+)\s*[Yy]?", s)
    return str(int(m.group(1))) if m else None


def resolve_feature_name(name):
    s = str(name).strip()
    parsed = _parse(s)
    if parsed is not None:
        return parsed
    m = re.fullmatch(r"%s(\d+)" % re.escape(BBG_PREFIX), s, re.I)
    return str(int(m.group(1))) if m else s


def clean_raw_columns(data_raw):
    rename = {c: _parse(c) for c in data_raw.columns if _parse(c) is not None}
    missing = [c for c in data_raw.columns if c not in rename]
    if missing:
        print("  dropping unparsed columns: %s" % missing, flush=True)
    out = data_raw.rename(columns=rename)[list(rename.values())].copy()
    return out.loc[:, ~out.columns.duplicated()].apply(pd.to_numeric, errors="coerce")


def _structs(raw, specs, avail, key=str):
    built, skipped = [], []
    for name, *legs in specs:
        if not all(key(x) in avail for x in legs):
            skipped.append(name)
            continue
        c = [raw[key(x)] for x in legs]
        s = (c[1] - c[0]) * 100.0 if len(c) == 2 else (2.0 * c[1] - c[0] - c[2]) * 100.0
        built.append(s.rename(name))
    return built, skipped


def build_full_data(
    data_raw,
    curves=None,
    flies=None,
    fwd_curves=FWD_CURVES,
    fwd_flies=FWD_FLIES,
    clean=True,
    prefer_bbg=None,
    force_compute=False,
):
    curves = list(curves) if curves is not None else all_curves()
    flies = list(flies) if flies is not None else all_flies()
    """
    Build tenors + swap curves/flies + fwd structures.

    prefer_bbg (default USE_BBG_SWAP_STRUCTURES): use BBG USOS/UF for CURVES/FLIES
    when present; only compute missing swap structures from USOSFR outrights.
    FWD_CURVES / FWD_FLIES are always computed from forward points (never BBG-listed).
    force_compute: ignore BBG USOS/UF cols and always compute swap structs (cross-check).
    """
    raw = clean_raw_columns(data_raw) if clean else data_raw.copy()
    prefer_bbg = USE_BBG_SWAP_STRUCTURES if prefer_bbg is None else bool(prefer_bbg)
    tenors = {c for c in raw.columns if str(c).isdigit()}
    fwds = {c for c in raw.columns if _FWD_CLEAN_RE.fullmatch(str(c))}
    curve_names = [s[0] for s in curves]
    fly_names = [s[0] for s in flies]

    if force_compute:
        drop = [n for n in curve_names + fly_names if n in raw.columns]
        if drop:
            raw = raw.drop(columns=drop)
        prefer_bbg = False

    bbg_cur, bbg_fly = [], []
    missing_curves, missing_flies = list(curves), list(flies)
    if prefer_bbg:
        have_c = [n for n in curve_names if n in raw.columns]
        have_f = [n for n in fly_names if n in raw.columns]
        bbg_cur = [raw[n].rename(n) for n in have_c]
        bbg_fly = [raw[n].rename(n) for n in have_f]
        missing_curves = [s for s in curves if s[0] not in raw.columns]
        missing_flies = [s for s in flies if s[0] not in raw.columns]
        if missing_curves or missing_flies:
            print(
                "  [structures] compute missing: %s"
                % ([s[0] for s in missing_curves + missing_flies] or "—"),
                flush=True,
            )

    cur, sk_c = _structs(raw, missing_curves, tenors)
    fly, sk_f = _structs(raw, missing_flies, tenors)
    if sk_c or sk_f:
        print(
            "  skip structures (missing tenors): curves=%s  flies=%s  have_tenors=%s"
            % (sk_c or "—", sk_f or "—", sorted(int(t) for t in tenors)), flush=True,
        )
    fcur, sk_fc = _structs(raw, fwd_curves, fwds)
    ffly, sk_ff = _structs(raw, fwd_flies, fwds)
    if sk_fc or sk_ff:
        print(
            "  skip fwd structures (missing forwards): curves=%s  flies=%s"
            % (sk_fc or "—", sk_ff or "—"), flush=True,
        )

    def _fkey(n):
        m = _FWD_CLEAN_RE.fullmatch(n)
        return (int(m.group(2)), int(m.group(1)))

    # Cross-assets + any USSFCT* / other extras already cleaned in raw
    struct_names = set(curve_names) | set(fly_names)
    xa_names = [n for n, _ in CROSS_ASSET if n in raw.columns]
    ussfct = sorted(
        c for c in raw.columns
        if str(c).upper().startswith("USSFCT") and c not in xa_names
    )
    kept = set(tenors) | set(fwds) | struct_names | set(xa_names) | set(ussfct)
    extras = [c for c in raw.columns if c not in kept]
    # Prefer BBG series; computed fills gaps; keep cross-assets / USSFCT / extras
    pieces = (
        [raw[t].rename(t) for t in sorted(tenors, key=int)]
        + bbg_cur + cur + bbg_fly + fly
        + [raw[c].rename(c) for c in sorted(fwds, key=_fkey)] + fcur + ffly
        + [raw[n].rename(n) for n in xa_names]
        + [raw[n].rename(n) for n in ussfct]
        + [raw[n].rename(n) for n in extras]
    )
    if not pieces:
        raise ValueError("No structures could be built — check raw tenors/forwards")
    full = pd.concat(pieces, axis=1)
    full = full.loc[:, ~full.columns.duplicated(keep="first")]
    full.index = raw.index
    return full


def build_predicted_data(full_data, predicted=PREDICTED):
    missing = [c for c in predicted if c not in full_data.columns]
    if missing:
        print("  skip predicted targets not in full_data: %s" % missing, flush=True)
    cols = [c for c in predicted if c in full_data.columns]
    if not cols:
        raise KeyError(
            "None of the PREDICTED targets are in full_data. Wanted %s; have %s"
            % (list(predicted), list(full_data.columns))
        )
    return full_data.loc[:, cols].copy()


def build_datasets(data_raw, curves=None, flies=None, predicted=PREDICTED):
    cleaned = clean_raw_columns(data_raw)
    full = build_full_data(cleaned, curves=curves, flies=flies, clean=False)
    return cleaned, full, build_predicted_data(full, predicted=predicted)


def _quiet_bbg_logs():
    logging.disable(logging.WARNING)
    root = logging.getLogger()
    root.setLevel(logging.ERROR)
    for h in list(root.handlers):
        h.setLevel(logging.ERROR)
    for name in set(logging.root.manager.loggerDict):
        if any(k in str(name).lower() for k in _BBG_LOG_KEYS):
            lg = logging.getLogger(name)
            lg.handlers, lg.propagate, lg.disabled = [], False, True
            lg.setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")


def _securities():
    tenors = set()
    for _, *legs in list(all_curves()) + list(all_flies()):
        tenors.update(legs)
    secs = (
        ["%s%d %s" % (BBG_PREFIX, int(t), BBG_SUFFIX) for t in sorted(tenors)]
        + ["%s %dY%dY %s" % (BBG_FWD_PREFIX, int(s), int(g), BBG_FWD_SUFFIX) for s, g in FORWARDS]
    )
    if USE_BBG_SWAP_STRUCTURES:
        secs = secs + structure_securities()
    secs = secs + [t for _, t in CROSS_ASSET]
    # de-dupe preserving order
    seen, out = set(), []
    for s in secs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _try_import_bbgapi():
    import importlib
    importlib.invalidate_caches()
    sys.modules.pop("bbgapi", None)
    import bbgapi  # noqa: F401
    return bbgapi


def _discover_sites(pkg_name):
    tag = "python%d.%d" % sys.version_info[:2]
    patterns = [
        "/ms/dist/python/PROJ/%s/*/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/python/PROJ/%s/*/exec/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/datazone/PROJ/%s/*/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/datazone/PROJ/%s/*/exec/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/datazone/PROJ/%s/*/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/python/PROJ/%s/*/%s/site-packages" % (pkg_name, tag),
    ]
    found, marker = [], pkg_name.replace("-", "_")
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if d not in found and (
                Path(d, marker).is_dir() or Path(d, pkg_name).is_dir()
                or Path(d, marker + ".py").is_file()
            ):
                found.append(d)
    return found


def _addpkg_versions(ms_version, name, versions, kw=None):
    kw, import_name = kw or {}, name.replace("-", "_")
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    for ver in versions:
        try:
            try:
                ms_version.addpkg(name, ver, **kw)
            except TypeError:
                ms_version.addpkg(name, ver)
        except Exception as exc:
            print("  [bbgapi] skip dep %s %s: %s" % (name, ver, exc), flush=True)
            continue
        try:
            sys.modules.pop(import_name, None)
            __import__(import_name)
            return True
        except ImportError:
            continue
    for site in _discover_sites(name):
        if site not in sys.path:
            sys.path.insert(0, site)
        try:
            sys.modules.pop(import_name, None)
            __import__(import_name)
            print("  [bbgapi] %s from %s" % (name, site), flush=True)
            return True
        except ImportError:
            continue
    print("  [bbgapi] could not load dep %s" % name, flush=True)
    return False


def _V(*parts):
    return tuple(x for p in parts for x in p.split() if x)


_DEPS = (
    ("urllib3", _V("prod latest 1.26.7 1.26.6 1.26.5 1.26.4 1.25.11 1.25.10 1.24.3"), {}),
    ("certifi", _V("prod latest 2021.10.8 2021.5.30 2020.12.5 2020.6.20",
                   "2019.11.28 2022.12.7 2022.9.24 2023.7.22"), {}),
    ("idna", _V("prod latest 3.3 3.2 2.10 2.9 2.8"), {}),
    ("chardet", _V("prod latest 4.0.0 3.0.4"), {}),
    ("charset-normalizer", _V("prod latest 2.0.7 2.0.12 2.0.4"), {}),
    ("blpapi", _V("prod latest 3.16.5 3.15.2"), {"shadow": True}),
    ("toml", _V("prod latest 0.10.1 0.10.2"), {}),
    ("requests", _V("prod latest 2.26.0 2.25.1 2.22.0 2.21.0"), {"shadow": True}),
    ("retrying", _V("prod latest 1.3.3"), {}),
    ("requests-kerberos", _V("prod latest 0.14.0 0.12.0"), {}),
    ("pyspnego", _V("prod latest 0.3.1 0.1.5"), {}),
    ("cryptography", _V("prod latest 3.4.8 3.3.2"), {}),
)
_DEPS_UNIX = (
    ("gssapi", _V("1.6.5 1.6.1 prod latest"), {}),
    ("krb5", _V("0.2.0 0.1.2 prod latest"), {}),
    ("python-scvlib", _V("1.0.25 prod latest"), {"meta": "sec"}),
    ("cffi", _V("1.14.5 1.14.0 prod latest"), {}),
    ("PyJWT", _V("1.5.3 1.7.1 prod latest"), {}),
)


def _add_bbgapi_deps(ms_version):
    for name, versions, kw in _DEPS + (() if sys.platform == "win32" else _DEPS_UNIX):
        _addpkg_versions(ms_version, name, versions, kw)


def _ensure_bbgapi():
    try:
        _try_import_bbgapi()
        return
    except ImportError:
        pass
    path = str(BBGAPI_PATH or "").strip()
    if path:
        p = Path(path)
        if p.is_file():
            p = p.parent
        if p.name == "bbgapi":
            p = p.parent
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        try:
            _try_import_bbgapi()
            print("  [bbgapi] loaded from BBGAPI_PATH=%s" % p, flush=True)
            return
        except ImportError:
            pass
    for site in _discover_sites("bbgapi"):
        if site not in sys.path:
            sys.path.insert(0, site)
        try:
            _try_import_bbgapi()
            print("  [bbgapi] loaded from %s" % site, flush=True)
            return
        except ImportError:
            continue
    try:
        import ms.version
    except ImportError as exc:
        raise ImportError(
            "bbgapi not installed and ms.version unavailable. "
            "Use: python run_dz.py prediction --once --pull-history "
            "(re-execs under Python 3.7). Or set curve_config.BBGAPI_PATH."
        ) from exc
    last_err, versions = None, []
    for v in (BBGAPI_VERSION, "prod", "latest", "2022.09.26-2"):
        if v and v not in versions:
            versions.append(v)
    for ver in versions:
        try:
            print("  [bbgapi] ms.version.addpkg  v=%r  py=%s.%s" % (
                ver, sys.version_info[0], sys.version_info[1]), flush=True)
            _add_bbgapi_deps(ms.version)
            try:
                ms.version.addpkg("bbgapi", ver, meta="datazone")
            except TypeError:
                ms.version.addpkg("bbgapi", ver)
            _try_import_bbgapi()
            print("  [bbgapi] import OK  v=%r" % (ver,), flush=True)
            return
        except Exception as exc:
            last_err = exc
            print("  [bbgapi] failed v=%r: %s" % (ver, exc), flush=True)
    raise ImportError(
        "Could not import bbgapi (Python %s.%s; desk usually needs 3.7). "
        "Run: python run_dz.py prediction --once --pull-history "
        "or set curve_config.BBGAPI_PATH. Last error: %r"
        % (sys.version_info[0], sys.version_info[1], last_err)
    ) from last_err


def _bquery():
    global _BQ
    _quiet_bbg_logs()
    if _BQ is not None and hasattr(_BQ, "bdib"):
        return _BQ
    _ensure_bbgapi()
    from bbgapi.query import BlpQuery
    bq = BlpQuery()
    started = bq.start()
    client = started if started is not None else bq  # start() may return None
    if client is None or not hasattr(client, "bdib"):
        raise RuntimeError("BlpQuery().start() did not yield .bdib client (got %r)" % (started,))
    _BQ = client
    print("  [bbgapi] BlpQuery ready  type=%s  file=%s" % (
        type(_BQ).__name__, getattr(sys.modules.get("bbgapi"), "__file__", "?")), flush=True)
    _quiet_bbg_logs()
    return _BQ


def _as_dt(value, default_now=False):
    text = "" if value is None else str(value).strip()
    if not text:
        if default_now:
            return pd.Timestamp.now().floor("min")
        raise ValueError("empty Bloomberg datetime")
    return pd.Timestamp(text)


def _interval_minutes(freq=None):
    if freq is None:
        return int(BBG_INTERVAL_MINUTES)
    if isinstance(freq, (int, float)):
        return int(freq)
    f = str(freq).strip().lower()
    if f.endswith("min"):
        return int(f[:-3])
    if f.endswith("h"):
        return int(float(f[:-1]) * 60)
    if f in {"1d", "d", "daily", "1day"}:
        return 24 * 60
    return int(BBG_INTERVAL_MINUTES)


def _series_from_bars(df, security, time_col="time", value_col="close"):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if time_col not in frame.columns or value_col not in frame.columns:
        raise ValueError(
            "bars(%s) expected %s/%s, got %s" % (security, time_col, value_col, list(frame.columns))
        )
    s = pd.to_numeric(frame.set_index(time_col)[value_col], errors="coerce").rename(security)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.index.name = "time"
    return s.sort_index().dropna()


def _bdib_close(bq, security, start, end, interval):
    _quiet_bbg_logs()
    bars = bq.bdib(
        security=security, event_type=BBG_EVENT_TYPE, interval=int(interval),
        start_datetime=start.strftime(_DT_FMT), end_datetime=end.strftime(_DT_FMT),
    )
    return _series_from_bars(bars, security)


def _is_entitlement_error(exc):
    text = str(exc)
    return (
        "NOT_ENTITLED" in text
        or "Entitlement" in text
        or "entitlement" in text.lower()
    )


def _bdh_px_last(bq, security, start, end):
    """
    Daily PX_LAST via bdh — often entitled when intradaily bdib is NOT_ENTITLED
    (common for CO1/CL1 Comdty on desk feeds).
    """
    if not hasattr(bq, "bdh"):
        raise RuntimeError("BlpQuery has no .bdh — cannot fallback from bdib")
    _quiet_bbg_logs()
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    # Try common bbgapi signatures
    last_err = None
    raw = None
    attempts = [
        lambda: bq.bdh(security, ["PX_LAST"], start_s, end_s),
        lambda: bq.bdh([security], ["PX_LAST"], start_s, end_s),
        lambda: bq.bdh(
            securities=[security], fields=["PX_LAST"],
            start_date=start_s, end_date=end_s,
        ),
        lambda: bq.bdh(
            security=security, field="PX_LAST",
            start_date=start_s, end_date=end_s,
        ),
    ]
    for fn in attempts:
        try:
            raw = fn()
            if raw is not None:
                break
        except TypeError as exc:
            last_err = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _is_entitlement_error(exc):
                raise
            continue
    if raw is None:
        raise RuntimeError("bdh(%s) failed: %s" % (security, last_err))

    if isinstance(raw, pd.Series):
        s = pd.to_numeric(raw, errors="coerce").rename(security)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None)
        s.index.name = "time"
        return s.sort_index().dropna()

    df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    if df.empty:
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # prefer PX_LAST level / matching security
        try:
            if security in df.columns.get_level_values(0):
                df = df[security]
            elif "PX_LAST" in df.columns.get_level_values(-1):
                df = df.xs("PX_LAST", axis=1, level=-1)
        except Exception:  # noqa: BLE001
            df.columns = ["_".join(str(x) for x in c if x is not None) for c in df.columns]

    if isinstance(df, pd.Series):
        s = pd.to_numeric(df, errors="coerce").rename(security)
    else:
        # date as index or column
        if not isinstance(df.index, pd.DatetimeIndex):
            for cand in ("date", "Date", "time", "TIME"):
                if cand in df.columns:
                    df = df.set_index(cand)
                    break
        val_col = None
        for cand in ("PX_LAST", "close", "Close", security):
            if cand in df.columns:
                val_col = cand
                break
        if val_col is None and df.shape[1] == 1:
            val_col = df.columns[0]
        if val_col is None:
            raise ValueError("bdh(%s) unexpected columns: %s" % (security, list(df.columns)))
        s = pd.to_numeric(df[val_col], errors="coerce").rename(security)

    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index.name = "time"
    return s.sort_index().dropna()


def _fetch_one_security(bq, security, start, end, interval):
    """bdib first; on NOT_ENTITLED / empty → daily bdh PX_LAST."""
    try:
        s = _bdib_close(bq, security, start, end, interval)
        if s is not None and not s.empty:
            return s, "bdib"
        print("  [bdib] empty %s — trying daily bdh" % security, flush=True)
    except Exception as exc:  # noqa: BLE001
        if _is_entitlement_error(exc):
            print(
                "  [bdib] NOT_ENTITLED %s (intraday) — trying daily bdh PX_LAST" % security,
                flush=True,
            )
        else:
            print("  [bdib] skip %s: %s — trying daily bdh" % (security, exc), flush=True)
    try:
        s = _bdh_px_last(bq, security, start, end)
        if s is not None and not s.empty:
            print(
                "  [bdh] %s OK  bars=%d  %s → %s (daily; will ffill into live grid)"
                % (security, len(s), s.index[0].date(), s.index[-1].date()),
                flush=True,
            )
            return s, "bdh"
        print("  [bdh] empty %s" % security, flush=True)
    except Exception as exc:  # noqa: BLE001
        print("  [bdh] skip %s: %s" % (security, exc), flush=True)
    return None, None


def _fetch_bdib_panel(start_date=None, end_date=None, freq=None, securities=None):
    start = _as_dt(BBG_START_DATE if start_date is None else start_date)
    end = _as_dt(BBG_END_DATE if end_date is None else end_date, default_now=True)
    interval = _interval_minutes(freq)
    names = list(securities) if securities is not None else _securities()
    bq = _bquery()
    print("  [bdib] %s → %s  interval=%sm  event=%s  n_securities=%d" % (
        start, end, interval, BBG_EVENT_TYPE, len(names)), flush=True)
    series_list = []
    for i, sec in enumerate(names, 1):
        s, src = _fetch_one_security(bq, sec, start, end, interval)
        if s is None or s.empty:
            continue
        series_list.append(s)
        if i in (1, len(names)) or i % 10 == 0:
            print(
                "  [pull] %d/%d  last=%s  bars=%d  via=%s"
                % (i, len(names), sec, len(s), src or "?"),
                flush=True,
            )
    if not series_list:
        raise ValueError("bdib/bdh returned no series for any security")
    panel = pd.concat(series_list, axis=1).sort_index()
    panel.index.name = "time"
    print("  [pull] panel n=%d cols=%d  %s → %s" % (
        len(panel), panel.shape[1], panel.index[0], panel.index[-1]), flush=True)
    return panel


def _fetch_latest():
    end = pd.Timestamp.now().floor("min")
    panel = _fetch_bdib_panel(
        start_date=end - pd.Timedelta(days=5), end_date=end, freq=BBG_INTERVAL_MINUTES,
    )
    if panel.empty:
        raise ValueError("bdib latest window empty")
    return panel.iloc[[-1]]


def pull_history_into_store(path=None, start_date=None, end_date=None, freq=None):
    dest = path or LIVE_STORE_PATH
    hist = _fetch_bdib_panel(start_date=start_date, end_date=end_date, freq=freq)
    if hist.empty:
        raise ValueError("Bloomberg history pull returned no rows")
    save_live_store(hist, dest)
    print("  [history] wrote %s  n=%d  %s → %s" % (
        dest, len(hist), hist.index[0], hist.index[-1]), flush=True)
    return hist


def bbg_ticker_for_name(name):
    """Map clean panel name → BBG ticker (CROSS_ASSET / USSFCT / USOSFR digit)."""
    s = str(name).strip()
    # Already a full BBG ticker (e.g. user pasted "CL1 Comdty" into CORR_SERIES)
    if re.search(r"\s+(Comdty|Index|Curncy|Equity|Corp|Govt)\s*$", s, re.I):
        return s
    for clean, ticker in CROSS_ASSET:
        if clean == s or str(ticker).strip().lower() == s.lower():
            return ticker
    if re.fullmatch(r"USSFCT\d+", s, re.I):
        return "%s Curncy" % s.upper()
    if str(s).isdigit():
        return "%s%d %s" % (BBG_PREFIX, int(s), BBG_SUFFIX)
    # listed curve / fly
    for n, a, b in all_curves():
        if n == s:
            return bbg_curve_ticker(a, b)
    for n, a, b, c in all_flies():
        if n == s:
            return bbg_fly_ticker(a, b, c)
    return None


def ensure_series_in_live_store(names, path=None):
    """
    If clean names are missing from the live store, pull their BBG tickers
    (bdib intradaily, then bdh daily on NOT_ENTITLED) and merge.
    Returns the updated raw live panel.
    """
    dest = path or LIVE_STORE_PATH
    existing = load_live_store(dest) if Path(dest).exists() else pd.DataFrame()
    have = set(clean_raw_columns(existing).columns) if not existing.empty else set()
    need = [n for n in names if n not in have]
    if not need:
        return existing
    tickers, unresolved = [], []
    for n in need:
        t = bbg_ticker_for_name(n)
        if t:
            tickers.append(t)
        else:
            unresolved.append(n)
    if unresolved:
        print("  [ensure] no BBG ticker for: %s" % unresolved, flush=True)
    if not tickers:
        return existing
    print("  [ensure] pulling missing series: %s" % need, flush=True)
    try:
        extra = _fetch_bdib_panel(securities=tickers)
    except Exception as exc:  # noqa: BLE001
        print("  [ensure] pull failed: %s" % exc, flush=True)
        return existing
    if extra is None or extra.empty:
        print(
            "  [ensure] empty after bdib+bdh for %s "
            "(desk may lack commodity entitlement entirely)" % tickers,
            flush=True,
        )
        return existing
    return append_snapshot(extra, path=dest)


def print_live_levels(data_raw, src=""):
    if data_raw is None or data_raw.empty:
        print("  [levels] empty panel", flush=True)
        return
    cleaned = clean_raw_columns(data_raw)
    last, asof = cleaned.iloc[-1], cleaned.index[-1]
    tenors = sorted((c for c in cleaned.columns if str(c).isdigit()), key=int)
    fwds = [c for c in cleaned.columns if _FWD_CLEAN_RE.match(str(c))]
    structs = [s[0] for s in list(CURVES)[:4] + list(FLIES)[:4] if s[0] in cleaned.columns]
    show = tenors[:8] + structs + fwds[:4] or list(cleaned.columns[:12])
    parts = [
        "%s=nan" % c if pd.isna(last[c]) else "%s=%.4f" % (c, float(last[c])) for c in show
    ]
    tag = (" src=%s" % src) if src else ""
    print("  [levels] asof=%s  n=%d%s\n           %s" % (
        asof, len(cleaned), tag, "  ".join(parts)), flush=True)


def append_live_bar(max_bars=MAX_HISTORY_BARS):
    panel = append_snapshot(_fetch_latest(), path=LIVE_STORE_PATH, max_bars=max_bars)
    print("  [live] +1 → %s  n=%d  last=%s" % (
        LIVE_STORE_PATH, len(panel), panel.index[-1]), flush=True)
    return panel


def refresh_live_store(force_history=False):
    store = Path(LIVE_STORE_PATH)
    try:
        existing = load_live_store(store) if store.exists() else pd.DataFrame()
        if force_history or existing.empty:
            print("  [history] %s → %s  interval=%sm" % (
                BBG_START_DATE, BBG_END_DATE or "now", BBG_INTERVAL_MINUTES), flush=True)
            pull_history_into_store()
        return append_live_bar(), "live:%s" % store
    except Exception as exc:
        print("  [refresh] pull failed (%s); using existing panel" % exc, flush=True)
        if not store.exists():
            ensure_seed_history(store)
        data, src = load_panel()
        return data, "fallback:%s" % src


def load_panel():
    live, path = Path(LIVE_STORE_PATH), Path(DATA_PATH)
    if live.exists():
        return load_live_store(live), "live:%s" % live
    if path.exists():
        return load_raw(path), "file:%s" % path
    return generate_synthetic_sofr(n_bars=400, freq="%dmin" % int(DATA_BAR_MINUTES)), "synthetic"


def build_panel(data_raw, save_structures=True):
    if not isinstance(data_raw.index, pd.DatetimeIndex):
        raise TypeError("data_raw must be indexed by time")
    # Drop Sat/Sun + US holidays before ffill so we never invent holiday prints.
    raw = drop_non_trading_days(
        data_raw.apply(pd.to_numeric, errors="coerce")
    )
    cleaned, full, pred = build_datasets(raw.ffill())
    if save_structures:
        try:
            save_structures_store(full)
        except Exception as exc:  # noqa: BLE001
            print("  [structures] save failed: %s" % exc, flush=True)
    return cleaned, full, pred
