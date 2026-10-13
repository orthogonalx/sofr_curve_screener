"""Data layer: load/live store, structure building, Bloomberg pull, panel loaders."""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import curve_config as _cfg
from curve_config import (
    BBG_END_DATE, BBG_EVENT_TYPE, BBG_FWD_PREFIX, BBG_FWD_SUFFIX,
    BBG_INTERVAL_MINUTES, BBG_PREFIX, BBG_START_DATE, BBG_SUFFIX, CURVES,
    DATA_BAR_MINUTES, DATA_PATH, FLIES, FORWARDS, FWD_CURVES, FWD_FLIES,
    LIVE_STORE_PATH, MAX_HISTORY_BARS, PREDICTED,
)

# Optional knobs (older clones may not define them yet)
BBGAPI_VERSION = getattr(_cfg, "BBGAPI_VERSION", "2022.09.26-2")
BBGAPI_PATH = getattr(_cfg, "BBGAPI_PATH", "")

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

    # Load without ffill so we don't bake forward-filled values into disk twice
    path = Path(path)
    if path.exists():
        existing = load_raw(path, ffill=False)
    else:
        existing = pd.DataFrame()
    if existing.empty:
        combined = snap
    else:
        combined = pd.concat([existing, snap], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    # Partial BBG bars (e.g. forwards OK, USOSFR empty) → carry last known level
    combined = combined.apply(pd.to_numeric, errors="coerce").ffill()
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


# ── Bloomberg pull via bdib (desk Ticker Extract pattern) ─────────────────────
#   bquery = BlpQuery().start()
#   bars = bquery.bdib(security=sec, event_type="TRADE", interval=180,
#                      start_datetime=..., end_datetime=...)
#   close = bars.set_index("time")["close"]

_BQ = None
_DT_FMT = "%Y-%m-%dT%H:%M:%S"
_BBG_LOG_NAMES = (
    "bbgapi",
    "blpapi",
    "dz",
    "dzparts",
    "datazone",
    "urllib3",
    "requests",
    "ms",
    "BBGActivityTracker",
    "logging_service",
)


def _quiet_bbg_logs() -> None:
    """Silence bbgapi/dz INFO (pink/red blocks in Jupyter)."""
    # Hard stop: drop everything below ERROR process-wide
    logging.disable(logging.WARNING)
    root = logging.getLogger()
    root.setLevel(logging.ERROR)
    for h in list(root.handlers):
        h.setLevel(logging.ERROR)
    for name in set(_BBG_LOG_NAMES) | set(logging.root.manager.loggerDict):
        low = str(name).lower()
        if name in _BBG_LOG_NAMES or any(
            k in low for k in ("bbg", "blp", "dz", "datazone", "activity", "kerberos", "ms.")
        ):
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
            lg.disabled = True
    try:
        import warnings
        warnings.filterwarnings("ignore")
    except Exception:
        pass


def _securities() -> List[str]:
    tenors = set()
    for _, *legs in list(CURVES) + list(FLIES):
        tenors.update(legs)
    return ([f"{BBG_PREFIX}{int(t)} {BBG_SUFFIX}" for t in sorted(tenors)]
            + [f"{BBG_FWD_PREFIX} {int(s)}Y{int(g)}Y {BBG_FWD_SUFFIX}" for s, g in FORWARDS])


def _try_import_bbgapi():
    import importlib
    importlib.invalidate_caches()
    if "bbgapi" in sys.modules:
        del sys.modules["bbgapi"]
    import bbgapi  # noqa: F401
    return bbgapi


def _addpkg_versions(ms_version, name: str, versions, kw=None) -> bool:
    """Try several addpkg versions until import of `name` (or import alias) works."""
    kw = kw or {}
    # charset-normalizer imports as charset_normalizer
    import_name = name.replace("-", "_")
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
        except Exception as exc:  # noqa: BLE001
            print(f"  [bbgapi] skip dep {name} {ver}: {exc}", flush=True)
            continue
        try:
            if import_name in sys.modules:
                del sys.modules[import_name]
            __import__(import_name)
            return True
        except ImportError:
            continue
    # Disk fallback under /ms/dist
    import glob as _glob

    tag = "python%d.%d" % (sys.version_info[0], sys.version_info[1])
    for pat in (
        "/ms/dist/python/PROJ/%s/*/lib/%s/site-packages" % (name, tag),
        "/ms/dist/python/PROJ/%s/*/exec/lib/%s/site-packages" % (name, tag),
        "/ms/dist/datazone/PROJ/%s/*/lib/%s/site-packages" % (name, tag),
    ):
        for site in sorted(_glob.glob(pat)):
            if site not in sys.path:
                sys.path.insert(0, site)
            try:
                if import_name in sys.modules:
                    del sys.modules[import_name]
                __import__(import_name)
                print(f"  [bbgapi] {name} from {site}", flush=True)
                return True
            except ImportError:
                continue
    print(f"  [bbgapi] could not load dep {name}", flush=True)
    return False


def _add_bbgapi_deps(ms_version) -> None:
    """Deps from bbgapi_jumpstart.bbgapi_dependencies() (+ urllib3 stack)."""
    _addpkg_versions(
        ms_version,
        "urllib3",
        ("prod", "latest", "1.26.7", "1.26.6", "1.26.5", "1.26.4", "1.25.11", "1.25.10", "1.24.3"),
    )
    for name, versions, kw in (
        (
            "certifi",
            (
                "prod",
                "latest",
                "2021.10.8",
                "2021.5.30",
                "2020.12.5",
                "2020.6.20",
                "2019.11.28",
                "2022.12.7",
                "2022.9.24",
                "2023.7.22",
            ),
            {},
        ),
        ("idna", ("prod", "latest", "3.3", "3.2", "2.10", "2.9", "2.8"), {}),
        ("chardet", ("prod", "latest", "4.0.0", "3.0.4"), {}),
        ("charset-normalizer", ("prod", "latest", "2.0.7", "2.0.12", "2.0.4"), {}),
        ("blpapi", ("prod", "latest", "3.16.5", "3.15.2"), {"shadow": True}),
        ("toml", ("prod", "latest", "0.10.1", "0.10.2"), {}),
        ("requests", ("prod", "latest", "2.26.0", "2.25.1", "2.22.0", "2.21.0"), {"shadow": True}),
        ("retrying", ("prod", "latest", "1.3.3"), {}),
        ("requests-kerberos", ("prod", "latest", "0.14.0", "0.12.0"), {}),
        ("pyspnego", ("prod", "latest", "0.3.1", "0.1.5"), {}),
        ("cryptography", ("prod", "latest", "3.4.8", "3.3.2"), {}),
    ):
        _addpkg_versions(ms_version, name, versions, kw)
    if sys.platform != "win32":
        for name, versions, kw in (
            ("gssapi", ("1.6.5", "1.6.1", "prod", "latest"), {}),
            ("krb5", ("0.2.0", "0.1.2", "prod", "latest"), {}),
            ("python-scvlib", ("1.0.25", "prod", "latest"), {"meta": "sec"}),
            ("cffi", ("1.14.5", "1.14.0", "prod", "latest"), {}),
            ("PyJWT", ("1.5.3", "1.7.1", "prod", "latest"), {}),
        ):
            _addpkg_versions(ms_version, name, versions, kw)


def _discover_bbgapi_sites() -> List[str]:
    """Find bbgapi site-packages dirs under /ms/dist for this Python version."""
    import glob as _glob

    tag = "python%d.%d" % (sys.version_info[0], sys.version_info[1])
    patterns = [
        "/ms/dist/datazone/PROJ/bbgapi/*/lib/%s/site-packages" % tag,
        "/ms/dist/datazone/PROJ/bbgapi/*/exec/lib/%s/site-packages" % tag,
        "/ms/dist/datazone/PROJ/bbgapi/*/%s/site-packages" % tag,
        "/ms/dist/python/PROJ/bbgapi/*/lib/%s/site-packages" % tag,
        "/ms/dist/python/PROJ/bbgapi/*/exec/lib/%s/site-packages" % tag,
    ]
    found: List[str] = []
    for pat in patterns:
        for d in sorted(_glob.glob(pat)):
            pkg = Path(d) / "bbgapi"
            if (pkg.is_dir() or Path(d, "bbgapi.py").is_file()) and d not in found:
                found.append(d)
    return found


def _ensure_bbgapi() -> None:
    """
    Load bbgapi for terminal/notebook on DataZone.

    Order:
      1) already importable
      2) optional BBGAPI_PATH
      3) filesystem discovery under /ms/dist/.../bbgapi (matching py version)
      4) ms.version.addpkg (several versions) + jumpstart deps
    """
    try:
        _try_import_bbgapi()
        return
    except ImportError:
        pass

    # Manual path override (from notebook: print(bbgapi.__file__))
    path = str(BBGAPI_PATH or "").strip()
    if path:
        p = Path(path)
        if p.is_file():
            p = p.parent
        # if user pasted .../bbgapi/__init__.py or .../bbgapi, go to site-packages
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

    # Auto-discover matching this interpreter (3.10 won't see py3.7 wheels)
    for site in _discover_bbgapi_sites():
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
            "bbgapi not installed and ms.version unavailable.\n"
            "Use:  python run_dz.py prediction --once --pull-history\n"
            "(it re-execs under Python 3.7). Or set curve_config.BBGAPI_PATH."
        ) from exc

    last_err: Optional[BaseException] = None
    versions = []
    for v in (BBGAPI_VERSION, "prod", "latest", "2022.09.26-2"):
        if v and v not in versions:
            versions.append(v)

    for ver in versions:
        try:
            print("  [bbgapi] ms.version.addpkg  v=%r  py=%s.%s" % (
                ver, sys.version_info[0], sys.version_info[1]
            ), flush=True)
            _add_bbgapi_deps(ms.version)
            try:
                ms.version.addpkg("bbgapi", ver, meta="datazone")
            except TypeError:
                ms.version.addpkg("bbgapi", ver)
            _try_import_bbgapi()
            print("  [bbgapi] import OK  v=%r" % (ver,), flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print("  [bbgapi] failed v=%r: %s" % (ver, exc), flush=True)

    raise ImportError(
        "Could not import bbgapi.\n"
        "You are on Python %s.%s — desk bbgapi is usually Python 3.7 only.\n"
        "Run:\n"
        "  python run_dz.py prediction --once --pull-history\n"
        "Or in a working notebook:\n"
        "  import sys, bbgapi; print(sys.executable); print(bbgapi.__file__)\n"
        "Then set curve_config.BBGAPI_PATH to that site-packages dir.\n"
        "Last error: %r" % (sys.version_info[0], sys.version_info[1], last_err)
    ) from last_err


def _bquery():
    """Return a started BlpQuery. start() often returns None (mutates in place)."""
    global _BQ
    _quiet_bbg_logs()
    if _BQ is not None and hasattr(_BQ, "bdib"):
        return _BQ
    _ensure_bbgapi()
    from bbgapi.query import BlpQuery

    bq = BlpQuery()
    started = bq.start()
    # Desk jumpstart: BlpQuery().start() → self; some builds return None
    client = started if started is not None else bq
    if client is None or not hasattr(client, "bdib"):
        raise RuntimeError(
            "BlpQuery().start() did not yield a client with .bdib "
            "(got %r). Check bbgapi version / session." % (started,)
        )
    _BQ = client
    print(
        "  [bbgapi] BlpQuery ready  type=%s  file=%s"
        % (type(_BQ).__name__, getattr(sys.modules.get("bbgapi"), "__file__", "?")),
        flush=True,
    )
    _quiet_bbg_logs()
    return _BQ


def _as_dt(value, default_now: bool = False) -> pd.Timestamp:
    text = "" if value is None else str(value).strip()
    if not text:
        if default_now:
            return pd.Timestamp.now().floor("min")
        raise ValueError("empty Bloomberg datetime")
    return pd.Timestamp(text)


def _interval_minutes(freq=None) -> int:
    """Map config / override → bdib interval minutes."""
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


def _bdib_close(bq, security: str, start: pd.Timestamp, end: pd.Timestamp,
                interval: int) -> Optional[pd.Series]:
    """One ticker → close series indexed by time (desk extract)."""
    _quiet_bbg_logs()
    bars = bq.bdib(
        security=security,
        event_type=BBG_EVENT_TYPE,
        interval=int(interval),
        start_datetime=start.strftime(_DT_FMT),
        end_datetime=end.strftime(_DT_FMT),
    )
    if bars is None or (isinstance(bars, pd.DataFrame) and bars.empty):
        return None
    df = bars if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "time" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"bdib({security}) expected columns time/close, got {list(df.columns)}"
        )
    s = pd.to_numeric(df.set_index("time")["close"], errors="coerce").rename(security)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s.index.name = "time"
    return s.sort_index()


def _fetch_bdib_panel(start_date=None, end_date=None, freq=None,
                      securities: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Loop securities with bdib → wide panel (index=time, columns=tickers).
    Same logic as the working Ticker Extract notebook.
    """
    start = _as_dt(BBG_START_DATE if start_date is None else start_date)
    end = _as_dt(BBG_END_DATE if end_date is None else end_date, default_now=True)
    interval = _interval_minutes(freq)
    names = list(securities) if securities is not None else _securities()
    bq = _bquery()
    print(
        f"  [bdib] {start} → {end}  interval={interval}m  event={BBG_EVENT_TYPE}  "
        f"n_securities={len(names)}",
        flush=True,
    )
    series_list: List[pd.Series] = []
    for i, sec in enumerate(names, 1):
        try:
            s = _bdib_close(bq, sec, start, end, interval)
        except Exception as exc:  # noqa: BLE001
            print(f"  [bdib] skip {sec}: {exc}", flush=True)
            continue
        if s is None or s.empty:
            print(f"  [bdib] empty {sec}", flush=True)
            continue
        series_list.append(s)
        if i == 1 or i == len(names) or i % 10 == 0:
            print(f"  [bdib] {i}/{len(names)}  last={sec}  bars={len(s)}", flush=True)
    if not series_list:
        raise ValueError("bdib returned no series for any security")
    panel = pd.concat(series_list, axis=1).sort_index()
    panel.index.name = "time"
    print(
        f"  [bdib] panel n={len(panel)} cols={panel.shape[1]}  "
        f"{panel.index[0]} → {panel.index[-1]}",
        flush=True,
    )
    return panel


def _fetch_history(start_date=None, end_date=None, freq=None) -> pd.DataFrame:
    return _fetch_bdib_panel(start_date=start_date, end_date=end_date, freq=freq)


def _fetch_latest() -> pd.DataFrame:
    """Latest bar: short bdib window → last row (same close source as history)."""
    end = pd.Timestamp.now().floor("min")
    start = end - pd.Timedelta(days=5)
    panel = _fetch_bdib_panel(start_date=start, end_date=end, freq=BBG_INTERVAL_MINUTES)
    if panel.empty:
        raise ValueError("bdib latest window empty")
    return panel.iloc[[-1]]


def pull_history_into_store(path: Optional[str] = None, start_date=None, end_date=None,
                            freq=None) -> pd.DataFrame:
    """Pull history via bdib and write LIVE_STORE_PATH."""
    dest = path or LIVE_STORE_PATH
    hist = _fetch_history(start_date, end_date, freq)
    if hist.empty:
        raise ValueError("Bloomberg history pull returned no rows")
    save_live_store(hist, dest)
    print(f"  [history] wrote {dest}  n={len(hist)}  {hist.index[0]} → {hist.index[-1]}", flush=True)
    return hist


def print_live_levels(data_raw: pd.DataFrame, src: str = "") -> None:
    """Print last-bar outright/forward levels so a live pull is obvious."""
    if data_raw is None or data_raw.empty:
        print("  [levels] empty panel", flush=True)
        return
    cleaned = clean_raw_columns(data_raw)
    last = cleaned.iloc[-1]
    asof = cleaned.index[-1]
    # Prefer clean tenors (2,5,10,…) then a few forwards
    tenor_cols = [c for c in cleaned.columns if str(c).isdigit()]
    tenor_cols = sorted(tenor_cols, key=lambda x: int(x))
    fwd_cols = [c for c in cleaned.columns if _FWD_CLEAN_RE.match(str(c))]
    # show all tenors + a short fwd sample (first 8)
    show_cols = tenor_cols + fwd_cols[:8]
    if not show_cols:
        show_cols = list(cleaned.columns[:12])
    parts = []
    for c in show_cols:
        v = last[c]
        if pd.isna(v):
            parts.append("%s=nan" % c)
        else:
            parts.append("%s=%.4f" % (c, float(v)))
    tag = (" src=%s" % src) if src else ""
    print(
        "  [levels] asof=%s  n=%d%s\n           %s"
        % (asof, len(cleaned), tag, "  ".join(parts)),
        flush=True,
    )


def append_live_bar(max_bars: int = MAX_HISTORY_BARS) -> pd.DataFrame:
    """Pull latest bdib bar and append onto the live store."""
    panel = append_snapshot(_fetch_latest(), path=LIVE_STORE_PATH, max_bars=max_bars)
    print(f"  [live] +1 → {LIVE_STORE_PATH}  n={len(panel)}  last={panel.index[-1]}", flush=True)
    return panel


def refresh_live_store(force_history: bool = False) -> Tuple[pd.DataFrame, str]:
    """History if store empty/forced, then append latest bar; fallback to load_panel."""
    store = Path(LIVE_STORE_PATH)
    try:
        existing = load_live_store(store) if store.exists() else pd.DataFrame()
        if force_history or existing.empty:
            print(
                f"  [history] {BBG_START_DATE} → {BBG_END_DATE or 'now'}  "
                f"interval={BBG_INTERVAL_MINUTES}m",
                flush=True,
            )
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
    return generate_synthetic_sofr(n_bars=400, freq=f"{int(DATA_BAR_MINUTES)}min"), "synthetic"


def build_panel(data_raw: pd.DataFrame):
    """raw → (cleaned, full_data, predicted_data)."""
    if not isinstance(data_raw.index, pd.DatetimeIndex):
        raise TypeError("data_raw must be indexed by time")
    return build_datasets(data_raw.apply(pd.to_numeric, errors="coerce").ffill())
