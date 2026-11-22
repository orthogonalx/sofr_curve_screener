#!/usr/bin/env python3
"""
Offline CSV analysis — no Bloomberg.

Load a BBG Excel/CSV dump (USOSFR outrights + USSFCT + extras), clean columns,
compute CURVES/FLIES (bp), export a analysis panel, and plot starter metrics.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from curve_config import CURVES, FLIES

# Expected extras in the dump (clean name → accepted raw patterns)
EXTRA_MAP = {
    "ESA": ("ESA Index", "ESA"),
    "CO1": ("CO1 Comdty", "CO1"),
    "XAU": ("XAU Comdty", "XAU"),
}
SPREAD_NAMES = ("USSFCT05", "USSFCT10", "USSFCT30")
DEFAULT_CSV = "data_11_22.csv"
OUT_DIR = Path("exports/csv_analysis")


def _norm_key(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def parse_column(col) -> Optional[str]:
    """Raw BBG / messy header → clean panel name."""
    s = str(col).strip()
    if not s:
        return None
    # USOSFR{N} [BGN] Curncy  OR bare USOSFR{N}
    m = re.match(r"^USOSFR\s*(\d+)(?:\s+BGN)?(?:\s+Curncy)?$", s, re.I)
    if m:
        return str(int(m.group(1)))
    # already a tenor digit
    if re.fullmatch(r"\d{1,2}", s):
        return str(int(s))
    # USSFCT
    m = re.match(r"^USSFCT\s*(\d+)\s*(?:Curncy)?$", s, re.I)
    if m:
        return "USSFCT%02d" % int(m.group(1))
    if re.fullmatch(r"USSFCT\d+", s, re.I):
        return s.upper()
    # extras
    key = _norm_key(s)
    for clean, aliases in EXTRA_MAP.items():
        if key in {_norm_key(a) for a in aliases} or key == clean.lower():
            return clean
    return None


def load_csv(path=DEFAULT_CSV, ffill: bool = True) -> pd.DataFrame:
    """Read CSV/Excel; index = time; keep raw columns (not yet cleaned)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            "Missing %s — put the BBG dump in the repo root (cwd=%s)"
            % (p.resolve(), Path.cwd())
        )
    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    # timestamp column
    ts_col = next(
        (
            c
            for c in df.columns
            if str(c).strip().lower() in {
                "time", "date", "datetime", "timestamp", "dates", "unnamed: 0",
            }
        ),
        df.columns[0],
    )
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    df.index.name = "time"
    return df


def clean_panel(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Rename/keep USOSFR tenors, USSFCT spreads, ESA/CO1/XAU.
    Returns cleaned numeric panel + report dict.
    """
    rename, skipped = {}, []
    for c in raw.columns:
        name = parse_column(c)
        if name is None:
            skipped.append(str(c))
        else:
            rename[c] = name
    out = raw.rename(columns=rename)[list(rename.values())].copy()
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    out = out.apply(pd.to_numeric, errors="coerce")
    # drop all-NaN columns
    out = out.dropna(axis=1, how="all")
    tenors = sorted((c for c in out.columns if str(c).isdigit()), key=lambda x: int(x))
    spreads = [c for c in SPREAD_NAMES if c in out.columns]
    extras = [c for c in EXTRA_MAP if c in out.columns]
    report = {
        "n_rows": len(out),
        "asof": out.index[-1] if len(out) else None,
        "start": out.index[0] if len(out) else None,
        "tenors": tenors,
        "spreads": spreads,
        "extras": extras,
        "skipped": skipped,
        "n_cols": out.shape[1],
    }
    print(
        "  [clean] %s → %s  n=%d  tenors=%d  spreads=%s  extras=%s"
        % (report["start"], report["asof"], report["n_rows"], len(tenors), spreads, extras),
        flush=True,
    )
    if skipped:
        print("  [clean] skipped columns: %s" % skipped, flush=True)
    missing_t = [str(t) for t in range(1, 51) if str(t) not in out.columns]
    if missing_t:
        print("  [clean] missing USOSFR tenors: %s" % missing_t, flush=True)
    return out, report


def compute_structures(
    rates: pd.DataFrame,
    curves=None,
    flies=None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Curves: (long - short) * 100  → bp
    Flies:  (2*body - left - right) * 100 → bp
    Rates assumed in percent.
    """
    curves = list(curves if curves is not None else CURVES)
    flies = list(flies if flies is not None else FLIES)
    tenors = {c for c in rates.columns if str(c).isdigit()}
    built, skipped = [], []
    for name, a, b in curves:
        ca, cb = str(a), str(b)
        if ca not in tenors or cb not in tenors:
            skipped.append(name)
            continue
        built.append(((rates[cb] - rates[ca]) * 100.0).rename(name))
    for name, a, b, c in flies:
        legs = [str(a), str(b), str(c)]
        if not all(x in tenors for x in legs):
            skipped.append(name)
            continue
        built.append(
            ((2.0 * rates[legs[1]] - rates[legs[0]] - rates[legs[2]]) * 100.0).rename(name)
        )
    if skipped:
        print("  [structs] skipped (missing tenors): %s" % skipped, flush=True)
    if not built:
        return pd.DataFrame(index=rates.index), skipped
    return pd.concat(built, axis=1), skipped


def build_analysis_panel(cleaned: pd.DataFrame) -> pd.DataFrame:
    """rates + computed curves/flies + spreads + extras."""
    tenors = [c for c in cleaned.columns if str(c).isdigit()]
    rates = cleaned[tenors].copy()
    structs, _ = compute_structures(rates)
    spreads = [c for c in SPREAD_NAMES if c in cleaned.columns]
    extras = [c for c in EXTRA_MAP if c in cleaned.columns]
    parts = [rates]
    if not structs.empty:
        parts.append(structs)
    if spreads:
        parts.append(cleaned[spreads])
    if extras:
        parts.append(cleaned[extras])
    panel = pd.concat(parts, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated(keep="last")].sort_index()
    panel.index.name = "time"
    print(
        "  [panel] n=%d  cols=%d  %s → %s"
        % (len(panel), panel.shape[1], panel.index[0], panel.index[-1]),
        flush=True,
    )
    return panel


def save_panel(panel: pd.DataFrame, out_dir=OUT_DIR, stem="analysis_panel") -> Path:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    csv = dest / ("%s.csv" % stem)
    panel.to_csv(csv)
    print("  [panel] wrote %s" % csv.resolve(), flush=True)
    try:
        xlsx = dest / ("%s.xlsx" % stem)
        panel.to_excel(xlsx)
        print("  [panel] wrote %s" % xlsx.resolve(), flush=True)
    except ImportError:
        pass
    return csv.resolve()


# ── metrics / plots ──────────────────────────────────────────────────────────

KEY_CURVES = ["2s5s", "5s10s", "10s30s", "20s30s"]
KEY_FLIES = ["2s5s10s", "5s7s10s", "5s10s30s", "10s20s30s", "3s4s5s"]
KEY_RATES = ["2", "5", "10", "30"]


def _ensure_dir(out_dir=OUT_DIR) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def recent_moves_table(
    panel: pd.DataFrame,
    cols: Sequence[str] = None,
    windows=("1d", "5d", "21d"),
) -> pd.DataFrame:
    """Change over calendar-ish windows (uses last available bar as end)."""
    cols = [c for c in (cols or list(KEY_RATES) + KEY_CURVES + KEY_FLIES + list(SPREAD_NAMES) + list(EXTRA_MAP)) if c in panel.columns]
    end = panel.index[-1]
    rows = []
    for c in cols:
        s = pd.to_numeric(panel[c], errors="coerce").dropna()
        if s.empty:
            continue
        row = {"name": c, "level": float(s.iloc[-1])}
        for w in windows:
            # map window label → days
            days = {"1d": 1, "5d": 5, "21d": 21, "1w": 5, "1m": 21}.get(w, 5)
            t0 = end - pd.Timedelta(days=int(days))
            # nearest prior bar
            hist = s.loc[:t0]
            if hist.empty:
                row["chg_%s" % w] = np.nan
                continue
            row["chg_%s" % w] = float(s.iloc[-1] - hist.iloc[-1])
        # z of last 1d-ish move vs rolling std of 1-bar diffs (proxy)
        d1 = s.diff().dropna()
        if len(d1) > 20:
            row["z_1bar"] = float(d1.iloc[-1] / d1.iloc[-60:].std(ddof=1)) if d1.iloc[-60:].std(ddof=1) else np.nan
        else:
            row["z_1bar"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outright_curve(panel: pd.DataFrame, out_dir=OUT_DIR) -> Path:
    tenors = sorted((c for c in panel.columns if str(c).isdigit()), key=int)
    if not tenors:
        raise ValueError("no outright tenors in panel")
    last = panel[tenors].iloc[-1]
    # compare to ~1m ago if possible
    t_ref = panel.index[-1] - pd.Timedelta(days=21)
    ref = panel[tenors].loc[:t_ref]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [int(t) for t in tenors]
    ax.plot(x, last.values, "o-", label="asof %s" % panel.index[-1].date())
    if len(ref):
        ax.plot(x, ref.iloc[-1].values, "o--", alpha=0.7, label="~21d ago")
    ax.set_xlabel("tenor (y)")
    ax.set_ylabel("rate (%)")
    ax.set_title("USOSFR outright curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    dest = _ensure_dir(out_dir) / "curve_snapshot.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    print("  [plot] %s" % dest, flush=True)
    return dest


def plot_timeseries(
    panel: pd.DataFrame,
    cols: Sequence[str],
    title: str,
    ylabel: str,
    fname: str,
    out_dir=OUT_DIR,
) -> Optional[Path]:
    use = [c for c in cols if c in panel.columns]
    if not use:
        print("  [plot] skip %s — missing cols" % fname, flush=True)
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in use:
        ax.plot(panel.index, panel[c], label=c, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(4, len(use)), fontsize=8)
    dest = _ensure_dir(out_dir) / fname
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    print("  [plot] %s" % dest, flush=True)
    return dest


def plot_rolling_z(
    panel: pd.DataFrame,
    cols: Sequence[str] = None,
    window: int = 60,
    out_dir=OUT_DIR,
) -> Optional[Path]:
    cols = [c for c in (cols or KEY_FLIES) if c in panel.columns]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in cols:
        s = pd.to_numeric(panel[c], errors="coerce")
        mu = s.rolling(window, min_periods=max(10, window // 3)).mean()
        sd = s.rolling(window, min_periods=max(10, window // 3)).std(ddof=1)
        z = (s - mu) / sd
        ax.plot(panel.index, z, label=c, linewidth=1.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(2, color="gray", ls="--", lw=0.8)
    ax.axhline(-2, color="gray", ls="--", lw=0.8)
    ax.set_title("Rolling z-score (window=%d)" % window)
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(4, len(cols)), fontsize=8)
    dest = _ensure_dir(out_dir) / "rolling_z_flies.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    print("  [plot] %s" % dest, flush=True)
    return dest


def plot_corr_heatmap(
    panel: pd.DataFrame,
    cols: Sequence[str] = None,
    out_dir=OUT_DIR,
) -> Optional[Path]:
    cols = [
        c
        for c in (
            cols
            or KEY_RATES + KEY_CURVES + KEY_FLIES[:3] + list(SPREAD_NAMES) + list(EXTRA_MAP)
        )
        if c in panel.columns
    ]
    if len(cols) < 2:
        return None
    corr = panel[cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Correlation (full sample)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    dest = _ensure_dir(out_dir) / "corr_heatmap.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    print("  [plot] %s" % dest, flush=True)
    return dest


def plot_extras_vs_rates(panel: pd.DataFrame, out_dir=OUT_DIR) -> Optional[Path]:
    """ESA / CO1 / XAU normalized vs 10y rate."""
    extras = [c for c in EXTRA_MAP if c in panel.columns]
    if "10" not in panel.columns or not extras:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    r = pd.to_numeric(panel["10"], errors="coerce")
    r_n = (r - r.mean()) / r.std(ddof=1)
    ax.plot(panel.index, r_n, label="10y (z)", color="black", lw=1.4)
    for c in extras:
        s = pd.to_numeric(panel[c], errors="coerce")
        z = (s - s.mean()) / s.std(ddof=1)
        ax.plot(panel.index, z, label="%s (z)" % c, alpha=0.85)
    ax.set_title("Extras vs 10y (full-sample z)")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    dest = _ensure_dir(out_dir) / "extras_vs_10y.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    print("  [plot] %s" % dest, flush=True)
    return dest


def run_all_metrics(panel: pd.DataFrame, out_dir=OUT_DIR) -> Dict[str, Path]:
    """Starter pack of tables + charts → exports/csv_analysis/."""
    out = _ensure_dir(out_dir)
    paths = {}
    moves = recent_moves_table(panel)
    moves_path = out / "recent_moves.csv"
    moves.to_csv(moves_path, index=False)
    print("  [table] %s" % moves_path.resolve(), flush=True)
    paths["moves"] = moves_path

    for fn, args in (
        (plot_outright_curve, dict(panel=panel, out_dir=out)),
        (
            plot_timeseries,
            dict(
                panel=panel, cols=KEY_RATES, title="Key outrights",
                ylabel="%", fname="ts_outrights.png", out_dir=out,
            ),
        ),
        (
            plot_timeseries,
            dict(
                panel=panel, cols=KEY_CURVES, title="Key curves (bp)",
                ylabel="bp", fname="ts_curves.png", out_dir=out,
            ),
        ),
        (
            plot_timeseries,
            dict(
                panel=panel, cols=KEY_FLIES, title="Key flies (bp)",
                ylabel="bp", fname="ts_flies.png", out_dir=out,
            ),
        ),
        (
            plot_timeseries,
            dict(
                panel=panel, cols=list(SPREAD_NAMES), title="USSFCT spreads",
                ylabel="bp / BBG", fname="ts_ussfct.png", out_dir=out,
            ),
        ),
        (plot_rolling_z, dict(panel=panel, out_dir=out)),
        (plot_corr_heatmap, dict(panel=panel, out_dir=out)),
        (plot_extras_vs_rates, dict(panel=panel, out_dir=out)),
    ):
        try:
            p = fn(**args)
            if p is not None:
                paths[Path(p).stem] = Path(p)
        except Exception as exc:  # noqa: BLE001
            print("  [plot] %s failed: %s" % (getattr(fn, "__name__", fn), exc), flush=True)
    return paths


def run_pipeline(csv_path=DEFAULT_CSV, out_dir=OUT_DIR, ffill: bool = True):
    """End-to-end: load → clean → structures → save panel → metrics."""
    raw = load_csv(csv_path, ffill=False)
    cleaned, report = clean_panel(raw)
    if ffill:
        cleaned = cleaned.ffill()
    panel = build_analysis_panel(cleaned)
    panel_path = save_panel(panel, out_dir=out_dir)
    paths = run_all_metrics(panel, out_dir=out_dir)
    paths["panel"] = panel_path
    return panel, report, paths


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Offline CSV SOFR analysis")
    p.add_argument("--csv", default=DEFAULT_CSV, help="Path to BBG dump CSV/xlsx")
    p.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    p.add_argument("--no-ffill", action="store_true")
    args = p.parse_args(argv)
    panel, report, paths = run_pipeline(
        args.csv, out_dir=args.out, ffill=not args.no_ffill,
    )
    print(
        "done  asof=%s  panel=%s  n_plots=%d"
        % (report.get("asof"), paths.get("panel"), len(paths) - 1),
        flush=True,
    )


if __name__ == "__main__":
    main()
