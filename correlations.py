#!/usr/bin/env python3
"""
Correlation matrices for CORR_SERIES (structures + CROSS_ASSET).

1) Level correlation matrix (pairwise, last CORR_WINDOW bars).
2) Percentile of rolling correlation: for each pair, rolling corr over CORR_WINDOW,
   then pct ∈ [0, 100] of how high corr_now ranks vs the last CORR_PCT_LOOKBACK
   rolling-corr values (100 = highest in the window). Heatmap of that matrix.

  python run_dz.py correlations
  # or run_correlations.ipynb
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

import curve_config as _cfg
from curve_config import CORR_SERIES
from data import build_panel, ensure_series_in_live_store, load_panel

CORR_WINDOW = int(getattr(_cfg, "CORR_WINDOW", 50))
# lookback of rolling corrs for the percentile rank (alias CORR_Z_LOOKBACK kept)
CORR_PCT_LOOKBACK = int(
    getattr(_cfg, "CORR_PCT_LOOKBACK", getattr(_cfg, "CORR_Z_LOOKBACK", 200))
)
GRAPH_DIR_CORR = getattr(_cfg, "GRAPH_DIR_CORR", "graphs/correlations")


def _available(panel: pd.DataFrame, series: Sequence[str]) -> List[str]:
    cols = [c for c in series if c in panel.columns]
    missing = [c for c in series if c not in panel.columns]
    if missing:
        print("  [corr] missing (skip): %s" % missing, flush=True)
    if len(cols) < 2:
        raise ValueError("Need ≥2 CORR_SERIES columns in panel; have %s" % cols)
    return cols


def corr_level_matrix(panel: pd.DataFrame, series=None, window=None) -> pd.DataFrame:
    """Pairwise correlation of selected series (last `window` bars if set)."""
    cols = _available(panel, series or CORR_SERIES)
    w = CORR_WINDOW if window is None else window
    sub = panel[cols].apply(pd.to_numeric, errors="coerce")
    if w and w > 0:
        sub = sub.iloc[-int(w):]
    return sub.corr()


def rolling_corr_pair(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=window).corr(b)


def _percentile_of_last(hist: pd.Series) -> float:
    """Empirical percentile of the last value in hist: 100 = max, 0 = min."""
    if hist is None or len(hist) < 2:
        return float("nan")
    vals = hist.to_numpy(dtype=float)
    last = vals[-1]
    if not np.isfinite(last):
        return float("nan")
    # share of history ≤ current (includes last → max ranks 100)
    return float(100.0 * np.mean(vals <= last))


def corr_percentile_matrix(
    panel: pd.DataFrame,
    series=None,
    corr_window=None,
    pct_lookback=None,
) -> pd.DataFrame:
    """
    For each pair (i,j): rolling corr over `corr_window`, then percentile of
    the latest corr among the last `pct_lookback` rolling-corr values.
    100 = highest in that window. Diagonal = NaN.
    """
    cols = _available(panel, series or CORR_SERIES)
    cw = int(CORR_WINDOW if corr_window is None else corr_window)
    pl = int(CORR_PCT_LOOKBACK if pct_lookback is None else pct_lookback)
    sub = panel[cols].apply(pd.to_numeric, errors="coerce")
    n = len(cols)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rc = rolling_corr_pair(sub.iloc[:, i], sub.iloc[:, j], cw).dropna()
            if len(rc) < max(10, pl // 10):
                continue
            hist = rc.iloc[-pl:] if len(rc) >= pl else rc
            pct = _percentile_of_last(hist)
            pmat.iloc[i, j] = pct
            pmat.iloc[j, i] = pct
    return pmat


# Back-compat alias
def corr_zscore_matrix(*args, z_lookback=None, **kwargs):
    return corr_percentile_matrix(*args, pct_lookback=z_lookback, **kwargs)


def plot_corr_heatmap(
    mat: pd.DataFrame,
    title: str,
    path: Path,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    fmt="%.2f",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [corr] matplotlib missing — skip heatmap %s" % path, flush=True)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(mat)), max(4, 0.5 * len(mat))))
    data = mat.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if vmin is None or vmax is None:
        if len(finite):
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            # symmetric around 0 only if span crosses 0 and looks like corr
            if lo >= 0 and hi <= 100 and hi > 50:
                vmin, vmax = 0.0, 100.0
            elif lo >= -1.05 and hi <= 1.05:
                vmin, vmax = -1.0, 1.0
            else:
                pad = max(abs(lo), abs(hi), 1e-6)
                vmin, vmax = -pad, pad
        else:
            vmin, vmax = 0.0, 1.0
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=8)
    mid = 0.5 * (float(vmin) + float(vmax))
    span = max(abs(float(vmax) - float(vmin)), 1e-6)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j, i, fmt % v, ha="center", va="center", fontsize=6,
                color="white" if abs(v - mid) > 0.35 * span else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [corr] wrote %s" % path, flush=True)
    return path


def _resolve_rates_col(panel: pd.DataFrame, rates: str) -> str:
    """Prefer configured rates name; fall back to outright '5' (USOSFR5)."""
    if rates in panel.columns:
        return rates
    if "5" in panel.columns:
        print("  [corr] rates %r missing → using outright '5'" % rates, flush=True)
        return "5"
    raise KeyError("Rates series %r not in panel (and no '5'). Add to CROSS_ASSET." % rates)


def dual_corr_scatter_frame(
    panel: pd.DataFrame,
    rates=None,
    asset_x=None,
    asset_y=None,
    corr_window=None,
    highlight_days=None,
) -> pd.DataFrame:
    """
    Build scatter data: x = 100*ρ(rates, asset_x), y = 100*ρ(rates, asset_y).
    `recent` = True for bars in the last `highlight_days` calendar days.
    """
    rates = rates or getattr(_cfg, "CORR_SCATTER_RATES", "USGG5YR")
    asset_x = asset_x or getattr(_cfg, "CORR_SCATTER_X", "CO1")
    asset_y = asset_y or getattr(_cfg, "CORR_SCATTER_Y", "ESA")
    cw = int(CORR_WINDOW if corr_window is None else corr_window)
    days = int(
        getattr(_cfg, "CORR_SCATTER_HIGHLIGHT_DAYS", 4)
        if highlight_days is None else highlight_days
    )
    rcol = _resolve_rates_col(panel, rates)
    for c in (asset_x, asset_y):
        if c not in panel.columns:
            raise KeyError(
                "Need %r in panel for scatter. Add to CROSS_ASSET and --pull-history. Have=%s"
                % (c, list(panel.columns)[:30])
            )
    sub = panel[[rcol, asset_x, asset_y]].apply(pd.to_numeric, errors="coerce")
    rx = rolling_corr_pair(sub[rcol], sub[asset_x], cw) * 100.0
    ry = rolling_corr_pair(sub[rcol], sub[asset_y], cw) * 100.0
    out = pd.DataFrame({"corr_x": rx, "corr_y": ry}).dropna()
    out.index.name = "time"
    if out.empty:
        raise ValueError("No overlapping rolling corrs for scatter (check window / data)")
    cutoff = pd.Timestamp(out.index[-1]).normalize() - pd.Timedelta(days=max(days - 1, 0))
    out["recent"] = pd.DatetimeIndex(out.index).normalize() >= cutoff
    out.attrs["rates"] = rcol
    out.attrs["asset_x"] = asset_x
    out.attrs["asset_y"] = asset_y
    out.attrs["corr_window"] = cw
    out.attrs["highlight_days"] = days
    return out


def plot_dual_corr_scatter(
    panel: pd.DataFrame,
    path=None,
    rates=None,
    asset_x=None,
    asset_y=None,
    corr_window=None,
    highlight_days=None,
):
    """
    MS-style scatter: x = ρ(US rates, CO1), y = ρ(US rates, ESA).
    Historical points khaki; last `highlight_days` calendar days in blue.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [corr] matplotlib missing — skip dual-corr scatter", flush=True)
        return None, None

    frame = dual_corr_scatter_frame(
        panel,
        rates=rates,
        asset_x=asset_x,
        asset_y=asset_y,
        corr_window=corr_window,
        highlight_days=highlight_days,
    )
    rcol = frame.attrs["rates"]
    ax_name = frame.attrs["asset_x"]
    ay_name = frame.attrs["asset_y"]
    cw = frame.attrs["corr_window"]
    days = frame.attrs["highlight_days"]

    hist = frame.loc[~frame["recent"]]
    recent = frame.loc[frame["recent"]]
    last = frame.iloc[-1]

    path = Path(path or (Path(GRAPH_DIR_CORR) / "corr_scatter_rates_assets.png"))
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_facecolor("white")
    ax.grid(True, color="#d0d0d0", lw=0.7, zorder=0)
    ax.axhline(0.0, color="#888", lw=0.9, zorder=1)
    ax.axvline(0.0, color="#888", lw=0.9, zorder=1)

    if len(hist):
        ax.scatter(
            hist["corr_x"], hist["corr_y"],
            s=28, c="#c4a574", alpha=0.85, edgecolors="none",
            label="history", zorder=2,
        )
    if len(recent):
        ax.scatter(
            recent["corr_x"], recent["corr_y"],
            s=42, c="#1f77d0", alpha=0.95, edgecolors="white", linewidths=0.4,
            label="last %d days" % days, zorder=3,
        )
        # label near recent cluster
        ax.annotate(
            "last %d days" % days,
            xy=(float(recent["corr_x"].median()), float(recent["corr_y"].median())),
            xytext=(12, 12), textcoords="offset points",
            color="#1f77d0", fontsize=9, fontweight="bold",
        )

    # callout last point
    ax.scatter(
        [float(last["corr_x"])], [float(last["corr_y"])],
        s=55, c="#5c4033", zorder=4, edgecolors="white", linewidths=0.5,
    )
    ax.annotate(
        "%.1f, %.1f" % (float(last["corr_x"]), float(last["corr_y"])),
        xy=(float(last["corr_x"]), float(last["corr_y"])),
        xytext=(14, -18), textcoords="offset points",
        fontsize=8, color="#333",
        arrowprops=dict(arrowstyle="-", color="#555", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.6),
    )

    # optional quadrant box (ρ_x>0 & ρ_y>0) — rates co-move with both risk assets
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # expand to nice ±100-ish bounds if data is within
    xmin = min(-100.0, float(frame["corr_x"].min()) - 5)
    xmax = max(100.0, float(frame["corr_x"].max()) + 5)
    ymin = min(-100.0, float(frame["corr_y"].min()) - 5)
    ymax = max(100.0, float(frame["corr_y"].max()) + 5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$\rho$(%s, %s)   ×100" % (rcol, ax_name), fontsize=10)
    ax.set_ylabel(r"$\rho$(%s, %s)   ×100" % (rcol, ay_name), fontsize=10)
    ax.set_title(
        "Rolling corr scatter  |  window=%d  |  asof %s"
        % (cw, pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d %H:%M")),
        fontsize=11, pad=10,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [corr] wrote %s  (n=%d recent=%d)" % (path, len(frame), int(frame["recent"].sum())), flush=True)
    return path, frame


def run_correlations(
    *,
    series=None,
    corr_window=None,
    pct_lookback=None,
    z_lookback=None,  # alias
    out_dir=None,
    make_scatter=True,
    ensure_missing=True,
):
    cols = list(series or CORR_SERIES)
    # Also ensure scatter legs exist when possible
    for extra in (
        getattr(_cfg, "CORR_SCATTER_RATES", None),
        getattr(_cfg, "CORR_SCATTER_X", None),
        getattr(_cfg, "CORR_SCATTER_Y", None),
    ):
        if extra and extra not in cols:
            cols.append(extra)

    if ensure_missing:
        ensure_series_in_live_store(cols)

    data_raw, src = load_panel()
    _, full, _ = build_panel(data_raw)
    print("  [corr] src=%s  n=%d  asof=%s" % (src, len(full), full.index[-1]), flush=True)
    present = [c for c in cols if c in full.columns]
    missing = [c for c in cols if c not in full.columns]
    if missing:
        print(
            "  [corr] still missing after ensure (not in live store / BBG): %s" % missing,
            flush=True,
        )
    print("  [corr] using: %s" % present, flush=True)

    cw = int(CORR_WINDOW if corr_window is None else corr_window)
    if pct_lookback is not None:
        pl = int(pct_lookback)
    elif z_lookback is not None:
        pl = int(z_lookback)
    else:
        pl = int(CORR_PCT_LOOKBACK)

    level = corr_level_matrix(full, series=present, window=cw)
    pmat = corr_percentile_matrix(full, series=present, corr_window=cw, pct_lookback=pl)

    dest = Path(out_dir or GRAPH_DIR_CORR)
    dest.mkdir(parents=True, exist_ok=True)
    level.to_csv(dest / "corr_level.csv")
    pmat.to_csv(dest / "corr_percentile.csv")
    plot_corr_heatmap(
        level,
        title="Correlation (last %d bars)" % cw,
        path=dest / "corr_level_heatmap.png",
        vmin=-1.0,
        vmax=1.0,
        fmt="%.2f",
    )
    plot_corr_heatmap(
        pmat,
        title="Corr percentile (roll=%d, lookback=%d)  — 100=highest" % (cw, pl),
        path=dest / "corr_percentile_heatmap.png",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=100.0,
        fmt="%.0f",
    )
    scatter_path, scatter_df = None, None
    if make_scatter:
        try:
            scatter_path, scatter_df = plot_dual_corr_scatter(
                full, path=dest / "corr_scatter_rates_assets.png", corr_window=cw,
            )
            if scatter_df is not None:
                scatter_df.to_csv(dest / "corr_scatter_rates_assets.csv")
        except Exception as exc:  # noqa: BLE001
            print("  [corr] scatter skipped: %s" % exc, flush=True)
    return level, pmat, full, scatter_path, scatter_df


def main(argv=None):
    p = argparse.ArgumentParser(description="Correlation + percentile + rates/asset scatter")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--pct-lookback", type=int, default=None)
    p.add_argument("--z-lookback", type=int, default=None, help="alias of --pct-lookback")
    p.add_argument("--no-scatter", action="store_true")
    args = p.parse_args(argv)
    level, pmat, _, scatter_path, _ = run_correlations(
        corr_window=args.window,
        pct_lookback=args.pct_lookback,
        z_lookback=args.z_lookback,
        make_scatter=not args.no_scatter,
    )
    print("\n=== level corr ===", flush=True)
    print(level.round(3).to_string(), flush=True)
    print("\n=== corr percentile (100=highest) ===", flush=True)
    print(pmat.round(1).to_string(), flush=True)
    if scatter_path:
        print("\nscatter → %s" % scatter_path, flush=True)


if __name__ == "__main__":
    main()
