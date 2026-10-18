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
from data import build_panel, load_panel

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


def run_correlations(
    *,
    series=None,
    corr_window=None,
    pct_lookback=None,
    z_lookback=None,  # alias
    out_dir=None,
):
    data_raw, src = load_panel()
    _, full, _ = build_panel(data_raw)
    print("  [corr] src=%s  n=%d  asof=%s" % (src, len(full), full.index[-1]), flush=True)

    cw = int(CORR_WINDOW if corr_window is None else corr_window)
    if pct_lookback is not None:
        pl = int(pct_lookback)
    elif z_lookback is not None:
        pl = int(z_lookback)
    else:
        pl = int(CORR_PCT_LOOKBACK)
    cols = series or CORR_SERIES

    level = corr_level_matrix(full, series=cols, window=cw)
    pmat = corr_percentile_matrix(full, series=cols, corr_window=cw, pct_lookback=pl)

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
    return level, pmat, full


def main(argv=None):
    p = argparse.ArgumentParser(description="Correlation + corr percentile heatmaps")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--pct-lookback", type=int, default=None)
    p.add_argument("--z-lookback", type=int, default=None, help="alias of --pct-lookback")
    args = p.parse_args(argv)
    level, pmat, _ = run_correlations(
        corr_window=args.window,
        pct_lookback=args.pct_lookback,
        z_lookback=args.z_lookback,
    )
    print("\n=== level corr ===", flush=True)
    print(level.round(3).to_string(), flush=True)
    print("\n=== corr percentile (100=highest) ===", flush=True)
    print(pmat.round(1).to_string(), flush=True)


if __name__ == "__main__":
    main()
