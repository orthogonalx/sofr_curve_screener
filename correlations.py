#!/usr/bin/env python3
"""
Correlation matrices for CORR_SERIES (structures + CROSS_ASSET).

1) Level correlation matrix (pairwise, last CORR_WINDOW bars by default —
   or full-sample; see corr_level_matrix).
2) Z-score of rolling correlation: for each pair, rolling corr over CORR_WINDOW,
   then z = (corr_now − μ) / σ over the last CORR_Z_LOOKBACK rolling-corr values.
   Heatmap of that z-matrix.

  python run_dz.py correlations
  # or run_correlations.ipynb
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import curve_config as _cfg
from curve_config import CORR_SERIES
from data import build_panel, load_panel

CORR_WINDOW = int(getattr(_cfg, "CORR_WINDOW", 50))
CORR_Z_LOOKBACK = int(getattr(_cfg, "CORR_Z_LOOKBACK", 200))
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
    """
    Pairwise correlation of selected series.
    If window is set, use only the last `window` bars; else full sample.
    """
    cols = _available(panel, series or CORR_SERIES)
    w = CORR_WINDOW if window is None else window
    sub = panel[cols].apply(pd.to_numeric, errors="coerce")
    if w and w > 0:
        sub = sub.iloc[-int(w):]
    return sub.corr()


def rolling_corr_pair(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=window).corr(b)


def corr_zscore_matrix(
    panel: pd.DataFrame,
    series=None,
    corr_window=None,
    z_lookback=None,
) -> pd.DataFrame:
    """
    For each pair (i,j): rolling corr over `corr_window`, then
    z = (last_corr − mean) / std using the last `z_lookback` rolling-corr values.
    Diagonal = NaN.
    """
    cols = _available(panel, series or CORR_SERIES)
    cw = int(CORR_WINDOW if corr_window is None else corr_window)
    zl = int(CORR_Z_LOOKBACK if z_lookback is None else z_lookback)
    sub = panel[cols].apply(pd.to_numeric, errors="coerce")
    n = len(cols)
    zmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rc = rolling_corr_pair(sub.iloc[:, i], sub.iloc[:, j], cw).dropna()
            if len(rc) < max(10, zl // 10):
                continue
            hist = rc.iloc[-zl:] if len(rc) >= zl else rc
            mu, sd = float(hist.mean()), float(hist.std(ddof=1))
            last = float(rc.iloc[-1])
            z = (last - mu) / sd if sd and abs(sd) > 1e-12 else np.nan
            zmat.iloc[i, j] = z
            zmat.iloc[j, i] = z
    return zmat


def plot_corr_heatmap(mat: pd.DataFrame, title: str, path: Path, cmap="RdBu_r", vlim=None):
    plt_mod = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt_mod = plt
    except ImportError:
        print("  [corr] matplotlib missing — skip heatmap %s" % path, flush=True)
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt_mod.subplots(figsize=(max(5, 0.55 * len(mat)), max(4, 0.5 * len(mat))))
    data = mat.to_numpy(dtype=float)
    if vlim is None:
        finite = data[np.isfinite(data)]
        vmax = float(np.nanmax(np.abs(finite))) if len(finite) else 1.0
        vmax = max(vmax, 1e-6)
        vmin, vmax = -vmax, vmax
    else:
        vmin, vmax = -float(vlim), float(vlim)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt_mod.close(fig)
    print("  [corr] wrote %s" % path, flush=True)
    return path


def run_correlations(
    *,
    series=None,
    corr_window=None,
    z_lookback=None,
    out_dir=None,
):
    data_raw, src = load_panel()
    _, full, _ = build_panel(data_raw)
    print("  [corr] src=%s  n=%d  asof=%s" % (src, len(full), full.index[-1]), flush=True)

    cw = int(CORR_WINDOW if corr_window is None else corr_window)
    zl = int(CORR_Z_LOOKBACK if z_lookback is None else z_lookback)
    cols = series or CORR_SERIES

    level = corr_level_matrix(full, series=cols, window=cw)
    zmat = corr_zscore_matrix(full, series=cols, corr_window=cw, z_lookback=zl)

    dest = Path(out_dir or GRAPH_DIR_CORR)
    dest.mkdir(parents=True, exist_ok=True)
    level.to_csv(dest / "corr_level.csv")
    zmat.to_csv(dest / "corr_zscore.csv")
    plot_corr_heatmap(
        level,
        title="Correlation (last %d bars)" % cw,
        path=dest / "corr_level_heatmap.png",
        vlim=1.0,
    )
    plot_corr_heatmap(
        zmat,
        title="Corr z-score (roll=%d, lookback=%d)" % (cw, zl),
        path=dest / "corr_zscore_heatmap.png",
    )
    return level, zmat, full


def main(argv=None):
    p = argparse.ArgumentParser(description="Correlation + corr z-score heatmaps")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--z-lookback", type=int, default=None)
    args = p.parse_args(argv)
    level, zmat, _ = run_correlations(corr_window=args.window, z_lookback=args.z_lookback)
    print("\n=== level corr ===", flush=True)
    print(level.round(3).to_string(), flush=True)
    print("\n=== corr z-score ===", flush=True)
    print(zmat.round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
