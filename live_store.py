# =============================================================================
# live_store.py  — Append hourly BBG prints; keep a fixed-length rolling file
# =============================================================================
# Policy:
#   1. STACK: append the new row(s) onto LIVE_STORE_PATH (CSV in this folder).
#   2. DEDUPE: if the timestamp already exists, replace that row.
#   3. TRIM: if len > MAX_HISTORY_BARS, drop the oldest rows (constant size).
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from curve_config import LIVE_STORE_PATH, MAX_HISTORY_BARS
from data_loader import load_raw


def load_live_store(path: Union[str, Path] = LIVE_STORE_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return load_raw(path, ffill=True)


def save_live_store(
    df: pd.DataFrame,
    path: Union[str, Path] = LIVE_STORE_PATH,
) -> Path:
    path = Path(path)
    out = df.sort_index()
    out.index.name = "time"
    out.to_csv(path)
    return path.resolve()


def append_snapshot(
    snapshot: pd.DataFrame,
    path: Union[str, Path] = LIVE_STORE_PATH,
    max_bars: int = MAX_HISTORY_BARS,
) -> pd.DataFrame:
    """
    Stack `snapshot` onto the on-disk panel and trim to `max_bars`.

    Returns the updated panel (DatetimeIndex).
    """
    if snapshot.empty:
        raise ValueError("snapshot is empty")

    snap = snapshot.copy()
    if not isinstance(snap.index, pd.DatetimeIndex):
        raise TypeError("snapshot must have a DatetimeIndex named time")
    snap.index = pd.DatetimeIndex(snap.index).tz_localize(None)
    snap.index.name = "time"
    snap = snap.apply(pd.to_numeric, errors="coerce")

    existing = load_live_store(path)
    if existing.empty:
        combined = snap
    else:
        # Align columns (union); missing → NaN then ffill later at model time
        combined = pd.concat([existing, snap], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

    if max_bars is not None and len(combined) > max_bars:
        combined = combined.iloc[-max_bars:]

    save_live_store(combined, path)
    return combined


def ensure_seed_history(
    path: Union[str, Path] = LIVE_STORE_PATH,
    seed_path: Optional[Union[str, Path]] = None,
    n_synthetic: int = 400,
) -> pd.DataFrame:
    """
    If live store missing, seed from DATA_PATH / synthetic so models have history.
    """
    path = Path(path)
    if path.exists():
        return load_live_store(path)

    from curve_config import DATA_PATH
    from data_loader import generate_synthetic_sofr

    src = Path(seed_path) if seed_path else Path(DATA_PATH)
    if src.exists():
        hist = load_raw(src)
    else:
        hist = generate_synthetic_sofr(n_bars=n_synthetic, freq="h")
    save_live_store(hist, path)
    return hist
