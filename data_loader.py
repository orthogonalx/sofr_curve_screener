# =============================================================================
# data_loader.py  — SOFR curve panel (Excel / CSV) and synthetic fallback
# =============================================================================
#   from data_loader import load_raw, generate_synthetic_sofr
#   data_raw = load_raw("sofr_3h.xlsx")
# =============================================================================

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_raw(
    filepath: Union[str, Path],
    timestamp_col: str = "time",
    sheet_name: Union[int, str] = 0,
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Load Bloomberg-style SOFR swap panel into data_raw.

    Expected layout (matches screenshot):
        time | USOSFR1 BGN Curncy | USOSFR2 BGN Curncy | ...
        3h frequency timestamps; columns = swap tenors.

    Returns
    -------
    DataFrame with DatetimeIndex and original column names (pre-structure build).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()
    log.info("Loading raw SOFR panel: %s", path)

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .xlsx or .csv")

    # Flexible timestamp column
    if timestamp_col not in df.columns:
        candidates = ["time", "datetime", "date", "Date", "Time", "timestamp"]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            # first column often is the timestamp in BBG exports
            found = df.columns[0]
            log.warning(
                "timestamp_col '%s' missing; using first column '%s'",
                timestamp_col, found,
            )
        timestamp_col = found

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    df.index.name = "time"
    df = df.apply(pd.to_numeric, errors="coerce")
    if ffill:
        df = df.ffill()

    log.info(
        "data_raw: %d tenors, %d bars | %s → %s",
        df.shape[1], len(df), df.index[0], df.index[-1],
    )
    return df


def generate_synthetic_sofr(
    tenors: Optional[Sequence[int]] = None,
    n_bars: int = 400,
    freq: str = "3h",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic upward-sloping SOFR curve panel with Bloomberg-like column names.
    Useful for local dry-runs before the real Excel is available.
    """
    if tenors is None:
        tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30, 35]

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq)

    # Level factor + slope factor (correlated random walks)
    level = 3.5 + np.cumsum(rng.normal(0, 0.01, n_bars))
    slope = 0.15 + np.cumsum(rng.normal(0, 0.002, n_bars))

    data = {}
    for t in tenors:
        # Simple Nelson-Siegel-ish shape + idiosyncratic noise
        rate = level + slope * np.log1p(t) / np.log1p(30) + rng.normal(0, 0.005, n_bars)
        data[f"USOSFR{t} BGN Curncy"] = rate

    # Inject a mild dislocation in 30s35s space near the end (via 35y)
    shock = np.zeros(n_bars)
    shock[-20:] = 0.08
    data["USOSFR35 BGN Curncy"] = data["USOSFR35 BGN Curncy"] + shock

    df = pd.DataFrame(data, index=idx)
    df.index.name = "time"
    return df
