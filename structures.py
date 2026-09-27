# =============================================================================
# structures.py  — Build curves / flies from raw SOFR tenors
# =============================================================================
# Pipeline:
#   data_raw  →  full_data (all curves + flies)
#             →  predicted_data (isolated target columns)
# =============================================================================

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from curve_config import (
    BBG_PREFIX,
    BBG_SUFFIX,
    CURVES,
    FLIES,
    PREDICTED,
    CurveSpec,
    FlySpec,
)

log = logging.getLogger(__name__)


# ── Column cleaning ───────────────────────────────────────────────────────────

_TENOR_RE = re.compile(
    rf"^{re.escape(BBG_PREFIX)}(\d+)\s*{re.escape(BBG_SUFFIX)}$",
    re.IGNORECASE,
)


def parse_tenor_column(col: str) -> Optional[int]:
    """
    Map a raw column name to an integer tenor in years.

    Accepts:
      - 'USOSFR10 BGN Curncy'  (Bloomberg)
      - '10', '10Y', '10y'     (already cleaned)
    """
    s = str(col).strip()
    m = _TENOR_RE.match(s)
    if m:
        return int(m.group(1))
    m2 = re.fullmatch(r"(\d+)\s*[Yy]?", s)
    if m2:
        return int(m2.group(1))
    return None


def clean_raw_columns(data_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Bloomberg-style columns to integer tenors (as strings: '1','2',...).
    Drops columns that cannot be parsed. Keeps DatetimeIndex.
    """
    rename: Dict[str, str] = {}
    for col in data_raw.columns:
        tenor = parse_tenor_column(col)
        if tenor is not None:
            rename[col] = str(tenor)

    missing = [c for c in data_raw.columns if c not in rename]
    if missing:
        log.warning("Dropping unparsed columns: %s", missing)

    out = data_raw.rename(columns=rename)[list(rename.values())].copy()
    # de-dupe if both '10' and 'USOSFR10...' somehow present
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def required_tenors(
    curves: Sequence[CurveSpec] = CURVES,
    flies: Sequence[FlySpec] = FLIES,
) -> Set[int]:
    tenors: Set[int] = set()
    for _, a, b in curves:
        tenors.update((a, b))
    for _, a, b, c in flies:
        tenors.update((a, b, c))
    return tenors


# ── Structure builders ────────────────────────────────────────────────────────

def _tenor_col(df: pd.DataFrame, tenor: int) -> pd.Series:
    key = str(tenor)
    if key not in df.columns:
        raise KeyError(
            f"Missing tenor {tenor}Y in raw data. "
            f"Available: {sorted(df.columns, key=lambda x: int(x) if str(x).isdigit() else 0)}"
        )
    return df[key]


def build_curve(data_raw: pd.DataFrame, name: str, short: int, long: int) -> pd.Series:
    """Curve XsYs = long - short (steepener convention)."""
    s = _tenor_col(data_raw, long) - _tenor_col(data_raw, short)
    s.name = name
    return s


def build_fly(
    data_raw: pd.DataFrame, name: str, left: int, body: int, right: int
) -> pd.Series:
    """Butterfly XsYsZs = 2*body - left - right."""
    s = (
        2.0 * _tenor_col(data_raw, body)
        - _tenor_col(data_raw, left)
        - _tenor_col(data_raw, right)
    )
    s.name = name
    return s


def build_full_data(
    data_raw: pd.DataFrame,
    curves: Sequence[CurveSpec] = CURVES,
    flies: Sequence[FlySpec] = FLIES,
    clean: bool = True,
) -> pd.DataFrame:
    """
    From cleaned (or raw Bloomberg) tenor panel → DataFrame of all structures.

    Columns = curve names + fly names. Index = same DatetimeIndex as raw.
    """
    raw = clean_raw_columns(data_raw) if clean else data_raw.copy()

    needed = required_tenors(curves, flies)
    available = {int(c) for c in raw.columns if str(c).isdigit()}
    missing = sorted(needed - available)
    if missing:
        raise KeyError(
            f"Raw data missing tenors required by structures: {missing}. "
            f"Have: {sorted(available)}"
        )

    pieces: List[pd.Series] = []
    for name, a, b in curves:
        pieces.append(build_curve(raw, name, a, b))
    for name, a, b, c in flies:
        pieces.append(build_fly(raw, name, a, b, c))

    full = pd.concat(pieces, axis=1)
    full.index = raw.index
    log.info(
        "full_data: %d structures (%d curves, %d flies), %d bars",
        full.shape[1], len(curves), len(flies), len(full),
    )
    return full


def build_predicted_data(
    full_data: pd.DataFrame,
    predicted: Sequence[str] = PREDICTED,
) -> pd.DataFrame:
    """
    Isolate target columns. Same values live in full_data; this is a view-copy
    so the prediction layer has a clean, expandable surface.
    """
    missing = [c for c in predicted if c not in full_data.columns]
    if missing:
        raise KeyError(
            f"Predicted targets not in full_data: {missing}. "
            f"Add them to CURVES/FLIES in curve_config.py."
        )
    pred = full_data.loc[:, list(predicted)].copy()
    log.info("predicted_data: %s", list(pred.columns))
    return pred


def build_datasets(
    data_raw: pd.DataFrame,
    curves: Sequence[CurveSpec] = CURVES,
    flies: Sequence[FlySpec] = FLIES,
    predicted: Sequence[str] = PREDICTED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience: raw → (cleaned_raw, full_data, predicted_data).
    """
    cleaned = clean_raw_columns(data_raw)
    full = build_full_data(cleaned, curves=curves, flies=flies, clean=False)
    pred = build_predicted_data(full, predicted=predicted)
    return cleaned, full, pred
