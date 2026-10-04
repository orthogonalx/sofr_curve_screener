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
    BBG_FWD_PREFIX,
    BBG_FWD_SUFFIX,
    BBG_PREFIX,
    BBG_SUFFIX,
    CURVES,
    FLIES,
    FWD_CURVES,
    FWD_FLIES,
    FWD_GROUPS,
    PREDICTED,
    CurveSpec,
    FlySpec,
    FwdCurveSpec,
    FwdFlySpec,
)

log = logging.getLogger(__name__)


# ── Column cleaning ───────────────────────────────────────────────────────────

_TENOR_RE = re.compile(
    rf"^{re.escape(BBG_PREFIX)}(\d+)\s*{re.escape(BBG_SUFFIX)}$",
    re.IGNORECASE,
)
# S0490FS 20Y5Y BLC Curncy  →  20y5y
_FWD_RE = re.compile(
    rf"^{re.escape(BBG_FWD_PREFIX)}\s+(\d+)Y(\d+)Y\s*{re.escape(BBG_FWD_SUFFIX)}$",
    re.IGNORECASE,
)
_FWD_CLEAN_RE = re.compile(r"^(\d+)[Yy](\d+)[Yy]$")


def parse_column_name(col: str) -> Optional[str]:
    """
    Map a raw Bloomberg column to a cleaned name.

    Accepts:
      - 'USOSFR10 BGN Curncy'           → '10'
      - '10', '10Y', '10y'              → '10'
      - 'S0490FS 20Y5Y BLC Curncy'      → '20y5y'
      - '20y5y', '20Y5Y'               → '20y5y'
    """
    s = str(col).strip()

    m = _TENOR_RE.match(s)
    if m:
        return str(int(m.group(1)))

    m = _FWD_RE.match(s)
    if m:
        return f"{int(m.group(1))}y{int(m.group(2))}y"

    m = _FWD_CLEAN_RE.fullmatch(s)
    if m:
        return f"{int(m.group(1))}y{int(m.group(2))}y"

    m2 = re.fullmatch(r"(\d+)\s*[Yy]?", s)
    if m2:
        return str(int(m2.group(1)))

    return None


def parse_tenor_column(col: str) -> Optional[int]:
    """Map a column to an integer tenor in years, or None if not a single tenor."""
    name = parse_column_name(col)
    if name is not None and name.isdigit():
        return int(name)
    return None


def resolve_feature_name(name: str) -> str:
    """
    Map a feature label from settings to a full_data column.
      'USOSFR5' / 'USOSFR5 BGN Curncy' / '5Y' → '5'
      '2s5s10s' / '5y5y' → unchanged (if not a tenor alias)
    """
    s = str(name).strip()
    parsed = parse_column_name(s)
    if parsed is not None:
        return parsed
    # bare USOSFR{N} without suffix
    m = re.fullmatch(rf"{re.escape(BBG_PREFIX)}(\d+)", s, re.IGNORECASE)
    if m:
        return str(int(m.group(1)))
    return s


def is_tenor_column(name: str) -> bool:
    return str(name).isdigit()


def is_fwd_structure_column(name: str) -> bool:
    return bool(_FWD_CLEAN_RE.fullmatch(str(name)))


def clean_raw_columns(data_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Bloomberg-style columns to cleaned names:
      - outright swaps → '1', '2', … (tenor years)
      - S0490FS forwards/spreads → '20y5y', '25y5y', …
    Drops columns that cannot be parsed. Keeps DatetimeIndex.
    """
    rename: Dict[str, str] = {}
    for col in data_raw.columns:
        cleaned = parse_column_name(col)
        if cleaned is not None:
            rename[col] = cleaned

    missing = [c for c in data_raw.columns if c not in rename]
    if missing:
        log.warning("Dropping unparsed columns: %s", missing)

    out = data_raw.rename(columns=rename)[list(rename.values())].copy()
    # de-dupe if both '10' and 'USOSFR10...' (or two aliases) somehow present
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
    """Swap curve XsYs = (long - short) * 100  (percent → bp)."""
    s = (_tenor_col(data_raw, long) - _tenor_col(data_raw, short)) * 100.0
    s.name = name
    return s


def build_fly(
    data_raw: pd.DataFrame, name: str, left: int, body: int, right: int
) -> pd.Series:
    """Swap butterfly XsYsZs = (2*body - left - right) * 100  (percent → bp)."""
    s = (
        2.0 * _tenor_col(data_raw, body)
        - _tenor_col(data_raw, left)
        - _tenor_col(data_raw, right)
    ) * 100.0
    s.name = name
    return s


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(
            f"Missing column '{name}' in data. "
            f"Available: {list(df.columns)}"
        )
    return df[name]


def build_fwd_curve(df: pd.DataFrame, name: str, left: str, right: str) -> pd.Series:
    """Forward curve A_B = (B - A) * 100  (percent → bp)."""
    s = (_col(df, right) - _col(df, left)) * 100.0
    s.name = name
    return s


def build_fwd_fly(
    df: pd.DataFrame, name: str, left: str, body: str, right: str
) -> pd.Series:
    """Forward fly A_B_C = (2*B - A - C) * 100  (percent → bp)."""
    s = (2.0 * _col(df, body) - _col(df, left) - _col(df, right)) * 100.0
    s.name = name
    return s


def required_forwards(
    fwd_curves: Sequence[FwdCurveSpec] = FWD_CURVES,
    fwd_flies: Sequence[FwdFlySpec] = FWD_FLIES,
) -> Set[str]:
    needed: Set[str] = set()
    for _, a, b in fwd_curves:
        needed.update((a, b))
    for _, a, b, c in fwd_flies:
        needed.update((a, b, c))
    return needed


def build_full_data(
    data_raw: pd.DataFrame,
    curves: Sequence[CurveSpec] = CURVES,
    flies: Sequence[FlySpec] = FLIES,
    fwd_curves: Sequence[FwdCurveSpec] = FWD_CURVES,
    fwd_flies: Sequence[FwdFlySpec] = FWD_FLIES,
    clean: bool = True,
) -> pd.DataFrame:
    """
    From cleaned (or raw Bloomberg) panel → all structures:
      - swap curves / flies (from USOSFR tenors)
      - forward levels (1y1y, 5y5y, …)
      - forward curves / flies (consecutive within each gap group)
    """
    raw = clean_raw_columns(data_raw) if clean else data_raw.copy()

    available = {int(c) for c in raw.columns if is_tenor_column(c)}
    available_fwd = {c for c in raw.columns if is_fwd_structure_column(c)}

    # Keep only structures whose legs exist in the panel (skip the rest)
    curves_ok = [(n, a, b) for n, a, b in curves if a in available and b in available]
    flies_ok = [
        (n, a, b, c)
        for n, a, b, c in flies
        if a in available and b in available and c in available
    ]
    skipped_curves = [n for n, a, b in curves if (n, a, b) not in curves_ok]
    skipped_flies = [n for n, a, b, c in flies if (n, a, b, c) not in flies_ok]
    if skipped_curves or skipped_flies:
        print(
            f"  skip structures (missing tenors): "
            f"curves={skipped_curves or '—'}  flies={skipped_flies or '—'}  "
            f"have_tenors={sorted(available)}",
            flush=True,
        )

    fwd_curves_ok = [
        (n, a, b) for n, a, b in fwd_curves if a in available_fwd and b in available_fwd
    ]
    fwd_flies_ok = [
        (n, a, b, c)
        for n, a, b, c in fwd_flies
        if a in available_fwd and b in available_fwd and c in available_fwd
    ]
    skipped_fc = [n for n, a, b in fwd_curves if (n, a, b) not in fwd_curves_ok]
    skipped_ff = [n for n, a, b, c in fwd_flies if (n, a, b, c) not in fwd_flies_ok]
    if skipped_fc or skipped_ff:
        print(
            f"  skip fwd structures (missing forwards): "
            f"curves={skipped_fc or '—'}  flies={skipped_ff or '—'}",
            flush=True,
        )

    pieces: List[pd.Series] = []

    # Outright swap tenors (so regressions can use e.g. '5' / USOSFR5 as X)
    for t in sorted(available):
        pieces.append(raw[str(t)].rename(str(t)))

    for name, a, b in curves_ok:
        pieces.append(build_curve(raw, name, a, b))
    for name, a, b, c in flies_ok:
        pieces.append(build_fly(raw, name, a, b, c))

    # Forward levels (cleaned S0490FS columns)
    def _fwd_key(name: str) -> Tuple[int, int]:
        m = _FWD_CLEAN_RE.fullmatch(name)
        return (int(m.group(2)), int(m.group(1))) if m else (0, 0)

    for col in sorted(available_fwd, key=_fwd_key):
        pieces.append(raw[col].rename(col))

    for name, a, b in fwd_curves_ok:
        pieces.append(build_fwd_curve(raw, name, a, b))
    for name, a, b, c in fwd_flies_ok:
        pieces.append(build_fwd_fly(raw, name, a, b, c))

    if not pieces:
        raise ValueError("No structures could be built — check raw tenors/forwards")

    full = pd.concat(pieces, axis=1)
    full.index = raw.index
    log.info(
        "full_data: %d cols | tenors=%d | swap curves=%d flies=%d | "
        "fwd levels=%d curves=%d flies=%d | groups=%s | bars=%d",
        full.shape[1],
        len(available),
        len(curves_ok),
        len(flies_ok),
        len(available_fwd),
        len(fwd_curves_ok),
        len(fwd_flies_ok),
        {k: len(v) for k, v in FWD_GROUPS.items()},
        len(full),
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
        print(
            f"  skip predicted targets not in full_data: {missing}",
            flush=True,
        )
    cols = [c for c in predicted if c in full_data.columns]
    if not cols:
        raise KeyError(
            f"None of the PREDICTED targets are in full_data. "
            f"Wanted {list(predicted)}; have {list(full_data.columns)}"
        )
    pred = full_data.loc[:, cols].copy()
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
