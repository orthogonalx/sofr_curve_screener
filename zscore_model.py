# =============================================================================
# zscore_model.py  — Rolling z-score screen on forward flies
# =============================================================================
# For each lookback W in ZSCORE_LOOKBACKS and timestamp t:
#   mu_W, sigma_W from bars (t-W … t-1)   # no look-ahead
#   z_W = (x_t - mu_W) / sigma_W
# fair (predicted) uses ZSCORE_WINDOW.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from curve_config import (
    ZSCORE_LOOKBACKS,
    ZSCORE_MODELS,
    ZSCORE_WINDOW,
    ZScoreSpec,
)


@dataclass
class ZScoreResult:
    spec: ZScoreSpec
    panel: pd.DataFrame   # value, fair, z_150, z_100, z_50, z_10, …
    lookbacks: List[int]
    fair_window: int


def _z_for_window(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=window).mean().shift(1)
    sigma = x.rolling(window, min_periods=window).std(ddof=1).shift(1)
    return (x - mu) / sigma.replace(0.0, np.nan)


def rolling_zscore_multi(
    series: pd.Series,
    lookbacks: Sequence[int] = ZSCORE_LOOKBACKS,
    fair_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Multi-lookback z-scores (past bars only).

    Columns: value | fair | z_150 | z_100 | z_50 | z_10 | …
    """
    x = series.astype(float)
    fair = x.rolling(fair_window, min_periods=fair_window).mean().shift(1)

    data = {"value": x, "fair": fair}
    for w in lookbacks:
        data[f"z_{w}"] = _z_for_window(x, w)

    out = pd.DataFrame(data, index=x.index)
    out.index.name = "time"
    return out


def run_zscore(
    full_data: pd.DataFrame,
    spec: ZScoreSpec,
    lookbacks: Sequence[int] = ZSCORE_LOOKBACKS,
    fair_window: int = ZSCORE_WINDOW,
) -> ZScoreResult:
    if spec.series not in full_data.columns:
        raise KeyError(
            f"Z-score series '{spec.series}' not in full_data. "
            f"Have: {list(full_data.columns)}"
        )
    panel = rolling_zscore_multi(
        full_data[spec.series],
        lookbacks=lookbacks,
        fair_window=fair_window,
    )
    panel.insert(0, "series", spec.series)
    panel.insert(1, "model", spec.param)
    return ZScoreResult(
        spec=spec,
        panel=panel,
        lookbacks=list(lookbacks),
        fair_window=fair_window,
    )


def run_zscore_models(
    full_data: pd.DataFrame,
    specs: Sequence[ZScoreSpec] = ZSCORE_MODELS,
    lookbacks: Sequence[int] = ZSCORE_LOOKBACKS,
    fair_window: int = ZSCORE_WINDOW,
) -> Dict[str, ZScoreResult]:
    results: Dict[str, ZScoreResult] = {}
    for spec in specs:
        results[spec.name] = run_zscore(
            full_data,
            spec,
            lookbacks=lookbacks,
            fair_window=fair_window,
        )
    return results
