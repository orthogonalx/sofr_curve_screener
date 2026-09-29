# =============================================================================
# bbg_pull.py  — Placeholder for Bloomberg hourly snapshot
# =============================================================================
# Replace `fetch_bloomberg_latest` with your real BBG pull.
# Contract: return a 1-row DataFrame with DatetimeIndex named "time" and the
# same Bloomberg columns as the historical panel (USOSFR…, S0490FS …).
# =============================================================================

from __future__ import annotations

from typing import Optional

import pandas as pd

from data_loader import generate_synthetic_sofr


def fetch_bloomberg_latest(
    asof: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    STUB: simulate one new hourly BBG print.

    Swap this body for your Bloomberg API call. Keep the return shape identical.
    """
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.utcnow().floor("h")
    # Synthetic one-bar panel with a unique seed so values move over time
    seed = int(asof.value % (2**31 - 1))
    panel = generate_synthetic_sofr(n_bars=1, freq="h", seed=seed)
    panel.index = pd.DatetimeIndex([asof], name="time")
    return panel
