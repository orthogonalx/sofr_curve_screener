# =============================================================================
# report.py  — Residual report table (CHOSEN_MODELS / default models)
# =============================================================================
# On each request:
#   1) optional data refresh (BBG pull → append live store)
#   2) fit last-k OLS per CHOSEN_MODELS
#   3) return a single table:
#        parametro | fly | residual | y_real | y_pred | regressors | model | periodo
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from bbg_pull import append_live_bar, pull_history_into_store
from curve_config import (
    BBG_END_DATE,
    BBG_FREQ,
    BBG_START_DATE,
    CHOSEN_MODELS,
    LIVE_STORE_PATH,
    REFRESH_DATA_ON_REQUEST,
    TRAIN_SIZE,
    RegressionSpec,
)
from live_store import ensure_seed_history, load_live_store
from rolling_model import fit_predict_next
from screener_core import build_panel, load_panel


REPORT_COLS = [
    "parametro",
    "fly",
    "residual",
    "y_real",
    "y_pred",
    "regressors",
    "model",
    "periodo",
]


def refresh_data(force_history: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    1. If the live store is missing (or force_history), pull
       [BBG_START_DATE, BBG_END_DATE] at BBG_FREQ into LIVE_STORE_PATH.
    2. Pull the latest print and append it onto that dataset.

    Falls back to the existing panel (or a synthetic seed) if Bloomberg is down.
    """
    store = Path(LIVE_STORE_PATH)
    try:
        existing = load_live_store(store) if store.exists() else pd.DataFrame()
        if force_history or existing.empty:
            end = BBG_END_DATE or "today"
            print(
                f"  [history] {BBG_START_DATE} → {end}  freq={BBG_FREQ}",
                flush=True,
            )
            pull_history_into_store()
        panel = append_live_bar()
        return panel, f"live+pull:{store}"
    except Exception as exc:  # noqa: BLE001
        print(f"  [refresh] pull failed ({exc}); using existing panel", flush=True)
        if not store.exists():
            ensure_seed_history(store)
        data, src = load_panel()
        return data, f"fallback:{src}"


def build_residual_report(
    specs: Sequence[RegressionSpec] = CHOSEN_MODELS,
    train_size: int = TRAIN_SIZE,
    refresh: bool = REFRESH_DATA_ON_REQUEST,
    force_history: bool = False,
) -> pd.DataFrame:
    """
    Refresh data (optional) → fit default models → residual report table only.
    """
    if refresh or force_history:
        data_raw, src = refresh_data(force_history=force_history)
    else:
        data_raw, src = load_panel()

    _, full_data, predicted_data = build_panel(data_raw)
    asof = data_raw.index[-1]
    print(
        f"report  src={src}  n={len(data_raw)}  asof={asof}  "
        f"models={[s.name for s in specs]}  k={train_size}",
        flush=True,
    )

    rows = []
    for spec in specs:
        try:
            r = fit_predict_next(
                full_data, predicted_data, spec, train_size=train_size
            )
        except (KeyError, ValueError) as exc:
            print(f"  skip {spec.name}: {exc}", flush=True)
            continue
        periodo = f"{r['train_start']} → {r['train_end']}  (k={r['k']}; pred @ {r['time']})"
        rows.append(
            {
                "parametro": r["model"],
                "fly": r["structure"],
                "residual": round(float(r["residual"]), 4),
                "y_real": round(float(r["nivel"]), 4),
                "y_pred": round(float(r["y_hat_now"]), 4),
                "regressors": r["features"],
                "model": r["equation"],
                "periodo": periodo,
            }
        )

    table = pd.DataFrame(rows, columns=REPORT_COLS)
    table.attrs["asof"] = str(asof)
    table.attrs["src"] = src
    return table


def format_report_text(table: pd.DataFrame, asof: Optional[str] = None) -> str:
    """Plain-text body: only the residual report table."""
    asof = asof or table.attrs.get("asof", "")
    header = f"SOFR residual report  |  asof {asof}"
    if table.empty:
        return header + "\n\n(no rows)\n"
    return header + "\n\n" + table.to_string(index=False) + "\n"


def run_report(
    *,
    refresh: bool = REFRESH_DATA_ON_REQUEST,
    force_history: bool = False,
    save_csv: bool = True,
    outfile: str = "residual_report.csv",
) -> pd.DataFrame:
    """Build report, print table, optionally save CSV."""
    table = build_residual_report(refresh=refresh, force_history=force_history)
    text = format_report_text(table)
    print(text, flush=True)
    if save_csv:
        Path(outfile).write_text(table.to_csv(index=False))
        print(f"  wrote {outfile}", flush=True)
    table.attrs["text"] = text
    return table
