#!/usr/bin/env python3
"""One-shot screen: load panel → summary table only."""

from pathlib import Path

from curve_config import DATA_PATH, LIVE_STORE_PATH, LIVE_TUNE_WINDOW
from data_loader import generate_synthetic_sofr, load_raw
from live_store import load_live_store
from screener_core import emit_summary, run_screen


def _load_panel():
    live = Path(LIVE_STORE_PATH)
    if live.exists():
        return load_live_store(live), f"live:{live}"
    path = Path(DATA_PATH)
    if path.exists():
        return load_raw(path), f"file:{path}"
    return generate_synthetic_sofr(n_bars=400, freq="h"), "synthetic"


def main() -> None:
    data_raw, src = _load_panel()
    asof = str(data_raw.index[-1])
    summary, _, _ = run_screen(
        data_raw, tune_window=LIVE_TUNE_WINDOW, quiet=True
    )
    print(f"src={src}  n={len(data_raw)}  asof={asof}", flush=True)
    emit_summary(summary, asof=asof)


if __name__ == "__main__":
    main()
