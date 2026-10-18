#!/usr/bin/env python3
"""
Custom prediction — same next-bar OLS as run_prediction, but CUSTOM_MODELS.

Edit in curve_config.py:
  CUSTOM_MODELS  e.g. 2s10s ~ SFRH7
  CUSTOM_CURVES / CUSTOM_FLIES / CROSS_ASSET  if the series is not in the DB yet
Then:  python run_dz.py custom --once --pull-history
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import curve_config as _curve_cfg
from curve_config import CUSTOM_MODELS, DATA_BAR_MINUTES, REFRESH_DATA_ON_REQUEST, TRAIN_SIZE
from data import (
    build_panel,
    build_predicted_data,
    load_panel,
    print_live_levels,
    refresh_live_store,
)
from models import (
    PREDICTION_COLS,
    format_prediction_html,
    format_prediction_table,
    run_chosen_predictions,
    save_evolution_plots,
)

LAST_ASOF = ""
LAST_TEXT = ""
LAST_HTML = ""
LAST_GRAPH_PATHS = []
LAST_TABLE = None


def preview_html(html=None, inline_images=None):
    import base64

    html = LAST_HTML if html is None else html
    paths = LAST_GRAPH_PATHS if inline_images is None else list(inline_images or [])
    out = html or ""
    for i, p in enumerate(paths):
        fp = Path(p)
        if not fp.exists():
            continue
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        out = out.replace("cid:graph%d" % i, "data:image/png;base64,%s" % b64)
    return out


def run_once(
    *,
    refresh: bool = True,
    force_history: bool = False,
    make_graphs: bool = True,
    specs=None,
):
    global LAST_ASOF, LAST_TEXT, LAST_HTML, LAST_GRAPH_PATHS, LAST_TABLE

    specs = list(specs if specs is not None else CUSTOM_MODELS)
    if not specs:
        raise ValueError("CUSTOM_MODELS is empty — add RegressionSpec entries in curve_config.py")

    if refresh or force_history:
        data_raw, src = refresh_live_store(force_history=force_history)
    else:
        data_raw, src = load_panel()
    print_live_levels(data_raw, src=src)
    _, full, _ = build_panel(data_raw)

    targets = []
    for s in specs:
        if s.target not in targets:
            targets.append(s.target)
    missing_t = [t for t in targets if t not in full.columns]
    missing_f = sorted({
        f for s in specs for f in s.features if f not in full.columns
    })
    if missing_t or missing_f:
        raise KeyError(
            "Custom series missing from panel. "
            "Add to CUSTOM_CURVES / CUSTOM_FLIES / CROSS_ASSET and re-run with --pull-history.\n"
            "  missing targets=%s  features=%s  have=%s"
            % (missing_t or "—", missing_f or "—", list(full.columns)[:40])
        )

    pred = build_predicted_data(full, predicted=targets)
    asof = str(full.index[-1])
    print(
        "custom prediction  src=%s  n=%d  asof=%s  models=%s"
        % (src, len(full), asof, [m.name for m in specs]),
        flush=True,
    )
    table = format_prediction_table(
        run_chosen_predictions(full, pred, specs=specs, train_size=TRAIN_SIZE)
    )
    cols = [c for c in PREDICTION_COLS if c in table.columns]
    show = table[cols]
    text = (
        "SOFR CUSTOM prediction  |  asof %s  |  k=%s  |  next ~%sm\n\n%s\n"
        % (asof, TRAIN_SIZE, DATA_BAR_MINUTES, show.to_string(index=False))
    )
    print(text, flush=True)
    Path("predictions_custom.csv").write_text(table.to_csv(index=False))

    graph_paths = []
    if make_graphs:
        graph_paths = save_evolution_plots(
            full,
            days=getattr(_curve_cfg, "GRAPH_LOOKBACK_DAYS", 9),
            specs=specs,
            out_dir="graphs/custom_evolution",
        )
    graph_cids = []
    for i, p in enumerate(graph_paths):
        graph_cids.append(("graph%d" % i, Path(p).stem.replace("evolution_", "")))

    html = format_prediction_html(table, asof=asof, graph_cids=graph_cids)
    LAST_ASOF, LAST_TEXT, LAST_HTML = asof, text, html
    LAST_GRAPH_PATHS = list(graph_paths)
    LAST_TABLE = table
    return table


def main() -> None:
    p = argparse.ArgumentParser(description="Custom-model prediction (CUSTOM_MODELS)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--pull-history", action="store_true")
    p.add_argument("--no-graphs", action="store_true")
    args = p.parse_args()
    refresh = REFRESH_DATA_ON_REQUEST and not args.no_refresh
    run_once(
        refresh=refresh,
        force_history=args.pull_history,
        make_graphs=not args.no_graphs,
    )


if __name__ == "__main__":
    main()
