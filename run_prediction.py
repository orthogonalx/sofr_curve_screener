#!/usr/bin/env python3
"""Prediction: refresh live data → CHOSEN_MODELS next-bar forecast."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import curve_config as _curve_cfg
from curve_config import (
    CHOSEN_MODELS,
    DATA_BAR_MINUTES,
    REFRESH_DATA_ON_REQUEST,
    TRAIN_SIZE,
)
from data import build_panel, load_panel, print_live_levels, refresh_live_store
from models import (
    PREDICTION_COLS,
    append_zscore_latest,
    format_prediction_html,
    format_prediction_table,
    run_chosen_predictions,
    save_evolution_plots,
    send_summary_email,
)

LAST_ASOF = ""
LAST_TEXT = ""
LAST_HTML = ""
LAST_GRAPH_PATHS = []

SEND_EMAIL_ON_PREDICTION = getattr(_curve_cfg, "SEND_EMAIL_ON_PREDICTION", True)
EMAIL_INLINE_GRAPHS = getattr(_curve_cfg, "EMAIL_INLINE_GRAPHS", True)
GRAPH_LOOKBACK_DAYS = getattr(_curve_cfg, "GRAPH_LOOKBACK_DAYS", 9)


def _loop_minutes():
    if hasattr(_curve_cfg, "PREDICTION_EVERY_MINUTES"):
        return float(_curve_cfg.PREDICTION_EVERY_MINUTES)
    return float(getattr(_curve_cfg, "PREDICTION_EMAIL_EVERY_HOURS", 0.5)) * 60.0


def run_once(
    *,
    refresh: bool = True,
    force_history: bool = False,
    send_email: Optional[bool] = None,
    make_graphs: bool = True,
):
    global LAST_ASOF, LAST_TEXT, LAST_HTML, LAST_GRAPH_PATHS

    if refresh or force_history:
        data_raw, src = refresh_live_store(force_history=force_history)
    else:
        data_raw, src = load_panel()
    print_live_levels(data_raw, src=src)
    _, full, pred = build_panel(data_raw)
    asof = str(data_raw.index[-1])
    print(
        "prediction  src=%s  n=%d  asof=%s  models=%s"
        % (src, len(data_raw), asof, [m.name for m in CHOSEN_MODELS]),
        flush=True,
    )
    table = format_prediction_table(
        append_zscore_latest(
            run_chosen_predictions(full, pred, specs=CHOSEN_MODELS, train_size=TRAIN_SIZE),
            full,
        )
    )
    cols = [c for c in PREDICTION_COLS if c in table.columns]
    show = table[cols]
    text = (
        "SOFR prediction  |  asof %s  |  k=%s  |  next ~%sm\n\n%s\n"
        % (asof, TRAIN_SIZE, DATA_BAR_MINUTES, show.to_string(index=False))
    )
    print(text, flush=True)
    Path("predictions_latest.csv").write_text(table.to_csv(index=False))
    LAST_ASOF, LAST_TEXT = asof, text

    graph_paths = []
    if make_graphs:
        graph_paths = save_evolution_plots(
            full, days=GRAPH_LOOKBACK_DAYS, specs=CHOSEN_MODELS
        )
    LAST_GRAPH_PATHS = list(graph_paths)

    # CID list for HTML (prefer combined chart first)
    graph_cids = []
    inline_paths = []
    if EMAIL_INLINE_GRAPHS and graph_paths:
        ordered = sorted(
            graph_paths,
            key=lambda p: (0 if Path(p).name.startswith("evolution_structures") else 1, Path(p).name),
        )
        for i, p in enumerate(ordered):
            cid = "graph%d" % i
            title = Path(p).stem.replace("evolution_", "").replace("_", " ")
            graph_cids.append((cid, title))
            inline_paths.append(p)

    html = format_prediction_html(table, asof=asof, graph_cids=graph_cids)
    LAST_HTML = html

    do_email = SEND_EMAIL_ON_PREDICTION if send_email is None else bool(send_email)
    if do_email:
        send_summary_email(
            table,
            asof=asof,
            subject="SOFR prediction %s" % asof,
            body=text,
            html_body=html,
            inline_images=inline_paths,
            attach_paths=None,
        )
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--pull-history", action="store_true")
    p.add_argument("--no-email", action="store_true")
    p.add_argument("--no-graphs", action="store_true")
    args = p.parse_args()
    refresh = REFRESH_DATA_ON_REQUEST and not args.no_refresh

    def _tick():
        run_once(
            refresh=refresh,
            force_history=args.pull_history,
            send_email=False if args.no_email else None,
            make_graphs=not args.no_graphs,
        )

    if args.once:
        _tick()
        return
    mins = max(_loop_minutes(), 1.0)
    print("prediction loop every %g min" % mins, flush=True)
    while True:
        try:
            _tick()
        except Exception as exc:  # noqa: BLE001
            print("[prediction] ERROR %s" % exc, flush=True)
        time.sleep(mins * 60.0)


if __name__ == "__main__":
    main()
