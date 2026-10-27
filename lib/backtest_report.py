#!/usr/bin/env python3
"""
Refresh latest 180m data → run CHOSEN_MODELS backtest → write a Spanish text report.

  python run_dz.py report
  # or run_backtest_report.ipynb
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from curve_config import (
    BACKTEST_END,
    BACKTEST_POSITION_MODE,
    BACKTEST_RESIDUAL_THR_BP,
    BACKTEST_START,
    BACKTEST_Z_LOOKBACK,
    BACKTEST_ZSCORE_THR,
    BBG_INTERVAL_MINUTES,
    CHOSEN_MODELS,
    TRAIN_SIZE,
)
from lib.data import build_panel, load_panel, refresh_live_store
from lib.models import backtest_chosen_models, save_backtest_plots, summary_table


def _fmt(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    if abs(v) >= 100:
        return "%.1f" % v
    return ("%." + str(nd) + "f") % v


def _model_blurb(s: dict) -> str:
    name = s.get("model", "?")
    target = s.get("structure", s.get("target", "?"))
    feats = s.get("features", "?")
    return (
        "### %s  (`%s ~ %s`)\n\n"
        "- **Fit OOS:** R²=%s  RMSE=%s  folds=%s\n"
        "- **Estrategia** (|z|≥%s residual, 1 barra): trades=%s  PnL=%s  "
        "hit=%s  Sharpe=%s  maxDD=%s  profit_factor=%s\n"
        % (
            name, target, feats,
            _fmt(s.get("oos_r2"), 3),
            _fmt(s.get("oos_rmse"), 3),
            s.get("n_folds", "n/a"),
            _fmt(s.get("entry_z", BACKTEST_ZSCORE_THR), 1)
            if "entry_z" in s else _fmt(BACKTEST_ZSCORE_THR, 1),
            s.get("n_trades", "n/a"),
            _fmt(s.get("total_pnl"), 3),
            _fmt(s.get("hit_rate"), 3),
            _fmt(s.get("sharpe"), 2),
            _fmt(s.get("max_dd"), 3),
            _fmt(s.get("profit_factor"), 2),
        )
    )


def build_report_text(
    summary: pd.DataFrame,
    *,
    asof,
    src: str,
    n_bars: int,
    freq_min: int,
    start,
    end,
    highlight: str = "x7",
) -> str:
    asof_s = pd.Timestamp(asof).strftime("%Y-%m-%d %H:%M") if asof is not None else "?"
    end_s = end or "último bar disponible"
    lines = []
    lines.append("# Informe de backtest — SOFR curve screener")
    lines.append("")
    lines.append("## 1. Datos")
    lines.append("")
    lines.append(
        "Se utiliza el panel live más reciente del screener (`sofr_live` → structures), "
        "muestreado a **%d minutos**. Fuente: `%s`. As-of **%s** (%d barras en el store)."
        % (freq_min, src, asof_s, int(n_bars))
    )
    lines.append("")
    lines.append(
        "Ventana de evaluación: **%s → %s**. "
        "Train walk-forward: **%d** barras; test: 1 barra; step: 1. "
        "Estrategia: mean-reversion del residual cuando `|z| ≥ %g` "
        "(z lookback=%d) y `|residual| ≥ %g` bp; posición **%s**."
        % (
            start, end_s, int(TRAIN_SIZE),
            float(BACKTEST_ZSCORE_THR), int(BACKTEST_Z_LOOKBACK),
            float(BACKTEST_RESIDUAL_THR_BP),
            BACKTEST_POSITION_MODE or "one_bar",
        )
    )
    lines.append("")
    lines.append("## 2. Metodología (resumen)")
    lines.append("")
    lines.append(
        "Para cada modelo de `CHOSEN_MODELS` se estima OLS con intercepto en una ventana "
        "rodante de %d observaciones, se predice la barra siguiente (OOS) y se forma el "
        "residual `r = y − ŷ`. La señal compra residuales baratos (`z` muy negativo) y "
        "vende residuales caros (`z` muy positivo). El PnL de cada barra es "
        "`posición_{t−1} × Δy_t`."
        % int(TRAIN_SIZE)
    )
    lines.append("")
    lines.append(
        "El modelo **x7** (`5s7s10s ~ 2s5s10s`) se ha contrastado a mano (fold 0: "
        "LINEST sobre las 160 filas alineadas vs `backtest_x7.csv`)."
    )
    lines.append("")
    lines.append("## 3. Resultados por modelo")
    lines.append("")

    if summary is None or summary.empty:
        lines.append("_Sin resultados — revisa panel / CHOSEN_MODELS._")
    else:
        # highlight first if present
        rows = list(summary.to_dict(orient="records"))
        rows_sorted = sorted(
            rows,
            key=lambda r: (0 if str(r.get("model")) == highlight else 1, str(r.get("model"))),
        )
        for s in rows_sorted:
            # attach z thr for blurb
            s = dict(s)
            lines.append(_model_blurb(s))
            lines.append("")

        # ranking by sharpe then pnl
        rank = summary.copy()
        if "sharpe" in rank.columns:
            rank = rank.sort_values("sharpe", ascending=False, na_position="last")
        lines.append("## 4. Lectura conjunta")
        lines.append("")
        best = rank.iloc[0] if len(rank) else None
        if best is not None:
            lines.append(
                "Por Sharpe OOS de la estrategia, el mejor en esta corrida es **%s** "
                "(Sharpe=%s, PnL=%s, hit=%s). "
                "Los R² OOS miden calidad de predicción del nivel; el PnL/Sharpe miden "
                "si la regla de residual aporta después de costes implícitos de 1 barra "
                "(sin comisiones modeladas)."
                % (
                    best.get("model"),
                    _fmt(best.get("sharpe"), 2),
                    _fmt(best.get("total_pnl"), 3),
                    _fmt(best.get("hit_rate"), 3),
                )
            )
            lines.append("")
        # x7 focus
        x7 = summary.loc[summary["model"] == highlight] if "model" in summary.columns else summary.iloc[0:0]
        if len(x7):
            s = x7.iloc[0]
            lines.append(
                "En particular **x7** (el verificado a mano): R² OOS=%s, RMSE=%s, "
                "trades=%s, PnL acumulado=%s, hit rate=%s, Sharpe=%s, drawdown máximo=%s. "
                "Interpretación: un R² OOS alto indica que `2s5s10s` explica bien `5s7s10s` "
                "fuera de muestra; el bloque de estrategia indica si apostar a la reversión "
                "del residual con umbral z=%g habría sido rentable en la muestra."
                % (
                    _fmt(s.get("oos_r2"), 3),
                    _fmt(s.get("oos_rmse"), 3),
                    s.get("n_trades"),
                    _fmt(s.get("total_pnl"), 3),
                    _fmt(s.get("hit_rate"), 3),
                    _fmt(s.get("sharpe"), 2),
                    _fmt(s.get("max_dd"), 3),
                    float(BACKTEST_ZSCORE_THR),
                )
            )
            lines.append("")

    lines.append("## 5. Archivos")
    lines.append("")
    lines.append(
        "- Scorecard: `backtest_summary.csv`\n"
        "- Paths por modelo: `backtest_<modelo>.csv` (p.ej. `backtest_x7.csv`)\n"
        "- Panel: `sofr_structures.csv` (freq %dm)\n"
        "- Verificación fold 0: `python run_dz.py verify`"
        % freq_min
    )
    lines.append("")
    lines.append("---")
    lines.append("Generado: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    return "\n".join(lines)


def run_report(
    *,
    refresh: bool = True,
    freq_minutes: int = None,
    start=None,
    end=None,
    out_dir: str = "exports/backtest",
    write_paths: bool = True,
) -> Dict:
    freq = int(freq_minutes if freq_minutes is not None else BBG_INTERVAL_MINUTES)
    start = BACKTEST_START if start is None else start
    end = BACKTEST_END if end is None else end

    if refresh:
        print("  [report] refresh live store @ %dm …" % freq, flush=True)
        try:
            refresh_live_store(force_history=False)
        except Exception as exc:  # noqa: BLE001
            print("  [report] refresh skipped: %s" % exc, flush=True)

    data_raw, src = load_panel()
    _, full, pred = build_panel(data_raw)
    asof = full.index[-1] if len(full) else None
    print(
        "  [report] src=%s  n=%d  asof=%s  freq=%dm  backtest=%s→%s"
        % (src, len(full), asof, freq, start, end or "end"),
        flush=True,
    )

    results = backtest_chosen_models(
        full, pred,
        specs=CHOSEN_MODELS,
        train_size=TRAIN_SIZE,
        start=start,
        end=end,
        residual_thr_bp=BACKTEST_RESIDUAL_THR_BP,
        z_thr=BACKTEST_ZSCORE_THR,
        z_lookback=BACKTEST_Z_LOOKBACK,
        position_mode=BACKTEST_POSITION_MODE,
    )
    if not results:
        raise RuntimeError("No backtest results — check data / CHOSEN_MODELS")

    if write_paths:
        for name, res in results.items():
            res.path.to_csv("backtest_%s.csv" % name)
            try:
                save_backtest_plots(res)
            except Exception:  # noqa: BLE001
                pass

    summary = summary_table(results)
    Path("backtest_summary.csv").write_text(summary.to_csv(index=False))

    text = build_report_text(
        summary,
        asof=asof,
        src=src,
        n_bars=len(full),
        freq_min=freq,
        start=start,
        end=end,
        highlight="x7",
    )

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md_path = dest / ("report_%s.md" % stamp)
    txt_path = dest / ("report_%s.txt" % stamp)
    latest_md = dest / "report_latest.md"
    latest_txt = dest / "report_latest.txt"
    for p in (md_path, txt_path, latest_md, latest_txt):
        p.write_text(text, encoding="utf-8")
    print("  [report] wrote %s" % latest_md.resolve(), flush=True)
    print("  [report] wrote %s" % latest_txt.resolve(), flush=True)
    print("\n" + text, flush=True)
    return {"summary": summary, "results": results, "text": text, "path": latest_md.resolve()}


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest report (180m latest data + text)")
    p.add_argument("--no-refresh", action="store_true", help="Skip live refresh")
    p.add_argument("--freq", type=int, default=None, help="Bar minutes (default 180)")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args(argv)
    run_report(
        refresh=not args.no_refresh,
        freq_minutes=args.freq,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
