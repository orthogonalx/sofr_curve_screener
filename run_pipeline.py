#!/usr/bin/env python3
"""Run the SOFR curve screener → regressions + z-scores → summary table."""

from pathlib import Path

from curve_config import (
    DATA_PATH,
    REGRESSIONS,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_WINDOWS,
    ZSCORE_LOOKBACKS,
    ZSCORE_MODELS,
    ZSCORE_WINDOW,
)
from data_loader import generate_synthetic_sofr, load_raw
from rolling_model import (
    instance_predictions,
    plot_residual_evolution,
    run_regressions,
)
from structures import build_datasets
from summary import build_summary
from zscore_model import run_zscore_models


def main() -> None:
    print("=" * 72, flush=True)
    print("SOFR curve screener", flush=True)
    print("=" * 72, flush=True)

    path = Path(DATA_PATH)
    if path.exists():
        data_raw = load_raw(path)
        print(f"data_raw: FILE {path.resolve()}  shape={data_raw.shape}", flush=True)
    else:
        data_raw = generate_synthetic_sofr(n_bars=400)
        print(f"data_raw: SYNTHETIC (missing {path})  shape={data_raw.shape}", flush=True)

    _, full_data, predicted_data = build_datasets(data_raw)
    print(
        f"full_data={full_data.shape}  predicted={list(predicted_data.columns)}",
        flush=True,
    )
    print(
        f"rolling: train in {list(TRAIN_WINDOWS)} (pick best)  "
        f"test={TEST_SIZE}  step={STEP_SIZE}",
        flush=True,
    )
    print(
        f"models: regs={[r.name for r in REGRESSIONS]}  "
        f"z={[z.param + ':' + z.series for z in ZSCORE_MODELS]}",
        flush=True,
    )

    results = run_regressions(
        full_data, predicted_data, specs=REGRESSIONS, tune_window=True
    )

    for spec in REGRESSIONS:
        res = results[spec.name]
        train_size = int(res.overall.get("train_size", 0))
        feat = "+".join(spec.features)

        print("\n" + "=" * 72, flush=True)
        print(
            f"Regression: {spec.name}  |  {spec.target} ~ {feat}  |  "
            f"train_size={train_size}  "
            f"OOS R2={res.overall.get('r2', float('nan')):.4f}  "
            f"RMSE={res.overall.get('rmse', float('nan')):.5f}",
            flush=True,
        )
        print("=" * 72, flush=True)

        plot_residual_evolution(res, outfile=f"residual_evolution_{spec.name}.png")
        preds = instance_predictions(res)
        preds.to_csv(Path(f"predictions_{spec.name}.csv"))
        print("Last 5:", flush=True)
        print(preds.tail(5).to_string(), flush=True)

    print(
        f"\nZ-score models: lookbacks={ZSCORE_LOOKBACKS}  "
        f"fair_window={ZSCORE_WINDOW}  (past bars only)",
        flush=True,
    )
    z_results = run_zscore_models(full_data, specs=ZSCORE_MODELS)

    z_cols = [f"z_{w}" for w in ZSCORE_LOOKBACKS]
    for spec in ZSCORE_MODELS:
        res = z_results[spec.name]
        # keep rows where the shortest lookback is available
        panel = res.panel.dropna(subset=[f"z_{min(ZSCORE_LOOKBACKS)}"])
        safe = spec.name.replace(":", "_")
        panel.to_csv(Path(f"zscore_{safe}.csv"))

        print("\n" + "=" * 72, flush=True)
        print(
            f"Z-score: {spec.param}  |  {spec.series}  |  "
            f"lookbacks={ZSCORE_LOOKBACKS}",
            flush=True,
        )
        print("=" * 72, flush=True)
        print("Last 5:", flush=True)
        print(panel[["value", "fair"] + z_cols].tail(5).to_string(), flush=True)

    summary = build_summary(results, z_results)
    out_sum = Path("summary.csv")
    summary.to_csv(out_sum, index=False)

    print("\n" + "=" * 72, flush=True)
    print("SUMMARY (latest observation per model)", flush=True)
    print("=" * 72, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSummary → {out_sum.resolve()}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
