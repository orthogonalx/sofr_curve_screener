# SOFR Curve Screener

Three modes:

| Mode | Command | Role |
|------|---------|------|
| **Explore** | `python3.8 -u run_pipeline.py` | Compare `REGRESSION_SCREENS` feature sets, pick winners |
| **Report** | `python3.8 -u run_report.py --once` | Residual table (CHOSEN_MODELS); email loop without `--once` every XXX h |
| **Prediction** | `python3.8 -u run_prediction.py --once` | Locked `CHOSEN_MODELS` → next-bar forecast (+ email loop without `--once`) |
| **Backtest** | `python3.8 -u run_backtest.py` | Residual mean-reversion on `CHOSEN_MODELS` (z + \|r\| thr → PnL) |

Optional BBG ingest only: `python3.8 -u run_live.py` (does **not** predict).

## Config (`curve_config.py`)

1. Add curves/flies in `CURVES` / `FLIES` (built from outright USOSFR levels).
2. Explore candidates in `REGRESSION_SCREENS.feature_sets`.
3. After explore, copy winners into **`CHOSEN_MODELS`** (prediction + backtest use only these).
4. Cadence: `DATA_BAR_MINUTES` (next bar ~15m), `REPORT_EMAIL_EVERY_HOURS` (XXX report email), `PREDICTION_EMAIL_EVERY_HOURS` (ZZZ), `TRAIN_SIZE` (k).

## Data

Point `DATA_PATH` at your Excel/CSV, or use `sofr_live.csv` if present. Missing file → synthetic panel.

```bash
pip install -r requirements.txt
python3.8 -u run_pipeline.py
# lock models in CHOSEN_MODELS, then:
python3.8 -u run_prediction.py --once
python3.8 -u run_backtest.py
```

Outputs (gitignored): `summary.csv`, `predictions_*.csv`, `predictions_latest.csv`, `backtest_*.csv`, z-score CSVs.
