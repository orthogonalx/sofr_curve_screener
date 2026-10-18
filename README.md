# SOFR Curve Screener

| Mode | Command |
|------|---------|
| Prediction | `run_prediction.ipynb` or `python run_dz.py prediction --once` |
| Explore | `python run_dz.py pipeline` |
| Backtest | `python run_dz.py backtest` (OOS fit + residual strategy scorecard) |
| Cross-check | `run_crosscheck.ipynb` or `python run_dz.py crosscheck` |
| Correlations | `run_correlations.ipynb` or `python run_dz.py correlations` |

Knobs: `PREDICTION_EVERY_MINUTES`, `GRAPH_LOOKBACK_DAYS`, `USE_BBG_SWAP_STRUCTURES`, `CROSS_ASSET` / `CORR_SERIES`, email `SMTP_*`.

CURVES/FLIES = BBG `USOS####` / `UF######`. Fwd flies = built. Cross-assets = `CROSS_ASSET`.
