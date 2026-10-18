# SOFR Curve Screener

| Mode | Command |
|------|---------|
| Prediction | `run_prediction.ipynb` or `python run_dz.py prediction --once` |
| Explore | `python run_dz.py pipeline` |
| Backtest | `python run_dz.py backtest` |
| Cross-check | `run_crosscheck.ipynb` or `python run_dz.py crosscheck` |

Knobs in `curve_config.py`: `PREDICTION_EVERY_MINUTES`, `GRAPH_LOOKBACK_DAYS`, `USE_BBG_SWAP_STRUCTURES`, email `SMTP_*` / `REPORT_TO`.

CURVES/FLIES = BBG `USOS####` / `UF######`. Fwd curves/flies = built from `S0490FS`.
