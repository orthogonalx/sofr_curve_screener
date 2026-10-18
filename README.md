# SOFR Curve Screener

| Mode | Command |
|------|---------|
| Prediction | `run_prediction.ipynb` or `python run_dz.py prediction --once` |
| Explore | `python run_dz.py pipeline` |
| Backtest | `python run_dz.py backtest` (OOS fit + residual strategy scorecard) |
| Cross-check | `run_crosscheck.ipynb` or `python run_dz.py crosscheck` |
| Correlations | `run_correlations.ipynb` — edit **`CORR_SERIES`** in `curve_config.py` |
| Custom prediction | `run_custom.ipynb` — edit **`CUSTOM_MODELS`** (+ `CUSTOM_CURVES` / `CROSS_ASSET` if new) |

Knobs: `CORR_SERIES`, `CUSTOM_MODELS`, `CUSTOM_CURVES`, `CROSS_ASSET`, `USE_BBG_SWAP_STRUCTURES`, email `SMTP_*`.

CURVES/FLIES = BBG `USOS`/`UF`. Custom series: add to `CUSTOM_*` / `CROSS_ASSET`, then `--pull-history`.
