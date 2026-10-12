# SOFR Curve Screener

| Mode | Command |
|------|---------|
| Explore | `python3.8 -u run_pipeline.py` |
| Backtest | `python3.8 -u run_backtest.py` |
| Prediction (live) | `python3.8 -u run_prediction.py --once` |

Layout: `curve_config.py` (knobs) · `data.py` (I/O + BBG + structures) · `models.py` (OLS / z / strategy / email) · three runners.

Prediction refreshes `sofr_live.csv` (BBG) then forecasts. `--no-refresh` skips BBG; `--pull-history` rebuilds history.

Lock explore winners in `CHOSEN_MODELS`.
