# SOFR Curve Screener

| Mode | Command / notebook |
|------|--------------------|
| Prediction + email | `run_prediction.ipynb` (DataZone) or `python run_dz.py prediction --once` |
| Explore | `python run_dz.py pipeline` |
| Backtest | `python run_dz.py backtest` |
| Structure cross-check | `python run_dz.py crosscheck` → `sofr_structures_crosscheck.xlsx` (python vs BBG `USOS####` / `UF######`) |

Knobs in `curve_config.py`:
- `PREDICTION_EVERY_MINUTES` — loop/email cadence (default **30**)
- `GRAPH_LOOKBACK_DAYS` — evolution charts use last **N calendar days**
- Email: `EMAIL_ENABLED`, `SMTP_*`, `REPORT_TO`

PNGs land under `graphs/evolution/`, `graphs/backtest_pnl/`, `graphs/backtest_vars/`.

Lock explore winners in `CHOSEN_MODELS`.
