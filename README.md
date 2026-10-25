# SOFR Curve Screener

Live SOFR curve/fly screener on Bloomberg (`bdib` → `sofr_live.csv`). Structures from BBG `USOS`/`UF` (+ computed fwd flies). Run everything via DataZone:

```bash
git pull origin main
python run_dz.py <cmd>
```

Or open the matching `run_*.ipynb`.

## Commands

| Cmd | What it does |
|-----|----------------|
| `prediction --once` | Fit OLS models → next-bar forecast + z-scores + email (if on) |
| `prediction --once --pull-history` | Same, but refresh full BBG history into live store first |
| `pipeline` | Explore / one-shot panel + graphs |
| `backtest` | OOS fit + residual strategy scorecard |
| `crosscheck` | Python vs BBG listed curves/flies → Excel |
| `correlations` | Level corr + percentile heatmap + scatter + Excel panel |
| `custom --once` | Your `CUSTOM_MODELS` prediction |
| `moves --start "…" --end "…"` | Level change / std / z between two timestamps |

## Correlations

1. Edit `CORR_SERIES` (and scatter legs) in `curve_config.py` — **clean names** (`CL1`, not `CL1 Comdty`)
2. `python run_dz.py correlations` or `run_correlations.ipynb`
3. Outputs:
   - `graphs/correlations/` — heatmaps + scatter
   - `exports/correlations/corr_panel.xlsx` — levels for Excel rolling corrs

Knobs: `CORR_WINDOW=50` (bars per corr), `CORR_PCT_LOOKBACK=200` (percentile history).  
Missing series auto-pull; `CL1`/`CO1` fall back to daily `bdh` if intradaily is `NOT_ENTITLED`.

## Config (`curve_config.py`)

| Knob | Role |
|------|------|
| `CURVES` / `FLIES` | Main structures (BBG USOS/UF) |
| `CUSTOM_CURVES` / `CUSTOM_FLIES` | Extra structures |
| `CROSS_ASSET` | Extra tickers `(clean, "TICKER BBG")` |
| `CORR_SERIES` | Corr matrix columns |
| `CUSTOM_MODELS` | Custom prediction specs |
| `MOVE_OUTRIGHTS` / `MOVE_STRUCTURES` | Period-move tables |
| `SMTP_*` / `EMAIL_*` | Prediction email |

New ticker → add to `CROSS_ASSET` (or `CUSTOM_*`) → `--pull-history` once.
