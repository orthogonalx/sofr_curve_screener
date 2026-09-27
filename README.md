# SOFR Curve Screener

Quick note for the team — this is a small local pipeline I’ve been building to screen SOFR swap / forward structures for dislocations.

## What it does

We load a 3h panel of SOFR swap tenors (`USOSFR…`) and forward rates (`S0490FS …Y…Y BLC Curncy`), clean the column names, and build the structures we care about:

- **Swap curves / flies** in bp: e.g. `10s30s = (30Y − 10Y) × 100`, `5s7s10s = (2×7 − 5 − 10) × 100`
- **Forwards** on 1y / 2y / 5y gaps (e.g. `1y1y…9y1y`, `1y2y…10y2y`, `5y5y…35y5y`)
- **Forward curves / flies** built *within* each gap group (consecutive pairs and triples), same bp convention

On top of that we run two types of screens:

1. **Walk-forward regressions** (1-step rolling): fit on the last *W* bars, predict the next bar, slide by 1. We compare a few train windows and keep the one with the best OOS RMSE.
2. **Z-score screens** on selected forward flies: for each lookback (150 / 100 / 50 / 10) we compute how many stdevs the current print is vs the *past* window only (no look-ahead).

The main deliverable is a **summary table** for the latest timestamp: level, model fair value, model id, residual (for regressions), and structure name. Full path histories and multi-window z-scores are printed / saved per model.

## Models currently wired

| Id | Type | Spec |
|---|---|---|
| `x7` | regression | `5s7s10s ~ 2s5s10s` |
| `x20` | regression | `10s20s30s ~ 10s30s` |
| `y15` | regression | `10s15s30s ~ 5s10s30s` |
| `y25` | regression | `20s25s30s ~ 10s30s` |
| `y4` | regression | `3s4s5s ~ 2s5s10s` |
| `z6` | z-score | fly `4y1y_5y1y_6y1y` (around 5y1y) |
| `z8` | z-score | fly `6y1y_7y1y_8y1y` (around 7y1y) |
| `z9` | z-score | fly `7y1y_8y1y_9y1y` (around 8y1y) |
| `z12` | — | TBD |

Easy to extend: add structures in `curve_config.py` and append a `RegressionSpec` / `ZScoreSpec`.

## How to run

```bash
pip install -r requirements.txt
python3.8 -u run_pipeline.py
```

Or open `helper.ipynb` if you prefer stepping through cells.

Point `DATA_PATH` in `curve_config.py` at your Excel/CSV (`sofr_3h.xlsx` by default). If the file isn’t there, the pipeline falls back to synthetic SOFR so you can still dry-run the code.

Outputs (gitignored):

- `summary.csv` — latest snapshot across models
- `predictions_<model>.csv` — full OOS path for each regression
- `zscore_*.csv` — z-score histories
- `residual_evolution_*.png` — residual charts

## Notes / caveats

- Units: structure diffs are in **bp** (`× 100` on percent rates).
- Missing data: load does a forward-fill; regressions drop rows where target/features are NaN; z-scores need a full lookback of past bars before they start printing.
- This is a research / screening tool, not a live trading system. Treat signals as a starting point for a look, not as automated trades.

Happy to walk anyone through the config if you want to add another point on the curve.
