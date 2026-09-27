# SOFR Curve Screener

Walk-forward screening of SOFR swap curves and flies on 3-hour bars.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook helper.ipynb
```

## Data

`DATA_PATH` in `curve_config.py` defaults to `sofr_3h.xlsx` in this directory. The file can be Excel or CSV with a `time` column and Bloomberg tenor columns such as `USOSFR10 BGN Curncy`.

If that file is missing, the notebook falls back to `generate_synthetic_sofr` so the rest of the pipeline still runs.
