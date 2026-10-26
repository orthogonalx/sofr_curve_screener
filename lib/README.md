# lib/

Internal modules (not entry points). Import as `from lib.data import …`, `import lib.correlations as correlations`, etc.

| Module | Role |
|--------|------|
| `data` | BBG pull, live store, panel/structures |
| `models` | OLS, z-scores, prediction email, legacy backtest |
| `correlations` | Corr matrices + Excel export |
| `pull_data` | Lightweight ticker pull |
| `backtest_lab` | Simple residual z-entry backtest |
| `crosscheck` | Python vs BBG structures |
| `period_moves` | Start/end move tables |
