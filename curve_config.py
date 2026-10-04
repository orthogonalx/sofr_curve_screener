# =============================================================================
# curve_config.py  — SOFR curve screener: structures & model params
# =============================================================================
# Swap tenors (USOSFR):
#   curve  XsYs   = (Y - X) * 100          # percent → bp
#   fly    XsYsZs = (2*Y - X - Z) * 100    # percent → bp
#
# Forwards (S0490FS startYgapY → "{start}y{gap}y"):
#   curve  A_B     = (B - A) * 100         within a gap group
#   fly    A_B_C   = (2*B - A - C) * 100   consecutive triples in a group
# =============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ── Data paths (DataZone: point these at your uploaded file) ──────────────────
DATA_PATH = "sofr_3h.xlsx"          # .xlsx or .csv
TIMESTAMP_COL = "time"              # matches Bloomberg export screenshot
SHEET_NAME = 0

# Bloomberg outright swap: "USOSFR{N} BGN Curncy" → tenor "{N}"
BBG_PREFIX = "USOSFR"
BBG_SUFFIX = "BGN Curncy"

# Bloomberg forward: "S0490FS 20Y5Y BLC Curncy" → "20y5y"
BBG_FWD_PREFIX = "S0490FS"
BBG_FWD_SUFFIX = "BLC Curncy"

# ── Forward grid (start years, gap years) → cleaned name "{start}y{gap}y" ─────
# 1y gaps: 1y1y … 9y1y
# 2y gaps: 1y2y … 10y2y
# 5y gaps: 5y5y, 10y5y, … 35y5y
FwdPoint = Tuple[int, int]  # (start, gap)


def fwd_name(start: int, gap: int) -> str:
    return f"{start}y{gap}y"


FORWARDS: List[FwdPoint] = (
    [(s, 1) for s in range(1, 10)]          # 1y1y … 9y1y
    + [(s, 2) for s in range(1, 11)]        # 1y2y … 10y2y
    + [(s, 5) for s in (5, 10, 15, 20, 25, 30, 35)]
)

# Ordered groups used to build consecutive curves / flies
FWD_GROUPS: Dict[str, List[str]] = {
    "1y": [fwd_name(s, 1) for s in range(1, 10)],
    "2y": [fwd_name(s, 2) for s in range(1, 11)],
    "5y": [fwd_name(s, 5) for s in (5, 10, 15, 20, 25, 30, 35)],
}

# Forward structures: (name, col_a, col_b) / (name, left, body, right)
FwdCurveSpec = Tuple[str, str, str]
FwdFlySpec = Tuple[str, str, str, str]


def _expand_fwd_group_structures(
    groups: Dict[str, List[str]],
) -> Tuple[List[FwdCurveSpec], List[FwdFlySpec]]:
    curves: List[FwdCurveSpec] = []
    flies: List[FwdFlySpec] = []
    for cols in groups.values():
        for i in range(len(cols) - 1):
            a, b = cols[i], cols[i + 1]
            curves.append((f"{a}_{b}", a, b))
        for i in range(len(cols) - 2):
            a, b, c = cols[i], cols[i + 1], cols[i + 2]
            flies.append((f"{a}_{b}_{c}", a, b, c))
    return curves, flies


FWD_CURVES, FWD_FLIES = _expand_fwd_group_structures(FWD_GROUPS)

# ── Swap structures (built from outright USOSFR levels in the data file) ───────
# INPUT data = outright swap levels only (USOSFR1, USOSFR2, …).
# Curves / flies below are COMPUTED in structures.build_full_data from those
# outrights — you do NOT put curve columns in the spreadsheet.
#
# To add a curve  XsYs     → append to CURVES:  ("XsYs", X, Y)
# To add a fly   XsYsZs   → append to FLIES:   ("XsYsZs", X, Y, Z)
#   e.g. 2s3s5s fly → ("2s3s5s", 2, 3, 5)
# Formulae (percent → bp):
#   curve  = (Y − X) * 100
#   fly    = (2*Y − X − Z) * 100
#
# Required tenors must exist as outrights in the panel (e.g. 30s50s needs USOSFR50).

CurveSpec = Tuple[str, int, int]       # (name, short, long)
FlySpec = Tuple[str, int, int, int]    # (name, left, body, right)

CURVES: List[CurveSpec] = [
    ("2s5s", 2, 5),
    ("5s10s", 5, 10),
    ("10s20s", 10, 20),
    ("10s30s", 10, 30),
    ("20s30s", 20, 30),
    ("30s35s", 30, 35),
    ("30s40s", 30, 40),
    ("30s50s", 30, 50),
]

FLIES: List[FlySpec] = [
    ("2s5s10s", 2, 5, 10),
    ("3s4s5s", 3, 4, 5),
    ("5s7s10s", 5, 7, 10),
    ("5s10s30s", 5, 10, 30),
    ("10s15s30s", 10, 15, 30),
    ("10s20s30s", 10, 20, 30),
    ("10s30s35s", 10, 30, 35),
    ("20s25s30s", 20, 25, 30),
    # example: ("2s3s5s", 2, 3, 5),
]

# Targets we want to screen / predict (must appear in CURVES or FLIES)
PREDICTED: List[str] = [
    "5s7s10s",
    "10s20s30s",
    "10s15s30s",
    "20s25s30s",
    "3s4s5s",
    "30s35s",
    "30s40s",
    "30s50s",
]

# ── Regression specs ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RegressionSpec:
    """One fitted model: target ~ features (+ intercept) at fixed TRAIN_SIZE."""
    target: str
    features: Tuple[str, ...]
    name: str = ""                  # model id: x7, y15, … ; default target~feats

    def __post_init__(self):
        if not self.name:
            object.__setattr__(
                self, "name", f"{self.target}~{'+'.join(self.features)}"
            )


@dataclass(frozen=True)
class RegressionScreen:
    """
    One screening problem: fixed target + fixed k, compare candidate feature sets.

    feature_sets is the INPUT — each tuple is a candidate X (structures and/or
    tenors). Tenors may be written as '5', 'USOSFR5', etc. (resolved to '5').
    """
    name: str                       # screen id: x7, x20, …
    target: str
    feature_sets: Tuple[Tuple[str, ...], ...]


REGRESSION_SCREENS: List[RegressionScreen] = [
    RegressionScreen(
        name="x7",
        target="5s7s10s",
        feature_sets=(
            ("2s5s10s",),
            ("2s5s10s", "USOSFR5"),
        ),
    ),
    RegressionScreen(
        name="x20",
        target="10s20s30s",
        feature_sets=(
            ("10s30s",),
            ("10s30s", "USOSFR5"),
        ),
    ),
    RegressionScreen(
        name="y15",
        target="10s15s30s",
        feature_sets=(
            ("5s10s30s",),
            ("5s10s30s", "USOSFR5"),
        ),
    ),
    RegressionScreen(
        name="y25",
        target="20s25s30s",
        feature_sets=(
            ("10s30s",),
            ("10s30s", "USOSFR5"),
        ),
    ),
    RegressionScreen(
        name="y4",
        target="3s4s5s",
        feature_sets=(
            ("2s5s10s",),
            ("2s5s10s", "USOSFR5"),
        ),
    ),
    RegressionScreen(
        name="z35",
        target="30s35s",
        feature_sets=(
            ("10s30s",),
        ),
    ),
    RegressionScreen(
        name="z40",
        target="30s40s",
        feature_sets=(
            ("10s30s",),
        ),
    ),
    RegressionScreen(
        name="z50",
        target="30s50s",
        feature_sets=(
            ("10s30s",),
        ),
    ),
]

# Flat list of baseline specs (first feature set of each screen) — helpers / notebooks
REGRESSIONS: List[RegressionSpec] = [
    RegressionSpec(name=s.name, target=s.target, features=s.feature_sets[0])
    for s in REGRESSION_SCREENS
    if s.feature_sets
]

# Locked models for prediction + backtest (edit after run_pipeline chooses winners).
# No feature search — features here are final.
CHOSEN_MODELS: List[RegressionSpec] = [
    RegressionSpec(name="x7", target="5s7s10s", features=("2s5s10s",)),
    RegressionSpec(name="x20", target="10s20s30s", features=("10s30s",)),
    RegressionSpec(name="y15", target="10s15s30s", features=("5s10s30s",)),
    RegressionSpec(name="y25", target="20s25s30s", features=("10s30s",)),
    RegressionSpec(name="y4", target="3s4s5s", features=("2s5s10s",)),
    RegressionSpec(name="z35", target="30s35s", features=("10s30s",)),
    RegressionSpec(name="z40", target="30s40s", features=("10s30s",)),
    RegressionSpec(name="z50", target="30s50s", features=("10s30s",)),
]

# ── Z-score models ────────────────────────────────────────────────────────────
# Rolling z-score of a forward fly vs recent history → high / low / mid
# z12: TBD (not wired yet)
@dataclass(frozen=True)
class ZScoreSpec:
    """Screen a structure via rolling z-score."""
    series: str                     # column in full_data, e.g. '4y1y_5y1y_6y1y'
    param: str = "z6"               # model id: z6, z8, z9, …
    name: str = ""                  # defaults to "{param}:{series}"

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", f"{self.param}:{self.series}")


ZSCORE_MODELS: List[ZScoreSpec] = [
    # body of the fly = the "anchor" year in the name
    ZScoreSpec(series="4y1y_5y1y_6y1y", param="z6"),   # around 5y1y
    ZScoreSpec(series="6y1y_7y1y_8y1y", param="z8"),   # around 7y1y
    ZScoreSpec(series="7y1y_8y1y_9y1y", param="z9"),   # around 8y1y
    # z12 — specify later
]

# Z-score lookbacks shown in the last-N print / summary (past bars only)
ZSCORE_LOOKBACKS: List[int] = [150, 100, 50, 10]
ZSCORE_WINDOW = 50                  # default / fair-value window (for predicted)

# ── Walk-forward fitting ──────────────────────────────────────────────────────
# 1-step rolling: fit on the last TRAIN_SIZE (=k) bars → predict the next bar →
# slide by 1. k is FIXED. Explore mode (run_pipeline) compares feature_sets;
# prediction/backtest use CHOSEN_MODELS only.
TRAIN_SIZE = 160                    # fixed k
TEST_SIZE = 1                       # predict one bar ahead
STEP_SIZE = 1                       # then roll the window by one bar
ADD_INTERCEPT = True
MIN_TRAIN_OBS = 30                  # skip fit if too few clean rows after NaN drop

DATA_BAR_MINUTES = 15               # next-bar horizon (dataset cadence)
PREDICTION_EMAIL_EVERY_HOURS = 3    # ZZZ: email cadence for run_prediction
BACKTEST_START = "2024-01-01"       # inclusive
BACKTEST_END = None                 # None = through last available bar

# Residual mean-reversion strategy (run_backtest.py)
# residual r = y_true − y_pred (bp). z = (r − μ) / σ over past BACKTEST_Z_LOOKBACK bars only.
# Enter long  if z <= −BACKTEST_ZSCORE_THR and |r| >= BACKTEST_RESIDUAL_THR_BP
# Enter short if z >= +BACKTEST_ZSCORE_THR and |r| >= BACKTEST_RESIDUAL_THR_BP
# Else flat. PnL bar t = position[t-1] * (y[t] − y[t-1]).
#
# BACKTEST_POSITION_MODE:
#   "one_bar" (default) — each signal bar is a 1-period trade (close next bar); |pos|∈{0,1};
#                         n_trades = n_long + n_short (= # signal bars).
#   "hold"              — stay in the position while the signal persists (current behaviour);
#                         n_entries = # times we enter/change into a non-flat position.
BACKTEST_RESIDUAL_THR_BP = 0.5      # min |residual| in bp to take a trade
BACKTEST_ZSCORE_THR = 1.0           # |z(residual)| entry threshold
BACKTEST_Z_LOOKBACK = 50            # past bars for residual μ/σ (no look-ahead)
BACKTEST_POSITION_MODE = "one_bar"  # "one_bar" | "hold"

# ── Optional BBG ingest only (run_live.py) — not used by prediction ───────────
DATA_UPDATE_HOURS = 1                 # pull frequency if using run_live
LIVE_STORE_PATH = "sofr_live.csv"     # stacked panel on disk (same folder)
MAX_HISTORY_BARS = 2000               # rolling window length after each append
LIVE_TUNE_FEATURES = False            # explore uses run_pipeline; live ingest stays dumb
LIVE_TUNE_WINDOW = LIVE_TUNE_FEATURES

# Email (override via env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
#        REPORT_TO, REPORT_FROM)
SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASSWORD = ""
REPORT_FROM = ""
REPORT_TO: List[str] = []             # e.g. ["desk@firm.com"]
EMAIL_ENABLED = False                 # set True once SMTP + REPORT_TO are filled
