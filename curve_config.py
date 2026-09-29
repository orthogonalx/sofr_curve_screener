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

# ── Swap tenor structure definitions ──────────────────────────────────────────
# Curves: (name, short_tenor, long_tenor)  →  long - short
CurveSpec = Tuple[str, int, int]
# Flies:  (name, left, body, right)        →  2*body - left - right
FlySpec = Tuple[str, int, int, int]

CURVES: List[CurveSpec] = [
    ("2s5s", 2, 5),
    ("5s10s", 5, 10),
    ("10s20s", 10, 20),
    ("10s30s", 10, 30),
    ("20s30s", 20, 30),
    ("30s35s", 30, 35),
    ("30s40s", 30, 40),
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
]

# Targets we want to screen / predict (must appear in CURVES or FLIES)
PREDICTED: List[str] = [
    "5s7s10s",
    "10s20s30s",
    "10s15s30s",
    "20s25s30s",
    "3s4s5s",
]

# ── Regression specs ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RegressionSpec:
    """One prediction problem: target ~ features (+ intercept)."""
    target: str
    features: Tuple[str, ...]
    name: str = ""                  # model id: x7, y15, … ; default target~feats

    def __post_init__(self):
        if not self.name:
            object.__setattr__(
                self, "name", f"{self.target}~{'+'.join(self.features)}"
            )


REGRESSIONS: List[RegressionSpec] = [
    RegressionSpec(name="x7", target="5s7s10s", features=("2s5s10s",)),
    RegressionSpec(name="x20", target="10s20s30s", features=("10s30s",)),
    RegressionSpec(name="y15", target="10s15s30s", features=("5s10s30s",)),
    RegressionSpec(name="y25", target="20s25s30s", features=("10s30s",)),
    RegressionSpec(name="y4", target="3s4s5s", features=("2s5s10s",)),
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
# 1-step rolling: fit on the last TRAIN_SIZE bars → predict the next bar →
# slide by 1. TRAIN_SIZE is chosen by comparing TRAIN_WINDOWS on OOS RMSE.
TRAIN_WINDOWS: List[int] = [50, 80, 100, 160, 200]
TRAIN_SIZE = 160                    # default / fallback if search is skipped
TEST_SIZE = 1                       # predict one bar ahead
STEP_SIZE = 1                       # then roll the window by one bar
ADD_INTERCEPT = True
MIN_TRAIN_OBS = 30                  # skip fit if too few clean rows after NaN drop

# ── Live ops: hourly BBG ingest + scheduled email report ──────────────────────
# New BBG print arrives every DATA_UPDATE_HOURS → append to LIVE_STORE_PATH.
# Keep only the last MAX_HISTORY_BARS rows (drop oldest) so history size is capped.
DATA_UPDATE_HOURS = 1                 # XXX: pull frequency
REPORT_EVERY_HOURS = 3                # YYY: email cadence
LIVE_STORE_PATH = "sofr_live.csv"     # stacked panel on disk (same folder)
MAX_HISTORY_BARS = 2000               # rolling window length after each append
LIVE_TUNE_WINDOW = False              # True = re-search TRAIN_WINDOWS each report (slower)

# Email (override via env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
#        REPORT_TO, REPORT_FROM)
SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASSWORD = ""
REPORT_FROM = ""
REPORT_TO: List[str] = []             # e.g. ["desk@firm.com"]
EMAIL_ENABLED = False                 # set True once SMTP + REPORT_TO are filled
