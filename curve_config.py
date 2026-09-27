# =============================================================================
# curve_config.py  — SOFR curve screener: structures & model params
# =============================================================================
# Raw columns are SOFR swap tenors (years). Structures are built as:
#   curve  XsYs   = Y - X
#   fly    XsYsZs = 2*Y - X - Z
# =============================================================================

from dataclasses import dataclass
from typing import List, Tuple

# ── Data paths (DataZone: point these at your uploaded file) ──────────────────
DATA_PATH = "sofr_3h.xlsx"          # .xlsx or .csv
TIMESTAMP_COL = "time"              # matches Bloomberg export screenshot
SHEET_NAME = 0

# Bloomberg ticker pattern: "USOSFR{N} BGN Curncy" → tenor N (years)
BBG_PREFIX = "USOSFR"
BBG_SUFFIX = "BGN Curncy"

# ── Structure definitions ─────────────────────────────────────────────────────
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
]

FLIES: List[FlySpec] = [
    ("2s5s10s", 2, 5, 10),
    ("5s7s10s", 5, 7, 10),
    ("5s10s30s", 5, 10, 30),
    ("10s20s30s", 10, 20, 30),
    ("10s30s35s", 10, 30, 35),
]

# Targets we want to screen / predict (must appear in CURVES or FLIES)
PREDICTED: List[str] = [
    "30s35s",
]

# ── Regression specs (expand this list as you add more points) ────────────────
@dataclass(frozen=True)
class RegressionSpec:
    """One prediction problem: target ~ features (+ intercept)."""
    target: str
    features: Tuple[str, ...]
    name: str = ""                  # optional label; defaults to "target~f1+f2"

    def __post_init__(self):
        if not self.name:
            object.__setattr__(
                self, "name", f"{self.target}~{'+'.join(self.features)}"
            )


REGRESSIONS: List[RegressionSpec] = [
    RegressionSpec(target="30s35s", features=("10s30s",)),
]

# ── Walk-forward fitting ──────────────────────────────────────────────────────
# 3h bars: TRAIN_SIZE=160 ≈ 20 trading days; TEST_SIZE=8 ≈ 1 trading day
TRAIN_SIZE = 160
TEST_SIZE = 8
# How many bars to advance the window after each test block
STEP_SIZE = TEST_SIZE               # non-overlapping blocks; set 1 for denser preds
ADD_INTERCEPT = True
MIN_TRAIN_OBS = 30                  # skip fit if too few clean rows after NaN drop
