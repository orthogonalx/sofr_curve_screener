# SOFR curve screener — structures & model params
# Swap CURVES/FLIES: BBG USOS#### / UF######. Fwd curves/flies: built from S0490FS.
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

DATA_PATH = "sofr_3h.xlsx"

BBG_PREFIX = "USOSFR"
BBG_SUFFIX = "BGN Curncy"
BBG_FWD_PREFIX = "S0490FS"
BBG_FWD_SUFFIX = "BLC Curncy"

BBG_EVENT_TYPE = "TRADE"
BBG_INTERVAL_MINUTES = 180
MAX_HISTORY_DAYS = 100
BBG_START_DATE = (datetime.now() - timedelta(days=MAX_HISTORY_DAYS)).strftime("%Y%m%d")
BBG_END_DATE = ""
BBGAPI_VERSION = "2022.09.26-2"
BBGAPI_PATH = ""

FwdPoint = Tuple[int, int]


def fwd_name(start: int, gap: int) -> str:
    return f"{start}y{gap}y"


FORWARDS: List[FwdPoint] = (
    [(s, 1) for s in range(1, 10)]
    + [(s, 2) for s in range(1, 11)]
    + [(s, 5) for s in (5, 10, 15, 20, 25, 30, 35)]
)

FWD_GROUPS: Dict[str, List[str]] = {
    "1y": [fwd_name(s, 1) for s in range(1, 10)],
    "2y": [fwd_name(s, 2) for s in range(1, 11)],
    "5y": [fwd_name(s, 5) for s in (5, 10, 15, 20, 25, 30, 35)],
}

FwdCurveSpec = Tuple[str, str, str]
FwdFlySpec = Tuple[str, str, str, str]


def _expand_fwd_group_structures(groups):
    curves, flies = [], []
    for cols in groups.values():
        for i in range(len(cols) - 1):
            a, b = cols[i], cols[i + 1]
            curves.append((f"{a}_{b}", a, b))
        for i in range(len(cols) - 2):
            a, b, c = cols[i], cols[i + 1], cols[i + 2]
            flies.append((f"{a}_{b}_{c}", a, b, c))
    return curves, flies


FWD_CURVES, FWD_FLIES = _expand_fwd_group_structures(FWD_GROUPS)

# Swap CURVES/FLIES from BBG (USOS/UF). Fwd structures always computed.
USE_BBG_SWAP_STRUCTURES = True
CurveSpec = Tuple[str, int, int]
FlySpec = Tuple[str, int, int, int]

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
]

PREDICTED: List[str] = [
    "5s7s10s", "10s20s30s", "10s15s30s", "20s25s30s",
    "3s4s5s", "30s35s", "30s40s", "30s50s",
]


@dataclass(frozen=True)
class RegressionSpec:
    target: str
    features: Tuple[str, ...]
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(
                self, "name", f"{self.target}~{'+'.join(self.features)}"
            )


@dataclass(frozen=True)
class RegressionScreen:
    name: str
    target: str
    feature_sets: Tuple[Tuple[str, ...], ...]


REGRESSION_SCREENS: List[RegressionScreen] = [
    RegressionScreen("x7", "5s7s10s", (("2s5s10s",), ("2s5s10s", "USOSFR5"))),
    RegressionScreen("x20", "10s20s30s", (("10s30s",), ("10s30s", "USOSFR5"))),
    RegressionScreen("y15", "10s15s30s", (("5s10s30s",), ("5s10s30s", "USOSFR5"))),
    RegressionScreen("y25", "20s25s30s", (("10s30s",), ("10s30s", "USOSFR5"))),
    RegressionScreen("y4", "3s4s5s", (("2s5s10s",), ("2s5s10s", "USOSFR5"))),
    RegressionScreen("z35", "30s35s", (("10s30s",),)),
    RegressionScreen("z40", "30s40s", (("10s30s",),)),
    RegressionScreen("z50", "30s50s", (("10s30s",),)),
]

CHOSEN_MODELS: List[RegressionSpec] = [
    RegressionSpec("5s7s10s", ("2s5s10s",), "x7"),
    RegressionSpec("10s20s30s", ("10s30s",), "x20"),
    RegressionSpec("10s15s30s", ("5s10s30s",), "y15"),
    RegressionSpec("20s25s30s", ("10s30s",), "y25"),
    RegressionSpec("3s4s5s", ("2s5s10s",), "y4"),
    RegressionSpec("30s35s", ("10s30s",), "z35"),
    RegressionSpec("30s40s", ("10s30s",), "z40"),
    RegressionSpec("30s50s", ("10s30s",), "z50"),
]


@dataclass(frozen=True)
class ZScoreSpec:
    series: str
    param: str = "z6"
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", f"{self.param}:{self.series}")


ZSCORE_MODELS: List[ZScoreSpec] = [
    ZScoreSpec("4y1y_5y1y_6y1y", "z6"),
    ZScoreSpec("6y1y_7y1y_8y1y", "z8"),
    ZScoreSpec("7y1y_8y1y_9y1y", "z9"),
]

ZSCORE_LOOKBACKS: List[int] = [150, 100, 50, 10]
ZSCORE_WINDOW = 50

TRAIN_SIZE = 160
TEST_SIZE = 1
STEP_SIZE = 1
ADD_INTERCEPT = True
MIN_TRAIN_OBS = 30

DATA_BAR_MINUTES = BBG_INTERVAL_MINUTES
PREDICTION_EVERY_MINUTES = 30
PREDICTION_EMAIL_EVERY_HOURS = PREDICTION_EVERY_MINUTES / 60.0
REFRESH_DATA_ON_REQUEST = True

GRAPH_LOOKBACK_DAYS = 9
GRAPH_DIR_EVOLUTION = "graphs/evolution"
GRAPH_DIR_BACKTEST_PNL = "graphs/backtest_pnl"
GRAPH_DIR_BACKTEST_VARS = "graphs/backtest_vars"
GRAPH_DIR_CORR = "graphs/correlations"
BACKTEST_START = "2024-01-01"
BACKTEST_END = None
BACKTEST_RESIDUAL_THR_BP = 0.5
BACKTEST_ZSCORE_THR = 1.0
BACKTEST_Z_LOOKBACK = 50
BACKTEST_POSITION_MODE = "one_bar"

LIVE_STORE_PATH = "sofr_live.csv"
STRUCTURES_STORE_PATH = "sofr_structures.csv"
CROSSCHECK_PATH = "sofr_structures_crosscheck.xlsx"
BBG_CURVE_TICKER_PREFIX = "USOS"
BBG_FLY_TICKER_PREFIX = "UF"
BBG_STRUCT_SUFFIX = "Curncy"
CROSSCHECK_TOL_BP = 1e-4
MAX_HISTORY_BARS = MAX_HISTORY_DAYS * (24 * 60 // BBG_INTERVAL_MINUTES) + 50

US_HOLIDAY_START = "2018-01-01"
US_HOLIDAY_END = "2035-12-31"


def us_holiday_dates(start=None, end=None):
    import pandas as pd
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, GoodFriday, USFederalHolidayCalendar,
    )

    start = pd.Timestamp(start or US_HOLIDAY_START)
    end = pd.Timestamp(end or US_HOLIDAY_END)

    class _USRatesHolidayCalendar(AbstractHolidayCalendar):
        rules = list(USFederalHolidayCalendar.rules) + [GoodFriday]

    hol = _USRatesHolidayCalendar().holidays(start=start, end=end)
    return pd.DatetimeIndex(hol).tz_localize(None).normalize().unique().sort_values()


SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASSWORD = ""
SMTP_USE_TLS = True
REPORT_FROM = ""
REPORT_TO: List[str] = []
EMAIL_ENABLED = False
EMAIL_INLINE_GRAPHS = True
SEND_EMAIL_ON_PREDICTION = True

# ── Extra BBG series (pulled into live store; use clean name everywhere) ───────
# Add any ticker not already in USOSFR / S0490FS / USOS / UF, e.g. futures, FX, equity.
# (clean_name, bbg_ticker)
CROSS_ASSET: List[Tuple[str, str]] = [
    ("SPX", "SPX Index"),
    ("CO1", "CO1 Comdty"),
    ("DXY", "DXY Curncy"),
    ("USGG5YR", "USGG5YR Index"),   # US 5y rates — corr scatter
    ("USGG10YR", "USGG10YR Index"),
    ("ESA", "ESA Index"),           # Euro Stoxx futures — corr scatter
    ("SFRH7", "SFRH7 Comdty"),
    ("USSFCT05", "USSFCT05 Curncy"),  # SOFR vs Fed Funds 5y
    ("USSFCT10", "USSFCT10 Curncy"),
    ("USSFCT30", "USSFCT30 Curncy"),
]

# ── Correlation notebook: EDIT THIS LIST to choose matrix columns ─────────────
# Names must exist in the panel: CURVES/FLIES/CUSTOM_* clean names, or CROSS_ASSET.
# Example: add "2s5s", "5s10s", "2s10s", "SFRH7", …
CORR_SERIES: List[str] = [
    "5s7s10s", "10s20s30s", "10s15s30s", "3s4s5s", "10s30s",
    "SPX", "CO1", "DXY", "USGG10YR",
]
CORR_WINDOW = 50          # rolling window for pairwise correlation
CORR_PCT_LOOKBACK = 200   # history of rolling corrs for percentile rank (100=highest)
CORR_Z_LOOKBACK = CORR_PCT_LOOKBACK  # legacy alias

# Scatter: x = ρ(rates, asset_x), y = ρ(rates, asset_y); highlight last N calendar days
CORR_SCATTER_RATES = "USGG5YR"   # fallback: "5" (USOSFR5) if missing
CORR_SCATTER_X = "CO1"           # x-axis: corr(rates, CO1)
CORR_SCATTER_Y = "ESA"           # y-axis: corr(rates, ESA)
CORR_SCATTER_HIGHLIGHT_DAYS = 4  # last week ≈ 4 trading days

# ── Custom structures beyond the main CURVES/FLIES lists ──────────────────────
# Same BBG mapping: 2s10s → USOS0210 Curncy. Merged into the live pull + panel.
CUSTOM_CURVES: List[CurveSpec] = [
    ("2s10s", 2, 10),
]
CUSTOM_FLIES: List[FlySpec] = [
    # ("2s5s7s", 2, 5, 7),
]

# ── Custom prediction (run_custom_prediction / run_custom.ipynb) ───────────────
# Same OLS next-bar forecast as prediction, but YOUR specs — not x7/x20/…
# Target / features = clean names (structures or CROSS_ASSET). If missing in DB,
# add to CUSTOM_CURVES / CUSTOM_FLIES / CROSS_ASSET and --pull-history.
CUSTOM_MODELS: List[RegressionSpec] = [
    RegressionSpec(target="2s10s", features=("SFRH7",), name="c_2s10s_sfrh7"),
]

# ── Period move tables (run_period_moves.ipynb) ────────────────────────────────
# Outright tenors (clean names = years as strings from USOSFR)
MOVE_OUTRIGHTS: List[str] = ["2", "5", "10", "30"]
# Curves / flies / spreads (structures + CROSS_ASSET clean names)
MOVE_STRUCTURES: List[str] = [
    "2s5s", "5s10s", "10s30s", "2s5s10s", "5s10s30s",
    "USSFCT05", "USSFCT10", "USSFCT30",
]
