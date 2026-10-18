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
