"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SQLITE_PATH = DATA_RAW / "database.sqlite"
SQLITE_SHA256 = "4df8569777d59fdd690754b1cc8ca1f7989baf65f2eaddd0f1368285f11139a9"
SQLITE_URL = "https://huggingface.co/datasets/julien-c/kaggle-hugomathien-soccer/resolve/main/database.sqlite"

# Outcome encoding, from the home team's perspective. The order matters:
# RPS treats the classes as ordinal, and home-draw-away is the natural ordering.
HOME, DRAW, AWAY = 0, 1, 2
OUTCOMES = ("H", "D", "A")

# Bookmakers present in the Match table. Coverage is the fraction of matches with
# a complete H/D/A triple; anything below ~85% is too sparse to use as a primary
# reference price, so the consensus is built from the six dense books only.
BOOKMAKERS = {
    "B365": 0.870,
    "BW": 0.869,
    "IW": 0.867,
    "LB": 0.868,
    "WH": 0.869,
    "VC": 0.869,
    "SJ": 0.658,
    "GB": 0.545,
    "BS": 0.545,
    "PS": 0.430,
}
DENSE_BOOKMAKERS = tuple(b for b, cov in BOOKMAKERS.items() if cov >= 0.85)
REFERENCE_BOOKMAKER = "B365"

SEASONS = (
    "2008/2009",
    "2009/2010",
    "2010/2011",
    "2011/2012",
    "2012/2013",
    "2013/2014",
    "2014/2015",
    "2015/2016",
)
# The first two seasons are burn-in: Elo ratings and rolling form are undefined
# until teams have a match history, so they train but are never scored.
BURN_IN_SEASONS = 2
TEST_SEASONS = SEASONS[BURN_IN_SEASONS:]
