"""Shared fixtures.

Tests that need the real 313 MB database are marked ``integration`` and skip
cleanly when it is absent, so a fresh clone can run the unit suite immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_forecast.config import SQLITE_PATH


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires the downloaded SQLite database")


@pytest.fixture(scope="session")
def has_database() -> bool:
    return SQLITE_PATH.exists()


@pytest.fixture(scope="session")
def real_matches(has_database: bool) -> pd.DataFrame:
    if not has_database:
        pytest.skip("database.sqlite not present; run `football-forecast fetch`")
    from football_forecast.data.loader import load_matches

    return load_matches()


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    """A small deterministic league where the right answer is known.

    Four teams, home-and-away over two seasons. Team 1 is built to be strongest
    and team 4 weakest, so ratings and strengths have a known ordering to assert
    against rather than merely "did not crash".

    The generating strengths are spaced widely on purpose. With 72 matches per
    team, adjacent Poisson scoring rates of 1.0 and 0.6 overlap enough that
    sampling noise reorders the bottom two, which would make the ordering test
    flaky rather than wrong. The gaps here are large enough that a correct
    implementation recovers the ranking every time.
    """
    rng = np.random.default_rng(0)
    teams = [1, 2, 3, 4]
    strength = {1: 2.4, 2: 1.6, 3: 1.0, 4: 0.35}
    rows, day = [], pd.Timestamp("2010-08-01")
    idx = 0
    for season in ("2010/2011", "2011/2012"):
        for _round in range(6):
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    rows.append(
                        {
                            "match_index": idx,
                            "match_api_id": 1000 + idx,
                            "date": day,
                            "season": season,
                            "stage": _round + 1,
                            "league_id": 1,
                            "league": "Test League",
                            "country_id": 1,
                            "home_team_api_id": home,
                            "away_team_api_id": away,
                            "home_team": f"Team {home}",
                            "away_team": f"Team {away}",
                            "home_team_goal": int(rng.poisson(strength[home] * 1.3)),
                            "away_team_goal": int(rng.poisson(strength[away])),
                        }
                    )
                    idx += 1
                    day += pd.Timedelta(days=1)
    frame = pd.DataFrame(rows)
    frame["result"] = np.select(
        [frame.home_team_goal > frame.away_team_goal, frame.home_team_goal == frame.away_team_goal],
        [0, 1],
        default=2,
    ).astype("int8")
    frame["goal_diff"] = frame.home_team_goal - frame.away_team_goal
    frame["total_goals"] = frame.home_team_goal + frame.away_team_goal
    return frame
