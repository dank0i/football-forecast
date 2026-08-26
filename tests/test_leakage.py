"""Tests that assert the pipeline cannot see the future.

These are the tests that matter most in this project. A leak does not raise, it
does not fail a type check, and it does not look wrong in a notebook: it just
quietly makes every number better. The only defence is to state the invariant
and check it mechanically.

The strategy throughout is perturbation: change something that happens *after* a
match and assert the features for that match do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchcast.features.elo import compute_elo
from pitchcast.features.form import compute_form


def test_elo_is_unchanged_by_later_results(synthetic_matches: pd.DataFrame):
    """Rewriting the last 20 scorelines must not alter any earlier rating."""
    baseline = compute_elo(synthetic_matches)

    tampered = synthetic_matches.copy()
    tail = tampered.index[-20:]
    tampered.loc[tail, "home_team_goal"] = 9
    tampered.loc[tail, "away_team_goal"] = 0
    tampered.loc[tail, "goal_diff"] = 9
    perturbed = compute_elo(tampered)

    cutoff = len(tampered) - 20
    pd.testing.assert_frame_equal(
        baseline.iloc[:cutoff], perturbed.iloc[:cutoff], check_exact=False, rtol=1e-12
    )


def test_elo_of_first_match_is_the_initial_rating(synthetic_matches: pd.DataFrame):
    elo = compute_elo(synthetic_matches)
    assert elo["elo_home"].iloc[0] == pytest.approx(1500.0)
    assert elo["elo_away"].iloc[0] == pytest.approx(1500.0)
    assert elo["elo_matches_home"].iloc[0] == 0


def _final_ratings(matches: pd.DataFrame) -> dict[int, float]:
    """Each team's rating after its last appearance, home or away.

    Grouping on ``elo_home`` alone would sample only a team's home fixtures and
    miss whichever came later, so both sides are considered.
    """
    frame = matches.merge(compute_elo(matches), on="match_index")
    ratings: dict[int, float] = {}
    for team in set(frame.home_team_api_id) | set(frame.away_team_api_id):
        played = frame[(frame.home_team_api_id == team) | (frame.away_team_api_id == team)].sort_values(
            "match_index"
        )
        last = played.iloc[-1]
        ratings[team] = last.elo_home if last.home_team_api_id == team else last.elo_away
    return ratings


def test_elo_ranks_a_known_ordering(synthetic_matches: pd.DataFrame):
    """Team 1 is generated strongest and team 4 weakest; Elo must recover that."""
    ratings = _final_ratings(synthetic_matches)
    ordering = sorted(ratings, key=ratings.get, reverse=True)
    assert ordering == [1, 2, 3, 4]


def test_elo_is_zero_sum(synthetic_matches: pd.DataFrame):
    """Rating is transferred between teams, never created.

    Every update adds delta to one side and subtracts it from the other, so the
    pool mean must stay at the initial rating. This catches sign errors and
    asymmetric updates that a ranking test would not.
    """
    ratings = _final_ratings(synthetic_matches)
    assert float(np.mean(list(ratings.values()))) == pytest.approx(1500.0, abs=25.0)


def test_elo_rejects_unsorted_input(synthetic_matches: pd.DataFrame):
    shuffled = synthetic_matches.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        compute_elo(shuffled)


def test_form_excludes_the_current_match(synthetic_matches: pd.DataFrame):
    """A team's own result must not appear in its own pre-match form."""
    baseline = compute_form(synthetic_matches)

    tampered = synthetic_matches.copy()
    target = 40
    tampered.loc[target, "home_team_goal"] = 8
    tampered.loc[target, "away_team_goal"] = 0
    perturbed = compute_form(tampered)

    row_before = baseline[baseline.match_index == target].reset_index(drop=True)
    row_after = perturbed[perturbed.match_index == target].reset_index(drop=True)
    pd.testing.assert_frame_equal(row_before, row_after, check_exact=False, rtol=1e-12)


def test_first_match_of_a_team_has_no_form(synthetic_matches: pd.DataFrame):
    form = compute_form(synthetic_matches)
    first = form.sort_values("match_index").iloc[0]
    assert np.isnan(first["home_form5_points"])
    assert np.isnan(first["away_form5_points"])


def test_form_windows_use_only_prior_matches(synthetic_matches: pd.DataFrame):
    """Recompute one team's rolling points by hand and compare."""
    form = compute_form(synthetic_matches)
    team = 2
    played = synthetic_matches[
        (synthetic_matches.home_team_api_id == team) | (synthetic_matches.away_team_api_id == team)
    ].sort_values("date")

    points = []
    for row in played.itertuples():
        if row.home_team_api_id == team:
            diff = row.home_team_goal - row.away_team_goal
        else:
            diff = row.away_team_goal - row.home_team_goal
        points.append(3.0 if diff > 0 else (1.0 if diff == 0 else 0.0))

    target_pos = 8
    expected = float(np.mean(points[max(0, target_pos - 3) : target_pos]))
    match_index = played.iloc[target_pos].match_index
    row = form[form.match_index == match_index].iloc[0]
    side = "home" if played.iloc[target_pos].home_team_api_id == team else "away"
    assert row[f"{side}_form3_points"] == pytest.approx(expected)


def test_dixon_coles_ignores_matches_at_or_after_as_of(synthetic_matches: pd.DataFrame):
    """Changing matches on or after the as-of date must not move the fit."""
    from pitchcast.models.dixon_coles import fit_dixon_coles

    as_of = synthetic_matches["date"].iloc[60]
    baseline = fit_dixon_coles(synthetic_matches, as_of=as_of)

    tampered = synthetic_matches.copy()
    mask = tampered["date"] >= as_of
    tampered.loc[mask, "home_team_goal"] = 7
    tampered.loc[mask, "away_team_goal"] = 0
    perturbed = fit_dixon_coles(tampered, as_of=as_of)

    assert baseline.n_matches == perturbed.n_matches
    np.testing.assert_allclose(baseline.attack, perturbed.attack, atol=1e-8)
    np.testing.assert_allclose(baseline.defence, perturbed.defence, atol=1e-8)


@pytest.mark.integration
def test_squad_ratings_predate_kickoff(real_matches: pd.DataFrame):
    """No player rating used by a match may be dated on or after that match.

    Exercised against the real database because the as-of join is where a
    forward-reaching merge would hide, and the synthetic fixture has no players.
    """
    from pitchcast.data.loader import load_player_attributes
    from pitchcast.features.squad import _lineup_long

    sample = real_matches.iloc[::37].copy()
    long = _lineup_long(sample)
    long["player_api_id"] = long["player_api_id"].astype("int64")
    attrs = load_player_attributes().sort_values("date", kind="mergesort")

    joined = pd.merge_asof(
        long.sort_values("date", kind="mergesort"),
        attrs[["player_api_id", "date", "overall_rating"]].rename(columns={"date": "rating_date"}),
        left_on="date",
        right_on="rating_date",
        by="player_api_id",
        direction="backward",
        allow_exact_matches=False,
    )
    matched = joined.dropna(subset=["rating_date"])
    assert len(matched) > 0
    assert (matched["rating_date"] < matched["date"]).all()
