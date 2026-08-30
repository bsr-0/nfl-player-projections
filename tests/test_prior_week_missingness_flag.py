"""Week-1 rows have no prior week, and that absence must not become a zero.

`get_all_players_for_training` joins team and opponent stats on
`ts.week = pws.week - 1` so the current week's outcome cannot leak in. There is
no week 0, so week 1 always yields NULL -- and the blanket numeric fill in
`calculate_utilization_scores` turned that into a measured 0 for 100% of week-1
rows.

Measured 2026-08-29, before the fix:

  - `opp_fpts_allowed`, a DIRECT causal feature for all four positions with a
    mid-season median of 21.6, read 0.0 on every week-1 row. That asserts the
    opponent allows zero fantasy points -- the best defence possible, identical
    for all 32 teams -- in the week where form data is weakest.
  - The zero propagated through 3-game rolling means: 25-33% of
    `team_pace_sec_per_play_roll3_mean` rows in weeks 2-4 fell below 20 s/play
    against a true 1st percentile of 24.75, which is physically impossible.

This is why backfilling 19 seasons of team_stats from play-by-play produced no
accuracy change: the database was corrected and the fabrication was
reintroduced downstream.
"""
import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from config.settings import DB_PATH
from src.utils.database import DatabaseManager
from src.features.utilization_score import (
    MISSINGNESS_PRESERVED_COLS,
    PRIOR_WEEK_JOIN_COLS,
    UtilizationScoreCalculator,
    missingness_preserved_cols,
)


@pytest.fixture(autouse=True)
def _isolate_other_flags(monkeypatch):
    """Hold the sibling flags fixed; this module tests one flag."""
    monkeypatch.setattr(settings, "PRESERVE_HISTORY_MISSINGNESS", False)
    monkeypatch.setattr(settings, "PRESERVE_PERSONNEL_MISSINGNESS", False)


@pytest.fixture
def flag(monkeypatch):
    def _set(value):
        monkeypatch.setattr(settings, "PRESERVE_PRIOR_WEEK_MISSINGNESS", value)
    return _set


def test_default_is_on():
    """Unlike the personnel flag, this one ships enabled.

    'Allows 0.0 fantasy points' has no defensible reading as a measurement,
    and the results a default-OFF would stay comparable with were already
    superseded by the same day's label, horizon and backfill fixes.
    """
    assert settings.PRESERVE_PRIOR_WEEK_MISSINGNESS is True


def test_on_exempts_every_prior_week_column(flag):
    flag(True)
    preserved = missingness_preserved_cols()
    for col in PRIOR_WEEK_JOIN_COLS:
        assert col in preserved, f"{col} still exposed to the blanket fill"


def test_off_restores_previous_behaviour(flag):
    flag(False)
    preserved = missingness_preserved_cols()
    assert preserved == MISSINGNESS_PRESERVED_COLS
    for col in PRIOR_WEEK_JOIN_COLS:
        assert col not in preserved


def test_opp_fpts_allowed_is_covered():
    """The single worst case: a direct feature for all four positions."""
    assert "opp_fpts_allowed" in PRIOR_WEEK_JOIN_COLS


def test_opp_fpts_allowed_source_columns_are_covered():
    """Exempting the assembled name alone was not enough.

    `opp_fpts_allowed` is built in feature_engineering from
    `fantasy_points_allowed_{pos}`, so protecting only the output left the
    inputs to be zero-filled and the output stayed 0 on 100% of week-1 rows.
    A column can feed a causal feature under a different name; membership
    checks by name miss that.
    """
    for pos in ("qb", "rb", "wr", "te"):
        assert f"fantasy_points_allowed_{pos}" in PRIOR_WEEK_JOIN_COLS


@pytest.mark.skipif(not DB_PATH.exists(), reason="needs the local database")
def test_week1_join_yields_null_not_a_week_zero_row():
    """team_stats carries week-0 placeholders; week 1 must not join to them.

    186 such rows exist (31 per season, 2020-2025), all with total_plays = 0.
    `ts.week = pws.week - 1` resolves to 0 for week-1 rows, so without the
    `ts.week >= 1` guard they matched a fabricated record and came out of SQL
    as a measured zero -- 73% of week-1 RB rows before the fix.
    """
    import sqlite3

    con = sqlite3.connect(DB_PATH)
    try:
        n_week0 = con.execute(
            "SELECT COUNT(*) FROM team_stats WHERE week = 0").fetchone()[0]
    finally:
        con.close()
    if not n_week0:
        pytest.skip("no week-0 rows present; guard is unobservable here")

    df = DatabaseManager().get_all_players_for_training(
        position="RB", min_games=4)
    w1 = df[(df.week == 1) & df.season.between(2018, 2025)]
    plays = pd.to_numeric(w1["team_plays"], errors="coerce")

    assert (plays == 0).sum() == 0, (
        f"{(plays == 0).sum()} week-1 rows carry team_plays == 0; the week-0 "
        f"placeholder join is back"
    )
    assert plays.isna().all(), "week 1 has no prior week; team_plays must be NULL"


_STAT_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_tds", "air_yards",
    "rushing_attempts", "rushing_yards", "rushing_tds", "passing_attempts",
    "passing_yards", "passing_tds", "snap_count", "team_snaps",
    "redzone_targets", "redzone_carries", "team_targets", "team_carries",
    "team_pass_attempts", "team_rush_attempts", "team_redzone_attempts",
    "rush_inside_10", "targets_15_plus", "goal_line_carries",
]


def _week1_frame() -> pd.DataFrame:
    """Week 1 rows as the prior-week join leaves them: NULL, not zero.

    Ordinary stat columns are populated so the position calculators run; only
    the prior-week join columns are NaN, which is the real week-1 shape.
    """
    df = pd.DataFrame({
        "player_id": ["p1", "p2"],
        "position": ["WR", "RB"],
        "season": [2024, 2024],
        "week": [1, 1],
        "team": ["KC", "SF"],
        "fantasy_points": [12.0, 8.0],
        "opp_fpts_allowed": [np.nan, np.nan],
        "team_pace_sec_per_play": [np.nan, np.nan],
        "team_plays": [np.nan, np.nan],
    })
    for c in _STAT_COLS:
        df[c] = 5.0
    return df


def test_week1_nan_survives_the_blanket_fill(flag):
    flag(True)
    out = UtilizationScoreCalculator(weights=None).calculate_all_scores(
        _week1_frame(), pd.DataFrame())

    for col in ("opp_fpts_allowed", "team_pace_sec_per_play", "team_plays"):
        if col not in out.columns:
            continue
        assert out[col].isna().all(), (
            f"{col} was filled on week-1 rows; a zero here is a measurement "
            f"the prior-week join never made"
        )


def test_week1_zero_fill_is_what_we_prevented(flag):
    """Non-vacuous: with the flag off, the zeros come back."""
    flag(False)
    out = UtilizationScoreCalculator(weights=None).calculate_all_scores(
        _week1_frame(), pd.DataFrame())

    filled = [c for c in ("opp_fpts_allowed", "team_pace_sec_per_play")
              if c in out.columns and (out[c] == 0).any()]
    assert filled, (
        "expected the blanket fill to zero these with the flag off; if this "
        "fails the fill moved and this test no longer proves anything"
    )
