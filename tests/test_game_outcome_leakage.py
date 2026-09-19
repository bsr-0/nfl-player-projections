"""Leakage-safety invariants for the game-outcome feature builder.

home_score/away_score must never survive into the returned feature frame;
every returned feature column must be classified (blocked or documented as
pre-kickoff-known) by src/utils/leakage.py; tie games must be dropped, not
phantom-encoded; and team-form aggregates must be lagged to strictly before
the target game -- a sentinel value planted in one week's team_stats must
not appear in that week's feature row, only in later weeks'.
"""
import sqlite3

import pytest

from src.models.game_outcome.features import build_game_outcome_rows, feature_columns
from src.utils.database import DatabaseManager
from src.utils.leakage import audit_feature_availability

SEASON = 2020


def _seed_db(db: DatabaseManager, n_weeks: int, sentinel_week: int, sentinel_value: float = 999.0):
    for week in range(1, n_weeks + 1):
        db.insert_schedule(
            {
                "season": SEASON,
                "week": week,
                "home_team": "AAA",
                "away_team": "BBB",
                "spread_line": 3.0,
                "total_line": 45.0,
                "home_score": 24,
                "away_score": 20,
            }
        )
        for team, base in (("AAA", 20.0), ("BBB", 17.0)):
            # total_yards (not points_scored/points_allowed -- those are now
            # derived from schedule.home_score/away_score, see features.py's
            # module docstring on the team_stats points defect) carries the
            # sentinel here.
            yards = sentinel_value if (team == "AAA" and week == sentinel_week) else base + week
            db.insert_team_stats(
                {
                    "team": team,
                    "season": SEASON,
                    "week": week,
                    "opponent": "BBB" if team == "AAA" else "AAA",
                    "home_away": "home" if team == "AAA" else "away",
                    "points_scored": 20.0,
                    "points_allowed": 17.0,
                    "total_yards": yards,
                    "passing_yards": 220.0,
                    "rushing_yards": 130.0,
                    "turnovers": 1.0,
                    "third_down_conv": 0.4,
                    "drive_success_rate": 0.5,
                    "avg_drive_epa": 0.1,
                    "points_per_drive": 2.0,
                    "neutral_pass_rate_oe": 0.0,
                }
            )


def _add_tie_game(db: DatabaseManager, week: int):
    db.insert_schedule(
        {
            "season": SEASON,
            "week": week,
            "home_team": "CCC",
            "away_team": "DDD",
            "spread_line": 0.0,
            "total_line": 40.0,
            "home_score": 20,
            "away_score": 20,
        }
    )


@pytest.fixture
def synthetic_db(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_db(db, n_weeks=6, sentinel_week=3)
    _add_tie_game(db, week=1)
    return db


def _build(db: DatabaseManager, **kwargs):
    con = sqlite3.connect(str(db.db_path))
    try:
        return build_game_outcome_rows(seasons=[SEASON], con=con, **kwargs)
    finally:
        con.close()


def test_scores_never_survive_into_feature_frame(synthetic_db):
    rows = _build(synthetic_db)
    forbidden = {"home_score", "away_score"}
    assert forbidden.isdisjoint(rows.columns)


def test_all_feature_columns_are_classified(synthetic_db):
    rows = _build(synthetic_db)
    unclassified = audit_feature_availability(feature_columns(rows))
    assert unclassified == []


def test_tie_games_are_dropped_not_encoded(synthetic_db):
    rows = _build(synthetic_db)
    # The CCC/DDD tie at week 1 must not appear at all.
    assert not ((rows.home_team == "CCC") & (rows.away_team == "DDD")).any()
    assert set(rows["home_win"].unique()) <= {0, 1}


def test_sentinel_value_only_appears_in_later_weeks(synthetic_db):
    rows = _build(synthetic_db, include_weather=False)
    rows = rows.sort_values("week").set_index("week")

    # week 3 is when the sentinel is recorded for AAA -- the game AT week 3
    # must not reflect it (shift(1) excludes the current row).
    week3_s2d = rows.loc[3, "home_minus_away_total_yards_s2d"]
    assert week3_s2d != pytest.approx(999.0, rel=0.5)

    # week 4 has 3 prior games (weeks 1-3) for AAA, clearing min_prior_games_for_form,
    # so its season-to-date and roll3 means include week 3's sentinel.
    week4_s2d = rows.loc[4, "home_minus_away_total_yards_s2d"]
    week4_roll3 = rows.loc[4, "home_minus_away_total_yards_roll3"]
    assert week4_s2d > 300  # dominated by the 999 sentinel averaged with 2 normal values
    assert week4_roll3 > 300


def test_no_unclassified_columns_survive_audit_for_full_default_build(synthetic_db):
    rows = _build(synthetic_db, include_weather=True)
    assert "home_win" in rows.columns
    assert audit_feature_availability(feature_columns(rows)) == []


def test_numpy_int64_seasons_return_same_rows_as_plain_int(synthetic_db):
    """Regression test for a silent bug found 2026-09-18 while tuning Elo
    hyperparameters: passing numpy.int64 season values (e.g. from a
    DataFrame's `.unique()`, rather than a plain Python int list) caused
    sqlite3 to bind them as something matching NO row, silently returning
    an empty-but-valid-looking DataFrame instead of raising or matching."""
    import numpy as np

    plain = _build(synthetic_db, include_weather=False)
    con = sqlite3.connect(str(synthetic_db.db_path))
    try:
        numpy_seasons = build_game_outcome_rows(
            seasons=[np.int64(SEASON)], con=con, include_weather=False
        )
    finally:
        con.close()
    assert len(numpy_seasons) == len(plain)
    assert len(numpy_seasons) > 0
