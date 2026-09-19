"""Leakage-safety invariants for the margin/total regression feature builder.

Reuses the synthetic DB fixture from test_game_outcome_leakage.py so both
targets are exercised against the identical underlying data -- home_score/
away_score must never survive into the feature frame, every feature column
must be classified, and the margin/total labels must match hand-computed
values from the seeded schedule scores.
"""
import sqlite3

from src.models.game_outcome.features import build_margin_total_rows, feature_columns
from src.utils.database import DatabaseManager
from src.utils.leakage import audit_feature_availability
from tests.test_game_outcome_leakage import SEASON, synthetic_db  # noqa: F401  (fixture import)


def _build(db: DatabaseManager, **kwargs):
    con = sqlite3.connect(str(db.db_path))
    try:
        return build_margin_total_rows(seasons=[SEASON], con=con, **kwargs)
    finally:
        con.close()


def test_scores_never_survive_into_feature_frame(synthetic_db):
    rows = _build(synthetic_db)
    assert {"home_score", "away_score"}.isdisjoint(rows.columns)


def test_all_feature_columns_are_classified(synthetic_db):
    rows = _build(synthetic_db)
    assert audit_feature_availability(feature_columns(rows)) == []


def test_labels_match_hand_computed_values_from_seeded_scores(synthetic_db):
    # _seed_db in test_game_outcome_leakage.py always sets home_score=24,
    # away_score=20 for every seeded (non-tie) game.
    rows = _build(synthetic_db)
    assert (rows["home_margin"] == 4).all()
    assert (rows["game_total"] == 44).all()


def test_tie_games_are_dropped_same_as_home_win_target(synthetic_db):
    rows = _build(synthetic_db)
    assert not ((rows.home_team == "CCC") & (rows.away_team == "DDD")).any()


def test_same_game_population_as_home_win_target(synthetic_db):
    from src.models.game_outcome.features import build_game_outcome_rows

    con = sqlite3.connect(str(synthetic_db.db_path))
    try:
        win_rows = build_game_outcome_rows(seasons=[SEASON], con=con)
        margin_rows = build_margin_total_rows(seasons=[SEASON], con=con)
    finally:
        con.close()
    win_keys = set(zip(win_rows.season, win_rows.week, win_rows.home_team, win_rows.away_team))
    margin_keys = set(zip(margin_rows.season, margin_rows.week, margin_rows.home_team, margin_rows.away_team))
    assert win_keys == margin_keys
