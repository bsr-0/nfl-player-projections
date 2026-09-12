"""NFLPredictor._refresh_injury_status_for_target_week must re-point
injury_score at the week actually being predicted, not the player's last
completed game.

`latest_data` starts as each player's last completed game with season/week
already overwritten to the prediction target -- exactly like
refresh_matchup_features does for team-relative features. Before this,
injury_score (a model feature, and the source of the availability discount)
still held the report from whatever week the player last actually played,
which has no relationship to their status for the week being forecast.
"""
import pandas as pd
import pytest

from src.predict import NFLPredictor
from src.utils import database as db_mod
from src.utils.database import DatabaseManager


@pytest.fixture
def predictor(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path=db_path)
    db.insert_player({"player_id": "P1", "name": "Player One", "position": "WR"})
    db.insert_player({"player_id": "P2", "name": "Player Two", "position": "WR"})
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO player_injuries (player_id, season, week, team, report_status, date_modified) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("P1", 2025, 10, "AAA", "Out", "2025-11-04T12:00:00Z"),
        )
        conn.commit()
    # _merge_injury_data_from_cache instantiates a fresh DatabaseManager()
    # internally rather than taking one as an argument, so it must be pointed
    # at this test's private db the same way the leakage-guard tests do.
    monkeypatch.setattr(db_mod, "DatabaseManager", lambda: db)

    p = NFLPredictor.__new__(NFLPredictor)
    p.db = db
    from src.features.feature_engineering import FeatureEngineer
    p.feature_engineer = FeatureEngineer()
    return p


def test_refresh_uses_the_target_weeks_own_report_not_the_stale_one(predictor):
    # P1's last COMPLETED game (week 9) had no issue; the target week (10) has
    # a real "Out" report. Simulates predict()'s state right after season/week
    # are overwritten to the target: latest_data still carries week 9's stale
    # injury_score, but season/week columns already say week 10.
    latest_data = pd.DataFrame({
        "player_id": ["P1", "P2"],
        "season": [2025, 2025],
        "week": [10, 10],                 # already overwritten to the target week
        "injury_score": [1.0, 1.0],       # stale: week 9's (healthy) report
        "is_injured": [0, 0],
    })

    out = predictor._refresh_injury_status_for_target_week(latest_data)

    assert out.set_index("player_id").loc["P1", "injury_score"] == 0.0   # real target-week report: Out
    assert out.set_index("player_id").loc["P2", "injury_score"] == 1.0   # no report -> healthy default


def test_stale_value_does_not_leak_through_as_a_fallback(predictor):
    """A player with no report for the target week must default to healthy,
    not silently inherit whatever was in the frame from their last game."""
    latest_data = pd.DataFrame({
        "player_id": ["P2"], "season": [2025], "week": [10],
        "injury_score": [0.0], "is_injured": [1],   # stale: was "Out" in their last game
    })
    out = predictor._refresh_injury_status_for_target_week(latest_data)
    assert out.loc[0, "injury_score"] == 1.0


def test_target_season_with_no_injury_data_at_all_defaults_healthy(predictor):
    """_merge_injury_data_from_cache early-returns the frame UNCHANGED when its
    query finds nothing for the queried season (e.g. a future season with no
    injury data loaded yet) or raises. Since injury_score was already dropped
    before that call, the naive result would be a MISSING column, not a
    healthy default -- this is exactly what silently emptied injury_adjustment/
    expected_points for every player on the live board (2026-09-11)."""
    latest_data = pd.DataFrame({
        "player_id": ["P1", "P2"], "season": [2099, 2099], "week": [1, 1],
        "injury_score": [0.0, 0.0], "is_injured": [1, 1],   # stale, irrelevant season
    })
    out = predictor._refresh_injury_status_for_target_week(latest_data)
    assert "injury_score" in out.columns
    assert out["injury_score"].tolist() == [1.0, 1.0]
    assert out["is_injured"].tolist() == [0, 0]


def test_noop_when_no_injury_columns_present(predictor):
    latest_data = pd.DataFrame({"player_id": ["P1"], "season": [2025], "week": [10]})
    out = predictor._refresh_injury_status_for_target_week(latest_data)
    pd.testing.assert_frame_equal(out, latest_data)
