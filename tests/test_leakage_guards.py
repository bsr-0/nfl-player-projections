"""Regression tests for leakage defenses.

Covers:
- src/utils/database.py: get_all_players_for_training must join team_stats
  (own-team) and team_defense_stats (opponent) strictly from week - 1.
- src/utils/leakage.py: is_leakage_feature / FEATURE_AVAILABILITY registry /
  audit_feature_availability classify feature columns correctly.
- src/data/external_data.py: InjuryDataLoader must drop injury reports
  modified after that week's kickoff (GAPS.md §7.6).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager
from src.data import external_data
from src.data.external_data import InjuryDataLoader
from src.utils.leakage import (
    is_leakage_feature,
    audit_feature_availability,
    FEATURE_AVAILABILITY,
)


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "test.db")


def _insert_player(db, player_id="P1", name="Test Player", position="WR"):
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO players (player_id, name, position) VALUES (?, ?, ?)",
            (player_id, name, position),
        )
        conn.commit()


def _insert_weekly_stats(db, player_id, season, week, team="AAA", opponent="BBB"):
    with db._get_connection() as conn:
        conn.execute(
            """INSERT INTO player_weekly_stats
               (player_id, season, week, team, opponent, receiving_yards)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (player_id, season, week, team, opponent, 50),
        )
        conn.commit()


def _insert_team_stats(db, team, season, week, total_yards=300):
    with db._get_connection() as conn:
        conn.execute(
            """INSERT INTO team_stats (team, season, week, total_yards)
               VALUES (?, ?, ?, ?)""",
            (team, season, week, total_yards),
        )
        conn.commit()


class TestOwnTeamStatsNotLeaked:
    def test_own_team_stats_week_is_always_prior_week(self, db):
        """get_all_players_for_training must never attach same-week team_stats."""
        _insert_player(db)
        for week in range(1, 5):
            _insert_weekly_stats(db, "P1", 2024, week)
            # Insert team_stats for every week including the current one, so
            # a same-week join (the bug) would be observable if it existed.
            _insert_team_stats(db, "AAA", 2024, week)

        df = db.get_all_players_for_training(min_games=1)

        assert "own_team_stats_week" in df.columns
        has_data = df["own_team_stats_week"].notna()
        assert has_data.any(), "expected at least one row with own-team stats joined"
        assert (df.loc[has_data, "own_team_stats_week"] == df.loc[has_data, "week"] - 1).all()
        assert not (df["own_team_stats_week"].fillna(-1) >= df["week"]).any()

    def test_week_one_has_no_prior_own_team_stats(self, db):
        """Week 1 has no week-0 team_stats to join, mirroring opp_defense_week behavior."""
        _insert_player(db)
        _insert_weekly_stats(db, "P1", 2024, 1)
        _insert_team_stats(db, "AAA", 2024, 1)  # same-week only, must NOT be joined

        df = db.get_all_players_for_training(min_games=1)
        assert df.loc[df["week"] == 1, "own_team_stats_week"].isna().all()


class TestLeakageFeatureGuard:
    @pytest.mark.parametrize(
        "col",
        [
            "predicted_fantasy_points",
            "projection_1w",
            "projection_18w",
            "target",
            "target_util_4w",
            "baseline_fp",
            "actual_for_backtest",
            "player_id",
            "utilization_score",
        ],
    )
    def test_blocks_known_leakage_columns(self, col):
        assert is_leakage_feature(col) is True

    @pytest.mark.parametrize(
        "col",
        [
            "targets_roll3_mean",
            "opp_fpts_allowed",
            "sos_next_4",
            "expected_games_next_4",
            "implied_team_total",
        ],
    )
    def test_allows_legitimate_features(self, col):
        assert is_leakage_feature(col) is False


class TestFeatureAvailabilityRegistry:
    def test_registry_is_nonempty_and_well_formed(self):
        assert len(FEATURE_AVAILABILITY) > 0
        for pattern, rule in FEATURE_AVAILABILITY:
            assert isinstance(pattern, str) and pattern
            assert isinstance(rule, str) and rule

    @pytest.mark.parametrize(
        "col",
        [
            "targets_roll3_mean",
            "team_yards",
            "team_pass_attempts",
            "opp_fpts_allowed_s2d_lag1",
            "implied_team_total",
            "spread",
            "injury_score",
            "is_rookie",
            "prev_season_ppg",
        ],
    )
    def test_known_feature_families_are_classified(self, col):
        assert col not in audit_feature_availability([col])

    def test_unclassified_synthetic_column_is_flagged(self):
        flagged = audit_feature_availability(["totally_new_unaudited_metric_xyz"])
        assert flagged == ["totally_new_unaudited_metric_xyz"]

    def test_missing_values_are_not_flagged_as_leakage(self):
        """Registry classifies column identity only; nulls are not audited here."""
        # A column with NaNs is still a column name — audit only inspects names.
        assert audit_feature_availability(["targets_roll3_mean"]) == []


class TestInjuryTimingGuard:
    @pytest.fixture(autouse=True)
    def _mock_schedule(self, monkeypatch):
        """AAA hosts BBB in week 5, 2024, kicking off at 13:00 America/New_York."""
        schedule = pd.DataFrame({
            "season": [2024],
            "week": [5],
            "gameday": ["2024-10-06"],
            "gametime": ["13:00"],
            "home_team": ["AAA"],
            "away_team": ["BBB"],
        })
        monkeypatch.setattr(external_data.nfl, "import_schedules", lambda seasons: schedule)

    def test_drops_report_modified_after_kickoff(self):
        injuries = pd.DataFrame({
            "gsis_id": ["P1", "P2"],
            "season": [2024, 2024],
            "week": [5, 5],
            "team": ["AAA", "AAA"],
            "report_status": ["Questionable", "Out"],
            "date_modified": [
                pd.Timestamp("2024-10-04 20:00:00", tz="UTC"),  # Friday before kickoff — OK
                pd.Timestamp("2024-10-06 18:30:00", tz="UTC"),  # after 17:00 UTC kickoff — leaks
            ],
        })

        result = InjuryDataLoader().get_player_injury_status(injuries)

        assert len(result) == 1
        assert result.iloc[0]["player_id"] == "P1"

    def test_keeps_reports_with_unmatched_schedule(self):
        """No leakage evidence for an unmatched team/week — keep the row (missingness != leakage)."""
        injuries = pd.DataFrame({
            "gsis_id": ["P3"],
            "season": [2024],
            "week": [99],  # no matching schedule row
            "team": ["ZZZ"],
            "report_status": ["Questionable"],
            "date_modified": [pd.Timestamp("2024-12-01 00:00:00", tz="UTC")],
        })

        result = InjuryDataLoader().get_player_injury_status(injuries)

        assert len(result) == 1
