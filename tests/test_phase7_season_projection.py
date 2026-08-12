"""Unit tests for Phase 7 (next_focus.md) 18-week season projection.

Synthetic data / monkeypatched DB calls only — no real DB/network for the
core logic tests (possible_weeks_for_team, estimate_availability_rate,
build_synthetic_week_row's carry-forward mechanics).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.season_projection import (
    possible_weeks_for_team,
    estimate_availability_rate,
    build_synthetic_week_row,
    resolve_week_source,
    REGULAR_SEASON_MAX_WEEK,
)


class TestPossibleWeeksForTeam:
    def test_excludes_bye_week(self):
        db = MagicMock()
        db.get_schedule.return_value = pd.DataFrame({
            "week": [1, 2, 3, 5, 6],  # week 4 = bye
            "home_team": ["KC"] * 5,
        })
        weeks = possible_weeks_for_team(db, "KC", 2023)
        assert weeks == [1, 2, 3, 5, 6]
        assert 4 not in weeks

    def test_excludes_playoff_weeks(self):
        db = MagicMock()
        db.get_schedule.return_value = pd.DataFrame({
            "week": list(range(1, 23)),  # includes playoff weeks 19-22
            "home_team": ["KC"] * 22,
        })
        weeks = possible_weeks_for_team(db, "KC", 2023)
        assert max(weeks) == REGULAR_SEASON_MAX_WEEK
        assert 19 not in weeks

    def test_empty_schedule_returns_empty(self):
        db = MagicMock()
        db.get_schedule.return_value = pd.DataFrame()
        assert possible_weeks_for_team(db, "KC", 2023) == []


class TestEstimateAvailabilityRate:
    def test_uses_prior_seasons_only_not_current(self):
        """Leakage check: must never use the season being evaluated."""
        db = MagicMock()
        # Prior season 2022: played 10 of 17 possible weeks.
        # Current season 2023 (being evaluated) should NOT affect the rate,
        # even though the mock history technically contains 2023 rows too.
        history = pd.DataFrame({
            "player_id": ["P1"] * 12,
            "season": [2022] * 10 + [2023] * 2,  # only 2 games in 2023 (injured all year)
            "week": list(range(1, 11)) + [1, 2],
            "team": ["KC"] * 12,
        })
        db.get_all_players_for_training.return_value = history
        db.get_schedule.return_value = pd.DataFrame({"week": list(range(1, 18)), "home_team": ["KC"] * 17})

        rate = estimate_availability_rate("P1", "WR", 2023, db)
        # Should reflect 2022's 10/17 rate, NOT be dragged down by 2023's 2 games.
        assert rate == pytest.approx(10 / 17, abs=0.01)

    def test_rookie_with_no_prior_season_uses_fallback(self):
        db = MagicMock()
        db.get_all_players_for_training.return_value = pd.DataFrame({
            "player_id": ["P1"], "season": [2023], "week": [1], "team": ["KC"],
        })
        rate = estimate_availability_rate("P1", "WR", 2023, db, position_avg_fallback=0.75)
        assert rate == 0.75


class TestResolveWeekSource:
    def test_no_row_at_all_is_synthetic(self):
        assert resolve_week_source(5, real_weeks={1, 2, 3}, data_source=None) is False

    def test_real_stats_row_is_not_discounted(self):
        assert resolve_week_source(2, real_weeks={1, 2, 3}, data_source="nflverse_stats") is True

    def test_snap_verified_zero_row_is_not_discounted(self):
        assert resolve_week_source(
            2, real_weeks={1, 2, 3}, data_source="inferred_snap_verified_zero",
        ) is True

    def test_pbp_confirmed_zero_row_is_not_discounted_by_default(self):
        assert resolve_week_source(
            2, real_weeks={1, 2, 3}, data_source="inferred_pbp_confirmed_zero",
        ) is True

    def test_exclude_flag_falls_back_to_synthetic_for_pbp_confirmed_only(self):
        assert resolve_week_source(
            2, real_weeks={1, 2, 3}, data_source="inferred_pbp_confirmed_zero",
            exclude_pbp_confirmed_zeros=True,
        ) is False
        # Real stats and snap-verified rows are unaffected by the toggle.
        assert resolve_week_source(
            2, real_weeks={1, 2, 3}, data_source="nflverse_stats",
            exclude_pbp_confirmed_zeros=True,
        ) is True
        assert resolve_week_source(
            2, real_weeks={1, 2, 3}, data_source="inferred_snap_verified_zero",
            exclude_pbp_confirmed_zeros=True,
        ) is True


class TestBuildSyntheticWeekRow:
    def test_returns_none_with_no_prior_history(self):
        history = pd.DataFrame({
            "player_id": ["P1"], "season": [2023], "week": [5],
        })
        result = build_synthetic_week_row(
            history, "P1", 2023, 3, "KC", MagicMock(), MagicMock(),
        )
        assert result is None

    def test_carries_forward_most_recent_prior_row(self, monkeypatch):
        history = pd.DataFrame({
            "player_id": ["P1", "P1", "P1"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "team": ["KC", "KC", "KC"],
            "targets_roll3_mean": [5.0, 6.0, 7.0],
        })
        db = MagicMock()
        db.get_schedule.return_value = pd.DataFrame()  # empty -> opponent unknown, that's fine
        feature_engineer = MagicMock()
        feature_engineer.refresh_matchup_features.side_effect = lambda df: df

        monkeypatch.setattr(
            "src.predict.get_schedule_map_for_week",
            lambda db, season, week: {"KC": ("BUF", "home")},
        )
        monkeypatch.setattr(
            "src.data.external_data.add_external_features",
            lambda df, seasons=None: df,
        )
        monkeypatch.setattr(
            "src.models.single_week_ppr.season_projection._compute_team_rolling_context",
            lambda team, season, week, n=3: {},
        )

        result = build_synthetic_week_row(history, "P1", 2023, 4, "KC", db, feature_engineer)
        assert result is not None
        # Carries forward week 3's rolling feature (most recent prior row).
        assert result["targets_roll3_mean"].iloc[0] == 7.0
        assert result["week"].iloc[0] == 4
        assert result["opponent"].iloc[0] == "BUF"

    def test_injury_score_set_neutral_not_looked_up(self, monkeypatch):
        """Conditional-on-playing assumption: never leak a real injury
        status for a week the player didn't play."""
        history = pd.DataFrame({
            "player_id": ["P1"], "season": [2023], "week": [1], "team": ["KC"],
            "injury_score": [0.0],  # was actually "Out" that game
        })
        db = MagicMock()
        feature_engineer = MagicMock()
        feature_engineer.refresh_matchup_features.side_effect = lambda df: df

        monkeypatch.setattr(
            "src.predict.get_schedule_map_for_week", lambda db, season, week: {},
        )
        monkeypatch.setattr(
            "src.data.external_data.add_external_features", lambda df, seasons=None: df,
        )
        monkeypatch.setattr(
            "src.models.single_week_ppr.season_projection._compute_team_rolling_context",
            lambda team, season, week, n=3: {},
        )

        result = build_synthetic_week_row(history, "P1", 2023, 2, "KC", db, feature_engineer)
        assert result["injury_score"].iloc[0] == 1.0
