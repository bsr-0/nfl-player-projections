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
    compute_player_week_predictions,
    resolve_week_source,
    _lookup_depth_chart_rank_asof,
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


class TestLookupDepthChartRankAsof:
    def _fake_table(self):
        # (gsis_id, _key=season*100+week, depth_chart_rank)
        return pd.DataFrame({
            "gsis_id": ["P1", "P1", "P1", "P2"],
            "_key": [202301, 202310, 202401, 202305],
            "depth_chart_rank": [1, 2, 1, 3],
        })

    def test_strictly_before_excludes_exact_week_match(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        # Week 10 (key=202310) has a snapshot, but target_week=10 itself
        # must NOT be able to use it -- only strictly earlier is allowed.
        result = _lookup_depth_chart_rank_asof("P1", 2023, 10)
        assert result == 1  # falls back to week 1's rank, not week 10's

    def test_uses_most_recent_strictly_prior_snapshot(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        result = _lookup_depth_chart_rank_asof("P1", 2023, 11)
        assert result == 2  # week 10's snapshot is now strictly prior

    def test_crosses_season_boundary_correctly(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        result = _lookup_depth_chart_rank_asof("P1", 2024, 1)
        assert result == 2  # 2023 week 10's snapshot, strictly before 2024 week 1

    def test_unknown_player_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        assert _lookup_depth_chart_rank_asof("UNKNOWN", 2023, 10) is None

    def test_no_prior_data_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        # P2's only snapshot is 2023 week 5 -- nothing strictly before it.
        assert _lookup_depth_chart_rank_asof("P2", 2023, 5) is None


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

    def test_depth_chart_rank_refreshed_on_detected_change(self, monkeypatch):
        """A demoted player's synthetic row must reflect the real, refreshed
        rank, not the stale carried-forward value -- the direct fix for
        the diagnosed real-vs-synthetic season-total bias."""
        history = pd.DataFrame({
            "player_id": ["P1", "P1"],
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "position": ["QB", "QB"],
            "depth_chart_rank": [1, 2],  # P1's own row = rank 1 (stale, healthy-starter)
            "snap_share_pct_roll3_mean": [0.9, 0.2],
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
        monkeypatch.setattr(
            "src.models.single_week_ppr.season_projection._lookup_depth_chart_rank_asof",
            lambda gsis_id, season, week: 2,  # depth chart now shows rank 2, not 1
        )

        result = build_synthetic_week_row(history, "P1", 2023, 3, "KC", db, feature_engineer)
        assert result["depth_chart_rank"].iloc[0] == 2
        # Usage-share rescaled toward rank-2's empirical average from history
        # (0.2), not left at rank-1's carried-forward stale value (0.9).
        assert result["snap_share_pct_roll3_mean"].iloc[0] < 0.9

    def test_depth_chart_rank_unchanged_when_no_asof_data(self, monkeypatch):
        """No prior depth-chart snapshot available (e.g. 2018/2019/2025) ->
        keep the carried-forward value, don't guess."""
        history = pd.DataFrame({
            "player_id": ["P1"], "season": [2018], "week": [1], "team": ["KC"],
            "position": ["QB"], "depth_chart_rank": [1], "snap_share_pct_roll3_mean": [0.9],
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
        monkeypatch.setattr(
            "src.models.single_week_ppr.season_projection._lookup_depth_chart_rank_asof",
            lambda gsis_id, season, week: None,
        )

        result = build_synthetic_week_row(history, "P1", 2018, 2, "KC", db, feature_engineer)
        assert result["depth_chart_rank"].iloc[0] == 1
        assert result["snap_share_pct_roll3_mean"].iloc[0] == 0.9


class TestPreseasonModeLeakInvariants:
    """Behavioural guards for `preseason_mode` (the August-legal arm used to
    compare Phase 7 against the three preseason models in
    scripts/walk_forward_preseason.py).

    Written behaviourally rather than as "this column is excluded", matching
    the exposure-leakage contract's reasoning in GAPS.md: the point is that
    the mode CANNOT consume target-season in-season data, not that some list
    happens to omit it.
    """

    @staticmethod
    def _patch_refreshes(monkeypatch, recorder):
        for name, target in (
            ("external", "src.data.external_data.add_external_features"),
            ("team_context",
             "src.models.single_week_ppr.season_projection._compute_team_rolling_context"),
            ("depth_chart",
             "src.models.single_week_ppr.season_projection._lookup_depth_chart_rank_asof"),
        ):
            def _make(n):
                def _fn(*a, **k):
                    recorder.append(n)
                    return {} if n == "team_context" else (None if n == "depth_chart" else a[0])
                return _fn
            monkeypatch.setattr(target, _make(name))
        monkeypatch.setattr(
            "src.predict.get_schedule_map_for_week",
            lambda db, season, week: {"KC": ("BUF", "home")},
        )

    def test_ignores_target_season_rows_when_carrying_forward(self, monkeypatch):
        """The whole point: a target-season game must not reach the row even
        though it sits in the history frame and is chronologically closer."""
        history = pd.DataFrame({
            "player_id": ["P1", "P1"],
            "season": [2024, 2025],
            "week": [17, 1],
            "team": ["KC", "KC"],
            "position": ["WR", "WR"],
            "targets_roll3_mean": [4.0, 99.0],  # 99.0 is the target-season value
        })
        calls = []
        self._patch_refreshes(monkeypatch, calls)
        fe = MagicMock()
        fe.refresh_matchup_features.side_effect = lambda df: df

        row = build_synthetic_week_row(
            history, "P1", 2025, 8, "KC", MagicMock(), fe, preseason_mode=True,
        )
        assert row is not None
        assert row["targets_roll3_mean"].iloc[0] == 4.0, "carried forward a 2025 game"

    def test_skips_every_in_season_refresh(self, monkeypatch):
        history = pd.DataFrame({
            "player_id": ["P1"], "season": [2024], "week": [17],
            "team": ["KC"], "position": ["WR"], "depth_chart_rank": [1],
            "injury_score": [0.2],
        })
        calls = []
        self._patch_refreshes(monkeypatch, calls)
        fe = MagicMock()
        fe.refresh_matchup_features.side_effect = lambda df: df

        row = build_synthetic_week_row(
            history, "P1", 2025, 8, "KC", MagicMock(), fe, preseason_mode=True,
        )
        assert calls == [], f"preseason mode consumed in-season data: {calls}"
        # Conditional-on-playing: never serve a real injury status.
        assert row["injury_score"].iloc[0] == 1.0

    def test_opponent_graded_on_prior_season_not_target_season(self, monkeypatch):
        """The schedule is August-knowable, so the opponent IS set -- but the
        strength probe must be stamped to the prior season."""
        history = pd.DataFrame({
            "player_id": ["P1"], "season": [2024], "week": [17],
            "team": ["KC"], "position": ["WR"], "opp_fpts_allowed_s2d_lag1": [0.0],
        })
        calls = []
        self._patch_refreshes(monkeypatch, calls)

        seen = {}
        def _refresh(df):
            seen["season"] = int(df["season"].iloc[0])
            df = df.copy()
            df["opp_fpts_allowed_s2d_lag1"] = 12.5
            return df
        fe = MagicMock()
        fe.refresh_matchup_features.side_effect = _refresh

        row = build_synthetic_week_row(
            history, "P1", 2025, 8, "KC", MagicMock(), fe, preseason_mode=True,
        )
        assert row["opponent"].iloc[0] == "BUF", "schedule is knowable in August"
        assert seen["season"] == 2024, "opponent strength probed in the target season"
        assert row["opp_fpts_allowed_s2d_lag1"].iloc[0] == 12.5
        assert int(row["season"].iloc[0]) == 2025, "probe must not restamp the real row"

    def test_no_week_is_ever_treated_as_known_played(self, monkeypatch):
        """Which weeks he played is target-season exposure. Even with real
        rows present for every week, preseason mode must synthesize all."""
        real = pd.DataFrame({
            "player_id": ["P1"], "season": [2025], "week": [1], "team": ["KC"],
            "fantasy_points": [30.0], "feat": [1.0], "data_source": ["nflverse_stats"],
        })
        model = MagicMock()
        model.predict.return_value = np.array([7.0])
        monkeypatch.setattr(
            "src.models.single_week_ppr.season_projection.build_synthetic_week_row",
            lambda *a, **k: pd.DataFrame({"feat": [1.0]}),
        )

        out = compute_player_week_predictions(
            "P1", {1: real}, {1}, {1: "KC"}, [1], model, ["feat"],
            pd.DataFrame({"player_id": ["P1"], "season": [2024], "week": [17], "feat": [1.0]}),
            MagicMock(), MagicMock(), 2025, preseason_mode=True,
        )
        assert len(out) == 1
        assert out[0]["is_real"] is False, "preseason mode treated a week as known-played"
        assert out[0]["actual_value"] is None
