"""Tests for the v30 offensive-line quality metrics and PBP-derived
pass-play participation rate (GAPS.md §11.1.C/D/H follow-up)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.features.utilization_score import UtilizationScoreCalculator
from src.data.pbp_stats_aggregator import get_pass_play_participation_from_pbp


def _base_player_rows(n_weeks=4):
    """Minimal player_weekly_stats-shaped rows for one team/player."""
    rows = []
    for week in range(1, n_weeks + 1):
        rows.append({
            "player_id": "00-0000001",
            "team": "KC",
            "season": 2023,
            "week": week,
        })
    return pd.DataFrame(rows)


class TestTeamOlFeatures:
    def test_pass_block_uses_starter_not_average(self, monkeypatch):
        """A backup QB's mop-up-duty row (low activity, e.g. one garbage-
        time sack) shouldn't dilute the starter's real pressure rate."""
        weekly_pfr = pd.DataFrame([
            # Starter: heavy activity, moderate pressure rate.
            {"team": "KC", "season": 2023, "week": 1, "stat_type": "pass",
             "times_pressured_pct": 0.20, "times_sacked": 2, "times_blitzed": 5,
             "times_hurried": 4, "times_hit": 3, "rushing_yards_before_contact_avg": None},
            # Backup: one mop-up snap, sacked once -> 100% pressured pct,
            # would badly skew a naive average toward "bad OL".
            {"team": "KC", "season": 2023, "week": 1, "stat_type": "pass",
             "times_pressured_pct": 1.00, "times_sacked": 1, "times_blitzed": 0,
             "times_hurried": 0, "times_hit": 0, "rushing_yards_before_contact_avg": None},
        ])

        monkeypatch.setattr(pd, "read_sql", lambda query, conn: weekly_pfr)

        engineer = FeatureEngineer(feature_mode="causal")
        df = _base_player_rows(n_weeks=1)
        out = engineer._add_team_ol_features(df)

        # No prior history yet (shift(1)), so roll3 is NaN->filled 0 for
        # week 1 itself; verify the underlying merge picked the starter by
        # re-running with a second week so week 2's roll3 reflects week 1's
        # starter value, not an average with the backup.
        df2 = _base_player_rows(n_weeks=2)
        out2 = engineer._add_team_ol_features(df2)
        week2_value = out2[out2["week"] == 2]["team_sack_rate_allowed_roll3_mean"].iloc[0]
        assert week2_value == pytest.approx(0.20), (
            f"Expected week 2's rolled value to reflect the starter's 0.20 "
            f"pressure rate, not an average with the backup's 1.00, got {week2_value}"
        )

    def test_run_block_averages_across_rbs(self, monkeypatch):
        weekly_pfr = pd.DataFrame([
            {"team": "KC", "season": 2023, "week": 1, "stat_type": "rush",
             "rushing_yards_before_contact_avg": 2.0, "times_pressured_pct": None,
             "times_sacked": None, "times_blitzed": None, "times_hurried": None, "times_hit": None},
            {"team": "KC", "season": 2023, "week": 1, "stat_type": "rush",
             "rushing_yards_before_contact_avg": 4.0, "times_pressured_pct": None,
             "times_sacked": None, "times_blitzed": None, "times_hurried": None, "times_hit": None},
        ])
        monkeypatch.setattr(pd, "read_sql", lambda query, conn: weekly_pfr)

        engineer = FeatureEngineer(feature_mode="causal")
        df = _base_player_rows(n_weeks=2)
        out = engineer._add_team_ol_features(df)
        week2_value = out[out["week"] == 2]["team_run_block_ybc_avg_roll3_mean"].iloc[0]
        assert week2_value == pytest.approx(3.0)

    def test_missing_columns_degrades_gracefully(self):
        engineer = FeatureEngineer(feature_mode="causal")
        df = pd.DataFrame({"player_id": ["p1"], "season": [2023]})  # no "team"/"week"
        out = engineer._add_team_ol_features(df)
        assert (out["team_sack_rate_allowed_roll3_mean"] == 0.0).all()
        assert (out["team_run_block_ybc_avg_roll3_mean"] == 0.0).all()


class TestPbpPassParticipationFeatures:
    def test_rolled_value_reflects_prior_week(self, monkeypatch):
        participation = pd.DataFrame([
            {"player_id": "00-0000001", "season": 2023, "week": 1,
             "pass_play_participation_pct": 0.85},
        ])
        monkeypatch.setattr(pd, "read_sql", lambda query, conn: participation)

        engineer = FeatureEngineer(feature_mode="causal")
        df = _base_player_rows(n_weeks=2)
        out = engineer._add_pbp_pass_participation_features(df)

        week1_value = out[out["week"] == 1]["pbp_pass_play_participation_pct_roll3_mean"].iloc[0]
        week2_value = out[out["week"] == 2]["pbp_pass_play_participation_pct_roll3_mean"].iloc[0]
        # Causal: week 1's own participation must not leak into week 1's feature.
        assert week1_value == pytest.approx(0.0)
        # Week 2 should reflect week 1's real value via shift(1).
        assert week2_value == pytest.approx(0.85)

    def test_missing_columns_degrades_gracefully(self):
        engineer = FeatureEngineer(feature_mode="causal")
        df = pd.DataFrame({"team": ["KC"]})  # no player_id/season/week
        out = engineer._add_pbp_pass_participation_features(df)
        assert (out["pbp_pass_play_participation_pct_roll3_mean"] == 0.0).all()


class TestGetPassPlayParticipationFromPbp:
    def test_computes_per_player_and_team_counts(self, monkeypatch):
        pbp = pd.DataFrame([
            # Play 1: 3 offensive players, KC.
            {"season": 2023, "week": 1, "play_type": "pass", "posteam": "KC",
             "offense_players": "00-0000001;00-0000002;00-0000003"},
            # Play 2: player 1 sits out (e.g. blocking sub), 2 and 3 stay in.
            {"season": 2023, "week": 1, "play_type": "pass", "posteam": "KC",
             "offense_players": "00-0000002;00-0000003;00-0000004"},
            # A run play should be excluded entirely.
            {"season": 2023, "week": 1, "play_type": "run", "posteam": "KC",
             "offense_players": "00-0000001;00-0000002;00-0000003"},
        ])

        class _FakeNfl:
            @staticmethod
            def import_pbp_data(seasons):
                return pbp

        # get_pass_play_participation_from_pbp does `import nfl_data_py as
        # nfl` internally; patching sys.modules makes that import resolve
        # to the fake module instead of hitting the network.
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "nfl_data_py", _FakeNfl())

        result = get_pass_play_participation_from_pbp(2023, use_cache=False)

        team_plays = result[["team", "season", "week", "team_pass_plays"]].drop_duplicates()
        assert team_plays["team_pass_plays"].iloc[0] == 2  # only the 2 real pass plays

        p1 = result[result["player_id"] == "00-0000001"].iloc[0]
        p2 = result[result["player_id"] == "00-0000002"].iloc[0]
        assert p1["player_pass_plays"] == 1
        assert p2["player_pass_plays"] == 2
        assert p1["pass_play_participation_pct"] == pytest.approx(0.5)
        assert p2["pass_play_participation_pct"] == pytest.approx(1.0)

    def test_empty_when_column_missing(self, monkeypatch):
        class _FakeNfl:
            @staticmethod
            def import_pbp_data(seasons):
                return pd.DataFrame({"season": [2010], "week": [1], "play_type": ["pass"]})

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "nfl_data_py", _FakeNfl())

        result = get_pass_play_participation_from_pbp(2010, use_cache=False)
        assert result.empty


class TestRouteParticipationFallbackChain:
    def test_prefers_pbp_derived_column_over_flat_proxy(self):
        calc = UtilizationScoreCalculator()
        n = 5
        df = pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n)],
            "season": [2023] * n,
            "week": [1] * n,
            "position": ["WR"] * n,
            "targets": np.random.uniform(2, 10, n),
            "team_targets": [30] * n,
            "receiving_yards": np.random.uniform(20, 100, n),
            "air_yards": np.random.uniform(50, 200, n),
            "snap_count": np.random.uniform(30, 60, n),
            "team_snaps": [65] * n,
            "receiving_tds": np.random.randint(0, 2, n),
            "pbp_pass_play_participation_pct_roll3_mean": [0.75] * n,
        })
        out = calc._calculate_wr_utilization(df.copy())
        # Should use the real PBP-derived column (0.75 -> 75%), not the
        # flat snap_share_pct * 0.8 estimate.
        expected_snap_share = (df["snap_count"] / df["team_snaps"]) * 100
        flat_proxy = (expected_snap_share * 0.8).clip(0, 100)
        assert np.allclose(out["route_participation_pct"], 75.0)
        assert not np.allclose(out["route_participation_pct"], flat_proxy)
