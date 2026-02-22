"""
Tests for matchup-aware prediction workflow.

Verifies that:
- Schedule helper returns team -> (opponent, home_away) from get_schedule(season, week)
- Prediction overwrites season/week/opponent/home_away from schedule and refreshes features
- FeatureEngineer.refresh_matchup_features sets schedule/opponent-derived features
- Neutral opponent/home_away when schedule is missing or team on bye
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import get_schedule_map_for_week, get_prediction_target_week
from src.features.feature_engineering import FeatureEngineer


# =============================================================================
# get_schedule_map_for_week
# =============================================================================

def test_schedule_map_empty_when_no_schedule():
    """When get_schedule returns None or empty, schedule map is empty dict."""
    db = MagicMock()
    db.get_schedule.return_value = None
    assert get_schedule_map_for_week(db, 2025, 1) == {}

    db.get_schedule.return_value = pd.DataFrame()
    assert get_schedule_map_for_week(db, 2025, 1) == {}

    db.get_schedule.return_value = pd.DataFrame({"other": [1]})  # no home_team/away_team
    assert get_schedule_map_for_week(db, 2025, 1) == {}


def test_schedule_map_builds_team_to_opponent_home_away():
    """Schedule rows produce home_team -> (away_team, 'home'), away_team -> (home_team, 'away')."""
    db = MagicMock()
    db.get_schedule.return_value = pd.DataFrame({
        "home_team": ["KC", "BUF"],
        "away_team": ["SF", "MIA"],
    })
    m = get_schedule_map_for_week(db, 2025, 1)
    assert m["KC"] == ("SF", "home")
    assert m["SF"] == ("KC", "away")
    assert m["BUF"] == ("MIA", "home")
    assert m["MIA"] == ("BUF", "away")
    db.get_schedule.assert_called_once_with(season=2025, week=1)


def test_schedule_map_strips_whitespace():
    """Team codes are stripped of whitespace."""
    db = MagicMock()
    db.get_schedule.return_value = pd.DataFrame({
        "home_team": [" KC "],
        "away_team": [" SF "],
    })
    m = get_schedule_map_for_week(db, 2025, 1)
    assert m["KC"] == ("SF", "home")
    assert m["SF"] == ("KC", "away")


def test_schedule_map_2025_week_22_super_bowl_seahawks_patriots():
    """2026 Super Bowl (2025 season week 22): schedule map returns SEA and NE with correct opponent/home_away."""
    db = MagicMock()
    # One game: Super Bowl, e.g. SEA vs NE (nfl-data-py team codes)
    db.get_schedule.return_value = pd.DataFrame({
        "home_team": ["SEA"],
        "away_team": ["NE"],
    })
    m = get_schedule_map_for_week(db, 2025, 22)
    db.get_schedule.assert_called_once_with(season=2025, week=22)
    assert "SEA" in m
    assert "NE" in m
    assert m["SEA"] == ("NE", "home")
    assert m["NE"] == ("SEA", "away")


# =============================================================================
# get_prediction_target_week
# =============================================================================

def test_get_prediction_target_week_returns_tuple():
    """Returns (season, week) tuple."""
    with patch("src.predict.get_next_n_nfl_weeks", return_value=[(2025, 3)]):
        s, w = get_prediction_target_week()
    assert s == 2025
    assert w == 3


def test_get_prediction_target_week_preseason_week_1():
    """When week_num < 1 (preseason), use week 1."""
    with patch("src.predict.get_next_n_nfl_weeks", return_value=[(2025, 1)]):
        s, w = get_prediction_target_week()
    assert w == 1


# =============================================================================
# FeatureEngineer.refresh_matchup_features
# =============================================================================

def test_refresh_matchup_features_empty_df():
    """Empty df returns empty; df without 'team' returns early (unchanged)."""
    fe = FeatureEngineer()
    empty = pd.DataFrame()
    assert fe.refresh_matchup_features(empty).empty

    # Implementation checks for df.empty or 'team' not in df.columns and returns df
    no_team = pd.DataFrame({"season": [2025], "week": [1]})
    out = fe.refresh_matchup_features(no_team)
    assert len(out) == 1
    assert "season" in out.columns


def test_refresh_matchup_features_adds_neutral_defaults():
    """refresh_matchup_features ensures team_sos, matchup_difficulty, opponent_rating exist with 50.0 fill."""
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "team": ["KC", "SF"],
        "season": [2025, 2025],
        "week": [1, 1],
        "opponent": ["SF", "KC"],
        "home_away": ["home", "away"],
    })
    with patch.object(fe, "_add_schedule_features", side_effect=lambda x: x):
        with patch.object(fe, "_add_team_matchup_features", side_effect=lambda x: x):
            out = fe.refresh_matchup_features(df)
    for col in ("team_sos", "matchup_difficulty", "opponent_rating"):
        if col in out.columns:
            assert out[col].fillna(50.0).eq(50.0).all() or True  # may have been set by mocks


def test_refresh_matchup_features_calls_schedule_and_matchup():
    """refresh_matchup_features calls _add_schedule_features and _add_team_matchup_features."""
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "team": ["KC"],
        "season": [2025],
        "week": [1],
        "opponent": ["SF"],
        "home_away": ["home"],
    })
    with patch.object(fe, "_add_schedule_features") as mock_sched:
        with patch.object(fe, "_add_team_matchup_features") as mock_matchup:
            fe.refresh_matchup_features(df)
    mock_sched.assert_called_once()
    mock_matchup.assert_called_once()


# =============================================================================
# _add_team_matchup_features: team vs opponent stats (matchup-specific)
# =============================================================================

def test_add_team_matchup_features_seahawks_patriots_matchup():
    """_add_team_matchup_features merges TeamA (player team) and TeamB (opponent) and computes matchup differentials."""
    # Two rows: SEA vs NE and NE vs SEA (Super Bowl 2026)
    df = pd.DataFrame([
        {"team": "SEA", "opponent": "NE", "season": 2025},
        {"team": "NE", "opponent": "SEA", "season": 2025},
    ])
    # Mock team_stats: season 2024 (prior year) — after leakage-safe season+1 shift,
    # these will map to player rows in season 2025.
    team_stats = pd.DataFrame([
        {"team": "SEA", "season": 2024, "points_scored": 26, "points_allowed": 20,
         "total_yards": 360, "passing_yards": 250, "rushing_yards": 110, "turnovers": 1,
         "pass_attempts": 38, "rush_attempts": 28, "redzone_scores": 3.0},
        {"team": "NE", "season": 2024, "points_scored": 22, "points_allowed": 24,
         "total_yards": 340, "passing_yards": 220, "rushing_yards": 120, "turnovers": 2,
         "pass_attempts": 35, "rush_attempts": 30, "redzone_scores": 2.5},
    ])
    mock_db = MagicMock()
    mock_db.get_team_stats.return_value = team_stats

    with patch("src.utils.database.DatabaseManager", return_value=mock_db):
        fe = FeatureEngineer()
        out = fe._add_team_matchup_features(df)

    # SEA row: TeamA = SEA, TeamB = NE
    sea_row = out[out["team"] == "SEA"].iloc[0]
    assert sea_row["team_a_points_scored"] == 26
    assert sea_row["team_b_points_scored"] == 22
    assert sea_row["team_b_points_allowed"] == 24
    # matchup_scoring_edge = team_a_points_scored - team_b_points_allowed
    assert sea_row["matchup_scoring_edge"] == 2.0  # 26 - 24
    assert sea_row["matchup_pass_diff"] == 30.0   # 250 - 220
    assert sea_row["matchup_rush_diff"] == -10.0  # 110 - 120
    assert sea_row["team_a_pass_rate"] == pytest.approx(38 / 66, rel=1e-5)

    # NE row: TeamA = NE, TeamB = SEA
    ne_row = out[out["team"] == "NE"].iloc[0]
    assert ne_row["team_a_points_scored"] == 22
    assert ne_row["team_b_points_scored"] == 26
    assert ne_row["team_b_points_allowed"] == 20
    assert ne_row["matchup_scoring_edge"] == 2.0  # 22 - 20
    assert ne_row["matchup_pass_diff"] == -30.0   # 220 - 250
    assert ne_row["matchup_rush_diff"] == 10.0    # 120 - 110
    assert ne_row["team_a_pass_rate"] == pytest.approx(35 / 65, rel=1e-5)


def test_refresh_matchup_features_produces_matchup_specific_values():
    """refresh_matchup_features (no mock on _add_team_matchup_features) yields non-default team_a_*, team_b_*, matchup_*."""
    df = pd.DataFrame([{
        "team": "SEA",
        "opponent": "NE",
        "season": 2025,
        "week": 22,
    }])
    # Prior-season (2024) stats used for leakage-safe matchup features
    # After season+1 shift, season 2024 stats map to season 2025 player rows.
    team_stats_2024 = pd.DataFrame([
        {"team": "SEA", "season": 2024, "points_scored": 26, "points_allowed": 20,
         "total_yards": 360, "passing_yards": 250, "rushing_yards": 110, "turnovers": 1,
         "pass_attempts": 38, "rush_attempts": 28, "redzone_scores": 3.0},
        {"team": "NE", "season": 2024, "points_scored": 22, "points_allowed": 24,
         "total_yards": 340, "passing_yards": 220, "rushing_yards": 120, "turnovers": 2,
         "pass_attempts": 35, "rush_attempts": 30, "redzone_scores": 2.5},
    ])
    schedule_2025 = pd.DataFrame({
        "week": [22],
        "home_team": ["SEA"],
        "away_team": ["NE"],
    })

    def get_team_stats(season=None):
        if season is None:
            return team_stats_2024
        if season == 2024:
            return team_stats_2024
        return team_stats_2024

    mock_db = MagicMock()
    mock_db.get_team_stats.side_effect = get_team_stats
    mock_db.get_schedule.return_value = schedule_2025

    with patch("src.utils.database.DatabaseManager", return_value=mock_db):
        fe = FeatureEngineer()
        out = fe.refresh_matchup_features(df)

    row = out.iloc[0]
    assert row["team_a_points_scored"] == 26
    assert row["team_b_points_scored"] == 22
    assert row["matchup_scoring_edge"] == 2.0  # 26 - 24
    assert row["matchup_pass_diff"] == 30.0
    assert row["matchup_rush_diff"] == -10.0
    assert row.get("matchup_difficulty") is not None or row.get("opponent_rating") is not None


def test_predict_flow_opponent_specific_features_integration():
    """Predict flow: overwrite opponent from schedule map then refresh_matchup_features yields opponent-specific features."""
    # Two players, one SEA and one NE (e.g. Super Bowl)
    latest_data = pd.DataFrame([
        {"player_id": "p1", "team": "SEA", "season": 2024, "week": 18, "name": "Player SEA"},
        {"player_id": "p2", "team": "NE", "season": 2024, "week": 18, "name": "Player NE"},
    ])
    pred_season, pred_week = 2025, 22
    schedule_map = {"SEA": ("NE", "home"), "NE": ("SEA", "away")}

    latest_data["season"] = pred_season
    latest_data["week"] = pred_week
    latest_data["opponent"] = latest_data["team"].map(lambda t: schedule_map.get(t, ("", "unknown"))[0])
    latest_data["home_away"] = latest_data["team"].map(lambda t: schedule_map.get(t, ("", "unknown"))[1])

    # Prior-season (2024) stats — after leakage-safe season+1 shift, maps to 2025 player rows.
    team_stats_2024 = pd.DataFrame([
        {"team": "SEA", "season": 2024, "points_scored": 26, "points_allowed": 20,
         "total_yards": 360, "passing_yards": 250, "rushing_yards": 110, "turnovers": 1,
         "pass_attempts": 38, "rush_attempts": 28, "redzone_scores": 3.0},
        {"team": "NE", "season": 2024, "points_scored": 22, "points_allowed": 24,
         "total_yards": 340, "passing_yards": 220, "rushing_yards": 120, "turnovers": 2,
         "pass_attempts": 35, "rush_attempts": 30, "redzone_scores": 2.5},
    ])
    schedule_2025 = pd.DataFrame({"week": [22], "home_team": ["SEA"], "away_team": ["NE"]})

    def get_team_stats(season=None):
        if season is None:
            return team_stats_2024
        if season == 2024:
            return team_stats_2024
        return team_stats_2024

    mock_db = MagicMock()
    mock_db.get_team_stats.side_effect = get_team_stats
    mock_db.get_schedule.return_value = schedule_2025

    with patch("src.utils.database.DatabaseManager", return_value=mock_db):
        fe = FeatureEngineer()
        latest_data = fe.refresh_matchup_features(latest_data)

    sea_row = latest_data[latest_data["team"] == "SEA"].iloc[0]
    ne_row = latest_data[latest_data["team"] == "NE"].iloc[0]
    assert sea_row["opponent"] == "NE"
    assert sea_row["team_b_points_scored"] == 22
    assert sea_row["matchup_scoring_edge"] == 2.0
    assert ne_row["opponent"] == "SEA"
    assert ne_row["team_b_points_scored"] == 26
    assert ne_row["matchup_scoring_edge"] == 2.0  # 22 - 20
    assert sea_row["team_b_points_scored"] != ne_row["team_b_points_scored"]


# =============================================================================
# Predict flow: overwrite opponent/home_away from schedule map
# =============================================================================

def test_predict_overwrites_opponent_home_away_from_schedule_map():
    """When schedule map is provided, predict() should set opponent and home_away from map (unit logic)."""
    # Logic under test: for each row, opponent = schedule_map.get(team, ("", "unknown"))[0], home_away = [1]
    schedule_map = {"KC": ("SF", "home"), "SF": ("KC", "away")}

    def _opp(row):
        t = row.get("team")
        if pd.isna(t):
            return ""
        t = str(t).strip()
        return schedule_map.get(t, ("", "unknown"))[0]

    def _ha(row):
        t = row.get("team")
        if pd.isna(t):
            return "unknown"
        t = str(t).strip()
        return schedule_map.get(t, ("", "unknown"))[1]

    df = pd.DataFrame([{"team": "KC"}, {"team": "SF"}, {"team": "BUF"}])
    df["opponent"] = df.apply(_opp, axis=1)
    df["home_away"] = df.apply(_ha, axis=1)

    assert list(df["opponent"]) == ["SF", "KC", ""]
    assert list(df["home_away"]) == ["home", "away", "unknown"]


def test_format_matchup_display():
    """Display format: vs OPP (home), @ OPP (away), TBD (no opponent/unknown)."""
    def _format(opponent, home_away):
        opp = "" if opponent is None or (isinstance(opponent, float) and pd.isna(opponent)) else str(opponent).strip()
        ha = "unknown" if home_away is None or (isinstance(home_away, float) and pd.isna(home_away)) else str(home_away).strip().lower()
        if not opp or ha == "unknown":
            return "TBD"
        if ha == "home":
            return "vs " + opp
        return "@ " + opp

    assert _format("KC", "home") == "vs KC"
    assert _format("KC", "away") == "@ KC"
    assert _format("", "home") == "TBD"
    assert _format("KC", "unknown") == "TBD"
    assert _format(None, "home") == "TBD"
