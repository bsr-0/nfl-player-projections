"""Schema and population contracts for the full-PPR share extension."""
import sqlite3

import pandas as pd

from scripts.build_team_week_player_shares import build_shares
from src.models.team_allocation.features import (
    ALL_SHARE_COLS,
    ALL_VOLUME_COLS,
    feature_columns,
    filter_population,
)


def test_full_ppr_targets_have_expected_populations():
    frame = pd.DataFrame({"position": ["QB", "RB", "WR", "TE"]})
    assert set(filter_population(frame, "receptions").position) == {"RB", "WR", "TE"}
    assert set(filter_population(frame, "receiving_tds").position) == {"RB", "WR", "TE"}
    assert set(filter_population(frame, "passing_yards").position) == {"QB"}
    assert set(filter_population(frame, "passing_tds").position) == {"QB"}
    assert set(filter_population(frame, "interceptions").position) == {"QB"}


def test_full_ppr_raw_columns_are_not_features():
    row = {"player_id": "p", "season": 2024, "week": 1, "team": "AAA", "position": "WR", "is_cold_start": 0}
    for col in ALL_VOLUME_COLS:
        row[col] = 1.0
        row[f"team_{col}"] = 2.0
        row[f"share_of_team_{col}"] = 0.5
        row[f"share_of_team_{col}_s2d"] = 0.4
        row[f"share_of_team_{col}_roll3"] = 0.45
    cols = set(feature_columns(pd.DataFrame([row])))
    assert set(ALL_VOLUME_COLS).isdisjoint(cols)
    assert {f"team_{c}" for c in ALL_VOLUME_COLS}.isdisjoint(cols)
    assert set(ALL_SHARE_COLS).isdisjoint(cols)
    assert "share_of_team_receptions_s2d" not in cols
    assert "share_of_team_passing_tds_roll3" not in cols
    full_cols = set(feature_columns(pd.DataFrame([row]), include_full_ppr=True))
    assert "share_of_team_receptions_s2d" in full_cols
    assert "share_of_team_passing_tds_roll3" in full_cols


def test_builder_emits_full_ppr_columns_and_compositional_shares():
    con = sqlite3.connect(":memory:")
    pd.DataFrame([
        {"player_id": "q", "season": 2024, "week": 1, "team": "AAA", "position": "QB"},
        {"player_id": "r", "season": 2024, "week": 1, "team": "AAA", "position": "RB"},
        {"player_id": "w", "season": 2024, "week": 1, "team": "AAA", "position": "WR"},
        {"player_id": "q", "season": 2024, "week": 2, "team": "AAA", "position": "QB"},
        {"player_id": "r", "season": 2024, "week": 2, "team": "AAA", "position": "RB"},
        {"player_id": "w", "season": 2024, "week": 2, "team": "AAA", "position": "WR"},
    ]).to_sql("canonical_player_weeks", con, index=False)
    stats = []
    for week in (1, 2):
        stats.extend([
            {"player_id": "q", "season": 2024, "week": week, "targets": 0, "rushing_attempts": 3, "receiving_yards": 0, "rushing_yards": 12, "receptions": 0, "receiving_tds": 0, "rushing_tds": 1, "passing_yards": 250, "passing_tds": 2, "interceptions": 1},
            {"player_id": "r", "season": 2024, "week": week, "targets": 3, "rushing_attempts": 15, "receiving_yards": 20, "rushing_yards": 70, "receptions": 2, "receiving_tds": 0, "rushing_tds": 0, "passing_yards": 0, "passing_tds": 0, "interceptions": 0},
            {"player_id": "w", "season": 2024, "week": week, "targets": 7, "rushing_attempts": 0, "receiving_yards": 90, "rushing_yards": 0, "receptions": 5, "receiving_tds": 1, "rushing_tds": 0, "passing_yards": 0, "passing_tds": 0, "interceptions": 0},
        ])
    pd.DataFrame(stats).to_sql("player_weekly_stats", con, index=False)
    panel = build_shares(con, 2024, 2024)
    assert set(ALL_VOLUME_COLS).issubset(panel.columns)
    assert set(ALL_SHARE_COLS).issubset(panel.columns)
    assert panel.loc[panel.position == "QB", "share_of_team_passing_yards"].eq(1.0).all()
    assert panel.groupby(["team", "season", "week"])["share_of_team_receptions"].sum().eq(1.0).all()
