"""Leakage-safety invariants for src/models/team_hierarchical/features.py."""
import pandas as pd
import pytest

from src.models.team_allocation.features import SHARE_COLS, VOLUME_COLS
from src.models.team_hierarchical.features import ID_COLS, feature_columns


def _row():
    row = {
        "player_id": "p1", "season": 2020, "week": 1, "team": "AAA", "position": "RB",
        "slot": "RB1", "slot_rank": 1, "depth_chart_rank": 1.0, "snap_share_s2d": 0.5,
        "team_week_id": "AAA_2020_1", "is_cold_start": 0,
    }
    for c in VOLUME_COLS:
        row[c] = 5.0
        row[f"team_{c}"] = 20.0
        row[f"share_of_team_{c}"] = 0.25
        row[f"share_of_team_{c}_s2d"] = 0.20
        row[f"share_of_team_{c}_roll3"] = 0.22
        row[f"team_{c}_s2d"] = 18.0
        row[f"team_{c}_roll3"] = 19.0
    return row


def test_roster_slot_columns_survive_as_features():
    df = pd.DataFrame([_row()])
    cols = set(feature_columns(df))
    for c in ("slot", "slot_rank", "depth_chart_rank", "snap_share_s2d"):
        assert c in cols


def test_lagged_share_and_team_total_columns_survive():
    df = pd.DataFrame([_row()])
    cols = set(feature_columns(df))
    for c in VOLUME_COLS:
        assert f"share_of_team_{c}_s2d" in cols
        assert f"share_of_team_{c}_roll3" in cols
        assert f"team_{c}_s2d" in cols
        assert f"team_{c}_roll3" in cols
    assert "is_cold_start" in cols


def test_same_week_raw_and_label_columns_never_survive():
    df = pd.DataFrame([_row()])
    cols = feature_columns(df)
    forbidden = set(VOLUME_COLS) | set(SHARE_COLS) | {f"team_{c}" for c in VOLUME_COLS}
    assert forbidden.isdisjoint(cols)


def test_id_and_group_columns_never_survive_as_features():
    df = pd.DataFrame([_row()])
    cols = feature_columns(df)
    assert set(ID_COLS).isdisjoint(cols)


def test_unclassified_column_raises():
    df = pd.DataFrame([_row()])
    df["some_new_unaudited_feature"] = 1.0
    with pytest.raises(ValueError, match="Unclassified team-hierarchical feature columns"):
        feature_columns(df)
