"""Leakage-safety invariants for src/models/team_allocation/features.py.

Every raw current-week column (volume counts, team totals, same-week
shares) must be excluded from feature_columns(); only the lagged
_s2d/_roll3 history and is_cold_start may survive.
"""
import pandas as pd
import pytest

from src.models.team_allocation.features import SHARE_COLS, VOLUME_COLS, feature_columns


def _row():
    row = {"player_id": "p1", "season": 2020, "week": 1, "team": "AAA", "position": "WR"}
    for c in VOLUME_COLS:
        row[c] = 5.0
        row[f"team_{c}"] = 20.0
        row[f"share_of_team_{c}"] = 0.25
        row[f"share_of_team_{c}_s2d"] = 0.20
        row[f"share_of_team_{c}_roll3"] = 0.22
    row["is_cold_start"] = 0
    return row


def test_only_lagged_and_cold_start_columns_survive():
    df = pd.DataFrame([_row()])
    cols = set(feature_columns(df))
    expected = {f"share_of_team_{c}_s2d" for c in VOLUME_COLS}
    expected |= {f"share_of_team_{c}_roll3" for c in VOLUME_COLS}
    expected.add("is_cold_start")
    assert cols == expected


def test_raw_same_week_columns_never_survive():
    df = pd.DataFrame([_row()])
    cols = feature_columns(df)
    forbidden = set(VOLUME_COLS) | set(SHARE_COLS) | {f"team_{c}" for c in VOLUME_COLS}
    assert forbidden.isdisjoint(cols)
    assert {"player_id", "season", "week", "team", "position"}.isdisjoint(cols)


def test_unclassified_column_raises():
    df = pd.DataFrame([_row()])
    df["some_new_unaudited_feature"] = 1.0
    with pytest.raises(ValueError, match="Unclassified team-allocation feature columns"):
        feature_columns(df)
