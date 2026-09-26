"""Leakage-safety invariants for src/models/team_hierarchical/features.py."""
import pandas as pd
import numpy as np
import pytest

from src.models.team_allocation.features import SHARE_COLS, VOLUME_COLS
from src.models.team_hierarchical.features import ID_COLS, feature_columns, join_slot_share_rows


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


def _join_inputs():
    shares = pd.DataFrame([_row(), dict(_row(), player_id="p2")]).drop(
        columns=["slot", "slot_rank", "depth_chart_rank", "team_week_id"])
    slots = pd.DataFrame([_row()])[["player_id", "season", "week", "team", "slot", "slot_rank", "depth_chart_rank", "snap_share_s2d"]]
    slots["snap_share_s2d"] = 0.7
    return shares, slots


def test_exact_roster_join_audits_capped_rows_and_preserves_snap_sources():
    shares, slots = _join_inputs()
    panel, coverage = join_slot_share_rows(shares, slots, "rushing_yards")
    assert len(panel) == 1 and coverage["excluded_rows"] == 1
    assert coverage["excluded_keys"][0]["player_id"] == "p2"
    assert panel.snap_share_s2d.iloc[0] == 0.5
    assert panel.roster_snap_share_s2d.iloc[0] == 0.7
    assert not any(c.endswith(("_x", "_y")) for c in panel)
    assert panel.share_of_team_rushing_yards.iloc[0] == 0.25  # never renormalize capped rows
    assert "roster_snap_share_s2d" in feature_columns(panel)


@pytest.mark.parametrize("problem", ["team", "duplicate_share", "duplicate_slot", "null_key", "position", "slot"])
def test_bad_roster_identity_rejected(problem):
    shares, slots = _join_inputs()
    if problem == "team":
        slots.loc[0, "team"] = "WRONG"
    elif problem == "duplicate_share":
        shares = pd.concat([shares, shares.iloc[:1]], ignore_index=True)
    elif problem == "duplicate_slot":
        slots = pd.concat([slots, slots.assign(player_id="p2")], ignore_index=True)
    elif problem == "null_key":
        shares.loc[0, "player_id"] = None
    elif problem == "position":
        shares.loc[0, "position"] = "WR"
    else:
        slots.loc[0, "slot"] = "RB2"
    with pytest.raises(ValueError):
        join_slot_share_rows(shares, slots, "rushing_yards")


@pytest.mark.parametrize("rank", [np.nan, np.inf, -1, 1.5])
def test_invalid_slot_rank_rejected(rank):
    shares, slots = _join_inputs()
    slots["slot_rank"] = rank
    with pytest.raises(ValueError, match="slot_rank"):
        join_slot_share_rows(shares, slots, "rushing_yards")


def test_expanded_raw_opportunity_columns_are_not_features():
    from src.models.team_allocation.features import OPPORTUNITY_COLS
    df = pd.DataFrame([_row()])
    for col in OPPORTUNITY_COLS:
        df[col] = df[f"team_{col}"] = df[f"share_of_team_{col}"] = 0.1
    assert set(OPPORTUNITY_COLS).isdisjoint(feature_columns(df))


def test_known_leakage_column_rejected():
    df = pd.DataFrame([_row()])
    df["fantasy_points_ppr"] = 10
    with pytest.raises(ValueError):
        feature_columns(df)
