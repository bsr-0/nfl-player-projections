"""Direct scoring, lineage, and target-game identity for the full-PPR comparison."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.full_ppr_head_to_head import (
    FULL_COMPONENTS, checked_full_truth, checked_production_fold,
    origin_to_target_map,
)
from src.utils.helpers import calculate_fantasy_points_df


def _inputs():
    key = ["player_id", "season", "week", "team", "position"]
    raw = pd.DataFrame([
        ["rookie", 2025, 1, "A", "QB", 0, -3, 2, 1, 0, 0, 40, 0, 1],
        ["returning", 2025, 1, "B", "WR", 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], columns=key + ["fold", "rushing_yards", "receiving_yards", "receptions",
                      "receiving_tds", "rushing_tds", "passing_yards", "passing_tds",
                      "interceptions"])
    raw["actual_ppr"] = calculate_fantasy_points_df(raw)
    stats = pd.DataFrame({"player_id": ["rookie"], "season": [2025], "week": [1],
                          "team": ["A"], "fumbles_lost": [1], "two_point_conversions": [1],
                          "fantasy_points": [float(raw.loc[0, "actual_ppr"])]})
    # One fumble and one two-point conversion cancel; the raw negative-yard
    # and off-position receiving/passing terms still matter.
    canonical = raw[key].copy()
    canonical["has_stats_row"] = [1, 0]
    canonical["has_snap_row"] = [1, 0]
    canonical["offense_snaps"] = [11, np.nan]
    canonical["fantasy_points"] = [stats.fantasy_points.iloc[0], np.nan]
    history = pd.DataFrame({"player_id": ["rookie", "returning"],
                            "first_canonical_season": [2025, 2020]})
    draft = pd.DataFrame({"player_id": ["rookie"], "draft_season": [2025]})
    return raw, stats, canonical, history, draft


def test_full_truth_scores_raw_ten_components_and_keeps_unknown_snaps():
    raw, stats, canonical, history, draft = _inputs()
    truth = checked_full_truth(raw, stats, canonical, history, draft)
    assert len(truth) == 2
    rookie = truth[truth.player_id.eq("rookie")].iloc[0]
    assert rookie.actual_full_ppr == rookie.actual_ppr
    assert rookie.actual_full_ppr == calculate_fantasy_points_df(
        pd.DataFrame([{**{name: rookie[name] for name in FULL_COMPONENTS}}])).iloc[0]
    assert bool(rookie.draft_rookie) and rookie.snap_segment == "nonzero"
    returning = truth[truth.player_id.eq("returning")].iloc[0]
    assert returning.actual_full_ppr == 0
    assert bool(returning.returning_player) and returning.snap_segment == "unknown"


@pytest.mark.parametrize("mutation", ["duplicate_stats", "missing_stats", "null_extra", "bad_canonical"])
def test_invalid_full_truth_fails(mutation):
    raw, stats, canonical, history, draft = _inputs()
    if mutation == "duplicate_stats":
        stats = pd.concat([stats, stats], ignore_index=True)
    elif mutation == "missing_stats":
        stats = stats.iloc[0:0]
    elif mutation == "null_extra":
        stats.loc[0, "fumbles_lost"] = np.nan
    else:
        canonical.loc[0, "fantasy_points"] = 99
    with pytest.raises(ValueError):
        checked_full_truth(raw, stats, canonical, history, draft)


def test_target_mapping_uses_next_observed_game_and_target_team():
    rows = []
    for week, team, points in [(1, "A", 2), (3, "A", 4), (6, "B", 6)]:
        row = {"player_id": "p", "season": 2025, "week": week, "team": team,
               "position": "WR", "fantasy_points": points}
        row.update({name: 0 for name in FULL_COMPONENTS})
        row["receptions"] = points
        rows.append(row)
    raw = pd.DataFrame(rows)
    mapping = origin_to_target_map(raw)
    assert mapping.loc[0, "target_week"] == 3
    assert mapping.loc[1, "target_week"] == 6
    assert mapping.loc[1, "target_team"] == "B"
    capture = pd.DataFrame({"player_id": ["p", "p"], "season": [2025, 2025],
                            "week": [1, 3], "team": ["A", "A"],
                            "position": ["WR", "WR"], "predicted_points": [3.0, 5.0],
                            "actual_points": [4.0, 6.0], "train_seasons": ["2023,2024"] * 2})
    export = checked_production_fold(capture, mapping)
    assert list(export.week) == [3, 6]
    assert list(export.team) == ["A", "B"]
    assert list(export.origin_week) == [1, 3]
    assert list(export.actual_ppr) == [4.0, 6.0]


@pytest.mark.parametrize("mutation", ["wrong_label", "missing_target", "leaked_fold", "duplicate_target"])
def test_bad_production_export_fails(mutation):
    rows = []
    for week in (1, 2, 3):
        row = {"player_id": "p", "season": 2025, "week": week,
               "team": "A", "position": "RB", "fantasy_points": float(week)}
        row.update({name: 0 for name in FULL_COMPONENTS})
        row["receptions"] = week
        rows.append(row)
    mapping = origin_to_target_map(pd.DataFrame(rows))
    capture = pd.DataFrame({"player_id": ["p", "p"], "season": [2025, 2025],
                            "week": [1, 2], "team": ["A", "A"],
                            "position": ["RB", "RB"], "predicted_points": [2.0, 3.0],
                            "actual_points": [2.0, 3.0], "train_seasons": ["2024"] * 2})
    if mutation == "wrong_label":
        capture.loc[0, "actual_points"] = 99
    elif mutation == "missing_target":
        capture.loc[1, "week"] = 3
    elif mutation == "leaked_fold":
        capture.loc[0, "train_seasons"] = "2024,2025"
    else:
        mapping.loc[1, "target_week"] = 2
    with pytest.raises(ValueError):
        checked_production_fold(capture, mapping)


def test_capture_carrying_its_own_target_columns_merges_and_is_cross_checked():
    rows = []
    for week in (1, 3, 6):
        row = {"player_id": "p", "season": 2025, "week": week, "team": "A",
               "position": "WR", "fantasy_points": float(week)}
        row.update({name: 0 for name in FULL_COMPONENTS})
        row["receptions"] = week
        rows.append(row)
    mapping = origin_to_target_map(pd.DataFrame(rows))
    capture = pd.DataFrame({"player_id": ["p", "p"], "season": [2025, 2025],
                            "week": [1, 3], "team": ["A", "A"], "position": ["WR", "WR"],
                            "predicted_points": [3.0, 5.0], "actual_points": [3.0, 6.0],
                            "train_seasons": ["2024"] * 2,
                            "target_week": [3, 6], "target_team": ["A", "A"]})
    export = checked_production_fold(capture, mapping)
    assert list(export.week) == [3, 6]
    capture.loc[0, "target_week"] = 4
    with pytest.raises(ValueError, match="disagrees"):
        checked_production_fold(capture, mapping)


def _winsor_fixture():
    rows = []
    for week in (1, 2, 3, 4):
        row = {"player_id": "p", "season": 2025, "week": week, "team": "A",
               "position": "WR", "fantasy_points": float(week * 10)}
        row.update({name: 0 for name in FULL_COMPONENTS})
        row["receptions"] = week * 10
        rows.append(row)
    mapping = origin_to_target_map(pd.DataFrame(rows))
    capture = pd.DataFrame({"player_id": ["p"] * 3, "season": [2025] * 3,
                            "week": [1, 2, 3], "team": ["A"] * 3, "position": ["WR"] * 3,
                            "predicted_points": [1.0, 2.0, 3.0],
                            "actual_points": [20.0, 30.0, 40.0],
                            "train_seasons": ["2024"] * 3})
    return mapping, capture


def test_clipped_capture_is_rejected():
    """Test targets are raw; a capped actual means the capture is not the label."""
    mapping, capture = _winsor_fixture()
    capture.loc[2, "actual_points"] = 35.0  # raw 40 capped at a training quantile
    with pytest.raises(ValueError, match="does not match"):
        checked_production_fold(capture, mapping)


def test_exact_capture_is_scored_on_raw_label():
    mapping, capture = _winsor_fixture()
    export = checked_production_fold(capture, mapping)
    assert list(export.actual_ppr) == [20.0, 30.0, 40.0]


def test_non_clip_mismatch_still_fails():
    mapping, capture = _winsor_fixture()
    capture.loc[1, "actual_points"] = 25.0
    with pytest.raises(ValueError, match="does not match"):
        checked_production_fold(capture, mapping)
