"""Tests for the 2025 depth-chart schema bridge.

The value of these is mostly in pinning down the judgment calls: which
snapshot represents a week, how ranks deeper than 3 are handled, and that
eliminated clubs stop appearing in the postseason.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backfill_depth_charts_2025.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("dc2025", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _snapshot(dt, team="KC", gsis="00-0000001", pos_abb="WR", rank=1):
    return {"dt": dt, "team": team, "gsis_id": gsis, "pos_abb": pos_abb,
            "pos_rank": rank, "player_name": "Test Player"}


def test_week_gets_last_snapshot_before_kickoff(mod):
    raw = pd.DataFrame([
        _snapshot("2025-09-01T07:00:00Z", rank=3),
        _snapshot("2025-09-03T07:00:00Z", rank=1),   # latest before kickoff
        _snapshot("2025-09-05T07:00:00Z", rank=2),   # after -- must not win
    ])
    weeks = pd.DataFrame({"week": [1], "first_game": [pd.Timestamp("2025-09-04")]})
    picked = mod._assign_snapshots(raw, weeks)

    assert len(picked) == 1
    assert picked["week"].iloc[0] == 1
    assert picked["pos_rank"].iloc[0] == 1


def test_snapshot_on_kickoff_day_is_excluded(mod):
    """game_time is date-only, so same-day snapshots are refused rather than
    assumed pre-kickoff. Conservative by design."""
    raw = pd.DataFrame([
        _snapshot("2025-09-02T07:00:00Z", rank=2),
        _snapshot("2025-09-04T07:00:00Z", rank=1),  # kickoff day
    ])
    weeks = pd.DataFrame({"week": [1], "first_game": [pd.Timestamp("2025-09-04")]})
    picked = mod._assign_snapshots(raw, weeks)

    assert picked["pos_rank"].iloc[0] == 2


def test_week_with_no_prior_snapshot_is_skipped(mod):
    raw = pd.DataFrame([_snapshot("2025-09-10T07:00:00Z")])
    weeks = pd.DataFrame({"week": [1], "first_game": [pd.Timestamp("2025-09-04")]})

    assert mod._assign_snapshots(raw, weeks).empty


def test_depth_team_clipped_to_old_feeds_vocabulary(mod):
    """The old feed only ever emits 1/2/3; an unclipped 7 would be a value
    the models have never seen."""
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z", rank=r), "week": 1}
        for r in (1, 2, 3, 4, 8)
    ])
    out = mod._to_target_schema(picked)

    assert out["depth_team"].tolist() == ["1", "2", "3", "3", "3"]
    assert out["depth_team"].map(type).eq(str).all()


def test_target_schema_matches_table_columns(mod):
    picked = pd.DataFrame([{**_snapshot("2025-09-03T07:00:00Z"), "week": 1}])
    out = mod._to_target_schema(picked)

    assert list(out.columns) == mod.COLUMNS
    assert out["season"].iloc[0] == mod.SEASON
    assert out["club_code"].iloc[0] == "KC"


def test_name_is_split_and_multiword_surnames_survive(mod):
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z"), "player_name": "Amon-Ra St. Brown", "week": 1},
    ])
    out = mod._to_target_schema(picked)

    assert out["first_name"].iloc[0] == "Amon-Ra"
    assert out["last_name"].iloc[0] == "St. Brown"
    assert out["full_name"].iloc[0] == "Amon-Ra St. Brown"


def test_slot_labels_normalize_to_standard_positions(mod):
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z", pos_abb=abb), "week": 1}
        for abb in ("LT", "RCB", "LDE", "SLB", "PK")
    ])
    out = mod._to_target_schema(picked)

    assert out["position"].tolist() == ["T", "CB", "DE", "OLB", "K"]
    # the fine-grained slot is preserved, matching the old depth_position
    assert out["depth_position"].tolist() == ["LT", "RCB", "LDE", "SLB", "PK"]


def test_returner_inherits_position_from_his_other_slot(mod):
    """KR/PR carry no position of their own; the old feed listed the player's
    real one."""
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z", pos_abb="WR", rank=2), "week": 1},
        {**_snapshot("2025-09-03T07:00:00Z", pos_abb="KR", rank=1), "week": 1},
    ])
    out = mod._to_target_schema(picked)

    assert out["position"].tolist() == ["WR", "WR"]
    assert out["depth_position"].tolist() == ["WR", "KR"]


def test_returner_with_no_other_slot_stays_unresolved(mod):
    """Better an honest NULL than a guessed position."""
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z", pos_abb="PR"), "week": 1},
    ])
    out = mod._to_target_schema(picked)

    assert pd.isna(out["position"].iloc[0])


def test_eliminated_clubs_dropped_only_in_postseason(mod):
    out = pd.DataFrame({
        "week": [18, 18, 19, 19],
        "club_code": ["KC", "CHI", "KC", "CHI"],  # CHI misses the playoffs
    })
    playing = pd.DataFrame({"week": [18, 18, 19], "team": ["KC", "CHI", "KC"]})
    kept = mod._drop_eliminated_teams(out, playing)

    # both survive week 18 (regular season keeps every club, byes included)
    assert sorted(kept[kept.week == 18].club_code) == ["CHI", "KC"]
    assert kept[kept.week == 19].club_code.tolist() == ["KC"]
