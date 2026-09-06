"""Tests for the 2025 depth-chart schema bridge.

The value of these is mostly in pinning down the judgment calls: which
snapshot represents a week, how ranks deeper than 3 are handled, and that
eliminated clubs stop appearing in the postseason.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backfill_depth_charts.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("backfill_depth_charts", SCRIPT)
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
    out = mod._to_target_schema(picked, 2025)

    assert out["depth_team"].tolist() == ["1", "2", "3", "3", "3"]
    assert out["depth_team"].map(type).eq(str).all()


def test_target_schema_matches_table_columns(mod):
    picked = pd.DataFrame([{**_snapshot("2025-09-03T07:00:00Z"), "week": 1}])
    out = mod._to_target_schema(picked, 2025)

    assert list(out.columns) == mod.COLUMNS
    # The season is an argument now, not a module constant -- hardcoding one
    # is what left every season after 2025 with no depth charts at all.
    assert out["season"].iloc[0] == 2025
    assert out["club_code"].iloc[0] == "KC"


def test_name_is_split_and_multiword_surnames_survive(mod):
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z"), "player_name": "Amon-Ra St. Brown", "week": 1},
    ])
    out = mod._to_target_schema(picked, 2025)

    assert out["first_name"].iloc[0] == "Amon-Ra"
    assert out["last_name"].iloc[0] == "St. Brown"
    assert out["full_name"].iloc[0] == "Amon-Ra St. Brown"


def test_slot_labels_normalize_to_standard_positions(mod):
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z", pos_abb=abb), "week": 1}
        for abb in ("LT", "RCB", "LDE", "SLB", "PK")
    ])
    out = mod._to_target_schema(picked, 2025)

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
    out = mod._to_target_schema(picked, 2025)

    assert out["position"].tolist() == ["WR", "WR"]
    assert out["depth_position"].tolist() == ["WR", "KR"]


def test_returner_with_no_other_slot_stays_unresolved(mod):
    """Better an honest NULL than a guessed position."""
    picked = pd.DataFrame([
        {**_snapshot("2025-09-03T07:00:00Z", pos_abb="PR"), "week": 1},
    ])
    out = mod._to_target_schema(picked, 2025)

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


def test_a_preseason_snapshot_belongs_to_week_one(mod):
    """Before kickoff every snapshot precedes every week's first game.

    Keyed by snapshot, the last iteration used to win, so a season with no
    games played had its whole chart stamped week 18 and a week-1 consumer
    found nothing.
    """
    raw = pd.DataFrame({"dt": ["2026-09-05T11:00:00Z", "2026-09-06T11:00:00Z"],
                        "gsis_id": ["00-1", "00-2"]})
    weeks = pd.DataFrame({
        "week": [1, 2, 18],
        "first_game": pd.to_datetime(["2026-09-09", "2026-09-16", "2027-01-03"])})

    picked = mod._assign_snapshots(raw, weeks)

    assert set(picked["week"]) == {1}
    # The latest snapshot before kickoff, not the earliest.
    assert picked["gsis_id"].tolist() == ["00-2"]


def test_each_played_week_keeps_its_own_snapshot(mod):
    """The normal in-season case must be unchanged: one chart per week."""
    raw = pd.DataFrame({"dt": ["2026-09-08T11:00:00Z", "2026-09-15T11:00:00Z"],
                        "gsis_id": ["00-1", "00-2"]})
    weeks = pd.DataFrame({
        "week": [1, 2],
        "first_game": pd.to_datetime(["2026-09-09", "2026-09-16"])})

    picked = mod._assign_snapshots(raw, weeks)

    assert dict(zip(picked["gsis_id"], picked["week"])) == {"00-1": 1, "00-2": 2}
