"""Depth-chart frame preparation for the historical (2013-2019) backfill.

The one real judgment call: exact duplicates collapse, conflicting ranks
survive for MIN to resolve at load time.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
          / "backfill_historical_coverage.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("hist", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _row(week=1, gsis="00-0000001", depth_team="1", depth_position="WR",
         formation="Offense"):
    return {"season": 2013, "club_code": "DEN", "week": week,
            "depth_team": depth_team, "last_name": "Decker",
            "first_name": "Eric", "football_name": "Eric", "position": "WR",
            "jersey_number": 87, "gsis_id": gsis,
            "depth_position": depth_position, "full_name": "Eric Decker",
            "formation": formation, "game_type": "REG"}


def test_rows_differing_only_by_dropped_columns_collapse(mod):
    """`formation` isn't stored, so base + sub-package listings arrive
    identical."""
    raw = pd.DataFrame([_row(formation="Offense"), _row(formation="Nickel")])

    assert len(mod.prepare_depth_chart_frame(raw)) == 1


def test_conflicting_depth_team_is_preserved(mod):
    """A player listed WR1 and WR2 in the same week is a real conflict; MIN
    resolves it downstream, not here."""
    raw = pd.DataFrame([_row(depth_team="1"), _row(depth_team="2")])
    out = mod.prepare_depth_chart_frame(raw)

    assert len(out) == 2
    assert sorted(out.depth_team) == ["1", "2"]


def test_rows_without_week_are_dropped(mod):
    """The Super Bowl bye (SBBYE) carries no week and is unusable."""
    raw = pd.DataFrame([_row(week=1), _row(week=None)])
    out = mod.prepare_depth_chart_frame(raw)

    assert out.week.tolist() == [1]


def test_rows_without_gsis_id_are_dropped(mod):
    raw = pd.DataFrame([_row(gsis="00-0000001"), _row(gsis=None)])

    assert len(mod.prepare_depth_chart_frame(raw)) == 1


def test_output_is_restricted_to_stored_columns(mod):
    out = mod.prepare_depth_chart_frame(pd.DataFrame([_row()]))

    assert list(out.columns) == mod.DEPTH_CHART_COLUMNS
    assert "formation" not in out.columns
    assert "game_type" not in out.columns


def test_postseason_weeks_pass_through_untouched(mod):
    """2013 encodes 18=WC .. 21=SB, already matching the stored convention."""
    raw = pd.DataFrame([_row(week=w) for w in (17, 18, 19, 20, 21)])
    out = mod.prepare_depth_chart_frame(raw)

    assert out.week.tolist() == [17, 18, 19, 20, 21]
