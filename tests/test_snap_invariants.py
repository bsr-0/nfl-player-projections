"""Snap data must satisfy two physical invariants before it reaches the DB.

Both of these actually shipped, silently, into a validation season and were
found by hand months later (GAPS.md 2026-08-19):

1. `team_snaps` summed over players rather than counting team plays -- ~11
   offensive players each credited with the same ~60 snaps inflated it ~12x
   (2025 held avg 646, max 2112). This makes snap_share ~12x too SMALL, so
   no ratio bound catches it; only a plausibility band on the play count.
2. `snap_count` doubled for players appearing in the passing frame plus
   another (QBs who also rush), because snap_count was present on the frame
   before a duplicate-collapse that sums numeric columns. Shows up as
   snap_count > team_snaps.

Neither is caught by any model test -- they corrupt a feature, not an
output -- so they belong at the ingest boundary.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema_validator import (
    SchemaValidationError, TEAM_SNAPS_MAX, TEAM_SNAPS_MIN,
    _check_snap_invariants, validate_weekly_data,
)


def _frame(**over):
    base = dict(player_id=["p1"], season=[2025], week=[3], position=["QB"],
                name=["A B"], team=["KC"], snap_count=[60], team_snaps=[65])
    base.update(over)
    return pd.DataFrame(base)


class TestSnapCountNeverExceedsTeamSnaps:
    def test_clean_frame_passes(self):
        assert _check_snap_invariants(_frame()) == []

    def test_doubling_is_caught(self):
        """The real 2025 shape: Josh Allen wk13 stored 148 against a 74-play game."""
        issues = _check_snap_invariants(_frame(snap_count=[148], team_snaps=[74]))
        assert any("snap_count > team_snaps" in i for i in issues)
        assert all(i.startswith("CRITICAL") for i in issues)

    def test_equality_is_allowed(self):
        """An every-down QB legitimately plays 100% of snaps."""
        assert _check_snap_invariants(_frame(snap_count=[74], team_snaps=[74])) == []

    def test_unknown_team_snaps_is_not_flagged(self):
        """team_snaps == 0 means 'no snap data', not a violation."""
        assert _check_snap_invariants(_frame(snap_count=[0], team_snaps=[0])) == []


class TestTeamSnapsPlausibilityBand:
    def test_summed_inflation_is_caught(self):
        """The other real shape: 814 = sum over ~47 credited players."""
        issues = _check_snap_invariants(_frame(snap_count=[74], team_snaps=[814]))
        assert any("outside" in i for i in issues)

    def test_both_teams_summed_is_caught(self):
        """KC wk3 2025 held 1584 -- both teams' player snaps added together."""
        issues = _check_snap_invariants(_frame(snap_count=[148], team_snaps=[1584]))
        assert any("outside" in i for i in issues)

    def test_inflation_MASKS_the_doubling(self):
        """The two invariants are complementary, not redundant, and the real
        2025 data needed both in sequence.

        Josh Allen wk13 was stored as snap_count=148 against team_snaps=1628.
        148 < 1628, so the ratio check cannot fire -- the inflated denominator
        HID the doubled numerator. Only after team_snaps was corrected to 74
        did 148 > 74 become visible. That is precisely why the backfill's
        dry run surfaced 511 rows over 1.0 only once the denominator was
        recomputed: fixing one defect is what exposed the other.
        """
        as_stored = _check_snap_invariants(_frame(snap_count=[148], team_snaps=[1628]))
        assert not any("snap_count > team_snaps" in i for i in as_stored), (
            "ratio check must NOT fire while the denominator is inflated"
        )
        assert any("outside" in i for i in as_stored), "band check must fire"

        denominator_fixed = _check_snap_invariants(
            _frame(snap_count=[148], team_snaps=[74]))
        assert any("snap_count > team_snaps" in i for i in denominator_fixed), (
            "once the denominator is right, the doubling must surface"
        )

    @pytest.mark.parametrize("value", [TEAM_SNAPS_MIN, 65, TEAM_SNAPS_MAX])
    def test_plausible_values_pass(self, value):
        assert _check_snap_invariants(
            _frame(snap_count=[10], team_snaps=[value])) == []

    def test_band_covers_observed_range(self):
        """2018-2025 observed 34-100. The band must not fire on real data."""
        assert TEAM_SNAPS_MIN < 34 and TEAM_SNAPS_MAX > 100

    def test_missing_columns_are_not_an_error(self):
        assert _check_snap_invariants(pd.DataFrame({"player_id": ["p1"]})) == []


class TestWiredIntoTheWritePath:
    """validate_weekly_data(strict=True) runs immediately before the DB write,
    so the invariant must fire there rather than only in isolation."""

    def test_strict_validation_raises_on_doubling(self):
        with pytest.raises(SchemaValidationError, match="snap_count > team_snaps"):
            validate_weekly_data(_frame(snap_count=[148], team_snaps=[74]), strict=True)

    def test_strict_validation_raises_on_inflation(self):
        with pytest.raises(SchemaValidationError, match="team_snaps outside"):
            validate_weekly_data(_frame(snap_count=[74], team_snaps=[814]), strict=True)

    def test_non_strict_reports_without_raising(self):
        issues = validate_weekly_data(_frame(snap_count=[148], team_snaps=[74]))
        assert any("snap_count > team_snaps" in i for i in issues)

    def test_clean_frame_does_not_raise(self):
        validate_weekly_data(_frame(), strict=True)


class TestCollapseOrderingConstraint:
    """`aggregate_all_stats` sums every numeric column for rows sharing
    (player_id, season, week) -- which QBs do, appearing in both the passing
    and rushing frames. snap_count must therefore be attached AFTER that
    collapse. The ordering is load-bearing and nothing else enforces it."""

    def test_snap_merge_happens_after_the_duplicate_collapse(self):
        src = Path(__file__).parent.parent / "src/data/pbp_stats_aggregator.py"
        lines = src.read_text().split("\n")
        collapse = next(i for i, l in enumerate(lines) if "to_collapse = all_stats[" in l)
        merge = next(i for i, l in enumerate(lines) if "merge_with_snaps(all_stats)" in l)
        assert collapse < merge, (
            "merge_with_snaps must run AFTER the duplicate-collapse: the collapse "
            "sums numeric columns, so a snap_count present beforehand is doubled "
            "for every player appearing in two PBP frames (QBs who rush). This "
            "produced 563 corrupted rows in 2025."
        )

    def test_collapse_sums_numerics_as_assumed(self):
        """If the collapse ever stops summing, the ordering test above is
        guarding something that no longer exists -- fail loudly instead."""
        src = Path(__file__).parent.parent / "src/data/pbp_stats_aggregator.py"
        text = src.read_text()
        assert '_agg = {c: "sum" for c in _num}' in text
