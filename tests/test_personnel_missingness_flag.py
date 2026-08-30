"""Guards for PRESERVE_PERSONNEL_MISSINGNESS.

This flag read INERT twice while being built, in two different ways, and both
failures were invisible in the headline output -- the pipeline ran, the columns
existed, the dtypes were right, and the NaN counts were byte-identical with the
flag on and off. Same signature as this session's empty-donor-pool and
inert-preseason-mode bugs.

The two failures, encoded below:

  1. Only `utilization_score`'s blanket fill was exempted. Two fillers sit
     between the raw table and the model, so `_impute_missing` median-filled
     the column afterwards regardless.
  2. Only the BASE columns were exempted. The model never sees
     `team_pct_12_personnel`; it sees `team_pct_12_personnel_roll3_mean`, a
     different name, which was still filled.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as settings
from src.features.feature_engineering import (
    _STRUCTURALLY_MISSING,
    _structurally_missing_cols,
)
from src.features.utilization_score import (
    MISSINGNESS_PRESERVED_COLS,
    PERSONNEL_MISSINGNESS_COLS,
    missingness_preserved_cols,
)


@pytest.fixture(autouse=True)
def _isolate_history_flag(monkeypatch):
    """Pin the OTHER missingness flags OFF for this module.

    `_structurally_missing_cols()` and `missingness_preserved_cols()` union
    several flags' column sets. Since the history flag was flipped to default ON
    (2026-08-28), leaving it at its default made these assertions depend on a
    flag they are not testing -- `test_off_matches_historical_behaviour` failed
    with the history rolling columns present. The personnel flag is what is
    under test here, so the others are held fixed rather than the expectation
    being widened.

    PRESERVE_PRIOR_WEEK_MISSINGNESS joined the same union on 2026-08-29, also
    defaulting ON, and broke the same assertion for the same reason. Pinned here
    for the same reason; it has its own coverage in
    tests/test_prior_week_missingness_flag.py.
    """
    monkeypatch.setattr(settings, "PRESERVE_HISTORY_MISSINGNESS", False)
    monkeypatch.setattr(settings, "PRESERVE_PRIOR_WEEK_MISSINGNESS", False)


@pytest.fixture
def flag(monkeypatch):
    def _set(value):
        monkeypatch.setattr(settings, "PRESERVE_PERSONNEL_MISSINGNESS", value)
    return _set


class TestFlagIsNotInert:
    def test_off_matches_historical_behaviour(self, flag):
        """Default OFF must be byte-identical to pre-flag behaviour: every
        prior result was produced this way."""
        flag(False)
        assert missingness_preserved_cols() == MISSINGNESS_PRESERVED_COLS
        assert _structurally_missing_cols() == _STRUCTURALLY_MISSING

    def test_on_actually_changes_both_fillers(self, flag):
        """Failure 1: exempting one filler leaves the other to fill anyway."""
        flag(True)
        util = missingness_preserved_cols()
        impute = _structurally_missing_cols()
        assert util != MISSINGNESS_PRESERVED_COLS, "utilization_score filler not exempted"
        assert impute != _STRUCTURALLY_MISSING, "_impute_missing filler not exempted"
        for col in PERSONNEL_MISSINGNESS_COLS:
            assert col in util
            assert col in impute

    def test_on_exempts_the_roll3_derivatives(self, flag):
        """Failure 2: the model-facing feature is the roll3 mean, not the base
        column, so exempting only base names leaves the feature filled."""
        flag(True)
        impute = _structurally_missing_cols()
        for col in PERSONNEL_MISSINGNESS_COLS:
            assert f"{col}_roll3_mean" in impute, (
                f"{col}_roll3_mean is what the model actually consumes and is "
                f"still being imputed")

    def test_snap_exemptions_survive_both_states(self, flag):
        """The pre-existing snap exemption must not be disturbed either way."""
        for state in (False, True):
            flag(state)
            assert {"snap_count", "team_snaps", "snap_share", "snap_share_pct"} <= (
                missingness_preserved_cols())
            assert "team_motion_rate" in _structurally_missing_cols()

    def test_resolved_at_call_time_not_import_time(self, flag):
        """Captured-at-import would make the toggle unsettable -- GAPS.md
        7.7/7.8 records that monkeypatching module constants here is
        unreliable, which is why both helpers read the setting live."""
        flag(False)
        before = _structurally_missing_cols()
        flag(True)
        assert _structurally_missing_cols() != before
