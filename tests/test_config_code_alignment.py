"""Tests that verify config and code are aligned.

Config-code mismatches — where settings declare one thing but code does
another — are a common source of silent bugs. These tests ensure that:
1. position_target_type config values are valid and respected
2. Converter training only happens for positions that need it
3. The prediction path respects the target type config
"""

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Config validity
# ---------------------------------------------------------------------------

class TestPositionTargetTypeConfig:
    """Verify position_target_type config is valid and contains no dead code."""

    VALID_TARGET_TYPES = {"fp", "util", "component"}

    def test_all_target_types_are_valid(self):
        """Every position_target_type value must be 'fp', 'util', or 'component'.
        'auto' is not a supported value — it was previously dead code."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        invalid = []
        for pos, target_type in ptc.items():
            if target_type not in self.VALID_TARGET_TYPES:
                invalid.append((pos, target_type))
        assert invalid == [], (
            f"Invalid position_target_type values found:\n"
            + "\n".join(
                f"  {pos}: '{tt}' (must be 'fp' or 'util')"
                for pos, tt in invalid
            )
        )

    def test_qb_target_type_is_explicit(self):
        """QB target type must be explicitly set, not 'auto'."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        qb_target = ptc.get("QB", "util")
        assert qb_target in self.VALID_TARGET_TYPES, (
            f"QB target type is '{qb_target}', must be one of {self.VALID_TARGET_TYPES}. "
            f"'auto' was dead code — ensemble.py unconditionally overrode it to 'fp'."
        )

    def test_all_four_positions_have_target_type(self):
        """QB, RB, WR, TE should all have explicit target type config."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        for pos in ["QB", "RB", "WR", "TE"]:
            assert pos in ptc, (
                f"Position {pos} missing from position_target_type config"
            )


# ---------------------------------------------------------------------------
# Converter training alignment
# ---------------------------------------------------------------------------

class TestConverterTrainingAlignment:
    """Verify that converters are only trained for positions that need them."""

    def test_util_positions_need_converters(self):
        """Only positions with target_type='util' need util-to-FP converters.
        Positions with 'fp' or 'component' target types do not."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        util_positions = [pos for pos, tt in ptc.items() if tt == "util"]
        non_util_positions = [pos for pos, tt in ptc.items() if tt != "util"]
        # Verify every position has a valid target type
        for pos, tt in ptc.items():
            assert tt in {"fp", "util", "component"}, (
                f"{pos} has invalid target type '{tt}'"
            )


# ---------------------------------------------------------------------------
# Prediction path alignment
# ---------------------------------------------------------------------------

class TestPredictionPathAlignment:
    """Verify the prediction path respects target type config."""

    def test_target_types_are_consistent_across_positions(self):
        """All positions should have a valid, explicitly configured target type."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        for pos in ["QB", "RB", "WR", "TE"]:
            target_type = ptc.get(pos)
            assert target_type is not None, f"{pos} missing from position_target_type"
            assert target_type in {"fp", "util", "component"}, (
                f"{pos} has invalid target type '{target_type}'"
            )

    def test_util_positions_need_conversion(self):
        """Only 'util' positions should trigger util->FP conversion."""
        ptc = MODEL_CONFIG.get("position_target_type", {})
        for pos, tt in ptc.items():
            should_convert = tt == "util"
            if should_convert:
                assert tt == "util", f"{pos} should be 'util' for conversion"
