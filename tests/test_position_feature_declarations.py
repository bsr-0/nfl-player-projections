"""POSITION_FEATURES must declare features that actually get built.

Seven declarations per position referenced columns no builder produces --
`snap_share_roll3` among them, a name that never existed. `get_position_features`
filtered them out silently, so walk_forward_multiweek.py trained on fewer
features than its config implied and nothing said so.
"""
import warnings

import pytest

from config.settings import CAUSAL_FEATURES
from src.models.train_position_models import POSITION_FEATURES, get_position_features


def _declared(position):
    groups = POSITION_FEATURES[position]
    return (groups.get("primary", []) + groups.get("derived", [])
            + groups.get("rolling", []) + groups.get("team_context", []))


def test_no_declaration_references_the_phantom_name():
    """`snap_share_roll3` is never constructed anywhere. The real column is
    snap_share_pct_roll3_mean."""
    for position in POSITION_FEATURES:
        assert "snap_share_roll3" not in _declared(position)


def test_rolling_declarations_use_the_builders_naming_convention():
    """The roll loop emits `{col}_roll3_mean`; a bare `_roll3` silently
    matches nothing."""
    for position, groups in POSITION_FEATURES.items():
        for feature in groups.get("rolling", []):
            assert feature.endswith("_mean"), (
                f"{position}: {feature!r} does not follow the _roll3_mean "
                f"convention and will be dropped silently"
            )


def test_absent_features_warn_instead_of_vanishing():
    with pytest.warns(RuntimeWarning, match="declared feature"):
        resolved = get_position_features("WR", ["targets", "receptions"])
    assert resolved == ["targets", "receptions"]


def test_no_warning_when_everything_resolves():
    declared = _declared("WR")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = get_position_features("WR", declared)
    assert resolved == declared


def test_causal_features_is_the_authoritative_production_list():
    """POSITION_FEATURES is consumed only by walk_forward_multiweek.py.
    Production trains on CAUSAL_FEATURES. This pins that distinction so the
    two lists are not mistaken for each other again."""
    for position in ("RB", "WR", "TE"):
        assert CAUSAL_FEATURES.get(position), f"{position} missing from CAUSAL_FEATURES"

    # the snap feature production actually trains on
    for position in ("RB", "WR", "TE"):
        snap = [f for f in CAUSAL_FEATURES[position] if "snap" in f]
        assert "snap_share_pct_roll3_mean" in snap
    assert not [f for f in CAUSAL_FEATURES.get("QB", []) if "snap" in f], (
        "QB has no snap features; a new one needs its own justification"
    )
