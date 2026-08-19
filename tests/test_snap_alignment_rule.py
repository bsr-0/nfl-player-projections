"""The three-way rule that aligns player_weekly_stats with snap_counts.

Each case exists because getting it wrong loses real information:
overwriting blindly discards values the source can't regenerate, and
NULLing blindly destroys them outright.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
          / "backfill_snap_counts_to_pws.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("align", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _resolve(mod, current, authoritative):
    return mod.resolve(pd.Series(current, dtype="object"),
                       pd.Series(authoritative, dtype="object"))


def test_authoritative_value_wins(mod):
    out = _resolve(mod, [0, 12, 99], [40, 40, 40])
    assert out.tolist() == [40, 40, 40]


def test_authoritative_zero_is_a_real_measurement(mod):
    """A source row of 0 snaps means the player dressed and did not play.
    It must land as 0, not NULL, or the hurdle model loses its negatives."""
    out = _resolve(mod, [0, 25], [0, 0])
    assert out.tolist() == [0, 0]
    assert out.notna().all()


def test_absent_source_over_a_zero_becomes_null(mod):
    """The placeholder zero was never a measurement."""
    out = _resolve(mod, [0], [None])
    assert out.isna().all()


def test_absent_source_preserves_an_existing_positive(mod):
    """1,831 real 2018+ values have no id-match; NULLing them would destroy
    data this script cannot regenerate."""
    out = _resolve(mod, [37], [None])
    assert out.tolist() == [37]


def test_absent_source_over_an_existing_null_stays_null(mod):
    out = _resolve(mod, [None], [None])
    assert out.isna().all()


def test_result_is_nullable_integer(mod):
    """A plain int column silently coerces NA back to a number."""
    assert _resolve(mod, [0], [None]).dtype == "Int64"


def test_all_four_states_together(mod):
    current = [0, 0, 50, None]
    auth = [40, None, None, None]
    out = _resolve(mod, current, auth)

    assert out[0] == 40      # overwritten from source
    assert pd.isna(out[1])   # placeholder zero -> unknown
    assert out[2] == 50      # preserved, source silent
    assert pd.isna(out[3])   # unknown stays unknown


def test_share_is_null_when_either_side_unknown(mod):
    """Mirrors how the script derives snap_share after resolving."""
    snaps = _resolve(mod, [10, 0, 10], [10, 0, None])
    team = _resolve(mod, [50, 50, 0], [50, 50, None])
    share = (snaps / team).where(team > 0)

    assert share[0] == pytest.approx(0.2)
    assert share[1] == 0.0          # known zero of a known total
    assert pd.isna(share[2])        # unknown team total
