import numpy as np
import pandas as pd

from scripts.repair_pbp_advanced_columns import plan_updates

KEY = ["player_id", "season", "week"]


def _frame(rows):
    return pd.DataFrame(rows, columns=KEY + ["recv_epa", "redzone_targets"])


def test_defaults_are_filled_and_nonzero_values_are_kept():
    stored = _frame([("a", 2026, 1, 0.0, 0), ("b", 2026, 1, 2.0, 0), ("c", 2026, 1, np.nan, 0)])
    derived = _frame([("a", 2026, 1, 1.5, 3), ("b", 2026, 1, 9.0, 1), ("c", 2026, 1, -0.5, 0)])
    plan = plan_updates(stored, derived)
    got = {(r.player_id, r.column): r.value for r in plan.itertuples()}
    assert got == {("a", "recv_epa"): 1.5, ("a", "redzone_targets"): 3.0,
                   ("b", "redzone_targets"): 1.0, ("c", "recv_epa"): -0.5}


def test_rows_absent_from_the_stored_frame_are_never_created():
    stored = _frame([("a", 2026, 1, 0.0, 0)])
    derived = _frame([("a", 2026, 1, 1.5, 0), ("z", 2026, 1, 4.0, 4)])
    assert set(plan_updates(stored, derived).player_id) == {"a"}


def test_a_missing_derived_value_and_an_equal_value_change_nothing():
    stored = _frame([("a", 2026, 1, 0.0, 0), ("b", 2026, 1, 0.0, 0)])
    derived = _frame([("a", 2026, 1, np.nan, 0), ("b", 2026, 1, 0.0, 0)])
    assert plan_updates(stored, derived).empty


def test_applying_the_plan_makes_a_second_plan_empty():
    stored = _frame([("a", 2026, 1, 0.0, 0), ("b", 2026, 1, 0.0, 0)])
    derived = _frame([("a", 2026, 1, 1.5, 2), ("b", 2026, 1, -1.0, 0)])
    plan = plan_updates(stored, derived)
    fixed = stored.copy().set_index(KEY)
    for r in plan.itertuples():
        fixed.loc[(r.player_id, r.season, r.week), r.column] = r.value
    assert plan_updates(fixed.reset_index(), derived).empty
