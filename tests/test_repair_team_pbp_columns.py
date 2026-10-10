import numpy as np
import pandas as pd

from scripts.repair_team_pbp_columns import COLS, KEY, plan_updates


def _frame(rows):
    df = pd.DataFrame(rows, columns=KEY + ["pace_sec_per_play", "points_scored"])
    for c in COLS:
        if c not in df.columns:
            df[c] = 1.0
    return df


def test_only_null_cells_of_played_weeks_are_filled():
    stored = _frame([("KC", 2026, 2, np.nan, np.nan), ("KC", 2026, 3, 28.0, 21.0), ("KC", 2026, 5, np.nan, np.nan)])
    derived = _frame([("KC", 2026, 2, 27.5, 24.0), ("KC", 2026, 3, 99.0, 99.0), ("KC", 2026, 5, 30.0, 10.0)])
    plan = plan_updates(stored, derived, last_week=4)
    assert sorted(zip(plan.week, plan.column, plan.value)) == [(2, "pace_sec_per_play", 27.5), (2, "points_scored", 24.0)]


def test_a_missing_derived_value_and_absent_rows_change_nothing():
    stored = _frame([("KC", 2026, 2, np.nan, np.nan)])
    derived = _frame([("KC", 2026, 2, np.nan, np.nan), ("BUF", 2026, 2, 27.0, 20.0)])
    assert plan_updates(stored, derived, last_week=4).empty
