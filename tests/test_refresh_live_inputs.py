import numpy as np
import pandas as pd

from scripts.refresh_live_inputs import compare, new_rows, seasonal_pfr_frame  # noqa: F401

KEY = ["season", "week", "pid"]


def _f(rows):
    return pd.DataFrame(rows, columns=KEY + ["x", "name"])


def test_new_rows_returns_only_absent_keys_and_never_duplicates():
    stored = _f([(2026, 1, "a", 1.0, "A")])[KEY]
    fetched = _f([(2026, 1, "a", 9.0, "A"), (2026, 2, "a", 2.0, "A"), (2026, 2, "a", 2.0, "A"), (2026, 1, "b", 3.0, "B")])
    out = new_rows(stored, fetched, KEY)
    assert sorted(zip(out.week, out.pid)) == [(1, "b"), (2, "a")]


def test_new_rows_treats_a_missing_key_part_as_a_value():
    stored = pd.DataFrame({"season": [2026], "week": [np.nan], "pid": ["a"]})
    fetched = pd.DataFrame({"season": [2026, 2026], "week": [np.nan, 1.0], "pid": ["a", "a"], "x": [1, 2]})
    assert new_rows(stored, fetched, KEY).week.tolist() == [1.0]


def test_identical_data_has_no_revisions_or_null_mismatches():
    d = _f([(2026, 1, "a", 1.0, "A"), (2026, 1, "b", np.nan, "B")])
    r = compare(d, d.copy(), KEY)
    assert r["revised"] == {} and r["null_mismatch"] == {} and r["shared"] == 2


def test_a_changed_value_is_a_revision_and_a_one_sided_missing_is_a_mapping_error():
    stored = _f([(2026, 1, "a", 1.0, "A"), (2026, 1, "b", 2.0, "B"), (2026, 1, "c", 3.0, "C")])
    fetched = _f([(2026, 1, "a", 1.5, "A"), (2026, 1, "b", np.nan, "B"), (2026, 1, "c", 3.0, "C")])
    r = compare(stored, fetched, KEY)
    assert r["revised"] == {"x": 1} and r["null_mismatch"] == {"x": 1}


def test_rows_on_only_one_side_are_counted():
    stored = _f([(2026, 1, "a", 1.0, "A"), (2026, 1, "old", 1.0, "O")])
    fetched = _f([(2026, 1, "a", 1.0, "A"), (2026, 1, "new", 1.0, "N")])
    r = compare(stored, fetched, KEY)
    assert (r["shared"], r["only_stored"], r["only_fetched"]) == (1, 1, 1)


def test_only_played_weeks_are_kept_and_week_zero_is_dropped():
    import sqlite3
    from scripts.refresh_live_inputs import played_weeks_only
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE player_weekly_stats (season INT, week INT)")
    con.executemany("INSERT INTO player_weekly_stats VALUES (2026, ?)", [(1,), (2,), (4,)])
    df = pd.DataFrame({"week": [0, 1, 2, 3, 4, 5], "x": range(6)})
    assert played_weeks_only(con, df, 2026).week.tolist() == [1, 2, 3, 4]
    assert played_weeks_only(con, df, 2031).empty
    assert played_weeks_only(con, df.drop(columns="week"), 2026).equals(df.drop(columns="week"))
