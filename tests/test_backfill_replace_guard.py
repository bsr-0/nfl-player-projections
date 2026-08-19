"""The `replace` guard in backfill_all_data.py.

Every loader in that script drops and recreates its table while fetching
only SEASONS (2018+), so any season backfilled elsewhere used to vanish on
the next run. These pin the guard that now refuses.
"""
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backfill_all_data.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("backfill_all", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def _seed(conn, seasons):
    pd.DataFrame({"season": seasons, "v": range(len(seasons))}).to_sql(
        "t", conn, if_exists="replace", index=False)


def test_refuses_replace_that_would_drop_earlier_seasons(mod, conn):
    _seed(conn, [2013, 2018, 2019])
    incoming = pd.DataFrame({"season": [2018, 2019], "v": [1, 2]})

    with pytest.raises(RuntimeError, match=r"refusing to replace"):
        mod._assert_no_history_lost(incoming, "t", conn)


def test_error_names_the_seasons_at_risk(mod, conn):
    _seed(conn, [2013, 2014, 2020])
    incoming = pd.DataFrame({"season": [2020], "v": [1]})

    with pytest.raises(RuntimeError, match=r"\[2013, 2014\]"):
        mod._assert_no_history_lost(incoming, "t", conn)


def test_allows_replace_covering_everything_present(mod, conn):
    _seed(conn, [2018, 2019])
    incoming = pd.DataFrame({"season": [2018, 2019, 2020], "v": [1, 2, 3]})

    mod._assert_no_history_lost(incoming, "t", conn)  # must not raise


def test_missing_table_is_not_an_error(mod, conn):
    mod._assert_no_history_lost(pd.DataFrame({"season": [2020]}), "nope", conn)


def test_seasonless_frame_is_skipped(mod, conn):
    """Contracts and similar have no season column; nothing to compare."""
    _seed(conn, [2013])
    mod._assert_no_history_lost(pd.DataFrame({"player": ["x"]}), "t", conn)


def test_save_df_enforces_the_guard(mod, conn):
    """The guard has to be wired into the write path, not just importable."""
    _seed(conn, [2013, 2018])
    incoming = pd.DataFrame({"season": [2018], "v": [9]})

    with pytest.raises(RuntimeError, match=r"refusing to replace"):
        mod._save_df(incoming, "t", conn)

    # the original rows must still be there
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2


def test_save_df_still_writes_when_nothing_is_lost(mod, conn):
    _seed(conn, [2018])
    mod._save_df(pd.DataFrame({"season": [2018, 2019], "v": [1, 2]}), "t", conn)

    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
