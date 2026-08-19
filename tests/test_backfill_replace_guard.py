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


def test_seasons_start_at_the_shared_floor(mod):
    assert mod.MIN_SEASON == 2013
    assert mod.SEASONS[0] == 2013


def test_upper_bound_is_the_nfl_season_not_the_calendar_year(mod):
    """datetime.now().year names an unplayed season for most of the year."""
    from config.settings import CURRENT_NFL_SEASON

    assert mod.SEASONS[-1] == CURRENT_NFL_SEASON


def test_dataset_without_its_own_floor_uses_the_global_one(mod):
    assert mod.seasons_for("qbr")[0] == mod.MIN_SEASON


def test_dataset_with_a_later_upstream_floor_is_clamped(mod):
    """NGS genuinely begins in 2016; asking for 2013 returns nothing."""
    assert mod.seasons_for("ngs")[0] == 2016
    assert mod.seasons_for("ngs")[-1] == mod.SEASONS[-1]


def test_every_declared_floor_is_within_the_global_range(mod):
    """A floor above the current season would yield an empty pull."""
    for dataset, floor in mod.DATASET_MIN_SEASON.items():
        assert mod.seasons_for(dataset), f"{dataset} resolves to no seasons"
        assert floor >= mod.MIN_SEASON, f"{dataset} floor is below MIN_SEASON"


def test_classic_depth_chart_range_stops_before_the_schema_change(mod):
    seasons = [s for s in mod.seasons_for("depth_charts")
               if s < mod.DEPTH_CHART_NEW_SCHEMA_SEASON]

    assert seasons[-1] == 2024
    assert mod.DEPTH_CHART_NEW_SCHEMA_SEASON not in seasons
