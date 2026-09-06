"""The roster load replaces the whole table, so what it fetches matters.

Loading only the upcoming season would delete 2019-2025 and silently break
every backtest that refits step 8, which is the failure this guards.
"""
import importlib.util

import pytest

from config.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "ingest_rosters.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("ingest_rosters", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_narrow_request_keeps_the_seasons_already_stored(mod):
    assert mod.seasons_to_fetch([2026], [2019, 2020, 2021]) == [
        2019, 2020, 2021, 2026]


def test_an_empty_table_fetches_exactly_what_was_asked_for(mod):
    assert mod.seasons_to_fetch(range(2019, 2022), []) == [2019, 2020, 2021]


def test_overlap_is_not_duplicated(mod):
    assert mod.seasons_to_fetch([2025, 2026], [2025]) == [2025, 2026]
