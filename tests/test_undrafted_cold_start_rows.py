"""Undrafted rookies must reach Step 8's cold-start population too.

Until 2026-09-18, _cold_start_rows_incoming (formerly _cold_start_rows_
from_draft) sourced its rookie class from draft_picks_v2 only, so an
undrafted player -- ~39% of everyone who reaches the league -- never
got a season projection before his first game, drafted-class rookie or
not. career_static_by_player already resolves an unmatched player_id to
is_undrafted=1 and the UNDRAFTED_ROUND/UNDRAFTED_PICK sentinel on its own;
this only had to widen the POPULATION, not touch that resolution.

DB-backed (creates the tables it needs on a tmp_path DB); no monkeypatching
of the query itself, so a real SQL mistake would show up here.
"""
import pandas as pd
import pytest

import config.settings as settings
from src.models.preseason_features import _cold_start_rows_incoming
from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path, monkeypatch):
    # _cold_start_rows_incoming (like its sibling _inference_season_teams)
    # opens its own connection via config.settings.DB_PATH rather than the
    # `db` object passed in -- an existing pattern in this module, not
    # something this test can route around, so DB_PATH is what gets
    # redirected to the temp DB.
    path = tmp_path / "cold_start.db"
    monkeypatch.setattr(settings, "DB_PATH", path)
    d = DatabaseManager(path)
    with d._get_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rosters (player_id TEXT, position TEXT, "
            "season INTEGER, years_exp REAL)")
        conn.executemany(
            "INSERT INTO draft_picks_v2 (player_id, position, draft_season, "
            "draft_round, draft_pick) VALUES (?,?,?,?,?)",
            [("drafted", "WR", 2026, 3, 80),
             # Drafted in an EARLIER season -- must not count as this year's
             # incoming class even though he'd otherwise look like a rookie.
             ("old_draftee", "RB", 2020, 5, 160)],
        )
        conn.executemany(
            "INSERT INTO rosters (player_id, position, season, years_exp) VALUES (?,?,?,?)",
            [("udfa_rookie", "RB", 2026, 0),
             ("udfa_year1", "WR", 2026, 1),
             ("camp_body", "TE", 2026, 3),      # long roster tenure, never played
             ("old_draftee", "RB", 2026, 6),    # on the 2026 roster, drafted in 2020
             ("veteran", "QB", 2026, 8)],
        )
        conn.commit()
    return d


def _history(rows):
    return pd.DataFrame(rows, columns=["player_id", "season"])


def test_drafted_and_undrafted_both_appear(db):
    # veteran/old_draftee have history predating 2026; everyone else does not.
    hist = _history([("veteran", 2019), ("old_draftee", 2020)])
    out = _cold_start_rows_incoming(db, 2026, hist)
    ids = set(out["player_id"])
    assert "drafted" in ids
    assert "udfa_rookie" in ids
    assert "udfa_year1" in ids, "years_exp<=1 must not exclude a one-year case"


def test_long_tenured_camp_body_is_excluded(db):
    hist = _history([("veteran", 2019), ("old_draftee", 2020)])
    out = _cold_start_rows_incoming(db, 2026, hist)
    assert "camp_body" not in set(out["player_id"])


def test_veteran_and_old_draftee_are_excluded(db):
    hist = _history([("veteran", 2019), ("old_draftee", 2020)])
    out = _cold_start_rows_incoming(db, 2026, hist)
    ids = set(out["player_id"])
    assert "veteran" not in ids
    assert "old_draftee" not in ids, \
        "drafted in 2020, not part of the 2026 incoming class regardless of roster presence"


def test_undrafted_players_carry_no_draft_columns_for_the_static_merge_to_resolve(db):
    """_cold_start_rows_incoming itself must not fabricate draft_round/pick
    -- that resolution belongs to career_static_by_player's unmatched-join
    fallback, which only fires on a genuinely absent column/value."""
    hist = _history([])
    out = _cold_start_rows_incoming(db, 2026, hist)
    row = out.set_index("player_id").loc["udfa_rookie"]
    assert "draft_round" not in out.columns or pd.isna(row.get("draft_round"))


def test_a_player_with_history_before_target_is_never_cold_start(db):
    hist = _history([("udfa_rookie", 2025)])
    out = _cold_start_rows_incoming(db, 2026, hist)
    assert "udfa_rookie" not in set(out["player_id"])


def test_empty_when_nothing_qualifies(db):
    with db._get_connection() as conn:
        conn.execute("DELETE FROM draft_picks_v2")
        conn.execute("DELETE FROM rosters")
        conn.commit()
    out = _cold_start_rows_incoming(db, 2026, _history([]))
    assert out.empty
