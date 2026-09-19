"""Undrafted rookies must reach the draft board's identity set too.

Until 2026-09-18, _load_rookie_identities sourced only draft_picks_v2, so
an undrafted rookie the season model already had a real Step 8 projection
for (confirmed: a genuine 2026 UDFA cold-start candidate was present in
the projection table and absent from the board) stayed invisible on
players_{POS}.json regardless. _load_undrafted_rookie_identities mirrors
the same population `preseason_features._cold_start_rows_incoming` and
`NFLPredictor._drafted_rookie_stub_rows` use: a roster row, no
draft_picks_v2 record ever, no history before the season, years_exp <= 1.

DB-backed (sqlite3.connect(DB_PATH) directly, like its sibling functions
in this module -- config.settings.DB_PATH is what gets redirected).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import config.settings as settings
from src.utils.database import DatabaseManager
from generate_draft_data import _load_rookie_identities, _load_undrafted_rookie_identities


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "board_rookies.db"
    monkeypatch.setattr(settings, "DB_PATH", path)
    d = DatabaseManager(path)
    with d._get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS rosters "
                     "(player_id TEXT, position TEXT, team TEXT, season INTEGER, years_exp REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS combine_data_v2 "
                     "(pfr_id TEXT, player_name TEXT)")
        conn.executemany(
            "INSERT INTO players (player_id, name, position) VALUES (?,?,?)",
            [("udfa_named", "U. Named", "WR"),
             ("old_draftee", "Old Draftee", "RB")],
            # "udfa_no_name" is deliberately absent from `players` -- a real
            # 2026 case (Rivaldo Fairweather, 00-0040046): on a roster with
            # a rookie-eligible years_exp but no `players` row yet at all.
        )
        conn.executemany(
            "INSERT INTO draft_picks_v2 (player_id, position, draft_season, "
            "draft_round, draft_pick) VALUES (?,?,?,?,?)",
            [("old_draftee", "RB", 2019, 6, 190)],
        )
        conn.executemany(
            "INSERT INTO rosters (player_id, position, team, season, years_exp) VALUES (?,?,?,?,?)",
            [("udfa_named", "WR", "KC", 2026, 0),
             ("udfa_no_name", "TE", "ARI", 2026, 1),      # a real 2026 case: no players.name yet
             ("old_draftee", "RB", "MIA", 2026, 6),       # drafted years ago, on the 2026 roster
             ("camp_body", "RB", "DAL", 2026, 3)],        # long tenure, never drafted, not a rookie
        )
        conn.commit()
    return d


def test_a_named_undrafted_rookie_is_included(db):
    out = _load_undrafted_rookie_identities(2026)
    assert set(out["player_id"]) == {"udfa_named"}
    row = out.set_index("player_id").loc["udfa_named"]
    assert row["team"] == "KC"
    assert row["draft_id"] == "udfa_named", \
        "already a real GSIS id -- no PFR/GSIS resolution needed, unlike the drafted half"


def test_a_player_with_no_name_on_file_is_excluded_rather_than_shown_blank(db):
    """A board row with no name is worse than no row -- same philosophy as
    the drafted half dropping a pick none of its three name sources can
    resolve."""
    out = _load_undrafted_rookie_identities(2026)
    assert "udfa_no_name" not in set(out["player_id"])


def test_drafted_years_ago_is_not_this_years_undrafted_class(db):
    out = _load_undrafted_rookie_identities(2026)
    assert "old_draftee" not in set(out["player_id"])


def test_long_tenured_never_drafted_player_is_excluded(db):
    out = _load_undrafted_rookie_identities(2026)
    assert "camp_body" not in set(out["player_id"])


def test_combined_identities_includes_the_undrafted_half(db):
    out = _load_rookie_identities(2026)
    assert "udfa_named" in set(out["player_id"])


def test_combined_identities_has_no_duplicate_players(db):
    out = _load_rookie_identities(2026)
    assert out["player_id"].is_unique
