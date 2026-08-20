"""Position must come from rosters, and a stat line must never redefine it.

The bug (GAPS.md 2026-08-20): `aggregate_passing_stats` stamps
position='QB' on anyone with a pass attempt. One trick-play pass by
Christian McCaffrey, Derrick Henry or Cooper Kupp reached `players` via
INSERT OR REPLACE, and `_infer_position` then read it back out of that same
table on the next ingest -- so the corruption fed itself. 11.4% of the QB
training population were not quarterbacks.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(tmp_path / "t.db")
    yield d


def _position(d, pid):
    with d._get_connection() as conn:
        row = conn.execute("SELECT position FROM players WHERE player_id=?", (pid,)).fetchone()
    return row[0] if row else None


def _col(d, pid, col):
    with d._get_connection() as conn:
        row = conn.execute(f"SELECT {col} FROM players WHERE player_id=?", (pid,)).fetchone()
    return row[0] if row else None


def test_untrusted_position_cannot_overwrite_a_known_one(db):
    """The core regression: McCaffrey throws one pass, stays an RB."""
    db.insert_player({"player_id": "p1", "name": "C.McCaffrey", "position": "RB"})
    db.insert_player({"player_id": "p1", "name": "C.McCaffrey", "position": "QB"},
                     trust_position=False)
    assert _position(db, "p1") == "RB"


def test_untrusted_position_may_still_fill_a_blank(db):
    """Deriving a position is better than having none at all."""
    db.insert_player({"player_id": "p2", "name": "Unknown", "position": ""})
    db.insert_player({"player_id": "p2", "name": "Unknown", "position": "WR"},
                     trust_position=False)
    assert _position(db, "p2") == "WR"


def test_trusted_position_does_overwrite(db):
    """A reported position from the weekly feed is allowed to correct."""
    db.insert_player({"player_id": "p3", "name": "X", "position": "QB"})
    db.insert_player({"player_id": "p3", "name": "X", "position": "TE"})
    assert _position(db, "p3") == "TE"


def test_insert_does_not_wipe_fields_the_caller_omitted(db):
    """INSERT OR REPLACE rewrote the whole row, so every weekly ingest
    NULLed birth_date and college -- college was NULL for all 2,985
    players as a result."""
    db.insert_player({"player_id": "p4", "name": "X", "position": "WR",
                      "birth_date": "1995-03-02", "college": "Stanford"})
    db.insert_player({"player_id": "p4", "name": "X", "position": "WR"})
    assert _col(db, "p4", "birth_date") == "1995-03-02"
    assert _col(db, "p4", "college") == "Stanford"


def test_insert_preserves_created_at(db):
    """created_at was being reset on every ingest, which made a re-ingest
    look like a fresh player and hid the real write history."""
    db.insert_player({"player_id": "p5", "name": "X", "position": "WR"})
    first = _col(db, "p5", "created_at")
    db.insert_player({"player_id": "p5", "name": "X", "position": "WR"})
    assert _col(db, "p5", "created_at") == first


def test_empty_name_does_not_erase_a_real_one(db):
    db.insert_player({"player_id": "p6", "name": "Real Name", "position": "TE"})
    db.insert_player({"player_id": "p6", "name": "", "position": "TE"})
    assert _col(db, "p6", "name") == "Real Name"


def test_position_lookup_prefers_rosters_over_the_players_table(monkeypatch):
    """The fallback that made the corruption self-sustaining."""
    from src.data.pbp_stats_aggregator import PBPStatsAggregator
    monkeypatch.setattr(PBPStatsAggregator, "_PLAYERS_POSITION_CACHE",
                        {"00-0033280": "RB"})
    row = pd.Series({"player_id": "00-0033280", "position": np.nan,
                     "passing_attempts": 1, "rushing_attempts": 20,
                     "targets": 6, "receiving_yards": 45})
    assert PBPStatsAggregator()._infer_position(row) == "RB"


def test_single_trick_play_pass_does_not_infer_qb(monkeypatch):
    """With no lookup at all, the heuristic must still not call a
    20-carry rusher a quarterback because he threw once."""
    from src.data.pbp_stats_aggregator import PBPStatsAggregator
    monkeypatch.setattr(PBPStatsAggregator, "_PLAYERS_POSITION_CACHE", {})
    row = pd.Series({"player_id": "unknown", "position": np.nan,
                     "passing_attempts": 1, "rushing_attempts": 20,
                     "targets": 6, "receiving_yards": 45})
    assert PBPStatsAggregator()._infer_position(row) == "RB"


def test_real_quarterback_still_infers_qb(monkeypatch):
    from src.data.pbp_stats_aggregator import PBPStatsAggregator
    monkeypatch.setattr(PBPStatsAggregator, "_PLAYERS_POSITION_CACHE", {})
    row = pd.Series({"player_id": "unknown", "position": np.nan,
                     "passing_attempts": 35, "rushing_attempts": 3,
                     "targets": 0, "receiving_yards": 0})
    assert PBPStatsAggregator()._infer_position(row) == "QB"


def test_quality_gate_catches_a_reintroduced_mismatch(tmp_path, monkeypatch):
    """If this ever regresses, the gate must fail rather than train on it."""
    from src.data import quality_gates
    d = DatabaseManager(tmp_path / "g.db")
    for pid, pos in [("a", "QB"), ("b", "RB"), ("c", "WR")]:
        d.insert_player({"player_id": pid, "name": pid, "position": pos})
    monkeypatch.setattr(DatabaseManager, "get_authoritative_player_positions",
                        lambda self: {"a": "RB", "b": "RB", "c": "WR"})
    out = quality_gates.check_position_integrity(db_path=tmp_path / "g.db")
    assert out["passed"] is False
    assert out["mismatched"] == 1
    assert out["examples"][0]["stored"] == "QB"


def test_quality_gate_passes_on_clean_data(tmp_path, monkeypatch):
    from src.data import quality_gates
    d = DatabaseManager(tmp_path / "g2.db")
    for pid, pos in [("a", "RB"), ("b", "WR")]:
        d.insert_player({"player_id": pid, "name": pid, "position": pos})
    monkeypatch.setattr(DatabaseManager, "get_authoritative_player_positions",
                        lambda self: {"a": "RB", "b": "WR"})
    assert quality_gates.check_position_integrity(db_path=tmp_path / "g2.db")["passed"]


def test_fullback_is_not_flagged_as_a_mismatched_running_back(tmp_path, monkeypatch):
    """The ingest pipeline folds FB into RB, so the gate must too."""
    from src.data import quality_gates
    d = DatabaseManager(tmp_path / "g3.db")
    d.insert_player({"player_id": "a", "name": "a", "position": "RB"})
    monkeypatch.setattr(DatabaseManager, "get_authoritative_player_positions",
                        lambda self: {"a": "FB"})
    assert quality_gates.check_position_integrity(db_path=tmp_path / "g3.db")["passed"]
