"""The session-end cleanup may only ever delete an EMPTY database.

tests/conftest.py removes an empty data/nfl_data.db that a test session
itself created (fifteen tests construct a real DatabaseManager()). Deleting
the wrong file would destroy the production database, so the rule -- delete
only what has no rows, and nothing that is unreadable -- is pinned here.
"""
import sqlite3

from tests.conftest import _holds_no_rows


def _db(path, rows=0, table="player_weekly_stats"):
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE {table} (player_id TEXT)")
        conn.executemany(f"INSERT INTO {table} VALUES (?)", [(f"p{i}",) for i in range(rows)])
    return path


def test_zero_byte_file_is_empty(tmp_path):
    path = tmp_path / "a.db"
    path.touch()
    assert _holds_no_rows(path)


def test_schema_only_database_is_empty(tmp_path):
    assert _holds_no_rows(_db(tmp_path / "a.db"))


def test_one_row_in_any_table_protects_the_database(tmp_path):
    path = _db(tmp_path / "a.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE schedule (game_id TEXT)")
        conn.execute("INSERT INTO schedule VALUES ('g')")
    assert not _holds_no_rows(path)


def test_populated_database_is_protected(tmp_path):
    assert not _holds_no_rows(_db(tmp_path / "a.db", rows=3))


def test_unreadable_or_missing_files_are_never_called_empty(tmp_path):
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"this is not a sqlite database" * 50)
    assert not _holds_no_rows(garbage)
    assert not _holds_no_rows(tmp_path / "missing.db")
