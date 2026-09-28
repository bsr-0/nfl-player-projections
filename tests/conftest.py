"""Shared test fixtures.

AUDIT_REPORT.md #19: a test run overwrote the committed
data/data_availability_cache.json (DataManager.check_data_availability()
writes it on every call). No test has a legitimate reason to mutate that
file, so this redirects it to a per-test tmp_path unconditionally --
unlike DB_PATH, there's no test that intentionally needs the real cache
file's committed content, so no opt-out marker is needed here.

DB_PATH itself is deliberately NOT redirected globally: several tests are
real read-only integration tests against the actual historical DB (guarded
individually with `skipif(not DB_PATH.exists())`), and a blanket redirect
would silently turn them into false-pass tests against an empty fixture
DB instead of the state the audit describes (an empty DB literally being
created on disk). Per-file guards stay the source of truth for that half
of the finding.

That half is now handled at session end (see `_remove_database_created_by_the_session`):
fifteen tests across seven files construct a real DatabaseManager(), which
creates an empty data/nfl_data.db when none exists. The `skipif` guards are
evaluated at collection, so the SECOND run in a fresh checkout collected with
the file present, stopped skipping, and reported five real-DB tests as
failures (2026-09-28).
"""
import sqlite3

import pytest

from config.settings import DB_PATH

from src.utils.data_manager import DataManager


@pytest.fixture(autouse=True)
def _isolate_data_availability_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(DataManager, "CACHE_FILE", tmp_path / "data_availability_cache.json")


def _holds_no_rows(path) -> bool:
    """True only when `path` is a readable SQLite file with no rows anywhere.
    Any doubt returns False: an unreadable file is never deleted."""
    try:
        if path.stat().st_size == 0:
            return True
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = [name for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
            return not any(conn.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone()
                           for name in tables)
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return False


@pytest.fixture(scope="session", autouse=True)
def _remove_database_created_by_the_session():
    """Leave no empty database behind for the next run to trip over.

    Acts only if the database did not exist when the session started AND
    holds no rows now, so a real database (present at the start) and one a
    test genuinely populated are both left alone. Behaviour during the run
    is unchanged.
    """
    existed = DB_PATH.exists()
    yield
    if existed or not DB_PATH.exists() or not _holds_no_rows(DB_PATH):
        return
    for suffix in ("", "-shm", "-wal"):
        leftover = DB_PATH.with_name(DB_PATH.name + suffix)
        if leftover.exists():
            leftover.unlink()
