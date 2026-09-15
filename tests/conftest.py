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
"""
import pytest

from src.utils.data_manager import DataManager


@pytest.fixture(autouse=True)
def _isolate_data_availability_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(DataManager, "CACHE_FILE", tmp_path / "data_availability_cache.json")
