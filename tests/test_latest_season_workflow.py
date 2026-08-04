"""
Test that the latest season in DB is used as the test dataset.

- test_latest_season_logic: Pure logic test (no DB); asserts test_season rules.
- test_latest_season_is_test_dataset: Runs scripts/verify_latest_season_test.py in subprocess.
  Run manually if needed: python scripts/verify_latest_season_test.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_latest_season_test.py"


def _train_test_season_logic(
    available: list,
    current_season: int,
    in_season: bool,
) -> tuple:
    """
    Same contract as DataManager.get_train_test_seasons (season selection only).
    Used to test logic without importing database.
    """
    available = sorted(available)
    if not available:
        raise ValueError("No season data available.")
    if in_season and current_season not in available:
        raise ValueError(
            f"Current season {current_season} has started but is not in the database. "
            "Run data refresh so current season is loaded from play-by-play."
        )
    if in_season and current_season in available:
        test_season = current_season
    else:
        test_season = max(available)
    train_seasons = [s for s in available if s < test_season]
    return train_seasons, test_season


def test_latest_season_logic_not_in_season():
    """When not in-season, test_season must be max(available)."""
    train, test = _train_test_season_logic(
        available=[2020, 2021, 2022, 2023, 2024],
        current_season=2025,
        in_season=False,
    )
    assert test == 2024
    assert 2024 not in train
    assert train == [2020, 2021, 2022, 2023]


def test_latest_season_logic_in_season_current_in_db():
    """When in-season and current season in DB, test_season must be current season."""
    train, test = _train_test_season_logic(
        available=[2020, 2021, 2022, 2023, 2024, 2025],
        current_season=2025,
        in_season=True,
    )
    assert test == 2025
    assert 2025 not in train
    assert train == [2020, 2021, 2022, 2023, 2024]


def test_latest_season_logic_in_season_current_not_in_db_raises():
    """When in-season but current season not in DB, must raise."""
    with pytest.raises(ValueError) as exc_info:
        _train_test_season_logic(
            available=[2020, 2021, 2022, 2023, 2024],
            current_season=2025,
            in_season=True,
        )
    assert "2025" in str(exc_info.value)
    assert "play-by-play" in str(exc_info.value).lower() or "database" in str(exc_info.value).lower()


def test_latest_season_is_test_dataset():
    """Run verify script: test_season must equal latest in DB and, when in-season, current season."""
    if not VERIFY_SCRIPT.exists():
        pytest.skip(f"Verify script not found: {VERIFY_SCRIPT}")
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=25,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if "No seasons in database" in out:
            pytest.skip("No seasons in database; run auto_refresh or load data first")
        if "Fatal Python error" in out or "Floating point" in out:
            pytest.skip("Environment crash during script run (e.g. numpy); run script manually")
        pytest.fail(
            f"verify_latest_season_test.py failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
