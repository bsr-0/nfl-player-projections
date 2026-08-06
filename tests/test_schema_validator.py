import pandas as pd
import pytest

from src.data.schema_validator import (
    SchemaValidationError,
    validate_schedule_data,
    validate_weekly_data,
)


def test_validate_weekly_data_rejects_invalid_position_in_strict_mode():
    df = pd.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2024],
            "week": [1],
            "position": ["FB"],
            "team": ["BUF"],
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_weekly_data(df, strict=True)


def test_validate_weekly_data_rejects_negative_stats_in_strict_mode():
    # `passing_yards` is deliberately excluded from this check (see
    # schema_validator.py's NEGATIVE_WARN_COLUMNS): sacks can legitimately
    # put single-game passing yards below zero, so it's warn-only, not a
    # critical error, even in strict mode. Use `receptions` instead, which
    # is in NEGATIVE_DISALLOWED_COLUMNS and has no legitimate negative
    # value. Test updated 2026-08-06 after finding the original assertion
    # failed against passing_yards for this reason, not a code regression.
    df = pd.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2024],
            "week": [1],
            "position": ["QB"],
            "receptions": [-1],
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_weekly_data(df, strict=True)


def test_validate_weekly_data_rejects_duplicate_key_rows_in_strict_mode():
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["RB", "RB"],
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_weekly_data(df, strict=True)


def test_validate_weekly_data_rejects_critical_null_threshold_in_strict_mode():
    df = pd.DataFrame(
        {
            "player_id": ["p1", None, None, None],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "position": ["QB", "QB", "QB", "QB"],
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_weekly_data(df, strict=True, critical_null_threshold=0.20)


def test_validate_schedule_data_rejects_invalid_postseason_week_in_strict_mode():
    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [25],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "game_type": ["POST"],
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_schedule_data(df, strict=True)


def test_validate_schedule_data_rejects_duplicate_games_in_strict_mode():
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "home_team": ["KC", "KC"],
            "away_team": ["BUF", "BUF"],
        }
    )

    with pytest.raises(SchemaValidationError):
        validate_schedule_data(df, strict=True)
