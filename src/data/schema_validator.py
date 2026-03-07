"""
Schema validation for NFL data ingestion.

Validates that incoming DataFrames match expected schemas before processing,
preventing silent failures when upstream data sources change column names,
types, or structure.

Per Agent Directive V7, Section 19: Data pipeline must include schema validation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""
    pass


# ---------------------------------------------------------------------------
# Expected schemas for each data source
# ---------------------------------------------------------------------------

# Weekly player stats: columns that MUST exist
WEEKLY_REQUIRED_COLUMNS: Set[str] = {
    "player_id",
    "season",
    "week",
    "position",
}

# Weekly player stats: columns expected but not fatal if missing
WEEKLY_EXPECTED_COLUMNS: Set[str] = {
    "name",
    "team",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "targets",
    "fumbles_lost",
}

# Schedule: required columns
SCHEDULE_REQUIRED_COLUMNS: Set[str] = {
    "season",
    "week",
    "home_team",
    "away_team",
}

# Snap counts: required columns
SNAP_COUNT_REQUIRED_COLUMNS: Set[str] = {
    "player",
    "position",
}

# Column type expectations (column_name -> expected dtype kind)
# 'i' = integer, 'f' = float, 'O'/'U' = string/object
COLUMN_TYPE_EXPECTATIONS: Dict[str, str] = {
    "season": "numeric",
    "week": "numeric",
    "passing_yards": "numeric",
    "passing_tds": "numeric",
    "rushing_yards": "numeric",
    "rushing_tds": "numeric",
    "receptions": "numeric",
    "receiving_yards": "numeric",
    "receiving_tds": "numeric",
    "targets": "numeric",
    "fumbles_lost": "numeric",
    "fantasy_points": "numeric",
}


def validate_weekly_data(df: pd.DataFrame, strict: bool = False) -> List[str]:
    """Validate weekly player stats DataFrame against expected schema.

    Args:
        df: DataFrame to validate.
        strict: If True, raise SchemaValidationError on failures.
                If False, return list of warning messages.

    Returns:
        List of validation warning/error messages (empty if all OK).
    """
    issues: List[str] = []

    if df.empty:
        issues.append("WARNING: Weekly data DataFrame is empty")
        if strict:
            raise SchemaValidationError("Weekly data DataFrame is empty")
        return issues

    # Check required columns
    missing_required = WEEKLY_REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        msg = f"CRITICAL: Missing required columns: {sorted(missing_required)}"
        issues.append(msg)
        if strict:
            raise SchemaValidationError(msg)

    # Check expected columns
    missing_expected = WEEKLY_EXPECTED_COLUMNS - set(df.columns)
    if missing_expected:
        issues.append(
            f"WARNING: Missing expected columns (data may be incomplete): "
            f"{sorted(missing_expected)}"
        )

    # Check for unexpected NaN rates in critical columns
    for col in ["player_id", "season", "week", "position"]:
        if col in df.columns:
            nan_rate = df[col].isna().mean()
            if nan_rate > 0.05:
                issues.append(
                    f"WARNING: Column '{col}' has {nan_rate:.1%} null values"
                )

    # Check column types
    for col, expected_type in COLUMN_TYPE_EXPECTATIONS.items():
        if col not in df.columns:
            continue
        if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(
                f"WARNING: Column '{col}' expected numeric but got {df[col].dtype}"
            )

    # Check season range sanity
    if "season" in df.columns:
        min_season = df["season"].min()
        max_season = df["season"].max()
        if min_season < 1999 or max_season > 2030:
            issues.append(
                f"WARNING: Season range [{min_season}, {max_season}] looks suspicious"
            )

    # Check week range sanity
    if "week" in df.columns:
        max_week = df["week"].max()
        if max_week > 22:
            issues.append(
                f"WARNING: Max week {max_week} exceeds expected range (1-22)"
            )

    # Check for duplicate rows
    if all(c in df.columns for c in ["player_id", "season", "week"]):
        dup_count = df.duplicated(subset=["player_id", "season", "week"]).sum()
        if dup_count > 0:
            issues.append(
                f"WARNING: {dup_count} duplicate (player_id, season, week) rows"
            )

    # Log issues
    for issue in issues:
        if issue.startswith("CRITICAL"):
            logger.error(issue)
        else:
            logger.warning(issue)

    return issues


def validate_schedule_data(df: pd.DataFrame, strict: bool = False) -> List[str]:
    """Validate schedule DataFrame against expected schema."""
    issues: List[str] = []

    if df.empty:
        issues.append("WARNING: Schedule DataFrame is empty")
        return issues

    missing_required = SCHEDULE_REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        msg = f"CRITICAL: Missing required schedule columns: {sorted(missing_required)}"
        issues.append(msg)
        if strict:
            raise SchemaValidationError(msg)

    return issues


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: Set[str],
    source_name: str = "data",
    strict: bool = False,
) -> List[str]:
    """Generic schema validation for any DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: Set of column names that must exist.
        source_name: Human-readable name for error messages.
        strict: If True, raise on critical failures.

    Returns:
        List of validation messages.
    """
    issues: List[str] = []

    if df.empty:
        issues.append(f"WARNING: {source_name} DataFrame is empty")
        return issues

    missing = required_columns - set(df.columns)
    if missing:
        msg = f"CRITICAL: {source_name} missing required columns: {sorted(missing)}"
        issues.append(msg)
        if strict:
            raise SchemaValidationError(msg)

    return issues
