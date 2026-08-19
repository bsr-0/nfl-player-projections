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

VALID_POSITIONS: Set[str] = {"QB", "RB", "WR", "TE", "K", "DST"}

# These columns legitimately have negative values in NFL data (tackles behind LOS
# produce negative rushing/receiving yards; avoid treating them as critical errors).
NEGATIVE_WARN_COLUMNS: Set[str] = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
}

NEGATIVE_DISALLOWED_COLUMNS: Set[str] = {
    "passing_attempts",
    "passing_completions",
    "passing_tds",
    "rushing_attempts",
    "rushing_tds",
    "receptions",
    "receiving_tds",
    "targets",
    "fumbles_lost",
    "interceptions",
    "games_played",
    "pass_plays",
    "rush_plays",
    "recv_targets",
    "neutral_targets",
    "neutral_rushes",
    "third_down_targets",
    "short_yardage_rushes",
    "redzone_targets",
    "goal_line_touches",
    "two_minute_targets",
    "high_leverage_touches",
}


def _collect_critical_issues(issues: List[str]) -> List[str]:
    return [issue for issue in issues if issue.startswith("CRITICAL")]


def _raise_if_strict_critical(issues: List[str], strict: bool) -> None:
    if strict:
        critical_issues = _collect_critical_issues(issues)
        if critical_issues:
            raise SchemaValidationError("; ".join(critical_issues))


def _is_valid_week_for_phase(week: int, phase: Optional[str]) -> bool:
    if phase is None:
        return 1 <= week <= 22

    normalized = str(phase).strip().upper()
    if normalized in {"REG", "REGULAR", "R"}:
        return 1 <= week <= 18
    if normalized in {"PRE", "PRESEASON", "P"}:
        return 0 <= week <= 4
    if normalized in {"POST", "POSTSEASON", "PLAYOFF", "WC", "DIV", "CON", "SB"}:
        # Some feeds encode postseason as relative weeks (1-5), others as
        # absolute NFL weeks. Pre-2021 seasons had a 17-week regular season,
        # so postseason absolute numbering there correctly starts at 18
        # (Wild Card=18 ... Super Bowl=21); 2021+ seasons (18-week regular
        # season) start postseason at 19. Accept 18-22 rather than trying
        # to thread the season-dependent boundary here.
        return (1 <= week <= 5) or (18 <= week <= 22)
    return 1 <= week <= 22


def validate_weekly_data(
    df: pd.DataFrame,
    strict: bool = False,
    critical_null_threshold: float = 0.20,
) -> List[str]:
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
            if nan_rate > critical_null_threshold:
                issues.append(
                    f"CRITICAL: Column '{col}' null rate {nan_rate:.1%} exceeds "
                    f"critical threshold {critical_null_threshold:.1%}"
                )
            if nan_rate > 0.05:
                issues.append(
                    f"WARNING: Column '{col}' has {nan_rate:.1%} null values"
                )

    # Position membership check
    if "position" in df.columns:
        invalid_positions = sorted(
            {
                str(pos)
                for pos in df["position"].dropna().astype(str).str.upper()
                if pos not in VALID_POSITIONS
            }
        )
        if invalid_positions:
            issues.append(
                f"CRITICAL: Invalid position values found: {invalid_positions}"
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

    # Check week range sanity by season phase (if present)
    if "week" in df.columns:
        phases = None
        if "season_type" in df.columns:
            phases = df["season_type"]
        elif "game_type" in df.columns:
            phases = df["game_type"]

        for idx, week_value in df["week"].items():
            if pd.isna(week_value):
                continue
            try:
                week = int(week_value)
            except (TypeError, ValueError):
                issues.append(f"CRITICAL: Week value '{week_value}' is not an integer")
                continue
            phase = phases.loc[idx] if phases is not None else None
            if not _is_valid_week_for_phase(week, phase):
                issues.append(
                    f"CRITICAL: Invalid week {week} for season phase '{phase}'"
                )
                break

    # Yardage columns: negative values are valid NFL data (tackles behind LOS) — warn only
    for col in NEGATIVE_WARN_COLUMNS:
        if col not in df.columns:
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        neg_count = (numeric_col < 0).sum()
        if neg_count > 0:
            issues.append(
                f"WARNING: Column '{col}' has {int(neg_count)} negative values (valid for tackles behind LOS)"
            )

    # No negative values for count/attempt fields
    for col in NEGATIVE_DISALLOWED_COLUMNS:
        if col not in df.columns:
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        neg_count = (numeric_col < 0).sum()
        if neg_count > 0:
            issues.append(
                f"CRITICAL: Column '{col}' has {int(neg_count)} negative values"
            )

    # Check for duplicate rows on required keys
    if all(c in df.columns for c in ["player_id", "season", "week"]):
        dup_count = df.duplicated(subset=["player_id", "season", "week"]).sum()
        if dup_count > 0:
            issues.append(
                f"CRITICAL: {dup_count} duplicate (player_id, season, week) rows"
            )

    # Stricter uniqueness when team/opponent are present
    unique_subsets = [
        ["player_id", "season", "week", "team"],
        ["player_id", "season", "week", "team", "opponent"],
    ]
    for subset in unique_subsets:
        if all(c in df.columns for c in subset):
            dup_count = df.duplicated(subset=subset).sum()
            if dup_count > 0:
                issues.append(f"CRITICAL: {dup_count} duplicate {tuple(subset)} rows")

    issues.extend(_check_snap_invariants(df))

    # Log issues
    for issue in issues:
        if issue.startswith("CRITICAL"):
            logger.error(issue)
        else:
            logger.warning(issue)

    _raise_if_strict_critical(issues, strict)

    return issues


# A team runs roughly 40-90 offensive plays; the widest band observed across
# 2018-2025 is 34-100. Anything outside 20-120 is not a play count.
TEAM_SNAPS_MIN, TEAM_SNAPS_MAX = 20, 120


def _check_snap_invariants(df: pd.DataFrame) -> List[str]:
    """Catch the two ways snap data has actually gone wrong in this repo.

    Both shipped silently into a validation season and were found only by
    hand months later (GAPS.md 2026-08-19):

    1. `team_snaps` summed over players instead of counting team plays --
       ~11 offensive players each credited with the same ~60 snaps inflated
       it ~12x (2025 held avg 646, max 2112). snap_share came out ~12x too
       SMALL, so no ratio bound catches it; only a plausibility band on the
       play count does.
    2. `snap_count` doubled for players appearing in the passing frame plus
       another (QBs who also rush), because snap_count was present before a
       duplicate-collapse that sums numeric columns. That one shows up as
       snap_count > team_snaps.

    Checked here rather than in the aggregator so every write path is
    covered, not just the PBP one that happened to break.
    """
    issues: List[str] = []
    if not {"snap_count", "team_snaps"}.issubset(df.columns):
        return issues

    snap_count = pd.to_numeric(df["snap_count"], errors="coerce")
    team_snaps = pd.to_numeric(df["team_snaps"], errors="coerce")
    known = team_snaps > 0

    over = int((known & (snap_count > team_snaps)).sum())
    if over:
        worst = float((snap_count[known] / team_snaps[known]).max())
        issues.append(
            f"CRITICAL: {over} row(s) have snap_count > team_snaps "
            f"(max ratio {worst:.2f}) -- a player cannot take more snaps than "
            f"his team ran. Usually snap_count double-counted before a "
            f"duplicate-collapse."
        )

    outside = known & ((team_snaps < TEAM_SNAPS_MIN) | (team_snaps > TEAM_SNAPS_MAX))
    n_outside = int(outside.sum())
    if n_outside:
        issues.append(
            f"CRITICAL: {n_outside} row(s) have team_snaps outside "
            f"[{TEAM_SNAPS_MIN},{TEAM_SNAPS_MAX}] "
            f"(min {int(team_snaps[outside].min())}, max {int(team_snaps[outside].max())}) "
            f"-- not a plausible offensive play count. Usually team_snaps "
            f"summed over players instead of counting team plays."
        )
    return issues


def validate_schedule_data(
    df: pd.DataFrame,
    strict: bool = False,
    critical_null_threshold: float = 0.20,
) -> List[str]:
    """Validate schedule DataFrame against expected schema."""
    issues: List[str] = []

    if df.empty:
        issues.append("WARNING: Schedule DataFrame is empty")
        return issues

    missing_required = SCHEDULE_REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        msg = f"CRITICAL: Missing required schedule columns: {sorted(missing_required)}"
        issues.append(msg)

    # Critical null-rate checks for key columns
    for col in ["season", "week", "home_team", "away_team"]:
        if col in df.columns:
            null_rate = df[col].isna().mean()
            if null_rate > critical_null_threshold:
                issues.append(
                    f"CRITICAL: Column '{col}' null rate {null_rate:.1%} exceeds "
                    f"critical threshold {critical_null_threshold:.1%}"
                )

    # Week and phase checks
    if "week" in df.columns:
        phase_col = None
        if "game_type" in df.columns:
            phase_col = df["game_type"]
        elif "season_type" in df.columns:
            phase_col = df["season_type"]
        for idx, week_value in df["week"].items():
            if pd.isna(week_value):
                continue
            try:
                week = int(week_value)
            except (TypeError, ValueError):
                issues.append(f"CRITICAL: Week value '{week_value}' is not an integer")
                continue
            phase = phase_col.loc[idx] if phase_col is not None else None
            if not _is_valid_week_for_phase(week, phase):
                issues.append(
                    f"CRITICAL: Invalid week {week} for schedule phase '{phase}'"
                )
                break

    # Uniqueness checks
    if all(c in df.columns for c in ["season", "week", "home_team", "away_team"]):
        dup_games = df.duplicated(subset=["season", "week", "home_team", "away_team"]).sum()
        if dup_games > 0:
            issues.append(
                f"CRITICAL: {dup_games} duplicate (season, week, home_team, away_team) rows"
            )
    if "game_id" in df.columns:
        dup_game_ids = df["game_id"].duplicated().sum()
        if dup_game_ids > 0:
            issues.append(f"CRITICAL: {dup_game_ids} duplicate game_id values")

    _raise_if_strict_critical(issues, strict)

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


# ---------------------------------------------------------------------------
# Data freshness SLA (Directive V7 §19.3)
# ---------------------------------------------------------------------------

def check_data_freshness(
    df: pd.DataFrame,
    max_staleness_days: int = 7,
    date_col: str = "game_date",
    season_col: str = "season",
    week_col: str = "week",
) -> dict:
    """Check if data is stale beyond the freshness SLA.

    Per Directive V7 Section 19.3: every data source must have a documented
    freshness SLA. Alert when data is too stale.

    Args:
        df: DataFrame to check.
        max_staleness_days: Maximum acceptable staleness in days.
        date_col: Column with game dates.
        season_col: Column with season year.
        week_col: Column with week number.

    Returns:
        Dict with freshness assessment.
    """
    from datetime import datetime, timedelta

    result = {
        "is_fresh": True,
        "max_staleness_days": max_staleness_days,
        "warnings": [],
    }

    if df.empty:
        result["is_fresh"] = False
        result["warnings"].append("DataFrame is empty")
        return result

    if date_col in df.columns:
        try:
            latest_date = pd.to_datetime(df[date_col]).max()
            if pd.notna(latest_date):
                staleness = (datetime.now() - latest_date).days
                result["latest_date"] = str(latest_date.date())
                result["staleness_days"] = staleness
                if staleness > max_staleness_days:
                    result["is_fresh"] = False
                    result["warnings"].append(
                        f"Data is {staleness} days old (SLA: {max_staleness_days} days)"
                    )
        except Exception as e:
            # A silent failure here would let a stale-data problem pass
            # this freshness gate undetected (GAPS.md §9 audit) — record it
            # in the report's own warnings list, matching this function's
            # existing pattern, rather than swallowing it.
            result["warnings"].append(f"Staleness check failed: {e}")

    if season_col in df.columns and week_col in df.columns:
        try:
            max_season = int(df[season_col].max())
            max_week = int(df[df[season_col] == max_season][week_col].max())
            result["latest_season"] = max_season
            result["latest_week"] = max_week
        except Exception as e:
            result["warnings"].append(f"Latest season/week lookup failed: {e}")

    return result


def validate_on_load(source_name: str = "data", strict: bool = False):
    """Decorator to auto-validate DataFrames after loading.

    Per Directive V7 Section 19: pipeline must include schema validation.

    Usage:
        @validate_on_load("weekly_stats")
        def load_weekly_data():
            return pd.read_csv("weekly.csv")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            if isinstance(df, pd.DataFrame):
                issues = validate_weekly_data(df, strict=strict)
                if issues:
                    logger.warning(
                        "Schema validation for %s: %d issues found",
                        source_name, len(issues),
                    )
            return df
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
