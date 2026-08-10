"""
Centralized leakage defenses for training/evaluation pipelines.

This module provides:
- Feature column filtering to block target/model-output leakage
- Schedule sanitization to remove final scores from feature inputs
- Explicit allowlist for schedule-derived features that are safe during training
  (backward-looking or rate-based, despite containing "_next" in names)
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple
import re

import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Feature leakage guards
# -----------------------------------------------------------------------------

# Only match true ML targets (avoid collisions like target_share).
_TARGET_COL_RE = re.compile(r"^target(_util)?_\d+w$|^target$")

# Model-output columns that should never be used as features.
_MODEL_OUTPUT_PREFIXES: Tuple[str, ...] = (
    "predicted_",
    "projection_",
)

# Backtest artifacts that must never be features.
_BACKTEST_PREFIXES: Tuple[str, ...] = (
    "baseline_",
    "actual_for_backtest",
)

# Generic forward-looking naming patterns to avoid.
_FORWARD_SUBSTRINGS: Tuple[str, ...] = (
    "_future",
    "_next",
    "_forward",
    "_upcoming",
)

# Identifier / row-index columns that must never be features (memorization risk).
_IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "id",
    "player_id",
    "player_name",
    "name",
    "gsis_id",
    "espn_id",
    "yahoo_id",
)

# Schedule-derived feature prefixes that contain "_next" but are safe during
# training because multiweek_features.py computes them backward-looking
# (rolling mean of past opponents) or from rate-based projections.
# These are explicitly allowed through the forward-looking filter.
SAFE_SCHEDULE_PREFIXES: Tuple[str, ...] = (
    "sos_next_",
    "sos_rank_next_",
    "favorable_matchups_next_",
    "expected_games_next_",
    "injury_prob_next_",
    "injury_risk_score_",
    "expected_missed_games_",
)


def _is_safe_schedule_feature(col: str) -> bool:
    """Return True if column is a known-safe schedule-derived feature."""
    col_l = col.lower()
    return any(col_l.startswith(p) for p in SAFE_SCHEDULE_PREFIXES)


def is_leakage_feature(col: str, *, ban_utilization_score: bool = True) -> bool:
    """Return True if a column name indicates leakage risk.

    Checks (in order):
    1. Identifier columns (id, player_id, etc.) — memorization risk
    2. Utilization score (when ban_utilization_score=True) — target leakage
    3. Target columns (target_1w, target_util_4w, etc.)
    4. Model-output prefixes (predicted_*, projection_*)
    5. Backtest artifact prefixes (baseline_*, actual_for_backtest)
    6. Forward-looking substrings (_future, _forward) — BUT allows
       known-safe schedule-derived features (sos_next_*, etc.)
    """
    if not col:
        return False
    col_l = col.lower()

    # Identifier columns must never be features.
    if col_l in _IDENTIFIER_COLUMNS:
        return True

    if ban_utilization_score and col_l == "utilization_score":
        return True

    # Raw utilization score is only used for target derivation, never as a feature.
    if col_l == "utilization_score_raw":
        return True

    if _TARGET_COL_RE.match(col_l):
        return True

    if any(col_l.startswith(p) for p in _MODEL_OUTPUT_PREFIXES):
        return True

    if any(col_l.startswith(p) for p in _BACKTEST_PREFIXES):
        return True

    # Forward-looking check: block _future/_forward, but allow safe schedule
    # features that use backward-looking computation during training.
    if any(s in col_l for s in _FORWARD_SUBSTRINGS):
        if not _is_safe_schedule_feature(col):
            return True

    return False


def find_leakage_columns(
    columns: Iterable[str],
    *,
    ban_utilization_score: bool = True,
) -> List[str]:
    """Return list of columns that look like leakage features."""
    return [c for c in columns if is_leakage_feature(c, ban_utilization_score=ban_utilization_score)]


def filter_feature_columns(
    feature_cols: Iterable[str],
    *,
    allow: Optional[Sequence[str]] = None,
    ban_utilization_score: bool = True,
) -> List[str]:
    """Filter out leakage-prone feature columns.

    Args:
        feature_cols: Candidate feature column names.
        allow: Optional explicit allow-list (bypasses leakage checks).
        ban_utilization_score: If True, remove utilization_score as a feature.
    """
    allow_set = set(allow or [])
    out: List[str] = []
    for col in feature_cols:
        if col in allow_set:
            out.append(col)
            continue
        if is_leakage_feature(col, ban_utilization_score=ban_utilization_score):
            continue
        out.append(col)
    return out


def drop_leakage_columns(
    df: pd.DataFrame,
    *,
    allow: Optional[Sequence[str]] = None,
    ban_utilization_score: bool = True,
) -> pd.DataFrame:
    """Return DataFrame with leakage columns removed."""
    if df.empty:
        return df
    to_drop = find_leakage_columns(df.columns, ban_utilization_score=ban_utilization_score)
    if allow:
        to_drop = [c for c in to_drop if c not in set(allow)]
    if not to_drop:
        return df
    return df.drop(columns=to_drop, errors="ignore")


def assert_no_leakage_columns(
    feature_cols: Iterable[str],
    *,
    ban_utilization_score: bool = True,
    context: str = "features",
) -> None:
    """Raise ValueError if leakage columns are present."""
    leaked = find_leakage_columns(feature_cols, ban_utilization_score=ban_utilization_score)
    if leaked:
        leaked_sorted = ", ".join(sorted(leaked)[:10])
        raise ValueError(f"Leakage columns detected in {context}: {leaked_sorted}")


# -----------------------------------------------------------------------------
# Schedule sanitization
# -----------------------------------------------------------------------------

_SCHEDULE_SCORE_COLUMNS: Tuple[str, ...] = (
    "home_score",
    "away_score",
    "home_team_score",
    "away_team_score",
    "home_points",
    "away_points",
    "home_pts",
    "away_pts",
    "points_home",
    "points_away",
)


def sanitize_schedule_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any final-score columns from schedule data used as features."""
    if df is None or df.empty:
        return df
    cols_to_drop = [c for c in df.columns if c in _SCHEDULE_SCORE_COLUMNS]
    if not cols_to_drop:
        return df
    return df.drop(columns=cols_to_drop, errors="ignore")


# -----------------------------------------------------------------------------
# Feature availability registry
# -----------------------------------------------------------------------------
#
# Formalizes, per feature family, when that family's data is actually known
# relative to the week being predicted. This is the single source of truth
# for "available_timestamp <= prediction_timestamp" — it classifies column
# *identity*, not row-level nulls (a missing value is not leakage; a column
# whose value could only be known after kickoff is).
#
# Each entry: (prefix_or_substring_match, human-readable availability rule).
# Order matters only in that the first match wins, most-specific first.
FEATURE_AVAILABILITY: Tuple[Tuple[str, str], ...] = (
    # Model outputs / targets / backtest artifacts are never inputs at all —
    # already hard-blocked by is_leakage_feature(), listed here for completeness.
    ("predicted_", "MODEL OUTPUT — never a valid input feature"),
    ("projection_", "MODEL OUTPUT — never a valid input feature"),
    ("target_", "TARGET — never a valid input feature"),
    ("baseline_", "BACKTEST ARTIFACT — never a valid input feature"),
    # Opponent / own-team box-score aggregates: joined from week - 1 in
    # get_all_players_for_training (src/utils/database.py), enforced by a
    # runtime ValueError assertion on opp_defense_week / own_team_stats_week.
    ("opp_fpts_allowed", "week - 1 (runtime-asserted in database.py)"),
    ("fantasy_points_allowed_", "week - 1 (runtime-asserted in database.py)"),
    ("team_points", "week - 1 (runtime-asserted in database.py)"),
    ("team_yards", "week - 1 (runtime-asserted in database.py)"),
    ("team_pass_attempts", "week - 1 (runtime-asserted in database.py)"),
    ("team_rush_attempts", "week - 1 (runtime-asserted in database.py)"),
    ("team_redzone_attempts", "week - 1 (runtime-asserted in database.py)"),
    ("team_plays", "week - 1 (runtime-asserted in database.py)"),
    ("team_neutral_", "week - 1 (runtime-asserted in database.py)"),
    ("team_drive_", "week - 1 (runtime-asserted in database.py)"),
    ("team_avg_drive_epa", "week - 1 (runtime-asserted in database.py)"),
    ("team_points_per_drive", "week - 1 (runtime-asserted in database.py)"),
    ("team_pace_sec_per_play", "week - 1 (runtime-asserted in database.py)"),
    # Player rolling / season-to-date aggregates: shift(1) applied before
    # .rolling()/.expanding() throughout feature_engineering.py, database.py,
    # season_long_features.py, qb_features.py, evaluation/baselines.py.
    ("_roll3", "week - 1 and earlier (shift(1) before .rolling())"),
    ("_roll5", "week - 1 and earlier (shift(1) before .rolling())"),
    ("_rolling_", "week - 1 and earlier (shift(1) before .rolling())"),
    ("_s2d_lag1", "week - 1 and earlier (shift(1) before .expanding())"),
    ("prev_season_", "prior season (fully known pre-kickoff)"),
    ("_season_prior", "prior season (fully known pre-kickoff)"),
    ("career_year_flag", "prior season and earlier (fully known pre-kickoff)"),
    ("bayesian_prior_ppg", "prior data only, expanding mean (shift(1) applied)"),
    # Schedule / market / environment: legitimately known before kickoff of
    # the predicted week (Vegas lines close pre-game, schedule is fixed,
    # weather forecasts are pre-game estimates).
    ("implied_team_total", "known pre-kickoff (Vegas line)"),
    ("spread", "known pre-kickoff (Vegas line)"),
    ("sos_next_", "prediction week (schedule is fixed in advance)"),
    ("sos_rank_next_", "prediction week (schedule is fixed in advance)"),
    ("favorable_matchups_next_", "prediction week (schedule is fixed in advance)"),
    ("expected_games_next_", "prediction week (derived from prior injury history)"),
    ("wind_speed_mph", "pre-kickoff forecast"),
    ("is_dome", "static stadium attribute"),
    ("precipitation_flag", "pre-kickoff forecast"),
    ("temperature_bucket", "pre-kickoff forecast"),
    # Draft capital / combine / identity: fixed at draft time, static per player.
    ("is_rookie", "static player attribute"),
    ("rookie_draft_value", "fixed at draft time"),
    ("combine_score", "fixed at combine, static per player"),
    ("depth_chart_rank", "most recent depth chart snapshot, prior to prediction week"),
    # Injury/status: enforced pre-kickoff via nfl_data_py's date_modified
    # timestamp vs. schedule kickoff time (InjuryDataLoader._load_kickoff_times
    # in src/data/external_data.py); post-kickoff reports are dropped before
    # feature construction. See GAPS.md §7.6.
    ("injury_score", "pre-kickoff (runtime-filtered in external_data.py)"),
    ("injury_prob_next_", "pre-kickoff (runtime-filtered in external_data.py)"),
    ("injury_risk_score_", "pre-kickoff (runtime-filtered in external_data.py)"),
    ("expected_missed_games_", "assumed week - 1 report (UNVERIFIED — see GAPS.md)"),
    ("availability_3yr", "prior seasons (fully known pre-kickoff)"),
)


def audit_feature_availability(columns: Iterable[str]) -> List[str]:
    """Return columns not covered by is_leakage_feature() or FEATURE_AVAILABILITY.

    Existing feature columns should either be blocked outright (is_leakage_feature)
    or have a documented availability rule (FEATURE_AVAILABILITY). Anything left
    over is unclassified and should be reviewed before use as a model input —
    this does NOT mean it is leaking, only that its availability timing hasn't
    been audited yet.
    """
    unclassified: List[str] = []
    for col in columns:
        if not col:
            continue
        col_l = col.lower()
        if is_leakage_feature(col):
            continue
        if any(pattern in col_l for pattern, _rule in FEATURE_AVAILABILITY):
            continue
        unclassified.append(col)
    return unclassified
