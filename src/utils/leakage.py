"""
Centralized leakage defenses for training/evaluation pipelines.

This module provides:
- Feature column filtering to block target/model-output leakage
- Schedule sanitization to remove final scores from feature inputs
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple
import re

import pandas as pd


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
)


def is_leakage_feature(col: str, *, ban_utilization_score: bool = True) -> bool:
    """Return True if a column name indicates leakage risk."""
    if not col:
        return False
    col_l = col.lower()

    if ban_utilization_score and col_l == "utilization_score":
        return True

    if _TARGET_COL_RE.match(col_l):
        return True

    if any(col_l.startswith(p) for p in _MODEL_OUTPUT_PREFIXES):
        return True

    if any(col_l.startswith(p) for p in _BACKTEST_PREFIXES):
        return True

    if any(s in col_l for s in _FORWARD_SUBSTRINGS):
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
