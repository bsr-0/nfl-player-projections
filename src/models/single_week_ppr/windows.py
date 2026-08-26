"""Phase 3 (next_focus.md) training-window and recency-weighting definitions.

Deliberately bypasses the codebase's production 2018+ training floor
(src/utils/data_manager.py TRAINING_START_YEAR_DEFAULT) for this experiment
only — see GAPS.md Phase 3 notes and the Phase 3 plan. Nothing here changes
production behavior; run_fold(train_seasons_override=...) in evaluate.py is
the only caller.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from config.settings import MIN_HISTORICAL_YEAR, MODEL_CONFIG

# The measurement-era floor, as a candidate rather than an imposed rule.
#
# snap_counts, depth_charts and player_injuries all begin in 2013. Before that,
# 12 of WR's 71 causal features are entirely absent and mean missingness is
# 17.3% against 4.5% from 2018 (measured 2026-08-25). The question of whether
# those seasons help is empirical, and this file already exists to answer that
# kind of question, so it competes against the relative windows instead of
# being hardcoded.
#
# Note this is an ABSOLUTE floor, unlike 3y/5y/7y/10y which are relative to the
# test season -- "since2013" spans 12 seasons for a 2025 fold and 5 for a 2018
# one. It is deliberately NOT expressed as a year count for that reason.
SEASON_FLOOR_WINDOWS = {"since2013": 2013}

WINDOW_CANDIDATES = ("3y", "5y", "7y", "10y", "since2013", "all")
WEIGHTING_SCHEMES = ("none", "linear", "exponential")

_WINDOW_YEARS = {"3y": 3, "5y": 5, "7y": 7, "10y": 10}


def window_to_season_list(window: str, test_season: int, available_seasons: Sequence[int]) -> List[int]:
    """Returns the season list for a given window, bypassing the 2018 floor
    down to MIN_HISTORICAL_YEAR. Degrades gracefully (fewer seasons than
    requested) rather than erroring when history is short.
    """
    prior_seasons = sorted(s for s in available_seasons if MIN_HISTORICAL_YEAR <= s < test_season)
    if window == "all":
        return prior_seasons
    if window in SEASON_FLOOR_WINDOWS:
        floor = SEASON_FLOOR_WINDOWS[window]
        return [s for s in prior_seasons if s >= floor]
    if window not in _WINDOW_YEARS:
        raise ValueError(f"Unknown window: {window!r} (expected one of {WINDOW_CANDIDATES})")
    n_years = _WINDOW_YEARS[window]
    return prior_seasons[-n_years:]


def compute_recency_weights(seasons: pd.Series, scheme: str) -> Optional[np.ndarray]:
    """Per-row sample weights for a given recency-weighting scheme.

    'none' -> None (uniform, matches sklearn/LightGBM default when
      sample_weight is omitted).
    'linear' -> ramps linearly from ~1/N (oldest season) to 1.0 (most
      recent), new — no existing scheme in the codebase to reuse.
    'exponential' -> reuses the EXACT formula already live in production
      (src/models/position_models.py:_horizon_recency_weights,
      src/models/ensemble.py), so results here are comparable to what the
      existing methodology already does: decay = 0.5**((max-season)/halflife),
      normalized to max 1.0. Halflife from
      MODEL_CONFIG["horizon_recency_halflife"][1] (single-week horizon),
      falling back to MODEL_CONFIG["recency_decay_halflife"].
    """
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"Unknown weighting scheme: {scheme!r} (expected one of {WEIGHTING_SCHEMES})")
    if scheme == "none":
        return None

    seasons_arr = np.asarray(seasons, dtype=float)
    if len(seasons_arr) == 0:
        return None
    max_season = np.nanmax(seasons_arr)
    min_season = np.nanmin(seasons_arr)
    if not np.isfinite(max_season) or max_season <= min_season:
        return np.ones(len(seasons_arr))

    if scheme == "linear":
        raw = seasons_arr - min_season + 1
        return raw / raw.max()

    # exponential
    halflife = MODEL_CONFIG.get("horizon_recency_halflife", {}).get(1) or MODEL_CONFIG.get("recency_decay_halflife")
    decay = np.power(0.5, (max_season - seasons_arr) / float(halflife))
    return decay / decay.max()
