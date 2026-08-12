"""Final per-position (architecture, window, weighting) choice from Phases 2-3.

Single source of truth for "the chosen config" — referenced by Phase 4's
validation run and any later phase that needs it. See data/experiments/
phase2_single_week_comparison_v2_corrected.csv and
phase3_training_window_comparison_v2_corrected.csv for the underlying
results, and GAPS.md §7.9 / the 2026-08-10 re-run entry for the Phase 2/3
findings. Superseded 2026-08-10/11: re-run against the corrected
Complete Player-Game Panel data (see GAPS.md "SUPERSEDED — 2026-08-10
(Complete Player-Game Panel prerequisite)"). The pre-fix values are
preserved in phase2/3_..._comparison.csv (no _v2_corrected suffix) for
reference.
"""
from __future__ import annotations

from typing import Dict, TypedDict


class PositionConfig(TypedDict):
    architecture: str  # key into _architectures_for_fold() in evaluate.py
    window: str        # key into windows.WINDOW_CANDIDATES
    weighting: str      # key into windows.WEIGHTING_SCHEMES


FINAL_CONFIG: Dict[str, PositionConfig] = {
    "QB": {"architecture": "F_yeojohnson_huber", "window": "7y", "weighting": "none"},
    "RB": {"architecture": "C_gbm_mae", "window": "all", "weighting": "exponential"},
    "WR": {"architecture": "C_gbm_mae", "window": "3y", "weighting": "none"},
    "TE": {"architecture": "B_gbm_huber", "window": "7y", "weighting": "none"},
}
