"""Final per-position (architecture, window, weighting) choice from Phases 2-3.

Single source of truth for "the chosen config" — referenced by Phase 4's
validation run and any later phase that needs it. See data/experiments/
phase2_single_week_comparison.csv and phase3_training_window_comparison.csv
for the underlying results, and GAPS.md §7.9 for the Phase 3 findings.
"""
from __future__ import annotations

from typing import Dict, TypedDict


class PositionConfig(TypedDict):
    architecture: str  # key into _architectures_for_fold() in evaluate.py
    window: str        # key into windows.WINDOW_CANDIDATES
    weighting: str      # key into windows.WEIGHTING_SCHEMES


FINAL_CONFIG: Dict[str, PositionConfig] = {
    "QB": {"architecture": "C_gbm_mae", "window": "all", "weighting": "none"},
    "RB": {"architecture": "F_yeojohnson_huber", "window": "all", "weighting": "none"},
    "WR": {"architecture": "C_gbm_mae", "window": "all", "weighting": "none"},
    "TE": {"architecture": "C_gbm_mae", "window": "10y", "weighting": "none"},
}
