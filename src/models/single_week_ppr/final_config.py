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

Superseded again 2026-08-18 (FEATURE_VERSION 33, depth_chart_rank as-of
fix): re-run against `phase2_single_week_comparison_v33.csv` /
`phase3_training_window_comparison_v33.csv`. Architectures are UNCHANGED
at every position; only windows/weightings moved, and only slightly.

Caveat worth keeping in view when reading these choices: the spread
between the best and worst (window, weighting) combination within a
position is small -- QB 0.176, RB 0.106, WR 0.087, TE 0.086 MAE. The
"winners" below are therefore best-of-a-narrow-band, not decisive; e.g.
QB's all/none (6.337) beats 10y/none (6.362) and 7y/none (6.385) by
margins plausibly within noise. Treat a change here as low-stakes.
"""
from __future__ import annotations

from typing import Dict, TypedDict


class PositionConfig(TypedDict):
    architecture: str  # key into _architectures_for_fold() in evaluate.py
    window: str        # key into windows.WINDOW_CANDIDATES
    weighting: str      # key into windows.WEIGHTING_SCHEMES


FINAL_CONFIG: Dict[str, PositionConfig] = {
    "QB": {"architecture": "F_yeojohnson_huber", "window": "all", "weighting": "none"},
    "RB": {"architecture": "C_gbm_mae", "window": "all", "weighting": "linear"},
    "WR": {"architecture": "C_gbm_mae", "window": "3y", "weighting": "none"},
    "TE": {"architecture": "B_gbm_huber", "window": "10y", "weighting": "none"},
}
