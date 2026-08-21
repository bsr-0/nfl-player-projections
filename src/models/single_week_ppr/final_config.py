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

RE-VALIDATED 2026-08-21 on the corrected panel (GAPS.md, search
"FINAL_CONFIG re-validation"). The 2026-08-18 selection above predated the
2026-08-20 data corrections -- position corruption (11.4% of the QB
population were not QBs), home_away/is_dome (RB's 4th most important
feature, broken across 96K rows), team_plays_roll3_mean, the PBP
situational backfill, the 2009 receiving floor, and age. Phases 2 and 3
were re-run with methodology unchanged and the original selection rule
(point estimators only; E_quantile_gbm remains excluded as a
floor/median/ceiling tool, not a competing point estimator).

Outcome, stated so a future reader does not mistake retention for
staleness:

  * ARCHITECTURE: TE changed materially, B_gbm_huber -> C_gbm_mae, margin
    0.106 MAE (~4x the +/-0.046 per-fold noise band this session
    established). QB/RB/WR did NOT change: QB and RB swapped with each
    other at margins of 0.025 and 0.022, and WR was identical -- ties
    broken differently by sort order, not evidence of reversal.

  * WINDOW / WEIGHTING: no meaningful changes; incumbent settings
    RETAINED at every position. The re-run nominally selected different
    winners everywhere, but by margins of 0.002-0.022 MAE, all below half
    the noise floor. Adopting them would fit sort order rather than data.
    In particular RB's floored run selected 10y/linear over the incumbent
    all/linear by 0.007 -- deliberately NOT adopted.

  * RB RECEIVING-FLOOR SENSITIVITY: the floor is applied. Only the "all"
    window differs between floored and unfloored runs (every other window
    is identical to 4dp, as it must be). Winner labels differ
    (10y/linear vs all/exponential) but the two are 0.0002 MAE apart, and
    the unfloored "all" is better on exponential yet worse on linear and
    none -- mixed signs consistent with noise from 758 extra rows rather
    than signal. Population choice is therefore decided on VALIDITY, not
    performance.

LINEAGE WARNING: the Phase 7 2013-2025 benchmark (25.08 MAE matched, the
figure that superseded the historical 26-40% result) was generated with
TE = B_gbm_huber, i.e. the PRE-edit config. Anything consuming the new
TE architecture is not directly comparable to that artifact without a
re-run.
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
    # TE architecture re-validated 2026-08-21: B_gbm_huber -> C_gbm_mae
    # (2.900 -> 2.794 MAE). Window/weighting deliberately unchanged.
    "TE": {"architecture": "C_gbm_mae", "window": "10y", "weighting": "none"},
}
