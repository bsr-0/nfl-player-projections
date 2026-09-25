"""Eight-component PPR reconstruction against checked raw player outcomes."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS, _ppr_for_choices
from src.evaluation.ppr_truth import KEY, validate_inputs
from src.evaluation.team_share_backtester import bootstrap_mae_delta


def evaluate_reconstruction(
    allocation_results: Mapping[str, dict],
    team_results: Mapping[str, dict],
    *,
    arms: Mapping[str, str],
    raw_rows: pd.DataFrame,
) -> dict:
    """Compare one allocation configuration to both rolling-share baselines."""
    if set(arms) != set(PPR_RECONSTRUCTION_TARGETS):
        raise ValueError("reconstruction requires all eight component arms")
    truth, audit = validate_inputs(allocation_results, team_results, raw_rows)
    folds = set(truth.fold.unique())
    team_arms = dict.fromkeys(PPR_RECONSTRUCTION_TARGETS, "blend")
    candidate = _ppr_for_choices(allocation_results, team_results, arms, team_arms, folds, truth, return_rows=True)
    report = {
        "scoring_definition": "raw_eight_component_ppr_excludes_fumbles_and_two_point_conversions",
        "candidate_mae": float(np.abs(candidate.actual_ppr - candidate.predicted_ppr).mean()),
        "candidate_rmse": float(np.sqrt(mean_squared_error(candidate.actual_ppr, candidate.predicted_ppr))),
        "n": len(candidate),
        "input_audit": audit,
        "baseline_comparisons": {},
    }
    for name in ("rolling3", "rolling3_renorm"):
        baseline_arms = dict.fromkeys(PPR_RECONSTRUCTION_TARGETS, name)
        baseline = _ppr_for_choices(allocation_results, team_results, baseline_arms, team_arms, folds, truth, return_rows=True)
        joined = candidate.merge(baseline, on=KEY, how="outer", validate="one_to_one",
                                 suffixes=("_candidate", "_baseline"), indicator=True)
        if not joined._merge.eq("both").all() or not np.allclose(joined.actual_ppr_candidate, joined.actual_ppr_baseline):
            raise ValueError("candidate and baseline truth populations differ")
        report["baseline_comparisons"][name] = {
            "mae": float(np.abs(joined.actual_ppr_baseline - joined.predicted_ppr_baseline).mean()),
            "rmse": float(np.sqrt(mean_squared_error(joined.actual_ppr_baseline, joined.predicted_ppr_baseline))),
            "vs_candidate": bootstrap_mae_delta(joined.actual_ppr_candidate.to_numpy(),
                                                 joined.predicted_ppr_candidate.to_numpy(),
                                                 joined.predicted_ppr_baseline.to_numpy()),
        }
    report["arms"] = dict(arms)
    return report
