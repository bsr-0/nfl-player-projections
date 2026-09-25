"""Attribution for the fold-local joint Plan A PPR selector.

The report asks a deliberately narrow counterfactual question: on each held
out fold, how much reconstructed PPR error returns when one component is
reverted to rolling-3 while every other selected component remains fixed?
It is attribution, not causal feature importance, and never retrains on a
test fold.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping
import pandas as pd

from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS, _ppr_for_choices


def component_contribution_report(
    allocation_results: Mapping[str, dict],
    team_results: Mapping[str, dict],
    selector_report: dict,
    truth: pd.DataFrame,
) -> dict:
    """Measure allocation and team-total contribution per component/fold.

    A positive delta means the selected component lowered PPR MAE relative
    to its rolling-3 counterfactual.  Allocation and team-total arms are
    isolated separately so a good player-share allocation is not credited
    for a team-total forecast (or vice versa).
    """
    rows: list[dict] = []
    for fold_report in selector_report["folds"]:
        fold = int(fold_report["fold"])
        allocation = dict(fold_report["allocation_arms"])
        totals = dict(fold_report["team_total_arms"])
        selected_mae, n = _ppr_for_choices(
            allocation_results, team_results, allocation, totals, {fold}, truth
        )
        for target in PPR_RECONSTRUCTION_TARGETS:
            for dimension in ("allocation", "team_total", "both"):
                trial_allocation = dict(allocation)
                trial_totals = dict(totals)
                if dimension in {"allocation", "both"}:
                    trial_allocation[target] = "rolling3"
                if dimension in {"team_total", "both"}:
                    trial_totals[target] = "rolling3"
                mae, trial_n = _ppr_for_choices(
                    allocation_results, team_results, trial_allocation, trial_totals, {fold}, truth
                )
                if trial_n != n:
                    raise ValueError(
                        f"counterfactual row population changed for {target}/{dimension}: {trial_n} != {n}"
                    )
                rows.append({
                    "fold": fold,
                    "target": target,
                    "dimension": dimension,
                    "n": n,
                    "selected_ppr_mae": selected_mae,
                    "counterfactual_ppr_mae": mae,
                    "delta_mae_when_reverted": mae - selected_mae,
                    "selected_allocation_arm": allocation[target],
                    "selected_team_total_arm": totals[target],
                })

    pooled: dict[str, dict] = defaultdict(dict)
    for target in PPR_RECONSTRUCTION_TARGETS:
        for dimension in ("allocation", "team_total", "both"):
            matching = [row for row in rows if row["target"] == target and row["dimension"] == dimension]
            weight = sum(row["n"] for row in matching)
            pooled[target][dimension] = {
                "n": int(weight),
                "weighted_delta_mae_when_reverted": (
                    sum(row["delta_mae_when_reverted"] * row["n"] for row in matching) / weight
                    if weight else None
                ),
                "positive_means_selected_component_helped": True,
            }
    return {"folds": rows, "pooled": dict(pooled)}
