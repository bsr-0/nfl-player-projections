"""Walk-forward selection of allocation/team-total arms on reconstructed PPR."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.evaluation.team_share_backtester import bootstrap_mae_delta
from src.evaluation.ppr_truth import TARGETS, KEY
from src.models.team_allocation.features import filter_population
from src.utils.helpers import calculate_fantasy_points_df


SPARSE = {"receiving_tds", "rushing_tds", "passing_tds", "interceptions"}
PPR_RECONSTRUCTION_TARGETS = TARGETS
DEFAULT_NONINFERIORITY_MARGIN = 0.0005
INNER_GATE_BOOTSTRAPS = 500
BLEND_WEIGHTS = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))


def _parse_blend_spec(arm: str) -> tuple[str, float] | None:
    """Parse a serializable convex blend specification.

    ``blend:<arm>:<weight>`` means ``weight * arm + (1 - weight) *
    rolling3``.  The string form makes each fold's chosen blend auditable in
    the JSON report and lets the serving/export path reconstruct it exactly.
    """
    if not arm.startswith("blend:"):
        return None
    _, component_arm, weight = arm.rsplit(":", 2)
    value = float(weight)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1], got {value}")
    return component_arm, value


def _allocation_rows(allocation: pd.DataFrame, allocation_arm: str) -> pd.DataFrame:
    """Return one share prediction per row for a direct or blended arm."""
    blend = _parse_blend_spec(allocation_arm)
    if blend is None:
        rows = allocation[allocation.arm.eq(allocation_arm)].copy()
        if rows[KEY].duplicated().any():
            raise ValueError(f"duplicate prediction rows for allocation arm {allocation_arm!r}")
        return rows
    component_arm, weight = blend
    baseline = allocation[allocation.arm.eq("rolling3")][KEY + ["actual_share", "predicted_share"]].copy()
    candidate = allocation[allocation.arm.eq(component_arm)][KEY + ["actual_share", "predicted_share"]].copy()
    if baseline[KEY].duplicated().any() or candidate[KEY].duplicated().any():
        raise ValueError(f"duplicate prediction rows in blend {allocation_arm!r}")
    rows = baseline.merge(
        candidate,
        on=KEY,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(rows) != len(baseline) or len(rows) != len(candidate):
        raise ValueError(f"row population mismatch in blend {allocation_arm!r}")
    if not np.allclose(
        rows.actual_share_baseline.to_numpy(float),
        rows.actual_share_candidate.to_numpy(float),
        equal_nan=True,
    ):
        raise ValueError(f"actual labels differ in blend {allocation_arm!r}")
    result = rows[KEY].copy()
    result["actual_share"] = rows.actual_share_baseline
    result["predicted_share"] = (
        (1.0 - weight) * rows.predicted_share_baseline + weight * rows.predicted_share_candidate
    )
    return result


def _choice_component(
    target: str,
    allocation: pd.DataFrame,
    totals: pd.DataFrame,
    allocation_arm: str,
    team_arm: str,
) -> pd.DataFrame:
    rows = _allocation_rows(allocation, allocation_arm)
    rows = rows.merge(
        totals[["fold", "team", "season", "week", team_arm]],
        on=["fold", "team", "season", "week"],
        how="left",
        validate="many_to_one",
    )
    rows["predicted_component"] = rows.predicted_share * rows[team_arm]
    if rows[team_arm].isna().any() and team_arm != "rolling3":
        raise ValueError(f"missing {target}/{team_arm} team forecast")
    return rows[KEY + ["predicted_component"]]


def _ppr_for_choices(
    allocation_results: Mapping[str, dict],
    team_results: Mapping[str, dict],
    allocation_arms: Mapping[str, str],
    team_arms: Mapping[str, str],
    folds: set[int],
    truth: pd.DataFrame,
    *,
    return_rows: bool = False,
) -> tuple[float, int] | pd.DataFrame:
    components_pred: list[pd.DataFrame] = []
    for target in PPR_RECONSTRUCTION_TARGETS:
        allocation = allocation_results[target]["predictions"]
        allocation = allocation[allocation.fold.isin(folds)]
        totals = team_results[target]["predictions"]
        totals = totals[totals.fold.isin(folds)]
        comp = _choice_component(target, allocation, totals, allocation_arms[target], team_arms[target])
        expected_keys = filter_population(truth[truth.fold.isin(folds)], target)[KEY]
        coverage = expected_keys.merge(comp[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
        if not coverage._merge.eq("both").all():
            raise ValueError(f"missing expected {target} predictions or unexpected rows")
        components_pred.append(comp[KEY + ["predicted_component"]])
    pred = truth[truth.fold.isin(folds)][KEY].copy()
    if pred.empty:
        raise ValueError("no truth rows for requested folds")
    for target, frame in zip(PPR_RECONSTRUCTION_TARGETS, components_pred):
        if frame.duplicated(KEY).any():
            raise ValueError(f"duplicate {target} component predictions")
        pred = pred.merge(frame.rename(columns={"predicted_component": target}), on=KEY, how="left", validate="one_to_one")
        expected = pred.position.ne("QB") if target in {"receiving_yards", "receptions", "receiving_tds"} else (
            pred.position.eq("QB") if target in {"passing_yards", "passing_tds", "interceptions"} else pd.Series(True, index=pred.index)
        )
        if pred.loc[expected, target].isna().any():
            # Rolling-3 team totals have one known cold-start null per team in week 1.
            if not (team_arms[target] == "rolling3" and pred.loc[expected & pred[target].isna(), "week"].eq(1).all()):
                raise ValueError(f"missing expected {target} predictions")
        pred[target] = pred[target].fillna(0.0)
    pred["ppr"] = calculate_fantasy_points_df(pred[PPR_RECONSTRUCTION_TARGETS])
    joined = pred.merge(truth[KEY + ["actual_ppr"]], on=KEY, how="left", validate="one_to_one")
    if return_rows:
        return joined[KEY + ["actual_ppr", "ppr"]].rename(columns={"ppr": "predicted_ppr"})
    return float(mean_absolute_error(joined.actual_ppr, joined.ppr)), int(len(joined))


def eligible_allocation_arms(
    predictions: pd.DataFrame,
    prior_folds: set[int],
    *,
    margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    n_bootstrap: int = INNER_GATE_BOOTSTRAPS,
) -> tuple[list[str], dict[str, dict]]:
    """Return share-safe arms using *only* completed out-of-fold rows.

    The joint objective can otherwise select an arm that helps reconstructed
    PPR by exploiting a component-level error trade-off while materially
    degrading that component's share forecast.  This is the fold-local
    equivalent of the Plan A acceptance gate: an arm is eligible only when
    the upper end of its paired share-MAE interval is within ``margin`` of
    rolling-3.  The incumbent is always available, including fold zero when
    there is no completed OOF evidence yet.
    """
    baseline = predictions[
        predictions.fold.isin(prior_folds) & predictions.arm.eq("rolling3")
    ].copy()
    if not prior_folds or baseline.empty:
        return ["rolling3"], {
            "rolling3": {"eligible": True, "reason": "incumbent_no_prior_evidence"}
        }
    if baseline[KEY].duplicated().any():
        raise ValueError("rolling3 contains duplicate row keys in prior folds")

    decisions: dict[str, dict] = {}
    allowed: list[str] = []
    for arm in sorted(predictions.arm.dropna().astype(str).unique()):
        if arm == "rolling3":
            decisions[arm] = {"eligible": True, "reason": "incumbent"}
            allowed.append(arm)
            continue
        candidate = predictions[
            predictions.fold.isin(prior_folds) & predictions.arm.eq(arm)
        ].copy()
        if candidate.empty or candidate[KEY].duplicated().any():
            decisions[arm] = {"eligible": False, "reason": "missing_or_duplicate_prior_rows"}
            continue
        joined = baseline[KEY + ["actual_share", "predicted_share"]].merge(
            candidate[KEY + ["actual_share", "predicted_share"]],
            on=KEY,
            suffixes=("_baseline", "_candidate"),
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(baseline) or len(joined) != len(candidate):
            decisions[arm] = {
                "eligible": False,
                "reason": "prior_row_population_mismatch",
                "baseline_rows": int(len(baseline)),
                "candidate_rows": int(len(candidate)),
                "matched_rows": int(len(joined)),
            }
            continue
        if not np.allclose(
            joined.actual_share_baseline.to_numpy(float),
            joined.actual_share_candidate.to_numpy(float),
            equal_nan=True,
        ):
            decisions[arm] = {"eligible": False, "reason": "actual_label_mismatch"}
            continue
        comparison = bootstrap_mae_delta(
            joined.actual_share_baseline.to_numpy(float),
            joined.predicted_share_candidate.to_numpy(float),
            joined.predicted_share_baseline.to_numpy(float),
            n_bootstrap=n_bootstrap,
        )
        comparison["margin"] = float(margin)
        comparison["eligible"] = bool(comparison["ci_high"] <= margin)
        comparison["reason"] = "noninferior" if comparison["eligible"] else "ci_exceeds_noninferiority_margin"
        decisions[arm] = comparison
        if comparison["eligible"]:
            allowed.append(arm)
    return allowed, decisions


def eligible_allocation_candidates(
    predictions: pd.DataFrame,
    prior_folds: set[int],
    *,
    margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    n_bootstrap: int = INNER_GATE_BOOTSTRAPS,
) -> tuple[list[str], dict[str, dict]]:
    """Add only share-safe convex blends to direct eligible allocation arms.

    Blend weights are not selected from the test fold: each candidate is
    formed from completed OOF folds and must itself clear the same paired
    share-MAE non-inferiority gate as a direct learned arm.  This permits a
    learned arm to contribute incremental signal without allowing it to
    replace rolling-3 wholesale.
    """
    direct, decisions = eligible_allocation_arms(
        predictions, prior_folds, margin=margin, n_bootstrap=n_bootstrap
    )
    if not prior_folds:
        return direct, decisions
    baseline = _allocation_rows(predictions, "rolling3")
    baseline = baseline[baseline.fold.isin(prior_folds)]
    candidates = list(direct)
    for arm in direct:
        if arm == "rolling3":
            continue
        for weight in BLEND_WEIGHTS:
            spec = f"blend:{arm}:{weight:.2f}"
            blended = _allocation_rows(predictions, spec)
            blended = blended[blended.fold.isin(prior_folds)]
            if len(blended) != len(baseline):
                decisions[spec] = {"eligible": False, "reason": "prior_row_population_mismatch"}
                continue
            comparison = bootstrap_mae_delta(
                baseline.actual_share.to_numpy(float),
                blended.predicted_share.to_numpy(float),
                baseline.predicted_share.to_numpy(float),
                n_bootstrap=n_bootstrap,
            )
            comparison["margin"] = float(margin)
            comparison["eligible"] = bool(comparison["ci_high"] <= margin)
            comparison["reason"] = "noninferior" if comparison["eligible"] else "ci_exceeds_noninferiority_margin"
            decisions[spec] = comparison
            if comparison["eligible"]:
                candidates.append(spec)
    return candidates, decisions


def select_walk_forward_arms(
    allocation_results: Mapping[str, dict],
    team_results: Mapping[str, dict],
    *,
    truth: pd.DataFrame,
    noninferiority_margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    inner_gate_bootstraps: int = INNER_GATE_BOOTSTRAPS,
) -> dict:
    """Select PPR arms from completed folds, subject to share safety gates."""
    fold_values = sorted(
        set.intersection(*[
            set(allocation_results[t]["predictions"].fold.unique()) for t in PPR_RECONSTRUCTION_TARGETS
        ])
    )
    # Fold zero has no completed OOF evidence.  Start from the incumbent
    # instead of granting an unvalidated learned arm a free test fold.
    alloc_choices = {target: "rolling3" for target in PPR_RECONSTRUCTION_TARGETS}
    team_choices = {target: "rolling3" for target in PPR_RECONSTRUCTION_TARGETS}
    fold_reports = []
    for fold in fold_values:
        prior = {f for f in fold_values if f < fold}
        eligibility = {}
        if prior:
            # Coordinate descent over the actual PPR objective.  Each choice
            # is evaluated on prior OOF folds only; the current fold remains
            # untouched until the final scoring call below.
            for target in PPR_RECONSTRUCTION_TARGETS:
                candidates, decisions = eligible_allocation_candidates(
                    allocation_results[target]["predictions"],
                    prior,
                    margin=noninferiority_margin,
                    n_bootstrap=inner_gate_bootstraps,
                )
                eligibility[target] = decisions
                best_arm, best_score = alloc_choices[target], np.inf
                for candidate in candidates:
                    trial = dict(alloc_choices)
                    trial[target] = candidate
                    score, _ = _ppr_for_choices(
                        allocation_results, team_results, trial, team_choices, prior, truth
                    )
                    if score < best_score:
                        best_arm, best_score = candidate, score
                alloc_choices[target] = best_arm
            for target in PPR_RECONSTRUCTION_TARGETS:
                candidates = team_results[target]["predictions"].columns.intersection(
                    ["rolling3", "ridge", "xgb", "blend"]
                ).tolist()
                best_arm, best_score = team_choices[target], np.inf
                for candidate in candidates:
                    trial = dict(team_choices)
                    trial[target] = candidate
                    score, _ = _ppr_for_choices(
                        allocation_results, team_results, alloc_choices, trial, prior, truth
                    )
                    if score < best_score:
                        best_arm, best_score = candidate, score
                team_choices[target] = best_arm
        else:
            eligibility = {
                target: {"rolling3": {"eligible": True, "reason": "incumbent_no_prior_evidence"}}
                for target in PPR_RECONSTRUCTION_TARGETS
            }

        current_score, n = _ppr_for_choices(
            allocation_results, team_results, alloc_choices, team_choices, {fold}, truth
        )
        fold_reports.append({
            "fold": int(fold),
            "allocation_arms": dict(alloc_choices),
            "allocation_eligibility": eligibility,
            "team_total_arms": dict(team_choices),
            "ppr_mae": current_score,
            "n": n,
        })
    return {
        "noninferiority_margin": noninferiority_margin,
        "inner_gate_bootstraps": inner_gate_bootstraps,
        "folds": fold_reports,
        "final_allocation_arms": alloc_choices,
        "final_team_total_arms": team_choices,
    }


def score_selector_report(allocation_results: Mapping[str, dict], team_results: Mapping[str, dict],
                          selector_report: dict, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score each frozen fold choice and the rolling-3 incumbent on exact rows."""
    selected_parts = []
    baseline_parts = []
    incumbent = dict.fromkeys(PPR_RECONSTRUCTION_TARGETS, "rolling3")
    for row in selector_report["folds"]:
        fold = {int(row["fold"])}
        selected_parts.append(_ppr_for_choices(allocation_results, team_results,
                                               row["allocation_arms"], row["team_total_arms"],
                                               fold, truth, return_rows=True))
        baseline_parts.append(_ppr_for_choices(allocation_results, team_results,
                                               incumbent, incumbent, fold, truth, return_rows=True))
    selected = pd.concat(selected_parts, ignore_index=True)
    baseline = pd.concat(baseline_parts, ignore_index=True)
    joined = selected.merge(baseline, on=KEY, how="outer", validate="one_to_one",
                            suffixes=("_selected", "_baseline"), indicator=True)
    if not joined._merge.eq("both").all() or not np.allclose(joined.actual_ppr_selected, joined.actual_ppr_baseline):
        raise ValueError("selected and baseline PPR populations differ")
    joined = joined.drop(columns=["_merge", "actual_ppr_baseline"]).rename(columns={"actual_ppr_selected": "actual_ppr"})
    joined = joined.merge(truth[KEY + ["negative_yards", "off_position_stat"]], on=KEY,
                          how="left", validate="one_to_one")
    joined["selected_abs_error"] = (joined.actual_ppr - joined.predicted_ppr_selected).abs()
    joined["baseline_abs_error"] = (joined.actual_ppr - joined.predicted_ppr_baseline).abs()

    def metrics(part: pd.DataFrame) -> dict:
        return {"n": len(part), "selected_mae": float(part.selected_abs_error.mean()),
                "baseline_mae": float(part.baseline_abs_error.mean()),
                "delta_selected_minus_baseline": float((part.selected_abs_error - part.baseline_abs_error).mean())}

    report = {
        "pooled": metrics(joined),
        "folds": {str(int(k)): metrics(v) for k, v in joined.groupby("fold")},
        "positions": {str(k): metrics(v) for k, v in joined.groupby("position")},
        "negative_yards": {str(k): metrics(v) for k, v in joined.groupby("negative_yards")},
        "off_position_stat": {str(k): metrics(v) for k, v in joined.groupby("off_position_stat")},
        "paired_bootstrap": bootstrap_mae_delta(joined.actual_ppr.to_numpy(float),
                                                  joined.predicted_ppr_selected.to_numpy(float),
                                                  joined.predicted_ppr_baseline.to_numpy(float), n_bootstrap=500),
    }
    return joined, report
