import numpy as np
import pandas as pd

from src.evaluation.joint_ppr_selector import (
    _allocation_rows,
    eligible_allocation_arms,
    eligible_allocation_candidates,
    select_walk_forward_arms,
    PPR_RECONSTRUCTION_TARGETS,
)


def test_share_gate_keeps_only_noninferior_prior_fold_arms():
    n = 40
    base = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "season": [2024] * n,
        "week": list(range(1, n + 1)),
        "team": ["A"] * n,
        "position": ["RB"] * n,
        "fold": [0] * n,
        "actual_share": np.linspace(0.0, 1.0, n),
    })
    rolling = base.assign(arm="rolling3", predicted_share=base.actual_share)
    safe = base.assign(arm="safe", predicted_share=base.actual_share)
    unsafe = base.assign(arm="unsafe", predicted_share=np.clip(base.actual_share + 0.1, 0, 1))
    allowed, decisions = eligible_allocation_arms(
        pd.concat([rolling, safe, unsafe], ignore_index=True), {0}, n_bootstrap=100
    )

    assert set(allowed) == {"rolling3", "safe"}
    assert decisions["safe"]["eligible"] is True
    assert decisions["unsafe"]["eligible"] is False
    assert decisions["unsafe"]["reason"] == "ci_exceeds_noninferiority_margin"


def test_share_gate_defaults_to_incumbent_without_prior_folds():
    predictions = pd.DataFrame(columns=["fold", "arm"])
    allowed, decisions = eligible_allocation_arms(predictions, set())
    assert allowed == ["rolling3"]
    assert decisions["rolling3"]["reason"] == "incumbent_no_prior_evidence"


def test_safe_convex_blends_are_auditable_and_preserve_row_population():
    n = 40
    base = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)], "season": [2024] * n,
        "week": list(range(1, n + 1)), "team": ["A"] * n,
        "position": ["RB"] * n, "fold": [0] * n,
        "actual_share": np.linspace(0.0, 1.0, n),
    })
    rolling = base.assign(arm="rolling3", predicted_share=base.actual_share + 0.01)
    learned = base.assign(arm="learned", predicted_share=base.actual_share)
    predictions = pd.concat([rolling, learned], ignore_index=True)
    allowed, decisions = eligible_allocation_candidates(predictions, {0}, n_bootstrap=100)
    specs = [arm for arm in allowed if arm.startswith("blend:learned:")]
    assert specs
    rows = _allocation_rows(predictions, specs[0])
    assert len(rows) == n
    assert decisions[specs[0]]["eligible"] is True


def test_future_fold_truth_cannot_change_prior_fold_selection(monkeypatch):
    import src.evaluation.joint_ppr_selector as selector

    keys = pd.DataFrame({
        "player_id": ["p0", "p1", "p2"], "season": [2023, 2024, 2025],
        "week": [1, 1, 1], "team": ["A"] * 3, "position": ["RB"] * 3,
        "fold": [0, 1, 2],
    })
    prediction = keys.assign(actual_share=0.5, predicted_share=0.5, arm="rolling3")
    learned = prediction.assign(arm="ridge_blend_renorm")
    allocation = {t: {"predictions": pd.concat([prediction, learned], ignore_index=True)}
                  for t in PPR_RECONSTRUCTION_TARGETS}
    totals = {t: {"predictions": keys[["fold", "team", "season", "week"]].assign(
        rolling3=1.0, ridge=1.0, xgb=1.0, blend=1.0)} for t in PPR_RECONSTRUCTION_TARGETS}

    def scorer(_a, _t, allocation_arms, _team_arms, folds, truth):
        prior_value = truth[truth.fold.isin(folds)].actual_ppr.sum()
        return (float(prior_value if allocation_arms["rushing_yards"] == "rolling3" else -prior_value), len(folds))

    monkeypatch.setattr(selector, "_ppr_for_choices", scorer)
    truth = keys.assign(actual_ppr=[1.0, 2.0, 3.0])
    first = select_walk_forward_arms(allocation, totals, truth=truth, inner_gate_bootstraps=10)
    changed = select_walk_forward_arms(allocation, totals, truth=truth.assign(actual_ppr=[1.0, 2.0, 999.0]),
                                       inner_gate_bootstraps=10)
    assert first["folds"][:2] == changed["folds"][:2]
