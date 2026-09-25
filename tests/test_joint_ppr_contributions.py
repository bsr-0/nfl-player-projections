import pandas as pd

from src.evaluation.joint_ppr_contributions import component_contribution_report
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS


def test_component_attribution_reverts_only_one_component_on_same_rows():
    keys = pd.DataFrame({
        "player_id": ["p1", "p2"], "season": [2025, 2025], "week": [1, 1],
        "team": ["A", "A"], "position": ["RB", "RB"], "fold": [0, 0],
    })
    allocation, totals = {}, {}
    for target in PPR_RECONSTRUCTION_TARGETS:
        actual_share = [1.0, 0.0]
        prediction = keys.copy()
        if target in {"passing_yards", "passing_tds", "interceptions"}:
            prediction = prediction.iloc[:0].copy()
            actual_share = []
        prediction["actual_share"] = actual_share
        prediction["arm"] = "rolling3"
        prediction["predicted_share"] = [0.0, 1.0][:len(prediction)]
        selected = prediction.copy()
        selected["arm"] = "selected"
        selected["predicted_share"] = actual_share
        allocation[target] = {"predictions": pd.concat([prediction, selected], ignore_index=True)}
        team = pd.DataFrame({
            "fold": [0], "team": ["A"], "season": [2025], "week": [1],
            "actual": [1.0], "rolling3": [0.0], "ridge": [1.0],
        })
        totals[target] = {"predictions": team}
    selector = {"folds": [{
        "fold": 0,
        "allocation_arms": {target: "selected" for target in PPR_RECONSTRUCTION_TARGETS},
        "team_total_arms": {target: "ridge" for target in PPR_RECONSTRUCTION_TARGETS},
    }]}
    truth = keys.assign(actual_ppr=[13.2, 0.0])
    report = component_contribution_report(allocation, totals, selector, truth)
    row = next(
        value for value in report["folds"]
        if value["target"] == "rushing_yards" and value["dimension"] == "allocation"
    )
    assert row["n"] == 2
    assert row["delta_mae_when_reverted"] > 0
