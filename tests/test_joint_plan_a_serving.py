import numpy as np
import pandas as pd
import pytest

from src.models.team_allocation.joint_serving import (
    JointPlanAArtifact, _fit_team_arm, _predict_share_arm, _predict_team_arm,
)
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS


class ZeroModel:
    def predict(self, rows):
        return np.zeros(len(rows))


def _rows():
    return pd.DataFrame({
        "team": ["A", "A", "B"], "season": [2025] * 3, "week": [1] * 3,
        "share_of_team_passing_tds_roll3": [0.0] * 3,
        "share_of_team_receiving_yards_roll3": [0.0] * 3,
        "team_passing_tds_roll3": [np.nan, np.nan, 1.25],
    })


def test_sparse_learned_zero_sum_matches_backtester_zero_fallback():
    rows = _rows()
    payload = {"target": "passing_tds", "kind": "ridge", "arm": "ridge_blend_renorm",
               "roll_column": "share_of_team_passing_tds_roll3", "feature_columns": [],
               "model": ZeroModel(), "blend_alpha": 0.5, "renormalize": True}
    # No explicit zero_fallback simulates a previously frozen v1 artifact.
    assert np.array_equal(_predict_share_arm(payload, rows), np.zeros(3))
    outer = {"target": "passing_tds", "kind": "outer_rolling3_blend", "arm": "blend:ridge_blend_renorm:0.05",
             "roll_column": "share_of_team_passing_tds_roll3", "weight": 0.05, "candidate": payload}
    assert np.array_equal(_predict_share_arm(outer, rows), np.zeros(3))


def test_rolling3_renorm_and_dense_learned_keep_equal_fallback():
    rows = _rows()
    rolling = {"target": "passing_tds", "kind": "rolling3", "arm": "rolling3_renorm",
               "roll_column": "share_of_team_passing_tds_roll3", "renormalize": True}
    dense = {"target": "receiving_yards", "kind": "ridge", "arm": "ridge_blend_renorm",
             "roll_column": "share_of_team_receiving_yards_roll3", "feature_columns": [],
             "model": ZeroModel(), "blend_alpha": 0.5, "renormalize": True}
    assert np.allclose(_predict_share_arm(rolling, rows), [0.5, 0.5, 1.0])
    assert np.allclose(_predict_share_arm(dense, rows), [0.5, 0.5, 1.0])


def test_rolling3_team_arm_is_artifact_backed_and_preserves_cold_start_null():
    rows = _rows()
    payload = _fit_team_arm(rows, "passing_tds", "rolling3")
    assert payload["arm"] == "rolling3"
    assert payload["roll_column"] == "team_passing_tds_roll3"
    assert "models" not in payload
    pred = _predict_team_arm(payload, rows)
    assert np.isnan(pred[:2]).all()
    assert pred[2] == pytest.approx(1.25)
    with pytest.raises(ValueError, match="missing rolling team-total"):
        _predict_team_arm(payload, rows.drop(columns="team_passing_tds_roll3"))


def test_fold_zero_all_rolling3_artifact_roundtrips(tmp_path):
    rows = pd.DataFrame({
        "player_id": ["q", "r", "w"], "position": ["QB", "RB", "WR"],
        "season": [2025] * 3, "week": [2] * 3, "team": ["A"] * 3,
    })
    for target in PPR_RECONSTRUCTION_TARGETS:
        rows[target] = 0.0
        rows[f"share_of_team_{target}_roll3"] = 0.0
        rows[f"team_{target}_roll3"] = 1.0
    arms = dict.fromkeys(PPR_RECONSTRUCTION_TARGETS, "rolling3")
    artifact = JointPlanAArtifact.fit(rows, allocation_arms=arms, team_total_arms=arms,
                                      train_through=2025, version="test-fold0", source_selector="test")
    path = tmp_path / "fold0.joblib"
    artifact.save(path)
    out = JointPlanAArtifact.load(path).predict(rows, include_actual=False)
    assert len(out) == 3
    assert out.predicted_ppr.eq(0).all()
    assert set(artifact.payload["team_totals"][t]["arm"] for t in PPR_RECONSTRUCTION_TARGETS) == {"rolling3"}
