"""Tests for Directive V7 Section 18: Staged Deployment Pipeline."""
import json
import tempfile
from pathlib import Path

import pytest

from src.deployment.staged_deployment import (
    DeploymentStage,
    StagedDeploymentManager,
)


@pytest.fixture
def deployer(tmp_path):
    return StagedDeploymentManager(deploy_dir=tmp_path)


class TestShadowDeployment:
    def test_start_shadow(self, deployer):
        state = deployer.start_shadow(
            position="QB",
            candidate_model_path="/models/qb_candidate.joblib",
            candidate_label="v2.1",
            metadata={"rmse": 5.2},
        )
        assert state["stage"] == DeploymentStage.SHADOW
        assert "QB" in state["candidates"]
        assert state["candidates"]["QB"]["label"] == "v2.1"

    def test_shadow_predictions_logging(self, deployer):
        deployer.start_shadow("QB", "/models/qb.joblib")
        deployer.log_shadow_predictions(
            position="QB",
            week=1,
            production_predictions={"p1": 20.5, "p2": 15.3},
            candidate_predictions={"p1": 21.0, "p2": 14.8},
            actuals={"p1": 22.0, "p2": 16.0},
        )
        assert deployer.predictions_log.exists()
        state = deployer.get_state()
        assert state["candidates"]["QB"]["weeks_observed"] == 1

    def test_evaluate_insufficient_weeks(self, deployer):
        deployer.start_shadow("QB", "/models/qb.joblib")
        result = deployer.evaluate_shadow("QB", min_weeks=3)
        assert result["promote_candidate"] is False
        assert "Insufficient" in result.get("reason", "")


class TestPromotion:
    def test_promote_to_production(self, deployer):
        deployer.start_shadow("RB", "/models/rb.joblib")
        result = deployer.promote_to_production("RB")
        assert "promoted_at" in result
        state = deployer.get_state()
        assert state["candidates"]["RB"]["stage"] == DeploymentStage.PRODUCTION

    def test_rollback(self, deployer):
        deployer.start_shadow("WR", "/models/wr.joblib")
        deployer.promote_to_production("WR")
        result = deployer.rollback("WR", reason="RMSE degradation")
        assert result["rolled_back"] is True
        state = deployer.get_state()
        assert state["candidates"]["WR"]["stage"] == DeploymentStage.ROLLED_BACK


class TestDeploymentState:
    def test_empty_state(self, deployer):
        state = deployer.get_state()
        assert state["stage"] == DeploymentStage.PRODUCTION

    def test_summary(self, deployer):
        deployer.start_shadow("QB", "/models/qb.joblib")
        summary = deployer.get_deployment_summary()
        assert summary["current_stage"] == DeploymentStage.SHADOW
        assert "QB" in summary["active_shadows"]
        assert summary["n_candidates"] == 1
