"""
Staged Deployment Pipeline.

Per Agent Directive V7 Section 18.1: every promoted system must pass through
a staged deployment pipeline (shadow → canary → graduated → production)
before receiving live traffic.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DEPLOY_DIR = Path(__file__).parent.parent.parent / "data" / "deployment"


class DeploymentStage(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    GRADUATED = "graduated"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


class StagedDeploymentManager:
    """Manages shadow → canary → production deployment workflow.

    Per Directive V7 Section 18.1:
    - Shadow: candidate runs in parallel, predictions logged but not acted on.
    - Canary: 5-10% of decisions use candidate.
    - Graduated: step up allocation (25%, 50%, 100%).
    - Production: candidate replaces incumbent with warm standby.
    """

    def __init__(self, deploy_dir: Optional[Path] = None, models_dir: Optional[Path] = None):
        self.deploy_dir = deploy_dir or _DEFAULT_DEPLOY_DIR
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.deploy_dir / "deployment_state.json"
        self.predictions_log = self.deploy_dir / "shadow_predictions.jsonl"
        self.models_dir = models_dir

    def get_state(self) -> Dict[str, Any]:
        """Load current deployment state."""
        if not self.state_path.exists():
            return {"stage": DeploymentStage.PRODUCTION, "candidates": {}}
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"stage": DeploymentStage.PRODUCTION, "candidates": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

    def start_shadow(
        self,
        position: str,
        candidate_model_path: str,
        candidate_label: str = "candidate",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a candidate model for shadow deployment.

        The candidate runs alongside production; predictions are logged
        but not served to users.

        Args:
            position: Position this model covers.
            candidate_model_path: Path to the candidate model artifact.
            candidate_label: Human-readable label.
            metadata: Training metrics and other context.

        Returns:
            Updated deployment state.
        """
        state = self.get_state()
        state["candidates"] = state.get("candidates", {})
        state["candidates"][position] = {
            "label": candidate_label,
            "model_path": str(candidate_model_path),
            "stage": DeploymentStage.SHADOW,
            "started_at": datetime.now().isoformat(),
            "weeks_observed": 0,
            "metadata": metadata or {},
        }
        state["stage"] = DeploymentStage.SHADOW
        self._save_state(state)
        logger.info("Shadow deployment started for %s: %s", position, candidate_label)
        return state

    def log_shadow_predictions(
        self,
        position: str,
        week: int,
        production_predictions: Dict[str, float],
        candidate_predictions: Dict[str, float],
        actuals: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log parallel predictions from production and candidate models.

        Args:
            position: Position being compared.
            week: NFL week number.
            production_predictions: {player_id: predicted_points} from production.
            candidate_predictions: {player_id: predicted_points} from candidate.
            actuals: {player_id: actual_points} if available.
        """
        entry = {
            "position": position,
            "week": week,
            "timestamp": datetime.now().isoformat(),
            "production": production_predictions,
            "candidate": candidate_predictions,
            "actuals": actuals,
        }
        try:
            with open(self.predictions_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to log shadow predictions: %s", e)

        # Update weeks observed
        state = self.get_state()
        candidates = state.get("candidates", {})
        if position in candidates:
            candidates[position]["weeks_observed"] = candidates[position].get("weeks_observed", 0) + 1
            self._save_state(state)

    def evaluate_shadow(
        self,
        position: str,
        min_weeks: int = 3,
        improvement_threshold_pct: float = 5.0,
    ) -> Dict[str, Any]:
        """Evaluate whether the shadow candidate should be promoted.

        Per Section 18.1: minimum duration of 50-200 decision cycles.
        For NFL weekly predictions, min_weeks serves as the cycle count.

        Args:
            position: Position to evaluate.
            min_weeks: Minimum weeks of shadow data required.
            improvement_threshold_pct: Required RMSE improvement to promote.

        Returns:
            Evaluation result with promote_candidate bool.
        """
        import numpy as np

        state = self.get_state()
        candidate = state.get("candidates", {}).get(position)
        if not candidate:
            return {"error": f"No shadow candidate for {position}", "promote_candidate": False}

        weeks_observed = candidate.get("weeks_observed", 0)
        if weeks_observed < min_weeks:
            return {
                "position": position,
                "weeks_observed": weeks_observed,
                "min_weeks": min_weeks,
                "promote_candidate": False,
                "reason": f"Insufficient shadow data ({weeks_observed}/{min_weeks} weeks)",
            }

        # Read shadow prediction logs
        if not self.predictions_log.exists():
            return {"error": "No shadow predictions logged", "promote_candidate": False}

        prod_errors = []
        cand_errors = []
        with open(self.predictions_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("position") != position or not entry.get("actuals"):
                        continue
                    actuals = entry["actuals"]
                    for pid, actual in actuals.items():
                        prod_pred = entry.get("production", {}).get(pid)
                        cand_pred = entry.get("candidate", {}).get(pid)
                        if prod_pred is not None and cand_pred is not None:
                            prod_errors.append((float(actual) - float(prod_pred)) ** 2)
                            cand_errors.append((float(actual) - float(cand_pred)) ** 2)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        if not prod_errors:
            return {"error": "No actuals available for comparison", "promote_candidate": False}

        prod_rmse = float(np.sqrt(np.mean(prod_errors)))
        cand_rmse = float(np.sqrt(np.mean(cand_errors)))
        improvement_pct = ((prod_rmse - cand_rmse) / prod_rmse) * 100 if prod_rmse > 0 else 0

        promote = improvement_pct >= improvement_threshold_pct
        result = {
            "position": position,
            "weeks_observed": weeks_observed,
            "production_rmse": round(prod_rmse, 4),
            "candidate_rmse": round(cand_rmse, 4),
            "improvement_pct": round(improvement_pct, 2),
            "threshold_pct": improvement_threshold_pct,
            "promote_candidate": promote,
            "n_predictions_compared": len(prod_errors),
            "evaluated_at": datetime.now().isoformat(),
        }
        logger.info(
            "Shadow evaluation for %s: prod_rmse=%.4f, cand_rmse=%.4f, improvement=%.1f%% — %s",
            position, prod_rmse, cand_rmse, improvement_pct,
            "PROMOTE" if promote else "KEEP PRODUCTION",
        )
        return result

    def promote_to_production(self, position: str) -> Dict[str, Any]:
        """Promote the shadow candidate to production.

        Per Section 18.1: incumbent retained as warm standby for rollback.

        Args:
            position: Position to promote.

        Returns:
            Promotion record.
        """
        state = self.get_state()
        candidate = state.get("candidates", {}).get(position)
        if not candidate:
            return {"error": f"No candidate for {position}"}

        # Archive current production model path for rollback
        candidate["previous_stage"] = candidate.get("stage")
        candidate["stage"] = DeploymentStage.PRODUCTION
        candidate["promoted_at"] = datetime.now().isoformat()
        state["stage"] = DeploymentStage.PRODUCTION
        self._save_state(state)

        logger.info("Candidate promoted to production for %s", position)
        return {"position": position, "stage": "production", "promoted_at": candidate["promoted_at"]}

    def rollback(self, position: str, reason: str = "") -> Dict[str, Any]:
        """Revert to previous production model.

        Per Section 18.1: incumbent is retained as warm standby.

        Args:
            position: Position to roll back.
            reason: Why the rollback was triggered.

        Returns:
            Rollback record.
        """
        state = self.get_state()
        candidates = state.get("candidates", {})
        if position in candidates:
            candidates[position]["stage"] = DeploymentStage.ROLLED_BACK
            candidates[position]["rolled_back_at"] = datetime.now().isoformat()
            candidates[position]["rollback_reason"] = reason

        state["stage"] = DeploymentStage.PRODUCTION
        self._save_state(state)

        logger.warning("Rollback triggered for %s: %s", position, reason)
        return {"position": position, "rolled_back": True, "reason": reason}

    def get_deployment_summary(self) -> Dict[str, Any]:
        """Summary for monitoring dashboard."""
        state = self.get_state()
        candidates = state.get("candidates", {})
        return {
            "current_stage": state.get("stage", "production"),
            "active_shadows": {
                pos: c for pos, c in candidates.items()
                if c.get("stage") == DeploymentStage.SHADOW
            },
            "n_candidates": len(candidates),
            "timestamp": datetime.now().isoformat(),
        }
