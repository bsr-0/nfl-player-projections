"""
A/B Testing Framework for Model Updates (Requirements Section VI.D).

Provides shadow-mode evaluation where a new candidate model runs alongside
the production model. Predictions from both are logged and compared after
a configurable number of weeks. If the candidate improves RMSE by the
required threshold (default 5%), it becomes the new production model;
otherwise the update is rolled back.

Usage:
    ab = ABTestManager("QB")
    ab.register_candidate(new_model, label="v2.1")
    # Each prediction week:
    ab.log_predictions(week, player_ids, prod_preds, cand_preds, actuals)
    # After evaluation window:
    result = ab.evaluate()
    if result["promote_candidate"]:
        ab.promote()
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR


class ABTestManager:
    """Manages A/B testing between production and candidate models."""

    def __init__(
        self,
        position: str,
        min_weeks: int = 3,
        improvement_threshold_pct: float = 5.0,
        log_dir: Optional[Path] = None,
    ):
        self.position = position
        self.min_weeks = min_weeks
        self.improvement_threshold_pct = improvement_threshold_pct
        self.log_dir = log_dir or DATA_DIR / "ab_tests"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.candidate_label: Optional[str] = None
        self.candidate_model = None
        self.weekly_logs: List[Dict] = []

    def register_candidate(self, model, label: str = "candidate"):
        """Register a candidate model to test against production."""
        self.candidate_label = label
        self.candidate_model = model
        self.weekly_logs = []

    def log_predictions(
        self,
        week: int,
        player_ids: np.ndarray,
        production_preds: np.ndarray,
        candidate_preds: np.ndarray,
        actuals: np.ndarray,
    ):
        """Log one week of shadow-mode predictions for later comparison."""
        from sklearn.metrics import mean_squared_error

        prod_rmse = float(np.sqrt(mean_squared_error(actuals, production_preds)))
        cand_rmse = float(np.sqrt(mean_squared_error(actuals, candidate_preds)))

        self.weekly_logs.append({
            "week": int(week),
            "n_players": len(player_ids),
            "production_rmse": round(prod_rmse, 3),
            "candidate_rmse": round(cand_rmse, 3),
            "timestamp": datetime.now().isoformat(),
        })

    def evaluate(self) -> Dict:
        """Evaluate candidate vs production after min_weeks of data.

        Returns dict with recommendation to promote or rollback.
        """
        if len(self.weekly_logs) < self.min_weeks:
            return {
                "ready": False,
                "weeks_logged": len(self.weekly_logs),
                "weeks_needed": self.min_weeks,
            }

        prod_rmse_avg = np.mean([w["production_rmse"] for w in self.weekly_logs])
        cand_rmse_avg = np.mean([w["candidate_rmse"] for w in self.weekly_logs])

        improvement_pct = (
            (prod_rmse_avg - cand_rmse_avg) / prod_rmse_avg * 100
            if prod_rmse_avg > 0
            else 0.0
        )

        promote = improvement_pct >= self.improvement_threshold_pct

        result = {
            "ready": True,
            "position": self.position,
            "candidate_label": self.candidate_label,
            "weeks_evaluated": len(self.weekly_logs),
            "production_rmse_avg": round(float(prod_rmse_avg), 3),
            "candidate_rmse_avg": round(float(cand_rmse_avg), 3),
            "improvement_pct": round(float(improvement_pct), 1),
            "threshold_pct": self.improvement_threshold_pct,
            "promote_candidate": promote,
            "recommendation": "PROMOTE" if promote else "ROLLBACK",
        }

        # Persist evaluation
        out_path = self.log_dir / f"ab_eval_{self.position}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(out_path, "w") as f:
            json.dump({"result": result, "weekly_logs": self.weekly_logs}, f, indent=2)

        return result

    def promote(self) -> bool:
        """Save candidate model as the new production model.

        Returns True if promotion succeeded.
        """
        if self.candidate_model is None:
            return False
        try:
            save_path = MODELS_DIR / f"multiweek_{self.position.lower()}.joblib"
            if hasattr(self.candidate_model, "save"):
                self.candidate_model.save(save_path)
            else:
                import joblib
                joblib.dump(self.candidate_model, save_path)
            return True
        except Exception:
            return False
