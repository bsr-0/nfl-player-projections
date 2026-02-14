"""
A/B Testing Framework for Model Updates (Requirements Section VI.D).

Provides shadow-mode evaluation where a new candidate model runs alongside
the production model. Predictions from both are logged and compared after
a configurable number of weeks. If the candidate improves RMSE by the
required threshold (default 5%) with statistical significance, it becomes
the new production model; otherwise the update is rolled back.

Usage:
    ab = ABTestManager("QB")
    ab.register_candidate(new_model, label="v2.1")
    # Each prediction week:
    ab.log_predictions(week, player_ids, prod_preds, cand_preds, actuals)
    # After evaluation window:
    result = ab.evaluate()
    if result["promote_candidate"]:
        ab.promote()
    # To revert a bad promotion:
    ab.rollback()
"""
import json
import shutil
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
        significance_alpha: float = 0.05,
        log_dir: Optional[Path] = None,
    ):
        self.position = position
        self.min_weeks = min_weeks
        self.improvement_threshold_pct = improvement_threshold_pct
        self.significance_alpha = significance_alpha
        self.log_dir = log_dir or DATA_DIR / "ab_tests"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.candidate_label: Optional[str] = None
        self.candidate_model = None
        self.weekly_logs: List[Dict] = []
        # Per-player prediction logs for statistical tests
        self._player_errors_prod: List[np.ndarray] = []
        self._player_errors_cand: List[np.ndarray] = []

    def register_candidate(self, model, label: str = "candidate"):
        """Register a candidate model to test against production."""
        self.candidate_label = label
        self.candidate_model = model
        self.weekly_logs = []
        self._player_errors_prod = []
        self._player_errors_cand = []

    def log_predictions(
        self,
        week: int,
        player_ids: np.ndarray,
        production_preds: np.ndarray,
        candidate_preds: np.ndarray,
        actuals: np.ndarray,
    ):
        """Log one week of shadow-mode predictions for later comparison.
        
        Stores both aggregate RMSE per week and per-player absolute errors
        for statistical significance testing.
        """
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
        
        # Store per-player absolute errors for significance tests
        self._player_errors_prod.append(np.abs(np.asarray(actuals) - np.asarray(production_preds)))
        self._player_errors_cand.append(np.abs(np.asarray(actuals) - np.asarray(candidate_preds)))

    def _run_significance_test(self) -> Dict:
        """Run Wilcoxon signed-rank test on paired per-player errors.
        
        Returns dict with p_value and is_significant flag.
        """
        if not self._player_errors_prod or not self._player_errors_cand:
            return {"p_value": 1.0, "is_significant": False, "test": "none"}
        
        prod_all = np.concatenate(self._player_errors_prod)
        cand_all = np.concatenate(self._player_errors_cand)
        
        if len(prod_all) < 10:
            return {"p_value": 1.0, "is_significant": False, "test": "insufficient_data"}
        
        try:
            from scipy.stats import wilcoxon
            # Test if candidate errors are significantly different from production
            stat, p_value = wilcoxon(prod_all, cand_all, alternative='greater')
            return {
                "p_value": float(p_value),
                "is_significant": p_value < self.significance_alpha,
                "test": "wilcoxon_signed_rank",
                "statistic": float(stat),
                "n_samples": len(prod_all),
            }
        except ImportError:
            # Fall back to paired t-test if scipy not available
            try:
                diff = prod_all - cand_all
                mean_diff = np.mean(diff)
                se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
                if se_diff > 0:
                    t_stat = mean_diff / se_diff
                    # Approximate p-value using normal distribution for large n
                    from math import erfc, sqrt
                    p_value = 0.5 * erfc(t_stat / sqrt(2))
                    return {
                        "p_value": float(p_value),
                        "is_significant": p_value < self.significance_alpha,
                        "test": "paired_t_approx",
                        "t_statistic": float(t_stat),
                        "n_samples": len(prod_all),
                    }
            except Exception:
                pass
            return {"p_value": 1.0, "is_significant": False, "test": "fallback_failed"}

    def evaluate(self) -> Dict:
        """Evaluate candidate vs production after min_weeks of data.

        Uses both RMSE improvement threshold AND statistical significance.
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

        # Statistical significance test
        sig_result = self._run_significance_test()
        
        # Promote only if improvement exceeds threshold AND is statistically significant
        meets_threshold = improvement_pct >= self.improvement_threshold_pct
        is_significant = sig_result.get("is_significant", False)
        promote = meets_threshold and is_significant

        result = {
            "ready": True,
            "position": self.position,
            "candidate_label": self.candidate_label,
            "weeks_evaluated": len(self.weekly_logs),
            "production_rmse_avg": round(float(prod_rmse_avg), 3),
            "candidate_rmse_avg": round(float(cand_rmse_avg), 3),
            "improvement_pct": round(float(improvement_pct), 1),
            "threshold_pct": self.improvement_threshold_pct,
            "significance_test": sig_result,
            "meets_threshold": meets_threshold,
            "is_significant": is_significant,
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

        Creates a backup of the current production model before overwriting.
        Returns True if promotion succeeded.
        """
        if self.candidate_model is None:
            return False
        try:
            save_path = MODELS_DIR / f"multiweek_{self.position.lower()}.joblib"
            
            # Backup existing production model before overwriting
            if save_path.exists():
                backup_dir = MODELS_DIR / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"multiweek_{self.position.lower()}_{ts}.joblib"
                shutil.copy2(save_path, backup_path)
            
            if hasattr(self.candidate_model, "save"):
                self.candidate_model.save(save_path)
            else:
                import joblib
                joblib.dump(self.candidate_model, save_path)
            return True
        except Exception:
            return False

    def rollback(self) -> bool:
        """Restore the most recent backup of the production model.
        
        Returns True if rollback succeeded, False if no backup found.
        """
        try:
            backup_dir = MODELS_DIR / "backups"
            if not backup_dir.exists():
                return False
            
            pattern = f"multiweek_{self.position.lower()}_*.joblib"
            backups = sorted(backup_dir.glob(pattern), reverse=True)
            if not backups:
                return False
            
            latest_backup = backups[0]
            save_path = MODELS_DIR / f"multiweek_{self.position.lower()}.joblib"
            shutil.copy2(latest_backup, save_path)
            return True
        except Exception:
            return False
