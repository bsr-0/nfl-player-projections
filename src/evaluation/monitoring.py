"""
Online monitoring and alerting for NFL prediction models.

Provides real-time (per-prediction-batch) monitoring of:
- Prediction distribution shift vs training baselines
- Feature drift (input distribution changes)
- Error rate tracking when actuals become available
- Alert generation when thresholds are exceeded

Alerts are written to a JSON log and can be forwarded to external
systems (email, Slack, PagerDuty) via hooks.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ALERT_DIR = Path(__file__).parent.parent.parent / "data" / "monitoring"


class AlertLevel:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ModelMonitor:
    """Monitor model predictions and inputs for anomalies."""

    def __init__(
        self,
        alert_dir: Optional[Path] = None,
        alert_hooks: Optional[List[Callable[[Dict], None]]] = None,
    ):
        self.alert_dir = alert_dir or _DEFAULT_ALERT_DIR
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.alert_log = self.alert_dir / "alerts.jsonl"
        self.metrics_log = self.alert_dir / "metrics.jsonl"
        self.alert_hooks = alert_hooks or []

        # Thresholds (configurable)
        self.prediction_mean_drift_threshold = 0.30  # 30% shift in mean
        self.prediction_std_drift_threshold = 0.50   # 50% change in spread
        self.feature_drift_ks_threshold = 0.20       # KS statistic threshold
        self.error_rmse_degradation_threshold = 0.25 # 25% RMSE increase
        self.null_prediction_threshold = 0.05        # 5% null predictions

    # ------------------------------------------------------------------
    # Prediction monitoring
    # ------------------------------------------------------------------

    def check_predictions(
        self,
        predictions: np.ndarray,
        position: str = "ALL",
        reference_mean: Optional[float] = None,
        reference_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Check a batch of predictions for anomalies.

        Args:
            predictions: Array of predicted values.
            position: Position label for context.
            reference_mean: Expected mean from training distribution.
            reference_std: Expected std from training distribution.

        Returns:
            List of alert dicts (empty if no issues).
        """
        alerts: List[Dict[str, Any]] = []
        preds = np.asarray(predictions, dtype=float)
        finite = preds[np.isfinite(preds)]

        # Check null/NaN rate
        null_rate = 1.0 - len(finite) / max(len(preds), 1)
        if null_rate > self.null_prediction_threshold:
            alerts.append(self._make_alert(
                AlertLevel.CRITICAL,
                f"High null prediction rate for {position}: {null_rate:.1%}",
                {"null_rate": null_rate, "position": position},
            ))

        if len(finite) < 5:
            return alerts

        batch_mean = float(np.mean(finite))
        batch_std = float(np.std(finite))

        # Mean drift
        if reference_mean is not None and reference_mean > 0:
            drift = abs(batch_mean - reference_mean) / reference_mean
            if drift > self.prediction_mean_drift_threshold:
                alerts.append(self._make_alert(
                    AlertLevel.WARNING,
                    f"Prediction mean drift for {position}: "
                    f"{batch_mean:.2f} vs reference {reference_mean:.2f} ({drift:.1%})",
                    {"batch_mean": batch_mean, "reference_mean": reference_mean,
                     "drift_pct": drift, "position": position},
                ))

        # Spread drift
        if reference_std is not None and reference_std > 0:
            std_drift = abs(batch_std - reference_std) / reference_std
            if std_drift > self.prediction_std_drift_threshold:
                alerts.append(self._make_alert(
                    AlertLevel.WARNING,
                    f"Prediction spread drift for {position}: "
                    f"std={batch_std:.2f} vs reference {reference_std:.2f}",
                    {"batch_std": batch_std, "reference_std": reference_std,
                     "position": position},
                ))

        # Log metrics
        self._log_metrics({
            "timestamp": datetime.now().isoformat(),
            "position": position,
            "batch_size": len(preds),
            "mean": batch_mean,
            "std": batch_std,
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "null_rate": null_rate,
        })

        for alert in alerts:
            self._emit_alert(alert)

        return alerts

    def check_actuals(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        position: str = "ALL",
        reference_rmse: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Check prediction errors when actuals become available."""
        alerts: List[Dict[str, Any]] = []
        preds = np.asarray(predictions, dtype=float)
        acts = np.asarray(actuals, dtype=float)
        valid = np.isfinite(preds) & np.isfinite(acts)
        if valid.sum() < 5:
            return alerts

        errors = acts[valid] - preds[valid]
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        if reference_rmse is not None and reference_rmse > 0:
            degradation = (rmse - reference_rmse) / reference_rmse
            if degradation > self.error_rmse_degradation_threshold:
                alerts.append(self._make_alert(
                    AlertLevel.CRITICAL,
                    f"RMSE degradation for {position}: {rmse:.2f} vs "
                    f"reference {reference_rmse:.2f} ({degradation:+.1%})",
                    {"rmse": rmse, "reference_rmse": reference_rmse,
                     "degradation_pct": degradation, "position": position},
                ))

        self._log_metrics({
            "timestamp": datetime.now().isoformat(),
            "position": position,
            "type": "error_check",
            "rmse": rmse,
            "mae": mae,
            "n_valid": int(valid.sum()),
        })

        for alert in alerts:
            self._emit_alert(alert)

        return alerts

    # ------------------------------------------------------------------
    # Feature monitoring
    # ------------------------------------------------------------------

    def check_feature_drift(
        self,
        current_features: Dict[str, np.ndarray],
        reference_features: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Detect feature distribution drift using KS test."""
        alerts: List[Dict[str, Any]] = []
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            return alerts

        drifted = []
        for feat_name in current_features:
            if feat_name not in reference_features:
                continue
            cur = current_features[feat_name]
            ref = reference_features[feat_name]
            cur_f = cur[np.isfinite(cur)]
            ref_f = ref[np.isfinite(ref)]
            if len(cur_f) < 10 or len(ref_f) < 10:
                continue
            stat, pval = ks_2samp(cur_f, ref_f)
            if stat > self.feature_drift_ks_threshold and pval < 0.01:
                drifted.append({"feature": feat_name, "ks_stat": round(stat, 3),
                                "p_value": round(pval, 6)})

        if drifted:
            drifted.sort(key=lambda x: -x["ks_stat"])
            alerts.append(self._make_alert(
                AlertLevel.WARNING,
                f"{len(drifted)} features show significant drift",
                {"drifted_features": drifted[:10]},
            ))

        for alert in alerts:
            self._emit_alert(alert)

        return alerts

    # ------------------------------------------------------------------
    # Alert summary
    # ------------------------------------------------------------------

    def get_recent_alerts(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Read recent alerts from the log."""
        if not self.alert_log.exists():
            return []
        alerts = []
        with open(self.alert_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return alerts[-last_n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_alert(self, level: str, message: str,
                    details: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "details": details,
        }

    def _emit_alert(self, alert: Dict[str, Any]) -> None:
        """Write alert to log and invoke any registered hooks."""
        try:
            with open(self.alert_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str) + "\n")
        except Exception as e:
            logger.warning("Alert log write failed: %s", e)

        for hook in self.alert_hooks:
            try:
                hook(alert)
            except Exception as e:
                logger.warning("Alert hook failed: %s", e)

        if alert["level"] == AlertLevel.CRITICAL:
            logger.error("ALERT [%s]: %s", alert["level"], alert["message"])
        else:
            logger.warning("ALERT [%s]: %s", alert["level"], alert["message"])

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        try:
            with open(self.metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, default=str) + "\n")
        except Exception:
            pass
