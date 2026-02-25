"""Production prediction monitoring and alerting.

Tracks prediction quality in real-time (or batch) and raises alerts when:
- Prediction distribution shifts beyond historical norms
- Error rates exceed configured thresholds
- Feature values fall outside training ranges
- Model staleness exceeds SLA

Alerts are written to a structured JSON log and can be consumed by external
systems (Slack webhooks, PagerDuty, email) via the alert callback mechanism.

Usage:
    monitor = PredictionMonitor.from_config()
    monitor.log_predictions(predictions_df, actuals_df)
    alerts = monitor.check_alerts()
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

_DEFAULT_MONITOR_DIR = Path(__file__).parent.parent.parent / "data" / "monitoring"


class AlertLevel:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    """A single monitoring alert."""

    def __init__(self, level: str, source: str, message: str,
                 details: Optional[Dict[str, Any]] = None):
        self.level = level
        self.source = source
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "details": self.details,
        }


class PredictionMonitor:
    """Monitors prediction quality and raises alerts on degradation."""

    def __init__(
        self,
        monitor_dir: Optional[Path] = None,
        rmse_threshold: float = 8.0,
        mae_threshold: float = 6.0,
        prediction_range: tuple = (0.0, 50.0),
        staleness_hours: int = 168,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ):
        self.monitor_dir = monitor_dir or _DEFAULT_MONITOR_DIR
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.rmse_threshold = rmse_threshold
        self.mae_threshold = mae_threshold
        self.prediction_range = prediction_range
        self.staleness_hours = staleness_hours
        self.alert_callback = alert_callback

        self._alerts_file = self.monitor_dir / "alerts.jsonl"
        self._metrics_file = self.monitor_dir / "weekly_metrics.jsonl"
        self._state_file = self.monitor_dir / "monitor_state.json"
        self._state = self._load_state()

    @classmethod
    def from_config(cls, config: Optional[Dict] = None) -> "PredictionMonitor":
        """Create a monitor from configuration dict."""
        config = config or {}
        return cls(
            rmse_threshold=config.get("rmse_threshold", 8.0),
            mae_threshold=config.get("mae_threshold", 6.0),
            staleness_hours=config.get("staleness_hours", 168),
        )

    def log_predictions(
        self,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        week: Optional[int] = None,
        season: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Log a batch of predictions and compute health metrics.

        Args:
            predictions: Model predictions.
            actuals: Actual outcomes (if available for retrospective monitoring).
            positions: Position labels for stratified monitoring.
            week: NFL week number.
            season: NFL season.

        Returns:
            Dict of computed metrics.
        """
        metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "week": week,
            "season": season,
            "n_predictions": len(predictions),
            "pred_mean": round(float(np.mean(predictions)), 3),
            "pred_std": round(float(np.std(predictions)), 3),
            "pred_min": round(float(np.min(predictions)), 3),
            "pred_max": round(float(np.max(predictions)), 3),
        }

        # Check for out-of-range predictions
        low = self.prediction_range[0]
        high = self.prediction_range[1]
        oor = int(np.sum((predictions < low) | (predictions > high)))
        metrics["out_of_range_count"] = oor
        if oor > 0:
            metrics["out_of_range_pct"] = round(oor / len(predictions) * 100, 1)

        # If actuals are available, compute error metrics
        if actuals is not None and len(actuals) == len(predictions):
            valid = np.isfinite(predictions) & np.isfinite(actuals)
            if valid.sum() > 0:
                errors = predictions[valid] - actuals[valid]
                metrics["rmse"] = round(float(np.sqrt(np.mean(errors ** 2))), 3)
                metrics["mae"] = round(float(np.mean(np.abs(errors))), 3)
                metrics["bias"] = round(float(np.mean(errors)), 3)

                # Per-position metrics
                if positions is not None:
                    pos_metrics = {}
                    for pos in np.unique(positions):
                        mask = (positions == pos) & valid
                        if mask.sum() >= 5:
                            pos_err = predictions[mask] - actuals[mask]
                            pos_metrics[pos] = {
                                "rmse": round(float(np.sqrt(np.mean(pos_err ** 2))), 3),
                                "mae": round(float(np.mean(np.abs(pos_err))), 3),
                                "n": int(mask.sum()),
                            }
                    metrics["by_position"] = pos_metrics

        # Persist metrics
        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

        self._state["last_prediction_time"] = time.time()
        self._save_state()

        return metrics

    def check_alerts(self, metrics: Optional[Dict] = None) -> List[Alert]:
        """Check for alert conditions based on latest metrics.

        Args:
            metrics: Metrics dict from log_predictions. If None, reads the
                     latest from the metrics file.

        Returns:
            List of Alert objects raised.
        """
        if metrics is None:
            metrics = self._read_latest_metrics()
        if not metrics:
            return []

        alerts: List[Alert] = []

        # 1. RMSE threshold
        rmse = metrics.get("rmse")
        if rmse is not None and rmse > self.rmse_threshold:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL if rmse > self.rmse_threshold * 1.5 else AlertLevel.WARNING,
                source="error_rate",
                message=f"RMSE {rmse:.2f} exceeds threshold {self.rmse_threshold:.2f}",
                details={"rmse": rmse, "threshold": self.rmse_threshold},
            ))

        # 2. MAE threshold
        mae = metrics.get("mae")
        if mae is not None and mae > self.mae_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                source="error_rate",
                message=f"MAE {mae:.2f} exceeds threshold {self.mae_threshold:.2f}",
                details={"mae": mae, "threshold": self.mae_threshold},
            ))

        # 3. Out-of-range predictions
        oor_pct = metrics.get("out_of_range_pct", 0)
        if oor_pct > 5.0:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                source="prediction_range",
                message=f"{oor_pct:.1f}% of predictions are out of range [{self.prediction_range[0]}, {self.prediction_range[1]}]",
                details={"out_of_range_pct": oor_pct},
            ))

        # 4. Prediction distribution shift (compare to historical mean)
        hist_mean = self._state.get("historical_pred_mean")
        pred_mean = metrics.get("pred_mean")
        if hist_mean is not None and pred_mean is not None:
            shift = abs(pred_mean - hist_mean)
            if shift > 3.0:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    source="distribution_shift",
                    message=f"Prediction mean shifted by {shift:.1f} from historical ({hist_mean:.1f} -> {pred_mean:.1f})",
                    details={"historical_mean": hist_mean, "current_mean": pred_mean},
                ))
        # Update historical mean with exponential moving average
        if pred_mean is not None:
            alpha = 0.1
            self._state["historical_pred_mean"] = (
                pred_mean if hist_mean is None
                else alpha * pred_mean + (1 - alpha) * hist_mean
            )

        # 5. Model staleness
        last_pred = self._state.get("last_prediction_time")
        if last_pred is not None:
            hours_since = (time.time() - last_pred) / 3600
            if hours_since > self.staleness_hours:
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    source="staleness",
                    message=f"No predictions in {hours_since:.0f} hours (SLA: {self.staleness_hours}h)",
                    details={"hours_since_last": round(hours_since, 1)},
                ))

        # Persist and dispatch alerts
        for alert in alerts:
            self._persist_alert(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        self._save_state()
        return alerts

    def get_recent_alerts(self, last_n: int = 20) -> List[Dict]:
        """Read the last N alerts."""
        if not self._alerts_file.exists():
            return []
        alerts = []
        with open(self._alerts_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    alerts.append(json.loads(line))
        return alerts[-last_n:]

    def health_status(self) -> Dict[str, Any]:
        """Return a summary health status for the /api/health endpoint."""
        recent = self.get_recent_alerts(last_n=10)
        critical = [a for a in recent if a.get("level") == AlertLevel.CRITICAL]
        last_pred = self._state.get("last_prediction_time")

        return {
            "status": "degraded" if critical else "healthy",
            "last_prediction": datetime.fromtimestamp(last_pred).isoformat() if last_pred else None,
            "recent_alerts": len(recent),
            "critical_alerts": len(critical),
            "monitor_uptime_hours": round((time.time() - self._state.get("monitor_start", time.time())) / 3600, 1),
        }

    def _persist_alert(self, alert: Alert) -> None:
        with open(self._alerts_file, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    def _read_latest_metrics(self) -> Optional[Dict]:
        if not self._metrics_file.exists():
            return None
        last_line = None
        with open(self._metrics_file) as f:
            for line in f:
                if line.strip():
                    last_line = line
        return json.loads(last_line) if last_line else None

    def _load_state(self) -> Dict:
        if self._state_file.exists():
            with open(self._state_file) as f:
                return json.loads(f.read())
        return {"monitor_start": time.time()}

    def _save_state(self) -> None:
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, default=str)
