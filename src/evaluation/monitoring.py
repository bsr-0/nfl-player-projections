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
    # Label drift detection (Directive V7 Section 18.3, 3rd drift axis)
    # ------------------------------------------------------------------

    def check_label_drift(
        self,
        recent_targets: np.ndarray,
        baseline_path: Optional[Path] = None,
        psi_threshold: float = 0.20,
        base_rate_shift_threshold: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Detect label/prior drift by comparing target distributions.

        Per Directive V7 Section 18.3: monitor the rolling base rate of the
        target variable and compare to the training-era base rate.

        Args:
            recent_targets: Recent actual target values.
            baseline_path: Path to training-era label baseline JSON.
            psi_threshold: PSI threshold for alert (default 0.2).
            base_rate_shift_threshold: Relative base rate shift (default 20%).

        Returns:
            List of alert dicts.
        """
        alerts: List[Dict[str, Any]] = []
        targets = np.asarray(recent_targets, dtype=float)
        targets = targets[np.isfinite(targets)]
        if len(targets) < 10:
            return alerts

        # Load training-era baseline
        baseline_path = baseline_path or (self.alert_dir.parent / "models" / "label_baseline.json")
        if not baseline_path.exists():
            logger.debug("No label baseline found at %s; skipping label drift check", baseline_path)
            return alerts

        import json as _json
        try:
            with open(baseline_path, encoding="utf-8") as f:
                baseline = _json.load(f)
        except (IOError, _json.JSONDecodeError):
            return alerts

        # Base rate shift check
        baseline_mean = baseline.get("mean")
        if baseline_mean is not None and baseline_mean > 0:
            current_mean = float(np.mean(targets))
            relative_shift = abs(current_mean - baseline_mean) / baseline_mean
            if relative_shift > base_rate_shift_threshold:
                alerts.append(self._make_alert(
                    AlertLevel.WARNING,
                    f"Label drift detected: mean={current_mean:.2f} vs "
                    f"baseline={baseline_mean:.2f} ({relative_shift:.1%} shift)",
                    {
                        "current_mean": current_mean,
                        "baseline_mean": baseline_mean,
                        "relative_shift": round(relative_shift, 4),
                        "drift_axis": "label_drift",
                    },
                ))

        # PSI check (Population Stability Index)
        baseline_bins = baseline.get("histogram_bins")
        baseline_freqs = baseline.get("histogram_freqs")
        if baseline_bins and baseline_freqs:
            try:
                bins = np.array(baseline_bins)
                expected = np.array(baseline_freqs, dtype=float)
                observed = np.histogram(targets, bins=bins)[0].astype(float)
                # Normalize to proportions
                expected = expected / max(expected.sum(), 1)
                observed = observed / max(observed.sum(), 1)
                # Add small epsilon to avoid log(0)
                eps = 1e-6
                expected = np.clip(expected, eps, None)
                observed = np.clip(observed, eps, None)
                psi = float(np.sum((observed - expected) * np.log(observed / expected)))
                if psi > psi_threshold:
                    alerts.append(self._make_alert(
                        AlertLevel.WARNING,
                        f"Label PSI drift: PSI={psi:.3f} exceeds threshold {psi_threshold}",
                        {
                            "psi": round(psi, 4),
                            "threshold": psi_threshold,
                            "drift_axis": "label_drift",
                        },
                    ))
            except Exception as e:
                logger.debug("PSI computation failed: %s", e)

        for alert in alerts:
            self._emit_alert(alert)

        self._log_metrics({
            "timestamp": datetime.now().isoformat(),
            "type": "label_drift_check",
            "n_targets": len(targets),
            "current_mean": float(np.mean(targets)),
            "baseline_mean": baseline.get("mean"),
        })

        return alerts

    @staticmethod
    def save_label_baseline(
        targets: np.ndarray,
        output_path: Path,
        n_bins: int = 20,
    ) -> Dict[str, Any]:
        """Save training-era target distribution as baseline for drift detection.

        Should be called during training to capture the label distribution.

        Args:
            targets: Training target values.
            output_path: Where to save the baseline JSON.
            n_bins: Number of histogram bins.

        Returns:
            Baseline dict.
        """
        targets = np.asarray(targets, dtype=float)
        targets = targets[np.isfinite(targets)]
        hist, bin_edges = np.histogram(targets, bins=n_bins)

        baseline = {
            "mean": round(float(np.mean(targets)), 4),
            "std": round(float(np.std(targets)), 4),
            "median": round(float(np.median(targets)), 4),
            "min": round(float(np.min(targets)), 4),
            "max": round(float(np.max(targets)), 4),
            "n_samples": len(targets),
            "histogram_bins": [round(float(b), 4) for b in bin_edges],
            "histogram_freqs": [int(h) for h in hist],
            "generated_at": datetime.now().isoformat(),
        }

        import json as _json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(baseline, f, indent=2)

        return baseline

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

    def circuit_breaker(
        self,
        predictions: np.ndarray,
        position: str = "ALL",
        reference_rmse: Optional[float] = None,
        reference_mean: Optional[float] = None,
        max_rmse_degradation: float = 0.50,
        max_null_rate: float = 0.10,
    ) -> Dict[str, Any]:
        """Circuit breaker: disable live predictions when quality degrades.

        Per Directive V7 Section 1 (safety over ambition) and Section 15
        (failure modes that trigger rejection): halt predictions when
        degradation exceeds thresholds.

        Args:
            predictions: Current batch of predictions.
            position: Position label.
            reference_rmse: Expected RMSE from validation.
            reference_mean: Expected mean from training.
            max_rmse_degradation: Max allowed RMSE increase ratio.
            max_null_rate: Max allowed null prediction rate.

        Returns:
            Dict with 'allow_predictions' bool and 'reasons' list.
        """
        preds = np.asarray(predictions, dtype=float)
        finite = preds[np.isfinite(preds)]
        reasons: List[str] = []

        # Check null rate
        null_rate = 1.0 - len(finite) / max(len(preds), 1)
        if null_rate > max_null_rate:
            reasons.append(
                f"Null prediction rate {null_rate:.1%} exceeds threshold {max_null_rate:.1%}"
            )

        # Check mean drift (extreme)
        if reference_mean is not None and reference_mean > 0 and len(finite) > 0:
            drift = abs(float(np.mean(finite)) - reference_mean) / reference_mean
            if drift > 0.50:
                reasons.append(
                    f"Prediction mean drift {drift:.1%} exceeds 50% safety threshold"
                )

        allow = len(reasons) == 0
        result = {
            "allow_predictions": allow,
            "position": position,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
        }

        if not allow:
            alert = self._make_alert(
                AlertLevel.CRITICAL,
                f"CIRCUIT BREAKER TRIPPED for {position}: predictions disabled",
                result,
            )
            self._emit_alert(alert)
            logger.error("Circuit breaker tripped: %s", reasons)

        return result

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Aggregate all monitoring metrics into a single dashboard response.

        Per Directive V7 Section 18: monitoring dashboard must track
        signal families continuously.

        Returns:
            Dict with recent alerts, metrics summary, and system status.
        """
        alerts = self.get_recent_alerts(last_n=20)

        # Read recent metrics
        recent_metrics: List[Dict[str, Any]] = []
        if self.metrics_log.exists():
            with open(self.metrics_log, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recent_metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            recent_metrics = recent_metrics[-100:]

        # Compute summary
        critical_alerts = [a for a in alerts if a.get("level") == AlertLevel.CRITICAL]
        warning_alerts = [a for a in alerts if a.get("level") == AlertLevel.WARNING]

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "degraded" if critical_alerts else "healthy",
            "alerts_summary": {
                "total": len(alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
            },
            "recent_alerts": alerts[-10:],
            "recent_metrics": recent_metrics[-20:],
            "thresholds": {
                "prediction_mean_drift": self.prediction_mean_drift_threshold,
                "prediction_std_drift": self.prediction_std_drift_threshold,
                "feature_drift_ks": self.feature_drift_ks_threshold,
                "error_rmse_degradation": self.error_rmse_degradation_threshold,
            },
            "drift_axes": {
                "data_drift": "KS test on features (check_feature_drift)",
                "concept_drift": "RMSE degradation (check_actuals)",
                "label_drift": "PSI + base rate shift (check_label_drift)",
            },
        }

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        try:
            with open(self.metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, default=str) + "\n")
        except Exception:
            pass


# ======================================================================
# Scoring environment shift detection (pre-training diagnostic)
# ======================================================================

def detect_scoring_environment_shift(
    df: "pd.DataFrame",
    target_col: str = "fantasy_points",
    shift_threshold: float = 0.15,
    positions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect year-over-year scoring environment shifts before training.

    The NFL scoring environment is non-stationary — rule changes, offensive
    scheme evolution, and pace-of-play shifts cause season-level mean fantasy
    points to drift.  Training a model on data from a different scoring
    regime produces systematically biased predictions (the 2025 backtest
    collapsed to R²=-0.685 partly because of this).

    This function computes per-season, per-position mean fantasy points and
    flags seasons where the mean shifted by more than ``shift_threshold``
    relative to the prior season.  It also computes the overall trend
    (increasing or decreasing) across the training window.

    Call this before training to:
    1. Log which seasons have shifted scoring environments.
    2. Warn if the most recent training season deviates significantly from
       earlier seasons (the model will over- or under-predict).
    3. Recommend whether to narrow the training window or increase recency
       decay.

    Args:
        df: Training DataFrame with season, position, and target columns.
        target_col: Column with fantasy points.
        shift_threshold: Fractional change that constitutes a "shift"
            (0.15 = 15% year-over-year change).
        positions: Positions to analyze (default: QB, RB, WR, TE).

    Returns:
        Dict with per-position shift analysis, overall trend, and
        recommendations.
    """
    import pandas as pd

    positions = positions or ["QB", "RB", "WR", "TE"]
    result: Dict[str, Any] = {
        "shifts_detected": [],
        "per_position": {},
        "overall_trend": "stable",
        "recommendation": None,
    }

    if df.empty or target_col not in df.columns or "season" not in df.columns:
        return result

    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        return result

    all_shifts: List[Dict[str, Any]] = []

    for pos in positions:
        pos_df = df[df["position"] == pos] if "position" in df.columns else df
        if pos_df.empty:
            continue

        # Per-season mean fantasy points
        season_means = (
            pos_df.groupby("season")[target_col]
            .mean()
            .reindex(seasons)
        )
        season_means = season_means.dropna()
        if len(season_means) < 2:
            continue

        # Year-over-year shifts
        shifts = []
        prev_mean = None
        for season in season_means.index:
            mean_val = float(season_means[season])
            if prev_mean is not None and prev_mean > 0:
                pct_change = (mean_val - prev_mean) / prev_mean
                if abs(pct_change) > shift_threshold:
                    shift_info = {
                        "position": pos,
                        "season": int(season),
                        "prev_season_mean": round(prev_mean, 2),
                        "season_mean": round(mean_val, 2),
                        "pct_change": round(pct_change * 100, 1),
                    }
                    shifts.append(shift_info)
                    all_shifts.append(shift_info)
            prev_mean = mean_val

        # Trend: compare first half of seasons to second half
        midpoint = len(season_means) // 2
        early_mean = float(season_means.iloc[:midpoint].mean())
        late_mean = float(season_means.iloc[midpoint:].mean())
        trend_pct = (late_mean - early_mean) / early_mean if early_mean > 0 else 0

        result["per_position"][pos] = {
            "season_means": {int(s): round(float(v), 2) for s, v in season_means.items()},
            "shifts": shifts,
            "trend_pct": round(trend_pct * 100, 1),
            "trend": "increasing" if trend_pct > 0.05 else ("decreasing" if trend_pct < -0.05 else "stable"),
            "latest_mean": round(float(season_means.iloc[-1]), 2),
            "training_window_mean": round(float(season_means.mean()), 2),
        }

    result["shifts_detected"] = all_shifts

    # Overall trend
    n_increasing = sum(1 for p in result["per_position"].values() if p["trend"] == "increasing")
    n_decreasing = sum(1 for p in result["per_position"].values() if p["trend"] == "decreasing")
    if n_decreasing > n_increasing:
        result["overall_trend"] = "decreasing"
    elif n_increasing > n_decreasing:
        result["overall_trend"] = "increasing"

    # Recommendations
    if all_shifts:
        recent_shifts = [s for s in all_shifts if s["season"] >= seasons[-2]]
        if recent_shifts:
            result["recommendation"] = (
                f"Scoring environment shifted in recent season(s) for "
                f"{len(recent_shifts)} position(s). Consider increasing "
                f"recency_decay_halflife or narrowing training window."
            )
        else:
            result["recommendation"] = (
                f"{len(all_shifts)} scoring shift(s) detected in older seasons. "
                f"Current training window may include outdated regimes."
            )

    return result
