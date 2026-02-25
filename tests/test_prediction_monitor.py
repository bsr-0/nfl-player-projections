"""Tests for the production prediction monitor."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.prediction_monitor import (
    PredictionMonitor,
    Alert,
    AlertLevel,
)


class TestPredictionMonitor:
    def _make_monitor(self, tmpdir, **kwargs):
        return PredictionMonitor(monitor_dir=Path(tmpdir), **kwargs)

    def test_log_predictions_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir)
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 15.0, 20.0]),
                week=1, season=2025,
            )
            assert "pred_mean" in metrics
            assert "n_predictions" in metrics
            assert metrics["n_predictions"] == 3

    def test_log_predictions_with_actuals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir)
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 15.0, 20.0]),
                actuals=np.array([12.0, 14.0, 18.0]),
            )
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "bias" in metrics

    def test_log_predictions_with_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir)
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 15.0, 20.0, 25.0, 8.0, 12.0]),
                actuals=np.array([12.0, 14.0, 18.0, 22.0, 10.0, 11.0]),
                positions=np.array(["QB", "QB", "RB", "RB", "WR", "WR"]),
            )
            assert "by_position" in metrics

    def test_out_of_range_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir, prediction_range=(0.0, 40.0))
            metrics = mon.log_predictions(
                predictions=np.array([10.0, -5.0, 50.0]),
            )
            assert metrics["out_of_range_count"] == 2

    def test_rmse_alert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir, rmse_threshold=5.0)
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 20.0, 30.0]),
                actuals=np.array([1.0, 10.0, 20.0]),
            )
            alerts = mon.check_alerts(metrics)
            alert_sources = [a.source for a in alerts]
            assert "error_rate" in alert_sources

    def test_no_alert_when_within_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir, rmse_threshold=10.0)
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 15.0, 20.0]),
                actuals=np.array([11.0, 14.0, 19.0]),
            )
            alerts = mon.check_alerts(metrics)
            error_alerts = [a for a in alerts if a.source == "error_rate"]
            assert len(error_alerts) == 0

    def test_health_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir)
            mon.log_predictions(predictions=np.array([10.0, 15.0]))
            status = mon.health_status()
            assert "status" in status
            assert status["status"] in ("healthy", "degraded")

    def test_alert_callback(self):
        received = []
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(
                tmpdir, rmse_threshold=1.0,
                alert_callback=lambda a: received.append(a),
            )
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 20.0]),
                actuals=np.array([1.0, 5.0]),
            )
            mon.check_alerts(metrics)
            assert len(received) > 0
            assert isinstance(received[0], Alert)

    def test_alerts_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mon = self._make_monitor(tmpdir, rmse_threshold=1.0)
            metrics = mon.log_predictions(
                predictions=np.array([10.0, 20.0]),
                actuals=np.array([1.0, 5.0]),
            )
            mon.check_alerts(metrics)
            recent = mon.get_recent_alerts()
            assert len(recent) > 0
            assert "level" in recent[0]


class TestAlert:
    def test_to_dict(self):
        alert = Alert(
            level=AlertLevel.WARNING,
            source="test",
            message="Test alert",
            details={"key": "value"},
        )
        d = alert.to_dict()
        assert d["level"] == "warning"
        assert d["source"] == "test"
        assert "timestamp" in d
