"""Tests for production retrain/monitor script settings alignment."""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.production_retrain_and_monitor as monitor


def test_drift_threshold_comes_from_settings():
    expected = float(monitor.RETRAINING_CONFIG.get("degradation_threshold_pct", 20.0))
    assert monitor.DRIFT_THRESHOLD_PCT == expected


def test_is_retrain_day_respects_config(monkeypatch):
    monkeypatch.setitem(monitor.RETRAINING_CONFIG, "retrain_day", "Tuesday")

    class _FakeDatetime:
        @staticmethod
        def now():
            class _Now:
                @staticmethod
                def strftime(fmt):
                    return "Tuesday"
            return _Now()

    monkeypatch.setattr(monitor, "datetime", _FakeDatetime)
    assert monitor._is_retrain_day() is True


def test_check_drift_detects_degradation(tmp_path, monkeypatch):
    backtest_dir = tmp_path / "backtest_results"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    old = backtest_dir / "old.json"
    new = backtest_dir / "new.json"
    old.write_text('{"metrics": {"rmse": 8.0}}', encoding="utf-8")
    new.write_text('{"metrics": {"rmse": 10.0}}', encoding="utf-8")
    monkeypatch.setattr(monitor, "BACKTEST_RESULTS_DIR", backtest_dir)
    monkeypatch.setattr(monitor, "DRIFT_THRESHOLD_PCT", 20.0)
    out = monitor.check_drift()
    assert out["drift_detected"] is True
    assert out["pct_change"] == 25.0


def test_backup_and_rollback_models_include_metadata_files(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "model_qb_1w.joblib").write_text("model-v1", encoding="utf-8")
    (models_dir / "model_metadata.json").write_text('{"version": 1}', encoding="utf-8")
    (models_dir / "feature_version.txt").write_text("v1", encoding="utf-8")
    monkeypatch.setattr(monitor, "MODELS_DIR", models_dir)

    monitor.backup_models()
    assert (models_dir / "model_qb_1w.joblib.bak").exists()
    assert (models_dir / "model_metadata.json.bak").exists()
    assert (models_dir / "feature_version.txt.bak").exists()

    (models_dir / "model_qb_1w.joblib").write_text("model-v2", encoding="utf-8")
    restored = monitor.rollback_models()
    assert restored is True
    assert (models_dir / "model_qb_1w.joblib").read_text(encoding="utf-8") == "model-v1"
