"""Tests for rubric compliance checker script."""

from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as settings
from scripts.check_rubric_compliance import run_checks, summarize


def test_rubric_checker_core_checks_exist():
    results = run_checks(require_artifacts=False)
    payload = summarize(results)
    names = {c["name"] for c in payload["checks"]}
    assert "positions_all" in names
    assert "horizon_model_classes" in names
    assert payload["failed_errors"] == 0


def test_rubric_checker_artifacts_required_fail_when_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(settings, "MODELS_DIR", Path(tmp))
        results = run_checks(require_artifacts=True)
        payload = summarize(results)
        assert payload["failed_errors"] >= 1


def test_rubric_checker_artifacts_required_pass_when_present(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for fname in [
            "model_metadata.json",
            "model_monitoring_report.json",
            "top10_features_per_position.json",
        ]:
            (tmp_path / fname).write_text("{}", encoding="utf-8")

        monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
        results = run_checks(require_artifacts=True)
        payload = summarize(results)
        assert payload["failed_errors"] == 0
