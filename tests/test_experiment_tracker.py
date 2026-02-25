"""Tests for the experiment tracking system."""
import json
import tempfile
from pathlib import Path

import pytest

from src.utils.experiment_tracker import ExperimentTracker, ExperimentRun


class TestExperimentRun:
    def test_log_params(self):
        run = ExperimentRun("test-id", "test-run", Path("/tmp"))
        run.log_params({"alpha": 1.0, "n_estimators": 100})
        assert run.params["alpha"] == 1.0
        assert run.params["n_estimators"] == 100

    def test_log_metrics(self):
        run = ExperimentRun("test-id", "test-run", Path("/tmp"))
        run.log_metrics({"rmse": 2.5, "r2": 0.87})
        assert run.metrics["rmse"] == 2.5

    def test_log_fold_metrics(self):
        run = ExperimentRun("test-id", "test-run", Path("/tmp"))
        run.log_fold_metrics(0, {"rmse": 2.3})
        run.log_fold_metrics(1, {"rmse": 2.7})
        assert len(run.fold_metrics) == 2
        assert run.fold_metrics[0]["fold"] == 0

    def test_log_features(self):
        run = ExperimentRun("test-id", "test-run", Path("/tmp"))
        run.log_features(["feat_a", "feat_b", "feat_c"])
        assert len(run.features) == 3

    def test_to_dict_has_required_keys(self):
        run = ExperimentRun("test-id", "test-run", Path("/tmp"))
        run.log_params({"alpha": 1.0})
        run.log_metrics({"rmse": 2.5})
        d = run.to_dict()
        assert "run_id" in d
        assert "name" in d
        assert "params" in d
        assert "metrics" in d
        assert "start_time" in d
        assert "duration_seconds" in d


class TestExperimentTracker:
    def test_start_run_and_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=Path(tmpdir))
            with tracker.start_run(name="test_run") as run:
                run.log_params({"lr": 0.01})
                run.log_metrics({"rmse": 3.0})

            runs = tracker.list_runs()
            assert len(runs) == 1
            assert runs[0]["name"] == "test_run"
            assert runs[0]["metrics"]["rmse"] == 3.0

    def test_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=Path(tmpdir))
            for i in range(5):
                with tracker.start_run(name=f"run_{i}") as run:
                    run.log_metrics({"rmse": 5.0 - i})

            runs = tracker.list_runs()
            assert len(runs) == 5

    def test_best_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=Path(tmpdir))
            for rmse in [3.0, 1.5, 4.0, 2.0]:
                with tracker.start_run(name="trial") as run:
                    run.log_metrics({"rmse": rmse})

            best = tracker.best_run(metric="rmse", minimize=True)
            assert best is not None
            assert best["metrics"]["rmse"] == 1.5

    def test_compare_last_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=Path(tmpdir))
            with tracker.start_run(name="run1") as run:
                run.log_metrics({"rmse": 2.5})

            report = tracker.compare_last_n(n=5, metric="rmse")
            assert isinstance(report, str)
            assert "run1" in report

    def test_jsonl_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(log_dir=Path(tmpdir))
            with tracker.start_run(name="fmt_test") as run:
                run.log_params({"x": 1})

            log_file = Path(tmpdir) / "runs.jsonl"
            assert log_file.exists()
            with open(log_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "fmt_test"
