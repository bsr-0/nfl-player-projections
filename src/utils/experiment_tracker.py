"""Lightweight experiment tracking for NFL prediction models.

Provides structured logging of training runs without requiring external
services (MLflow, W&B). Each run is appended to a JSONL file with:
- Unique run ID and timestamp
- Git commit hash
- Hyperparameters and configuration
- Evaluation metrics (per-fold and aggregate)
- Feature set used
- Model artifacts path

Usage:
    tracker = ExperimentTracker()
    with tracker.start_run(name="QB_weekly_retrain") as run:
        run.log_params({"n_estimators": 100, "max_depth": 5})
        run.log_metrics({"rmse": 2.5, "r2": 0.87})
        run.log_features(feature_list)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "experiments"


class ExperimentRun:
    """A single experiment run."""

    def __init__(self, run_id: str, name: str, log_dir: Path):
        self.run_id = run_id
        self.name = name
        self.log_dir = log_dir
        self.start_time = time.time()
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.tags: Dict[str, str] = {}
        self.features: List[str] = []
        self.artifacts: List[str] = []
        self.fold_metrics: List[Dict[str, Any]] = []
        self._git_commit = _get_git_commit()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.params.update(params)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        self.metrics.update(metrics)

    def log_fold_metrics(self, fold_idx: int, metrics: Dict[str, float]) -> None:
        """Log per-fold CV metrics for stability analysis."""
        self.fold_metrics.append({"fold": fold_idx, **metrics})

    def log_features(self, features: List[str]) -> None:
        """Log the feature set used."""
        self.features = list(features)

    def log_artifact(self, path: str) -> None:
        """Log a model artifact path."""
        self.artifacts.append(str(path))

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        self.tags[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the run to a dictionary."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "git_commit": self._git_commit,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration_seconds": round(time.time() - self.start_time, 1),
            "params": self.params,
            "metrics": self.metrics,
            "fold_metrics": self.fold_metrics,
            "tags": self.tags,
            "n_features": len(self.features),
            "features": self.features[:20],  # First 20 for readability
            "artifacts": self.artifacts,
        }


class ExperimentTracker:
    """Append-only experiment tracker using JSONL files."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or _DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "runs.jsonl"

    @contextmanager
    def start_run(self, name: str = "unnamed"):
        """Context manager that creates, yields, and persists an experiment run."""
        run = ExperimentRun(
            run_id=str(uuid.uuid4())[:8],
            name=name,
            log_dir=self.log_dir,
        )
        try:
            yield run
        finally:
            self._persist(run)

    def _persist(self, run: ExperimentRun) -> None:
        """Append run record to the JSONL log."""
        record = run.to_dict()
        with open(self._log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info("Experiment run %s logged to %s", run.run_id, self._log_file)

    def list_runs(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Read the last N experiment runs."""
        if not self._log_file.exists():
            return []
        runs = []
        with open(self._log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
        return runs[-last_n:]

    def best_run(self, metric: str = "rmse", minimize: bool = True) -> Optional[Dict[str, Any]]:
        """Find the run with the best value for a given metric."""
        runs = self.list_runs(last_n=1000)
        valid = [r for r in runs if metric in r.get("metrics", {})]
        if not valid:
            return None
        return min(valid, key=lambda r: r["metrics"][metric]) if minimize else max(valid, key=lambda r: r["metrics"][metric])

    def compare_last_n(self, n: int = 5, metric: str = "rmse") -> str:
        """Format a comparison table of the last N runs."""
        runs = self.list_runs(last_n=n)
        if not runs:
            return "No experiment runs logged yet."
        lines = [
            f"{'Run ID':<10} {'Name':<25} {'Date':<20} {metric.upper():>8} {'Params':>30}",
            "-" * 95,
        ]
        for r in runs:
            val = r.get("metrics", {}).get(metric, "N/A")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            params_str = str(r.get("params", {}))[:28]
            lines.append(
                f"{r['run_id']:<10} {r['name']:<25} "
                f"{r.get('start_time', 'N/A'):<20} {val_str:>8} {params_str:>30}"
            )
        return "\n".join(lines)


def _get_git_commit() -> str:
    """Get current git short hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
