"""
Lightweight experiment tracking for NFL prediction model training runs.

Logs each training run's configuration, metrics, and metadata to a JSON-lines
file, making it easy to compare experiments over time without external
dependencies (MLflow, W&B, etc.).

Usage in train.py:
    tracker = ExperimentTracker()
    run_id = tracker.start_run(config={...})
    tracker.log_metrics(run_id, {"rmse": 5.2, "r2": 0.35})
    tracker.end_run(run_id)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "experiments"


class ExperimentTracker:
    """Append-only experiment log backed by a JSONL file."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or _DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiment_log.jsonl"
        self._active_runs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> str:
        """Begin a new experiment run.  Returns a unique run ID."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        run = {
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "description": description,
            "config": config or {},
            "tags": tags or {},
            "metrics": {},
            "git_commit": self._git_short_hash(),
            "git_dirty": self._git_is_dirty(),
        }
        self._active_runs[run_id] = run
        logger.info("Experiment run started: %s", run_id)
        return run_id

    def log_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        """Add/update metrics for an active run."""
        if run_id not in self._active_runs:
            logger.warning("Run %s not found; metrics not logged.", run_id)
            return
        self._active_runs[run_id]["metrics"].update(
            {k: self._serialize(v) for k, v in metrics.items()}
        )

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log additional configuration parameters mid-run."""
        if run_id not in self._active_runs:
            return
        self._active_runs[run_id]["config"].update(
            {k: self._serialize(v) for k, v in params.items()}
        )

    def end_run(self, run_id: str, status: str = "completed") -> None:
        """Finalize a run and flush to disk."""
        if run_id not in self._active_runs:
            return
        run = self._active_runs.pop(run_id)
        run["status"] = status
        run["ended_at"] = datetime.now().isoformat()
        self._append_to_log(run)
        logger.info("Experiment run ended: %s (%s)", run_id, status)

    def list_runs(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Read the most recent *last_n* runs from the log file."""
        if not self.log_file.exists():
            return []
        runs: List[Dict[str, Any]] = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return runs[-last_n:]

    def compare_runs(self, metric_keys: Optional[List[str]] = None,
                     last_n: int = 10) -> str:
        """Return a formatted comparison table of recent runs."""
        runs = self.list_runs(last_n=last_n)
        if not runs:
            return "No experiment runs found."
        if metric_keys is None:
            # Auto-detect common metrics
            all_keys: set = set()
            for r in runs:
                all_keys.update(r.get("metrics", {}).keys())
            metric_keys = sorted(all_keys)[:8]

        lines = [
            f"{'Run ID':<26} {'Status':<10} " + "  ".join(f"{k:>12}" for k in metric_keys)
        ]
        lines.append("-" * len(lines[0]))
        for r in runs:
            vals = []
            for k in metric_keys:
                v = r.get("metrics", {}).get(k)
                if isinstance(v, float):
                    vals.append(f"{v:>12.4f}")
                elif v is not None:
                    vals.append(f"{str(v):>12}")
                else:
                    vals.append(f"{'—':>12}")
            lines.append(
                f"{r['run_id']:<26} {r.get('status', '?'):<10} " + "  ".join(vals)
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append_to_log(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write experiment log: %s", e)

    @staticmethod
    def _git_short_hash() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"

    @staticmethod
    def _git_is_dirty() -> bool:
        try:
            out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return bool(out)
        except Exception:
            return False

    @staticmethod
    def _serialize(value: Any) -> Any:
        """Make value JSON-safe."""
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
