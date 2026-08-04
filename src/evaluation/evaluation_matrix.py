"""
Consolidated Evaluation Matrix Generator.

Per Agent Directive V7 Section 13: every serious candidate system must be
reported through a common evaluation matrix so results are comparable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "data" / "models" / "evaluation_matrix.json"


def generate_evaluation_matrix(
    training_metrics: Dict[str, Dict[str, Any]],
    positions: list,
    train_seasons: list,
    test_season: int,
    feature_version: str = "",
    experiment_id: str = "",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate a consolidated evaluation matrix per Directive V7 Section 13.

    Produces a per-position table with:
    - Predictive accuracy: RMSE, MAE, R²
    - Calibration: ECE (if available)
    - Decision utility: available metrics
    - Risk: worst-position performance
    - Stability: metric variance indicators

    Args:
        training_metrics: Per-position metrics dict from trainer.training_metrics.
        positions: List of positions trained.
        train_seasons: Seasons used for training.
        test_season: Holdout test season.
        feature_version: Feature set version string.
        experiment_id: Experiment tracker run ID.
        output_path: Where to write the JSON artifact.

    Returns:
        The evaluation matrix dict (also written to disk).
    """
    output_path = output_path or _DEFAULT_OUTPUT

    matrix = {}
    all_rmses = []
    all_maes = []

    for pos in positions:
        pos_metrics = training_metrics.get(pos, {})
        rmse = pos_metrics.get("rmse")
        mae = pos_metrics.get("mae")
        r2 = pos_metrics.get("r2")
        mape = pos_metrics.get("mape")

        if rmse is not None:
            all_rmses.append(rmse)
        if mae is not None:
            all_maes.append(mae)

        matrix[pos] = {
            "predictive_accuracy": {
                "rmse": _round(rmse),
                "mae": _round(mae),
                "r2": _round(r2),
                "mape_pct": _round(mape),
            },
            "calibration": {
                "ece": _round(pos_metrics.get("ece")),
            },
            "decision_utility": {
                "spearman_rank": _round(pos_metrics.get("spearman")),
                "directional_accuracy": _round(pos_metrics.get("directional_accuracy")),
            },
            "risk": {
                "prediction_std": _round(pos_metrics.get("prediction_std")),
            },
            "stability": {
                "cv_rmse_std": _round(pos_metrics.get("cv_rmse_std")),
            },
        }

    # Cross-position summary
    summary = {
        "mean_rmse": _round(np.mean(all_rmses)) if all_rmses else None,
        "worst_rmse": _round(max(all_rmses)) if all_rmses else None,
        "best_rmse": _round(min(all_rmses)) if all_rmses else None,
        "mean_mae": _round(np.mean(all_maes)) if all_maes else None,
        "positions_evaluated": len(matrix),
    }

    result = {
        "generated_at": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "feature_version": feature_version,
        "train_seasons": train_seasons,
        "test_season": test_season,
        "per_position": matrix,
        "cross_position_summary": summary,
        "directive_section": "Section 13 — Required Evaluation Matrix",
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Evaluation matrix written to %s", output_path)
    except Exception as e:
        logger.warning("Failed to write evaluation matrix: %s", e)

    return result


def _round(value, decimals=4):
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None
