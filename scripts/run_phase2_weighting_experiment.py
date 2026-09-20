"""Leakage-safe Phase 2 weighting experiment.

This runner evaluates weighted hist_gbm participation classifiers and opportunity
regressors against the incumbent unweighted model. It never treats unknown rows
as negatives and never changes the production Phase 2 artifacts.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from src.models.participation_opportunity import (
    FEATURES, PRIMARY_THRESHOLD, build_causal_features, expanding_season_folds,
    make_classifier, make_opportunity_regressor, target_name, validate_causal_contract,
)
from scripts.run_phase2_participation import calibration_rows, classification_metrics

DB_PATH = ROOT / "data" / "nfl_data.db"


def weights_for(train: pd.DataFrame, y: pd.Series, *, positive: float = 1.0,
                cold_start: float = 1.0, half_life: float | None = None) -> np.ndarray:
    """Build weights from the current training fold only, then normalize/cap."""
    w = np.where(y.to_numpy(dtype=int) == 1, positive, 1.0 / positive)
    if cold_start != 1.0:
        w = w * np.where(train["cold_start"].to_numpy(dtype=int) == 1, cold_start, 1.0)
    if half_life is not None:
        max_season = float(train["season"].max())
        w = w * np.power(0.5, (max_season - train["season"].to_numpy(float)) / half_life)
    w = w / np.mean(w)
    return np.clip(w, 0.5, 2.0)


def regression_weights(train: pd.DataFrame, *, high: float = 1.0,
                       boundary: float = 1.0) -> np.ndarray:
    w = np.ones(len(train), dtype=float)
    w = np.where(train["snap_share"].to_numpy(float) >= 0.70, high, w)
    w = np.where(np.abs(train["snap_share"].to_numpy(float) - 0.10) <= 0.05,
                 w * boundary, w)
    return np.clip(w / np.mean(w), 0.5, 2.0)


def fit_classifier(kind: str, train: pd.DataFrame, target: str, cfg: dict):
    y = train[target].astype(int)
    model = make_classifier(kind)
    w = weights_for(train, y, positive=cfg.get("positive", 1.0),
                    cold_start=cfg.get("cold_start", 1.0),
                    half_life=cfg.get("half_life"))
    if cfg.get("weighted", False):
        model.fit(train[FEATURES], y, model__sample_weight=w)
    else:
        model.fit(train[FEATURES], y)
    return model


def evaluate(frame: pd.DataFrame, min_train_seasons: int) -> dict:
    validate_causal_contract(frame)
    observed = frame[frame["label_observed"].eq(1)].copy()
    target = target_name(PRIMARY_THRESHOLD)
    configs = [
        {"name": "unweighted", "weighted": False},
        {"name": "class_1.25", "weighted": True, "positive": 1.25},
        {"name": "class_1.5", "weighted": True, "positive": 1.5},
        {"name": "class_2.0", "weighted": True, "positive": 2.0},
        {"name": "class_1.5_cold_1.25", "weighted": True, "positive": 1.5, "cold_start": 1.25},
        {"name": "class_1.5_decay3", "weighted": True, "positive": 1.5, "half_life": 3},
    ]
    rows, opp_rows = [], []
    for fold in expanding_season_folds(observed, min_train_seasons):
        train, test = observed.loc[fold.train_index], observed.loc[fold.test_index]
        if train[target].nunique() < 2:
            continue
        for cfg in configs:
            model = fit_classifier("hist_gbm", train, target, cfg)
            p = np.clip(model.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
            metrics = classification_metrics(test[target].astype(int), p)
            cal = calibration_rows(test[target].astype(int), p)
            rows.append({"season": fold.test_season, "model": cfg["name"],
                         "segment": "all", "n": len(test),
                         "brier": float(metrics["brier"]),
                         "log_loss": float(metrics["log_loss"]),
                         "ece": float(metrics["ece"]),
                         "calibration": cal,
                         "cold_start_brier": float(brier_score_loss(test.loc[test.cold_start.eq(1), target], p[test.cold_start.to_numpy() == 1])) if test.cold_start.eq(1).any() and test.loc[test.cold_start.eq(1), target].nunique() > 0 else None})
            # Conditional opportunity is evaluated on the same positive population.
            pos_train, pos_test = train[train[target].eq(1)], test[test[target].eq(1)]
            if len(pos_train) >= 100 and not pos_test.empty:
                reg = make_opportunity_regressor()
                rw = regression_weights(pos_train, high=1.5 if cfg["name"] == "class_1.5" else 1.0,
                                         boundary=1.25 if cfg["name"] == "class_1.5" else 1.0)
                if cfg["name"] == "class_1.5":
                    reg.fit(pos_train[FEATURES], pos_train["snap_share"], model__sample_weight=rw)
                else:
                    reg.fit(pos_train[FEATURES], pos_train["snap_share"])
                pred = np.clip(reg.predict(pos_test[FEATURES]), 0, 1)
                opp_rows.append({"season": fold.test_season, "model": cfg["name"],
                                 "segment": "all", "n": len(pos_test),
                                 "mae": float(mean_absolute_error(pos_test["snap_share"], pred))})
    return {"classification": rows, "opportunity": opp_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--min-train-seasons", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/experiments/participation_system/phase2/weighting")
    args = ap.parse_args()
    with sqlite3.connect(args.db) as conn:
        panel = pd.read_sql("SELECT * FROM canonical_player_weeks", conn)
    frame = build_causal_features(panel)
    report = evaluate(frame, args.min_train_seasons)
    report["contract"] = {"unknown_rows_are_negatives": False, "production_ppr_modified": False,
                           "weight_cap": [0.5, 2.0], "fold_local_weights": True}
    report["summary"] = {
        "classification": pd.DataFrame(report["classification"]).groupby("model", as_index=False)[["brier", "log_loss", "ece"]].mean().to_dict("records"),
        "opportunity": pd.DataFrame(report["opportunity"]).groupby("model", as_index=False)[["mae"]].mean().to_dict("records") if report["opportunity"] else [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (args.output_dir / "preregistration_v1.json").write_text(json.dumps({
        "primary_classifier": "hist_gbm", "configs": ["unweighted", "class_1.25", "class_1.5", "class_2.0", "class_1.5_cold_1.25", "class_1.5_decay3"],
        "weights": "fold-local, normalized and clipped to [0.5, 2.0]", "unknown_rows_as_negatives": False,
        "incumbent_unchanged": True,
    }, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
