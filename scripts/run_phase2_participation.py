#!/usr/bin/env python3
"""Run Phase 2 walk-forward participation/opportunity experiments.

This script reads ``canonical_player_weeks`` and writes only experiment
artifacts.  It does not update Phase 7, production features or PPR targets.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH
from src.models.participation_opportunity import (
    FEATURES, PRIMARY_THRESHOLD, TARGET_THRESHOLDS, build_causal_features,
    expanding_season_folds, feature_columns_for_ablation, make_classifier,
    make_opportunity_regressor, predict_asof_week, target_name, validate_causal_contract,
)


def calibration_rows(y: pd.Series, probability: np.ndarray, bins: int = 10) -> list[dict]:
    data = pd.DataFrame({"actual": y.astype(float).to_numpy(), "probability": probability})
    data["bin"] = pd.cut(data["probability"], np.linspace(0, 1, bins + 1),
                         include_lowest=True, duplicates="drop")
    return [
        {"bin": str(label), "n": int(len(g)), "mean_probability": float(g.probability.mean()),
         "observed_rate": float(g.actual.mean())}
        for label, g in data.groupby("bin", observed=True)
    ]


def classification_metrics(y: pd.Series, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def classification_slices(test: pd.DataFrame, target: str, probability: np.ndarray) -> list[dict]:
    """Return aggregate, position and cold-start diagnostics for one prediction."""
    scored = test[[target, "position", "cold_start"]].copy()
    scored["probability"] = probability
    masks = [("all", pd.Series(True, index=scored.index))]
    masks.extend((f"position:{p}", scored.position.eq(p)) for p in sorted(scored.position.unique()))
    masks.extend([
        ("cold_start", scored.cold_start.eq(1)),
        ("has_history", scored.cold_start.eq(0)),
    ])
    rows = []
    for segment, mask in masks:
        part = scored[mask]
        if part.empty:
            continue
        row = classification_metrics(part[target].astype(int), part.probability.to_numpy())
        row["segment"] = segment
        rows.append(row)
    return rows


def baseline_probability(train: pd.DataFrame, test: pd.DataFrame, target: str, mode: str) -> np.ndarray:
    global_rate = float(train[target].mean())
    position_rate = train.groupby("position")[target].mean().to_dict()
    fallback = test["position"].map(position_rate).fillna(global_rate).to_numpy(float)
    if mode == "position_rate":
        return fallback
    source = test["played_lag1"] if mode == "previous_game" else test["played_roll3"]
    return source.fillna(pd.Series(fallback, index=test.index)).to_numpy(float)


def evaluate(frame: pd.DataFrame, min_train_seasons: int = 3) -> tuple[pd.DataFrame, dict]:
    validate_causal_contract(frame)
    observed = frame[frame["label_observed"].eq(1)].copy()
    predictions: list[pd.DataFrame] = []
    report: dict = {"contract": {
        "unknown_rows_are_negatives": False,
        "features": FEATURES,
        "primary_threshold": PRIMARY_THRESHOLD,
        "production_ppr_modified": False,
    }, "coverage": {}, "classification": [], "opportunity": [], "ablations": []}

    coverage = frame.groupby(["season", "position"], dropna=False).agg(
        rows=("player_id", "size"), observed_labels=("label_observed", "sum")
    ).reset_index()
    coverage["coverage"] = coverage.observed_labels / coverage.rows
    report["coverage"] = coverage.to_dict("records")

    for fold in expanding_season_folds(observed, min_train_seasons):
        train, test = observed.loc[fold.train_index], observed.loc[fold.test_index]
        for threshold in TARGET_THRESHOLDS:
            target = target_name(threshold)
            y_train, y_test = train[target].astype(int), test[target].astype(int)
            if y_train.nunique() < 2 or y_test.empty:
                continue
            for baseline in ("position_rate", "previous_game", "rolling3"):
                p = baseline_probability(train, test, target, baseline)
                for row in classification_slices(test, target, p):
                    row.update({"season": fold.test_season, "threshold": threshold, "model": baseline})
                    report["classification"].append(row)

            for kind in ("logistic", "hist_gbm"):
                model = make_classifier(kind)
                model.fit(train[FEATURES], y_train)
                p = model.predict_proba(test[FEATURES])[:, 1]
                for row in classification_slices(test, target, p):
                    row.update({"season": fold.test_season, "threshold": threshold, "model": kind})
                    if row["segment"] == "all":
                        row["calibration"] = calibration_rows(y_test, p)
                    report["classification"].append(row)
                if threshold == PRIMARY_THRESHOLD:
                    pred = test[["player_id", "season", "week", "team", "position"]].copy()
                    pred["participation_probability"] = p
                    pred["participation_actual"] = y_test.to_numpy()
                    pred["model"] = kind
                    predictions.append(pred)

        # Conditional opportunity uses only primary-positive rows.  Evaluation
        # is therefore conditional too; the unconditional downstream product
        # can later be formed from P(play) * E(share | play).
        primary = target_name(PRIMARY_THRESHOLD)
        opp_train = train[train[primary].eq(1)]
        opp_test = test[test[primary].eq(1)]
        if len(opp_train) >= 100 and not opp_test.empty:
            baseline = opp_test["snap_share_roll3"].fillna(opp_train.snap_share.median()).clip(0, 1)
            report["opportunity"].append({
                "season": fold.test_season, "model": "rolling3", "segment": "all", "n": len(opp_test),
                "mae": float(mean_absolute_error(opp_test.snap_share, baseline)),
                "rmse": float(mean_squared_error(opp_test.snap_share, baseline) ** 0.5),
            })
            reg = make_opportunity_regressor().fit(opp_train[FEATURES], opp_train.snap_share)
            all_opp_p = np.clip(reg.predict(test[FEATURES]), 0, 1)
            opp_p = pd.Series(all_opp_p, index=test.index).loc[opp_test.index].to_numpy()
            for segment, subset in [("all", opp_test)] + [
                (f"position:{p}", opp_test[opp_test.position.eq(p)])
                for p in sorted(opp_test.position.unique())
            ]:
                if subset.empty:
                    continue
                segment_p = pd.Series(opp_p, index=opp_test.index).loc[subset.index]
                report["opportunity"].append({
                    "season": fold.test_season, "model": "hist_gbm_mae", "segment": segment,
                    "n": len(subset), "mae": float(mean_absolute_error(subset.snap_share, segment_p)),
                    "rmse": float(mean_squared_error(subset.snap_share, segment_p) ** 0.5),
                })
            for pred in predictions[-2:]:
                if pred["season"].iloc[0] == fold.test_season:
                    pred["conditional_snap_share"] = all_opp_p
                    pred["expected_snap_share"] = (
                        pred["participation_probability"] * pred["conditional_snap_share"]
                    )

        # Primary-target ablations use logistic regression to isolate feature
        # value without conflating each addition with model-family changes.
        y_train, y_test = train[primary].astype(int), test[primary].astype(int)
        if y_train.nunique() >= 2:
            for ablation in ("history", "usage", "role"):
                cols = feature_columns_for_ablation(ablation)
                base = make_classifier("logistic")
                # Rebuild preprocessing through column selection by blanking
                # excluded values; the declared model schema remains stable.
                train_x, test_x = train[FEATURES].copy(), test[FEATURES].copy()
                for col in set(FEATURES) - set(cols):
                    train_x[col] = np.nan if col in train_x.select_dtypes(exclude="object") else "EXCLUDED"
                    test_x[col] = np.nan if col in test_x.select_dtypes(exclude="object") else "EXCLUDED"
                base.fit(train_x, y_train)
                p = base.predict_proba(test_x)[:, 1]
                row = classification_metrics(y_test, p)
                row.update({"season": fold.test_season, "ablation": ablation})
                report["ablations"].append(row)

    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), report


def summarize(report: dict) -> dict:
    out = {}
    for section, keys in (("classification", ["threshold", "model", "segment"]),
                          ("opportunity", ["model", "segment"]), ("ablations", ["ablation"])):
        data = pd.DataFrame(report[section])
        if data.empty:
            out[section] = []
            continue
        metrics = [c for c in ("brier", "log_loss", "mae", "rmse") if c in data]
        out[section] = data.groupby(keys, dropna=False)[metrics].mean().reset_index().to_dict("records")
    return out


def add_kickoff_filtered_injury_score(panel: pd.DataFrame) -> pd.DataFrame:
    """Reuse the one repository-authoritative pregame injury-cache guard."""
    from src.features.feature_engineering import FeatureEngineer
    return FeatureEngineer()._merge_injury_data_from_cache(panel.copy())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--min-train-seasons", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "phase2")
    ap.add_argument("--predict-season", type=int, default=None,
                    help="also write Phase 2 predictions for a target season/week")
    ap.add_argument("--predict-week", type=int, default=1,
                    help="target week for --predict-season; Week 1 is the pre-season path")
    ap.add_argument("--include-pregame-injury", action="store_true",
                    help="include only kickoff-filtered player_injuries cache scores as an optional ablation")
    args = ap.parse_args()
    with sqlite3.connect(args.db) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='canonical_player_weeks'").fetchone()
        if not exists:
            raise SystemExit("canonical_player_weeks is missing; run Phase 1 builder with --write first")
        panel = pd.read_sql("SELECT * FROM canonical_player_weeks", conn)

    if args.include_pregame_injury:
        panel = add_kickoff_filtered_injury_score(panel)
    frame = build_causal_features(panel, include_pregame_injury=args.include_pregame_injury)
    predictions, report = evaluate(frame, args.min_train_seasons)
    report["summary"] = summarize(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.predict_season is not None:
        future = predict_asof_week(frame, args.predict_season, args.predict_week)
        future.to_csv(args.output_dir / f"predictions_{args.predict_season}_week_{args.predict_week}.csv", index=False)
        print(f"wrote {len(future):,} strictly as-of prediction rows for {args.predict_season} week {args.predict_week}")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {len(predictions):,} OOF prediction rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
