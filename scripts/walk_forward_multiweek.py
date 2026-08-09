#!/usr/bin/env python3
"""Real walk-forward validation for in-season rest-of-season projections.

TRACKING.md §2d evaluated MultiWeekModel on a single train<=2022/test
2023-2025 pooled split -- the same one-shot-split pattern §6's "Known
pitfalls" warns is unstable on small/mismatched test sets.

This reruns it as a real expanding-window walk-forward: for each test
season, train fresh on every season strictly before it (via the same
target-creation/feature-selection helpers as
src/models/train_position_models.py::train_position_model), score
against that season alone, repeat.

NOTE: this deliberately does NOT call train_position_model() directly --
that function's `multi_model.save()` / `PositionModel.save()` calls
write to the *production* artifact paths (data/models/multiweek_<pos>.joblib,
data/models/model_<pos>_<n>w.joblib) with no fold-specific naming, which
would silently clobber the real deployed models on every fold. This
script reuses its pure helpers (create_targets, get_position_features)
and calls MultiWeekModel.fit()/.predict() directly instead, never saving.

Usage:
    python scripts/walk_forward_multiweek.py
    python scripts/walk_forward_multiweek.py --positions QB RB --test-seasons 2023 2024 2025
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import DATA_DIR
from src.utils.database import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.utils.leakage import filter_feature_columns, assert_no_leakage_columns
from src.models.position_models import MultiWeekModel
from src.models.train_position_models import create_targets, get_position_features

POSITIONS = ["QB", "RB", "WR", "TE"]
N_WEEKS_LIST = [1, 4, 18]


def _load_all(position: str, min_games: int = 4) -> pd.DataFrame:
    db = DatabaseManager()
    all_data = db.get_all_players_for_training(position=position, min_games=min_games)
    if all_data.empty:
        return all_data
    engineer = FeatureEngineer()
    return engineer.create_features(all_data)


def _run_fold(position: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    train_data = create_targets(train_data, N_WEEKS_LIST)
    test_data = create_targets(test_data, N_WEEKS_LIST)

    exclude_cols = [
        "player_id", "name", "position", "team", "season", "week",
        "fantasy_points", "opponent", "home_away",
    ] + [f"target_{n}w" for n in N_WEEKS_LIST]

    available_features = [
        c for c in train_data.columns
        if c not in exclude_cols and train_data[c].dtype in ["int64", "float64"]
    ]
    feature_cols = get_position_features(position, available_features)
    if len(feature_cols) < 5:
        feature_cols = available_features
    feature_cols = filter_feature_columns(feature_cols)
    assert_no_leakage_columns(feature_cols, context=f"walk_forward_multiweek ({position})")

    X_train = train_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    y_dict = {n: train_data[f"target_{n}w"] for n in N_WEEKS_LIST}
    valid_mask = ~y_dict[1].isna()
    X_train = X_train[valid_mask]
    y_dict = {k: v[valid_mask] for k, v in y_dict.items()}

    if len(X_train) < 20:
        return {"error": f"insufficient training rows ({len(X_train)})"}

    multi_model = MultiWeekModel(position)
    multi_model.fit(X_train, y_dict, tune_hyperparameters=False)

    metrics = {}
    for n_weeks in N_WEEKS_LIST:
        target_col = f"target_{n_weeks}w"
        if target_col not in test_data.columns:
            continue
        valid_test = ~test_data[target_col].isna()
        X_test_valid = X_test[valid_test]
        y_test = test_data.loc[valid_test, target_col]
        if len(y_test) == 0:
            continue
        preds = multi_model.predict(X_test_valid, n_weeks)
        metrics[f"{n_weeks}w"] = {
            "n": int(len(y_test)),
            "r2": round(float(r2_score(y_test, preds)), 3) if len(y_test) > 1 else None,
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 1),
            "mae": round(float(mean_absolute_error(y_test, preds)), 1),
        }

    return {
        "n_train": int(len(X_train)),
        "n_features": len(feature_cols),
        "metrics": metrics,
    }


def _aggregate(fold_metrics: list, horizon: str) -> dict:
    vals = [f["metrics"][horizon] for f in fold_metrics if horizon in f.get("metrics", {})]
    if not vals:
        return {}
    r2s = [v["r2"] for v in vals if v["r2"] is not None]
    return {
        "n_folds": len(vals),
        "total_n": sum(v["n"] for v in vals),
        "r2_mean": round(float(np.mean(r2s)), 3) if r2s else None,
        "r2_std": round(float(np.std(r2s)), 3) if r2s else None,
        "mae_mean": round(float(np.mean([v["mae"] for v in vals])), 1),
        "rmse_mean": round(float(np.mean([v["rmse"] for v in vals])), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", nargs="+", default=POSITIONS)
    parser.add_argument("--test-seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--min-train-seasons", type=int, default=3)
    args = parser.parse_args()

    all_fold_records = {}
    for position in args.positions:
        print(f"\n{'='*60}\n{position}\n{'='*60}")
        data = _load_all(position)
        if data.empty:
            print(f"  No data for {position}, skipping")
            all_fold_records[position] = []
            continue
        available_seasons = sorted(int(s) for s in data["season"].dropna().unique())

        fold_records = []
        for test_season in args.test_seasons:
            train_seasons = [s for s in available_seasons if s < test_season]
            if len(train_seasons) < args.min_train_seasons or test_season not in available_seasons:
                print(f"  Skipping test_season={test_season}: insufficient history")
                continue
            train_data = data[data["season"].isin(train_seasons)]
            test_data = data[data["season"] == test_season]
            result = _run_fold(position, train_data, test_data)
            if "error" in result:
                print(f"  test_season={test_season}: {result['error']}")
                continue
            result["test_season"] = test_season
            fold_records.append(result)
            print(f"  test_season={test_season}: n_train={result['n_train']} {result['metrics']}")
        all_fold_records[position] = fold_records

    print(f"\n{'='*70}\nAGGREGATE (mean across folds, unweighted)\n{'='*70}")
    summary = {}
    for position, fold_records in all_fold_records.items():
        summary[position] = {}
        for horizon in ["1w", "4w", "18w"]:
            agg = _aggregate(fold_records, horizon)
            summary[position][horizon] = agg
            if agg:
                print(
                    f"  {position} {horizon}: n_folds={agg['n_folds']} "
                    f"R2={agg['r2_mean']}±{agg['r2_std']} MAE={agg['mae_mean']} "
                    f"RMSE={agg['rmse_mean']} (total_n={agg['total_n']})"
                )

    out = {
        "generated_at": datetime.now().isoformat(),
        "test_seasons": args.test_seasons,
        "min_train_seasons": args.min_train_seasons,
        "fold_records": all_fold_records,
        "summary": summary,
    }
    out_dir = DATA_DIR / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"walk_forward_multiweek_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
