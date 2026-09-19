#!/usr/bin/env python3
"""Train + walk-forward-evaluate the phase-2 margin (spread) / total (O-U) models.

Standalone from `src/models/train.py`'s QB/RB/WR/TE loop and from
scripts/train_game_outcome_model.py's win/loss classifier -- see
src/models/game_outcome/ and config.settings.GAME_OUTCOME_MODEL_CONFIG.

Usage:
    python scripts/train_game_margin_model.py                    # both targets, full history
    python scripts/train_game_margin_model.py --target margin
    python scripts/train_game_margin_model.py --seasons 2010 2025
    python scripts/train_game_margin_model.py --tune              # minimal fixed-grid search, see tuning.py

Writes, per target per model arm, a joblib artifact + JSON metadata sidecar
to data/models/ (MODELS_DIR), and prints the walk-forward metrics table.
Non-goal for phase 2: this does NOT wire into any serving/site path -- it
is a training + backtest artifact only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import mean_absolute_error

from config.settings import GAME_OUTCOME_MODEL_CONFIG, MODELS_DIR
from src.evaluation.game_margin_backtester import TARGET_CONFIG, TUNABLE_ARM_CLASSES, run_walk_forward_backtest
from src.models.game_outcome.features import build_margin_total_rows, feature_columns
from src.models.game_outcome.models import (
    GameMarginRidgeModel,
    GameRegressionRFModel,
    GameRegressionXGBModel,
    save_model,
)
from src.models.game_outcome.optuna_tuning import REGRESSOR_SEARCH_SPACES, tune_all_arms_optuna
from src.models.game_outcome.tuning import REGRESSOR_GRIDS, tune_all_arms


def _print_report(report: dict) -> None:
    pick_name = TARGET_CONFIG[report["target"]]["pick_name"]
    print(f"\n[{report['target']}] {report['n_games_total']:,} total games, {report['n_folds']} walk-forward fold(s)\n")
    for fold in report["folds"]:
        print(
            f"Fold {fold['fold']}: train seasons {fold['train_seasons'][0]}-{fold['train_seasons'][-1]} "
            f"({fold['n_train']} games) -> test season(s) {fold['test_seasons']} ({fold['n_test']} games)"
        )
        for arm, m in fold["arms"].items():
            pick = m[pick_name]
            pick_str = f"{pick:.3f}" if pick == pick else "n/a"  # NaN check
            adj_r2_str = f"{m['adj_r2']:.3f}" if m["adj_r2"] == m["adj_r2"] else "n/a"
            print(
                f"    {arm:14s} mae={m['mae']:.3f}  rmse={m['rmse']:.3f}  r2={m['r2']:.3f}  "
                f"adj_r2={adj_r2_str}  {pick_name}={pick_str} (n_picks={m['n_picks']})"
            )
    print("\nPooled across all folds:")
    for arm, m in report["pooled"].items():
        pick = m[pick_name]
        pick_str = f"{pick:.3f}" if pick == pick else "n/a"
        adj_r2_str = f"{m['adj_r2']:.3f}" if m["adj_r2"] == m["adj_r2"] else "n/a"
        print(
            f"    {arm:14s} n={m['n']:5d}  mae={m['mae']:.3f}  rmse={m['rmse']:.3f}  "
            f"r2={m['r2']:.3f}  adj_r2={adj_r2_str}  {pick_name}={pick_str}"
        )
    market_mae = report["pooled"].get("market_line", {}).get("mae")
    for arm in ("ridge", "xgboost", "random_forest"):
        mae = report["pooled"].get(arm, {}).get("mae")
        if mae is not None and market_mae is not None:
            status = "beats" if mae < market_mae else "does NOT beat"
            print(f"    {arm} {status} the market line's own MAE ({mae:.3f} vs {market_mae:.3f})")


def _tune(target: str, seasons, n_test_seasons: int) -> dict:
    """Tune once on data strictly before the earliest walk-forward test season
    (same rationale as scripts/train_game_outcome_model.py's `_tune`)."""
    label_col = TARGET_CONFIG[target]["label_col"]
    df = build_margin_total_rows(seasons=seasons)
    unique_seasons = sorted(df["season"].unique())
    n_splits = n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    holdout_start = unique_seasons[-n_splits]
    pre_holdout = df[df["season"] < holdout_start]
    print(
        f"Tuning {target} on {len(pre_holdout):,} games from seasons "
        f"{unique_seasons[0]}-{holdout_start - 1} (holdout starts {holdout_start})..."
    )

    feat_cols = feature_columns(pre_holdout)
    X = pre_holdout[feat_cols]
    y = pre_holdout[label_col].to_numpy()
    season_arr = pre_holdout["season"].to_numpy()

    def scorer(model, X_test, y_test):
        return mean_absolute_error(y_test, model.predict(X_test))

    return tune_all_arms(REGRESSOR_GRIDS, TUNABLE_ARM_CLASSES, X, y, season_arr, scorer)


def _tune_optuna(target: str, seasons, n_test_seasons: int, n_trials: int) -> dict:
    """Same pre-holdout-only boundary as `_tune` above, but a real Optuna
    TPE search over continuous ranges instead of a handful of fixed grid
    combos -- see optuna_tuning.py."""
    label_col = TARGET_CONFIG[target]["label_col"]
    df = build_margin_total_rows(seasons=seasons)
    unique_seasons = sorted(df["season"].unique())
    n_splits = n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    holdout_start = unique_seasons[-n_splits]
    pre_holdout = df[df["season"] < holdout_start]
    print(
        f"Optuna-tuning {target} ({n_trials} trials/arm) on {len(pre_holdout):,} games from seasons "
        f"{unique_seasons[0]}-{holdout_start - 1} (holdout starts {holdout_start})..."
    )

    feat_cols = feature_columns(pre_holdout)
    X = pre_holdout[feat_cols]
    y = pre_holdout[label_col].to_numpy()
    season_arr = pre_holdout["season"].to_numpy()

    def scorer(model, X_test, y_test):
        return mean_absolute_error(y_test, model.predict(X_test))

    return tune_all_arms_optuna(REGRESSOR_SEARCH_SPACES, TUNABLE_ARM_CLASSES, X, y, season_arr, scorer, n_trials=n_trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGET_CONFIG), default=None, help="Default: run both.")
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"), default=None)
    parser.add_argument("--n-test-seasons", type=int, default=None)
    parser.add_argument("--tune", action="store_true", help="Minimal fixed-grid hyperparameter search (see tuning.py).")
    parser.add_argument(
        "--tune-optuna", type=int, default=0, metavar="N_TRIALS",
        help="Real Optuna TPE search with N_TRIALS trials per arm (see optuna_tuning.py). Overrides --tune if both given.",
    )
    parser.add_argument("--no-save", action="store_true", help="Run the backtest only; skip persisting final models.")
    args = parser.parse_args()

    seasons = list(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else None
    n_test_seasons = args.n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    targets = [args.target] if args.target else list(TARGET_CONFIG)

    for target in targets:
        tuned_params = {}
        if args.tune_optuna:
            tuned_params = _tune_optuna(target, seasons, n_test_seasons, args.tune_optuna)
        elif args.tune:
            tuned_params = _tune(target, seasons, n_test_seasons)

        print(f"\nRunning walk-forward backtest for target={target!r}...")
        report = run_walk_forward_backtest(
            target, seasons=seasons, n_test_seasons=args.n_test_seasons, tuned_params=tuned_params
        )
        _print_report(report)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "seasons_requested": seasons,
            "tuned_params": tuned_params,
            "config": GAME_OUTCOME_MODEL_CONFIG,
            "backtest": report,
        }

        if args.no_save:
            continue

        print(f"\nFitting final {target} models on the full available history...")
        df = build_margin_total_rows(seasons=seasons)
        feat_cols = feature_columns(df)
        label_col = TARGET_CONFIG[target]["label_col"]
        X, y = df[feat_cols], df[label_col].to_numpy()

        ridge = GameMarginRidgeModel(**tuned_params.get("ridge", {})).fit(X, y)
        xgb_model = GameRegressionXGBModel(**tuned_params.get("xgboost", {})).fit(X, y)
        rf_model = GameRegressionRFModel(**tuned_params.get("random_forest", {})).fit(X, y)

        save_model(ridge, MODELS_DIR / f"game_{target}_ridge.joblib")
        save_model(xgb_model, MODELS_DIR / f"game_{target}_xgb.joblib")
        save_model(rf_model, MODELS_DIR / f"game_{target}_rf.joblib")
        metadata["feature_columns"] = feat_cols
        metadata["n_training_games"] = int(len(df))

        metadata_path = MODELS_DIR / f"game_{target}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Saved {target} models + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
