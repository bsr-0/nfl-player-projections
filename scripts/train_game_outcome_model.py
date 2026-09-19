#!/usr/bin/env python3
"""Train + walk-forward-evaluate the phase-1 game-outcome model.

Standalone from `src/models/train.py`'s QB/RB/WR/TE loop -- see
src/models/game_outcome/ and config.settings.GAME_OUTCOME_MODEL_CONFIG.

Usage:
    python scripts/train_game_outcome_model.py                # full history, default fold count
    python scripts/train_game_outcome_model.py --seasons 2010 2025
    python scripts/train_game_outcome_model.py --n-test-seasons 3
    python scripts/train_game_outcome_model.py --tune          # minimal fixed-grid search, see tuning.py

Writes, per model arm, a joblib artifact + JSON metadata sidecar to
data/models/ (MODELS_DIR), and prints the walk-forward metrics table.
Non-goal for phase 1: this does NOT wire into any serving/site path -- it
is a training + backtest artifact only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import log_loss

from config.settings import GAME_OUTCOME_MODEL_CONFIG, MODELS_DIR
from src.evaluation.game_outcome_backtester import TUNABLE_ARM_CLASSES, run_walk_forward_backtest
from src.models.game_outcome.features import build_game_outcome_rows, feature_columns
from src.models.game_outcome.models import (
    GameOutcomeLogisticModel,
    GameOutcomeRFModel,
    GameOutcomeXGBModel,
    save_model,
)
from src.models.game_outcome.optuna_tuning import CLASSIFIER_SEARCH_SPACES, tune_all_arms_optuna
from src.models.game_outcome.tuning import CLASSIFIER_GRIDS, tune_all_arms


def _print_report(report: dict) -> None:
    print(f"\n{report['n_games_total']:,} total games, {report['n_folds']} walk-forward fold(s)\n")
    for fold in report["folds"]:
        print(
            f"Fold {fold['fold']}: train seasons {fold['train_seasons'][0]}-{fold['train_seasons'][-1]} "
            f"({fold['n_train']} games) -> test season(s) {fold['test_seasons']} ({fold['n_test']} games)"
        )
        for arm, m in fold["arms"].items():
            print(
                f"    {arm:16s} acc={m['accuracy']:.3f}  log_loss={m['log_loss']:.3f}  "
                f"roc_auc={m['roc_auc']:.3f}  brier={m['brier']:.3f}"
            )
    print("\nPooled across all folds:")
    for arm, m in report["pooled"].items():
        print(
            f"    {arm:16s} n={m['n']:5d}  acc={m['accuracy']:.3f}  log_loss={m['log_loss']:.3f}  "
            f"roc_auc={m['roc_auc']:.3f}  brier={m['brier']:.3f}"
        )
    home_field_acc = report["pooled"].get("home_field", {}).get("accuracy")
    for arm in ("logistic", "xgboost", "random_forest", "vegas_favorite"):
        acc = report["pooled"].get(arm, {}).get("accuracy")
        if acc is not None and home_field_acc is not None:
            status = "beats" if acc > home_field_acc else "DOES NOT beat"
            print(f"    {arm} {status} the home-field floor ({acc:.3f} vs {home_field_acc:.3f})")


def _tune(seasons, n_test_seasons: int) -> dict:
    """Tune once on data strictly before the earliest walk-forward test season.

    That season boundary is, by construction, strictly before every season
    the outer walk-forward ever scores -- so nothing tuned here can ever see
    a season it (or a later fold) is evaluated on.
    """
    df = build_game_outcome_rows(seasons=seasons)
    unique_seasons = sorted(df["season"].unique())
    n_splits = n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    holdout_start = unique_seasons[-n_splits]
    pre_holdout = df[df["season"] < holdout_start]
    print(
        f"Tuning on {len(pre_holdout):,} games from seasons "
        f"{unique_seasons[0]}-{holdout_start - 1} (holdout starts {holdout_start})..."
    )

    feat_cols = feature_columns(pre_holdout)
    X = pre_holdout[feat_cols]
    y = pre_holdout["home_win"].to_numpy()
    season_arr = pre_holdout["season"].to_numpy()

    def scorer(model, X_test, y_test):
        proba = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, proba, labels=[0, 1])

    return tune_all_arms(CLASSIFIER_GRIDS, TUNABLE_ARM_CLASSES, X, y, season_arr, scorer)


def _tune_optuna(seasons, n_test_seasons: int, n_trials: int) -> dict:
    """Same pre-holdout-only data boundary as `_tune` above, but a real
    Optuna TPE search over continuous ranges instead of a handful of fixed
    grid combos -- see optuna_tuning.py."""
    df = build_game_outcome_rows(seasons=seasons)
    unique_seasons = sorted(df["season"].unique())
    n_splits = n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    holdout_start = unique_seasons[-n_splits]
    pre_holdout = df[df["season"] < holdout_start]
    print(
        f"Optuna-tuning ({n_trials} trials/arm) on {len(pre_holdout):,} games from seasons "
        f"{unique_seasons[0]}-{holdout_start - 1} (holdout starts {holdout_start})..."
    )

    feat_cols = feature_columns(pre_holdout)
    X = pre_holdout[feat_cols]
    y = pre_holdout["home_win"].to_numpy()
    season_arr = pre_holdout["season"].to_numpy()

    def scorer(model, X_test, y_test):
        proba = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, proba, labels=[0, 1])

    return tune_all_arms_optuna(CLASSIFIER_SEARCH_SPACES, TUNABLE_ARM_CLASSES, X, y, season_arr, scorer, n_trials=n_trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    tuned_params = {}
    if args.tune_optuna:
        tuned_params = _tune_optuna(seasons, n_test_seasons, args.tune_optuna)
    elif args.tune:
        tuned_params = _tune(seasons, n_test_seasons)

    print("\nRunning walk-forward backtest...")
    report = run_walk_forward_backtest(seasons=seasons, n_test_seasons=args.n_test_seasons, tuned_params=tuned_params)
    _print_report(report)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons_requested": seasons,
        "tuned_params": tuned_params,
        "config": GAME_OUTCOME_MODEL_CONFIG,
        "backtest": report,
    }

    if args.no_save:
        return

    print("\nFitting final models on the full available history...")
    df = build_game_outcome_rows(seasons=seasons)
    feat_cols = feature_columns(df)
    X, y = df[feat_cols], df["home_win"].to_numpy()

    logistic = GameOutcomeLogisticModel(**tuned_params.get("logistic", {})).fit(X, y)
    xgb_model = GameOutcomeXGBModel(**tuned_params.get("xgboost", {})).fit(X, y)
    rf_model = GameOutcomeRFModel(**tuned_params.get("random_forest", {})).fit(X, y)

    save_model(logistic, MODELS_DIR / "game_outcome_logistic.joblib")
    save_model(xgb_model, MODELS_DIR / "game_outcome_xgb.joblib")
    save_model(rf_model, MODELS_DIR / "game_outcome_rf.joblib")
    metadata["feature_columns"] = feat_cols
    metadata["n_training_games"] = int(len(df))

    metadata_path = MODELS_DIR / "game_outcome_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved models + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
