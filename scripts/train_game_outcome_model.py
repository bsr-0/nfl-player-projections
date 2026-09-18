#!/usr/bin/env python3
"""Train + walk-forward-evaluate the phase-1 game-outcome model.

Standalone from `src/models/train.py`'s QB/RB/WR/TE loop -- see
src/models/game_outcome/ and config.settings.GAME_OUTCOME_MODEL_CONFIG.

Usage:
    python scripts/train_game_outcome_model.py                # full history, default fold count
    python scripts/train_game_outcome_model.py --seasons 2010 2025
    python scripts/train_game_outcome_model.py --n-test-seasons 3

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

from config.settings import GAME_OUTCOME_MODEL_CONFIG, MODELS_DIR
from src.evaluation.game_outcome_backtester import run_walk_forward_backtest
from src.models.game_outcome.features import build_game_outcome_rows, feature_columns
from src.models.game_outcome.models import GameOutcomeLogisticModel, GameOutcomeXGBModel, save_model


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
    for arm in ("logistic", "xgboost", "vegas_favorite"):
        acc = report["pooled"].get(arm, {}).get("accuracy")
        if acc is not None and home_field_acc is not None:
            status = "beats" if acc > home_field_acc else "DOES NOT beat"
            print(f"    {arm} {status} the home-field floor ({acc:.3f} vs {home_field_acc:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"), default=None)
    parser.add_argument("--n-test-seasons", type=int, default=None)
    parser.add_argument("--no-save", action="store_true", help="Run the backtest only; skip persisting final models.")
    args = parser.parse_args()

    seasons = list(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else None

    print("Running walk-forward backtest...")
    report = run_walk_forward_backtest(seasons=seasons, n_test_seasons=args.n_test_seasons)
    _print_report(report)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons_requested": seasons,
        "config": GAME_OUTCOME_MODEL_CONFIG,
        "backtest": report,
    }

    if args.no_save:
        return

    print("\nFitting final models on the full available history...")
    df = build_game_outcome_rows(seasons=seasons)
    feat_cols = feature_columns(df)
    X, y = df[feat_cols], df["home_win"].to_numpy()

    logistic = GameOutcomeLogisticModel().fit(X, y)
    xgb_model = GameOutcomeXGBModel().fit(X, y)

    save_model(logistic, MODELS_DIR / "game_outcome_logistic.joblib")
    save_model(xgb_model, MODELS_DIR / "game_outcome_xgb.joblib")
    metadata["feature_columns"] = feat_cols
    metadata["n_training_games"] = int(len(df))

    metadata_path = MODELS_DIR / "game_outcome_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved models + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
