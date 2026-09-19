#!/usr/bin/env python3
"""One-off ablation: does `home_minus_away_rest_days` actually help the
game-outcome/margin models, or is it neutral noise?

Not part of the regular training pipeline -- run manually when curious
whether a specific feature is pulling weight. Reuses the real walk-forward
backtesters, just re-running each with and without the one column.

Usage:
    python scripts/ablate_rest_days.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score

from src.models.game_outcome.features import build_game_outcome_rows, build_margin_total_rows, feature_columns
from src.models.game_outcome.models import GameOutcomeLogisticModel, GameOutcomeXGBModel, GameMarginRidgeModel, GameRegressionXGBModel
from src.models.position_models import SeasonAwareTimeSeriesSplit

REST_COL = "home_minus_away_rest_days"
N_SPLITS = 5


def _classification_ablation(seasons=None) -> None:
    df = build_game_outcome_rows(seasons=seasons)
    feat_cols = feature_columns(df)
    y = df["home_win"].to_numpy()
    season_arr = df["season"].to_numpy()
    splitter_seasons = season_arr

    for with_rest in (True, False):
        cols = feat_cols if with_rest else [c for c in feat_cols if c != REST_COL]
        X = df[cols]
        splitter = SeasonAwareTimeSeriesSplit(n_splits=N_SPLITS, seasons=splitter_seasons, gap_seasons=0)
        for arm_name, arm_cls in (("logistic", GameOutcomeLogisticModel), ("xgboost", GameOutcomeXGBModel)):
            accs, losses, aucs = [], [], []
            for train_idx, test_idx in splitter.split(X):
                model = arm_cls().fit(X.iloc[train_idx], y[train_idx])
                proba = model.predict_proba(X.iloc[test_idx])[:, 1]
                accs.append(accuracy_score(y[test_idx], proba >= 0.5))
                losses.append(log_loss(y[test_idx], proba, labels=[0, 1]))
                aucs.append(roc_auc_score(y[test_idx], proba))
            tag = "WITH rest_days   " if with_rest else "WITHOUT rest_days"
            print(
                f"  [{arm_name:8s}] {tag}  acc={np.mean(accs):.4f}  "
                f"log_loss={np.mean(losses):.4f}  roc_auc={np.mean(aucs):.4f}"
            )


def _regression_ablation(target: str, label_col: str, seasons=None) -> None:
    df = build_margin_total_rows(seasons=seasons)
    feat_cols = feature_columns(df)
    y = df[label_col].to_numpy()
    season_arr = df["season"].to_numpy()

    for with_rest in (True, False):
        cols = feat_cols if with_rest else [c for c in feat_cols if c != REST_COL]
        X = df[cols]
        splitter = SeasonAwareTimeSeriesSplit(n_splits=N_SPLITS, seasons=season_arr, gap_seasons=0)
        for arm_name, arm_cls in (("ridge", GameMarginRidgeModel), ("xgboost", GameRegressionXGBModel)):
            maes = []
            for train_idx, test_idx in splitter.split(X):
                model = arm_cls().fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[test_idx])
                maes.append(mean_absolute_error(y[test_idx], pred))
            tag = "WITH rest_days   " if with_rest else "WITHOUT rest_days"
            print(f"  [{arm_name:8s}] {tag}  mae={np.mean(maes):.4f}")


def main() -> None:
    print("=== Win/loss classification ===")
    _classification_ablation()
    print("\n=== Margin regression ===")
    _regression_ablation("margin", "home_margin")
    print("\n=== Total regression ===")
    _regression_ablation("total", "game_total")


if __name__ == "__main__":
    main()
