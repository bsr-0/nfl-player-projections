#!/usr/bin/env python3
"""One-off ablation: do multi-bookmaker `game_odds` features add signal
beyond `spread_line`/`total_line` (already in every model), on the 2020+
population where `game_odds` actually has coverage?

Not part of the regular training pipeline (see features.py's
`include_market_odds` opt-in flag and market_odds.py's module docstring for
why this isn't a default) -- game_odds only covers 2020-2025 (6 seasons),
which is too little history for a fair 5-fold outer walk-forward backtest
the way phase 1/2's committed models get (that reserves 5 whole seasons as
holdout, leaving basically nothing to train on here). Uses a 3-fold inner
walk-forward split instead and reports the real, non-cherry-picked result,
same discipline as scripts/ablate_rest_days.py.

Usage:
    python scripts/ablate_market_odds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score

from src.models.game_outcome.features import build_game_outcome_rows, build_margin_total_rows, feature_columns
from src.models.game_outcome.models import (
    GameMarginRidgeModel,
    GameOutcomeLogisticModel,
    GameOutcomeXGBModel,
    GameRegressionXGBModel,
)
from src.models.position_models import SeasonAwareTimeSeriesSplit

MARKET_COLS_PREFIX = "market_"
N_SPLITS = 3
SEASONS = list(range(2020, 2026))  # game_odds coverage


def _classification_ablation() -> None:
    df = build_game_outcome_rows(seasons=SEASONS, include_market_odds=True)
    feat_cols = feature_columns(df)
    y = df["home_win"].to_numpy()
    season_arr = df["season"].to_numpy()

    for with_market in (True, False):
        cols = feat_cols if with_market else [c for c in feat_cols if not c.startswith(MARKET_COLS_PREFIX)]
        X = df[cols]
        splitter = SeasonAwareTimeSeriesSplit(n_splits=N_SPLITS, seasons=season_arr, gap_seasons=0)
        for arm_name, arm_cls in (("logistic", GameOutcomeLogisticModel), ("xgboost", GameOutcomeXGBModel)):
            accs, losses, aucs = [], [], []
            for train_idx, test_idx in splitter.split(X):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                model = arm_cls().fit(X.iloc[train_idx], y[train_idx])
                proba = model.predict_proba(X.iloc[test_idx])[:, 1]
                accs.append(accuracy_score(y[test_idx], proba >= 0.5))
                losses.append(log_loss(y[test_idx], proba, labels=[0, 1]))
                aucs.append(roc_auc_score(y[test_idx], proba))
            tag = "WITH market_odds   " if with_market else "WITHOUT market_odds"
            print(
                f"  [{arm_name:8s}] {tag}  acc={np.mean(accs):.4f}  "
                f"log_loss={np.mean(losses):.4f}  roc_auc={np.mean(aucs):.4f}"
            )


def _regression_ablation(label_col: str) -> None:
    df = build_margin_total_rows(seasons=SEASONS, include_market_odds=True)
    feat_cols = feature_columns(df)
    y = df[label_col].to_numpy()
    season_arr = df["season"].to_numpy()

    for with_market in (True, False):
        cols = feat_cols if with_market else [c for c in feat_cols if not c.startswith(MARKET_COLS_PREFIX)]
        X = df[cols]
        splitter = SeasonAwareTimeSeriesSplit(n_splits=N_SPLITS, seasons=season_arr, gap_seasons=0)
        for arm_name, arm_cls in (("ridge", GameMarginRidgeModel), ("xgboost", GameRegressionXGBModel)):
            maes = []
            for train_idx, test_idx in splitter.split(X):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                model = arm_cls().fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[test_idx])
                maes.append(mean_absolute_error(y[test_idx], pred))
            tag = "WITH market_odds   " if with_market else "WITHOUT market_odds"
            print(f"  [{arm_name:8s}] {tag}  mae={np.mean(maes):.4f}")


def main() -> None:
    print(f"Restricted to {SEASONS[0]}-{SEASONS[-1]} (game_odds coverage), {N_SPLITS}-fold inner CV\n")
    print("=== Win/loss classification ===")
    _classification_ablation()
    print("\n=== Margin regression ===")
    _regression_ablation("home_margin")
    print("\n=== Total regression ===")
    _regression_ablation("game_total")


if __name__ == "__main__":
    main()
