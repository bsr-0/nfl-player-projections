#!/usr/bin/env python3
"""Minimal fixed-grid tuning for the Elo rating's own hyperparameters
(k, home_advantage, season_regression) -- see src/models/game_outcome/elo.py.

Same discipline as scripts/train_game_outcome_model.py's `--tune` flag
(src/models/game_outcome/tuning.py): tuned ONCE on data strictly before the
earliest walk-forward holdout season, scored via inner walk-forward CV, then
reported. Not folded into tuning.py's `grid_search`/`tune_all_arms` because
those tune a MODEL's hyperparameters against a fixed X -- Elo params affect
FEATURE CONSTRUCTION itself (a different combo means rebuilding the whole
feature frame, not refitting one estimator), so this is its own small loop.

Scored against the win/loss classifier's log-loss (the same primary metric
scripts/train_game_outcome_model.py's own `--tune` uses) using the baseline
GameOutcomeLogisticModel -- Elo is a single shared feature consumed by both
phase-1 (classification) and phase-2 (margin/total regression) models, and
tuning it separately per target would just be more grid combinations for
the same underlying rating; log-loss on the classifier is the cheapest
reasonable proxy for "is this a better power rating."

Usage:
    python scripts/tune_elo_params.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import log_loss

from config.settings import GAME_OUTCOME_MODEL_CONFIG
from src.models.game_outcome.features import build_game_outcome_rows, feature_columns
from src.models.game_outcome.models import GameOutcomeLogisticModel
from src.models.position_models import SeasonAwareTimeSeriesSplit

# Deliberately small -- 3 x 3 x 3 = 27 combinations, each requiring one
# feature-frame rebuild + a few-fold inner CV fit. A "minimal" check that
# the documented 538-style defaults (k=20, home_advantage=55,
# season_regression=0.75) aren't badly wrong, not a real hyperparameter
# search.
ELO_GRID = {
    "k": [10.0, 20.0, 30.0],
    "home_advantage": [25.0, 55.0, 85.0],
    "season_regression": [0.5, 0.75, 1.0],
}
INNER_CV_SPLITS = 3


def _inner_cv_log_loss(elo_params: dict, pre_holdout_seasons: list[int]) -> float:
    full_params = {"initial": 1500.0, "mov_multiplier": True, **elo_params}
    df = build_game_outcome_rows(seasons=pre_holdout_seasons, elo_params=full_params)
    feat_cols = feature_columns(df)
    X = df[feat_cols]
    y = df["home_win"].to_numpy()
    season_arr = df["season"].to_numpy()

    cv = SeasonAwareTimeSeriesSplit(n_splits=INNER_CV_SPLITS, seasons=season_arr, gap_seasons=0)
    scores = []
    for train_idx, test_idx in cv.split(X):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = GameOutcomeLogisticModel().fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(log_loss(y[test_idx], proba, labels=[0, 1]))
    return float(np.mean(scores)) if scores else float("inf")


def main() -> None:
    full = build_game_outcome_rows()
    unique_seasons = sorted(full["season"].unique())
    n_splits = GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    holdout_start = unique_seasons[-n_splits]
    pre_holdout_seasons = [s for s in unique_seasons if s < holdout_start]
    print(
        f"Tuning Elo params on {pre_holdout_seasons[0]}-{pre_holdout_seasons[-1]} "
        f"(holdout starts {holdout_start}, never scored here)...\n"
    )

    default = GAME_OUTCOME_MODEL_CONFIG["elo_params"]
    default_score = _inner_cv_log_loss(
        {"k": default["k"], "home_advantage": default["home_advantage"], "season_regression": default["season_regression"]},
        pre_holdout_seasons,
    )
    print(f"Committed default {dict(k=default['k'], home_advantage=default['home_advantage'], season_regression=default['season_regression'])}: log_loss={default_score:.4f}\n")

    keys = list(ELO_GRID)
    best_params, best_score = None, default_score
    for combo in itertools.product(*ELO_GRID.values()):
        params = dict(zip(keys, combo))
        score = _inner_cv_log_loss(params, pre_holdout_seasons)
        marker = ""
        if score < best_score:
            best_score, best_params = score, params
            marker = "  <- best so far"
        print(f"  {params}  log_loss={score:.4f}{marker}")

    print()
    if best_params is None:
        print(f"No grid combination beat the committed default (log_loss={default_score:.4f}) -- keeping it as-is.")
    else:
        improvement = default_score - best_score
        print(f"Best: {best_params}  log_loss={best_score:.4f}  (default was {default_score:.4f}, improvement={improvement:.4f})")
        if improvement < 0.001:
            print("Improvement is below noise (< 0.001 log_loss) -- not worth changing the committed default over.")
        else:
            print("This beats the committed default by a non-trivial margin -- consider updating GAME_OUTCOME_MODEL_CONFIG['elo_params'].")


if __name__ == "__main__":
    main()
