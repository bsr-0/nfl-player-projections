"""Phase 6c (next_focus.md follow-up): feature-count ablation for
FINAL_CONFIG architectures.

Never done in Phases 2-5 — CAUSAL_FEATURES was held fixed throughout every
architecture/window/weighting/hyperparameter experiment. Answers: does OOS
MAE improve, plateau, or degrade as features are added?

All FINAL_CONFIG architectures are LightGBM-based gradient-boosted trees
(confirmed: src/models/single_week_ppr/architectures.py, HAS_LIGHTGBM=True
in this environment) — robust to multicollinearity by construction (each
split picks whichever correlated feature is locally best; doesn't blow up
like linear-model coefficients would). So this test measures overfitting/
diminishing-returns from feature count, NOT collinearity. Note: the
actually-deployed production system (src/models/component_predictor.py,
Ridge regression) is a different, linear model that DOES care about
multicollinearity — this ablation says nothing about that path.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COUNT_CANDIDATES: Sequence[Union[int, str]] = (10, 20, 30, "all")


def _get_lgbm_model(fitted_model):
    """Unwraps GBMRegressor / YeoJohnsonHuber / BoxCoxHuber down to the
    underlying fitted lgb.LGBMRegressor (or sklearn fallback), for
    .feature_importances_.
    """
    inner = fitted_model
    for _ in range(3):  # at most 2 levels of wrapping (transformer -> GBMRegressor -> lgb model)
        if hasattr(inner, "feature_importances_"):
            return inner
        if hasattr(inner, "model"):
            inner = inner.model
        else:
            break
    raise ValueError(f"Could not find feature_importances_ on {type(fitted_model)}")


def rank_features_by_importance(fitted_model, feature_names: Sequence[str]) -> List[str]:
    """Returns feature_names sorted by descending gain-based importance."""
    inner = _get_lgbm_model(fitted_model)
    importances = np.asarray(inner.feature_importances_)
    order = np.argsort(importances)[::-1]
    return [feature_names[i] for i in order]


def top_n_features(ranked_features: Sequence[str], n: Union[int, str]) -> List[str]:
    if n == "all":
        return list(ranked_features)
    return list(ranked_features[:n])


def run_ablation(
    positions=None,
    seasons=None,
    feature_counts: Sequence[Union[int, str]] = FEATURE_COUNT_CANDIDATES,
    output_path=None,
) -> pd.DataFrame:
    """For each position's FINAL_CONFIG fold: fit once on the full feature
    set to rank importance, then refit fresh on each candidate top-N subset
    and score on the true outer test season. Appends rows incrementally
    (Phase 3 lesson: a killed/timed-out run shouldn't lose completed work).
    """
    from pathlib import Path

    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.models.single_week_ppr.evaluate import (
        DEFAULT_VALIDATION_SEASONS, run_fold, compute_metrics,
        FoldFailureTracker,
        _build_feature_matrices, _architectures_for_fold, _append_row_to_csv,
    )
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.windows import window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    seasons = seasons or DEFAULT_VALIDATION_SEASONS
    output_path = output_path or Path("data/experiments/phase6c_feature_ablation.csv")
    positions = list(positions) if positions else POSITIONS
    rows: List[dict] = []

    tracker = FoldFailureTracker("Phase 6c (feature ablation)")
    for position in positions:
        cfg = FINAL_CONFIG[position]
        arch_name = cfg["architecture"]
        available_seasons = sorted(
            DatabaseManager().get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )

        for season in seasons:
            print(f"\n=== ABLATION {position} / test_season={season} / {cfg} ===")
            train_seasons = window_to_season_list(cfg["window"], season, available_seasons)
            if not train_seasons:
                logger.warning("Skipping %s/%s: no training seasons available", position, season)
                continue
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons,
                )
            except Exception as e:
                tracker.record(position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position].reset_index(drop=True)
            pos_test = test_df[test_df["position"] == position]
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue

            feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
            if not feature_cols:
                logger.warning("Skipping %s/%s: no CAUSAL_FEATURES columns present", position, season)
                continue

            X_train, y_train, X_test, y_test = _build_feature_matrices(pos_train, pos_test, feature_cols)

            # Fit once on the FULL feature set purely to get an importance ranking.
            try:
                ranking_model = _architectures_for_fold()[arch_name]
                ranking_model.fit(X_train, y_train)
                ranked = rank_features_by_importance(ranking_model, feature_cols)
            except Exception as e:
                logger.warning("Importance ranking failed for %s/%s: %s", position, season, e)
                continue

            for n in feature_counts:
                subset = top_n_features(ranked, n)
                try:
                    model = _architectures_for_fold()[arch_name]
                    model.fit(X_train[subset], y_train)
                    pred = pd.Series(model.predict(X_test[subset]), index=X_test.index)
                    row = {
                        "position": position, "season": season, "model": arch_name,
                        "n_features_requested": str(n), "n_features_actual": len(subset),
                        "n_features_available": len(feature_cols),
                        **compute_metrics(y_test, pred),
                    }
                    rows.append(row)
                    _append_row_to_csv(row, output_path)
                except Exception as e:
                    logger.warning("Ablation fit failed for %s/%s/n=%s: %s", position, season, n, e)

    result = pd.DataFrame(rows)
    print(f"\n{len(result)} rows appended to {output_path}")
    tracker.report(output_path)
    return result
