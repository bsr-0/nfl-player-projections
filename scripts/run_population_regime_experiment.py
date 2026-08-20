#!/usr/bin/env python3
"""Does the 2006-2012 era belong in the production-training population?

Four arms, one shared walk-forward evaluation. Every arm is fit with its
position's FINAL_CONFIG architecture and weighting, and scored on an
IDENTICAL held-out population (snap-confirmed rows only), so the only thing
that moves between A/B/C is which rows were eligible as training targets.

    P0_current          FINAL_CONFIG window, no participation filter.
                        The status quo, for reference only -- it varies the
                        window too, so it is not a controlled contrast.
    A_clean_modern      2013+, offense_snaps > 0.
    B_extended          + 2006-2012 admitted via the PPR > 0 proxy.
    C_extended_flagged  B, with participation_quality handed to the model.

A/B/C all use window="all" so the era rule is the only thing that differs.
That deliberately departs from WR's 3y and TE's 10y FINAL_CONFIG windows,
which would have made A and B identical at those positions (their windows
never reach 2012) and the experiment vacuous there.

Features are engineered over the full panel BEFORE the population filter
runs, so 2006-2012 still supplies career history, aging curves and rolling
windows to every arm. The contrast is about target eligibility, not about
throwing away history.

Usage:
    python scripts/run_population_regime_experiment.py
    python scripts/run_population_regime_experiment.py --positions TE --seasons 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CAUSAL_FEATURES, MIN_HISTORICAL_YEAR, POSITIONS
from src.models.single_week_ppr.evaluate import (
    DEFAULT_VALIDATION_SEASONS, FoldFailureTracker, _append_df_to_csv,
    _architectures_for_fold, compute_metrics, run_fold,
)
from src.models.single_week_ppr.final_config import FINAL_CONFIG
from src.models.single_week_ppr.population import (
    QUALITY_COLUMN, REGIMES, age_bucket, apply_regime, evaluation_population,
    regime_feature_columns, snap_bucket, tenure_bucket, with_participation_quality,
)
from src.models.single_week_ppr.windows import compute_recency_weights, window_to_season_list
from src.utils.database import DatabaseManager
from src.utils.leakage import filter_feature_columns
from src.models.single_week_ppr.architectures import HAS_LIGHTGBM

OUT_DIR = Path("data/experiments")
OVERALL_PATH = OUT_DIR / "population_regime_overall.csv"
BUCKET_PATH = OUT_DIR / "population_regime_by_bucket.csv"
SEASON_PATH = OUT_DIR / "population_regime_season_conditional.csv"
ROWS_PATH = OUT_DIR / "population_regime_predictions.csv"

ARMS = ("P0_current",) + REGIMES


def _train_population(pos_train: pd.DataFrame, arm: str, position: str, test_season: int,
                      available_seasons) -> pd.DataFrame:
    """Rows eligible as training targets for one arm."""
    if arm == "P0_current":
        seasons = window_to_season_list(FINAL_CONFIG[position]["window"], test_season, available_seasons)
        return pos_train[pos_train["season"].isin(seasons)]
    return apply_regime(pos_train, arm)


def _fit_predict(arm: str, position: str, train_rows: pd.DataFrame, test_rows: pd.DataFrame,
                 base_features) -> pd.Series:
    feature_cols = regime_feature_columns(base_features, arm)
    X_train = train_rows[feature_cols]
    X_test = test_rows[feature_cols]
    if not HAS_LIGHTGBM:
        X_train, X_test = X_train.fillna(0), X_test.fillna(0)

    weights = compute_recency_weights(train_rows["season"], FINAL_CONFIG[position]["weighting"])
    model = _architectures_for_fold()[FINAL_CONFIG[position]["architecture"]]
    model.fit(X_train, train_rows["fantasy_points"], sample_weight=weights)
    return pd.Series(model.predict(X_test), index=test_rows.index)


def _bucket_metrics(test_rows: pd.DataFrame, pred: pd.Series, meta: dict) -> pd.DataFrame:
    """MAE/bias per snap, tenure and age bucket, in one long frame."""
    buckets = {
        "snap": snap_bucket(test_rows["snap_count"]),
        "tenure": tenure_bucket(test_rows),
        "age": age_bucket(test_rows),
    }
    out = []
    for bucket_type, labels in buckets.items():
        for label, idx in labels.groupby(labels, observed=True).groups.items():
            if len(idx) == 0:
                continue
            out.append({**meta, "bucket_type": bucket_type, "bucket": str(label),
                        "mean_actual": float(test_rows.loc[idx, "fantasy_points"].mean()),
                        "mean_predicted": float(pred.loc[idx].mean()),
                        **compute_metrics(test_rows.loc[idx, "fantasy_points"], pred.loc[idx])})
    return pd.DataFrame(out)


def _season_conditional_metrics(test_rows: pd.DataFrame, pred: pd.Series, meta: dict) -> dict:
    """Season totals over PARTICIPATED weeks only.

    This is the conditional-production season total -- sum of E[PPR|plays]
    over the weeks the player actually played. It is NOT the Phase 7
    unconditional season projection, which additionally has to answer how
    many weeks the player plays at all. Reported because it is the
    season-level quantity this population definition is actually
    responsible for.
    """
    per_player = pd.DataFrame({
        "player_id": test_rows["player_id"],
        "actual": test_rows["fantasy_points"],
        "pred": pred,
    }).groupby("player_id").sum()
    m = compute_metrics(per_player["actual"], per_player["pred"])
    return {**meta, **{f"season_{k}": v for k, v in m.items()}}


def run(positions, seasons) -> None:
    db = DatabaseManager()
    tracker = FoldFailureTracker("Population regime experiment")

    for position in positions:
        available_seasons = sorted(
            db.get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )
        base_features_all = filter_feature_columns(CAUSAL_FEATURES.get(position, []))

        for test_season in seasons:
            full_history = [s for s in available_seasons if MIN_HISTORICAL_YEAR <= s < test_season]
            print(f"\n=== {position} / test={test_season} / train={full_history[0]}-{full_history[-1]} ===")
            try:
                train_df, test_df, _, _ = run_fold(
                    position, test_season, False, train_seasons_override=full_history,
                )
            except Exception as e:
                tracker.record(position, test_season, e)
                continue

            pos_train = with_participation_quality(
                train_df[train_df["position"] == position].reset_index(drop=True))
            pos_test = with_participation_quality(
                test_df[test_df["position"] == position].reset_index(drop=True))
            test_rows = evaluation_population(pos_test)
            if len(test_rows) < 20:
                print(f"  skipped: only {len(test_rows)} snap-confirmed test rows")
                continue

            base_features = [c for c in base_features_all
                             if c in pos_train.columns and c in test_rows.columns]
            print(f"  eval rows={len(test_rows)} (of {len(pos_test)} raw), features={len(base_features)}")

            overall, buckets, season_rows, row_frames = [], [], [], []
            for arm in ARMS:
                train_rows = _train_population(pos_train, arm, position, test_season, available_seasons)
                if len(train_rows) < 20:
                    print(f"  {arm:20s} skipped: {len(train_rows)} train rows")
                    continue
                pred = _fit_predict(arm, position, train_rows, test_rows, base_features)

                meta = {"position": position, "test_season": test_season, "arm": arm,
                        "architecture": FINAL_CONFIG[position]["architecture"],
                        "weighting": FINAL_CONFIG[position]["weighting"]}
                pop = {
                    "n_train": len(train_rows),
                    "train_season_min": int(train_rows["season"].min()),
                    "train_season_max": int(train_rows["season"].max()),
                    "n_train_pre2013": int((train_rows["season"] < 2013).sum()),
                    "train_zero_ppr_share": float((train_rows["fantasy_points"] <= 0).mean()),
                    "train_quality1_share": float((train_rows[QUALITY_COLUMN] == 1).mean()),
                }
                m = compute_metrics(test_rows["fantasy_points"], pred)
                overall.append({**meta, **pop, **m})
                buckets.append(_bucket_metrics(test_rows, pred, meta))
                season_rows.append(_season_conditional_metrics(test_rows, pred, meta))
                row_frames.append(pd.DataFrame({
                    **meta,
                    "player_id": test_rows["player_id"].to_numpy(),
                    "week": test_rows["week"].to_numpy(),
                    "snap_count": test_rows["snap_count"].to_numpy(),
                    "actual_ppr": test_rows["fantasy_points"].to_numpy(),
                    "prediction": pred.to_numpy(),
                }))
                print(f"  {arm:20s} n_train={pop['n_train']:6d} "
                      f"({pop['n_train_pre2013']:5d} pre-2013)  "
                      f"MAE={m['mae']:.3f}  bias={m['bias']:+.3f}  RMSE={m['rmse']:.3f}")

            if overall:
                _append_df_to_csv(pd.DataFrame(overall), OVERALL_PATH)
                _append_df_to_csv(pd.concat(buckets, ignore_index=True), BUCKET_PATH)
                _append_df_to_csv(pd.DataFrame(season_rows), SEASON_PATH)
                _append_df_to_csv(pd.concat(row_frames, ignore_index=True), ROWS_PATH)

    tracker.report(OVERALL_PATH)
    print(f"\nWrote:\n  {OVERALL_PATH}\n  {BUCKET_PATH}\n  {SEASON_PATH}\n  {ROWS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", nargs="+", default=list(POSITIONS))
    ap.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS))
    args = ap.parse_args()
    run(args.positions, args.seasons)


if __name__ == "__main__":
    main()
