#!/usr/bin/env python3
"""Does explicitly modelling opportunity improve PPR projection?

Pre-registered in GAPS.md (2026-08-20) BEFORE this was run. Criteria, all
of which must hold to adopt:

  1. mean weekly MAE improvement (B vs A) >= 0.05 at a position
  2. improvement at >= 3 of 4 positions
  3. sign holds in >= 2 of 3 folds wherever a position improves
  4. snap-bucket slope moves toward 1.0 (the mechanism check)
  5. the opportunity model actually predicts opportunity

Arms share folds, evaluation population, target definition, metrics and
configuration; only the estimator structure differs.

  A  baseline            FINAL_CONFIG architecture/window/weighting,
                         predicting fantasy_points directly. Frozen.
  B  multiplicative      E[PPR] = E[snaps] x E[PPR per snap]
  C  opportunity feature baseline + predicted snaps as one extra input,
                         to separate "the signal helps" from "the
                         multiplicative structure helps"

Usage:
    python scripts/run_opportunity_experiment.py
    python scripts/run_opportunity_experiment.py --positions RB --seasons 2025
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
from src.models.single_week_ppr.opportunity import (
    OPPORTUNITY_TARGET, combine, fit_opportunity_model, fit_rate_model,
    opportunity_training_rows, predict_opportunity, snap_bucket_slope,
)
from src.models.single_week_ppr.population import (
    SNAP_BUCKET_EDGES, SNAP_BUCKET_LABELS, apply_regime, evaluation_population,
    with_participation_quality,
)
from src.models.single_week_ppr.windows import compute_recency_weights, window_to_season_list
from src.utils.database import DatabaseManager
from src.utils.leakage import filter_feature_columns
from src.models.single_week_ppr.architectures import HAS_LIGHTGBM

OUT_DIR = Path("data/experiments")
OVERALL_PATH = OUT_DIR / "opportunity_overall.csv"
BUCKET_PATH = OUT_DIR / "opportunity_by_bucket.csv"
SEASON_PATH = OUT_DIR / "opportunity_season.csv"
ROWS_PATH = OUT_DIR / "opportunity_predictions.csv"

ARMS = ("A_baseline", "B_multiplicative", "C_opportunity_feature")


def _matrices(train, test, cols):
    X_tr, X_te = train[cols], test[cols]
    if not HAS_LIGHTGBM:
        X_tr, X_te = X_tr.fillna(0), X_te.fillna(0)
    return X_tr, X_te


def run(positions, seasons) -> None:
    db = DatabaseManager()
    tracker = FoldFailureTracker("Opportunity experiment")

    for position in positions:
        cfg = FINAL_CONFIG[position]
        available = sorted(
            db.get_all_players_for_training(position=position)["season"].dropna().unique().tolist())
        base_features_all = filter_feature_columns(CAUSAL_FEATURES.get(position, []))

        for season in seasons:
            train_seasons = window_to_season_list(cfg["window"], season, available)
            if not train_seasons:
                continue
            print(f"\n=== {position} / test={season} / window={cfg['window']} "
                  f"({train_seasons[0]}-{train_seasons[-1]}) ===", flush=True)
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons)
            except Exception as e:
                tracker.record(position, season, e)
                continue

            pos_train = with_participation_quality(
                train_df[train_df["position"] == position].reset_index(drop=True))
            pos_test = with_participation_quality(
                test_df[test_df["position"] == position].reset_index(drop=True))
            trd = apply_regime(pos_train, "B_extended")
            ted = evaluation_population(pos_test)
            if len(ted) < 20 or len(trd) < 20:
                print(f"  skipped: train={len(trd)} test={len(ted)}")
                continue

            feats = [c for c in base_features_all if c in trd.columns and c in ted.columns]
            weights = compute_recency_weights(trd["season"], cfg["weighting"])
            y = ted["fantasy_points"]
            snaps_actual = ted[OPPORTUNITY_TARGET]

            # --- opportunity model (shared by B and C) -----------------
            opp_train = opportunity_training_rows(trd)
            opp_w = compute_recency_weights(opp_train["season"], cfg["weighting"])
            opp_model = fit_opportunity_model(opp_train, feats, sample_weight=opp_w)
            pred_snaps = predict_opportunity(opp_model, ted, feats)
            opp_metrics = compute_metrics(snaps_actual, pred_snaps)
            # Criterion 5: an opportunity model that cannot beat the
            # positional mean is not an opportunity model.
            naive = pd.Series(opp_train[OPPORTUNITY_TARGET].mean(), index=ted.index)
            naive_mae = float((naive - snaps_actual).abs().mean())
            print(f"  opportunity: MAE={opp_metrics['mae']:.2f} snaps  "
                  f"bias={opp_metrics['bias']:+.2f}  r={opp_metrics['spearman']:.3f}  "
                  f"(mean actual {snaps_actual.mean():.1f}; positional-mean baseline "
                  f"MAE={naive_mae:.2f})", flush=True)

            preds = {}

            # --- A: frozen baseline ------------------------------------
            X_tr, X_te = _matrices(trd, ted, feats)
            m = _architectures_for_fold()[cfg["architecture"]]
            m.fit(X_tr, trd["fantasy_points"], sample_weight=weights)
            preds["A_baseline"] = pd.Series(m.predict(X_te), index=ted.index)

            # --- B: E[snaps] x E[PPR per snap] -------------------------
            rate_model = fit_rate_model(opp_train, feats, recency_weight=opp_w)
            pred_rate = pd.Series(rate_model.predict(ted[feats]), index=ted.index)
            preds["B_multiplicative"] = combine(pred_snaps, pred_rate)

            # --- C: baseline + predicted snaps as a feature ------------
            trd_c = trd.copy()
            ted_c = ted.copy()
            # In-sample opportunity predictions for the training rows, so the
            # extra column means the same thing at fit and predict time.
            trd_c["pred_snaps"] = predict_opportunity(opp_model, trd, feats)
            ted_c["pred_snaps"] = pred_snaps
            feats_c = feats + ["pred_snaps"]
            Xc_tr, Xc_te = _matrices(trd_c, ted_c, feats_c)
            mc = _architectures_for_fold()[cfg["architecture"]]
            mc.fit(Xc_tr, trd_c["fantasy_points"], sample_weight=weights)
            preds["C_opportunity_feature"] = pd.Series(mc.predict(Xc_te), index=ted.index)

            overall, buckets, season_rows, rows = [], [], [], []
            for arm, p in preds.items():
                meta = {"position": position, "test_season": season, "arm": arm}
                slope = snap_bucket_slope(y, p, snaps_actual, SNAP_BUCKET_EDGES, SNAP_BUCKET_LABELS)
                overall.append({**meta, **compute_metrics(y, p), "bucket_slope": slope,
                                "n_train": len(trd),
                                "opp_mae": opp_metrics["mae"], "opp_bias": opp_metrics["bias"],
                                "opp_spearman": opp_metrics["spearman"],
                                "opp_mae_naive_mean": naive_mae})
                b = pd.cut(snaps_actual, bins=SNAP_BUCKET_EDGES, labels=SNAP_BUCKET_LABELS, right=True)
                for label, idx in b.groupby(b, observed=True).groups.items():
                    if len(idx) == 0:
                        continue
                    buckets.append({**meta, "bucket": str(label), "n": len(idx),
                                    "mean_actual": float(y.loc[idx].mean()),
                                    "mean_predicted": float(p.loc[idx].mean()),
                                    "mae": float((p.loc[idx] - y.loc[idx]).abs().mean()),
                                    "bias": float((p.loc[idx] - y.loc[idx]).mean())})
                per_player = pd.DataFrame({"player_id": ted["player_id"], "actual": y, "pred": p}) \
                    .groupby("player_id").sum()
                sm = compute_metrics(per_player["actual"], per_player["pred"])
                season_rows.append({**meta, **{f"season_{k}": v for k, v in sm.items()}})
                rows.append(pd.DataFrame({**meta,
                                          "player_id": ted["player_id"].to_numpy(),
                                          "week": ted["week"].to_numpy(),
                                          "snap_count": snaps_actual.to_numpy(),
                                          "pred_snaps": pred_snaps.to_numpy(),
                                          "actual_ppr": y.to_numpy(),
                                          "prediction": p.to_numpy()}))
                print(f"  {arm:24s} MAE={overall[-1]['mae']:.4f}  bias={overall[-1]['bias']:+.4f}  "
                      f"RMSE={overall[-1]['rmse']:.4f}  bucket_slope={slope:.3f}", flush=True)

            _append_df_to_csv(pd.DataFrame(overall), OVERALL_PATH)
            _append_df_to_csv(pd.DataFrame(buckets), BUCKET_PATH)
            _append_df_to_csv(pd.DataFrame(season_rows), SEASON_PATH)
            _append_df_to_csv(pd.concat(rows, ignore_index=True), ROWS_PATH)

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
