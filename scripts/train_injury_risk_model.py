#!/usr/bin/env python3
"""Train and persist the injury-risk model (injury_prob_advanced / combined).

Replaces two prior approaches, both measured against real next-week
injury-report outcomes and found worse than a trivial "always predict the
base rate" baseline:

  - The hand-tuned heuristic (position/age/workload/history multipliers,
    capped at 25%): Brier 0.057 vs 0.033, overpredicting risk ~5.5x
    uniformly (mean predicted 18.8% vs true weekly incidence 3.4%). This was
    the ~19% availability floor predict.py used to apply to everyone before
    the 2026-09-11 fix that switched availability to the real injury report.
  - A RandomForestClassifier with class_weight='balanced' (+ optional
    SMOTE), coded but never actually wired to run (no caller ever passed
    fit_classifier=True): tested anyway, raw predict_proba averaged 48.5%,
    Brier 0.267 -- far worse. Balancing techniques fix RANKING under severe
    imbalance; they do not produce a calibrated probability on their own.

A plain, unweighted logistic regression on the same underlying inputs (age,
week, workload, prior-injury count, position) is calibrated by construction
for a rare binary outcome, and out-of-sample (train<=2023, test 2024-2025)
beat every alternative tried, including a properly isotonic-recalibrated
version of the RandomForestClassifier above (Brier 0.0339 vs 0.0331 --
statistically indistinguishable, without needing a second held-out slice
just for calibration). See src/features/advanced_rookie_injury.py for the
full comparison and fit_injury_risk_model for the persisted artifact.

Usage:
    python scripts/train_injury_risk_model.py                # train on all history
    python scripts/train_injury_risk_model.py --held-out-eval # also report OOS metrics
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.database import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.features.advanced_rookie_injury import AdvancedInjuryPredictor, fit_injury_risk_model


def _labeled_frame() -> pd.DataFrame:
    """Real historical rows with workload/age/position features and the
    actual next-week injury-report outcome, matching the exact frame
    add_advanced_injury_features builds in production (minus the risk-model
    prediction step itself, which is what we're about to fit).
    """
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=1)
    df = FeatureEngineer()._merge_injury_data_from_cache(df)  # real injury_score/is_injured

    predictor = AdvancedInjuryPredictor()
    # Reuses the same workload-shift and prior-injury-count logic
    # add_advanced_injury_features runs, without needing a persisted model to
    # exist yet (that's what building this frame is for).
    if "age" not in df.columns:
        df["age"] = 26
    if "rushing_attempts" in df.columns and "targets" in df.columns:
        raw_weekly_workload = df["rushing_attempts"].fillna(0) + df["targets"].fillna(0)
    else:
        raw_weekly_workload = pd.Series(15.0, index=df.index)
    order = df.sort_values(["player_id", "season", "week"]).index
    shifted = raw_weekly_workload.loc[order].groupby(
        [df.loc[order, "player_id"], df.loc[order, "season"]]).shift(1).fillna(0)
    df["weekly_workload"] = shifted.reindex(df.index)
    df["season_workload"] = shifted.groupby(
        [df.loc[order, "player_id"], df.loc[order, "season"]]).cumsum().reindex(df.index)
    df = predictor._compute_prior_injury_counts(df)

    df = df.sort_values(["player_id", "season", "week"])
    df["is_injured_next_week"] = df.groupby(["player_id", "season"])["is_injured"].shift(-1)
    df = df.dropna(subset=["is_injured_next_week"])
    df["is_injured_next_week"] = df["is_injured_next_week"].astype(int)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out-eval", action="store_true",
                    help="Also report train<=2023/test 2024-2025 metrics before "
                         "refitting on the full history for the persisted artifact.")
    args = ap.parse_args()

    print("Loading historical injury outcomes...")
    df = _labeled_frame()
    print(f"  {len(df)} labeled player-weeks, {df.is_injured_next_week.sum():.0f} positive "
          f"({df.is_injured_next_week.mean():.2%} prevalence)")

    if args.held_out_eval:
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from src.features.advanced_rookie_injury import (
            _build_injury_risk_matrix, INJURY_RISK_NUMERIC_FEATURES)
        train, test = df[df.season <= 2023], df[df.season.isin([2024, 2025])]
        artifact = fit_injury_risk_model(train, path=Path("/tmp/_injury_risk_eval_only.joblib"))
        from src.features.advanced_rookie_injury import predict_injury_risk
        p = predict_injury_risk(test, artifact)
        p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
        print(f"  Held-out (train<=2023, test 2024-2025): "
              f"brier={brier_score_loss(test.is_injured_next_week, p_clipped):.4f} "
              f"logloss={log_loss(test.is_injured_next_week, p_clipped):.4f} "
              f"auc={roc_auc_score(test.is_injured_next_week, p_clipped):.3f} "
              f"mean_pred={p.mean():.4f} (true prevalence {test.is_injured_next_week.mean():.4f})")

    print("Fitting production model on the full history...")
    artifact = fit_injury_risk_model(df)
    print(f"  Trained on {artifact['n_train_rows']} rows, "
          f"{artifact['train_prevalence']:.2%} prevalence.")
    print(f"  Risk-level thresholds: high>{artifact['risk_level_thresholds']['high']:.4f}, "
          f"medium>{artifact['risk_level_thresholds']['medium']:.4f}")
    from config.settings import MODELS_DIR
    from src.features.advanced_rookie_injury import INJURY_RISK_MODEL_FILENAME
    print(f"  Saved to {MODELS_DIR / INJURY_RISK_MODEL_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
