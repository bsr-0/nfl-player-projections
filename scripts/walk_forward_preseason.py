#!/usr/bin/env python3
"""Real walk-forward validation for season-total preseason projections.

TRACKING.md §2b/§2c compared PreseasonProjector (production) against a
multi-year + team-aware candidate (preseason_features.py) on a single
pooled 2023-2025 holdout -- exactly the kind of one-shot split §6's
"Known pitfalls" warns is unstable on small test sets (an RB test once
showed R2 moving the wrong direction on a 17-50 row split).

This reruns both as a real expanding-window walk-forward: for each test
season, fit fresh on every season strictly before it, score against that
season alone, repeat, then report per-fold and aggregate metrics.

The candidate model here is a single Ridge-on-full-features fit per
position (same architecture as the WR/TE winners in §2c). It does not
reproduce the bespoke per-position variants (QB's nonlinear PositionModel,
RB's UpstreamCalibrator) -- those were never captured in a reusable script
and are out of scope for this pass; see TRACKING.md for the note.

Phase 8 (next_focus.md) adds a third arm: "phase7" -- the summed-weekly
season projection from src/models/single_week_ppr/season_projection.py,
loaded from its already-computed output CSV rather than retrained here.
Two honesty caveats, not smoothed over: (1) each arm is scored on its own
natural eligible population (production/candidate already don't share one
either -- different SQL-level filters), so `n` is printed per arm per fold
for the reader to judge comparability, not forced into an intersection;
(2) the phase7 CSV only covers 2023-2025, so its arm is silently skipped
for folds outside that range while production/candidate still run their
full default walk-forward (back to ~2009) unaffected.

Usage:
    python scripts/walk_forward_preseason.py
    python scripts/walk_forward_preseason.py --test-seasons 2022 2023 2024 2025
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import DATA_DIR
from src.utils.database import DatabaseManager
from src.utils.data_manager import DataManager
from src.models.preseason_projector import (
    PreseasonProjector,
    RIDGE_ALPHA_BY_POSITION,
    MIN_SAMPLES,
)
from src.models.preseason_features import build_multiyear_season_pairs

POSITIONS = ["QB", "RB", "WR", "TE"]

CANDIDATE_EXCLUDE = {
    "player_id", "player_name", "position", "birth_date", "season",
    "target_season", "season_total", "dest_team", "prior_team",
}


def _fit_candidate(pos_df: pd.DataFrame, alpha: float):
    features = [
        c for c in pos_df.columns
        if c not in CANDIDATE_EXCLUDE and pos_df[c].dtype.kind in "fi"
    ]
    X = pos_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pos_df["season_total"].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return model, scaler, features


def _predict_candidate(model, scaler, features, pos_df: pd.DataFrame) -> np.ndarray:
    X = pos_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict(scaler.transform(X))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "r2": round(float(r2_score(y_true, y_pred)), 3) if len(y_true) > 1 else None,
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 1),
    }


def _aggregate(fold_metrics: list) -> dict:
    """Mean/std across folds (unweighted -- each season counts once,
    regardless of player count) plus a pooled figure (all folds' predictions
    concatenated) for direct comparison against the old single-split numbers."""
    if not fold_metrics:
        return {}
    r2s = [m["r2"] for m in fold_metrics if m["r2"] is not None]
    maes = [m["mae"] for m in fold_metrics]
    rmses = [m["rmse"] for m in fold_metrics]
    return {
        "n_folds": len(fold_metrics),
        "total_n": sum(m["n"] for m in fold_metrics),
        "r2_mean": round(float(np.mean(r2s)), 3) if r2s else None,
        "r2_std": round(float(np.std(r2s)), 3) if r2s else None,
        "mae_mean": round(float(np.mean(maes)), 1),
        "rmse_mean": round(float(np.mean(rmses)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--test-seasons", nargs="+", type=int, default=None,
        help="Seasons to hold out one at a time (default: every season with "
             "--min-train-seasons+ prior seasons of data available)",
    )
    parser.add_argument(
        "--min-train-seasons", type=int, default=3,
        help="Minimum prior seasons required before a season can be a test fold",
    )
    parser.add_argument(
        "--phase7-csv", type=Path,
        default=Path("data/experiments/phase7_season_projection.csv"),
        help="Phase 7 (summed-weekly) season projection output CSV -- loaded "
             "as-is, not retrained. Pass a missing/empty path to skip that arm.",
    )
    parser.add_argument(
        "--intersect-populations", action="store_true",
        help="Restrict all 3 arms to the common player_id intersection per "
             "position/fold before scoring (isolates architecture from "
             "population-breadth differences; models are still trained on "
             "each arm's own full training set, only the *scored* test rows "
             "are restricted). Skips a position/fold if the intersection is "
             "too small (<MIN_INTERSECTION) to trust.",
    )
    args = parser.parse_args()

    db = DatabaseManager()
    dm = DataManager()
    all_seasons = sorted(dm.get_available_seasons_from_db())
    print(f"Seasons in DB: {all_seasons}")

    print("\nBuilding production (PreseasonProjector) season pairs...")
    prod_pairs = PreseasonProjector._build_season_pairs(db, all_seasons)
    print(f"  {len(prod_pairs)} rows")

    print("Building candidate (multi-year + team-aware) season pairs...")
    cand_pairs = build_multiyear_season_pairs(db, all_seasons)
    print(f"  {len(cand_pairs)} rows")

    if args.phase7_csv.exists():
        phase7_df = pd.read_csv(args.phase7_csv)
        print(f"Loaded phase7 (summed-weekly) projections: {len(phase7_df)} rows "
              f"from {args.phase7_csv} (seasons {sorted(phase7_df['season'].unique())})")
    else:
        phase7_df = pd.DataFrame()
        print(f"No phase7 CSV found at {args.phase7_csv} -- that arm will be skipped.")

    if args.test_seasons:
        test_seasons = args.test_seasons
    else:
        earliest = min(all_seasons)
        test_seasons = [
            s for s in all_seasons
            if (s - earliest) >= args.min_train_seasons
            and s in set(prod_pairs["curr_season"]) if not prod_pairs.empty
        ]
    print(f"\nTest seasons (walk-forward folds): {test_seasons}")

    VARIANTS = ("production", "candidate", "phase7")
    fold_records = {v: {p: [] for p in POSITIONS} for v in VARIANTS}

    MIN_INTERSECTION = 15

    for test_season in test_seasons:
        print(f"\n--- Fold: test_season={test_season} ---")

        train_p = prod_pairs[prod_pairs["curr_season"] < test_season]
        test_p = prod_pairs[prod_pairs["curr_season"] == test_season]
        proj = None
        if not train_p.empty and not test_p.empty:
            proj = PreseasonProjector()
            proj.fit(train_p)

        train_c = cand_pairs[cand_pairs["target_season"] < test_season] if not cand_pairs.empty else cand_pairs
        test_c = cand_pairs[cand_pairs["target_season"] == test_season] if not cand_pairs.empty else cand_pairs

        for pos in POSITIONS:
            pos_test_p = test_p[test_p["position"] == pos] if proj is not None else test_p.iloc[0:0]
            pos_train_c = train_c[train_c["position"] == pos] if not train_c.empty else train_c
            pos_test_c = test_c[test_c["position"] == pos] if not test_c.empty else test_c
            pos_test_p7 = (
                phase7_df[(phase7_df["position"] == pos) & (phase7_df["season"] == test_season)]
                if not phase7_df.empty else phase7_df
            )

            if args.intersect_populations:
                common_ids = (
                    set(pos_test_p["player_id"]) & set(pos_test_c["player_id"])
                    & set(pos_test_p7["player"])
                )
                if len(common_ids) < MIN_INTERSECTION:
                    print(f"  [{pos}] intersection too small ({len(common_ids)} < "
                          f"{MIN_INTERSECTION}) -- skipping this position/fold entirely")
                    continue
                pos_test_p = pos_test_p[pos_test_p["player_id"].isin(common_ids)]
                pos_test_c = pos_test_c[pos_test_c["player_id"].isin(common_ids)]
                pos_test_p7 = pos_test_p7[pos_test_p7["player"].isin(common_ids)]
                print(f"  [{pos}] intersected population: {len(common_ids)} players")

            # --- Production ---
            if proj is not None and pos in proj.models and not pos_test_p.empty:
                preds = proj.predict(pos_test_p, pos)
                actual = pos_test_p["season_total"].to_numpy(dtype=float)
                m = _metrics(actual, preds)
                m["test_season"] = test_season
                fold_records["production"][pos].append(m)
                print(f"  production {pos}: n={m['n']} R2={m['r2']} MAE={m['mae']}")

            # --- Candidate ---
            if len(pos_train_c) >= MIN_SAMPLES and not pos_test_c.empty:
                model, scaler, features = _fit_candidate(pos_train_c, RIDGE_ALPHA_BY_POSITION[pos])
                preds = _predict_candidate(model, scaler, features, pos_test_c)
                actual = pos_test_c["season_total"].to_numpy(dtype=float)
                m = _metrics(actual, preds)
                m["test_season"] = test_season
                fold_records["candidate"][pos].append(m)
                print(f"  candidate  {pos}: n={m['n']} R2={m['r2']} MAE={m['mae']}")

            # --- Phase 7 (summed-weekly) -- loaded, not retrained ---
            if not pos_test_p7.empty:
                actual = pos_test_p7["actual_season_total"].to_numpy(dtype=float)
                preds = pos_test_p7["predicted_season_total"].to_numpy(dtype=float)
                m = _metrics(actual, preds)
                m["test_season"] = test_season
                fold_records["phase7"][pos].append(m)
                print(f"  phase7     {pos}: n={m['n']} R2={m['r2']} MAE={m['mae']}")

    print(f"\n{'='*70}\nAGGREGATE (mean across folds, unweighted)\n{'='*70}")
    summary = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for pos in POSITIONS:
            agg = _aggregate(fold_records[variant][pos])
            summary[variant][pos] = agg
            if agg:
                print(
                    f"  {variant:10s} {pos}: n_folds={agg['n_folds']} "
                    f"R2={agg['r2_mean']}±{agg['r2_std']} MAE={agg['mae_mean']} "
                    f"RMSE={agg['rmse_mean']} (total_n={agg['total_n']})"
                )

    out = {
        "generated_at": datetime.now().isoformat(),
        "test_seasons": test_seasons,
        "min_train_seasons": args.min_train_seasons,
        "fold_records": fold_records,
        "summary": summary,
    }
    out_dir = DATA_DIR / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"walk_forward_preseason_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
