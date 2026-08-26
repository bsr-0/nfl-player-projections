#!/usr/bin/env python
"""Explain the ~40-point bias step that appears at a player's FIRST synthetic week.

The availability comparison (data/experiments/availability_comparison.csv)
showed the synthetic-share bias gradient is essentially invariant to the
availability formulation: all five estimators produce a max bucket-to-bucket
step of 40.0-41.1. A weight applied to the synthetic contribution cannot move
a discontinuity that appears as soon as ONE synthetic week exists, so the
mechanism lives in how the synthetic ROW is built, not in how it is weighted.

Two outputs, deliberately kept separate:

  synthetic_row_weeks.csv    one row per player-week, with the prediction, the
                             real/synthetic status, the position within the
                             synthetic run, and whether the week falls before
                             the first / after the last real appearance.

  synthetic_row_features.csv per-feature values on the last real row before a
                             player's first synthetic week vs. the first
                             synthetic row itself, so carried-forward usage and
                             role variables can be inspected directly.

Track B (the -30.6 pt real-row bias on continuously-observed QBs) is NOT
addressed here on purpose -- it is a per-week calibration problem on genuine
observations and must not contaminate this diagnostic.

Usage:
    python scripts/run_synthetic_row_diagnostic.py --positions QB
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS


def classify_weeks(wk_preds):
    """Annotate each week with its position relative to the player's real
    appearances. `synth_index` counts within a consecutive synthetic run
    (1 = first synthetic week of that run)."""
    real_weeks = [w["week"] for w in wk_preds if w["is_real"]]
    first_real = min(real_weeks) if real_weeks else None
    last_real = max(real_weeks) if real_weeks else None

    run = 0
    for rec in sorted(wk_preds, key=lambda r: r["week"]):
        if rec["is_real"]:
            run = 0
            rec["synth_index"] = 0
            rec["position"] = "real"
        else:
            run += 1
            rec["synth_index"] = run
            if first_real is not None and rec["week"] < first_real:
                rec["position"] = "before_first_real"
            elif last_real is not None and rec["week"] > last_real:
                rec["position"] = "after_last_real"
            else:
                rec["position"] = "interior"
    return wk_preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", nargs="+", default=["QB"])
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=list(DEFAULT_VALIDATION_SEASONS))
    ap.add_argument("--outdir", type=Path, default=Path("data/experiments"))
    args = ap.parse_args()

    from config.settings import CAUSAL_FEATURES
    from src.features.feature_engineering import PositionFeatureEngineer
    from src.models.single_week_ppr.evaluate import run_fold, _architectures_for_fold
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.season_projection import (
        possible_weeks_for_player, compute_player_week_predictions,
        REGULAR_SEASON_MAX_WEEK, regular_season_max_week, WeekSkipTracker,
    )
    from src.models.single_week_ppr.windows import window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    db = DatabaseManager()
    week_skips = WeekSkipTracker("synthetic row diagnostic")
    week_rows, feat_rows = [], []

    for position in args.positions:
        cfg = FINAL_CONFIG[position]
        fe = PositionFeatureEngineer(position)
        full_hist = db.get_all_players_for_training(position=position)
        available_seasons = sorted(full_hist["season"].dropna().unique().tolist())

        for season in args.seasons:
            print(f"\n=== SYNTHETIC ROW DIAGNOSTIC {position} / {season} ===")
            train_seasons = window_to_season_list(cfg["window"], season, available_seasons)
            if not train_seasons:
                continue
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons)
            except Exception as e:
                print(f"  fold failed: {e}")
                continue

            pos_train = train_df[train_df["position"] == position].reset_index(drop=True)
            pos_test = test_df[test_df["position"] == position].copy()
            if len(pos_test) < 20:
                continue

            feat = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feat = [c for c in feat if c in pos_train.columns and c in pos_test.columns]
            model = _architectures_for_fold()[cfg["architecture"]]
            model.fit(pos_train[feat], pos_train["fantasy_points"])
            full_history = pd.concat([pos_train, pos_test], ignore_index=True)

            for player_id, g_all in pos_test.groupby("player_id"):
                g = g_all[g_all["week"] <= regular_season_max_week(season)]
                if g.empty:
                    continue
                g_by_week = {int(w): sub for w, sub in g.groupby(g["week"].astype(int))}
                real_weeks = set(g_by_week.keys())
                real_team_by_week = {int(w): sub["team"].iloc[0] for w, sub in g_by_week.items()}
                possible, team_by_week = possible_weeks_for_player(
                    db, real_team_by_week, season, player_id=player_id,
                    skip_tracker=week_skips)
                if not possible:
                    continue

                wk_preds = compute_player_week_predictions(
                    player_id, g_by_week, real_weeks, team_by_week, possible, model,
                    feat, full_history, db, fe, season, capture_rows=True,
                    skip_tracker=week_skips)
                if not wk_preds:
                    continue

                n_real = sum(1 for w in wk_preds if w["is_real"])
                n_synth = len(wk_preds) - n_real
                # The step is a within-player transition, so it is only
                # identifiable on players who have BOTH kinds of week.
                if n_real == 0 or n_synth == 0:
                    continue

                wk_preds = classify_weeks(wk_preds)
                for rec in wk_preds:
                    week_rows.append({
                        "position": position, "season": season, "player": player_id,
                        "week": rec["week"], "is_real": rec["is_real"],
                        "synth_index": rec["synth_index"], "position_class": rec["position"],
                        "prediction": rec["point_prediction"],
                        "actual": rec["actual_value"],
                        "n_real_weeks": n_real, "n_synth_weeks": n_synth,
                    })

                # Feature-level: last real row strictly before the first
                # synthetic week, vs. that first synthetic row.
                synths = sorted([r for r in wk_preds if not r["is_real"]],
                                key=lambda r: r["week"])
                reals = sorted([r for r in wk_preds if r["is_real"]],
                               key=lambda r: r["week"])
                first_synth = synths[0]
                prior_reals = [r for r in reals if r["week"] < first_synth["week"]]
                if not prior_reals:
                    continue
                last_real = prior_reals[-1]

                rrow, srow = last_real["feature_row"], first_synth["feature_row"]
                for col in feat:
                    rv, sv = rrow.get(col), srow.get(col)
                    feat_rows.append({
                        "position": position, "season": season, "player": player_id,
                        "last_real_week": last_real["week"],
                        "first_synth_week": first_synth["week"],
                        "feature": col,
                        "real_value": pd.to_numeric(rv, errors="coerce"),
                        "synth_value": pd.to_numeric(sv, errors="coerce"),
                        "real_pred": last_real["point_prediction"],
                        "synth_pred": first_synth["point_prediction"],
                    })

    args.outdir.mkdir(parents=True, exist_ok=True)
    wdf = pd.DataFrame(week_rows)
    fdf = pd.DataFrame(feat_rows)
    fdf["delta"] = fdf["synth_value"] - fdf["real_value"]
    wpath = args.outdir / "synthetic_row_weeks.csv"
    fpath = args.outdir / "synthetic_row_features.csv"
    wdf.to_csv(wpath, index=False)
    fdf.to_csv(fpath, index=False)
    print(f"\nWrote {len(wdf)} player-weeks -> {wpath}")
    print(f"Wrote {len(fdf)} feature comparisons -> {fpath}")
    week_skips.report(wpath)

    if not wdf.empty:
        print("\nMEAN PREDICTION BY WEEK TYPE")
        summary = wdf.groupby(wdf.apply(
            lambda r: "real" if r.is_real else f"synthetic #{min(int(r.synth_index), 4)}"
                      + ("+" if r.synth_index >= 4 else ""), axis=1)
        )["prediction"].agg(["count", "mean"])
        print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
