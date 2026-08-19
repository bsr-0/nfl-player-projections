#!/usr/bin/env python
"""Compare causal P(plays) formulations for Phase 7's synthetic weeks.

Phase 7's season-total bias was traced to `estimate_availability_rate`
using prior seasons only: for synthetic-heavy QBs it assumed 0.466
availability where 0.244 was realized, and the bias tracked that gap
monotonically. Rather than pick a replacement formula a priori, this
compares five strictly-causal formulations on the two things that matter:
the real-vs-synthetic bias split, and season-level accuracy.

Efficiency note: availability is applied AFTER the per-week predictions and
the predictions don't depend on it, so each fold is fit ONCE and its cached
per-week predictions are re-weighted five ways. This costs one Phase 7 run,
not five.

Usage:
    python scripts/run_availability_comparison.py --positions QB
    python scripts/run_availability_comparison.py --positions QB RB WR TE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.availability import (
    AVAILABILITY_ESTIMATORS, PlayerAvailabilityHistory,
)
from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", nargs="+", default=None)
    ap.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experiments/availability_comparison.csv"))
    args = ap.parse_args()

    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.features.feature_engineering import PositionFeatureEngineer
    from src.models.single_week_ppr.evaluate import run_fold, _architectures_for_fold
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.season_projection import (
        possible_weeks_for_team, possible_weeks_for_player,
        compute_player_week_predictions,
        REGULAR_SEASON_MAX_WEEK, WeekSkipTracker,
    )
    from src.models.single_week_ppr.windows import window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    positions = list(args.positions) if args.positions else list(POSITIONS)
    db = DatabaseManager()
    week_skips = WeekSkipTracker("availability comparison")
    rows = []

    for position in positions:
        cfg = FINAL_CONFIG[position]
        fe = PositionFeatureEngineer(position)
        full_hist = db.get_all_players_for_training(position=position)
        available_seasons = sorted(full_hist["season"].dropna().unique().tolist())

        for season in args.seasons:
            print(f"\n=== AVAILABILITY COMPARISON {position} / {season} ===")
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

            # Availability history: real rows only, all seasons (estimators
            # enforce their own causal cutoffs internally).
            team_weeks = {}
            for t in full_hist["team"].dropna().unique():
                for s in set(list(train_seasons) + [season]):
                    wk = possible_weeks_for_team(db, t, s)
                    if wk:
                        team_weeks[(t, s)] = wk
            hist = PlayerAvailabilityHistory(full_hist, team_weeks)

            for player_id, g_all in pos_test.groupby("player_id"):
                g = g_all[g_all["week"] <= REGULAR_SEASON_MAX_WEEK]
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

                # Fit once, re-weight many.
                wk_preds = compute_player_week_predictions(
                    player_id, g_by_week, real_weeks, team_by_week, possible, model,
                    feat, full_history, db, fe, season, skip_tracker=week_skips)
                if not wk_preds:
                    continue

                actual = float(g["fantasy_points"].sum())
                known = sum(w["point_prediction"] for w in wk_preds if w["is_real"])
                synth = [w for w in wk_preds if not w["is_real"]]
                synth_pred_sum = sum(w["point_prediction"] for w in synth)
                # Weekly predictions can be negative (QB PPR: INTs, fumbles,
                # sacks), so this sum can be a near-zero residue of cancelling
                # terms. effective_rate divides by it, and is only a weighted
                # mean of the per-week rates when the weights share a sign --
                # otherwise it is unstable and lands outside [0,1] while every
                # underlying rate is perfectly valid. Report it only when the
                # denominator is dominated by same-sign terms.
                synth_pred_abs_sum = sum(abs(w["point_prediction"]) for w in synth)
                rate_is_meaningful = (
                    synth_pred_sum > 0
                    and synth_pred_sum >= 0.5 * synth_pred_abs_sum
                )
                realized_rate = len(real_weeks) / len(possible)

                for name, est in AVAILABILITY_ESTIMATORS.items():
                    total = known
                    for w in synth:
                        total += w["point_prediction"] * est(
                            hist, player_id, season, team_by_week[w["week"]], w["week"])
                    rows.append({
                        "estimator": name, "player": player_id, "position": position,
                        "season": season, "possible_weeks": len(possible),
                        "weeks_synthetic": len(synth),
                        "games_actually_played": len(real_weeks),
                        "realized_play_rate": realized_rate,
                        "predicted_season_total": total,
                        "actual_season_total": actual,
                        # season_total(r) = known_sum + r * synth_pred_sum for
                        # any CONSTANT rate r -- makes the 0->100% availability
                        # sensitivity curve pure post-hoc arithmetic.
                        "known_sum": known,
                        "synth_pred_sum": synth_pred_sum,
                        "synth_pred_abs_sum": synth_pred_abs_sum,
                        "effective_rate": ((total - known) / synth_pred_sum)
                                          if rate_is_meaningful else np.nan,
                    })

    if not rows:
        print("No results produced.")
        return
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n{len(df)} rows -> {args.output}")
    week_skips.report(args.output)

    df["bias"] = df.predicted_season_total - df.actual_season_total
    df["ae"] = df.bias.abs()
    df["synth_share"] = df.weeks_synthetic / df.possible_weeks
    df["bucket"] = pd.cut(df.synth_share, [-.01, .001, .25, .5, .75, 1.0],
                          labels=["none", "0-25%", "25-50%", "50-75%", "75-100%"])

    print("\n" + "=" * 78)
    print("Season-total MAE / bias by estimator (lower |bias| and MAE better)")
    print("=" * 78)
    summary = df.groupby("estimator").agg(mae=("ae", "mean"), bias=("bias", "mean"),
                                          n=("ae", "size")).round(2)
    print(summary.sort_values("mae").to_string())

    print("\n" + "=" * 78)
    print("Bias by synthetic-week share — the diagnostic that exposed the problem")
    print("(flat across buckets = availability no longer drives the error)")
    print("=" * 78)
    piv = df.pivot_table(index="estimator", columns="bucket", values="bias",
                         aggfunc="mean", observed=False).round(1)
    print(piv.to_string())


if __name__ == "__main__":
    main()
