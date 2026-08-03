"""Outlier projection diagnostic.

Runs the full feature + prediction pipeline, then flags any player whose
projected PPG is >1.5σ from their position mean and identifies which
specific features are most likely driving the anomaly.

Usage:
    python scripts/diagnose_outliers.py [--weeks 18] [--sigma 1.5] [--position WR]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")


# ── Feature-level risk checks ─────────────────────────────────────────────────
# Each entry: (feature_name, condition_fn, human label)
# condition_fn receives the scalar value and position-median dict.
RISK_CHECKS = [
    (
        "preseason_ecr",
        lambda v, _: v >= 290,
        "preseason_ecr=300 → ADP match failed; model treats as undrafted",
    ),
    (
        "prev_season_ppg",
        lambda v, _: v < 2.0,
        "prev_season_ppg≈0 → missed most/all of prior season",
    ),
    (
        "team_changed",
        lambda v, _: v == 1,
        "team_changed=1 → share features reflect old team's system",
    ),
    (
        "fp_late6_vs_season",
        lambda v, _: v >= 2.4,
        "fp_late6_vs_season near clip-max → single outlier game inflating trend",
    ),
    (
        "fp_late6_vs_season",
        lambda v, _: v <= 0.32,
        "fp_late6_vs_season near clip-min → tiny late-season slump depressing trend",
    ),
    (
        "snap_share_pct_roll3_mean",
        lambda v, _: v < 0.05,
        "snap_share≈0 → no recent snap data (new player or ADP miss)",
    ),
    (
        "target_share_pct_roll3_mean",
        lambda v, _: v < 0.02,
        "target_share≈0 → no recent target data",
    ),
    (
        "injury_score",
        lambda v, _: v < 0.5,
        "injury_score<0.5 → significant injury concern flagged",
    ),
    (
        "depth_chart_rank",
        lambda v, _: v >= 3,
        "depth_chart_rank≥3 → listed as backup or depth",
    ),
    (
        "current_qb_epa_per_att",
        lambda v, _: v == 0.0,
        "current_qb_epa=0 → QB has no historical EPA data (rookie or missing)",
    ),
    (
        "rush_share_pct_roll3_mean",
        lambda v, _: v < 0.02,
        "rush_share≈0 → no recent carry data (RB only)",
    ),
]


def _check_risks(row: pd.Series) -> list[str]:
    """Return list of triggered risk labels for a player row."""
    triggered = []
    for feat, cond, label in RISK_CHECKS:
        val = row.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            if cond(float(val), {}):
                triggered.append(label)
        except Exception:
            continue
    return triggered


def _z_scores(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def run_diagnostic(n_weeks: int = 18, sigma_threshold: float = 1.5, position: str = None):
    from src.predict import NFLPredictor, get_prediction_target_week, get_schedule_map_for_week
    from config.settings import POSITIONS, MIN_GAMES_FOR_PREDICTION

    predictor = NFLPredictor()
    if not predictor.initialize():
        print("No trained models found. Run: python -m src.models.train")
        return

    pred_season, pred_week = get_prediction_target_week()

    # ── Build features + predictions ─────────────────────────────────────────
    print(f"Loading player data (season={pred_season}, week={pred_week})...")
    player_data = predictor._load_player_data(position=position, min_games=1)
    if player_data.empty:
        print("No player data available.")
        return

    player_data = predictor._prepare_features(player_data)
    games_per_player = player_data.groupby("player_id").size().reset_index(name="games_count")
    latest = player_data.groupby("player_id").last().reset_index()
    latest = latest.merge(games_per_player, on="player_id")

    latest["season"] = pred_season
    latest["week"] = pred_week
    schedule_map = get_schedule_map_for_week(predictor.db, pred_season, pred_week)
    if schedule_map:
        latest["opponent"] = latest["team"].map(lambda t: schedule_map.get(str(t).strip(), ("", ""))[0])
        latest["home_away"] = latest["team"].map(lambda t: schedule_map.get(str(t).strip(), ("", "unknown"))[1])
    else:
        latest["opponent"] = ""
        latest["home_away"] = "unknown"
    latest = predictor.feature_engineer.refresh_matchup_features(latest)

    print("Running predictions...")
    results = predictor.predictor.predict(latest, n_weeks=n_weeks)
    results["predicted_ppg"] = results["predicted_points"] / n_weeks

    # Join features back onto results
    feature_cols = [c for c in latest.columns if c not in results.columns or c == "player_id"]
    combined = results.merge(latest[feature_cols], on="player_id", how="left")

    # ── Position-by-position outlier detection ────────────────────────────────
    positions_to_check = [position] if position else POSITIONS
    any_outliers = False

    print(f"\n{'='*70}")
    print(f"  PROJECTION OUTLIER REPORT  —  {n_weeks}w  |  >{sigma_threshold}σ threshold")
    print(f"{'='*70}")

    for pos in positions_to_check:
        pos_df = combined[combined["position"] == pos].copy()
        if len(pos_df) < 5:
            continue

        pos_df["_z"] = _z_scores(pos_df["predicted_ppg"])
        outliers = pos_df[pos_df["_z"].abs() >= sigma_threshold].copy()
        if outliers.empty:
            continue

        any_outliers = True
        mean_ppg = pos_df["predicted_ppg"].mean()
        print(f"\n{pos}  —  position mean {mean_ppg:.1f} ppg  |  {len(outliers)} outlier(s)")
        print("-" * 70)

        for direction, subset in [("HIGH", outliers[outliers["_z"] > 0]),
                                   ("LOW",  outliers[outliers["_z"] < 0])]:
            if subset.empty:
                continue
            subset = subset.sort_values("_z", ascending=(direction == "LOW"))
            print(f"\n  {direction} OUTLIERS")
            for _, row in subset.iterrows():
                name = row.get("name", row.get("player_name", row.get("player_id", "?")))
                team = row.get("team", "?")
                ppg  = row["predicted_ppg"]
                z    = row["_z"]
                risks = _check_risks(row)

                print(f"\n  {name:<28} {team:<5}  {ppg:>5.1f} ppg  z={z:+.2f}")
                if risks:
                    for r in risks:
                        print(f"    ↳ {r}")
                else:
                    # Fall back to raw feature values for the most relevant features
                    relevant = [
                        "preseason_ecr", "prev_season_ppg", "snap_share_pct_roll3_mean",
                        "target_share_pct_roll3_mean", "fp_late6_vs_season",
                        "team_changed", "injury_score", "depth_chart_rank",
                    ]
                    shown = []
                    for feat in relevant:
                        val = row.get(feat)
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            pos_median = pos_df[feat].median() if feat in pos_df.columns else None
                            shown.append(f"{feat}={val:.2f}" + (f" (pos median {pos_median:.2f})" if pos_median is not None else ""))
                    if shown:
                        print(f"    ↳ No known risk pattern. Key features: {', '.join(shown[:4])}")

    if not any_outliers:
        print(f"\n  No outliers found at >{sigma_threshold}σ threshold. All projections look within normal range.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose outlier player projections")
    parser.add_argument("--weeks", type=int, default=18,
                        help="Projection horizon in weeks (default: 18)")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Z-score threshold for outlier flag (default: 1.5)")
    parser.add_argument("--position", choices=["QB", "RB", "WR", "TE"], default=None,
                        help="Limit to one position (default: all)")
    args = parser.parse_args()
    run_diagnostic(n_weeks=args.weeks, sigma_threshold=args.sigma, position=args.position)
