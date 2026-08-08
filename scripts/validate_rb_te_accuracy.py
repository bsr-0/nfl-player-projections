"""Real, apples-to-apples accuracy check for the live serving path.

Calls the actual `NFLPredictor` class (same pattern as generate_app_data.py:
initialize() + predict(n_weeks=1, top_n=2000), no retraining) against past
weeks, and joins every prediction 1:1 to the real fantasy_points actual for
that exact player_id/season/week. This replaces the earlier, invalid
comparison in TRACKING.md (full predicted-population aggregate mean/median
vs. a vague "real typical" benchmark) with matched-population metrics.

CAVEAT: the currently-serving model's train_seasons include 2025
(data/models/model_metadata.json), so any 2025 week validated here is
in-sample. This script therefore checks serving-path PLUMBING (does the
2026-08-07 injury-prob fix produce sane, correctly-matched per-player
numbers) -- it is NOT a generalization/skill claim. That's covered
separately by src/evaluation/ts_backtester.py's walk-forward retrain
(TRACKING.md Sec 2a, real out-of-sample RB/TE R^2).

Historical targeting without retraining: NFLPredictor.predict() always
targets "the real upcoming week" via the module-level get_prediction_target_week()
(src/predict.py) and takes the most recent row per player from an unfiltered
full-history pull. HistoricalNFLPredictor below truncates that pull to rows
strictly before the requested (season, week), and get_prediction_target_week
is patched per-call to return that (season, week), so the currently-loaded
model artifacts are reused unmodified -- only the input snapshot changes.

Residual leakage caveat: the matchup-feature refresh's opponent-defense
lookup (_add_opp_fpts_allowed_from_db) is already week-relative/leakage-safe
by construction. Other feature blocks in _prepare_features were not
individually re-audited for this script; flagged here rather than blocking.

Usage:
    python -m scripts.validate_rb_te_accuracy --seasons 2025 --weeks 10-17
"""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import NFLPredictor
from src.utils.database import DatabaseManager
from src.evaluation.ts_backtester import _calc_metrics

IN_SAMPLE_WARNING = (
    "=" * 78 + "\n"
    "CAVEAT: the currently-serving model was trained on seasons including\n"
    "the ones validated below (see data/models/model_metadata.json). This is\n"
    "a SERVING-PATH PLUMBING check (matched-population sanity check of the\n"
    "2026-08-07 injury-prob fix), NOT an out-of-sample skill measurement.\n"
    "Real generalization numbers live in TRACKING.md Sec 2a (ts_backtester.py).\n"
    + "=" * 78
)


class HistoricalNFLPredictor(NFLPredictor):
    """NFLPredictor variant that predicts a specified past week using only
    data strictly before it, reusing the currently-loaded (not retrained)
    model artifacts."""

    def __init__(self, *args, as_of_season=None, as_of_week=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._as_of_season = as_of_season
        self._as_of_week = as_of_week

    def _load_player_data(self, position: str = None, min_games: int = 1) -> pd.DataFrame:
        df = super()._load_player_data(position, min_games)
        if df.empty:
            return df
        cutoff = self._as_of_season * 100 + self._as_of_week
        key = df["season"] * 100 + df["week"]
        return df[key < cutoff]


def predict_for_week(predictor_cache: dict, season: int, week: int, positions) -> pd.DataFrame:
    """Get real-serving-path predictions for one historical week, reusing
    already-loaded model artifacts across weeks."""
    if "predictor" not in predictor_cache:
        p = HistoricalNFLPredictor(as_of_season=season, as_of_week=week)
        if not p.initialize():
            raise RuntimeError("No trained models found; run training first.")
        predictor_cache["predictor"] = p
    else:
        predictor_cache["predictor"]._as_of_season = season
        predictor_cache["predictor"]._as_of_week = week

    predictor = predictor_cache["predictor"]
    with patch("src.predict.get_prediction_target_week", return_value=(season, week)):
        preds = predictor.predict(n_weeks=1, position=None, top_n=2000)
    if preds.empty:
        return preds
    preds = preds[preds["position"].isin(positions)].copy()
    preds["season"] = season
    preds["week"] = week
    return preds


def load_actuals(db: DatabaseManager, season: int, week: int) -> pd.DataFrame:
    query = """
        SELECT pws.player_id, pws.fantasy_points AS actual_fantasy_points,
               pws.snap_share, pws.snap_count,
               us.snap_share AS util_snap_share,
               us.target_share AS util_target_share,
               us.rush_share AS util_rush_share
        FROM player_weekly_stats pws
        LEFT JOIN utilization_scores us
            ON pws.player_id = us.player_id
            AND pws.season = us.season AND pws.week = us.week
        WHERE pws.season = ? AND pws.week = ?
    """
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    try:
        return pd.read_sql_query(query, conn, params=(season, week))
    finally:
        conn.close()


def is_starter(row, threshold: float) -> bool:
    # utilization_scores.snap_share is on a 0-100 scale (real, correctly
    # distributed: mean ~45, max 100). player_weekly_stats.snap_share is
    # broken for season 2025 (capped ~0-0.18 instead of the expected 0-1
    # range seen in every other season -- see GAPS.md 2026-08-07 entry) so
    # it is NOT used as a fallback; rows missing util_snap_share are
    # treated as unknown (non-starter) rather than trusting the broken column.
    share = row["util_snap_share"]
    if pd.isna(share):
        return False
    return share >= threshold * 100


def parse_weeks(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def build_joined(seasons, weeks, positions, predictor_cache, db):
    rows = []
    for season in seasons:
        for week in weeks:
            print(f"  predicting season={season} week={week} ...")
            preds = predict_for_week(predictor_cache, season, week, positions)
            if preds.empty:
                print(f"    no predictions returned, skipping")
                continue
            actuals = load_actuals(db, season, week)
            if actuals.empty:
                print(f"    no actuals in player_weekly_stats for {season} wk{week}, skipping")
                continue
            merged = preds.merge(actuals, on="player_id", how="inner")
            n_unmatched = len(preds) - len(merged)
            if n_unmatched:
                print(f"    {n_unmatched} predicted players had no actual row (likely DNP) for {season} wk{week}")
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def report_cut(df: pd.DataFrame, position: str, cut_name: str):
    sub = df[df["position"] == position]
    if sub.empty:
        print(f"  {position:4s} {cut_name:14s}  (no rows)")
        return
    m = _calc_metrics(sub["actual_fantasy_points"], sub["predicted_points"])
    bias = (
        sub["predicted_points"].mean() / sub["actual_fantasy_points"].mean()
        if sub["actual_fantasy_points"].mean() not in (0, np.nan) else float("nan")
    )
    print(
        f"  {position:4s} {cut_name:14s}  n={m['n']:4d}  MAE={m['mae']:.2f}  "
        f"RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}  bias(pred/actual)={bias:.2f}"
    )


def report_zero_prediction_diagnostic(df: pd.DataFrame, position: str):
    sub = df[(df["position"] == position) & (df["predicted_points"] < 0.5)]
    if sub.empty:
        print(f"  {position}: no near-zero predictions")
        return
    n_starters = sub.apply(lambda r: is_starter(r, 0.5), axis=1).sum()
    pct_actual_gt3 = (sub["actual_fantasy_points"] > 3.0).mean() * 100
    print(
        f"  {position} near-zero predictions (predicted<0.5): n={len(sub)}, "
        f"actual median={sub['actual_fantasy_points'].median():.2f}, "
        f"pct(actual>3.0)={pct_actual_gt3:.1f}%, n_flagged_starters={n_starters}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", nargs="+", type=int, default=[2025])
    parser.add_argument("--weeks", type=str, default="10-15")
    parser.add_argument("--positions", nargs="+", default=["QB", "RB", "WR", "TE"])
    parser.add_argument("--starter-snap-share-threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    weeks = parse_weeks(args.weeks)
    print(IN_SAMPLE_WARNING)
    print(f"\nValidating seasons={args.seasons} weeks={weeks} positions={args.positions}\n")

    db = DatabaseManager()
    predictor_cache = {}
    joined = build_joined(args.seasons, weeks, args.positions, predictor_cache, db)

    if joined.empty:
        print("No matched prediction/actual rows produced. Aborting.")
        return

    joined["is_starter"] = joined.apply(lambda r: is_starter(r, args.starter_snap_share_threshold), axis=1)
    n_starters = joined["is_starter"].sum()
    print(f"\nMatched rows: {len(joined)} ({n_starters} starters, {len(joined) - n_starters} backups, "
          f"threshold={args.starter_snap_share_threshold})\n")

    print("Metrics by position (full population vs. starters-only):")
    for pos in args.positions:
        report_cut(joined, pos, "full")
        report_cut(joined[joined["is_starter"]], pos, "starters-only")

    print("\nNear-zero-prediction diagnostic (RB/TE focus):")
    for pos in [p for p in args.positions if p in ("RB", "TE")]:
        report_zero_prediction_diagnostic(joined, pos)

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("data/backtest_results") / f"rb_te_accuracy_check_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["player_id", "name", "position", "season", "week", "predicted_points",
            "actual_fantasy_points", "is_starter"]
    cols = [c for c in cols if c in joined.columns]
    joined[cols].to_csv(out_path, index=False)
    print(f"\nRow-level output written to: {out_path}")


if __name__ == "__main__":
    main()
