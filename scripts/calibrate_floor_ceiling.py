"""Fit the draft board's floor/ceiling bands to Step 8's own residuals.

The board has served Step 8 season totals since 2026-08-28, but the
FLOOR_ASYM_COEF / CEILING_ASYM_COEF constants in generate_draft_data.py were
fit to PreseasonProjector residuals (a model that was deleted 2026-09-17).
A band is a statement about ONE model's error distribution; applying it to
another model's numbers is not calibrated, however well it once scored.

Method (unchanged from the 2026-08-28 fit, only the model differs):

  * For each season S in 2021-2025, refit Step 8 strictly on seasons < S
    (the 2019+ window the live board uses), project S, and keep
    (predicted_total, confidence_score, position) with the realised total.
    Every row is therefore out-of-sample.
  * Relative error  r = actual / predicted - 1.
  * Two one-sided quantile regressions on [1, log(pred), confidence, pos]:
    floor at q = (1 - coverage) / 2, ceiling at q = 1 - that, so the band is
    a two-sided `coverage` interval (default 0.50, as the board publishes).
  * Holdout first (fit 2021-2023, test 2024-2025): report the breach rate
    on each side against its target. Then fit on every season and print the
    coefficients to paste into generate_draft_data.py.

Rows require at least one game played in S: a player who never took a
snap that season is not a draft-board outcome the band is meant to cover.

Usage:
    python scripts/calibrate_floor_ceiling.py [--target-coverage 0.50]
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, MODELS_DIR  # noqa: E402

TRAIN_FROM = 2019          # Step 8's training window, as generate_draft_data.py
SEASONS = list(range(2021, 2026))
HOLDOUT_FROM = 2024
FEATURES = ["const", "log_pred", "confidence_score", "pos_RB", "pos_WR", "pos_TE"]
OUT_PATH = MODELS_DIR / "floor_ceiling_step8_calibration.json"


def step8_out_of_sample(season: int) -> pd.DataFrame:
    from src.utils.database import DatabaseManager
    from src.models.preseason_features import build_multiyear_season_pairs
    from src.models.single_week_ppr.season_availability import load_player_seasons
    from src.models.season_step8 import (
        Step8SeasonModel, possible_games_for_players, with_board_metadata,
    )
    db = DatabaseManager()
    panel = load_player_seasons()
    pairs = build_multiyear_season_pairs(db, list(range(TRAIN_FROM, season)),
                                         inference_season=season)
    train = pairs[pairs["target_season"] < season]
    infer = pairs[pairs["target_season"] == season].copy()
    if train.empty or infer.empty:
        return pd.DataFrame()
    model = Step8SeasonModel().fit(train, panel, before_season=season)
    preds = model.predict(infer, possible_games=possible_games_for_players(infer, season))
    out = with_board_metadata(infer, preds)
    out["position"] = infer["position"].to_numpy()
    out["season"] = season
    return out[["player_id", "season", "position", "predicted_total", "confidence_score"]]


def actual_totals(season: int) -> pd.DataFrame:
    with sqlite3.connect(str(DB_PATH)) as conn:
        return pd.read_sql(
            "SELECT player_id, SUM(fantasy_points) AS actual_total, COUNT(*) AS games "
            "FROM player_weekly_stats WHERE season = ? AND week <= 18 "
            "AND fantasy_points IS NOT NULL GROUP BY player_id",
            conn, params=(int(season),))


def design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "const": 1.0,
        "log_pred": np.log(np.maximum(df["predicted_total"].to_numpy(dtype=float), 1.0)),
        "confidence_score": df["confidence_score"].to_numpy(dtype=float),
    }, index=df.index)
    for pos in ("RB", "WR", "TE"):
        X[f"pos_{pos}"] = (df["position"] == pos).astype(float)
    return X[FEATURES]


def fit_side(df: pd.DataFrame, q: float) -> dict:
    from statsmodels.regression.quantile_regression import QuantReg
    res = QuantReg(df["rel_error"].to_numpy(), design(df)).fit(q=q, max_iter=5000)
    return {k: float(v) for k, v in zip(FEATURES, res.params)}


def breach_rates(df: pd.DataFrame, floor_coef: dict, ceil_coef: dict) -> dict:
    X = design(df)
    lo = X.to_numpy() @ np.array([floor_coef[k] for k in FEATURES])
    hi = X.to_numpy() @ np.array([ceil_coef[k] for k in FEATURES])
    r = df["rel_error"].to_numpy()
    return {"floor_breached": float((r < lo).mean()), "ceiling_breached": float((r > hi).mean()),
            "n": int(len(df))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-coverage", type=float, default=0.50)
    args = ap.parse_args()
    q_floor = (1.0 - args.target_coverage) / 2.0
    q_ceil = 1.0 - q_floor

    frames = []
    for season in SEASONS:
        s8 = step8_out_of_sample(season)
        if s8.empty:
            print(f"  {season}: no Step 8 rows"); continue
        s8 = s8.merge(actual_totals(season), on="player_id", how="inner")
        s8 = s8[(s8["games"] >= 1) & (s8["predicted_total"] > 0)].copy()
        s8["rel_error"] = s8["actual_total"] / s8["predicted_total"] - 1.0
        frames.append(s8)
        print(f"  {season}: {len(s8)} player-seasons, MAE {np.abs(s8.actual_total - s8.predicted_total).mean():.1f}, "
              f"median rel_error {s8.rel_error.median():+.3f}")
    df = pd.concat(frames, ignore_index=True)

    fit, test = df[df.season < HOLDOUT_FROM], df[df.season >= HOLDOUT_FROM]
    fl, ce = fit_side(fit, q_floor), fit_side(fit, q_ceil)
    hold = breach_rates(test, fl, ce)
    print(f"\nholdout (fit {fit.season.min()}-{fit.season.max()}, test {test.season.min()}-{test.season.max()}, "
          f"n={hold['n']}): floor breached {hold['floor_breached']:.3f}, "
          f"ceiling breached {hold['ceiling_breached']:.3f}, target {q_floor:.3f} each side; "
          f"per-side miscalibration {abs(hold['floor_breached']-q_floor)+abs(hold['ceiling_breached']-q_floor):.3f}")

    floor_coef, ceil_coef = fit_side(df, q_floor), fit_side(df, q_ceil)
    insample = breach_rates(df, floor_coef, ceil_coef)
    print(f"final fit on {df.season.min()}-{df.season.max()} (n={len(df)}): in-sample floor "
          f"{insample['floor_breached']:.3f}, ceiling {insample['ceiling_breached']:.3f}")

    def fmt(c):
        return ("{\n    \"const\": %.6f, \"log_pred\": %.6f, \"confidence_score\": %.6f,\n"
                "    \"pos_RB\": %.6f, \"pos_WR\": %.6f, \"pos_TE\": %.6f,\n}") % tuple(c[k] for k in FEATURES)
    print("\nFLOOR_ASYM_COEF = " + fmt(floor_coef))
    print("CEILING_ASYM_COEF = " + fmt(ceil_coef))

    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "step8", "train_from": TRAIN_FROM, "seasons": SEASONS,
        "target_coverage": args.target_coverage, "q_floor": q_floor, "q_ceiling": q_ceil,
        "n_player_seasons": int(len(df)),
        "holdout": {"fit_seasons": [int(fit.season.min()), int(fit.season.max())],
                    "test_seasons": [int(test.season.min()), int(test.season.max())], **hold},
        "in_sample": insample,
        "floor_coef": floor_coef, "ceiling_coef": ceil_coef,
    }, indent=2))
    print(f"\nwritten {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
