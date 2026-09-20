#!/usr/bin/env python3
"""Predict scheduled-but-not-yet-played games using the saved game-outcome/
game-margin models.

This is the serving path for src/models/game_outcome/ -- everything else in
that package (features.py's build_game_outcome_rows/build_margin_total_rows,
the two training scripts) only ever operates on COMPLETED historical games.
This script is what actually answers "who wins next week": it calls
build_prediction_rows() (same feature pipeline, sourced from schedule rows
that have no score yet) and runs the models saved by
scripts/train_game_outcome_model.py / scripts/train_game_margin_model.py
against them.

Usage:
    python scripts/predict_upcoming_games.py --season 2026 --week 2
    python scripts/predict_upcoming_games.py --season 2026          # all remaining scheduled weeks

Requires the model artifacts already saved to data/models/ -- run (at
least once, after any feature change):
    python scripts/train_game_outcome_model.py
    python scripts/train_game_margin_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.settings import MODELS_DIR
from src.models.game_outcome.features import build_prediction_rows, feature_columns
from src.models.game_outcome.models import load_model
from src.models.game_outcome.market_picks import spread_pick, total_pick


def _load_if_exists(path: Path):
    if not path.exists():
        return None
    return load_model(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None, help="Omit to predict every remaining scheduled week.")
    parser.add_argument(
        "--market-model", choices=("ridge", "xgb", "rf"), default="ridge",
        help="Margin/total model used for the explicit ATS and O/U picks (default: ridge).",
    )
    args = parser.parse_args()

    rows = build_prediction_rows(args.season, week=args.week)
    if rows.empty:
        print(f"No scheduled-but-unplayed games found for season={args.season} week={args.week}.")
        print("(Either everything through that point is already final, or the schedule doesn't reach that far yet.)")
        return

    feat_cols = feature_columns(rows)
    # SQLite can return newly scraped market lines as strings even though the
    # historical training frame held them as floats.  Coerce the complete
    # numeric feature matrix here so strict estimators (notably XGBoost) see
    # the same schema at serving time as they saw during fitting.
    X = rows[feat_cols].apply(pd.to_numeric, errors="coerce")

    # (output column, artifact filename, "proba" for classifiers or "point" for regressors)
    MODEL_SPECS = [
        ("home_win_prob_logistic", "game_outcome_logistic.joblib", "proba"),
        ("home_win_prob_xgb", "game_outcome_xgb.joblib", "proba"),
        ("home_win_prob_rf", "game_outcome_rf.joblib", "proba"),
        ("predicted_margin_ridge", "game_margin_ridge.joblib", "point"),
        ("predicted_margin_xgb", "game_margin_xgb.joblib", "point"),
        ("predicted_margin_rf", "game_margin_rf.joblib", "point"),
        ("predicted_total_ridge", "game_total_ridge.joblib", "point"),
        ("predicted_total_xgb", "game_total_xgb.joblib", "point"),
        ("predicted_total_rf", "game_total_rf.joblib", "point"),
    ]

    out = rows[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
    missing = []
    for out_col, filename, kind in MODEL_SPECS:
        model = _load_if_exists(MODELS_DIR / filename)
        if model is None:
            missing.append(filename)
            continue
        if kind == "proba":
            out[out_col] = model.predict_proba(X)[:, 1].round(3)
        else:
            out[out_col] = model.predict(X).round(1)

    if missing:
        print(f"Warning: missing model artifact(s) in {MODELS_DIR}, skipping: {missing}")
        print("Run scripts/train_game_outcome_model.py and scripts/train_game_margin_model.py to produce them.\n")

    # Point forecasts are useful, but these columns answer the betting-market
    # question directly.  They are intentionally blank when a line/model is
    # unavailable or exactly agrees with the market.
    margin_col = f"predicted_margin_{args.market_model}"
    total_col = f"predicted_total_{args.market_model}"
    if margin_col in out:
        ats = [
            spread_pick(row["home_team"], row["away_team"], row[margin_col], row["spread_line"])
            for row in out.to_dict(orient="records")
        ]
        out["ats_pick"] = [item["pick"] for item in ats]
        out["ats_edge"] = [item["edge"] for item in ats]
    if total_col in out:
        ou = [total_pick(row[total_col], row["total_line"]) for row in out.to_dict(orient="records")]
        out["ou_pick"] = [item["pick"] for item in ou]
        out["ou_edge"] = [item["edge"] for item in ou]

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
