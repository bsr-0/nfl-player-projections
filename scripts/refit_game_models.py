#!/usr/bin/env python3
"""Refit the saved game-outcome models on the games completed so far (seconds, no backtest).

scripts/train_game_outcome_model.py and train_game_margin_model.py run a full
walk-forward backtest (and optional tuning) before saving; that is the right
tool when the model design changes and the wrong one for taking in a new week of
results. This refits the nine saved models (win/loss, margin, total x logistic/
ridge, XGBoost, random forest) with the hyperparameters already recorded in the
metadata sidecars, saves them (the joblib files are gitignored) and records
what they were fitted on in data/models/game_models_manifest.json.

    python scripts/refit_game_models.py                # production set: skip if current
    python scripts/refit_game_models.py --force        # refit even if current
    python scripts/refit_game_models.py --check        # exit 1 if stale; fit nothing
    python scripts/refit_game_models.py --walkforward 2026
                                                       # season-pure set: seasons before 2026 only,
                                                       # for src/evaluation/game_backtest.py

scripts/generate_game_predictions_data.py refits the production set itself when
it is stale, so after a week's results are loaded the whole update is:

    python scripts/generate_game_predictions_data.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.game_outcome import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walkforward", type=int, metavar="SEASON", default=None,
                    help="fit the season-pure set that scores SEASON out of sample")
    ap.add_argument("--force", action="store_true", help="refit even if the saved set is current")
    ap.add_argument("--check", action="store_true", help="report whether the saved set is current; fit nothing")
    args = ap.parse_args()

    name = registry.walkforward_name(args.walkforward) if args.walkforward else "production"
    before = args.walkforward
    con = registry.open_db()
    try:
        ok, why = registry.check_set(name, con=con, before_season=before)
        print(f"{name}: {'current' if ok else 'STALE'} -- {why}")
        if args.check:
            return 0 if ok else 1
        if ok and not args.force:
            return 0
        t = time.time()
        entry = registry.fit_set(name, con=con, before_season=before)
        print(f"{name}: fitted {entry['n_games']} games through {entry['trained_through']['season']} "
              f"wk{entry['trained_through']['week']} in {time.time() - t:.0f}s -> {entry['dir']}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
