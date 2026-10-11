#!/usr/bin/env python3
"""Replay the served weekly model as of each completed week and score it (docs/data/weekly_replay_{season}.json).

Runs NFLPredictor.predict(as_of=(season, week)) -- the weekly ensemble plus the
pace blend, i.e. the served path -- for every week whose games are all final, on
the database as it now stands, with the forecast universe, rookie teams and
positions recomputed as of the week (scripts/build_prekickoff_population.py).
About 75 seconds per week. Nothing is retrained and no pinned file is touched.
See src/evaluation/weekly_replay.py for what the replay does and does not prove.

Usage:
    python scripts/replay_weekly_backtest.py --season 2026
    python scripts/replay_weekly_backtest.py --season 2026 --weeks 1 4
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.settings import DB_PATH, MODELS_DIR, PACE_BLEND_KAPPA  # noqa: E402
from src.evaluation.game_backtest import complete_weeks  # noqa: E402
from src.evaluation.published_record import (  # noqa: E402
    POSITIONS, kickoff_times, player_actuals, player_metrics, player_week,
)
from src.evaluation.weekly_replay import ROW_COLUMNS, compare_with_published, score_replay  # noqa: E402

OUT_DIR = ROOT / "docs" / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None)
    args = ap.parse_args()

    from scripts.build_prekickoff_population import population
    from src.predict import NFLPredictor

    con = sqlite3.connect(str(DB_PATH))
    games = kickoff_times(args.season, con)
    weeks = complete_weeks(games)
    if args.weeks:
        weeks = [w for w in weeks if args.weeks[0] <= w <= args.weeks[1]]
    if not weeks:
        print("no complete weeks to replay")
        return 1

    predictor = NFLPredictor()
    if not predictor.initialize():
        print("no trained models; run `python -m src.models.train` first")
        return 1
    meta_path = MODELS_DIR / "model_metadata.json"
    model_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    seasons_trained = model_meta.get("train_seasons") or []
    if seasons_trained and max(seasons_trained) >= args.season:
        print(f"weekly models were trained through {max(seasons_trained)}: not out of sample for {args.season}")
        return 1

    now = pd.Timestamp.now(tz="UTC")
    results, rows_by_week, frames = {}, {}, []
    for wk in weeks:
        pop, pmeta, out = population(predictor, args.season, wk, replay=True)
        rows = out[out["player_id"].isin(pop["player_id"])].copy()
        rows["position"] = rows["player_id"].map(dict(zip(pop["player_id"], pop["position"])))
        playing = set(games.loc[games["week"] == wk, ["home_team", "away_team"]].to_numpy().ravel())
        acts = player_actuals(con, args.season, wk)
        acts = acts[acts["team"].isin(playing) & acts["position"].isin(POSITIONS)]
        scored, _, summary = score_replay(rows, acts)
        published, _ = player_week(con, args.season, wk, games, now)
        summary["vs_published"] = compare_with_published(scored, published)
        summary["population"] = {k: pmeta[k] for k in ("eligible_ids_as_of", "production_rows", "repositioned_as_of",
                                                       "dropped_bye_or_unknown_team", "players")}
        results[str(wk)] = summary
        rows_by_week[str(wk)] = json.loads(scored[[c for c in ROW_COLUMNS if c in scored.columns]]
                                           .round(2).to_json(orient="records"))
        frames.append(scored)
        print(f"  wk{wk}: replay MAE {summary.get('mae')} on {summary.get('n')} players "
              f"(published, same players: {(summary['vs_published'] or {}).get('mae_published')})", flush=True)

    pooled = pd.concat(frames, ignore_index=True)
    track = player_metrics(pooled)
    track["by_position"] = {p: player_metrics(pooled[pooled["position"] == p])
                            for p in POSITIONS if (pooled["position"] == p).any()}
    payload = {
        "season": args.season,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weeks": weeks,
        "basis": ("served weekly model + pace blend (kappa=%s) run as of each week on the current database; weekly models "
                  "fitted on seasons %s-%s only; forecast universe, rookie teams and positions recomputed as of the week"
                  % (PACE_BLEND_KAPPA, min(seasons_trained or [0]), max(seasons_trained or [0]))),
        "weekly_models_trained_at": model_meta.get("training_date"),
        "week_results": results,
        "track_record": track,
        "rows": rows_by_week,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"weekly_replay_{args.season}.json").write_text(json.dumps(payload))
    print(f"\nWrote weeks {weeks[0]}-{weeks[-1]}: pooled replay MAE {track['mae']} on {track['n']} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
