"""
Generate schedule-adjusted weekly projections for every player in board.json.

Trains (or loads) a WeeklyMatchupPredictor that learns — from 2019–2024
historical player-week data — how opponent defensive strength, home/away,
and week of season modulate a player's output relative to their baseline PPG.

For 2026 predictions each player-week is scored with:
  - 2024 opponent position-specific defensive rating (latest complete season)
  - Home/away flag from the 2026 schedule
  - Week number

The 17 per-game multipliers are normalised so the weekly projections sum
exactly to the preseason model's season_proj.

Run:
    python scripts/generate_weekly_projections.py          # train + generate
    python scripts/generate_weekly_projections.py --retrain  # force retrain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.weekly_matchup_predictor import (
    get_predictor,
    get_2024_def_ratings,
)

BOARD_PATH = PROJECT_ROOT / "docs" / "data" / "board.json"
SCHED_PATH = PROJECT_ROOT / "docs" / "data" / "schedule.json"


def build_matchup_list(
    team_matchups: list[dict],
    def_ratings:   dict[str, dict[str, float]],
    position:      str,
) -> list[dict]:
    """
    Convert raw schedule matchup dicts into the format expected by the predictor.
    Attaches opp_def_ppg for each game week.
    """
    result = []
    for m in team_matchups:
        opp     = m.get("opp", "")
        pos_def = def_ratings.get(opp, {}).get(position)
        result.append({
            "week":        int(m["week"]),
            "opp":         opp,
            "home":        bool(m.get("home", False)),
            "opp_def_ppg": pos_def,   # None → model uses league avg fallback
        })
    return result


def main(retrain: bool = False) -> None:
    predictor    = get_predictor(retrain=retrain)
    def_ratings  = get_2024_def_ratings()

    print("Loading 2026 schedule...")
    with open(SCHED_PATH) as f:
        schedule = json.load(f)
    teams_sched = schedule.get("teams", {})

    print("Loading board.json...")
    with open(BOARD_PATH) as f:
        board = json.load(f)
    print(f"  {len(board)} players")

    updated = skipped = 0
    for player in board:
        team = player.get("t", "")
        pos  = player.get("p", "")
        proj = player.get("proj")

        team_data = teams_sched.get(team)
        if not team_data or not proj or proj <= 0:
            player["w"] = None
            skipped += 1
            continue

        matchups = build_matchup_list(
            team_data.get("matchups", []), def_ratings, pos
        )
        player["w"] = predictor.project_weekly(proj, pos, matchups)
        updated += 1

    print(f"  {updated} players updated, {skipped} skipped (no schedule / no proj)")

    with open(BOARD_PATH, "w") as f:
        json.dump(board, f, separators=(",", ":"))
    print(f"Saved → {BOARD_PATH}")

    # Sanity check
    for pos in ["QB", "WR", "RB", "TE"]:
        p = next((x for x in board if x["p"] == pos and x.get("w")), None)
        if p:
            games = [v for v in p["w"] if v is not None]
            bye   = next((i + 1 for i, v in enumerate(p["w"]) if v is None), "–")
            print(f"  {p['p']} {p['n']} ({p['t']}): "
                  f"proj={p['proj']}  sum={sum(games):.1f}  bye=W{bye}")
            print(f"    weeks: {p['w']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining even if saved model exists")
    args = parser.parse_args()
    main(retrain=args.retrain)
