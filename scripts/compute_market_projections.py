#!/usr/bin/env python3
"""Aggregate game-level player prop lines into market-implied season projections.

For each player:
  - Median prop line per (market, week) across bookmakers → sum across regular-season weeks
  - Convert to PPR fantasy points
  - Supplement with actual 2025 TDs (rush_tds, rec_tds) since those prop markets
    weren't scraped — pass_tds ARE captured in the prop data for QBs.

Usage:
    python scripts/compute_market_projections.py            # 2025 season (default)
    python scripts/compute_market_projections.py --season 2024

Output: docs/data/market_projections.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH

# PPR scoring weights per prop market
SCORING = {
    "player_pass_yds":           1 / 25,
    "player_pass_tds":           4.0,
    "player_pass_interceptions": -1.0,
    "player_rush_yds":           1 / 10,
    "player_rush_tds":           6.0,    # rarely in DB; included for completeness
    "player_receptions":         1.0,    # PPR
    "player_reception_yds":      1 / 10,
    "player_reception_tds":      6.0,    # rarely in DB; supplemented from actuals
    "player_kicking_points":     1.0,
}

# Human-readable stat output fields
STAT_FIELDS = {
    "player_pass_yds":           "pass_yds",
    "player_pass_tds":           "pass_tds",
    "player_pass_interceptions": "pass_ints",
    "player_rush_yds":           "rush_yds",
    "player_receptions":         "receptions",
    "player_reception_yds":      "rec_yds",
    "player_kicking_points":     "kicking_pts",
}

REGULAR_SEASON_MAX_WEEK = 18


def _load_props(season: int) -> dict[tuple[str, str, int], list[float]]:
    """Return {(player_name, market, week): [point values]} for Over lines only."""
    markets_in = ",".join(f"'{m}'" for m in SCORING)
    query = f"""
        SELECT player_name, market, week, point
        FROM player_props_odds
        WHERE season = ?
          AND description = 'Over'
          AND week BETWEEN 1 AND {REGULAR_SEASON_MAX_WEEK}
          AND point IS NOT NULL
          AND market IN ({markets_in})
    """
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(query, (season,)).fetchall()

    data: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for player_name, market, week, point in rows:
        data[(player_name, market, week)].append(float(point))
    return data


def _load_actual_tds(season: int) -> dict[str, dict]:
    """Return {full_player_name: {rush_tds, rec_tds, pass_tds, position}}.

    Uses weekly_rosters_v2 (full names) joined to player_weekly_stats (TD counts).
    Pass TDs are included for reference but NOT added to market_fp since prop data
    already captures them for QBs.
    """
    query = """
        SELECT r.player_name, p.position,
               SUM(COALESCE(w.rushing_tds, 0)),
               SUM(COALESCE(w.receiving_tds, 0)),
               SUM(COALESCE(w.passing_tds, 0))
        FROM player_weekly_stats w
        JOIN weekly_rosters_v2 r
            ON w.player_id = r.player_id
           AND w.season = r.season
           AND w.week = r.week
        JOIN players p ON w.player_id = p.player_id
        WHERE w.season = ?
          AND w.week BETWEEN 1 AND 18
          AND p.position IN ('QB', 'RB', 'WR', 'TE', 'K')
        GROUP BY w.player_id, r.player_name
    """
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(query, (season,)).fetchall()

    result: dict[str, dict] = {}
    for full_name, position, rush_tds, rec_tds, pass_tds in rows:
        result[full_name] = {
            "position": position,
            "rush_tds": int(rush_tds),
            "rec_tds": int(rec_tds),
            "pass_tds": int(pass_tds),
        }
    return result


def _td_fp_supplement(td_record: dict) -> float:
    """FP from TDs not captured in prop lines.

    Pass TDs are already in prop data (player_pass_tds market).
    Rush TDs and reception TDs were not scraped — add them from actuals.
    """
    pos = td_record.get("position", "")
    rush_tds = td_record.get("rush_tds", 0)
    rec_tds = td_record.get("rec_tds", 0)

    if pos == "QB":
        # QB pass TDs captured via props; only add rush TDs
        return rush_tds * 6.0
    elif pos in ("RB", "WR", "TE"):
        return (rush_tds + rec_tds) * 6.0
    return 0.0


def compute(season: int) -> dict[str, dict]:
    """Aggregate prop lines + TD supplement into per-player season projections."""
    raw = _load_props(season)
    actual_tds = _load_actual_tds(season)

    # Median across bookmakers per (player, market, week)
    weekly_medians: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (player, market, week), points in raw.items():
        weekly_medians[(player, market)][week] = median(points)

    # Sum medians across weeks per (player, market)
    season_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    games_seen: dict[str, set[int]] = defaultdict(set)

    for (player, market), week_map in weekly_medians.items():
        for week, val in week_map.items():
            season_totals[player][market] += val
            games_seen[player].add(week)

    players: dict[str, dict] = {}
    for player, market_totals in season_totals.items():
        prop_fp = sum(
            market_totals.get(m, 0.0) * weight for m, weight in SCORING.items()
        )

        td_record = actual_tds.get(player, {})
        td_fp = _td_fp_supplement(td_record)

        record: dict = {
            "market_fp": round(prop_fp + td_fp, 1),
            "prop_fp": round(prop_fp, 1),
            "td_fp": round(td_fp, 1),
            "td_rush": td_record.get("rush_tds", 0),
            "td_rec": td_record.get("rec_tds", 0),
            "games": len(games_seen[player]),
            "position": td_record.get("position", ""),
        }
        for market_key, field in STAT_FIELDS.items():
            val = market_totals.get(market_key, 0.0)
            record[field] = round(val, 1) if val else 0

        players[player] = record

    return players


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "data" / "market_projections.json",
    )
    args = parser.parse_args()

    print(f"Computing market-implied projections for {args.season} season...")
    players = compute(args.season)
    print(f"  {len(players)} players with prop lines")

    sorted_players = dict(
        sorted(players.items(), key=lambda x: x[1]["market_fp"], reverse=True)
    )

    print("  Top 10 by market-implied FP:")
    for name, rec in list(sorted_players.items())[:10]:
        pos = rec.get("position") or "?"
        print(
            f"    [{pos}] {name}: {rec['market_fp']} fp "
            f"(props={rec['prop_fp']}, tds=+{rec['td_fp']}, {rec['games']}g)"
        )

    out = {
        "season": args.season,
        "note": (
            "market_fp = prop-implied FP (median book line, regular season weeks 1-18) "
            "+ actual TD FP (rush/rec TDs not in prop data). "
            "QB pass_tds are from prop lines."
        ),
        "players": sorted_players,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
