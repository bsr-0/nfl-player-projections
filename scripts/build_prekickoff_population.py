#!/usr/bin/env python3
"""Freeze each week's pre-kickoff player list: the population every model forecasts.

The list is production's own forecast universe -- the players
NFLPredictor.predict() returns -- grouped by the team production assigns
them. It is what the forward test (docs/PRODUCTION_SELECTION_RULE.md, G10)
hashes before kickoff, and what Plan A renormalizes team shares over instead
of the players who turned out to play.

    live      the real upcoming week, exactly as served.
    history   past weeks via predict(as_of=(season, week)). Production computes
              eligibility "now" (players with games in the last
              ELIGIBLE_SEASONS_LOOKBACK seasons of data, any week), which is
              correct live but sees the future when replayed. History mode
              recomputes it as of the week: games in the lookback seasons
              strictly before (season, week). Rookie teams come from
              get_current_team_map(), today's teams, so history mode maps them
              from the previous week's weekly_rosters instead (falling back to
              the draft team, as production does). Positions come from the
              latest roster snapshot, also today's, so history mode takes each
              listed player's position from the previous week's roster, then
              their last game before the week. Nothing else is changed.

Players whose team has no game that week (byes) are dropped.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import src.data.nfl_data_loader as loader
from config.settings import DB_PATH, ELIGIBLE_SEASONS_LOOKBACK
from src.evaluation.paired_ppr_comparison import file_sha256
from src.predict import NFLPredictor, get_prediction_target_week, get_schedule_map_for_week

COLUMNS = ["player_id", "season", "week", "team", "position"]


def as_of_eligible_ids(season: int, week: int) -> list[str]:
    seasons = list(range(season - ELIGIBLE_SEASONS_LOOKBACK + 1, season + 1))
    with sqlite3.connect(str(DB_PATH)) as con:
        ids = pd.read_sql(
            f"SELECT DISTINCT player_id FROM player_weekly_stats WHERE season IN "
            f"({','.join('?' * len(seasons))}) AND (season < ? OR week < ?)",
            con, params=(*seasons, season, week))
    return ids.player_id.tolist()


def as_of_team_map(season: int, week: int) -> dict[str, str]:
    """Teams from the last completed week's roster; empty before week 2."""
    if week <= 1:
        return {}
    with sqlite3.connect(str(DB_PATH)) as con:
        r = pd.read_sql("SELECT player_id, team FROM weekly_rosters WHERE season = ? AND week = ?",
                        con, params=(season, week - 1))
    return dict(zip(r.player_id, r.team))


def as_of_positions(season: int, week: int) -> dict[str, str]:
    """Position from the last completed week's roster, else the last game before the week."""
    with sqlite3.connect(str(DB_PATH)) as con:
        games = pd.read_sql("SELECT player_id, season, week, position FROM canonical_player_weeks "
                            "WHERE season < ? OR (season = ? AND week < ?)", con,
                            params=(season, season, week))
        roster = (pd.read_sql("SELECT player_id, position FROM weekly_rosters WHERE season = ? AND week = ?",
                              con, params=(season, week - 1)) if week > 1 else pd.DataFrame(columns=["player_id", "position"]))
    last = games.sort_values(["season", "week"]).groupby("player_id").position.last()
    out = last.to_dict()
    skill = roster[roster.position.isin(["QB", "RB", "WR", "TE"])]
    out.update(dict(zip(skill.player_id, skill.position)))
    return out


@contextlib.contextmanager
def as_of(predictor: NFLPredictor, season: int, week: int):
    """Point production's "now"-dependent lookups at (season, week) for one replay."""
    original_filter = loader.filter_to_eligible_players
    original_teams = predictor.db.get_current_team_map
    ids = as_of_eligible_ids(season, week)
    teams = as_of_team_map(season, week)
    loader.filter_to_eligible_players = lambda df, eligible_player_ids=None: original_filter(df, ids)
    predictor.db.get_current_team_map = lambda *args, **kwargs: dict(teams)
    try:
        yield len(ids)
    finally:
        loader.filter_to_eligible_players = original_filter
        predictor.db.get_current_team_map = original_teams


def teams_playing(predictor: NFLPredictor, season: int, week: int) -> set[str]:
    teams = set(get_schedule_map_for_week(predictor.db, season, week))
    if not teams:
        raise ValueError(f"no schedule for {season} week {week}")
    return teams


def population(predictor: NFLPredictor, season: int, week: int, *, replay: bool) -> tuple[pd.DataFrame, dict]:
    ctx = as_of(predictor, season, week) if replay else contextlib.nullcontext(None)
    with ctx as n_eligible:
        out = predictor.predict(n_weeks=1, top_n=1_000_000, as_of=(season, week) if replay else None)
    missing = {"player_id", "team", "position"} - set(out.columns)
    if missing:
        raise ValueError(f"production output lacks {sorted(missing)}")
    pop = out[["player_id", "team", "position"]].assign(season=season, week=week)[COLUMNS]
    n_repositioned = 0
    if replay:
        positions = pop.player_id.map(as_of_positions(season, week))
        n_repositioned = int((positions.notna() & (positions != pop.position)).sum())
        pop["position"] = positions.fillna(pop.position)
    pop = pop[pop.position.isin(["QB", "RB", "WR", "TE"])]
    if pop.player_id.duplicated().any():
        raise ValueError("production returned a player twice")
    playing = teams_playing(predictor, season, week)
    on_bye = ~pop.team.isin(playing)
    meta = {"season": season, "week": week, "replay": replay, "eligible_ids_as_of": n_eligible,
            "production_rows": int(len(out)), "repositioned_as_of": n_repositioned,
            "dropped_bye_or_unknown_team": int(on_bye.sum())}
    pop = pop[~on_bye].sort_values(COLUMNS).reset_index(drop=True)
    meta["players"] = int(len(pop))
    return pop, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("history")
    h.add_argument("--season", type=int, required=True)
    h.add_argument("--weeks", type=int, nargs="+", required=True)
    h.add_argument("--output-dir", type=Path, required=True)
    lv = sub.add_parser("live")
    lv.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = NFLPredictor()
    predictor.initialize()
    if a.mode == "live":
        season, week = get_prediction_target_week()
        targets, replay = [(season, week)], False
    else:
        targets, replay = [(a.season, w) for w in a.weeks], True
    for season, week in targets:
        path = a.output_dir / f"population_{season}_w{week:02d}.csv"
        if path.exists():
            raise ValueError(f"refusing to overwrite a frozen list: {path}")
        pop, meta = population(predictor, season, week, replay=replay)
        pop.to_csv(path, index=False)
        meta |= {"file": path.name, "sha256": file_sha256(path),
                 "written_at": datetime.now(timezone.utc).isoformat(),
                 "script_sha256": file_sha256(Path(__file__))}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
