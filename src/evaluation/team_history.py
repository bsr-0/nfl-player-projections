"""Your fantasy team, week by week, from ESPN box scores: the lineup as set,
the lineup the model's pre-kickoff projections would have picked from the
same roster, ESPN's pick and the hindsight best, all scored on actuals."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

def _week_lineups(league, rows, week, team_id, slots, xwalk, optimal_lineup) -> dict:
    """One week of your team: the lineup as set, the lineup the model's
    pre-kickoff projections would have picked from the same lineup and bench,
    ESPN's projections' pick, and the hindsight best -- all scored on actuals."""
    box = next(b for b in league.box_scores(week) if team_id in (b.home_team.team_id, b.away_team.team_id))
    home = box.home_team.team_id == team_id
    team = box.home_team if home else box.away_team
    lineup = box.home_lineup if home else box.away_lineup
    wk = rows[rows["week"] == week].dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")
    proj = wk["predicted_points"]
    stat_line = set(wk.index[wk["actual_points"].notna()])
    players, pending = [], 0
    for i, p in enumerate(lineup):
        played = (p.game_played or 0) >= 100
        pending += (not played) and not p.on_bye_week and p.slot_position not in ("BE", "IR")
        gsis = xwalk.get(str(p.playerId))
        model = proj.get(gsis)
        players.append({"index": i, "name": p.name, "position": p.position, "slot": p.slot_position,
                        "espn": None if p.on_bye_week else p.projected_points,
                        "model": None if p.on_bye_week or model is None else float(model),
                        "actual": p.points if played else None, "bye": bool(p.on_bye_week),
                        "available": not p.on_bye_week and p.slot_position != "IR",
                        # Same rule as the test set: a stat line in player_weekly_stats.
                        "scored": gsis in stat_line})

    def best(key):
        # No model number (kickers, defences, unprojected players): ESPN's.
        pool = [dict(x, points=(x[key] if x[key] is not None else (x["espn"] or 0.0))) for x in players]
        picked, _ = optimal_lineup(pool, slots)
        return sum((x["actual"] or 0.0) for x in picked), {x["index"] for x in picked}

    started = [x for x in players if x["slot"] not in ("BE", "IR")]
    model_total, model_pick = best("model")
    return {"week": week, "team": team.team_name, "pending": pending,
            "actual": sum((x["actual"] or 0.0) for x in started),
            "model_lineup": model_total, "espn_lineup": best("espn")[0], "hindsight": best("actual")[0],
            "changes": len(model_pick - {x["index"] for x in started}),
            "fallback": sum(1 for x in players if x["model"] is None and x["available"]),
            "opponent": box.away_score if home else box.home_score,
            "players": [x for x in players if x["scored"] and None not in (x["model"], x["espn"], x["actual"])]}


def fantasy_season(rows: pd.DataFrame, weeks: list, team_id: int) -> dict | None:
    """Your team week by week, and the season totals over completed weeks."""
    try:
        from espn_api.football import League
        from src.integrations.league_insights import optimal_lineup, starting_slots
        from src.integrations.league_join import espn_id_to_gsis, load_snapshot
        league = League(league_id=int(os.environ["ESPN_LEAGUE_ID"]), year=int(os.environ.get("ESPN_YEAR", 2026)))
        slots = starting_slots(load_snapshot().settings)
        xwalk = espn_id_to_gsis()
        by_week = [_week_lineups(league, rows, w, team_id, slots, xwalk, optimal_lineup) for w in weeks]
    except Exception as e:                       # noqa: BLE001
        print(f"  fantasy panel skipped ({e})")
        return None
    done = [w for w in by_week if not w["pending"]]
    result = lambda pts, opp: "W" if pts > opp else "L" if pts < opp else "T"
    for w in by_week:
        w["result"], w["model_result"] = result(w["actual"], w["opponent"]), result(w["model_lineup"], w["opponent"])
    record = lambda key: "\u2013".join(str(sum(w[key] == r for w in done)) for r in "WL") \
        + (f"\u2013{sum(w[key] == 'T' for w in done)}" if any(w[key] == "T" for w in done) else "")
    total = lambda key: sum(w[key] for w in done)
    pl = [x for w in done for x in w["players"]]
    err = lambda key: [x[key] - x["actual"] for x in pl]
    proj = {"n": len(pl), "weeks": len(done)}
    if pl:
        for key in ("model", "espn"):
            e = np.array(err(key))
            proj[f"{key}_mae"], proj[f"{key}_bias"] = float(np.abs(e).mean()), float(e.mean())
        proj["model_closer"] = sum(abs(x["model"] - x["actual"]) < abs(x["espn"] - x["actual"]) for x in pl)
    return {"team": by_week[0]["team"], "weeks": by_week, "done": len(done), "projections": proj,
            "actual": total("actual"), "model_lineup": total("model_lineup"),
            "espn_lineup": total("espn_lineup"), "hindsight": total("hindsight"),
            "record": record("result"), "model_record": record("model_result"),
            "model_better": sum(w["model_lineup"] > w["actual"] + 1e-9 for w in done),
            "model_worse": sum(w["model_lineup"] < w["actual"] - 1e-9 for w in done),
            "changes": sum(w["changes"] for w in done),
            "flips": [w for w in done if w["result"] != w["model_result"]]}



OUT_STATUSES = {"OUT", "INJURY_RESERVE", "SUSPENSION"}


def week_lineup(team_id: int, week: int, proj: pd.DataFrame, league=None) -> dict | None:
    """Our lineup for an upcoming week, from ESPN, beside the model's
    published projection and 80% range for each player. `proj` must already
    be whitelisted (review_brief.week_projections). Also the lineup the
    model's numbers would set from the same roster (ESPN's number stands in
    where the model has none: kickers, defences)."""
    try:
        from espn_api.football import League
        from src.integrations.league_insights import optimal_lineup, starting_slots
        from src.integrations.league_join import espn_id_to_gsis, load_snapshot
        league = league or League(league_id=int(os.environ["ESPN_LEAGUE_ID"]),
                                  year=int(os.environ.get("ESPN_YEAR", 2026)))
        slots = starting_slots(load_snapshot().settings)
        xwalk = espn_id_to_gsis()
        box = next(b for b in league.box_scores(week) if team_id in (b.home_team.team_id, b.away_team.team_id))
    except Exception as e:                       # noqa: BLE001
        print(f"  week-{week} lineup skipped ({e})")
        return None
    home = box.home_team.team_id == team_id
    team, opp = (box.home_team, box.away_team) if home else (box.away_team, box.home_team)
    by_id = proj.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")
    players = []
    for i, p in enumerate(box.home_lineup if home else box.away_lineup):
        row = by_id.loc[xwalk[str(p.playerId)]] if xwalk.get(str(p.playerId)) in by_id.index else None
        status = getattr(p, "injuryStatus", None)
        bye = bool(p.on_bye_week)
        players.append({
            "index": i, "name": p.name, "position": p.position, "slot": p.slot_position,
            "nfl_team": getattr(p, "proTeam", None),
            "opponent": (row["opponent"] if row is not None and "opponent" in row else None)
            or getattr(p, "pro_opponent", None),
            "espn": None if bye else p.projected_points,
            "model": None if bye or row is None else float(row["predicted_points"]),
            "lo": None if row is None or pd.isna(row.get("prediction_ci80_lower")) else float(row["prediction_ci80_lower"]),
            "hi": None if row is None or pd.isna(row.get("prediction_ci80_upper")) else float(row["prediction_ci80_upper"]),
            "status": status, "bye": bye,
            "available": not bye and p.slot_position != "IR" and status not in OUT_STATUSES})
    starters = [x for x in players if x["slot"] not in ("BE", "IR")]
    pool = [dict(x, points=x["model"] if x["model"] is not None else (x["espn"] or 0.0)) for x in players]
    picked, _ = optimal_lineup(pool, slots)
    pick_idx = {x["index"] for x in picked}
    set_idx = {x["index"] for x in starters}
    model_or_espn = lambda x: x["model"] if x["model"] is not None else (x["espn"] or 0.0)
    return {"week": week, "team": team.team_name, "opponent": opp.team_name, "players": players,
            "start": [x for x in players if x["index"] in pick_idx - set_idx],
            "sit": [x for x in players if x["index"] in set_idx - pick_idx],
            "model_total": sum(model_or_espn(x) for x in starters if x["available"]),
            "espn_total": sum((x["espn"] or 0.0) for x in starters if x["available"]),
            "fallback": sum(1 for x in starters if x["model"] is None)}
