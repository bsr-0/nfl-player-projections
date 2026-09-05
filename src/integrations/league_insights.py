"""The five things a manager actually asks, computed from a league snapshot.

Every number here is either this project's projection or ESPN's, and every row
says which. A player neither side can price is listed as unpriced rather than
given a zero, which would quietly rank him last instead of unknown.

    matchup      this team and its week's opponent, each at its best lineup
    starters     that lineup, and the bench behind it
    tough_calls  starter/bench pairs close enough that the projection is not
                 the thing that should decide them
    waivers      free agents who beat somebody currently rostered here
    trades       where this project and ESPN disagree most about a player

WEEK-SPECIFIC IS A CLAIM THIS CANNOT YET MAKE. In `season_prorated` mode every
week's number is the season total over 17 -- identical week to week, with only
byes, injuries and the opponent label moving. "Start X over Y this week" is
therefore really "X projects higher over a season". The mode travels into the
payload for exactly that reason: when generate_weekly_data.py flips to
`weekly_model`, this same code starts making a genuinely weekly claim, and the
report can say so.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.integrations.league_join import join_players, match_report

# Slots that are not a starting lineup.
BENCH_SLOTS = {"BE", "IR", "ER", ""}

# ESPN statuses that mean he is not playing. QUESTIONABLE and DAY_TO_DAY are
# deliberately absent: those are decisions, and the report should surface them
# rather than quietly bench the player.
UNAVAILABLE = {"OUT", "INJURY_RESERVE", "SUSPENSION", "DOUBTFUL"}

# Two points a week. Inside this, the projection is not separating the two
# players -- measured MAE is 3.0-7.1 points depending on position, so a
# sub-2-point gap is noise wearing a decimal point.
TOUGH_CALL_MARGIN = 2.0

# A waiver add has a real cost (the drop, the FAAB), so require daylight.
WAIVER_MARGIN = 1.0


def starting_slots(settings: dict) -> list:
    """The league's starting lineup, one entry per slot to fill."""
    counts = (settings or {}).get("position_slot_counts") or {}
    slots = []
    for slot, n in counts.items():
        if slot in BENCH_SLOTS or not n:
            continue
        slots.extend([slot] * int(n))
    return slots


def slot_positions(slot: str) -> set:
    """"RB/WR/TE" -> {RB, WR, TE}. "D/ST" is one position, not two."""
    return {slot} if slot in ("D/ST",) else set(slot.split("/"))


def eligible(slot: str, position) -> bool:
    return bool(position) and position in slot_positions(slot)


def price(row: dict):
    """(points, source) for one joined player.

    This project's number when there is one, ESPN's when there is not -- which
    is every kicker and defence, plus the handful of players with no board
    row. Labelled, never blended.
    """
    week = row.get("week_points")
    espn = row.get("espn_projected_avg")
    if week is not None and pd.notna(week):
        return round(float(week), 2), "model"
    if espn is not None and pd.notna(espn):
        return round(float(espn), 2), "espn"
    return None, None


def availability(row: dict):
    """(available, reason). A bye and an injury are different answers."""
    status = str(row.get("injury_status") or "")
    if status in UNAVAILABLE:
        return False, status
    if row.get("on_bye") is True:
        return False, "BYE"
    return True, None


def _players(rows: list) -> list:
    out = []
    for i, row in enumerate(rows):
        points, source = price(row)
        available, reason = availability(row)
        out.append({
            "index": i,
            "name": _clean(row.get("espn_name")),
            "position": _clean(row.get("position")),
            "nfl_team": _clean(row.get("nfl_team")),
            "lineup_slot": _clean(row.get("lineup_slot")),
            "points": points,
            "points_source": source,
            "available": available and points is not None,
            "unavailable_reason": reason,
            "injury_status": _clean(row.get("injury_status")),
            "percent_owned": _num(row.get("percent_owned")),
            "espn_projected_avg": _clean(row.get("espn_projected_avg")),
            "model_season_total": _num(row.get("season_total")),
            "model_season_ppg": _num(row.get("season_ppg")),
            "floor": _num(row.get("floor")),
            "ceiling": _num(row.get("ceiling")),
            "week_ci": [_num(row.get("week_ci_low")), _num(row.get("week_ci_high"))],
            "opponent": _clean(row.get("opponent")),
            "matched": row.get("match_method") is not None,
            "fantasy_team": _clean(row.get("fantasy_team")),
        })
    return out


def _num(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _clean(v):
    """NaN -> None. A player with no opponent has none; he does not have a
    float. The payload is written with allow_nan=False, and bare NaN is not
    JSON a browser will parse.
    """
    try:
        return None if v is None or pd.isna(v) else v
    except (TypeError, ValueError):        # lists, dicts
        return v


def optimal_lineup(players: list, slots: list):
    """Best legal lineup, filling the narrowest slots first.

    Greedy is exact for this slot structure because the slots nest: RB is a
    subset of RB/WR/TE, so taking the best RB for the RB slot can never cost
    the flex a player it would otherwise have started -- the flex still has
    every remaining option the RB slot did not want.
    """
    pool = sorted([p for p in players if p["available"]],
                  key=lambda p: -p["points"])
    used, filled = set(), []
    for slot in sorted(slots, key=lambda s: len(slot_positions(s))):
        pick = next((p for p in pool
                     if p["index"] not in used and eligible(slot, p["position"])),
                    None)
        if pick is not None:
            used.add(pick["index"])
            filled.append(dict(pick, slot=slot))
    bench = [p for p in players if p["index"] not in used]
    return filled, bench


def lineup_total(starters: list) -> float:
    return round(sum(p["points"] for p in starters), 2)


def tough_calls(starters: list, bench: list, margin: float = TOUGH_CALL_MARGIN):
    """Bench players within `margin` of the starter they would replace.

    The pairing is by slot eligibility, so a bench WR is compared against the
    weakest starter he could legally take the place of -- including the flex,
    which is where most of these actually live.
    """
    calls = []
    for b in bench:
        if not b["available"]:
            continue
        # Same pricing source both sides. This project's numbers run at
        # 73-85% of ESPN's (see trade_candidates), so a model-priced starter
        # against an ESPN-priced bench player would compare two scales.
        rivals = [s for s in starters if eligible(s["slot"], b["position"])
                  and s["points_source"] == b["points_source"]]
        if not rivals:
            continue
        weakest = min(rivals, key=lambda s: s["points"])
        gap = round(weakest["points"] - b["points"], 2)
        if gap <= margin:
            calls.append({
                "slot": weakest["slot"],
                "starting": {k: weakest[k] for k in
                             ("name", "position", "nfl_team", "points",
                              "points_source", "opponent", "injury_status")},
                "benched": {k: b[k] for k in
                            ("name", "position", "nfl_team", "points",
                             "points_source", "opponent", "injury_status")},
                "gap": gap,
            })
    return sorted(calls, key=lambda c: c["gap"])


def waiver_targets(free_agents: list, roster: list, slots: list,
                   margin: float = WAIVER_MARGIN, limit: int = 10):
    """Free agents who out-project the worst rostered player at their position.

    That is the honest bar: an add costs a drop, so the comparison is against
    the man who would leave, not against the starter.
    """
    startable = {pos for slot in slots for pos in slot_positions(slot)}
    worst = {}
    for p in roster:
        if p["points"] is None or p["position"] not in startable:
            continue
        current = worst.get(p["position"])
        if current is None or p["points"] < current["points"]:
            worst[p["position"]] = p

    targets = []
    for fa in free_agents:
        if not fa["available"] or fa["position"] not in startable:
            continue
        drop = worst.get(fa["position"])
        if drop is None or drop["points_source"] != fa["points_source"]:
            continue                      # never compare across pricing scales
        if fa["points"] <= drop["points"] + margin:
            continue
        targets.append({
            "name": fa["name"], "position": fa["position"],
            "nfl_team": fa["nfl_team"], "points": fa["points"],
            "points_source": fa["points_source"],
            "percent_owned": fa.get("percent_owned"),
            "over": round(fa["points"] - drop["points"], 2),
            "instead_of": {"name": drop["name"], "points": drop["points"]},
            "injury_status": fa["injury_status"],
        })
    return sorted(targets, key=lambda t: -t["over"])[:limit]


def trade_candidates(league: list, team_name: str, limit: int = 8,
                     min_gap: float = 0.4):
    """Where this project and ESPN disagree about a player's STANDING.

    Not about his points. Measured on this snapshot, this project's per-week
    numbers run at 73-85% of ESPN's by position (QB .835, RB .734, TE .845,
    WR .803) while correlating .57-.87 with them: a season total over 17
    carries the games a player is expected to MISS, where ESPN's average is
    per game he plays. Subtracting one from the other therefore sorts by
    scale -- every star on the roster comes out a "sell" -- so each side is
    standardised within position first and the disagreement is the difference
    in standing.

    Rank within position travels too, because "we have him RB14 and ESPN has
    him RB27" is the sentence a manager can act on.
    """
    priced = [dict(p, model_ppg=p["model_season_ppg"],
                   espn_ppg=_num(p["espn_projected_avg"]))
              for p in league
              if p["points_source"] == "model"
              and p["model_season_ppg"] is not None
              and _num(p["espn_projected_avg"]) is not None]
    if not priced:
        return {"buy_low": [], "sell_high": [], "basis": "none"}

    df = pd.DataFrame(priced)
    for col, out in (("model_ppg", "model"), ("espn_ppg", "espn")):
        grouped = df.groupby("position")[col]
        std = grouped.transform("std").fillna(0.0)
        centred = df[col] - grouped.transform("mean")
        df[f"{out}_z"] = (centred / std.where(std > 0, 1.0)).round(3)
        df[f"{out}_rank"] = grouped.rank(ascending=False,
                                         method="min").astype(int)
    df["gap"] = (df["model_z"] - df["espn_z"]).round(3)
    df["rank_gap"] = df["espn_rank"] - df["model_rank"]

    def shape(row):
        return {"name": row["name"], "position": row["position"],
                "nfl_team": row["nfl_team"], "fantasy_team": row["fantasy_team"],
                "model_ppg": row["model_ppg"], "espn_ppg": row["espn_ppg"],
                "model_rank": int(row["model_rank"]),
                "espn_rank": int(row["espn_rank"]),
                "rank_gap": int(row["rank_gap"]), "gap": float(row["gap"]),
                "injury_status": row["injury_status"]}

    mine = df["fantasy_team"] == team_name
    buy = df[~mine & (df["gap"] >= min_gap)].sort_values("gap", ascending=False)
    sell = df[mine & (df["gap"] <= -min_gap)].sort_values("gap")
    return {
        "basis": "within-position z-score, model minus ESPN",
        "buy_low": [shape(r) for _, r in buy.head(limit).iterrows()],
        "sell_high": [shape(r) for _, r in sell.head(limit).iterrows()],
    }


def find_team(snapshot, team) -> dict:
    """A roster by team id or by a case-insensitive piece of its name."""
    rosters = snapshot.rosters
    if team is None:
        raise ValueError("no team given; set ESPN_TEAM_ID or pass --team")
    text = str(team).strip().lower()
    for t in rosters:
        if str(t.get("team_id")) == text:
            return t
    matches = [t for t in rosters if text in str(t.get("team_name", "")).lower()]
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(f"{t['team_id']}: {t['team_name']}" for t in rosters)
    raise ValueError(
        f"{'no team' if not matches else 'more than one team'} matches "
        f"{team!r}. Teams are -- {names}")


def build_report(snapshot, projections: pd.DataFrame, team, week=None,
                 crosswalk=None) -> dict:
    """The whole payload for one team in one week."""
    week = snapshot.week if week is None else int(week)
    mine = find_team(snapshot, team)
    slots = starting_slots(snapshot.settings)

    league_rows = join_players(snapshot.rostered(), projections, crosswalk)
    league = _players(league_rows.to_dict("records"))
    roster = [p for p in league if p["fantasy_team"] == mine["team_name"]]

    starters, bench = optimal_lineup(roster, slots)

    matchup = snapshot.opponent_for(mine["team_id"], week)
    opponent_id = matchup.get("opponent_id")
    opponent = [p for p in league
                if p["fantasy_team"] == matchup.get("opponent_name")]
    opp_starters, _ = optimal_lineup(opponent, slots)

    fa_rows = join_players(snapshot.free_agents, projections, crosswalk)
    free_agents = _players(fa_rows.to_dict("records"))

    mode = _mode(projections)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": str(snapshot.path),
        "season": snapshot.season,
        "week": week,
        "projection_mode": mode,
        "mode_note": (
            "Season total over 17 with real opponents and byes; identical "
            "week to week, so ranking is season-long, not matchup-specific."
            if mode == "season_prorated" else
            "Weekly model output for this week, with 80% intervals."),
        "team": {"id": mine.get("team_id"), "name": mine.get("team_name"),
                 "wins": mine.get("wins"), "losses": mine.get("losses")},
        "matchup": {
            "week": week,
            "opponent_id": opponent_id,
            "opponent": matchup.get("opponent_name"),
            "projected_total": lineup_total(starters),
            "opponent_projected_total": lineup_total(opp_starters),
            "edge": round(lineup_total(starters)
                          - lineup_total(opp_starters), 2),
        },
        "caveats": _caveats(mode, starters),
        "starters": starters,
        "bench": bench,
        "tough_calls": tough_calls(starters, bench),
        "waivers": waiver_targets(free_agents, roster, slots),
        "trades": trade_candidates(league, mine["team_name"]),
        "coverage": {
            "roster": match_report(league_rows[
                league_rows["fantasy_team"] == mine["team_name"]]),
            "free_agents": match_report(fa_rows),
        },
    }


def _caveats(mode, starters: list) -> list:
    """What the reader has to know before trusting a number on this page."""
    notes = []
    if mode == "season_prorated":
        notes.append(
            "Every week's projection is the season total over 17, so the "
            "ranking is season-long. Only byes, injuries and the opponent "
            "label are week-specific.")
    borrowed = sorted({p["position"] for p in starters
                       if p["points_source"] == "espn"})
    if borrowed:
        notes.append(
            f"{', '.join(borrowed)} are priced by ESPN -- this project does "
            "not model them. Its own numbers run at roughly 80% of ESPN's "
            "scale, so the projected totals mix two scales; they are "
            "consistent between the two teams but not comparable to ESPN's.")
    return notes


def _mode(projections: pd.DataFrame):
    modes = projections["projection_mode"].dropna().unique()
    return modes[0] if len(modes) else None
