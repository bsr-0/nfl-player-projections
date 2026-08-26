#!/usr/bin/env python
"""Track B: does the real-week residual track current-season ROLE, or the player?

The availability branch is closed -- weighting synthetic weeks is a secondary
adjustment, not the mechanism (GAPS.md). What survives every correction is a
gradient in the residual on GENUINE observations: QBs who end up with fewer
real weeks are over-predicted on the weeks they do play.

Two competing readings, which this separates:

  A  player-level heterogeneity -- "fragile QBs are systematically
     over-predicted", a stable property of the player.
  B  state transition -- "QBs become over-predicted around the time they
     lose their role", a current-season state the model cannot see.

A implies the residual is flat within a player and differs between players.
B implies it climbs as a player approaches an absence, within the same
player.

No refit: real-week predictions already exist in the Phase 4 row-level dump
under FINAL_CONFIG, so this is a join plus arithmetic.

Absences are crudely typed from weekly_rosters status in the missed week --
RES (IR/PUP/NFI) reads as health-driven, ACT (dressed, didn't play) as
role-driven, INA as ambiguous. Not perfect labels, but enough to say whether
the gradient is predominantly health or role.

Usage:
    python scripts/run_track_b_exposure.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.settings import regular_season_max_week
from src.models.single_week_ppr.season_projection import REGULAR_SEASON_MAX_WEEK

# weekly_rosters status in a MISSED week -> why he wasn't out there.
ABSENCE_TYPE = {
    "RES": "health (IR/PUP/NFI)",
    "ACT": "role (dressed, did not play)",
    "INA": "ambiguous (declared inactive)",
    "DEV": "role (practice squad)",
    "CUT": "role (waived)",
    "RET": "role (retired)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path,
                    default=Path("data/experiments/phase4_v33.csv"))
    ap.add_argument("--position", default="QB")
    ap.add_argument("--output", type=Path,
                    default=Path("data/experiments/track_b_exposure.csv"))
    args = ap.parse_args()

    from config.settings import DB_PATH
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    conn = sqlite3.connect(str(DB_PATH))
    cfg = FINAL_CONFIG[args.position]

    p = pd.read_csv(args.preds)
    p = p[(p.position == args.position)
          & (p.model == cfg["architecture"])
          & (p.training_window == cfg["window"])].copy()
    p = p.rename(columns={"player": "player_id", "actual_ppr": "actual"})
    p["week"] = p.week.astype(int)

    ledger = {"rows_at_final_config": len(p)}
    # Playoff rows must go before anything counts "weeks in a season" --
    # season_projection.py guards this for the same reason. Leaving them in
    # gave 29 of 239 player-seasons an availability above 1.0 (up to 1.24),
    # which is the regressor the whole Track B analysis turns on.
    # Era-aware: 17 through 2020, 18 from 2021. The flat 18 this replaces
    # only fixed 2021+ -- pre-2021 it left the wild-card week in, so the
    # denominator was 17 against a true 16-game schedule and a player who
    # played every regular-season game but no playoff game scored 16/17.
    # That lands just under the availability_gt_1 assert below, which is
    # why the guard never fired again after the first fix.
    p = p[p.week <= p.season.map(regular_season_max_week)]
    ledger["excluded_playoff_weeks"] = ledger["rows_at_final_config"] - len(p)
    ledger["rows_regular_season"] = len(p)
    p["residual"] = p.prediction - p.actual
    seasons = sorted(p.season.unique().tolist())
    lo, hi = min(seasons), max(seasons)

    stats = pd.read_sql(
        f"""SELECT player_id, season, week, team, passing_attempts, rushing_attempts
            FROM player_weekly_stats WHERE season BETWEEN {lo} AND {hi}""", conn)
    stats["week"] = stats.week.astype(int)
    p = p.merge(stats, on=["player_id", "season", "week"], how="left")

    rosters = pd.read_sql(
        f"""SELECT player_id, season, week, status FROM weekly_rosters
            WHERE game_type='REG' AND season BETWEEN {lo} AND {hi}""", conn)
    rosters["week"] = rosters.week.astype(int)
    rosters = rosters.drop_duplicates(["player_id", "season", "week"])
    status_by = {(r.player_id, r.season, r.week): r.status
                 for r in rosters.itertuples()}

    sched = pd.read_sql(
        f"""SELECT season, week, home_team AS team FROM schedule
            WHERE season BETWEEN {lo} AND {hi}
            UNION SELECT season, week, away_team FROM schedule
            WHERE season BETWEEN {lo} AND {hi}""", conn)
    sched["week"] = pd.to_numeric(sched.week, errors="coerce")
    sched = sched.dropna(subset=["week"])
    sched["week"] = sched.week.astype(int)
    sched = sched[sched.week <= sched.season.map(regular_season_max_week)]
    team_weeks = sched.groupby(["team", "season"])["week"].apply(set).to_dict()

    depth = pd.read_sql(
        f"""SELECT gsis_id AS player_id, CAST(season AS INT) season,
                   CAST(week AS INT) week, MIN(CAST(depth_team AS INT)) depth_team
            FROM depth_charts WHERE position=? AND season BETWEEN {lo} AND {hi}
              AND gsis_id IS NOT NULL GROUP BY gsis_id, season, week""",
        conn, params=[args.position])
    p = p.merge(depth, on=["player_id", "season", "week"], how="left")

    prior = pd.read_sql(
        """SELECT player_id, season+1 AS season, COUNT(*) prior_games,
                  AVG(fantasy_points) prior_ppg
           FROM player_weekly_stats GROUP BY player_id, season""", conn)
    p = p.merge(prior, on=["player_id", "season"], how="left")

    rows = []
    for (pid, season), g in p.groupby(["player_id", "season"]):
        g = g.sort_values("week")
        played = sorted(g.week.tolist())
        team = g.team.dropna().iloc[0] if g.team.notna().any() else None
        possible = sorted(team_weeks.get((team, season), set())) or list(
            range(1, REGULAR_SEASON_MAX_WEEK + 1))
        missed = [w for w in possible if w not in played]

        # Type each missed week, then summarise the season.
        types = [ABSENCE_TYPE.get(status_by.get((pid, season, w)), "not on roster")
                 for w in missed]
        health = sum(1 for t in types if t.startswith("health"))
        role = sum(1 for t in types if t.startswith("role"))

        n_played = len(played)
        for i, r in enumerate(g.itertuples()):
            later = [w for w in missed if w > r.week]
            next_absence = min(later) if later else None
            rows.append({
                "player_id": pid, "season": season, "week": r.week,
                "prediction": r.prediction, "actual": r.actual,
                "residual": r.residual,
                "games_played_so_far": i + 1,
                "games_remaining": n_played - i - 1,
                "cum_pass_att": g[g.week <= r.week].passing_attempts.sum(),
                "roll3_pass_att": g[(g.week <= r.week)].passing_attempts.tail(3).mean(),
                "depth_team": r.depth_team,
                "prior_ppg": r.prior_ppg, "prior_games": r.prior_games,
                "season_real_weeks": n_played,
                "season_possible_weeks": len(possible),
                "eventual_availability": n_played / len(possible) if possible else np.nan,
                "weeks_until_absence": (next_absence - r.week) if next_absence else np.nan,
                "is_last_before_absence": int(next_absence == r.week + 1) if next_absence else 0,
                "season_absences": len(missed),
                "absences_health": health, "absences_role": role,
                "absence_profile": ("none" if not missed else
                                    "health" if health > role else
                                    "role" if role > health else "mixed"),
            })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Audit trail: this analysis went through enough revisions that the
    # exclusion ledger matters more than another hypothesis test.
    ledger.update({
        "player_weeks_out": len(out),
        "players": int(out.player_id.nunique()),
        "player_seasons": int(out.groupby(["player_id", "season"]).ngroups),
        "seasons": seasons,
        "architecture": cfg["architecture"], "window": cfg["window"],
        "availability_gt_1": int(
            (out.groupby(["player_id", "season"])["eventual_availability"].first() > 1).sum()),
        "max_week": int(out.week.max()),
        "spike_weeks": int(out.is_last_before_absence.sum()),
        "weeks_with_prior_season": int(out.prior_ppg.notna().sum()),
    })
    print(f"Wrote {len(out)} real {args.position}-weeks -> {args.output}\n")
    print("EXCLUSION LEDGER")
    for k, v in ledger.items():
        print(f"  {k:26s} {v}")
    assert ledger["availability_gt_1"] == 0, "availability > 1 — week filter is wrong"
    assert ledger["max_week"] <= REGULAR_SEASON_MAX_WEEK
    conn.close()


if __name__ == "__main__":
    main()
