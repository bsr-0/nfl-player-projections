#!/usr/bin/env python3
"""How well does the number the board publishes for week 1 predict week 1?

Before kickoff the site publishes `season total / 17` per week -- the step 8
season model divided by a season (`generate_weekly_data.py`, mode
`season_prorated`). Every existing measurement of that model is a SEASON
number: rookie season-total MAE 39.81, cold-start games-played MAE 3.3-4.8.
Nothing measures the thing a week-1 lineup decision actually reads.

This runs the production path backwards over past seasons and scores it
against real week-1 PPR, split by whether the player had any NFL history at
all, because that split is the whole question -- a cold-start player's pace is
almost entirely the shrinkage target that PR #96 changed.

Arms:
    team_share     the alternative: what the player's TEAM produces at his
                   position per week, times the share a player like him takes.
                   For a cold-start player "like him" means his draft round --
                   the only role signal that exists before he plays -- and the
                   share is fitted on past rookies, ratio-of-sums, shrunk
                   toward the position mean by cell size. For a veteran it is
                   his OWN share of his old team's position group, carried to
                   his new team's level. This is the "team tendencies and how
                   a team uses its players" hypothesis: a back landing in a
                   room that produces 22 PPR points a week is in a different
                   situation from one landing in a room that produces 12,
                   whatever his own history says.
    hurdle         the two-stage version: P(role) x E[points | role] plus the
                   complement, fit on the same features. The snap-share
                   diagnostic showed step8_pace wins the composite by
                   answering "does he play at all" and LOSES to team context
                   once a role is granted, so this gives each half the
                   features that win it. Role means a realised snap share
                   >= 0.25 -- a training label, never a feature.
    matchup        the same hypothesis carried one step further: a Ridge on
                   where he landed AND who he opens against -- draft capital,
                   his new team's prior-season production at his position and
                   its pace, and what the week-1 opponent gave up to that
                   position last year. team_share has the first half; this
                   adds the matchup. Cold start only, walk-forward, see
                   src/models/week1_matchup.py.
    step8_pace     the published number: step 8 season total / 17
    step8_per_game the same total over the model's OWN expected games, which
                   is the production term `E[PPR per game | played]` on its
                   own. `/ 17` averages in the games a player is expected to
                   MISS; this population is players who played, so the
                   published number should be biased low by exactly that
                   discount and this arm should remove it.
    position_mean  the position's mean week-1 points, fitted leave-one-season-
                   out. The floor any model has to clear to be worth running.
    prior_ppg      the player's own PPG last season. Undefined for cold-start
                   rows by construction, which is exactly why they are hard;
                   they fall back to position_mean.

Leakage: step 8 is refit per target season on pairs strictly before it, with
availability fit `before_season=S`. position_mean excludes the target season.
prior_ppg reads season S-1, which is known before week 1 is played. team_share
reads team levels from S-1 and fits its share table on seasons < S; the one
target-season fact it uses is which team the player is on in week 1, which is
public roster information before kickoff and is not the label.

POPULATION CAVEAT, stated because it flatters every arm: only players who
recorded a week-1 row are scored. A rookie drafted and inactive in week 1 is
not here, so this measures accuracy given that he played, not the harder
question of whether he would.

Usage:
    python scripts/run_week1_coldstart_experiment.py
    python scripts/run_week1_coldstart_experiment.py --seasons 2023 2025
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, DB_PATH

# Step 8's own training window, matching generate_draft_data.py.
TRAIN_FROM = 2019

# Snap-share cuts for the diagnostic slice. 0.0 is every row that played at
# all; 0.25 is roughly a rotational role; 0.50 is a starter.
SNAP_SHARE_THRESHOLDS = (0.0, 0.10, 0.25, 0.50)
OUT_DIR = DATA_DIR / "backtest_results"


def step8_pace(season: int) -> pd.DataFrame:
    """Refit step 8 as of `season` and return its published per-week number."""
    from src.utils.database import DatabaseManager
    from src.models.preseason_features import build_multiyear_season_pairs
    from src.models.single_week_ppr.season_availability import load_player_seasons
    from src.models.season_step8 import Step8SeasonModel, possible_games_for_players

    db = DatabaseManager()
    panel = load_player_seasons()
    pairs = build_multiyear_season_pairs(db, list(range(TRAIN_FROM, season)),
                                         inference_season=season)
    train = pairs[pairs["target_season"] < season]
    infer = pairs[pairs["target_season"] == season].copy()
    if train.empty or infer.empty:
        return pd.DataFrame()

    model = Step8SeasonModel().fit(train, panel, before_season=season)
    possible = np.asarray(possible_games_for_players(infer, season), dtype=float)
    preds = np.asarray(model.predict(infer, possible_games=possible))

    # The model's own games estimate, the other half of games x rate. Dividing
    # the total by it recovers the per-game production term exactly.
    games = np.asarray(model.availability.predict_rate(infer),
                       dtype=float) * possible
    cold = infer.get("is_cold_start")
    return pd.DataFrame({
        "player_id": infer["player_id"].values,
        "position": infer["position"].values,
        "cold_start": (cold.fillna(0).astype(int).values if cold is not None
                       else np.zeros(len(infer), dtype=int)),
        # The published week number is the season total over a season.
        "step8_pace": preds / 17.0,
        "step8_per_game": preds / np.where(games > 0, games, np.nan),
        "train_seasons": season - TRAIN_FROM,
    })


def week1_actuals(season: int) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    try:
        return pd.read_sql(
            "SELECT player_id, team, snap_share, fantasy_points AS actual "
            "FROM player_weekly_stats "
            "WHERE season = ? AND week = 1 AND fantasy_points IS NOT NULL",
            con, params=[int(season)])
    finally:
        con.close()


def _weekly(columns="player_id, season, week, team, fantasy_points"):
    con = sqlite3.connect(str(DB_PATH))
    try:
        w = pd.read_sql(
            f"SELECT {columns} FROM player_weekly_stats "
            "WHERE fantasy_points IS NOT NULL", con)
        pos = pd.read_sql(
            "SELECT player_id, position FROM players WHERE position IS NOT NULL",
            con)
    finally:
        con.close()
    return w.merge(pos.drop_duplicates("player_id"), on="player_id", how="left")


def team_position_levels(weekly: pd.DataFrame) -> pd.DataFrame:
    """PPR points a team's position room produces per week, by season.

    The room, not the player: this is the situation a rookie is drafted into,
    and it is knowable before he takes a snap.
    """
    per_week = (weekly[weekly["position"].isin(("QB", "RB", "WR", "TE"))]
                .groupby(["season", "team", "position", "week"])["fantasy_points"]
                .sum().reset_index())
    return (per_week.groupby(["season", "team", "position"])["fantasy_points"]
            .mean().reset_index().rename(columns={"fantasy_points": "level"}))


def fit_rookie_shares(weekly, levels, debut, draft_round, before: int,
                      bucket_k: float = 25.0) -> dict:
    """Share of his room's weekly output a rookie takes in week 1.

    Ratio of sums, not mean of ratios: a rookie landing in a room that
    produced 4 points a week would otherwise contribute a share of 3.0 and
    swamp the estimate. Shrunk toward the position mean by cell size, the same
    device and the same K as the availability prior's draft buckets.
    """
    wk1 = weekly[(weekly.week == 1) & (weekly.season < before)].copy()
    wk1["debut"] = wk1.player_id.map(debut)
    rookies = wk1[wk1.debut == wk1.season].copy()
    if rookies.empty:
        return {}
    prior = levels.assign(season=levels.season + 1)
    rookies = rookies.merge(prior, on=["season", "team", "position"], how="left")
    rookies = rookies[rookies.level > 0]
    rookies["round"] = rookies.player_id.map(draft_round).fillna(8).astype(int)
    rookies.loc[rookies["round"] > 7, "round"] = 8

    shares = {}
    for pos, g in rookies.groupby("position"):
        pos_share = g.fantasy_points.sum() / g.level.sum()
        shares[(pos, None)] = pos_share
        for rnd, cell in g.groupby("round"):
            w = len(cell) / (len(cell) + bucket_k)
            cell_share = cell.fantasy_points.sum() / cell.level.sum()
            shares[(pos, int(rnd))] = w * cell_share + (1 - w) * pos_share
    return shares


def veteran_shares(weekly, levels, season: int) -> pd.DataFrame:
    """Each player's share of his OWN room last season, to carry forward."""
    prev = weekly[weekly.season == season - 1]
    if prev.empty:
        return pd.DataFrame(columns=["player_id", "vet_share"])
    per_player = (prev.groupby(["player_id", "team", "position"])
                  ["fantasy_points"].mean().reset_index())
    per_player = per_player.merge(
        levels[levels.season == season - 1].drop(columns="season"),
        on=["team", "position"], how="left")
    per_player = per_player[per_player.level > 0]
    per_player["vet_share"] = per_player.fantasy_points / per_player.level
    return (per_player.sort_values("fantasy_points", ascending=False)
            .drop_duplicates("player_id")[["player_id", "vet_share"]])


def prior_ppg(season: int) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    try:
        return pd.read_sql(
            "SELECT player_id, AVG(fantasy_points) AS prior_ppg "
            "FROM player_weekly_stats WHERE season = ? "
            "GROUP BY player_id", con, params=[int(season) - 1])
    finally:
        con.close()


def draft_rounds() -> dict:
    con = sqlite3.connect(str(DB_PATH))
    try:
        d = pd.read_sql("SELECT player_id, draft_round FROM draft_picks_v2 "
                        "WHERE player_id IS NOT NULL AND draft_round IS NOT NULL",
                        con)
    finally:
        con.close()
    return dict(zip(d.player_id, d.draft_round))


def add_team_share(rows: pd.DataFrame, season: int, weekly, levels, debut,
                   rounds) -> pd.DataFrame:
    """level(new team, position, S-1) x the share a player like him takes."""
    shares = fit_rookie_shares(weekly, levels, debut, rounds, before=season)
    rows = rows.merge(levels[levels.season == season - 1].drop(columns="season"),
                      on=["team", "position"], how="left")
    rows = rows.merge(veteran_shares(weekly, levels, season), on="player_id",
                      how="left")
    rnd = rows.player_id.map(rounds).fillna(8).astype(int).clip(upper=8)

    share = []
    for cold, pos, r, vet in zip(rows.cold_start, rows.position, rnd,
                                 rows.vet_share):
        if cold:
            share.append(shares.get((pos, int(r)), shares.get((pos, None))))
        else:
            share.append(vet if pd.notna(vet) else np.nan)
    rows["team_share"] = rows["level"] * pd.Series(share, index=rows.index)
    return rows.drop(columns=["vet_share"])


def _score(g: pd.DataFrame, arm: str) -> dict:
    err = g[arm] - g["actual"]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((g["actual"] - g["actual"].mean()) ** 2).sum())
    return {"n": int(len(g)), "mae": round(float(err.abs().mean()), 2),
            "bias": round(float(err.mean()), 2),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
            "r2": round(1 - ss_res / ss_tot, 3) if ss_tot else None}


def evaluate(rows: pd.DataFrame, arms) -> dict:
    out = {}
    for label, g in (("all", rows), ("cold_start", rows[rows.cold_start == 1]),
                     ("veteran", rows[rows.cold_start == 0])):
        if g.empty:
            continue
        out[label] = {arm: _score(g, arm) for arm in arms}
        out[label]["mean_actual"] = round(float(g["actual"].mean()), 2)
    return out


def matchup_arm(seasons, pool_from: int) -> pd.DataFrame:
    """Walk-forward: each season predicted by a model fit only on cold-start
    week-1 rows from earlier seasons, the same discipline step 8 gets."""
    from src.models.week1_matchup import (
        Week1HurdleModel, Week1MatchupModel, build_week1_rows)

    pool = build_week1_rows(range(pool_from, max(seasons) + 1))
    out = []
    for season in sorted(seasons):
        train, test = pool[pool.season < season], pool[pool.season == season]
        if len(train) < 50 or test.empty:
            print(f"  {season}: skipped ({len(train)} training rows)")
            continue
        flat = Week1MatchupModel().fit(train)
        hurdle = Week1HurdleModel().fit(train)
        out.append(test[["player_id", "season"]].assign(
            matchup=flat.predict(test), hurdle=hurdle.predict(test),
            matchup_train_rows=len(train), matchup_alpha=flat.alpha_))
        print(f"  {season}: fit on {len(train)} rows, "
              f"matchup alpha {flat.alpha_}, hurdle {hurdle.params_}",
              flush=True)
    return (pd.concat(out, ignore_index=True) if out else
            pd.DataFrame(columns=["player_id", "season", "matchup", "hurdle"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=[2021, 2025])
    ap.add_argument("--pool-from", type=int, default=2013,
                    help="first season of cold-start rows the matchup arm may "
                         "train on (it needs season-1 team stats)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    first, last = args.seasons
    weekly = _weekly()
    levels = team_position_levels(weekly)
    debut = weekly.groupby("player_id")["season"].min().to_dict()
    rounds = draft_rounds()

    frames = []
    for season in range(first, last + 1):
        print(f"fitting step 8 as of {season} ...", flush=True)
        pace = step8_pace(season)
        if pace.empty:
            print(f"  no pairs for {season}; skipped")
            continue
        rows = (pace.merge(week1_actuals(season), on="player_id", how="inner")
                    .merge(prior_ppg(season), on="player_id", how="left"))
        rows["season"] = season
        rows = add_team_share(rows, season, weekly, levels, debut, rounds)
        print(f"  {len(rows)} players with a week-1 row "
              f"({int(rows.cold_start.sum())} cold start, "
              f"{int(rows.team_share.notna().sum())} with a team level)")
        frames.append(rows)

    if not frames:
        print("nothing to score")
        return 1
    rows = pd.concat(frames, ignore_index=True)

    # Leave-one-season-out so the floor is not fitted on the season it scores.
    # With a single season there is nothing to leave out; say so rather than
    # silently scoring against NaN and reporting an empty table.
    single_season = rows["season"].nunique() == 1
    parts = []
    for season, g in rows.groupby("season"):
        other = rows if single_season else rows[rows.season != season]
        means = other.groupby("position")["actual"].mean()
        parts.append(g.assign(position_mean=g["position"].map(means)))
    rows = pd.concat(parts, ignore_index=True).dropna(subset=["position_mean"])
    # A cold-start player has no last season; the floor stands in for it. Same
    # for a team room that did not exist last season, or a veteran with no
    # prior share to carry.
    rows["prior_ppg"] = rows["prior_ppg"].fillna(rows["position_mean"])
    rows["team_share"] = rows["team_share"].fillna(rows["position_mean"])

    # A fixed half-and-half, pre-registered rather than tuned: the question is
    # whether team context adds anything ON TOP of the published number, and a
    # weight fitted on these same rows could not answer that honestly.
    rows["blend_pace_team"] = 0.5 * rows["step8_pace"] + 0.5 * rows["team_share"]

    arms = ["step8_pace", "step8_per_game", "team_share", "blend_pace_team",
            "position_mean", "prior_ppg"]
    rows = rows.dropna(subset=["step8_per_game"])

    print("\nmatchup arm (cold start only)")
    rows = rows.merge(matchup_arm(sorted(rows["season"].unique()),
                                  args.pool_from),
                      on=["player_id", "season"], how="left")
    # The matchup arm exists only for cold-start players, so its comparison
    # has to be on the rows where every arm has a number.
    head_to_head = rows[rows["matchup"].notna()]

    result = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "seasons": [first, last],
        "train_from": TRAIN_FROM,
        "population": "players with a week-1 row in player_weekly_stats",
        "position_mean_basis": ("in-sample (single season)" if single_season
                                else "leave-one-season-out"),
        "pooled": evaluate(rows, arms),
        "by_season": {int(s): evaluate(g, arms)
                      for s, g in rows.groupby("season")},
        "by_position_cold_start": {
            pos: evaluate(g, arms).get("all")
            for pos, g in rows[rows.cold_start == 1].groupby("position")},
        "matchup_head_to_head": evaluate(head_to_head, arms + ["matchup", "hurdle"]),
        # DIAGNOSTIC, NOT A SERVABLE FILTER. Realised snap share is not known
        # before kickoff, so conditioning on it cannot be shipped. It answers a
        # different question: with the "did he even play a role" part of the
        # problem removed, does knowing his team and his opponent finally beat
        # draft capital? Every week-1 row in this table already has snaps > 0,
        # so the thresholds separate contributors from three-snap bench bodies
        # rather than players from inactives.
        "by_snap_share": {
            str(t): {
                "cold_start": evaluate(
                    head_to_head[head_to_head.snap_share >= t],
                    arms + ["matchup", "hurdle"]).get("cold_start"),
                "veteran": evaluate(rows[(rows.cold_start == 0)
                                         & (rows.snap_share >= t)],
                                    arms).get("veteran"),
            } for t in SNAP_SHARE_THRESHOLDS},
        "matchup_coverage": {
            "cold_start_rows": int((rows.cold_start == 1).sum()),
            "with_matchup": int(((rows.cold_start == 1)
                                 & rows["matchup"].notna()).sum())},
    }

    def table(title, scored, arm_names):
        print(f"\n=== {title} ===")
        for slice_name, scores in scored.items():
            print(f"\n{slice_name}  (mean actual {scores['mean_actual']})")
            for arm in arm_names:
                s = scores.get(arm)
                if s:
                    print(f"  {arm:<16} n={s['n']:<5} MAE {s['mae']:>5}  "
                          f"RMSE {s['rmse']:>5}  bias {s['bias']:>+6}  "
                          f"R2 {s['r2']:>+7}")

    table("matchup head-to-head (rows where every arm has a number)",
          result["matchup_head_to_head"], arms + ["matchup", "hurdle"])

    for t in SNAP_SHARE_THRESHOLDS:
        scored = result["by_snap_share"][str(t)]["cold_start"]
        if scored:
            print(f"\n=== cold start, snap share >= {t:.2f} "
                  f"(diagnostic; not knowable before kickoff) ===")
            print(f"  mean actual {scored['mean_actual']}")
            for arm in arms + ["matchup", "hurdle"]:
                sc = scored.get(arm)
                if sc:
                    print(f"  {arm:<16} n={sc['n']:<5} MAE {sc['mae']:>5}  "
                          f"RMSE {sc['rmse']:>5}  bias {sc['bias']:>+6}  "
                          f"R2 {sc['r2']:>+7}")
    table("all rows", result["pooled"], arms)

    out = args.out or (OUT_DIR / f"week1_coldstart_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")

    # Per-row output, so a later question about significance or per-season
    # consistency does not need another five refits of step 8.
    rows_out = out.with_suffix(".rows.csv")
    rows.to_csv(rows_out, index=False)
    print(f"      {rows_out.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
