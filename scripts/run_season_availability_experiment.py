#!/usr/bin/env python3
"""Does historical availability beat a CONSTANT availability assumption?

Step 8, season half. The weekly half is already closed negatively: the
weekly opportunity layer failed its pre-registered test in 12/12 folds
(GAPS.md 2026-08-20). This asks the separate, untested question --
season-level projections must estimate how many games a player gets, and
the weekly experiment never had to.

    E[season PPR] = E[games played] x E[PPR per game | played]

**Stage 1 holds the production term at ORACLE** (each player's realized
PPR per game in the target season) and varies ONLY the availability term.
That is deliberate: if a historical availability estimate cannot beat a
constant when production is perfect, it cannot help with a real production
model either, and the hypothesis dies without fitting anything. It is the
smallest experiment capable of falsifying the claim.

Arms (availability term only):
    const_position   position mean games-played rate, TRAINING SEASONS ONLY
    hist_player      the player's own rate across prior seasons
    hist_shrunk      hist_player shrunk toward const_position by games observed

Leakage: every arm reads seasons strictly before the target season. No
current-season, no realized snaps, no future roster status.

Usage:
    python scripts/run_season_availability_experiment.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, MIN_HISTORICAL_YEAR, POSITIONS
from src.models.single_week_ppr.population import (
    RECEIVING_CHARTING_MIN_SEASON, RECEIVING_DEPENDENT_POSITIONS,
    SNAP_LABEL_MIN_SEASON, label_participation,
)

REGULAR_SEASON_MAX_WEEK = 18
# Shrinkage strength: weight on the player's own history is
# n_prior_games / (n_prior_games + K).
SHRINKAGE_K = 16.0
OUT_PATH = Path("data/experiments/season_availability.csv")


def load_player_seasons() -> pd.DataFrame:
    """Participated games and PPR per player-season, participation contract applied."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(f"""
        SELECT pws.player_id, p.position, pws.season, pws.week, pws.team,
               pws.snap_count, pws.fantasy_points, pws.data_source
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE p.position IN ({','.join('?' * len(POSITIONS))})
          AND pws.week <= {REGULAR_SEASON_MAX_WEEK}
          AND pws.season >= {MIN_HISTORICAL_YEAR}
    """, conn, params=list(POSITIONS))
    sched = pd.read_sql(
        f"SELECT season, week, home_team AS team FROM schedule WHERE week <= {REGULAR_SEASON_MAX_WEEK} "
        f"UNION ALL SELECT season, week, away_team AS team FROM schedule "
        f"WHERE week <= {REGULAR_SEASON_MAX_WEEK}", conn)
    conn.close()

    df["participation_quality"] = label_participation(df)
    df = df[df["participation_quality"] >= 1]
    # Same receiving floor the production population uses.
    recv = df["position"].isin(RECEIVING_DEPENDENT_POSITIONS)
    df = df[~(recv & (df["season"] < RECEIVING_CHARTING_MIN_SEASON))]

    played = df.groupby(["player_id", "position", "season"]).agg(
        games_played=("week", "nunique"),
        ppr=("fantasy_points", "sum"),
        team=("team", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
    ).reset_index()
    possible = sched.groupby(["season", "team"])["week"].nunique().rename("possible_games").reset_index()
    played = played.merge(possible, on=["season", "team"], how="left")
    played["possible_games"] = played["possible_games"].fillna(REGULAR_SEASON_MAX_WEEK - 1)
    played["rate"] = played["games_played"] / played["possible_games"]
    played["ppr_per_game"] = played["ppr"] / played["games_played"]
    return played


def build_arms(hist: pd.DataFrame, target: pd.DataFrame, season: int) -> pd.DataFrame:
    """Availability estimates for `target`, from `hist` (seasons < season only)."""
    out = target.copy()
    pos_rate = hist.groupby("position")["rate"].mean()
    out["const_position"] = out["position"].map(pos_rate)

    per_player = hist.groupby("player_id").agg(
        prior_games=("games_played", "sum"), prior_possible=("possible_games", "sum")).reset_index()
    per_player["hist_player"] = per_player["prior_games"] / per_player["prior_possible"]
    out = out.merge(per_player[["player_id", "hist_player", "prior_games"]], on="player_id", how="left")

    w = out["prior_games"].fillna(0) / (out["prior_games"].fillna(0) + SHRINKAGE_K)
    out["hist_shrunk"] = w * out["hist_player"].fillna(out["const_position"]) + (1 - w) * out["const_position"]
    # No history -> the arms are identical by construction; keep them, but flag.
    out["has_history"] = out["hist_player"].notna()
    out["hist_player"] = out["hist_player"].fillna(out["const_position"])

    # Prior-season PPR per game: the production term for stage 2. Most recent
    # prior season only, so it is a plausible pre-season forecast rather than
    # a career average.
    recent = (hist.sort_values("season").groupby("player_id").tail(1)
              [["player_id", "ppr_per_game"]].rename(columns={"ppr_per_game": "prior_ppr_per_game"}))
    out = out.merge(recent, on="player_id", how="left")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = ap.parse_args()

    panel = load_player_seasons()
    print(f"player-seasons in the contract population: {len(panel)}")
    print(f"  snap-era floor {SNAP_LABEL_MIN_SEASON}, receiving floor "
          f"{RECEIVING_CHARTING_MIN_SEASON} for {sorted(RECEIVING_DEPENDENT_POSITIONS)}")

    arms = ["const_position", "hist_player", "hist_shrunk"]
    rows = []
    for season in args.seasons:
        hist = panel[panel["season"] < season]
        target = panel[panel["season"] == season]
        if hist.empty or target.empty:
            continue
        est = build_arms(hist, target, season)
        for pos, g in est.groupby("position"):
            for arm in arms:
                pred_games = g[arm] * g["possible_games"]
                # Stage 1: ORACLE production. Only availability varies.
                pred_ppr = pred_games * g["ppr_per_game"]
                rows.append({
                    "season": season, "position": pos, "arm": arm, "n": len(g),
                    "n_with_history": int(g["has_history"].sum()),
                    "games_mae": float((pred_games - g["games_played"]).abs().mean()),
                    "games_bias": float((pred_games - g["games_played"]).mean()),
                    "season_mae": float((pred_ppr - g["ppr"]).abs().mean()),
                    "season_bias": float((pred_ppr - g["ppr"]).mean()),
                })
                # Stage 2: REALISTIC production term -- the player's prior-season
                # PPR per game, a legitimate pre-season forecast. Both the
                # availability and production terms now come from prior seasons
                # and are correlated (starters have high rates AND high
                # availability), so this is where a multiplicative form can
                # double-count. Applied identically across arms.
                if "prior_ppr_per_game" in g.columns:
                    real = g["prior_ppr_per_game"]
                    ok = real.notna()
                    if ok.any():
                        pp = (pred_games[ok] * real[ok])
                        rows[-1]["season_mae_realistic"] = float((pp - g.loc[ok, "ppr"]).abs().mean())
                        rows[-1]["season_bias_realistic"] = float((pp - g.loc[ok, "ppr"]).mean())
                        rows[-1]["n_realistic"] = int(ok.sum())
                sub = g[g["has_history"]]
                if len(sub):
                    pg = sub[arm] * sub["possible_games"]
                    rows[-1]["games_mae_hist_only"] = float((pg - sub["games_played"]).abs().mean())
                    rows[-1]["season_mae_hist_only"] = float(
                        (pg * sub["ppr_per_game"] - sub["ppr"]).abs().mean())

    res = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, index=False)

    pd.set_option("display.width", 200)
    print("\n=== GAMES PLAYED: MAE by position (lower is better) ===")
    print(res.pivot_table(index="position", columns="arm", values="games_mae").round(3).to_string())
    print("\n=== GAMES PLAYED: MAE, players WITH prior history only ===")
    print(res.pivot_table(index="position", columns="arm", values="games_mae_hist_only").round(3).to_string())
    print("\n=== SEASON PPR (oracle production): MAE ===")
    print(res.pivot_table(index="position", columns="arm", values="season_mae").round(2).to_string())
    print("\n=== SEASON PPR (oracle production): bias ===")
    print(res.pivot_table(index="position", columns="arm", values="season_bias").round(2).to_string())
    print("\n=== SEASON PPR (REALISTIC prior-season production): MAE ===")
    print(res.pivot_table(index="position", columns="arm", values="season_mae_realistic").round(2).to_string())
    print("\n=== SEASON PPR (REALISTIC): bias ===")
    print(res.pivot_table(index="position", columns="arm", values="season_bias_realistic").round(2).to_string())
    print("\n=== per fold, games MAE ===")
    print(res.pivot_table(index=["position", "season"], columns="arm", values="games_mae").round(3).to_string())
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
