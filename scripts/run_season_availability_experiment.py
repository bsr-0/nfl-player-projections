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
    hist_shrunk_est  hist_shrunk, computed by the PRODUCTION estimator rather
                     than by the hand-rolled arithmetic above. Should match
                     hist_shrunk to floating point; it is here as a standing
                     check that the two have not drifted apart.
    hist_shrunk_k*   hist_shrunk, but shrunk toward the position x draft-round
                     mean instead of the position mean, at bucket weight K.
                     One arm per --draft-bucket-k value.

The draft-round arms exist to answer the question the position mean cannot:
a player with no history gets w = 0, so his estimate IS the shrinkage target,
and a target that ignores draft capital hands every rookie at a position the
same number. Read `n_with_history` alongside the MAE -- the cold-start rows
are the only ones where these arms can differ much from hist_shrunk.

Leakage: every arm reads seasons strictly before the target season. No
current-season, no realized snaps, no future roster status.

Usage:
    python scripts/run_season_availability_experiment.py
    python scripts/run_season_availability_experiment.py --draft-bucket-k 25 50 100 200
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
    RECEIVING_CHARTING_MIN_SEASON, RECEIVING_DEPENDENT_POSITIONS, SNAP_LABEL_MIN_SEASON,
)
# One definition of the panel and the estimator, shared with production.
from src.models.single_week_ppr.season_availability import (
    DRAFT_BUCKET_K, SHRINKAGE_K, SeasonAvailabilityEstimator, load_player_seasons,
)

OUT_PATH = Path("data/experiments/season_availability.csv")


def build_arms(hist: pd.DataFrame, target: pd.DataFrame, season: int,
               panel: pd.DataFrame, bucket_ks: list) -> pd.DataFrame:
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

    # Estimator arms. Fit on the FULL panel deliberately: fit() drops seasons
    # >= before_season itself, so causality here is structural rather than
    # dependent on this script having filtered correctly.
    est = SeasonAvailabilityEstimator(use_draft_prior=False).fit(panel, before_season=season)
    out["hist_shrunk_est"] = est.predict_rate(out).to_numpy()
    for k in bucket_ks:
        est_k = SeasonAvailabilityEstimator(
            draft_bucket_k=k, use_draft_prior=True).fit(panel, before_season=season)
        out[f"hist_shrunk_k{k:g}"] = est_k.predict_rate(out).to_numpy()

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
    ap.add_argument("--draft-bucket-k", nargs="+", type=float, default=[DRAFT_BUCKET_K],
                    help="Cell weight(s) for the position x draft-round target. "
                         "One arm per value; sweep to tune it.")
    args = ap.parse_args()

    panel = load_player_seasons()
    print(f"player-seasons in the contract population: {len(panel)}")
    print(f"  snap-era floor {SNAP_LABEL_MIN_SEASON}, receiving floor "
          f"{RECEIVING_CHARTING_MIN_SEASON} for {sorted(RECEIVING_DEPENDENT_POSITIONS)}")
    known_round = int(panel["draft_round"].notna().sum()) if "draft_round" in panel else 0
    print(f"  draft round known for {known_round} of {len(panel)} player-seasons; "
          f"the rest bucket as undrafted")

    arms = (["const_position", "hist_player", "hist_shrunk", "hist_shrunk_est"]
            + [f"hist_shrunk_k{k:g}" for k in args.draft_bucket_k])
    rows = []
    for season in args.seasons:
        hist = panel[panel["season"] < season]
        target = panel[panel["season"] == season]
        if hist.empty or target.empty:
            continue
        est = build_arms(hist, target, season, panel, args.draft_bucket_k)
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
                # The complement, and the only place the draft-round arms can
                # move much: these rows have w = 0, so the arm IS the target.
                cold = g[~g["has_history"]]
                if len(cold):
                    pg = cold[arm] * cold["possible_games"]
                    rows[-1]["n_cold"] = int(len(cold))
                    rows[-1]["games_mae_cold"] = float((pg - cold["games_played"]).abs().mean())
                    rows[-1]["games_bias_cold"] = float((pg - cold["games_played"]).mean())

    res = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, index=False)

    pd.set_option("display.width", 200)
    print("\n=== GAMES PLAYED: MAE by position (lower is better) ===")
    print(res.pivot_table(index="position", columns="arm", values="games_mae").round(3).to_string())
    print("\n=== GAMES PLAYED: MAE, players WITH prior history only ===")
    print(res.pivot_table(index="position", columns="arm", values="games_mae_hist_only").round(3).to_string())
    print("\n=== GAMES PLAYED: MAE, COLD START only (no prior history) ===")
    print("    NOTE: these are cold-start players who played at least one game.")
    print("    A pick who never dressed is absent from the panel entirely, so the")
    print("    round-to-round spread measured here understates the real one.")
    print(res.pivot_table(index="position", columns="arm", values="games_mae_cold").round(3).to_string())
    print("\n=== GAMES PLAYED: bias, COLD START only ===")
    print(res.pivot_table(index="position", columns="arm", values="games_bias_cold").round(3).to_string())
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
