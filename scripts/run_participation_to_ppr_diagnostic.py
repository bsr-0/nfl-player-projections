#!/usr/bin/env python3
"""Lightweight diagnostic: convert Phase 2 expected snap share to PPR.

This is deliberately not a Phase 3 promotion gate.  It asks only how much
PPR variance a deterministic opportunity-only conversion explains.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.single_week_ppr.participation_integration import (
    KEY, deduplicate_ppr_player_weeks, validate_phase2_manifest, validate_phase2_oof,
)


def metrics(g: pd.DataFrame) -> dict:
    g = g.dropna(subset=["fantasy_points", "prediction"])
    if g.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan}
    y, p = g.fantasy_points.to_numpy(), g.prediction.to_numpy()
    return {
        "n": int(len(g)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if len(g) > 1 else np.nan,
        "spearman": float(spearmanr(y, p).correlation) if len(g) > 1 else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2-oof", type=Path,
                    default=Path("data/experiments/participation_system/phase2/oof_predictions.csv"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("data/experiments/participation_system/phase3/participation_to_ppr"))
    ap.add_argument("--phase2-model", default="hist_gbm")
    args = ap.parse_args()
    validate_phase2_manifest(args.phase2_oof)
    raw = pd.read_csv(args.phase2_oof)
    oof = validate_phase2_oof(raw, model=args.phase2_model)
    oof = oof[KEY + ["p2_expected_snap_share"]].rename(columns={"p2_expected_snap_share": "prediction_share"})

    con = sqlite3.connect("data/nfl_data.db")
    ppr = pd.read_sql_query(
        """SELECT s.player_id, s.season, s.week, s.team, s.fantasy_points,
                  s.snap_count, s.team_snaps, p.position
           FROM player_weekly_stats s JOIN players p ON p.player_id=s.player_id
           WHERE p.position IN ('QB','RB','WR','TE') AND s.fantasy_points IS NOT NULL""", con)
    ppr = deduplicate_ppr_player_weeks(ppr)
    matched = ppr.merge(oof, on=KEY, how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError("no matched PPR/Phase 2 OOF rows")

    seasons = sorted(matched.season.unique())
    rows, fold_rates = [], []
    for season in seasons:
        prior = ppr[ppr.season < season].copy()
        test = matched[matched.season.eq(season)].copy()
        if prior.empty or test.empty:
            continue
        # Position-specific PPR per offensive snap, fit strictly before test season.
        rates = (prior.groupby("position")
                 .agg(prior_ppr=("fantasy_points", "sum"), prior_snaps=("snap_count", "sum"))
                 .assign(ppr_per_snap=lambda x: x.prior_ppr / x.prior_snaps.replace(0, np.nan)))
        # Team offensive snaps are repeated across player rows; deduplicate games
        # before estimating the prior-season team-snap environment.
        team_games = prior[["season", "week", "team", "team_snaps"]].drop_duplicates()
        team_rate = team_games.groupby("season")["team_snaps"].mean().mean()
        if not np.isfinite(team_rate) or team_rate <= 0:
            raise ValueError(f"invalid prior team-snap rate for {season}: {team_rate}")
        test = test.join(rates["ppr_per_snap"], on="position")
        test["prior_team_snaps_per_game"] = float(team_rate)
        test["prediction"] = (test.prediction_share * test.prior_team_snaps_per_game
                              * test.ppr_per_snap)
        prior_counts = prior.groupby("player_id").size()
        test["prior_player_weeks"] = test.player_id.map(prior_counts).fillna(0).astype(int)
        test["history_group"] = np.where(test.prior_player_weeks < 3, "sparse", "established")
        for position, g in test.groupby("position"):
            for subgroup, h in [("all", g), ("sparse", g[g.history_group.eq("sparse")]),
                                ("established", g[g.history_group.eq("established")])]:
                rows.append({"season": int(season), "position": position,
                             "subgroup": subgroup, **metrics(h)})
        fold_rates.extend({
            "test_season": int(season), "position": pos,
            "prior_ppr_per_snap": float(row.ppr_per_snap),
            "prior_team_snaps_per_game": float(team_rate),
            "prior_rows": int(len(prior[prior.position.eq(pos)])),
        } for pos, row in rates.iterrows() if np.isfinite(row.ppr_per_snap))

    result = pd.DataFrame(rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "diagnostic_metrics.csv", index=False)
    pd.DataFrame(fold_rates).to_csv(out / "conversion_rates.csv", index=False)
    matched.to_csv(out / "matched_inputs.csv", index=False)
    report = {
        "diagnostic": "participation_expected_snap_share_to_ppr",
        "phase2_model": args.phase2_model,
        "conversion": "expected_snap_share * prior_team_snaps_per_game * prior_position_ppr_per_snap",
        "leakage_rule": "conversion rates use only seasons strictly before each test season",
        "zero_point_weeks_retained": True,
        "promotion_gate": "none; standalone sanity-check only",
        "n_matched": int(len(matched)),
        "seasons": [int(s) for s in seasons],
    }
    (out / "diagnostic_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(result[result.subgroup.eq("all")].to_string(index=False))
    print(f"Saved diagnostic artifacts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
