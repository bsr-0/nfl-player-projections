#!/usr/bin/env python
"""Diagnostic (not tuning) for the QB-vs-TE Phase 9 calibration contrast
flagged after the 2026-08-21/22 TE empty-donor-pool fix.

Reads the regenerated, lineage-correct artifacts
(phase4_row_level_predictions_v2_corrected*.csv,
data/experiments/phase9_season_simulation.csv) and reports, per position:
  - the donor residual distribution feeding the block-bootstrap
  - donor-pool composition (player-season count, sequence-length spread)
  - how much of each season is actually simulated (synthetic week share)
  - empirical quantile coverage, broken out by season

Does not change season_simulation.py or FINAL_CONFIG. Purely descriptive,
to decide WHERE (donor distribution vs. simulation mechanics vs. small
synthetic-week share) the QB/TE calibration gap in
`run_phase9_season_simulation.py`'s printed summary comes from, before
touching anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.final_config import FINAL_CONFIG
from src.models.single_week_ppr.season_simulation import build_residual_donor_pools

RESIDUAL_CSV_PATHS = [
    Path("data/experiments/phase4_row_level_predictions_v2_corrected.csv"),
    Path("data/experiments/phase4_row_level_predictions_v2_corrected_2020_2022.csv"),
]
PHASE9_CSV = Path("data/experiments/phase9_season_simulation.csv")
SEASONS = (2023, 2024, 2025)
POSITIONS = ("QB", "RB", "WR", "TE")


def donor_residual_stats():
    print("=" * 78)
    print("1. Donor residual distribution feeding the block-bootstrap")
    print("   (pooled across seasons < 2025, i.e. the widest donor pool used)")
    print("=" * 78)
    rows = []
    for position in POSITIONS:
        arch = FINAL_CONFIG[position]["architecture"]
        pool = build_residual_donor_pools(RESIDUAL_CSV_PATHS, position, arch, before_season=2025)
        all_residuals = np.array([r for seq in pool.values() for _, r in seq])
        lengths = np.array([len(seq) for seq in pool.values()])
        rows.append({
            "position": position, "architecture": arch,
            "donor_player_seasons": len(pool),
            "total_residuals": len(all_residuals),
            "resid_mean": all_residuals.mean(), "resid_std": all_residuals.std(),
            "resid_p10": np.percentile(all_residuals, 10),
            "resid_p50": np.percentile(all_residuals, 50),
            "resid_p90": np.percentile(all_residuals, 90),
            "seq_len_median": np.median(lengths), "seq_len_p10": np.percentile(lengths, 10),
            "seq_len_p90": np.percentile(lengths, 90),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out.round(3).to_string(index=False))
    print()
    return out


def synthetic_week_share():
    print("=" * 78)
    print("2. Synthetic-week share of each season (drives how much a player's")
    print("   interval is actually simulated vs. known)")
    print("=" * 78)
    df = pd.read_csv(PHASE9_CSV)
    df["synthetic_weeks"] = df["weeks_predicted"] - df["games_actually_played"]
    df["synthetic_share"] = df["synthetic_weeks"] / df["weeks_predicted"].replace(0, np.nan)
    df["interval_width"] = df["p90"] - df["p25"]
    summary = df.groupby("position").agg(
        n=("player", "size"),
        mean_synthetic_weeks=("synthetic_weeks", "mean"),
        mean_synthetic_share=("synthetic_share", "mean"),
        pct_fully_known=("synthetic_weeks", lambda s: float((s == 0).mean())),
        mean_interval_width=("interval_width", "mean"),
        median_interval_width=("interval_width", "median"),
    )
    print(summary.round(3).to_string())
    print()
    return df


def calibration_by_position_season(df: pd.DataFrame):
    print("=" * 78)
    print("3. Empirical coverage by position AND season (not just pooled)")
    print("=" * 78)
    rows = []
    for (position, season), g in df.groupby(["position", "season"]):
        n = len(g)
        rows.append({
            "position": position, "season": season, "n": n,
            "below_p25": float((g["actual_season_total"] < g["p25"]).mean()),
            "below_p50": float((g["actual_season_total"] < g["p50"]).mean()),
            "below_p75": float((g["actual_season_total"] < g["p75"]).mean()),
            "below_p90": float((g["actual_season_total"] < g["p90"]).mean()),
            "in_p25_p75": float(((g["actual_season_total"] >= g["p25"]) & (g["actual_season_total"] <= g["p75"])).mean()),
        })
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    print()

    print("   Same, restricted to players with >=1 synthetic week")
    print("   (fully-known seasons are zero-width intervals by construction")
    print("   and mechanically satisfy/fail coverage depending on rounding --")
    print("   excluding them isolates the simulation's actual behavior).")
    print("-" * 78)
    sim_only = df[df["synthetic_weeks"] > 0]
    rows2 = []
    for (position, season), g in sim_only.groupby(["position", "season"]):
        n = len(g)
        rows2.append({
            "position": position, "season": season, "n": n,
            "below_p25": float((g["actual_season_total"] < g["p25"]).mean()),
            "below_p50": float((g["actual_season_total"] < g["p50"]).mean()),
            "below_p75": float((g["actual_season_total"] < g["p75"]).mean()),
            "below_p90": float((g["actual_season_total"] < g["p90"]).mean()),
            "in_p25_p75": float(((g["actual_season_total"] >= g["p25"]) & (g["actual_season_total"] <= g["p75"])).mean()),
        })
    out2 = pd.DataFrame(rows2)
    print(out2.round(3).to_string(index=False))
    print()
    return out, out2


def worst_qb_errors(df: pd.DataFrame):
    print("=" * 78)
    print("4. QB: the 10 largest |P50 - actual| errors (eyeball what's driving 12.60 MAE)")
    print("=" * 78)
    qb = df[df["position"] == "QB"].copy()
    qb["p50_abs_error"] = (qb["p50"] - qb["actual_season_total"]).abs()
    cols = ["player", "season", "possible_weeks", "weeks_predicted", "games_actually_played",
            "synthetic_weeks", "availability_rate", "p25", "p50", "p75", "p90", "actual_season_total", "p50_abs_error"]
    print(qb.sort_values("p50_abs_error", ascending=False)[cols].head(10).round(2).to_string(index=False))
    print()


if __name__ == "__main__":
    donor_residual_stats()
    df = synthetic_week_share()
    calibration_by_position_season(df)
    worst_qb_errors(df)
