"""Player-tier and predicted-score-bucket definitions for Phase 4
(next_focus.md) validation breakdowns.

Player tier is a property of the PLAYER (their established level of
production), not of any single week's score — distinct from "predicted-score
bucket," which bins a single week's model output. Tier is assigned from
PRIOR-SEASON PPG only (leakage-safe: a player's tier while evaluating season
N must never depend on season N data).

New tier edges, not reused from elsewhere: `src/features/season_long_
features.py`'s ROOKIE_ARCHETYPES (elite/high/mid/low PPG bands) is explicitly
scoped to rookie draft-capital comps ("Rookie comparable archetypes based on
draft capital and combine metrics") — reusing it here for general veteran
tiering would mislabel it. `src/evaluation/metrics.py:tier_classification_
accuracy` has a real general-purpose tier scheme but with one flat set of
edges (not position-adjusted) and it tiers a single score, not a player.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

# (lo, hi) PPG bands per position, informed by the position-scale pattern
# already visible in ROOKIE_ARCHETYPES (QB scores higher than TE, etc.) but
# built fresh for a general veteran population, not rookies specifically.
PLAYER_TIER_PPG_EDGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "QB": {"elite": (18.0, float("inf")), "starter": (12.0, 18.0), "depth": (6.0, 12.0), "waiver": (0.0, 6.0)},
    "RB": {"elite": (14.0, float("inf")), "starter": (9.0, 14.0), "depth": (4.0, 9.0), "waiver": (0.0, 4.0)},
    "WR": {"elite": (13.0, float("inf")), "starter": (8.0, 13.0), "depth": (3.0, 8.0), "waiver": (0.0, 3.0)},
    "TE": {"elite": (9.0, float("inf")), "starter": (5.0, 9.0), "depth": (2.0, 5.0), "waiver": (0.0, 2.0)},
}

PLAYER_TIERS = ("elite", "starter", "depth", "waiver", "rookie")

# Fixed ranges on the model's own weekly point PREDICTION, consistent with
# the edge-bucketing style already used in scripts/compute_confidence_tiers.py.
PREDICTED_SCORE_BUCKET_EDGES: Dict[str, Tuple[float, float]] = {
    "0-5": (0.0, 5.0),
    "5-10": (5.0, 10.0),
    "10-15": (10.0, 15.0),
    "15-20": (15.0, 20.0),
    "20+": (20.0, float("inf")),
}


def assign_player_tier(prior_season_ppg: Optional[float], position: str) -> str:
    """Rookies (no prior season) get an explicit 'rookie' tier rather than
    being forced into one of the four PPG bands — that's honest, not a gap.
    """
    if prior_season_ppg is None or pd.isna(prior_season_ppg):
        return "rookie"
    edges = PLAYER_TIER_PPG_EDGES.get(position)
    if edges is None:
        return "rookie"
    for name, (lo, hi) in edges.items():
        if lo <= prior_season_ppg < hi:
            return name
    return "waiver"


def assign_score_bucket(predicted_value: float) -> str:
    if predicted_value is None or pd.isna(predicted_value):
        return "unknown"
    if predicted_value < 0:
        return "0-5"  # negative single-game PPR (fumbles/INTs) buckets with the floor
    for name, (lo, hi) in PREDICTED_SCORE_BUCKET_EDGES.items():
        if lo <= predicted_value < hi:
            return name
    return "20+"


def compute_prior_season_ppg(df: pd.DataFrame, target_col: str = "fantasy_points") -> pd.DataFrame:
    """Returns df with a new 'prior_season_ppg' column: each row's player's
    mean target_col from the season immediately before that row's season.

    Leakage-safe by construction — mirrors the shift-by-one-season merge
    pattern in src/evaluation/baselines.py:positional_rank_baseline (lines
    72-105), reused conceptually (same idea, different output shape: a tier
    input here vs. a baseline prediction there).
    """
    season_avg = (
        df.groupby(["player_id", "season"])[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "prior_season_ppg"})
    )
    season_avg["_prior_season"] = season_avg["season"] + 1
    season_avg = season_avg.rename(columns={"season": "_src_season"})

    result = df.merge(
        season_avg[["player_id", "_prior_season", "prior_season_ppg"]],
        left_on=["player_id", "season"],
        right_on=["player_id", "_prior_season"],
        how="left",
    ).drop(columns=["_prior_season"])
    return result
