"""
Decision Optimization Layer for NFL Fantasy Predictions.

Per Agent Directive V7 Section 9: the system must optimize the downstream
action policy, because the best predictive model is not necessarily the
best decision engine.

This module converts point predictions + uncertainty estimates into
actionable decisions:
  1. VOR-based draft rankings with positional scarcity
  2. Start/Sit recommendations with abstention for low confidence
  3. Waiver wire pickup priority scoring
  4. Decision quality evaluation separate from prediction quality
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replacement-level baselines (PPR scoring, approximate waiver-wire floor)
# ---------------------------------------------------------------------------
DEFAULT_REPLACEMENT_LEVEL: Dict[str, float] = {
    "QB": 14.0,   # QB15-QB18 range (streaming)
    "RB": 7.0,    # RB30-RB40 range (waiver wire)
    "WR": 7.5,    # WR35-WR45 range (waiver wire)
    "TE": 5.0,    # TE12-TE15 range (streaming)
}

# Positional scarcity weights for draft ranking
# Higher = position is scarcer / drops off faster
POSITIONAL_SCARCITY: Dict[str, float] = {
    "QB": 0.8,   # Least scarce — many startable QBs
    "RB": 1.3,   # Most scarce — deep drop-off
    "WR": 1.0,   # Moderate scarcity
    "TE": 1.1,   # Moderate-high scarcity — elite TE advantage
}


def compute_vor_rankings(
    predictions: pd.DataFrame,
    pred_col: str = "predicted_points",
    position_col: str = "position",
    player_col: str = "player_name",
    replacement_level: Optional[Dict[str, float]] = None,
    apply_scarcity: bool = True,
) -> pd.DataFrame:
    """Compute Value Over Replacement (VOR) rankings for draft/lineup decisions.

    VOR measures how much better a player is than the best available
    replacement at their position. This is the standard framework for
    fantasy draft rankings.

    Args:
        predictions: DataFrame with player predictions.
        pred_col: Column containing point predictions.
        position_col: Column containing position labels.
        player_col: Column containing player names/IDs.
        replacement_level: Dict of position -> replacement-level points.
        apply_scarcity: If True, multiply VOR by positional scarcity weight.

    Returns:
        DataFrame with VOR rankings, sorted by adjusted VOR descending.
    """
    if replacement_level is None:
        replacement_level = DEFAULT_REPLACEMENT_LEVEL

    df = predictions.copy()

    # Compute raw VOR
    df["replacement_level"] = df[position_col].map(replacement_level).fillna(7.0)
    df["vor_raw"] = df[pred_col] - df["replacement_level"]

    # Apply positional scarcity adjustment
    if apply_scarcity:
        df["scarcity_weight"] = df[position_col].map(POSITIONAL_SCARCITY).fillna(1.0)
        df["vor_adjusted"] = df["vor_raw"] * df["scarcity_weight"]
    else:
        df["vor_adjusted"] = df["vor_raw"]

    # Rank by adjusted VOR
    df["vor_rank"] = df["vor_adjusted"].rank(ascending=False, method="min").astype(int)

    # Position rank
    df["position_rank"] = (
        df.groupby(position_col)[pred_col]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Tier assignment based on VOR
    df["vor_tier"] = pd.cut(
        df["vor_adjusted"],
        bins=[-np.inf, 0, 5, 10, 20, np.inf],
        labels=["below_replacement", "depth", "flex", "starter", "elite"],
    )

    sort_cols = ["vor_rank"]
    result_cols = [
        player_col, position_col, pred_col,
        "vor_raw", "vor_adjusted", "vor_rank", "position_rank", "vor_tier",
    ]
    existing = [c for c in result_cols if c in df.columns]
    return df[existing].sort_values("vor_rank")


def start_sit_recommendations(
    predictions: pd.DataFrame,
    pred_col: str = "predicted_points",
    std_col: str = "predicted_std",
    position_col: str = "position",
    player_col: str = "player_name",
    confidence_threshold: float = 0.3,
    min_projection: float = 5.0,
) -> pd.DataFrame:
    """Generate start/sit recommendations with abstention for low confidence.

    Per Directive V7 Section 9: abstention is a first-class policy.
    When prediction confidence is too low, the system recommends
    "uncertain" rather than making a bad recommendation.

    Args:
        predictions: DataFrame with player predictions and uncertainty.
        pred_col: Column with point predictions.
        std_col: Column with prediction std deviations.
        position_col: Column with positions.
        player_col: Column with player names.
        confidence_threshold: Coefficient of variation threshold above which
            the recommendation is "uncertain" (abstain). Default 0.3 means
            if std > 30% of prediction, abstain.
        min_projection: Minimum projected points to recommend starting.

    Returns:
        DataFrame with start/sit/uncertain recommendations.
    """
    df = predictions.copy()

    # Coefficient of variation (normalized uncertainty)
    if std_col in df.columns:
        df["coeff_of_variation"] = df[std_col] / np.maximum(df[pred_col], 1.0)
        df["confidence"] = np.where(
            df["coeff_of_variation"] <= confidence_threshold,
            "high",
            np.where(
                df["coeff_of_variation"] <= confidence_threshold * 1.5,
                "medium",
                "low",
            ),
        )
    else:
        df["coeff_of_variation"] = np.nan
        df["confidence"] = "unknown"

    # Recommendation logic
    def _recommend(row):
        pred = row[pred_col]
        confidence = row.get("confidence", "unknown")

        # Abstain when confidence is low
        if confidence == "low":
            return "uncertain"

        # Start/sit based on projection vs floor
        if pred >= min_projection:
            return "start"
        else:
            return "sit"

    df["recommendation"] = df.apply(_recommend, axis=1)

    # Boom/bust risk indicators
    if std_col in df.columns:
        from scipy.stats import norm

        std_safe = np.maximum(df[std_col].values, 0.01)
        df["boom_probability"] = 1.0 - norm.cdf(
            20.0, loc=df[pred_col].values, scale=std_safe
        )
        df["bust_probability"] = norm.cdf(
            5.0, loc=df[pred_col].values, scale=std_safe
        )

    result_cols = [
        player_col, position_col, pred_col,
        "recommendation", "confidence", "coeff_of_variation",
    ]
    if "boom_probability" in df.columns:
        result_cols.extend(["boom_probability", "bust_probability"])
    existing = [c for c in result_cols if c in df.columns]

    return df[existing].sort_values(pred_col, ascending=False)


def evaluate_decision_quality(
    decisions: pd.DataFrame,
    actuals: pd.DataFrame,
    pred_col: str = "predicted_points",
    actual_col: str = "fantasy_points",
    recommendation_col: str = "recommendation",
    player_col: str = "player_name",
) -> Dict[str, float]:
    """Evaluate decision quality separately from prediction quality.

    Per Directive V7 Section 9: separate model quality from policy quality
    to identify whether failure comes from the forecast or the decision rule.

    Args:
        decisions: DataFrame with recommendations.
        actuals: DataFrame with actual outcomes.
        pred_col: Prediction column.
        actual_col: Actual outcome column.
        recommendation_col: Decision column.
        player_col: Player identifier column.

    Returns:
        Dict with decision quality metrics.
    """
    # Merge decisions with actuals
    if player_col in decisions.columns and player_col in actuals.columns:
        merged = decisions.merge(
            actuals[[player_col, actual_col]],
            on=player_col,
            how="inner",
        )
    else:
        merged = decisions.copy()
        if actual_col not in merged.columns:
            return {"error": "Cannot evaluate without actual outcomes"}

    results: Dict[str, float] = {}

    # Start accuracy: what % of "start" recommendations scored above floor?
    starts = merged[merged[recommendation_col] == "start"]
    if len(starts) > 0:
        results["start_hit_rate"] = float(
            (starts[actual_col] >= 5.0).mean()
        )
        results["start_avg_points"] = float(starts[actual_col].mean())
        results["n_starts"] = len(starts)

    # Sit accuracy: what % of "sit" recommendations scored below ceiling?
    sits = merged[merged[recommendation_col] == "sit"]
    if len(sits) > 0:
        results["sit_correct_rate"] = float(
            (sits[actual_col] < 10.0).mean()
        )
        results["sit_avg_points"] = float(sits[actual_col].mean())
        results["n_sits"] = len(sits)

    # Abstention analysis: what would have happened if we followed "uncertain"?
    uncertain = merged[merged[recommendation_col] == "uncertain"]
    if len(uncertain) > 0:
        results["uncertain_avg_points"] = float(uncertain[actual_col].mean())
        results["uncertain_variance"] = float(uncertain[actual_col].var())
        results["n_uncertain"] = len(uncertain)
        # High variance in "uncertain" bucket validates the abstention policy
        results["abstention_justified"] = bool(
            results["uncertain_variance"]
            > merged[actual_col].var() * 1.2
        )

    # Overall decision value: did "start" players outscore "sit" players?
    if len(starts) > 0 and len(sits) > 0:
        results["start_vs_sit_differential"] = float(
            starts[actual_col].mean() - sits[actual_col].mean()
        )
        # Positive differential means the decision rule is adding value
        results["decision_adds_value"] = bool(
            results["start_vs_sit_differential"] > 0
        )

    # Top-K precision: among top-K recommended players, how many were actually top-K?
    for k in [10, 20]:
        if len(merged) >= k:
            top_k_pred = set(
                merged.nlargest(k, pred_col)[player_col].values
            )
            top_k_actual = set(
                merged.nlargest(k, actual_col)[player_col].values
            )
            results[f"top_{k}_precision"] = float(
                len(top_k_pred & top_k_actual) / k
            )

    return results


def waiver_wire_priority(
    predictions: pd.DataFrame,
    rostered_players: Optional[List[str]] = None,
    pred_col: str = "predicted_points",
    std_col: str = "predicted_std",
    position_col: str = "position",
    player_col: str = "player_name",
) -> pd.DataFrame:
    """Score waiver wire pickups by upside potential and VOR.

    Prioritizes players who are:
    1. Unrostered (available on waivers)
    2. Projected above replacement level
    3. High upside (positive skew in prediction distribution)

    Args:
        predictions: DataFrame with predictions.
        rostered_players: List of already-rostered player names to exclude.
        pred_col: Prediction column.
        std_col: Std deviation column.
        position_col: Position column.
        player_col: Player name column.

    Returns:
        DataFrame with waiver priority scores, sorted by priority.
    """
    df = predictions.copy()

    # Filter to unrostered players
    if rostered_players:
        df = df[~df[player_col].isin(rostered_players)]

    # Compute VOR
    df["replacement_level"] = df[position_col].map(DEFAULT_REPLACEMENT_LEVEL).fillna(7.0)
    df["vor"] = df[pred_col] - df["replacement_level"]

    # Upside score: prediction + 1 std (optimistic scenario)
    if std_col in df.columns:
        df["upside_score"] = df[pred_col] + df[std_col]
    else:
        df["upside_score"] = df[pred_col] * 1.2  # 20% upside assumption

    # Priority = weighted combination of VOR and upside
    df["waiver_priority"] = 0.6 * df["vor"] + 0.4 * df["upside_score"]
    df["waiver_rank"] = (
        df["waiver_priority"].rank(ascending=False, method="min").astype(int)
    )

    result_cols = [
        player_col, position_col, pred_col, "vor",
        "upside_score", "waiver_priority", "waiver_rank",
    ]
    existing = [c for c in result_cols if c in df.columns]
    return df[existing].sort_values("waiver_rank").head(50)
