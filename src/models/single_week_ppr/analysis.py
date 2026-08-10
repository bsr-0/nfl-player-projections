"""Phase 4 (next_focus.md) validation breakdowns: overall / position / season
/ predicted-score bucket / player tier, plus quantile calibration.

Consumes the row-level predictions produced by
evaluate.py:run_final_validation (data/experiments/
phase4_row_level_predictions.csv by default).
"""
from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from src.models.single_week_ppr.evaluate import compute_metrics, compute_quantile_metrics
from src.models.single_week_ppr.tiers import assign_player_tier, assign_score_bucket, compute_prior_season_ppg


def _grouped_metrics(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for keys, g in df.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update(compute_metrics(g["actual_ppr"], g["prediction"]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Overall + by-position + by-season breakdowns — next_focus.md Phase 4's
    "evaluate overall and by position/season" is nearly free once row-level
    data exists (no new logic beyond groupby)."""
    return {
        "overall": _grouped_metrics(df, ["model"]),
        "by_position": _grouped_metrics(df, ["position", "model"]),
        "by_season": _grouped_metrics(df, ["season", "model"]),
    }


def compute_bucket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Buckets on the model's own weekly PREDICTION — is error uniform
    across the model's predicted range, or worse for predicted big games?
    """
    d = df.copy()
    d["bucket"] = d["prediction"].apply(assign_score_bucket)
    return _grouped_metrics(d, ["position", "bucket", "model"])


def _load_full_history_for_tiers(positions: Sequence[str]) -> pd.DataFrame:
    from src.utils.database import DatabaseManager

    frames = []
    for pos in positions:
        d = DatabaseManager().get_all_players_for_training(position=pos)
        frames.append(d[["player_id", "season", "position", "fantasy_points"]])
    return pd.concat(frames, ignore_index=True)


def compute_tier_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Buckets on the PLAYER's prior-season PPG (leakage-safe — never uses
    the season being evaluated). Loads full player history from the DB
    (not just the row-level CSV, which only has test-season rows and can't
    supply an earlier season's PPG on its own).
    """
    positions = df["position"].unique().tolist()
    history = _load_full_history_for_tiers(positions)
    history_with_prior = compute_prior_season_ppg(history)
    prior_lookup = (
        history_with_prior[["player_id", "season", "prior_season_ppg"]]
        .drop_duplicates(subset=["player_id", "season"])
    )

    merged = df.merge(
        prior_lookup, left_on=["player", "season"], right_on=["player_id", "season"], how="left",
    )
    merged["tier"] = [
        assign_player_tier(ppg, pos) for ppg, pos in zip(merged["prior_season_ppg"], merged["position"])
    ]
    return _grouped_metrics(merged, ["position", "tier", "model"])


def compute_calibration_report(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile pinball loss + coverage, by position and season, for the
    E_quantile_gbm rows only."""
    qdf = df[df["model"] == "E_quantile_gbm"]
    rows: List[dict] = []
    for (position, season), g in qdf.groupby(["position", "season"]):
        preds = g[["p25", "p50", "p75", "p90"]].reset_index(drop=True)
        actual = g["actual_ppr"].reset_index(drop=True)
        metrics = compute_quantile_metrics(actual, preds)
        rows.append({"position": position, "season": season, "n": len(g), **metrics})
    return pd.DataFrame(rows)
