"""Walk-forward PARTIAL fantasy-points reconstruction for Plan A.

Wires src/models/team_allocation/reconstruct.py (renormalize_shares /
reconstruct_volume / reconstruct_partial_fantasy_points) into the
walk-forward out-of-fold predictions from
src/evaluation/team_share_backtester.py's `walk_forward_oof_predictions`,
producing a per-arm partial fantasy-points MAE -- see
docs/TEAM_LEVEL_ALLOCATION_MODELS.md's revised acceptance criterion 2 for
why this is a YARDAGE-ONLY partial reconstruction, not full PPR.

Only rushing_yards and receiving_yards are reconstructed into points (the
two POINTS_ELIGIBLE_VOLUME_COLS -- targets/rushing_attempts have no direct
scoring entry). rushing_yards' population is every position (QBs scramble);
receiving_yards' population excludes QB. A QB row therefore exists in the
rushing OOF set but not the receiving one -- rather than a gap, this is
correct: a QB really does contribute 0 receiving production, so filling
their missing receiving-share prediction with 0 (never invented via
imputation) reconstructs to 0 receiving yards, matching reality.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.team_share_backtester import (
    _regression_metrics,
    _segment_metrics,
    bootstrap_mae_delta,
    walk_forward_oof_predictions,
)
from src.models.team_allocation.reconstruct import (
    POINTS_ELIGIBLE_VOLUME_COLS,
    reconstruct_partial_fantasy_points,
    reconstruct_volume,
    renormalize_shares,
)

ID_COLS = ["player_id", "season", "week", "team", "position"]


def _reconstruct_for_arm(oof_by_target: Dict[str, pd.DataFrame], arm: str) -> pd.DataFrame:
    """One row per player-week present in either target's OOF set for
    `arm`, with reconstructed + actual partial fantasy points. Outer-joined
    on id columns since rushing_yards' population is a superset of
    receiving_yards' (see module docstring) -- a row present only on the
    rushing side is exactly the QB case above, not a bug.
    """
    frames = []
    for target in POINTS_ELIGIBLE_VOLUME_COLS:
        sub = oof_by_target[target]
        sub = sub[sub["arm"] == arm][ID_COLS + ["predicted_share", "actual_volume", "team_total_roll3"]]
        sub = sub.rename(columns={
            "predicted_share": f"predicted_share__{target}",
            "actual_volume": f"actual__{target}",
            "team_total_roll3": f"team_total__{target}",
        })
        frames.append(sub)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=ID_COLS, how="outer")

    for target in POINTS_ELIGIBLE_VOLUME_COLS:
        for prefix, fill in (("predicted_share__", 0.0), ("actual__", 0.0), ("team_total__", np.nan)):
            col = f"{prefix}{target}"
            if col not in merged.columns:
                merged[col] = fill
            merged[col] = merged[col].fillna(fill)

    group_keys = merged[["team", "season", "week"]]
    reconstructed_volumes = {}
    actual_volumes = {}
    for target in POINTS_ELIGIBLE_VOLUME_COLS:
        renorm = renormalize_shares(merged[f"predicted_share__{target}"].to_numpy(), group_keys)
        reconstructed_volumes[target] = reconstruct_volume(renorm, merged[f"team_total__{target}"].to_numpy())
        actual_volumes[target] = merged[f"actual__{target}"].to_numpy()

    merged["predicted_partial_points"] = reconstruct_partial_fantasy_points(reconstructed_volumes).to_numpy()
    merged["actual_partial_points"] = reconstruct_partial_fantasy_points(actual_volumes).to_numpy()
    return merged


def run_reconstruction_backtest(
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
) -> Dict:
    """Per-arm (ridge/xgboost/rolling3) walk-forward MAE on reconstructed
    PARTIAL fantasy points (rushing_yards + receiving_yards points only --
    see module docstring). `rolling3` here means the reconstruction used
    each player's own rolling-3 share for both targets, i.e. the same
    reconstruction pipeline as the tunable arms, just fed a naive share
    input -- the correct baseline for the revised acceptance criterion 2
    (compare reconstructed-vs-reconstructed, not this against the
    production model's full-PPR MAE, which predicts a different, larger
    target).
    """
    oof_by_target = {
        target: walk_forward_oof_predictions(target, seasons=seasons, n_test_seasons=n_test_seasons)
        for target in POINTS_ELIGIBLE_VOLUME_COLS
    }
    any_oof = next(iter(oof_by_target.values()))
    arms = sorted(any_oof["arm"].unique().tolist()) if not any_oof.empty else []

    per_arm: Dict[str, pd.DataFrame] = {arm: _reconstruct_for_arm(oof_by_target, arm) for arm in arms}

    pooled: Dict[str, Dict] = {}
    for arm, df in per_arm.items():
        y_true = df["actual_partial_points"].to_numpy()
        y_pred = df["predicted_partial_points"].to_numpy()
        position = df["position"].to_numpy()
        n_features = 0 if arm == "rolling3" else 1  # adjusted-R^2 is informational only here
        pooled[arm] = {
            **_regression_metrics(y_true, y_pred, n_features),
            "by_position": _segment_metrics(y_true, y_pred, position, n_features),
        }

    if "rolling3" in per_arm:
        baseline_df = per_arm["rolling3"]
        # Reconstruction can outer-join to slightly different row sets per
        # arm only if a model's own predictions differ in which rows exist
        # -- they don't (every arm is scored on the identical OOF rows), so
        # this alignment is exact, not an approximation.
        for arm, df in per_arm.items():
            if arm == "rolling3" or len(df) != len(baseline_df):
                continue
            pooled[arm]["vs_rolling3_bootstrap"] = bootstrap_mae_delta(
                df["actual_partial_points"].to_numpy(),
                df["predicted_partial_points"].to_numpy(),
                baseline_df["predicted_partial_points"].to_numpy(),
            )

    return {
        "targets_reconstructed": POINTS_ELIGIBLE_VOLUME_COLS,
        "n_rows_total": int(len(next(iter(per_arm.values())))) if per_arm else 0,
        "pooled": pooled,
    }
