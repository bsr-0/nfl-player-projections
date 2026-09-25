"""Row-level out-of-fold predictions from walk-forward validation.

No persisted row-level prediction artifact existed for the served weekly
model. `position_models.py` computes OOF predictions internally for
meta-learner stacking and isotonic calibration but keeps only aggregate
metrics; `train.py --walk-forward` reported four numbers per position and
discarded everything else. Two separate pieces of work were blocked on this:

  * Segment evaluation -- by position x tier, and returning players vs
    cold-start players with no prior starts. Aggregate MAE cannot tell you
    whether a change helps starters and hurts cold-start players equally.
  * The correlation layer (docs/GAME_SIMULATION_CORRELATION_PLAN.md, Phase
    2), whose stated design fits same-game residual covariance "from
    out-of-fold player predictions" and so had nothing to fit on.

Each walk-forward fold trains on seasons 1..N-1 and predicts season N, so a
fold's test-set predictions are genuinely out-of-fold for that season.
Concatenating the folds gives an OOF panel covering the validated seasons,
with no row ever predicted by a model that saw it.

The leakage guarantee is enforced, not assumed: `capture_fold_rows` raises if
a fold's test season appears among its training seasons, or if the frame
carries rows from any other season. A silently contaminated panel would make
every downstream segment number look better than it is.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Identity, context, and the (predicted, actual) pair. `opponent` and `team`
# are kept because the correlation layer groups residuals by (season, week,
# team) to build same-game vectors.
IDENTITY_COLUMNS = ("player_id", "season", "week", "team", "opponent", "position")
PREDICTION_COLUMN = "predicted_points"
ACTUAL_COLUMN = "actual_for_backtest"

OOF_PANEL_FILENAME = "walk_forward_oof_predictions.parquet"


class OOFLeakageError(RuntimeError):
    """A fold's captured rows are not out-of-fold."""


def capture_fold_rows(
    test_data: pd.DataFrame,
    *,
    train_seasons: Sequence[int],
    test_season: int,
) -> pd.DataFrame:
    """Extract one fold's out-of-fold rows.

    Raises OOFLeakageError if the fold's test season is also a training
    season, or if `test_data` contains rows outside that season -- either
    would mean the "held-out" predictions were partly in-sample.
    """
    if test_season in set(int(s) for s in train_seasons):
        raise OOFLeakageError(
            f"test season {test_season} is also a training season; these "
            f"predictions are in-sample, not out-of-fold")

    present = set(pd.to_numeric(test_data["season"], errors="coerce").dropna().astype(int))
    if present - {int(test_season)}:
        raise OOFLeakageError(
            f"fold for {test_season} carries rows from {sorted(present)}; "
            "a fold's test frame must hold only its held-out season")

    missing = [c for c in IDENTITY_COLUMNS if c not in test_data.columns]
    if missing:
        raise ValueError(f"test_data is missing identity columns: {missing}")
    for column in (PREDICTION_COLUMN, ACTUAL_COLUMN):
        if column not in test_data.columns:
            raise ValueError(f"test_data is missing {column!r}")

    rows = test_data.loc[:, [*IDENTITY_COLUMNS, PREDICTION_COLUMN, ACTUAL_COLUMN]].copy()
    rows = rows.rename(columns={ACTUAL_COLUMN: "actual_points"})

    # A row the model could not score is not evidence about the model. Drop
    # it here rather than letting a NaN-filled zero masquerade as a
    # prediction downstream -- the same silent-zero-fill defect this repo
    # has hit before in reconstruct_partial_fantasy_points.
    before = len(rows)
    rows = rows[rows[PREDICTION_COLUMN].notna() & rows["actual_points"].notna()]
    dropped = before - len(rows)
    if dropped:
        logger.info("fold %s: dropped %d/%d rows lacking a prediction or actual",
                    test_season, dropped, before)

    rows["train_seasons"] = ",".join(str(int(s)) for s in sorted(train_seasons))
    rows["n_train_seasons"] = len(set(int(s) for s in train_seasons))
    rows["residual"] = rows[PREDICTION_COLUMN] - rows["actual_points"]
    return rows.reset_index(drop=True)


def add_experience_segments(panel: pd.DataFrame) -> pd.DataFrame:
    """Label each row by the player's prior experience *within the panel*.

    `prior_weeks_in_panel` counts that player's earlier appearances in the
    OOF panel, ordered by (season, week), computed with a strict shift so a
    row never counts itself. `is_cold_start` marks a player's first appearance
    -- the "new players without prior starts" split that aggregate MAE hides.

    This is deliberately panel-relative, not career-relative: it answers "how
    much had this model seen of this player by this point in the validation",
    which is the question segment evaluation is actually asking. A
    career-relative version would need the full pre-panel history and is a
    different metric; do not conflate them.
    """
    out = panel.sort_values(["player_id", "season", "week"]).copy()
    out["prior_weeks_in_panel"] = out.groupby("player_id").cumcount()
    out["is_cold_start"] = out["prior_weeks_in_panel"] == 0
    return out.reset_index(drop=True)


def build_panel(fold_rows: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate captured folds into one OOF panel."""
    frames = [f for f in fold_rows if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=[*IDENTITY_COLUMNS, PREDICTION_COLUMN,
                                     "actual_points", "residual"])
    panel = pd.concat(frames, ignore_index=True)

    duplicated = panel.duplicated(subset=["player_id", "season", "week"], keep=False)
    if duplicated.any():
        # Two folds predicting the same player-week means overlapping test
        # seasons: the panel would double-count those rows and silently
        # reweight every aggregate computed from it.
        offenders = panel.loc[duplicated, ["player_id", "season", "week"]].head(5)
        raise OOFLeakageError(
            f"{int(duplicated.sum())} player-weeks appear in more than one "
            f"fold; test seasons must not overlap. First few:\n{offenders}")

    return add_experience_segments(panel)


def write_panel(panel: pd.DataFrame, path: Path) -> Path:
    """Persist the panel, atomically."""
    from src.utils.atomic_io import atomic_write_parquet
    return atomic_write_parquet(panel, path)


def segment_report(panel: pd.DataFrame, *, by: Sequence[str] = ("position",)) -> pd.DataFrame:
    """MAE / RMSE / bias / n by segment.

    Bias (mean signed residual) is reported alongside the error magnitudes
    because a change can leave MAE flat while shifting the whole
    distribution -- which is exactly what aggregate reporting hides.
    """
    if panel.empty:
        return pd.DataFrame(columns=[*by, "n", "mae", "rmse", "bias"])
    grouped = panel.groupby(list(by), dropna=False)
    report = grouped.agg(
        n=("residual", "size"),
        mae=("residual", lambda r: float(np.abs(r).mean())),
        rmse=("residual", lambda r: float(np.sqrt(np.mean(np.square(r))))),
        bias=("residual", "mean"),
    ).reset_index()
    return report.sort_values(list(by)).reset_index(drop=True)
