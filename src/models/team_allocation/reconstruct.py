"""Plan A reconstruction: renormalized share x lagged team total -> a
PARTIAL fantasy-points reconstruction.

IMPORTANT SCOPE LIMITATION (discovered while implementing this): Plan A's
four share targets (targets, rushing_attempts, receiving_yards,
rushing_yards) do not cover touchdowns, receptions, or passing production.
`calculate_fantasy_points_df` (src/utils/helpers.py, this repo's single
source of truth for PPR scoring) contributes 0 for any column not present
in its input, so reconstructing points from only rushing_yards/
receiving_yards necessarily yields a YARDAGE-ONLY PARTIAL fantasy-points
number, not the real target. `targets` and `rushing_attempts` don't carry
their own entry in config.settings.SCORING at all (attempts/targets aren't
scored directly) -- they remain useful model inputs (volume is a leading
indicator of receiving-yardage/reception opportunity) but are not part of
the points reconstruction here.

Consequence: Plan A's originally-stated acceptance criterion 2
("reconstructed fantasy-point MAE must beat the existing per-position
production model") cannot be evaluated as written until touchdown and
reception share targets exist -- this reconstruction's output is only ever
the yardage slice of a player's real PPR total, so comparing its MAE
against the production model's full-PPR MAE would not be apples-to-apples.
See docs/TEAM_LEVEL_ALLOCATION_MODELS.md's Plan A section for the revised
criterion this implies.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.utils.helpers import calculate_fantasy_points_df

# Only these two volume targets have a direct entry in config.settings.SCORING
# -- targets/rushing_attempts don't score points on their own and are
# excluded from the points reconstruction (see module docstring).
POINTS_ELIGIBLE_VOLUME_COLS = ["rushing_yards", "receiving_yards"]


def renormalize_shares(predicted_share: np.ndarray, group_keys: pd.DataFrame) -> np.ndarray:
    """Renormalize predicted shares so every (team, season, week) group
    sums to 1. Raw independent per-player regressor outputs have no reason
    to sum to 1 on their own -- see docs/TEAM_LEVEL_ALLOCATION_MODELS.md's
    Plan A section.

    `group_keys` must have columns team/season/week, same row order and
    length as `predicted_share`.

    Degenerate case: if every player on a team-week has a (near-)zero
    predicted share, the group sum is ~0 and dividing would produce NaN/inf
    -- fall back to an equal split across that group's rows instead. This
    is a symmetric, no-information fallback (never favors one player over
    another), used only when the model itself produced no usable signal
    for that team-week; it is not intended to be common and is worth
    watching for in the coverage report if it fires often on real data.
    """
    predicted_share = np.asarray(predicted_share, dtype=float)
    if len(predicted_share) != len(group_keys):
        raise ValueError("predicted_share and group_keys must have matching length")

    df = group_keys[["team", "season", "week"]].reset_index(drop=True).copy()
    # Negative predictions are nonsensical for a share and would otherwise
    # let one player's negative prediction inflate everyone else's
    # renormalized share -- clip before summing, not after.
    df["_pred"] = np.clip(predicted_share, a_min=0.0, a_max=None)

    group = df.groupby(["team", "season", "week"])
    group_sum = group["_pred"].transform("sum")
    group_size = group["_pred"].transform("size")

    degenerate = group_sum <= 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        normal_case = df["_pred"] / group_sum
    return np.where(degenerate, 1.0 / group_size, normal_case).astype(float)


def reconstruct_volume(renormalized_share: np.ndarray, predicted_team_total: np.ndarray) -> np.ndarray:
    """predicted_volume = renormalized_share x predicted_team_total.

    NaN team total (e.g. a team's first tracked weeks, no lag history yet
    -- see scripts/build_team_week_player_shares.py's `_lagged_team_totals`)
    propagates to NaN rather than being silently zero-filled, matching this
    repo's standing rule against inventing a value where there is no
    evidence yet.
    """
    return np.asarray(renormalized_share, dtype=float) * np.asarray(predicted_team_total, dtype=float)


def reconstruct_partial_fantasy_points(volumes: Dict[str, np.ndarray]) -> pd.Series:
    """PARTIAL fantasy points from reconstructed yardage volumes only --
    see module docstring for why this is not full fantasy points.

    `volumes` must include at least one of POINTS_ELIGIBLE_VOLUME_COLS,
    mapping to equal-length array-likes. Reuses
    src/utils/helpers.py:calculate_fantasy_points_df (this repo's single
    scoring-formula source of truth) rather than re-deriving point values
    here -- a column absent from `volumes` simply contributes 0, the same
    behavior that function already has for any other missing stat.
    """
    eligible = {c: v for c, v in volumes.items() if c in POINTS_ELIGIBLE_VOLUME_COLS}
    if not eligible:
        raise ValueError(
            f"reconstruct_partial_fantasy_points needs at least one of "
            f"{POINTS_ELIGIBLE_VOLUME_COLS}, got keys {list(volumes)}"
        )
    df = pd.DataFrame(eligible)
    points = calculate_fantasy_points_df(df)
    # calculate_fantasy_points_df fillna(0)s each column independently (its
    # contract for a column absent entirely, e.g. passing_yards for a
    # receiving-only reconstruction). Applied here, that would also
    # silently turn a NaN reconstructed volume -- reconstruct_volume's
    # documented signal for "no lagged team total yet, can't reconstruct"
    # -- into a fake 0 rather than propagating the missing-evidence signal.
    return points.where(~df.isna().any(axis=1), np.nan)
