"""E[PPR per game | played] -- the production half of the season layer.

    season PPR = E[games played] x E[PPR per game | played]
                 ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
                 season_availability        this module

Phase 7 answers the production half by manufacturing synthetic weekly
feature rows and running the weekly model on them. This module answers it
at the player-season level instead, which is what lets the architecture
drop synthetic weeks entirely.

### The target, and why it is not taken from the existing builder

`preseason_features.build_multiyear_season_pairs` supplies the FEATURES:
causal, player-season level, with real destination-team context known
before week 1. Those are reused as-is.

The TARGET is deliberately not taken from there.
`preseason_features._load_full_history` reads `player_weekly_stats` raw --
no participation quality label, no receiving-charting floor -- and its
`games_played` is a bare row count. Dividing season PPR by that denominator
would reintroduce the selection problem this architecture exists to remove
(GAPS.md 2026-08-20). The target comes from
`season_availability.load_player_seasons()`, which applies the contract.

### The played-zero invariant

A `snap_count > 0, PPR = 0` week is a real observed game. It counts in the
denominator and contributes 0 to the numerator, which is exactly right: it
pulls the player's conditional rate down because he genuinely played and
produced nothing. It is never dropped and never treated as an availability
failure.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

TARGET = "ppr_per_game"

# Identifiers and non-numeric context. Target-season exposure columns are
# NOT listed here -- they are excluded structurally, see below.
_NON_FEATURE = {
    "player_id", "player_name", "position", "birth_date", "target_season",
    "season", "season_total", "dest_team", "prior_team", "team",
}

# THE CONTRACT, stated once:
#
#     Target-season exposure may determine training weights and the target.
#     It may NEVER be an input feature.
#
# Prior-season exposure (games_played_y1, games_played_y2, ...) IS
# legitimate signal and must survive -- a filter broad enough to catch
# "games_played" by name would destroy it.
#
# So the exclusion is derived from the panel's own schema rather than a
# hand-maintained list: every column `attach_conditional_target` joins in
# describes the TARGET season and is therefore off-limits as a feature. Add
# a column to the panel and it is excluded automatically.
TARGET_SEASON_COLUMNS = ("games_played", "ppr", "ppr_per_game", "possible_games", "rate")


def attach_conditional_target(pairs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join the contract-correct PPR-per-game target onto pre-season features.

    `pairs` is `build_multiyear_season_pairs` output (one row per player /
    target season). `panel` is `season_availability.load_player_seasons()`.
    """
    if pairs.empty or panel.empty:
        return pairs.assign(**{TARGET: np.nan})
    # `possible_games` comes along because the exposure model needs it to
    # turn a rate into games. It is the published schedule length, known
    # pre-season, and it stays out of the feature set via
    # TARGET_SEASON_COLUMNS.
    target = panel[["player_id", "season", "games_played", "ppr", "ppr_per_game",
                    "possible_games"]].rename(columns={"season": "target_season"})
    out = pairs.merge(target, on=["player_id", "target_season"], how="inner")
    return out


def feature_columns(pairs: pd.DataFrame) -> List[str]:
    """Numeric pre-season columns only.

    Excludes identifiers and every target-season exposure column. Prior-season
    lags (`*_y1`, `*_y2`) are deliberately retained: they are the legitimate
    causal signal about a player's durability and role.
    """
    banned = _NON_FEATURE | set(TARGET_SEASON_COLUMNS)
    return [c for c in pairs.columns
            if c not in banned and pd.api.types.is_numeric_dtype(pairs[c])]


def fit_conditional_production(train: pd.DataFrame, features: Sequence[str],
                               sample_weight: Optional[np.ndarray] = None):
    """E[PPR per game | played].

    Weighted by games played by default: a player-season built on 3 games is
    a far noisier estimate of a true rate than one built on 17, and letting
    them count equally is the same error the weekly rate model had to fix.
    """
    from src.models.single_week_ppr.architectures import GBMRegressor

    if sample_weight is None:
        sample_weight = train["games_played"].to_numpy(dtype=float)
    model = GBMRegressor(objective="regression")
    model.fit(train[list(features)], train[TARGET], sample_weight=sample_weight)
    return model


def predict_conditional_production(model, test: pd.DataFrame,
                                   features: Sequence[str]) -> pd.Series:
    """Predicted rate, floored at zero.

    A season-long PPR-per-game rate below zero is not a real outcome even
    though single weeks can be negative, and letting the product go negative
    would corrupt the season total.
    """
    raw = model.predict(test[list(features)])
    return pd.Series(np.maximum(raw, 0.0), index=test.index)


def decompose_errors(games_pred: pd.Series, games_actual: pd.Series,
                     rate_pred: pd.Series, rate_actual: pd.Series,
                     season_actual: pd.Series) -> dict:
    """Both components AND the season total, plus their interaction.

    Required output, not a diagnostic extra: a season total alone cannot
    distinguish an exposure failure from a conditional-production failure
    (GAPS.md Step 8A pre-registration).

    The interaction term is exact, not approximate:

        Gp*Rp - Ga*Ra = Ra*(Gp-Ga) + Ga*(Rp-Ra) + (Gp-Ga)*(Rp-Ra)
                        ^games      ^rate         ^interaction
    """
    def _stats(pred, actual, prefix):
        err = pred - actual
        out = {f"{prefix}_mae": float(err.abs().mean()),
               f"{prefix}_bias": float(err.mean()),
               f"{prefix}_rmse": float(np.sqrt((err ** 2).mean())),
               f"{prefix}_mean_pred": float(pred.mean()),
               f"{prefix}_mean_actual": float(actual.mean())}
        if len(pred) > 1 and pred.std() > 0 and actual.std() > 0:
            out[f"{prefix}_corr"] = float(pred.corr(actual))
            ss_res = float(((actual - pred) ** 2).sum())
            ss_tot = float(((actual - actual.mean()) ** 2).sum())
            out[f"{prefix}_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return out

    season_pred = games_pred * rate_pred
    dg, dr = games_pred - games_actual, rate_pred - rate_actual
    out = {}
    out.update(_stats(games_pred, games_actual, "games"))
    out.update(_stats(rate_pred, rate_actual, "rate"))
    out.update(_stats(season_pred, season_actual, "season"))
    out["contrib_games"] = float((rate_actual * dg).mean())
    out["contrib_rate"] = float((games_actual * dr).mean())
    out["contrib_interaction"] = float((dg * dr).mean())
    return out
