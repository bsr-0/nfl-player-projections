"""Step 8 season projection as a production model: E[games] x E[PPR/game].

Extracted from `_step8_arm` in scripts/walk_forward_preseason.py, which was an
EVALUATION harness -- it required the target-season actuals to exist, so it
could not project a season that had not happened yet.

Why this arm: on the corrected 11-fold walk-forward (2026-08-28) it ranks first
of four, and it is best on rookies specifically.

    mean MAE rank   step8 1.25 | candidate 1.75 | phase7 3.00 | production 4.00
    rookies (n=1012) step8 39.81 | candidate 40.56 | phase7 44.96 | production 53.29

The arm the UI currently ships (`PreseasonProjector`, the "production" arm)
is last on both.

### The two halves

    season PPR = E[games played] x E[PPR per game | played]

Kept as a product rather than a single season-total regression because a
season total alone cannot separate an exposure failure (missed games) from a
conditional-production failure (played badly). `decompose_errors` reports that
split and is why the arm exists in this shape.

### Cold start is already handled, and is this arm's best case

No special rookie path is needed here. `build_multiyear_season_pairs` emits
cold-start rows (`_cold_start_rows`) with the `*_y1/_y2/_y3` lags left NaN and
draft/combine/college populated; the production half is a LightGBM regressor
which routes those NaN natively; and the exposure half falls back to the
position-mean rate at zero weight on player history. Measured, step8 is BETTER
on rookies (39.81) than on veterans (45.96).

### What actually had to change for production

`attach_conditional_target` inner-joins the panel to pick up the target AND
`possible_games`. The panel row for a season only exists once that season has
been played, so on an upcoming season the join drops every row. Inference
therefore takes `possible_games` from the SCHEDULE -- which is what it is,
"the published schedule length, known pre-season" -- rather than from the
panel. `possible_games_for_players` does that, era-aware (17-game seasons
through 2020, 18 from 2021).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class Step8SeasonModel:
    """Fit both halves on history; project a season with no target present."""

    def __init__(self) -> None:
        self.production_model = None
        self.availability = None
        self.features: List[str] = []
        self.fitted_before_season: Optional[int] = None

    # -- fit ---------------------------------------------------------------
    def fit(self, train_pairs: pd.DataFrame, panel: pd.DataFrame,
            before_season: int) -> "Step8SeasonModel":
        """Both halves, strictly on seasons before `before_season`.

        Causality is structural, not conventional: SeasonAvailabilityEstimator
        .fit() drops anything at or after `before_season` itself, so a caller
        cannot accidentally train the exposure half on the season being
        projected.
        """
        from src.models.single_week_ppr.season_conditional_production import (
            TARGET, attach_conditional_target, feature_columns,
            fit_conditional_production,
        )
        from src.models.single_week_ppr.season_availability import SeasonAvailabilityEstimator

        train = attach_conditional_target(train_pairs, panel).dropna(subset=[TARGET])
        if train.empty:
            raise ValueError("no training rows with a conditional target")

        self.features = feature_columns(train)
        self.production_model = fit_conditional_production(train, self.features)
        self.availability = SeasonAvailabilityEstimator().fit(panel, before_season=before_season)
        self.fitted_before_season = int(before_season)
        return self

    # -- predict -----------------------------------------------------------
    def predict(self, pairs: pd.DataFrame,
                possible_games: Optional[pd.Series] = None) -> pd.Series:
        """Projected season PPR. Requires NO target-season data.

        `possible_games` must be supplied when `pairs` has no such column --
        i.e. whenever projecting a season that has not been played. Use
        `possible_games_for_players`.
        """
        from src.models.single_week_ppr.season_conditional_production import (
            predict_conditional_production,
        )
        if self.production_model is None or self.availability is None:
            raise RuntimeError("call fit() before predict()")

        # Validate inputs BEFORE doing any work, so a bad call fails with the
        # explanation rather than with whatever the regressor raises first.
        if possible_games is None:
            if "possible_games" not in pairs.columns:
                raise ValueError(
                    "possible_games not in `pairs` and none supplied. For an "
                    "unplayed season it cannot come from the panel -- use "
                    "possible_games_for_players().")
            possible_games = pairs["possible_games"]

        missing = [c for c in self.features if c not in pairs.columns]
        if missing:
            # Not silently tolerated: a feature absent at inference that was
            # present at fit means the two frames were built by different
            # paths, which is the failure mode that produced a silently
            # weaker model elsewhere in this codebase.
            raise ValueError(
                f"{len(missing)} fitted feature(s) missing at predict time, "
                f"e.g. {missing[:5]}. Build inference pairs with the same "
                f"function used for training.")
        feats = [c for c in self.features if c in pairs.columns]

        rate = predict_conditional_production(self.production_model, pairs, feats)
        games = self.availability.predict_rate(pairs) * np.asarray(possible_games, dtype=float)
        return pd.Series(np.asarray(games) * np.asarray(rate), index=pairs.index)

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"production_model": self.production_model,
                     "availability": self.availability,
                     "features": self.features,
                     "fitted_before_season": self.fitted_before_season}, path)

    @classmethod
    def load(cls, path: Path) -> "Step8SeasonModel":
        import joblib
        d = joblib.load(Path(path))
        m = cls()
        m.production_model = d["production_model"]
        m.availability = d["availability"]
        m.features = d["features"]
        m.fitted_before_season = d.get("fitted_before_season")
        return m


def possible_games_for_players(players: pd.DataFrame, season: int,
                               team_col: str = "dest_team") -> pd.Series:
    """Scheduled games per player for `season`, from the SCHEDULE.

    The panel's `possible_games` is only available after a season is played.
    This is the pre-season equivalent and is era-aware via
    `regular_season_max_week` -- 16 games through 2020, 17 from 2021 -- so it
    cannot re-introduce the wild-card week that the flat-18 cap admitted into
    every pre-2021 season total (GAPS.md 2026-08-25).

    Falls back to the league-typical count for a player with no resolvable
    team, rather than dropping them.
    """
    import sqlite3
    from config.settings import DB_PATH, regular_season_max_week

    max_week = regular_season_max_week(season)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        sched = pd.read_sql(
            "SELECT season, week, home_team, away_team FROM schedule WHERE season = ?",
            conn, params=[int(season)])
    finally:
        conn.close()

    sched = sched[pd.to_numeric(sched["week"], errors="coerce") <= max_week]
    counts = pd.concat([sched["home_team"], sched["away_team"]]).value_counts()
    default = int(counts.median()) if len(counts) else max_week - 1

    teams = (players[team_col] if team_col in players.columns
             else pd.Series(index=players.index, dtype=object))
    return teams.map(counts).fillna(default).astype(float)
