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
which routes those NaN natively; and the exposure half falls back to its
shrinkage target at zero weight on player history. Measured, step8 is BETTER
on rookies (39.81) than on veterans (45.96).

That 39.81 was measured when the exposure fallback was the flat position mean,
which gave every rookie at a position the same games estimate. The target is
now conditioned on draft round (see `season_availability`), and `pairs`
carries `draft_round`, so this path picks it up automatically. The number
above therefore predates the change and has NOT been re-measured.

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


# ---------------------------------------------------------------------------
# Board metadata: confidence_score / support_class
# ---------------------------------------------------------------------------
# scripts/generate_draft_data.py sizes the board's asymmetric floor/ceiling
# bands from PreseasonProjector.predict_with_details()'s confidence_score and
# support_class. Swapping the projection model without supplying these would
# silently break the uncertainty bands rather than error.
#
# The thresholds and weights below are COPIED EXACTLY from
# PreseasonProjector._assign_support_class / _prepare_feature_frame, not
# re-derived. That is deliberate: FLOOR_CEILING_COEF in generate_draft_data.py
# was regression-FIT against that confidence_score's distribution
# (coefficients like confidence_score: 0.072483), so changing the formula
# would silently miscalibrate coefficients nobody would think to refit.
# Same inputs, same scale, same coefficients stay valid.
#
# The inputs are all PRIOR-season aggregates, which step8's pairs frame already
# carries as `*_y1`. This is a column mapping, not a new model.

SUPPORT_CLASS_ORDER = ("starter", "committee", "backup", "rotational")

_Y1 = {
    "snap_share": "snap_share_y1",
    "carries_pg": "carries_pg_y1",
    "targets_pg": "targets_pg_y1",
    "passing_yards_pg": "passing_yards_pg_y1",
    "ppg": "ppg_y1",
    "games_played": "games_played_y1",
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Prior-season column as float, NaN-safe, 0.0 when absent entirely.

    `snap_share` is clipped to [0, 1] to match
    PreseasonProjector._prepare_feature_frame line ~271, which does the same
    before computing support_class and confidence_score.

    That clip is arguably wrong at source -- snap_share is stored 0-100
    (median ~40), so clipping to 1.0 collapses almost every rostered player to
    the same value and discards the signal. It is replicated here ANYWAY and
    deliberately: FLOOR_CEILING_COEF in generate_draft_data.py was
    regression-fit against the resulting distribution, so "fixing" the clip
    here would silently shift confidence_score under coefficients nobody would
    think to refit. Without it, measured, veterans sat at 0.927 mean against
    the reference 0.511 -- a 0.42 shift straight into the board's bands.
    Logged in GAPS.md as a separate issue to fix at source, with the
    coefficients refit at the same time.
    """
    src = _Y1.get(col, col)
    if src not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    out = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
    if col == "snap_share":
        out = out.clip(0.0, 1.0)
    return out


def assign_support_class(pairs: pd.DataFrame) -> pd.Series:
    """starter / committee / backup / rotational, from prior-season usage.

    Thresholds identical to PreseasonProjector._assign_support_class.

    Cold-start rows have every `*_y1` NaN, which `_num` reads as 0.0, so a
    rookie falls to "backup" on the low-usage branches. That is the honest
    answer from THIS signal -- it says "no prior usage evidence", not "bad
    player". Draft capital carries the rookie signal and is already in the
    model's own features; it deliberately does not leak into this label.
    """
    position = pairs["position"].fillna("") if "position" in pairs.columns else pd.Series("", index=pairs.index)
    snap, carries = _num(pairs, "snap_share"), _num(pairs, "carries_pg")
    targets, passing = _num(pairs, "targets_pg"), _num(pairs, "passing_yards_pg")
    ppg = _num(pairs, "ppg")

    support = pd.Series("rotational", index=pairs.index, dtype=object)

    qb_starter = (position == "QB") & ((passing >= 220.0) | (ppg >= 16.0))
    qb_backup = (position == "QB") & (passing < 150.0) & (ppg < 12.0)
    rb_starter = (position == "RB") & ((carries >= 15.0) | (snap >= 0.60))
    rb_committee = (position == "RB") & ~rb_starter & (
        (carries >= 8.0) | (targets >= 3.0) | (snap >= 0.38))
    rb_backup = (position == "RB") & (carries < 6.0) & (snap < 0.30) & (targets < 2.5)
    wr_starter = (position == "WR") & ((targets >= 7.0) | (snap >= 0.78))
    wr_committee = (position == "WR") & ~wr_starter & ((targets >= 5.0) | (snap >= 0.60))
    wr_backup = (position == "WR") & (targets < 3.5) & (snap < 0.45)
    te_starter = (position == "TE") & ((targets >= 6.0) | (snap >= 0.72))
    te_committee = (position == "TE") & ~te_starter & ((targets >= 4.0) | (snap >= 0.55))
    te_backup = (position == "TE") & (targets < 2.8) & (snap < 0.45)

    support.loc[qb_starter | rb_starter | wr_starter | te_starter] = "starter"
    support.loc[rb_committee | wr_committee | te_committee] = "committee"
    support.loc[qb_backup | rb_backup | wr_backup | te_backup] = "backup"
    return support


def confidence_score(pairs: pd.DataFrame,
                     support: Optional[pd.Series] = None) -> pd.Series:
    """0.05-1.0 confidence in the projection, from prior-season evidence.

    Weights identical to PreseasonProjector._prepare_feature_frame. Low for a
    player with little prior usage -- including rookies, correctly: the board
    should widen their bands, and a rookie's genuine signal (draft capital)
    belongs in the projection, not in the confidence in it.
    """
    position = pairs["position"].fillna("") if "position" in pairs.columns else pd.Series("", index=pairs.index)
    if support is None:
        support = assign_support_class(pairs)

    workload_norm = np.where(
        position.eq("QB"), _num(pairs, "passing_yards_pg").clip(0.0, 300.0) / 300.0,
        np.where(position.eq("RB"), _num(pairs, "carries_pg").clip(0.0, 20.0) / 20.0,
                 _num(pairs, "targets_pg").clip(0.0, 10.0) / 10.0))
    experience_norm = _num(pairs, "years_exp").clip(0.0, 5.0) / 5.0
    support_bonus = (0.20 * (support == "starter").astype(float)
                     + 0.10 * (support == "committee").astype(float)
                     + 0.02 * (support == "rotational").astype(float))
    return pd.Series(np.clip(
        0.30 * (_num(pairs, "games_played").clip(0.0, 17.0) / 17.0)
        + 0.25 * _num(pairs, "snap_share")
        + 0.20 * workload_norm
        + 0.15 * experience_norm
        + support_bonus, 0.05, 1.0), index=pairs.index)


def with_board_metadata(pairs: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Projection plus the columns the draft board needs, in its column names."""
    support = assign_support_class(pairs)
    out = pd.DataFrame({
        "player_id": pairs["player_id"].to_numpy(),
        "predicted_total": np.asarray(predictions, dtype=float),
        "support_class": support.to_numpy(),
        "confidence_score": confidence_score(pairs, support).to_numpy(),
    }, index=pairs.index)
    if "position" in pairs.columns:
        out["position"] = pairs["position"].to_numpy()
    return out
