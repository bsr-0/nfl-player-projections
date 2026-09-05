"""Expected games played, for the season projection's opportunity term.

    season PPR = E[games played] x E[PPR per game | played]

The weekly model supplies the second factor and is NOT touched by anything
here. That separation is the point: the weekly model answers "how good is
this player when he plays", and this module answers "how often should I
expect him to play". Asking one estimator to do both is what the
2026-08-20 investigation removed.

Not to be confused with `availability.py`, which holds per-WEEK P(plays)
estimators built for the synthetic-week architecture that has since been
abandoned. Those are used only by two experiment scripts. This module is
season-level and is the one wired into projections.

### What this estimator does and does not claim

Validated (GAPS.md 2026-08-20): shrunk historical availability predicts
games played better than a constant in **12/12 position-folds**, by
0.49-1.07 games, and cuts mean |season PPR bias| from 15.02 to 4.42.

It improves season PPR MAE at QB and RB (3/3 folds each) and **not**
consistently at TE/WR (1/3 each). It is adopted for the games component on
the strength of the games-played and bias results, not on a general
season-MAE claim.

**Boundary, which the experiment could not cross:** the evaluation
population is players with >= 1 observed game, because a player with zero
games has no rows to evaluate against. So this answers *"conditional on
this player having a season, how many games does he play"* -- NOT *"will
this player play at all"*. It has not been shown to identify players who
miss a roster, start the season injured, retire, are cut, or are rookies
with no NFL history. Do not read the fallback below as a solution to that.

### The shrinkage target is conditioned on draft round

A player with no history gets `w = 0`, so his estimate is entirely the
target he is shrunk toward. When that target was the position mean, every
rookie at a position got an IDENTICAL number -- a first-round back and a
seventh-round back alike -- which discards the only exposure signal a
week-1 rookie has. The target is now the position x draft-round mean,
itself shrunk toward the position mean by cell size (`DRAFT_BUCKET_K`), so
a thin cell collapses back to exactly the old behaviour rather than
chasing noise.

Measured (2026-09-05, targets 2023-25, 343 cold-start rows), games-played
MAE on cold-start players against the position-mean target it replaced:

    QB 4.42 -> 3.32   RB 5.62 -> 4.75   TE 5.20 -> 3.80   WR 5.10 -> 4.14

4/4 positions, -0.87 to -1.40 games, and it repeats on a disjoint
2018-22 set (4/4 again). Over-prediction of rookie games falls with it
(RB bias +2.01 -> +1.47, WR +1.16 -> +0.77). Players WITH history move
only slightly, in the same direction (WR 3.81 -> 3.72), so this is not
bought from the veterans.

The underlying spread is large and monotonic -- over 2013-24, mean
participation rate runs 0.746 at round 1 down to 0.440 at round 7, with
undrafted at 0.449. Roughly five games between a first-rounder and a
seventh-rounder, all of which the position mean was discarding.

This is still conservative, and the reason is the boundary above: the
panel holds only player-seasons with >= 1 observed game, so a late pick
who never dressed is absent from it entirely. The measured
round-to-round gap is therefore the gap AMONG players who reached the
field, which understates the true one. The conditioning moves the estimate
in the right direction; it still does not answer "will this player play at
all".
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.single_week_ppr.population import (
    RECEIVING_CHARTING_MIN_SEASON, RECEIVING_DEPENDENT_POSITIONS, label_participation,
)
from src.models.preseason_features import UNDRAFTED_ROUND
from config.settings import regular_season_max_week, regular_season_week_sql

# Weight on the player's own history is n_prior_games / (n_prior_games + K).
# At K=16 (roughly one season), a player with one prior season gets ~50%
# weight and a five-year veteran ~84%. The raw per-player rate over-corrects
# (WR season bias +12.36 vs +7.85 shrunk), which is why this exists.
SHRINKAGE_K = 16.0

# Weight on a (position, draft round) cell's own mean is n / (n + K), n counted
# in player-seasons.
#
# Swept, not guessed. An earlier revision set this to 100 from a noise argument
# alone and said so; measured, 100 was far too much shrinkage. Cold-start games
# MAE falls monotonically as K drops, on two disjoint season sets:
#
#   K                 5      10      25      50     100     200   (position mean)
#   2023-25 WR     4.110   4.117   4.139   4.172   4.231   4.326        5.104
#   2018-22 WR     4.196   4.204   4.228   4.266   4.332      --        5.125
#
# 25 rather than 5: the curve is flat below 25 (WR moves 0.03 games between 25
# and 5, and the ordering of the two is not stable across positions), while a
# smaller K keeps stripping the thin-cell protection that makes this safe --
# at K=25 a 25-row cell still gets only half its own weight, at K=5 it gets 83%.
# Buying 0.03 games with that guard is a bad trade. Re-sweep with
# `run_season_availability_experiment.py --draft-bucket-k A B C`.
DRAFT_BUCKET_K = 25.0


def draft_bucket(draft_round) -> pd.Series:
    """Draft round as a prior bucket: 1-7, or UNDRAFTED_ROUND for everything else.

    A missing round reads as undrafted, matching
    `build_multiyear_season_pairs`, which fills the same NaN with
    UNDRAFTED_ROUND. That is not a guess: `draft_picks_v2` has no row at all
    for an undrafted player, so within this data "no draft row" and
    "undrafted" are the same observation. Out-of-range rounds (the 8-12 round
    era, or a bad value) land in the same bucket rather than opening cells
    that modern classes can never populate.
    """
    r = pd.to_numeric(pd.Series(draft_round).reset_index(drop=True), errors="coerce")
    return r.where(r.between(1, UNDRAFTED_ROUND - 1), UNDRAFTED_ROUND).astype(int)


def _observed_rate(df: pd.DataFrame) -> pd.Series:
    """games_played / possible_games, with an unusable denominator left NaN.

    A zero or missing `possible_games` would otherwise divide to inf and take
    the whole position mean with it -- means here are computed with skipna, so
    NaN drops the row instead of poisoning the aggregate.
    """
    possible = pd.to_numeric(df["possible_games"], errors="coerce")
    games = pd.to_numeric(df["games_played"], errors="coerce")
    return games / possible.where(possible > 0)


def load_draft_rounds(db_path: Optional[str] = None) -> pd.DataFrame:
    """player_id -> draft_round, one row per player.

    Deduplicated the same way `career_static_by_player` does it, on MIN
    (draft_pick): `draft_picks_v2` carries a handful of players twice, and the
    bare `draft_round` beside a MIN() aggregate resolves to that same
    minimum-pick row under SQLite's bare-column rule.

    Returns an empty frame when the table is absent, so an older DB degrades
    to the position-mean target rather than failing the load outright.
    """
    import sqlite3
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(db_path or DB_PATH))
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='draft_picks_v2'"
        ).fetchone()
        if not exists:
            return pd.DataFrame(columns=["player_id", "draft_round"])
        return pd.read_sql(
            "SELECT player_id, draft_round, MIN(draft_pick) AS _pick "
            "FROM draft_picks_v2 "
            "WHERE player_id IS NOT NULL AND player_id != '' "
            "GROUP BY player_id", conn)[["player_id", "draft_round"]]
    finally:
        conn.close()


def load_player_seasons(db_path: Optional[str] = None) -> pd.DataFrame:
    """Observed games and PPR per player-season, participation contract applied.

    Every row is an OBSERVED participation. Nothing is fabricated: a player
    with no rows in a season simply does not appear, and no zero-week is
    manufactured to fill the gap.
    """
    import sqlite3
    from config.settings import DB_PATH, MIN_HISTORICAL_YEAR, POSITIONS

    conn = sqlite3.connect(str(db_path or DB_PATH))
    df = pd.read_sql(f"""
        SELECT pws.player_id, p.position, pws.season, pws.week, pws.team,
               pws.snap_count, pws.fantasy_points, pws.data_source
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE p.position IN ({','.join('?' * len(POSITIONS))})
          AND {regular_season_week_sql('pws.week', 'pws.season')}
          AND pws.season >= {MIN_HISTORICAL_YEAR}
    """, conn, params=list(POSITIONS))
    sched = pd.read_sql(
        f"SELECT season, week, home_team AS team FROM schedule "
        f"WHERE {regular_season_week_sql()} UNION ALL "
        f"SELECT season, week, away_team AS team FROM schedule "
        f"WHERE {regular_season_week_sql()}", conn)
    conn.close()

    df["participation_quality"] = label_participation(df)
    df = df[df["participation_quality"] >= 1]
    recv = df["position"].isin(RECEIVING_DEPENDENT_POSITIONS)
    df = df[~(recv & (df["season"] < RECEIVING_CHARTING_MIN_SEASON))]

    played = df.groupby(["player_id", "position", "season"]).agg(
        games_played=("week", "nunique"),
        ppr=("fantasy_points", "sum"),
        team=("team", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
    ).reset_index()
    possible = (sched.groupby(["season", "team"])["week"].nunique()
                .rename("possible_games").reset_index())
    played = played.merge(possible, on=["season", "team"], how="left")
    # Fallback for a team with no schedule rows at all. Was a flat 18, wrong
    # twice: 18 is a WEEK count and this column counts GAMES (one bye), and it
    # ignored the era. Games = max regular-season week - 1, i.e. 16 through
    # 2020 and 17 from 2021.
    fallback = played["season"].map(lambda s: regular_season_max_week(s) - 1)
    played["possible_games"] = played["possible_games"].fillna(fallback)
    played["rate"] = _observed_rate(played)
    played["ppr_per_game"] = played["ppr"] / played["games_played"]

    # Draft round rides along so `fit` and `predict_rate` bucket players the
    # same way. Left NaN where unknown; `draft_bucket` reads that as undrafted.
    rounds = load_draft_rounds(db_path)
    if not rounds.empty:
        played = played.merge(rounds, on="player_id", how="left")
    else:
        played["draft_round"] = np.nan
    return played


class SeasonAvailabilityEstimator:
    """E[games played] from prior seasons only, shrunk toward a population rate.

    That rate is the player's position x draft-round mean where a round is
    available and the position mean where it is not, so a player with no
    history of his own is no longer indistinguishable from every other player
    at his position. `use_draft_prior=False` restores the position-mean-only
    behaviour exactly, for A/B against the 12/12 result in the module
    docstring.

    Causality is structural rather than conventional: `fit` requires the
    target season and drops anything at or after it, so a caller cannot
    accidentally train on the season being projected -- and that now covers
    the draft-round prior too, not just the per-player history.
    """

    def __init__(self, shrinkage_k: float = SHRINKAGE_K,
                 draft_bucket_k: float = DRAFT_BUCKET_K,
                 use_draft_prior: bool = True):
        self.shrinkage_k = float(shrinkage_k)
        self.draft_bucket_k = float(draft_bucket_k)
        self.use_draft_prior = bool(use_draft_prior)
        self.before_season: Optional[int] = None
        self.position_rate_: Optional[pd.Series] = None
        self.draft_prior_: Optional[pd.DataFrame] = None
        self.player_history_: Optional[pd.DataFrame] = None

    def fit(self, player_seasons: pd.DataFrame, before_season: int) -> "SeasonAvailabilityEstimator":
        required = {"player_id", "position", "season", "games_played", "possible_games"}
        missing = required - set(player_seasons.columns)
        if missing:
            raise ValueError(f"player_seasons missing columns: {sorted(missing)}")

        hist = player_seasons[pd.to_numeric(player_seasons["season"], errors="coerce")
                              < int(before_season)]
        if hist.empty:
            raise ValueError(f"no seasons before {before_season} to fit on")

        self.before_season = int(before_season)
        rate = _observed_rate(hist)
        self.position_rate_ = rate.groupby(hist["position"]).mean()
        self.draft_prior_ = self._fit_draft_prior(hist, rate)
        self.player_history_ = hist.groupby("player_id").agg(
            prior_games=("games_played", "sum"),
            prior_possible=("possible_games", "sum")).reset_index()
        return self

    def _fit_draft_prior(self, hist: pd.DataFrame, rate: pd.Series) -> Optional[pd.DataFrame]:
        """Position x draft-round mean rate, shrunk toward the position mean.

        Returns None when there is no round to condition on, which is what
        keeps this backward compatible: the estimator then targets the
        position mean exactly as it did before.
        """
        if not self.use_draft_prior or "draft_round" not in hist.columns:
            return None

        cells = pd.DataFrame({
            "position": hist["position"].to_numpy(),
            "draft_bucket": draft_bucket(hist["draft_round"]).to_numpy(),
            "rate": rate.to_numpy(),
        }).dropna(subset=["rate"])
        if cells.empty:
            return None

        agg = cells.groupby(["position", "draft_bucket"])["rate"].agg(
            ["mean", "size"]).reset_index()
        position_rate = agg["position"].map(self.position_rate_)
        # An unseen position cannot happen here (the cells come from `hist`,
        # which is what position_rate_ was built from), but the fill keeps the
        # arithmetic total rather than silently emitting NaN priors.
        position_rate = position_rate.fillna(self.position_rate_.mean())
        w = agg["size"] / (agg["size"] + self.draft_bucket_k)
        agg["prior"] = w * agg["mean"] + (1.0 - w) * position_rate
        return agg[["position", "draft_bucket", "prior"]]

    def _target_rate(self, merged: pd.DataFrame, position_rate: pd.Series) -> pd.Series:
        """The rate a row is shrunk toward: its draft-round cell, else position.

        Falls back per ROW, not per call: a frame where only some players carry
        a draft round still gets the conditioned target for the ones that do.
        """
        # getattr, not attribute access: `save`/`load` pickle this estimator, so
        # a model fitted before the draft prior existed unpickles without the
        # attribute at all and must keep predicting rather than raise.
        prior_table = getattr(self, "draft_prior_", None)
        if prior_table is None or "draft_round" not in merged.columns:
            return position_rate

        keys = pd.DataFrame({
            "position": merged["position"].to_numpy(),
            "draft_bucket": draft_bucket(merged["draft_round"]).to_numpy(),
        })
        joined = keys.merge(prior_table, on=["position", "draft_bucket"], how="left")
        prior = pd.Series(joined["prior"].to_numpy(), index=merged.index)
        return prior.fillna(position_rate)

    def predict_rate(self, players: pd.DataFrame) -> pd.Series:
        """Expected participation rate per row.

        Fallback for a player with no prior history: the shrinkage target, at
        zero weight on player history (`w = 0 / (0 + K)`). That target is the
        player's position x draft-round rate where a round is available and the
        position mean where it is not. This is a documented degradation, not an
        estimate of whether an unknown player will play at all -- see the
        module docstring.
        """
        if self.position_rate_ is None:
            raise RuntimeError("call fit() before predict_rate()")

        # Dropped rather than suffixed: a caller whose frame already carries
        # these names would otherwise silently be read as its own history.
        clash = [c for c in ("prior_games", "prior_possible") if c in players.columns]
        merged = (players.drop(columns=clash) if clash else players).merge(
            self.player_history_, on="player_id", how="left")

        prior_games = merged["prior_games"].fillna(0.0)
        prior_possible = merged["prior_possible"]
        player_rate = merged["prior_games"] / prior_possible.where(prior_possible > 0)

        position_rate = merged["position"].map(self.position_rate_)
        if position_rate.isna().any():
            position_rate = position_rate.fillna(self.position_rate_.mean())
        target = self._target_rate(merged, position_rate)

        w = prior_games / (prior_games + self.shrinkage_k)
        rate = w * player_rate.fillna(target) + (1.0 - w) * target
        return pd.Series(np.clip(rate.to_numpy(), 0.0, 1.0), index=players.index)

    def predict_games(self, players: pd.DataFrame,
                      possible_games: Optional[pd.Series] = None) -> pd.Series:
        if possible_games is None:
            possible_games = players["possible_games"]
        return self.predict_rate(players) * np.asarray(possible_games, dtype=float)

    def has_history(self, players: pd.DataFrame) -> pd.Series:
        known = set(self.player_history_["player_id"]) if self.player_history_ is not None else set()
        return players["player_id"].isin(known)


def project_season_ppr(expected_games: pd.Series, expected_ppr_per_game: pd.Series) -> pd.Series:
    """season PPR = E[games] x E[PPR per game].

    Deliberately the whole calculation. The season-bias improvement comes
    from this product; adding a separate post-hoc bias correction on top
    would double-count the same effect.
    """
    return pd.Series(np.asarray(expected_games, dtype=float)
                     * np.asarray(expected_ppr_per_game, dtype=float),
                     index=getattr(expected_games, "index", None))
