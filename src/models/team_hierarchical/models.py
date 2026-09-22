"""Mixed-effects share regressor for Plan B.

IMPORTANT DESIGN CORRECTION from this doc's original Plan B sketch (see
docs/TEAM_LEVEL_ALLOCATION_MODELS.md's Architecture section) -- found while
actually implementing this, not assumed away:

1. A random effect's value is a per-group deviation estimated FROM THAT
   GROUP'S OWN DATA. A walk-forward test row is, by construction, always a
   genuinely NEW (team, season, week) -- so a random effect GROUPED BY
   team-week has no observations to estimate from at prediction time and
   contributes nothing to a forecast. Grouping by `player_id` instead
   avoids this: a player with prior training history keeps the same group
   across future weeks, so their estimated persistent deviation (talent/
   role tendency beyond what slot+features predict) DOES carry forward. A
   player with no prior history (true cold start) correctly falls back to
   the fixed-effects-only prediction. Team-environment effects (what a
   team-week grouping was meant to capture) are instead carried by Plan
   A's existing lagged team-total features already in the fixed-effects
   design (`team_{stat}_s2d/roll3`) -- an observed, pre-game-known proxy,
   unlike an unobservable-until-the-fact team-week intercept.

2. statsmodels' `MixedLMResults.predict()` uses ONLY the fixed effects for
   EVERY row, known group or not (verified empirically: a known group's
   prediction via `.predict()` ignores its own estimated random effect
   entirely). Using a per-player random effect at prediction time requires
   manually adding `result.random_effects[group]["Group"]` on top of the
   fixed-effects prediction -- which `predict()` below does -- falling back
   to 0 (fixed-effects-only) for a group absent from `random_effects`
   (never seen in training).
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


class MixedEffectsShareModel:
    """Random intercept per `player_id`, fixed effects = `slot` (categorical)
    plus every other column in X. See module docstring for why `player_id`,
    not team-week, is the random-effect group.

    `fit(X, y)`: X must include `player_id` (the random-effect group) and
    `slot` (the key fixed effect) alongside the usual feature columns --
    deliberately NOT the same "X is exactly feature_columns(df)" contract
    Plan A's Ridge/XGB wrappers use, since this model has a structural
    dependency (groups) a tree/linear regressor doesn't. Documented here
    rather than silently pretending to match an interface it doesn't.
    """

    def __init__(self, target_col: str = "y", reml: bool = True):
        self.target_col = target_col
        self.reml = reml
        self.result_ = None
        self.converged_ = None
        self._fixed_effect_cols: List[str] = []
        self._slot_categories: Optional[pd.Index] = None
        self._fallback_mean_: Optional[float] = None

    def _formula(self) -> str:
        numeric = [c for c in self._fixed_effect_cols if c != "slot"]
        rhs = " + ".join(["C(slot)"] + numeric) if numeric else "C(slot)"
        return f"{self.target_col} ~ {rhs}"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MixedEffectsShareModel":
        if X.columns.duplicated().any():
            # Easy mistake to make: features.py's feature_columns() already
            # includes "slot" (it's a real fixed-effect feature, not an id
            # column) -- a caller building X as
            # df[["player_id", "slot"] + feature_columns(df)] double-counts
            # it. The intended contract is
            # df[["player_id"] + feature_columns(df)]. Failing clearly here
            # beats the confusing pandas AttributeError a duplicate "slot"
            # column produces downstream (X["slot"] returns a DataFrame,
            # not a Series, once .dropna()/.unique() promptly break on).
            dupes = sorted(set(X.columns[X.columns.duplicated()]))
            raise ValueError(
                f"X has duplicate columns {dupes} -- feature_columns(df) already "
                "includes 'slot', so X should be df[['player_id'] + feature_columns(df)], "
                "not with 'slot' added again separately"
            )
        if "player_id" not in X.columns or "slot" not in X.columns:
            raise ValueError("X must include 'player_id' (random-effect group) and 'slot' (fixed effect)")
        self._fixed_effect_cols = [c for c in X.columns if c != "player_id"]
        # Categories fixed to what training actually saw -- a slot with a
        # coefficient statsmodels never estimated (never appeared in this
        # fold's training data) is handled at predict() time by falling
        # back to that position's rank-1 slot (see predict()'s docstring),
        # not by pretending the category exists here.
        self._slot_categories = pd.Index(sorted(X["slot"].dropna().unique()))
        self._fallback_mean_ = float(np.nanmean(y))

        data = X.copy()
        data[self.target_col] = np.asarray(y, dtype=float)
        data["slot"] = pd.Categorical(data["slot"], categories=self._slot_categories)
        data = data.dropna(subset=[self.target_col, "slot"] + [
            c for c in self._fixed_effect_cols if c not in ("slot", "player_id")
        ])
        # A numeric fixed-effect column with zero variance (e.g. every
        # training row cold-start) makes the design matrix singular --
        # drop it rather than let statsmodels fail opaquely; a constant
        # feature carries no information for this fold regardless.
        for col in list(self._fixed_effect_cols):
            if col != "slot" and col in data.columns and data[col].nunique(dropna=True) <= 1:
                self._fixed_effect_cols.remove(col)

        try:
            with warnings.catch_warnings():
                # A degenerate fold (too little data, singular design) is an
                # expected, handled code path here, not a bug -- statsmodels/
                # numpy's convergence and numerical-stability warnings for it
                # are noise once the non-converged result below is discarded.
                warnings.simplefilter("ignore")
                model = smf.mixedlm(self._formula(), data=data, groups=data["player_id"])
                fitted = model.fit(reml=self.reml, method="lbfgs")
            self.converged_ = bool(fitted.converged)
        except Exception:
            fitted = None
            self.converged_ = False

        # A non-converged fit does not raise -- statsmodels returns a
        # result object with `converged=False` and, found empirically,
        # NaN/garbage parameters rather than an exception. Discarding it
        # here (not just on the exception path above) matters: keeping a
        # non-converged result would silently feed untrustworthy
        # coefficients into predict() instead of falling back to the
        # trivial constant, the same degenerate-fold treatment
        # VegasFavoriteBaseline (src/models/game_outcome/baseline.py)
        # already gives elsewhere in this repo.
        self.result_ = fitted if self.converged_ else None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._slot_categories is None:
            raise RuntimeError("call fit() before predict()")
        if self.result_ is None:
            # fit() hit the degenerate-fold fallback -- the trivial
            # constant is the only thing left to predict.
            return np.full(len(X), self._fallback_mean_)

        data = X.copy()
        # A slot unseen in training has no estimated coefficient. Fall back
        # to that same position's rank-1 slot (e.g. unseen "WR5" -> "WR1")
        # -- rank-1 slots are the ones virtually always populated (starters
        # are filled almost every week), so this is the least-arbitrary
        # known category to borrow, not a genuinely informed guess about
        # the unseen slot itself. Documented, not silently dropped.
        unseen = ~data["slot"].isin(self._slot_categories)
        if unseen.any():
            position = data.loc[unseen, "slot"].astype(str).str.extract(r"^([A-Za-z]+)")[0]
            fallback_slot = position + "1"
            still_unseen = ~fallback_slot.isin(self._slot_categories)
            # If even the rank-1 fallback was never seen in training,
            # there is no fixed-effect coefficient this fold can offer at
            # all for that position -- leave those rows to the constant
            # fallback via NaN-then-fillna below rather than guess further.
            fallback_slot = fallback_slot.where(~still_unseen, np.nan)
            data.loc[unseen, "slot"] = fallback_slot.values
        data["slot"] = pd.Categorical(data["slot"], categories=self._slot_categories)

        fixed_pred = self.result_.predict(exog=data)
        fixed_pred = fixed_pred.reindex(data.index)
        fixed_pred = fixed_pred.fillna(self._fallback_mean_)

        random_effects = self.result_.random_effects
        re_adjustment = data["player_id"].map(
            lambda pid: float(random_effects[pid]["Group"]) if pid in random_effects else 0.0
        )
        return (fixed_pred.to_numpy() + re_adjustment.to_numpy())
