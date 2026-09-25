"""Role-aware hierarchical allocation for sparse full-PPR events.

This module deliberately contains no same-week outcomes.  Role tiers are
derived from lagged snap/opportunity history, while event conditioning is
learned from historical team-week totals inside each walk-forward fold.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import logsumexp

from sklearn.linear_model import Ridge


SPARSE_EVENT_TARGETS = {"receiving_tds", "rushing_tds", "passing_tds", "interceptions"}

# Event-specific opportunity signals.  These are all lagged by the share-table
# builder before they reach this module.
EVENT_OPPORTUNITY_FEATURES = {
    "receiving_tds": (
        "redzone_targets", "share_of_team_redzone_targets", "targets_15_plus",
        "air_yards", "recv_targets",
    ),
    "rushing_tds": (
        "goal_line_touches", "share_of_team_goal_line_touches", "rush_inside_5",
        "share_of_team_rush_inside_5", "rush_inside_10", "short_yardage_rushes",
    ),
    "passing_tds": (
        "pass_plays", "team_pass_plays", "passing_yards", "team_passing_yards",
        "share_of_team_passing_yards",
    ),
    "interceptions": (
        "pass_plays", "team_pass_plays", "passing_yards", "team_passing_yards",
        "air_yards",
    ),
}

# These are deliberately lagged feature *families*, not a broad automatic
# sweep.  Sparse events are where a transparent incremental feature test is
# most useful: we can ask whether player event role helps before asking if
# team-level event environment adds anything beyond it.
EVENT_FEATURE_FAMILIES = {
    "player_event_role": EVENT_OPPORTUNITY_FEATURES,
    "team_event_context": {
        "receiving_tds": (
            "team_redzone_targets", "team_targets_15_plus", "team_air_yards",
            "team_recv_targets", "team_pass_plays", "team_neutral_targets",
        ),
        "rushing_tds": (
            "team_goal_line_touches", "team_rush_inside_5", "team_rush_inside_10",
            "team_short_yardage_rushes", "team_rush_plays", "team_neutral_rushes",
        ),
        "passing_tds": (
            "team_pass_plays", "team_passing_yards", "team_recv_targets",
            "team_redzone_targets", "team_neutral_targets",
        ),
        "interceptions": (
            "team_pass_plays", "team_passing_yards", "team_air_yards",
            "team_neutral_targets",
        ),
    },
}


class EmpiricalBayesShareModel:
    """Scalable random-intercept mixed model for player shares.

    This is the closed-form empirical-Bayes equivalent of a random-intercept
    Gaussian mixed model.  A fixed-effects Ridge model captures role and
    opportunity; player residual means are shrunk toward their role mean by
    reliability (sample count and estimated between-player variance).  Unlike
    statsmodels MixedLM, it remains practical with thousands of player groups.
    """

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.columns_: list[str] = []
        self.model_: Ridge | None = None
        self.global_residual_ = 0.0
        self.role_effects_: dict[str, float] = {}
        self.player_effects_: dict[str, float] = {}
        self.player_weight_: dict[str, float] = {}

    @staticmethod
    def _design(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        base = X.drop(columns=[c for c in ("player_id", "role_tier") if c in X], errors="ignore").copy()
        if "role_tier" in X:
            base = pd.concat([base, pd.get_dummies(X["role_tier"].astype(str), prefix="role")], axis=1)
        base = base.apply(pd.to_numeric, errors="coerce")
        if columns is not None:
            base = base.reindex(columns=columns, fill_value=0.0)
        return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EmpiricalBayesShareModel":
        self.columns_ = list(self._design(X).columns)
        design = self._design(X, self.columns_)
        self.model_ = Ridge(alpha=self.alpha).fit(design, np.asarray(y, dtype=float))
        residual = np.asarray(y, dtype=float) - self.model_.predict(design)
        frame = pd.DataFrame({
            "player_id": X["player_id"].astype(str).to_numpy(),
            "role_tier": X["role_tier"].astype(str).to_numpy(),
            "residual": residual,
        })
        self.global_residual_ = float(frame.residual.mean()) if len(frame) else 0.0
        role_mean = frame.groupby("role_tier").residual.mean()
        self.role_effects_ = {str(k): float(v) for k, v in role_mean.items()}
        role_centered = frame.residual - frame.role_tier.map(role_mean).fillna(self.global_residual_)
        noise = float(np.var(role_centered)) if len(role_centered) else 0.0
        grouped = frame.groupby("player_id").residual.agg(["mean", "count"])
        between = float(np.var(grouped["mean"])) if len(grouped) > 1 else 0.0
        tau2 = max(between - noise / max(float(grouped["count"].mean()), 1.0), 1e-8)
        for player, row in grouped.iterrows():
            role = frame.loc[frame.player_id.eq(player), "role_tier"].iloc[0]
            prior = self.role_effects_.get(str(role), self.global_residual_)
            weight = float(row["count"] / (row["count"] + noise / tau2))
            self.player_effects_[str(player)] = float(prior + weight * (row["mean"] - prior))
            self.player_weight_[str(player)] = weight
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() before predict()")
        fixed = self.model_.predict(self._design(X, self.columns_))
        role = X["role_tier"].astype(str).map(self.role_effects_).fillna(self.global_residual_).to_numpy(float)
        player = X["player_id"].astype(str).map(self.player_effects_).fillna(pd.Series(role, index=X.index)).to_numpy(float)
        return fixed + player


def add_role_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a lagged-history position/rank role label.

    The rank is computed from prior snap share, then prior opportunity share;
    no current-week volume or outcome is read.  A position-local rank is more
    stable than a global roster rank and gives MixedLM an interpretable fixed
    effect for starter/rotation roles.
    """
    out = df.copy()
    snap = out.get("snap_share_roll3", pd.Series(np.nan, index=out.index))
    opp = out.get("share_of_team_targets_roll3", pd.Series(np.nan, index=out.index))
    score = pd.DataFrame({"snap": snap.fillna(-1), "opp": opp.fillna(-1)}, index=out.index)
    score["player_id"] = out["player_id"].astype(str)
    score["position"] = out["position"].astype(str)
    score["team"] = out["team"].astype(str)
    score["season"] = out["season"].astype(int)
    score["week"] = out["week"].astype(int)
    score = score.sort_values(
        ["team", "season", "week", "position", "snap", "opp", "player_id"],
        ascending=[True, True, True, True, False, False, True],
    )
    score["rank"] = score.groupby(["team", "season", "week", "position"], sort=False).cumcount() + 1
    score["rank"] = score["rank"].clip(upper=4)
    role = score["position"] + score["rank"].astype(str)
    out["role_tier"] = role.reindex(out.index).fillna(out["position"].astype(str) + "1")
    return out


def hierarchical_feature_columns(
    df: pd.DataFrame,
    target: str,
    *,
    families: Iterable[str] | None = None,
) -> list[str]:
    """Small, stable fixed-effect design for MixedLM.

    Keeping the design intentionally compact avoids singular fits from the
    full 100+ column allocation contract while retaining role/opportunity
    signal relevant to each sparse event.
    """
    selected = set(families or EVENT_FEATURE_FAMILIES)
    unknown = selected.difference(EVENT_FEATURE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown sparse-event feature families: {sorted(unknown)}")
    names = ["is_cold_start"]
    for family in sorted(selected):
        tokens = EVENT_FEATURE_FAMILIES[family].get(target, ())
        for col in df.columns:
            is_team_feature = col.startswith("team_")
            wants_team_feature = family == "team_event_context"
            if (
                col.endswith(("_roll3", "_s2d"))
                and is_team_feature == wants_team_feature
                and any(token in col for token in tokens)
            ):
                names.append(col)
    # Preserve order and remove columns absent from a minimal test fixture.
    return list(dict.fromkeys(c for c in names if c in df.columns))


def _opportunity_exposure(df: pd.DataFrame, target: str) -> pd.Series:
    """Build a target-specific, lagged player opportunity exposure."""
    def val(name: str) -> pd.Series:
        for suffix in ("_roll3", "_s2d"):
            col = f"{name}{suffix}"
            if col in df:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    if target == "receiving_tds":
        exposure = val("redzone_targets") + 0.5 * val("targets_15_plus") + 0.25 * val("air_yards")
    elif target == "rushing_tds":
        exposure = val("goal_line_touches") + val("rush_inside_5") + 0.25 * val("rush_inside_10") + 0.5 * val("short_yardage_rushes")
    elif target in {"passing_tds", "interceptions"}:
        exposure = val("pass_plays") + 0.25 * val("passing_yards")
    else:
        raise ValueError(target)
    return exposure.clip(lower=0.0)


def _eligibility_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    """Identify players with plausible prior opportunity for this event."""
    def prior_sum(*names: str) -> pd.Series:
        values = []
        for name in names:
            for suffix in ("_roll3", "_s2d"):
                col = f"{name}{suffix}"
                if col in df:
                    values.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0))
                    break
        return sum(values, pd.Series(0.0, index=df.index))

    snap = prior_sum("snap_share")
    if target == "receiving_tds":
        opportunity = prior_sum(
            "redzone_targets", "recv_targets", "targets", "share_of_team_targets", "air_yards"
        )
        return ((opportunity > 0) | (snap >= 0.05)).to_numpy(bool)
    if target == "rushing_tds":
        opportunity = prior_sum(
            "rushing_attempts", "share_of_team_rushing_attempts", "goal_line_touches",
            "rush_inside_5", "rush_inside_10"
        )
        return ((opportunity > 0) | (snap >= 0.05)).to_numpy(bool)
    # Passing events are QB-only in TARGET_POPULATIONS; require prior pass
    # participation/dropback evidence rather than roster presence alone.
    opportunity = prior_sum(
        "pass_plays", "passing_yards", "share_of_team_passing_yards", "passing_attempts"
    )
    return (opportunity > 0).to_numpy(bool)


def _opportunity_share(
    df: pd.DataFrame,
    target: str,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    exposure = _opportunity_exposure(df, target)
    if eligible is not None:
        exposure = exposure.where(pd.Series(eligible, index=df.index), 0.0)
    group_sum = exposure.groupby([df["team"], df["season"], df["week"]]).transform("sum")
    return (exposure / group_sum.replace(0, np.nan)).fillna(0.0).to_numpy(float)


def _fit_calibrated_opportunity_model(
    active_train: pd.DataFrame,
    positive: pd.DataFrame,
    target: str,
    cols: list[str],
    label: str,
    *,
    residual_mode: bool = False,
) -> tuple[EmpiricalBayesShareModel, float]:
    """Fit the rate/residual model and select EB Ridge alpha fold-locally."""
    opportunity = _opportunity_share(active_train, target)
    positive_rows = active_train[label].to_numpy(float) > 0
    positive_opportunity = opportunity[positive_rows]
    y_positive = positive[label].to_numpy(float)
    if residual_mode:
        y_model = y_positive - positive_opportunity
    else:
        y_model = y_positive / np.maximum(positive_opportunity, 0.02)

    positive_seasons = positive["season"].to_numpy(int)
    last = int(active_train["season"].max())
    cal = positive_seasons == last
    fit = positive_seasons < last
    alphas = (1.0, 3.0, 10.0, 30.0, 100.0)
    selected = 10.0
    if fit.sum() >= 30 and cal.sum() >= 10:
        best_score = np.inf
        fit_X = positive[cols].copy()
        fit_X.insert(0, "role_tier", positive["role_tier"].astype(str).to_numpy())
        fit_X.insert(0, "player_id", positive["player_id"].astype(str).to_numpy())
        for alpha in alphas:
            candidate = EmpiricalBayesShareModel(alpha=alpha).fit(fit_X.loc[fit], y_model[fit])
            cal_X = positive[cols].copy()
            cal_X.insert(0, "role_tier", positive["role_tier"].astype(str).to_numpy())
            cal_X.insert(0, "player_id", positive["player_id"].astype(str).to_numpy())
            model_pred = candidate.predict(cal_X.loc[cal])
            if residual_mode:
                predicted_share = opportunity[positive_rows][cal] + model_pred
            else:
                predicted_share = positive_opportunity[cal] * np.clip(model_pred, 0.0, 5.0)
            score = float(np.mean(np.abs(y_positive[cal] - np.clip(predicted_share, 0.0, 1.0))))
            if score < best_score:
                selected, best_score = alpha, score

    full_X = positive[cols].copy()
    full_X.insert(0, "role_tier", positive["role_tier"].astype(str).to_numpy())
    full_X.insert(0, "player_id", positive["player_id"].astype(str).to_numpy())
    return EmpiricalBayesShareModel(alpha=selected).fit(full_X, y_model), selected


def _team_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    keys = ["team", "season", "week"]
    team = df.sort_values(keys).drop_duplicates(keys).copy()
    features = [
        c for c in df.columns
        if c.startswith("team_") and c.endswith(("_roll3", "_s2d"))
    ]
    # Team event history is always included when available.
    features = list(dict.fromkeys(features + [f"team_{target}_roll3", f"team_{target}_s2d"]))
    features = [c for c in features if c in team.columns]
    return team.reset_index(drop=True), features


def _fit_event_gate(train_df: pd.DataFrame, target: str):
    team, features = _team_frame(train_df, target)
    if not features or team[f"team_{target}"].nunique() < 2:
        return None, 0.5
    y = (team[f"team_{target}"].to_numpy(float) > 0).astype(int)
    if y.min() == y.max():
        return None, 0.5
    def make_model():
        return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    # Calibrate team event mass on the last *training* season.  The original
    # classifier was class-balanced, which is useful for identifying event
    # weeks but can badly overstate a rare event's absolute probability.  A
    # shrinkage calibration is intentionally simple and robust: it pulls the
    # raw probability toward the prior event rate by a weight selected only
    # on the temporal calibration season.  The final classifier is then
    # refit on every permitted training season.
    calibration = {"weight": 1.0, "base_rate": float(y.mean()), "method": "identity"}
    threshold = 0.5  # retained for backwards-compatible diagnostics only
    last = int(team["season"].max())
    cal = team[team["season"] == last]
    prior = team[team["season"] < last]
    prior_y = (prior[f"team_{target}"].to_numpy(float) > 0).astype(int)
    if len(prior) >= 30 and len(cal) >= 10 and np.unique(prior_y).size == 2:
        cal_model = make_model().fit(prior[features], (prior[f"team_{target}"] > 0).astype(int))
        raw_p = cal_model.predict_proba(cal[features])[:, 1]
        cal_y = (cal[f"team_{target}"].to_numpy(float) > 0).astype(int)
        prior_rate = float(prior_y.mean())
        weight = min(
            np.arange(0.0, 1.001, 0.05),
            key=lambda w: float(np.mean((cal_y - (w * raw_p + (1.0 - w) * prior_rate)) ** 2)),
        )
        calibration = {
            "weight": float(weight),
            "base_rate": float(y.mean()),
            "calibration_prior_rate": prior_rate,
            "method": "last_season_brier_shrinkage",
        }
    model = make_model().fit(team[features], y)
    return (model, features, calibration), float(threshold)


def _team_event_probabilities(fitted, test_df: pd.DataFrame, target: str) -> np.ndarray:
    if fitted is None:
        return np.ones(len(test_df), dtype=float)
    model, features, calibration = fitted
    team, _ = _team_frame(test_df, target)
    if len(team) == 0:
        return np.zeros(len(test_df), dtype=float)
    raw_p = model.predict_proba(team[features])[:, 1]
    weight = float(calibration.get("weight", 1.0))
    base_rate = float(calibration.get("base_rate", 0.5))
    p = np.clip(weight * raw_p + (1.0 - weight) * base_rate, 0.0, 1.0)
    lookup = team[["team", "season", "week"]].copy()
    lookup["_p"] = p
    out = test_df[["team", "season", "week"]].merge(
        lookup, on=["team", "season", "week"], how="left", validate="many_to_one"
    )
    return out["_p"].fillna(0.0).to_numpy(float)


def _fit_active_classifier(X: pd.DataFrame, y: np.ndarray):
    if len(np.unique(y)) < 2:
        return None
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X, y.astype(int))
    return model


def predict_conditional_hierarchical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    feature_cols: Iterable[str] | None = None,
) -> np.ndarray:
    """Predict a sparse event share using team-positive conditioning.

    Training rows are restricted to team-positive weeks for sparse targets.
    The team gate uses only lagged team features and is evaluated at the
    team-week level; player predictions are then activation probability times
    a positive-share mixed-effects estimate.
    """
    if target not in SPARSE_EVENT_TARGETS:
        raise ValueError(f"conditional hierarchical arm only supports {SPARSE_EVENT_TARGETS}")
    train = add_role_tier(train_df)
    test = add_role_tier(test_df)
    label = f"share_of_team_{target}"
    cols = list(feature_cols or hierarchical_feature_columns(train, target))
    cols = [c for c in cols if c in train.columns and c in test.columns]
    train_eligible = _eligibility_mask(train, target)
    test_eligible = _eligibility_mask(test, target)
    positive_team = train[f"team_{target}"].to_numpy(float) > 0
    positive_team &= train_eligible
    if positive_team.sum() < 30:
        return np.zeros(len(test), dtype=float)

    gate, threshold = _fit_event_gate(train.loc[positive_team | ~positive_team], target)
    event_prob = _team_event_probabilities(gate, test, target)
    # Soft conditioning avoids the discontinuity caused by a hard 0.5 gate.
    # A team with uncertain event probability receives a proportionally
    # uncertain player allocation instead of an all-zero roster.
    event_weight = np.clip(event_prob, 0.0, 1.0)
    pred = np.zeros(len(test), dtype=float)

    # Activation is learned only inside team-positive weeks.  The feature
    # matrix remains lagged; current event labels are used only as y.
    active_train = train.loc[positive_team].copy()
    X_active = active_train[cols]
    active_y = (active_train[label].to_numpy(float) > 0).astype(int)
    active_model = _fit_active_classifier(X_active, active_y)
    if active_model is None:
        active_probability = np.zeros(len(test), dtype=float)
        active_probability[test_eligible] = 1.0
    else:
        active_probability = np.zeros(len(test), dtype=float)
        active_probability[test_eligible] = active_model.predict_proba(test.loc[test_eligible, cols])[:, 1]

    positive_rows = active_train[label].to_numpy(float) > 0
    positive = active_train.loc[positive_rows]
    opportunity_test = _opportunity_share(test, target, test_eligible)
    if len(positive) < 30:
        raw_positive = np.full(len(test), float(positive[label].mean()) if len(positive) else 0.0)
    else:
        mixed, _ = _fit_calibrated_opportunity_model(active_train, positive, target, cols, label)
        test_X = test[cols].copy()
        test_X.insert(0, "role_tier", test["role_tier"].astype(str).to_numpy())
        test_X.insert(0, "player_id", test["player_id"].astype(str).to_numpy())
        predicted_rate = np.clip(mixed.predict(test_X), 0.0, 5.0)
        raw_positive = np.clip(opportunity_test * predicted_rate, 0, 1)
    pred = np.clip(event_weight * active_probability * raw_positive, 0, 1)
    pred[~test_eligible] = 0.0
    return pred


def predict_residual_over_opportunity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    feature_cols: Iterable[str] | None = None,
) -> np.ndarray:
    """Compare a residual model against the target-specific opportunity prior."""
    if target not in SPARSE_EVENT_TARGETS:
        raise ValueError(f"residual opportunity arm only supports {SPARSE_EVENT_TARGETS}")
    train = add_role_tier(train_df)
    test = add_role_tier(test_df)
    label = f"share_of_team_{target}"
    cols = list(feature_cols or hierarchical_feature_columns(train, target))
    cols = [c for c in cols if c in train.columns and c in test.columns]
    train_eligible = _eligibility_mask(train, target)
    test_eligible = _eligibility_mask(test, target)
    positive_team = train[f"team_{target}"].to_numpy(float) > 0
    positive_team &= train_eligible
    if positive_team.sum() < 30:
        return np.zeros(len(test), dtype=float)
    active_train = train.loc[positive_team].copy()
    positive = active_train.loc[active_train[label].to_numpy(float) > 0].copy()
    opportunity_test = _opportunity_share(test, target, test_eligible)
    gate, _ = _fit_event_gate(train, target)
    event_weight = np.clip(_team_event_probabilities(gate, test, target), 0.0, 1.0)
    if len(positive) < 30:
        raw = opportunity_test
    else:
        model, _ = _fit_calibrated_opportunity_model(
            active_train, positive, target, cols, label, residual_mode=True
        )
        test_X = test[cols].copy()
        test_X.insert(0, "role_tier", test["role_tier"].astype(str).to_numpy())
        test_X.insert(0, "player_id", test["player_id"].astype(str).to_numpy())
        raw = np.clip(opportunity_test + model.predict(test_X), 0.0, 1.0)
    out = np.clip(event_weight * raw, 0.0, 1.0)
    out[~test_eligible] = 0.0
    return out


def predict_poisson_count_allocation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    feature_cols: Iterable[str] | None = None,
) -> np.ndarray:
    """Poisson count benchmark, normalized as a team allocation.

    This is intentionally a benchmark arm: it models the nonnegative player
    event count directly, then converts expected counts to a compositional
    share. It is not promoted without beating both rolling-3 gates.
    """
    if target not in SPARSE_EVENT_TARGETS:
        raise ValueError(f"count allocation arm only supports {SPARSE_EVENT_TARGETS}")
    train = add_role_tier(train_df)
    test = add_role_tier(test_df)
    cols = list(feature_cols or hierarchical_feature_columns(train, target))
    cols = [c for c in cols if c in train.columns and c in test.columns]
    label = f"share_of_team_{target}"
    count_col = target
    if count_col not in train.columns:
        return np.zeros(len(test), dtype=float)
    train_eligible = _eligibility_mask(train, target)
    test_eligible = _eligibility_mask(test, target)
    if not train_eligible.any() or not test_eligible.any():
        return np.zeros(len(test), dtype=float)
    train_X = train.loc[train_eligible, cols].copy()
    train_X.insert(0, "role_tier", train.loc[train_eligible, "role_tier"].astype(str).to_numpy())
    test_X = test.loc[test_eligible, cols].copy()
    test_X.insert(0, "role_tier", test.loc[test_eligible, "role_tier"].astype(str).to_numpy())
    design_train = EmpiricalBayesShareModel._design(train_X)
    design_test = EmpiricalBayesShareModel._design(test_X, list(design_train.columns))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PoissonRegressor(alpha=1.0, max_iter=1000),
    )
    model.fit(design_train, np.clip(train.loc[train_eligible, count_col].to_numpy(float), 0.0, None))
    expected_eligible = np.clip(model.predict(design_test), 0.0, None)
    expected = np.zeros(len(test), dtype=float)
    expected[test_eligible] = expected_eligible
    keys = test[["team", "season", "week"]]
    sums = pd.Series(expected).groupby([keys["team"].to_numpy(), keys["season"].to_numpy(), keys["week"].to_numpy()]).transform("sum").to_numpy()
    return np.divide(expected, sums, out=np.zeros_like(expected), where=sums > 1e-9)


def predict_multinomial_count_allocation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    feature_cols: Iterable[str] | None = None,
    ridge_penalty: float = 1.0,
) -> np.ndarray:
    """Conditional multinomial-logit count allocation benchmark.

    Each team-week with at least one observed event is one multinomial trial;
    player counts are the trial frequencies.  A shared utility model is fit
    by maximizing the grouped multinomial likelihood, then softmax utilities
    are converted to player shares on the test team-weeks.  The team-event
    probability is applied separately so event-free weeks remain all-zero.

    This is deliberately a benchmark arm.  It has no player identity feature,
    so unseen players can be scored from lagged role/opportunity features, and
    it is only eligible for promotion if it clears both rolling-3 gates.
    """
    if target not in SPARSE_EVENT_TARGETS:
        raise ValueError(f"multinomial count allocation only supports {SPARSE_EVENT_TARGETS}")
    train = add_role_tier(train_df)
    test = add_role_tier(test_df)
    cols = list(feature_cols or hierarchical_feature_columns(train, target))
    cols = [c for c in cols if c in train.columns and c in test.columns]
    count_col = target
    if count_col not in train.columns or not cols:
        return np.zeros(len(test), dtype=float)
    train_eligible = _eligibility_mask(train, target)
    test_eligible = _eligibility_mask(test, target)
    eligible_train = train.loc[train_eligible].copy()
    if len(eligible_train) == 0 or not test_eligible.any():
        return np.zeros(len(test), dtype=float)

    # Build a numeric utility design with role dummies.  Player identity is
    # intentionally excluded so the arm is a genuine population model.
    train_X = eligible_train[cols].copy()
    train_X.insert(0, "role_tier", eligible_train["role_tier"].astype(str).to_numpy())
    test_X = test.loc[test_eligible, cols].copy()
    test_X.insert(0, "role_tier", test.loc[test_eligible, "role_tier"].astype(str).to_numpy())
    design_train = EmpiricalBayesShareModel._design(train_X)
    design_test = EmpiricalBayesShareModel._design(test_X, list(design_train.columns))
    X = design_train.to_numpy(float)
    counts = np.clip(eligible_train[count_col].to_numpy(float), 0.0, None)
    keys = eligible_train[["team", "season", "week"]].astype(str).agg("|".join, axis=1)
    group_codes, _ = pd.factorize(keys, sort=False)
    totals = np.bincount(group_codes, weights=counts)
    valid = totals[group_codes] > 0
    X = X[valid]
    counts = counts[valid]
    group_codes = group_codes[valid]
    if len(counts) == 0 or np.unique(group_codes).size < 2:
        return np.zeros(len(test), dtype=float)

    # Centering is not required for softmax, but improves optimizer scaling.
    scale = np.nanstd(X, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    X_scaled = X / scale
    n_groups = int(group_codes.max()) + 1

    def objective(beta: np.ndarray, gradient: bool = False):
        utilities = X_scaled @ beta
        normalizers = np.zeros(n_groups, dtype=float)
        for group in range(n_groups):
            rows = group_codes == group
            normalizers[group] = logsumexp(utilities[rows])
        loss = -float(np.sum(counts * (utilities - normalizers[group_codes])))
        loss += 0.5 * float(ridge_penalty) * float(beta @ beta)
        if not gradient:
            return loss
        probabilities = np.exp(utilities - normalizers[group_codes])
        expected = np.zeros_like(beta)
        for group in range(n_groups):
            rows = group_codes == group
            expected += X_scaled[rows].T @ (counts[rows].sum() * probabilities[rows] - counts[rows])
        expected += ridge_penalty * beta
        return loss, expected

    result = minimize(
        lambda beta: objective(beta, gradient=True),
        np.zeros(X_scaled.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-9},
    )
    beta = result.x if np.all(np.isfinite(result.x)) else np.zeros(X_scaled.shape[1], dtype=float)
    utilities = design_test.to_numpy(float) / scale @ beta
    test_keys = test.loc[test_eligible, ["team", "season", "week"]].astype(str).agg("|".join, axis=1)
    pred_eligible = np.zeros(len(test_X), dtype=float)
    for key in test_keys.unique():
        rows = test_keys.to_numpy() == key
        pred_eligible[rows] = np.exp(utilities[rows] - logsumexp(utilities[rows]))

    # Apply the same leakage-safe team-event gate as the conditional arm.
    gate, _ = _fit_event_gate(train, target)
    event_prob = np.clip(_team_event_probabilities(gate, test, target), 0.0, 1.0)
    out = np.zeros(len(test), dtype=float)
    out[test_eligible] = event_prob[test_eligible] * pred_eligible
    return np.clip(out, 0.0, 1.0)
