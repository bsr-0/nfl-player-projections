"""Leakage-resistant participation and opportunity modeling primitives.

Phase 2 is deliberately independent of the PPR model.  It estimates:

* P(meaningful offensive participation before kickoff); and
* E[offensive snap share | meaningful participation].

Only snap-observed rows are labels.  ``participation_state == 'unknown'`` is
never silently converted to a negative.  Every usage feature is shifted by at
least one player-game; the current row contributes only schedule/identity
information that is knowable before kickoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KEY = ["player_id", "season", "week"]
POSITIONS = ("QB", "RB", "WR", "TE")
TARGET_THRESHOLDS = (0.0, 0.10, 0.25)
PRIMARY_THRESHOLD = 0.10

NUMERIC_FEATURES = [
    "week", "is_home", "prior_games_observed", "prior_games_played",
    "prior_zero_snap_games", "prior_unknown_games", "weeks_since_observed",
    "snap_share_lag1", "snap_share_roll3", "snap_share_roll5",
    "snap_share_roll3_known", "played_lag1", "played_roll3",
    "team_position_rank_lag1", "cold_start",
    "injury_score",
]
CATEGORICAL_FEATURES = ["position", "status_lag1"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TEAM_COMPETITION_FEATURES = [
    "peer_history_count",
    "peer_snap_share_lag1_mean",
    "peer_snap_share_lag1_max",
    "peer_snap_share_lag1_sum",
    "peer_snap_share_lag1_hhi",
    "peer_snap_share_roll3_mean",
    "peer_snap_share_roll3_max",
    "peer_meaningful_10_lag1_count",
]
TEAM_COMPETITION_MODEL_FEATURES = FEATURES + TEAM_COMPETITION_FEATURES


def normalize_snap_share(values: pd.Series) -> pd.Series:
    """Return snap share on [0, 1], accepting fractions, percentages or 'N%'."""
    text = values.astype("string")
    percent = text.str.endswith("%").fillna(False)
    raw = text.str.rstrip("%")
    out = pd.to_numeric(raw, errors="coerce")
    # Normalize row-wise rather than guessing one dataset-wide unit.  This is
    # robust to migrations that temporarily mix 0.72, 72 and "72%".
    out = out.where((out <= 1) & ~percent, out / 100.0)
    invalid = out.notna() & ~out.between(0, 1)
    if invalid.any():
        examples = sorted(out[invalid].unique())[:5]
        raise ValueError(f"snap share outside [0, 1]: {examples}")
    return out.astype(float)


def add_targets(panel: pd.DataFrame, thresholds: Sequence[float] = TARGET_THRESHOLDS) -> pd.DataFrame:
    """Add observed-label eligibility and objective participation targets."""
    required = set(KEY + ["participation_state", "offense_pct", "offense_snaps"])
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"canonical panel missing required columns: {sorted(missing)}")
    if panel[KEY].isna().any().any():
        raise ValueError("canonical panel missing player/week identity")
    out = panel.copy()
    out["snap_share"] = normalize_snap_share(out["offense_pct"])
    observed = out["participation_state"].isin(["confirmed_played", "confirmed_zero_snaps"])
    if (observed & out["snap_share"].isna()).any():
        raise ValueError("snap-observed row has no usable offense_pct")
    out["label_observed"] = observed.astype("int8")
    for threshold in thresholds:
        if not 0 <= threshold < 1:
            raise ValueError(f"invalid participation threshold: {threshold}")
        name = target_name(threshold)
        if threshold == 0:
            label = pd.to_numeric(out["offense_snaps"], errors="coerce").gt(0)
        else:
            label = out["snap_share"].ge(threshold)
        out[name] = label.where(observed).astype("Float64")
    return out


def target_name(threshold: float) -> str:
    return "meaningful_any_snap" if threshold == 0 else f"meaningful_snap_share_{int(round(threshold * 100)):02d}"


def team_grouping_provenance(panel: pd.DataFrame) -> dict:
    """Summarize whether current team can safely form a non-model group key.

    ``team`` is never supplied as a categorical predictor.  It only groups
    players whose own *lagged* histories become peer aggregates.  Canonical
    construction resolves it roster-first, so retain the roster-source
    coverage in the experiment artifact for audit rather than assuming it.
    """
    if "team" not in panel:
        raise ValueError("team grouping requires canonical team column")
    team = panel["team"].astype("string").str.strip()
    has_team = team.notna() & team.ne("")
    roster_source = panel.get("roster_source", pd.Series(index=panel.index, dtype="object"))
    return {
        "grouping_key": ["season", "week", "team", "position"],
        "team_is_model_feature": False,
        "current_week_outcomes_used": False,
        "rows": int(len(panel)),
        "rows_with_team": int(has_team.sum()),
        "team_coverage": float(has_team.mean()) if len(panel) else 0.0,
        "roster_backed_team_rows": int((has_team & roster_source.notna()).sum()),
        "roster_backed_team_coverage": float((has_team & roster_source.notna()).mean()) if len(panel) else 0.0,
        "roster_sources": roster_source.dropna().value_counts().to_dict(),
    }


def _add_team_competition_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add team-position peer signals assembled exclusively from lagged data."""
    out = frame.copy()
    for feature in TEAM_COMPETITION_FEATURES:
        out[feature] = np.nan

    valid = (
        out["team"].notna() & out["team"].astype("string").str.strip().ne("")
        & out["position"].isin(POSITIONS)
    )
    group_columns = ["season", "week", "team", "position"]
    group = out.loc[valid, group_columns]

    def peer_statistics(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Return peer count, sum, mean and max for a lagged numeric signal."""
        known = values.notna()
        count = known.groupby([group[c] for c in group_columns], sort=False).transform("sum") - known.astype(int)
        total = values.fillna(0).groupby([group[c] for c in group_columns], sort=False).transform("sum") - values.fillna(0)
        mean = total / count.replace(0, np.nan)
        first_max = values.groupby([group[c] for c in group_columns], sort=False).transform("max")
        max_ties = values.eq(first_max).groupby([group[c] for c in group_columns], sort=False).transform("sum")
        second_max = values.where(values.lt(first_max)).groupby(
            [group[c] for c in group_columns], sort=False
        ).transform("max")
        peer_max = first_max.where(values.ne(first_max) | max_ties.gt(1), second_max)
        peer_max = peer_max.where(count.gt(0))
        return count.astype(float), total, mean, peer_max

    lag1 = pd.to_numeric(out.loc[valid, "snap_share_lag1"], errors="coerce")
    count, total, mean, peer_max = peer_statistics(lag1)
    out.loc[valid, "peer_history_count"] = count
    out.loc[valid, "peer_snap_share_lag1_sum"] = total.where(count.gt(0))
    out.loc[valid, "peer_snap_share_lag1_mean"] = mean
    out.loc[valid, "peer_snap_share_lag1_max"] = peer_max
    peer_square_sum = (lag1.fillna(0) ** 2).groupby([group[c] for c in group_columns], sort=False).transform("sum") - lag1.fillna(0) ** 2
    peer_total = total.where(count.gt(0))
    out.loc[valid, "peer_snap_share_lag1_hhi"] = (peer_square_sum / peer_total.pow(2)).where(peer_total.gt(0), 0.0)

    roll3 = pd.to_numeric(out.loc[valid, "snap_share_roll3"], errors="coerce")
    roll3_count, _, roll3_mean, roll3_max = peer_statistics(roll3)
    out.loc[valid, "peer_snap_share_roll3_mean"] = roll3_mean.where(roll3_count.gt(0))
    out.loc[valid, "peer_snap_share_roll3_max"] = roll3_max

    meaningful = pd.to_numeric(out.loc[valid, f"{target_name(PRIMARY_THRESHOLD)}_lag1"], errors="coerce")
    meaningful_known = meaningful.notna()
    peer_known = meaningful_known.groupby([group[c] for c in group_columns], sort=False).transform("sum") - meaningful_known.astype(int)
    peer_positive = meaningful.ge(1).groupby([group[c] for c in group_columns], sort=False).transform("sum") - meaningful.ge(1).astype(int)
    out.loc[valid, "peer_meaningful_10_lag1_count"] = peer_positive.where(peer_known.gt(0))
    return out


def build_causal_features(panel: pd.DataFrame, include_pregame_injury: bool = False) -> pd.DataFrame:
    """Build player-history features with a strict one-game minimum lag.

    The function sorts internally and restores chronological order.  A future
    row can never affect an earlier row.  Same-week usage, fantasy points,
    current status, team and opponent are intentionally absent from FEATURES.
    """
    out = add_targets(panel)
    if out.duplicated(KEY).any():
        raise ValueError("duplicate player/season/week rows")
    out = out.sort_values(KEY).reset_index(drop=True)
    out["is_home"] = out.get("home_away", pd.Series(index=out.index, dtype="object")).eq("home").astype("int8")
    # Injury score is a same-week feature only when it was enriched through
    # the repository's kickoff-filtered cache path.  Otherwise neutralize it
    # rather than accidentally trusting an unproven column from a caller.
    if include_pregame_injury and "injury_score" in out:
        out["injury_score"] = pd.to_numeric(out["injury_score"], errors="coerce").clip(0, 1).fillna(1.0)
    else:
        out["injury_score"] = 1.0

    # Rank is measured after each game, then lagged with the rest of player
    # history.  It is not a retrospective/manual WR1/RB1 label.
    out["team_position_rank_actual"] = out.groupby(
        ["season", "week", "team", "position"], dropna=False
    )["snap_share"].rank(method="min", ascending=False)

    parts = []
    for _, group in out.groupby("player_id", sort=False):
        g = group.sort_values(["season", "week"]).copy()
        observed = g["label_observed"].astype(bool)
        played = g[target_name(0)].astype(float)
        share = g["snap_share"]

        g["prior_games_observed"] = observed.cumsum().shift(fill_value=0)
        g["prior_games_played"] = played.fillna(0).cumsum().shift(fill_value=0)
        zero = (observed & played.eq(0)).astype(int)
        g["prior_zero_snap_games"] = zero.cumsum().shift(fill_value=0)
        unknown = (~observed).astype(int)
        g["prior_unknown_games"] = unknown.cumsum().shift(fill_value=0)
        g["snap_share_lag1"] = share.shift(1)
        g["played_lag1"] = played.shift(1)
        g["team_position_rank_lag1"] = g["team_position_rank_actual"].shift(1)
        g["status_lag1"] = g.get("status", pd.Series(index=g.index, dtype="object")).shift(1).fillna("UNKNOWN")

        prior_share = share.shift(1)
        prior_played = played.shift(1)
        g["snap_share_roll3"] = prior_share.rolling(3, min_periods=1).mean()
        g["snap_share_roll5"] = prior_share.rolling(5, min_periods=1).mean()
        g["snap_share_roll3_known"] = prior_share.notna().astype(int).rolling(3, min_periods=1).sum()
        g["played_roll3"] = prior_played.rolling(3, min_periods=1).mean()
        for threshold in TARGET_THRESHOLDS:
            target = target_name(threshold)
            prior_target = g[target].astype(float).shift(1)
            g[f"{target}_lag1"] = prior_target
            g[f"{target}_roll3"] = prior_target.rolling(3, min_periods=1).mean()
        g["cold_start"] = g["prior_games_observed"].eq(0).astype("int8")

        # Schedule distance since the last observed label.  This is a count of
        # player rows, not calendar weeks, and intentionally includes unknowns.
        observed_ord = pd.Series(np.where(observed, np.arange(len(g)), np.nan), index=g.index)
        last_observed = observed_ord.ffill().shift(1)
        g["weeks_since_observed"] = np.arange(len(g)) - last_observed.to_numpy()
        parts.append(g)

    result = pd.concat(parts).sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    result = _add_team_competition_features(result)
    absent = set(FEATURES) - set(result.columns)
    if absent:
        raise AssertionError(f"feature builder failed to create: {sorted(absent)}")
    return result


def _preprocessor(scale: bool, feature_columns: Sequence[str] = FEATURES) -> ColumnTransformer:
    numeric_features = [c for c in feature_columns if c in NUMERIC_FEATURES or c in TEAM_COMPETITION_FEATURES]
    categorical_features = [c for c in feature_columns if c in CATEGORICAL_FEATURES]
    numeric_steps = [("impute", SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    ))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_features),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ], remainder="drop")


def make_classifier(
    kind: str, random_state: int = 42, feature_columns: Sequence[str] = FEATURES,
) -> Pipeline:
    if kind == "logistic":
        model = LogisticRegression(max_iter=2000, C=1.0, class_weight=None, random_state=random_state)
        return Pipeline([("features", _preprocessor(scale=True, feature_columns=feature_columns)), ("model", model)])
    if kind == "hist_gbm":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=random_state,
        )
        return Pipeline([("features", _preprocessor(scale=False, feature_columns=feature_columns)), ("model", model)])
    raise ValueError(f"unknown classifier: {kind}")


def make_opportunity_regressor(
    random_state: int = 42, feature_columns: Sequence[str] = FEATURES,
) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(scale=False, feature_columns=feature_columns)),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=200,
            max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, random_state=random_state,
        )),
    ])


def fit_asof_week_models(
    frame: pd.DataFrame, predict_season: int, predict_week: int, random_state: int = 42,
) -> tuple[Pipeline, Pipeline]:
    """Fit only outcomes available before ``(predict_season, predict_week)``."""
    validate_causal_contract(frame)
    historical = frame["season"].lt(predict_season)
    same_season_past = frame["season"].eq(predict_season) & frame["week"].lt(predict_week)
    train = frame[(historical | same_season_past) & frame["label_observed"].eq(1)]
    primary = target_name(PRIMARY_THRESHOLD)
    if train.empty or train[primary].nunique() < 2:
        raise ValueError(f"insufficient observed primary labels before {predict_season} week {predict_week}")
    classifier = make_classifier("hist_gbm", random_state).fit(train[FEATURES], train[primary].astype(int))
    positive = train[train[primary].eq(1)]
    if len(positive) < 100:
        raise ValueError(f"insufficient meaningful-participation rows before {predict_season} week {predict_week}")
    regressor = make_opportunity_regressor(random_state).fit(positive[FEATURES], positive["snap_share"])
    return classifier, regressor


def predict_asof_week(
    frame: pd.DataFrame, predict_season: int, predict_week: int, random_state: int = 42,
) -> pd.DataFrame:
    """Predict one week using only labels and features knowable before kickoff."""
    classifier, regressor = fit_asof_week_models(frame, predict_season, predict_week, random_state)
    target = frame[frame["season"].eq(predict_season) & frame["week"].eq(predict_week)].copy()
    if target.empty:
        return pd.DataFrame(columns=KEY + ["team", "position", "participation_probability",
                                            "conditional_snap_share", "expected_snap_share"])
    probability = classifier.predict_proba(target[FEATURES])[:, 1]
    conditional = np.clip(regressor.predict(target[FEATURES]), 0, 1)
    return target[KEY + ["team", "position"]].assign(
        participation_probability=probability,
        conditional_snap_share=conditional,
        expected_snap_share=probability * conditional,
    )


def fit_preseason_models(frame: pd.DataFrame, predict_season: int, random_state: int = 42) -> tuple[Pipeline, Pipeline]:
    """Compatibility wrapper for a true pre-season (Week 1) fit."""
    return fit_asof_week_models(frame, predict_season, predict_week=1, random_state=random_state)


def predict_preseason(frame: pd.DataFrame, predict_season: int, random_state: int = 42) -> pd.DataFrame:
    """Produce Week 1-only pre-season predictions; later weeks use ``predict_asof_week``."""
    return predict_asof_week(frame, predict_season, predict_week=1, random_state=random_state)


@dataclass(frozen=True)
class SeasonFold:
    test_season: int
    train_index: pd.Index
    test_index: pd.Index


def expanding_season_folds(df: pd.DataFrame, min_train_seasons: int = 3) -> Iterator[SeasonFold]:
    """Yield strict expanding-window folds; no training season reaches test."""
    seasons = sorted(int(s) for s in pd.Series(df["season"]).dropna().unique())
    for offset in range(min_train_seasons, len(seasons)):
        test = seasons[offset]
        train_idx = df.index[df["season"] < test]
        test_idx = df.index[df["season"] == test]
        if len(train_idx) and len(test_idx):
            yield SeasonFold(test, train_idx, test_idx)


def feature_columns_for_ablation(name: str) -> list[str]:
    groups = {
        "history": ["week", "is_home", "position", "status_lag1", "cold_start",
                    "prior_games_observed", "prior_games_played", "prior_zero_snap_games",
                    "prior_unknown_games", "weeks_since_observed", "played_lag1", "played_roll3"],
        "usage": FEATURES,
        "role": FEATURES,
    }
    if name not in groups:
        raise ValueError(f"unknown ablation: {name}")
    cols = groups[name]
    if name == "usage":
        return [c for c in cols if c != "team_position_rank_lag1"]
    return list(cols)


def validate_causal_contract(frame: pd.DataFrame) -> None:
    """Fail on target/current-outcome columns accidentally entering FEATURES."""
    forbidden = {"snap_share", "offense_pct", "offense_snaps", "fantasy_points",
                 "participation_state", "has_stats_row", "label_observed", "status",
                 "team_position_rank_actual"}
    overlap = forbidden & set(FEATURES)
    if overlap:
        raise ValueError(f"current-game/leaky features declared: {sorted(overlap)}")
    if frame[FEATURES].shape[1] != len(FEATURES):
        raise ValueError("feature contract contains duplicate columns")
    missing = set(TEAM_COMPETITION_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"team competition features missing: {sorted(missing)}")
