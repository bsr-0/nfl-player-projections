"""A week-1 model for players with no NFL history. MEASURED AND REJECTED.

Kept because it is the record of the attempt, and because it is the arm that
showed opponent strength carries real signal. It is NOT production: scored
over 2021-2025 it lost to the published `season total / 17` by 0.32 MAE
(95% CI [+0.03, +0.61]), winning in 1 season of 5. See GAPS.md 2026-09-05.

`season total / 17` prices a cold-start player almost entirely on draft
capital, because that is what the season model has for someone with no prior
season. Two things it never looks at are known before kickoff and are
specifically about week 1:

    where he landed   how his new team distributed production at his position
                      last year, and how fast it played
    who he opens on   what that defence gave up to his position last year

This fits those directly. Everything about the teams comes from season S-1;
the only season-S facts used are the ones the schedule and a preseason depth
chart already tell you -- which team he is on, who they open against, and
whether it is at home.

Betting lines are deliberately absent. `schedule.spread_line` and `total_line`
are populated through 2024 and empty for 2025 and 2026, so a model leaning on
them could be measured but never served.

The population is players whose first season in `player_weekly_stats` is the
target season, scored on their week-1 row.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import DB_PATH

POSITIONS = ("QB", "RB", "WR", "TE")

# Round 8 and pick 300 stand in for undrafted, matching the encoding
# `preseason_features` already uses for the same idea.
UNDRAFTED_ROUND = 8
UNDRAFTED_PICK = 300

FEATURES = [
    "draft_round", "draft_pick", "is_undrafted",
    "team_fp_at_pos", "team_plays", "team_pass_att", "team_rush_att",
    "team_neutral_pass_rate", "team_points",
    "opp_fp_allowed_at_pos", "opp_points_allowed",
    "opp_pass_yards_allowed", "opp_rush_yards_allowed",
    "is_home", "pos_QB", "pos_RB", "pos_WR", "pos_TE",
]

RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def _connect(con):
    return sqlite3.connect(str(DB_PATH)) if con is None else con


def _season_tables(con) -> tuple:
    """Per-team, per-season means. Every row is a completed season."""
    offense = pd.read_sql("""
        SELECT team, season,
               AVG(fantasy_points_produced_qb) AS off_qb,
               AVG(fantasy_points_produced_rb) AS off_rb,
               AVG(fantasy_points_produced_wr) AS off_wr,
               AVG(fantasy_points_produced_te) AS off_te
        FROM team_offense_stats GROUP BY team, season""", con)
    pace = pd.read_sql("""
        SELECT team, season,
               AVG(total_plays) AS team_plays,
               AVG(pass_attempts) AS team_pass_att,
               AVG(rush_attempts) AS team_rush_att,
               AVG(neutral_pass_rate) AS team_neutral_pass_rate,
               AVG(points_scored) AS team_points
        FROM team_stats GROUP BY team, season""", con)
    defense = pd.read_sql("""
        SELECT team, season,
               AVG(points_allowed) AS opp_points_allowed,
               AVG(passing_yards_allowed) AS opp_pass_yards_allowed,
               AVG(rushing_yards_allowed) AS opp_rush_yards_allowed,
               AVG(fantasy_points_allowed_qb) AS def_qb,
               AVG(fantasy_points_allowed_rb) AS def_rb,
               AVG(fantasy_points_allowed_wr) AS def_wr,
               AVG(fantasy_points_allowed_te) AS def_te
        FROM team_defense_stats GROUP BY team, season""", con)
    return offense.merge(pace, on=["team", "season"], how="outer"), defense


def build_week1_rows(seasons, con=None) -> pd.DataFrame:
    """One row per cold-start player with a week-1 game, features attached."""
    owns_con = con is None
    con = _connect(con)
    try:
        debut = pd.read_sql(
            "SELECT player_id, MIN(season) AS debut FROM player_weekly_stats "
            "GROUP BY player_id", con)
        week1 = pd.read_sql(
            "SELECT player_id, season, team, opponent, home_away, snap_share, "
            "       fantasy_points AS actual "
            "FROM player_weekly_stats WHERE week = 1 "
            "AND fantasy_points IS NOT NULL", con)
        positions = pd.read_sql(
            "SELECT player_id, season, position FROM player_weekly_stats "
            "WHERE week = 1 AND position IS NOT NULL", con) \
            if _has_column(con, "player_weekly_stats", "position") else None
        draft = pd.read_sql(
            "SELECT player_id, draft_season, draft_round, draft_pick, position "
            "FROM draft_picks_v2 WHERE player_id IS NOT NULL", con)
        offense, defense = _season_tables(con)
    finally:
        if owns_con:
            con.close()

    rows = week1.merge(debut, on="player_id")
    rows = rows[rows["debut"] == rows["season"]]
    rows = rows[rows["season"].isin(list(seasons))]

    draft = draft.drop_duplicates("player_id")
    rows = rows.merge(draft.rename(columns={"draft_season": "_ds"}),
                      on="player_id", how="left")
    if positions is not None:
        rows = rows.merge(positions.drop_duplicates(["player_id", "season"]),
                          on=["player_id", "season"], how="left",
                          suffixes=("", "_wk"))
        rows["position"] = rows["position_wk"].fillna(rows["position"])
    rows = rows[rows["position"].isin(POSITIONS)].copy()

    rows["is_undrafted"] = rows["draft_round"].isna().astype(int)
    rows["draft_round"] = rows["draft_round"].fillna(UNDRAFTED_ROUND)
    rows["draft_pick"] = rows["draft_pick"].fillna(UNDRAFTED_PICK)

    # Everything about the teams is last season's. Nothing here has seen a
    # snap of the season being predicted.
    rows["prior"] = rows["season"] - 1
    rows = rows.merge(offense.rename(columns={"season": "prior"}),
                      left_on=["team", "prior"], right_on=["team", "prior"],
                      how="left")
    rows = rows.merge(defense.rename(columns={"season": "prior",
                                              "team": "opponent"}),
                      left_on=["opponent", "prior"],
                      right_on=["opponent", "prior"], how="left")

    pos = rows["position"]
    rows["team_fp_at_pos"] = _pick_by_position(rows, "off_", pos)
    rows["opp_fp_allowed_at_pos"] = _pick_by_position(rows, "def_", pos)
    rows["is_home"] = (rows["home_away"].astype(str).str.lower()
                       .eq("home").astype(int))
    for p in POSITIONS:
        rows[f"pos_{p}"] = pos.eq(p).astype(int)
    return rows


def _has_column(con, table, column) -> bool:
    cols = pd.read_sql(f"PRAGMA table_info({table})", con)["name"].tolist()
    return column in cols


def _pick_by_position(rows: pd.DataFrame, prefix: str, pos: pd.Series):
    """The column for each row's own position -- def_wr for a receiver."""
    out = pd.Series(np.nan, index=rows.index, dtype=float)
    for p in POSITIONS:
        col = f"{prefix}{p.lower()}"
        if col in rows.columns:
            out = out.mask(pos.eq(p), rows[col])
    return out


class Week1MatchupModel:
    """Ridge on the landing spot and the opening matchup.

    Ridge rather than a tree: a cold-start week-1 panel is a few hundred rows
    against eighteen features, which is where a boosted model spends its time
    memorising individual seasons. Alpha is chosen by inner CV on the training
    seasons rather than picked, since the sample changes as the walk-forward
    advances.
    """

    def __init__(self, alphas=RIDGE_ALPHAS, random_state: int = 0):
        self.alphas = alphas
        self.random_state = random_state
        self.model = None
        self.medians = None
        self.alpha_ = None

    def _matrix(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = rows.reindex(columns=FEATURES).astype(float)
        return x.fillna(self.medians)

    def fit(self, rows: pd.DataFrame):
        x_raw = rows.reindex(columns=FEATURES).astype(float)
        # A team that did not exist last season (relocation, expansion) leaves
        # holes; the training median stands in, and never a zero, which would
        # assert "this defence allowed nothing".
        self.medians = x_raw.median()
        x, y = self._matrix(rows), rows["actual"].astype(float)

        best, best_score = None, np.inf
        folds = KFold(n_splits=min(5, max(2, len(rows) // 40)), shuffle=True,
                      random_state=self.random_state)
        for alpha in self.alphas:
            errors = []
            for train_idx, test_idx in folds.split(x):
                pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
                errors.append(np.abs(pipe.predict(x.iloc[test_idx])
                                     - y.iloc[test_idx]).mean())
            score = float(np.mean(errors))
            if score < best_score:
                best, best_score = alpha, score
        self.alpha_ = best
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=best))
        self.model.fit(x, y)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        # Fantasy points do not go negative in this scoring; clip rather than
        # publish an impossible number.
        return np.clip(self.model.predict(self._matrix(rows)), 0.0, None)


class Week1HurdleModel:
    """Two questions instead of one: does he have a role, and what then?
    MEASURED AND REJECTED -- MAE 4.29 against the published number's 3.70.

        E[points] = P(role) x E[points | role] + (1 - P(role)) x E[points | none]

    The split is not decoration. Measured 2026-09-05, `season total / 17` wins
    the composite cold-start problem (MAE 3.70) almost entirely by answering
    the first question: condition on a rookie who actually played a quarter of
    his team's snaps and its R2 falls from +0.270 to +0.090, below every
    team-context arm, with bias -1.68. Team context and the opening matchup are
    the better answer to the second question and the worse answer to the first.
    This gives each half the features that win it.

    "Role" is a realised snap share at or above `role_threshold`. That is a
    label, not a feature -- nothing here reads snap share at predict time.
    """

    def __init__(self, role_threshold: float = 0.25,
                 alphas=RIDGE_ALPHAS, cs=(0.1, 1.0, 10.0),
                 random_state: int = 0):
        self.role_threshold = role_threshold
        self.alphas, self.cs = alphas, cs
        self.random_state = random_state
        self.role_model = self.given_role = None
        self.without_role = 0.0
        self.medians = None
        self.params_ = None

    def _matrix(self, rows: pd.DataFrame) -> pd.DataFrame:
        return rows.reindex(columns=FEATURES).astype(float).fillna(self.medians)

    def _fit_once(self, rows, c, alpha):
        x, y = self._matrix(rows), rows["actual"].astype(float)
        has_role = (rows["snap_share"].astype(float)
                    >= self.role_threshold).values
        role = make_pipeline(StandardScaler(),
                             LogisticRegression(C=c, max_iter=2000))
        given = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        # Degenerate fold (every row one class, or nobody without a role):
        # fall back to a single Ridge on everything rather than crash.
        if has_role.all() or not has_role.any():
            given.fit(x, y)
            return None, given, float(y.mean())
        role.fit(x, has_role)
        given.fit(x[has_role], y[has_role])
        return role, given, float(y[~has_role].mean())

    def _predict_with(self, parts, rows):
        role, given, without = parts
        x = self._matrix(rows)
        point = given.predict(x)
        if role is None:
            return np.clip(point, 0.0, None)
        p = role.predict_proba(x)[:, 1]
        return np.clip(p * point + (1 - p) * without, 0.0, None)

    def fit(self, rows: pd.DataFrame):
        self.medians = rows.reindex(columns=FEATURES).astype(float).median()
        folds = KFold(n_splits=min(5, max(2, len(rows) // 40)), shuffle=True,
                      random_state=self.random_state)
        best, best_score = None, np.inf
        for c in self.cs:
            for alpha in self.alphas:
                errors = []
                for train_idx, test_idx in folds.split(rows):
                    parts = self._fit_once(rows.iloc[train_idx], c, alpha)
                    pred = self._predict_with(parts, rows.iloc[test_idx])
                    errors.append(np.abs(
                        pred - rows.iloc[test_idx]["actual"].values).mean())
                score = float(np.mean(errors))
                if score < best_score:
                    best, best_score = (c, alpha), score
        self.params_ = {"C": best[0], "alpha": best[1]}
        self.role_model, self.given_role, self.without_role = self._fit_once(
            rows, *best)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.given_role is None:
            raise RuntimeError("call fit() first")
        return self._predict_with(
            (self.role_model, self.given_role, self.without_role), rows)
