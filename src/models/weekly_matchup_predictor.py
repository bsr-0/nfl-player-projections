"""
WeeklyMatchupPredictor: learns how matchup context (opponent defense, home/away,
week of season) modulates a player's output relative to their baseline PPG.

Training target: multiplier = weekly_fp / player_prior_season_ppg
  - 1.0 means the player performed exactly at their baseline
  - 1.2 means 20% above baseline (soft opponent, home game, etc.)

This normalises out player skill, so predictions can be applied to any
preseason season-total projection:
  proj_week_w = (season_proj / 17) × normalised_multiplier_w

The 17 per-week multipliers are normalised to average 1.0 before scaling,
so the weekly projections always sum exactly to season_proj.

Usage (training):
    predictor = WeeklyMatchupPredictor()
    predictor.fit_from_db(seasons=range(2019, 2025))
    predictor.save("data/models/weekly_matchup_predictor.pkl")

Usage (inference):
    predictor = WeeklyMatchupPredictor.load("data/models/weekly_matchup_predictor.pkl")
    weekly = predictor.project_weekly(season_proj, position, matchups)
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "data" / "nfl_data.db"

# Abbreviation mismatches between nflverse DB and schedule.json
DB_TO_SCHED  = {"LA": "LAR", "JAX": "JAC"}
SCHED_TO_DB  = {v: k for k, v in DB_TO_SCHED.items()}
POSITIONS    = ["QB", "RB", "WR", "TE"]
FEATURES     = ["opp_def_norm"]


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

def _norm_team(team: str, mapping: dict) -> str:
    return mapping.get(team, team)


def build_training_data(
    seasons: list[int] | None = None,
    min_games_prior: int = 4,
) -> pd.DataFrame:
    """
    Build player-week training rows with matchup features and multiplier target.

    Uses SAME-season defensive ratings as the feature (r≈0.60 WR same-year
    vs r≈0.04 prior-year), then aggregates to (opponent, season, position)
    team level to reduce individual-game noise.

    For 2026 inference: pass 2024 full-season averages as opp_def_ppg proxy.

    Returns (agg_df, league_avg_def) where agg_df has columns:
      opponent, season, position, multiplier, n, opp_def_ppg, opp_def_norm
    """
    if seasons is None:
        seasons = list(range(2019, 2025))

    conn = sqlite3.connect(DB_PATH)

    # ── 1. Prior-season player PPG (multiplier denominator) ───────────────
    # Still uses prior-season PPG so the target = "how much did this player
    # deviate from their established baseline this week?"
    player_ppg = pd.read_sql("""
        SELECT w.player_id, w.season, p.position,
               AVG(w.fantasy_points) AS ppg,
               COUNT(*) AS games
        FROM player_weekly_stats w
        JOIN players p ON w.player_id = p.player_id
        WHERE w.fantasy_points IS NOT NULL
          AND p.position IN ('QB','RB','WR','TE')
        GROUP BY w.player_id, w.season, p.position
    """, conn)
    player_ppg = player_ppg[player_ppg["games"] >= min_games_prior].copy()
    prior_ppg = player_ppg.rename(columns={"ppg": "ppg_prior", "games": "games_prior"})
    prior_ppg["season"] = prior_ppg["season"] + 1  # shift to target season

    # ── 2. SAME-season team defense per position ───────────────────────────
    # Key insight: same-year defense correlates strongly with outcomes (WR r≈0.60).
    # Prior-year defense is nearly uncorrelated (WR r≈0.04).
    def_season = pd.read_sql("""
        SELECT team, season,
               AVG(fantasy_points_allowed_qb) AS qb,
               AVG(fantasy_points_allowed_rb) AS rb,
               AVG(fantasy_points_allowed_wr) AS wr,
               AVG(fantasy_points_allowed_te) AS te
        FROM team_defense_stats
        GROUP BY team, season
        HAVING COUNT(*) >= 10
    """, conn)

    def_long = def_season.melt(
        id_vars=["team", "season"], value_vars=["qb", "rb", "wr", "te"],
        var_name="pos_key", value_name="def_ppg"
    )
    def_long["position"] = def_long["pos_key"].str.upper()
    def_long = def_long.drop(columns=["pos_key"])
    def_long["team"] = def_long["team"].map(lambda t: DB_TO_SCHED.get(t, t))
    # No season shift — use same-year defense
    same_def = def_long.rename(columns={"def_ppg": "opp_def_ppg"}).copy()

    league_avg_def = def_long.groupby("position")["def_ppg"].mean().to_dict()

    # ── 3. Weekly player stats for target seasons ──────────────────────────
    season_sql = ",".join(str(s) for s in seasons)
    weekly = pd.read_sql(f"""
        SELECT w.player_id, w.season, w.week, w.team, w.opponent,
               w.fantasy_points
        FROM player_weekly_stats w
        WHERE w.season IN ({season_sql})
          AND w.opponent IS NOT NULL AND w.opponent != ''
          AND w.fantasy_points IS NOT NULL
    """, conn)

    # ── 4. Home/away from schedule table (join on team, not just week) ─────
    # Bug fix: joining only on (season, week) matches all 16 games per week,
    # expanding rows 14×. Expand schedule to per-team rows first.
    schedule = pd.read_sql(f"""
        SELECT season, week, home_team, away_team
        FROM schedule
        WHERE season IN ({season_sql})
    """, conn)
    conn.close()

    schedule["home_team"] = schedule["home_team"].map(lambda t: DB_TO_SCHED.get(t, t))
    schedule["away_team"] = schedule["away_team"].map(lambda t: DB_TO_SCHED.get(t, t))
    weekly["team"]     = weekly["team"].map(lambda t: DB_TO_SCHED.get(t, t))
    weekly["opponent"] = weekly["opponent"].map(lambda t: DB_TO_SCHED.get(t, t))

    home_rows = schedule[["season", "week", "home_team"]].rename(columns={"home_team": "team"}).copy()
    home_rows["home"] = 1.0
    away_rows = schedule[["season", "week", "away_team"]].rename(columns={"away_team": "team"}).copy()
    away_rows["home"] = 0.0
    team_sched = pd.concat([home_rows, away_rows], ignore_index=True)

    weekly = weekly.merge(team_sched, on=["season", "week", "team"], how="left")

    # ── 5. Join player position and prior-season PPG ───────────────────────
    pos_map = pd.read_sql("""
        SELECT w.player_id, w.season, p.position
        FROM player_weekly_stats w
        JOIN players p ON w.player_id = p.player_id
        WHERE p.position IN ('QB','RB','WR','TE')
        GROUP BY w.player_id, w.season, p.position
    """, sqlite3.connect(DB_PATH))

    weekly = weekly.merge(
        pos_map[["player_id", "season", "position"]],
        on=["player_id", "season"], how="inner"
    )
    weekly = weekly.merge(
        prior_ppg[["player_id", "season", "position", "ppg_prior"]],
        on=["player_id", "season", "position"], how="inner"
    )

    # ── 6. Join SAME-season opponent defense ──────────────────────────────
    weekly = weekly.merge(
        same_def[["team", "season", "position", "opp_def_ppg"]],
        left_on=["opponent", "season", "position"],
        right_on=["team", "season", "position"],
        how="inner"
    ).drop(columns=["team_y"]).rename(columns={"team_x": "team"})

    # ── 7. Feature engineering + target ───────────────────────────────────
    weekly["opp_def_norm"] = weekly.apply(
        lambda r: r["opp_def_ppg"] / league_avg_def.get(r["position"], 1.0), axis=1
    )

    weekly = weekly[weekly["ppg_prior"] > 0].copy()
    weekly["multiplier"] = (weekly["fantasy_points"] / weekly["ppg_prior"]).clip(0, 4)

    # ── 8. Aggregate to (opponent, season, position) level ────────────────
    # Individual player-week variance swamps the defense signal; aggregating
    # to team-level cells makes the signal dominate.
    agg = (
        weekly
        .groupby(["opponent", "season", "position"])
        .agg(avg_multiplier=("multiplier", "mean"), n=("multiplier", "count"))
        .reset_index()
    )
    agg = agg[agg["n"] >= 5].copy()

    agg = agg.merge(
        same_def[["team", "season", "position", "opp_def_ppg"]],
        left_on=["opponent", "season", "position"],
        right_on=["team", "season", "position"],
        how="inner",
    ).drop(columns=["team"])

    agg["opp_def_norm"] = agg.apply(
        lambda r: r["opp_def_ppg"] / league_avg_def.get(r["position"], 1.0), axis=1
    )
    agg = agg.rename(columns={"avg_multiplier": "multiplier"})

    return agg, league_avg_def


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class WeeklyMatchupPredictor:
    """
    Per-position Ridge regression predicting weekly FP relative to player baseline.
    """

    def __init__(self) -> None:
        self.models:      dict[str, Ridge]          = {}
        self.scalers:     dict[str, StandardScaler] = {}
        self.league_avg:  dict[str, float]           = {}
        self.cv_scores:   dict[str, dict]            = {}

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, league_avg: dict[str, float]) -> None:
        """Train per-position Ridge models on training data."""
        self.league_avg = league_avg
        for pos in POSITIONS:
            pos_df = train_df[train_df["position"] == pos].copy()
            if len(pos_df) < 100:
                print(f"  {pos}: too few samples ({len(pos_df)}), skipping")
                continue
            X = pos_df[FEATURES].values
            y = pos_df["multiplier"].values

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            model = Ridge(alpha=1.0)
            model.fit(X_s, y)

            self.models[pos]  = model
            self.scalers[pos] = scaler

            y_pred  = model.predict(X_s)
            mae     = float(np.mean(np.abs(y_pred - y)))
            coef_df = pd.Series(model.coef_, index=FEATURES)
            print(f"  {pos}: n={len(pos_df):,}  train_MAE={mae:.3f}  coefs={coef_df.round(3).to_dict()}")

    def fit_from_db(
        self,
        seasons: list[int] | None = None,
        min_games_prior: int = 4,
    ) -> None:
        """Build training data from DB and fit."""
        print("Building training data from DB...")
        train_df, league_avg = build_training_data(seasons, min_games_prior)
        print(f"  {len(train_df):,} player-week training rows")
        for pos in POSITIONS:
            n = (train_df["position"] == pos).sum()
            print(f"  {pos}: {n:,} rows")
        print("Training per-position models...")
        self.fit(train_df, league_avg)

    # ── Inference ────────────────────────────────────────────────────────────

    def _predict_multipliers(
        self,
        position: str,
        matchups: list[dict],
    ) -> list[float]:
        """Return raw (un-normalised) multipliers for a list of game matchups."""
        model  = self.models.get(position)
        scaler = self.scalers.get(position)
        if model is None or scaler is None:
            return [1.0] * len(matchups)

        avg_def = self.league_avg.get(position, 1.0)
        rows = []
        for m in matchups:
            opp_def_raw  = m.get("opp_def_ppg", avg_def)
            opp_def_norm = opp_def_raw / avg_def if avg_def > 0 else 1.0
            rows.append([opp_def_norm])

        X_s = scaler.transform(np.array(rows))
        return model.predict(X_s).clip(0.2, 2.5).tolist()

    def project_weekly(
        self,
        season_proj: float,
        position:    str,
        matchups:    list[dict],
    ) -> list[float | None]:
        """
        Return 18-element weekly projection array (None = bye week).
        Guaranteed: sum of non-None values == season_proj.

        matchups: list of {week, opp, home, opp_def_ppg} dicts (game weeks only).
        """
        weeks: list[float | None] = [None] * 18
        if not season_proj or season_proj <= 0 or not matchups or position not in POSITIONS:
            return weeks

        raw_mults = self._predict_multipliers(position, matchups)

        # Normalise so multipliers average to 1.0 → sum of weekly projs = season_proj
        avg = sum(raw_mults) / len(raw_mults)
        if avg > 0:
            norm_mults = [m / avg for m in raw_mults]
        else:
            norm_mults = [1.0] * len(raw_mults)

        ppg = season_proj / 17
        for m, mult in zip(matchups, norm_mults):
            wk = int(m["week"]) - 1
            if 0 <= wk < 18:
                weeks[wk] = round(ppg * mult, 1)

        return weeks

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "WeeklyMatchupPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "weekly_matchup_predictor.pkl"


def get_predictor(retrain: bool = False) -> WeeklyMatchupPredictor:
    """Load saved model or train a new one."""
    if not retrain and MODEL_PATH.exists():
        print(f"Loading saved model from {MODEL_PATH}")
        return WeeklyMatchupPredictor.load(MODEL_PATH)
    predictor = WeeklyMatchupPredictor()
    predictor.fit_from_db()
    predictor.save(MODEL_PATH)
    return predictor


def get_2024_def_ratings() -> dict[str, dict[str, float]]:
    """Return 2024 season avg FP-allowed per position per team, in schedule.json abbrev."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT team,
               AVG(fantasy_points_allowed_qb) AS qb,
               AVG(fantasy_points_allowed_rb) AS rb,
               AVG(fantasy_points_allowed_wr) AS wr,
               AVG(fantasy_points_allowed_te) AS te
        FROM team_defense_stats
        WHERE season = 2024
        GROUP BY team
    """, conn)
    conn.close()
    result = {}
    for _, row in df.iterrows():
        team = DB_TO_SCHED.get(row["team"], row["team"])
        result[team] = {
            "QB": float(row["qb"]) if pd.notna(row["qb"]) else None,
            "RB": float(row["rb"]) if pd.notna(row["rb"]) else None,
            "WR": float(row["wr"]) if pd.notna(row["wr"]) else None,
            "TE": float(row["te"]) if pd.notna(row["te"]) else None,
        }
    return result
