"""Phase 7 (next_focus.md): 18-week season projection via weekly aggregation.

For real completed seasons (2023-2025): for every possible week a player's
team played (schedule-derived, excludes byes/playoffs), including weeks the
player didn't actually suit up, predict E[PPR | plays] and multiply by a
non-circular P(plays) estimate, sum, and compare to the real season total.

Reuses Predictor.predict()'s mechanism (src/predict.py:260-292) for building
a feature row for a not-yet-played week — carry forward a player's most
recent real game's rolling/lag features, overwrite season/week/opponent via
the schedule, refresh opponent-dependent DVP columns
(FeatureEngineer.refresh_matchup_features). EXTENDS that mechanism (a real
gap found and confirmed with the user this session): refresh_matchup_features
alone leaves Vegas lines and team-level rolling context stale on a carried-
forward row, since those come from a different pipeline stage
(add_external_features / raw team_stats joins) that Predictor.predict() only
runs BEFORE the season/week overwrite, not after. This module refreshes both
explicitly for the target week — see _refresh_vegas_and_weather and
_refresh_team_rolling_context.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGULAR_SEASON_MAX_WEEK = 18

# Team-level roll3_mean columns that come from raw team_stats/team_personnel_stats
# joins (get_all_players_for_training SQL) + _create_causal_rolling_features'
# roll_cols, NOT from FeatureEngineer.refresh_matchup_features or
# add_external_features. Must be refreshed by direct DB lookup for synthetic
# rows. (raw_col -> output roll3_mean col)
_TEAM_STATS_ROLL_COLS = {
    "total_plays": "team_plays_roll3_mean",
    "pace_sec_per_play": "team_pace_sec_per_play_roll3_mean",
    "neutral_pass_rate_oe": "team_neutral_pass_rate_oe_roll3_mean",
}
_TEAM_PERSONNEL_ROLL_COLS = {
    "pct_11": "team_pct_11_personnel_roll3_mean",
    "pct_12": "team_pct_12_personnel_roll3_mean",
    "pct_21": "team_pct_21_personnel_roll3_mean",
    "pct_13": "team_pct_13_personnel_roll3_mean",
}


def possible_weeks_for_team(db, team: str, season: int) -> List[int]:
    """Regular-season weeks (1-18) this team appears in the schedule —
    correctly excludes bye weeks (team absent that week) and playoffs
    (week > 18)."""
    sched = db.get_schedule(season=season, team=team)
    if sched is None or sched.empty:
        return []
    weeks = sorted(int(w) for w in sched["week"].unique() if int(w) <= REGULAR_SEASON_MAX_WEEK)
    return weeks


def estimate_availability_rate(player_id: str, position: str, season: int, db,
                                position_avg_fallback: float = 0.88) -> float:
    """Non-circular P(plays)-per-week estimate: the player's own games-played
    rate in seasons STRICTLY BEFORE `season` (leakage-safe — never uses the
    season being evaluated, since that would leak exactly what's being
    forecast). Falls back to a fixed position-average prior for players with
    no prior-season history (rookies) — same "be honest, don't force a
    number" spirit as tiers.py's 'rookie' tier fallback.

    Deliberately not built on GamesPlayedProjector (season_long_features.py)
    — that hardcodes a 17-game season regardless of actual length (18 weeks
    since 2021), which would systematically bias this rate for recent
    seasons. See GAPS.md Phase 7 notes.
    """
    history = db.get_all_players_for_training(position=position)
    prior = history[(history["player_id"] == player_id) & (history["season"] < season)]
    if prior.empty:
        return position_avg_fallback

    rates = []
    for prior_season, g in prior.groupby("season"):
        team = g["team"].mode().iloc[0] if not g["team"].mode().empty else None
        possible = len(possible_weeks_for_team(db, team, prior_season)) if team else 17
        if possible <= 0:
            continue
        games_played = g["week"].nunique()
        rates.append(min(games_played / possible, 1.0))
    if not rates:
        return position_avg_fallback
    return float(np.mean(rates))


def _compute_team_rolling_context(team: str, season: int, week: int, n: int = 3) -> Dict[str, float]:
    """Direct-lookup replacement for the per-player shift(1).rolling(3) team
    context features — computed from the team's own team_stats/
    team_personnel_stats rows for the n weeks strictly before `week`
    (independent of any specific player, so correct even when the player
    we're projecting for missed those weeks)."""
    out: Dict[str, float] = {}

    import sqlite3
    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))
    try:
        team_stats = pd.read_sql(
            "SELECT * FROM team_stats WHERE team = ? AND season = ? AND week < ? ORDER BY week DESC LIMIT ?",
            conn, params=[team, season, week, n],
        )
        personnel = pd.read_sql(
            "SELECT * FROM team_personnel_stats WHERE team = ? AND season = ? AND week < ? ORDER BY week DESC LIMIT ?",
            conn, params=[team, season, week, n],
        )
    finally:
        conn.close()

    for raw_col, out_col in _TEAM_STATS_ROLL_COLS.items():
        out[out_col] = float(team_stats[raw_col].mean()) if not team_stats.empty and raw_col in team_stats.columns else np.nan
    for raw_col, out_col in _TEAM_PERSONNEL_ROLL_COLS.items():
        out[out_col] = float(personnel[raw_col].mean()) if not personnel.empty and raw_col in personnel.columns else np.nan
    return out


def build_synthetic_week_row(
    player_history_df: pd.DataFrame,
    player_id: str,
    target_season: int,
    target_week: int,
    team: str,
    db,
    feature_engineer,
) -> Optional[pd.DataFrame]:
    """Builds a single-row DataFrame representing what player_id's feature
    row WOULD have been for (target_season, target_week), had they played.

    Generalizes Predictor.predict()'s mechanism (src/predict.py:260-292):
    carry forward the player's most recent REAL row strictly before the
    target week, overwrite season/week/opponent/home_away from the schedule,
    then refresh every week-dependent feature family (not just DVP — see
    module docstring for why refresh_matchup_features alone isn't enough).

    Returns None if the player has no real history before the target week
    (can't retrospectively project a debut without any prior data).
    """
    prior = player_history_df[
        (player_history_df["player_id"] == player_id)
        & ((player_history_df["season"] < target_season)
           | ((player_history_df["season"] == target_season) & (player_history_df["week"] < target_week)))
    ].sort_values(["season", "week"])
    if prior.empty:
        return None

    row = prior.iloc[[-1]].copy()
    row["season"] = target_season
    row["week"] = target_week
    row["team"] = team

    from src.predict import get_schedule_map_for_week
    schedule_map = get_schedule_map_for_week(db, target_season, target_week)
    opponent, home_away = schedule_map.get(team, ("", "unknown"))
    row["opponent"] = opponent
    row["home_away"] = home_away

    # 1. Opponent-dependent DVP (opp_fpts_allowed*) — existing mechanism.
    row = feature_engineer.refresh_matchup_features(row)

    # 2. Vegas lines + weather — team/schedule-keyed, not player-keyed, so
    # re-running the external-features loader against the overwritten
    # season/week/team correctly refreshes them for the target game.
    try:
        from src.data.external_data import add_external_features
        row = add_external_features(row, seasons=[target_season])
    except Exception as e:
        logger.warning("External feature refresh failed for %s %s/%s: %s", player_id, target_season, target_week, e)

    # Conditional-on-playing assumption: don't look up a real injury status
    # for a week the player didn't play (that would leak exactly what
    # estimate_availability_rate is separately, non-circularly, estimating).
    # Neutral/healthy default, matching InjuryDataLoader's own "no report"
    # fallback (src/data/external_data.py INJURY_STATUS_SCORES[None] = 1.0).
    if "injury_score" in row.columns:
        row["injury_score"] = 1.0

    # 3. Team-level rolling context (team_plays/pace/pass-rate-OE/personnel%)
    # — direct DB lookup, independent of this player's own row history.
    team_context = _compute_team_rolling_context(team, target_season, target_week)
    for col, val in team_context.items():
        if col in row.columns:
            row[col] = val

    return row


def run_season_projection(
    positions: Optional[Sequence[str]] = None,
    seasons: Optional[Sequence[int]] = None,
    output_path=None,
) -> pd.DataFrame:
    """For each position's FINAL_CONFIG fold: fit once, then for every player
    with >=1 real game that season, sum P(plays)*E[PPR|plays] across every
    possible week (real prediction where a real row exists, synthetic
    prediction otherwise), and compare to the real season total.
    """
    from pathlib import Path

    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.features.feature_engineering import PositionFeatureEngineer
    from src.models.single_week_ppr.evaluate import (
        DEFAULT_VALIDATION_SEASONS, run_fold, _architectures_for_fold, _append_df_to_csv,
    )
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.windows import window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    seasons = seasons or DEFAULT_VALIDATION_SEASONS
    output_path = output_path or Path("data/experiments/phase7_season_projection.csv")
    positions = list(positions) if positions else POSITIONS
    all_rows: List[dict] = []
    db = DatabaseManager()

    for position in positions:
        cfg = FINAL_CONFIG[position]
        feature_engineer = PositionFeatureEngineer(position)
        available_seasons = sorted(
            db.get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )

        for season in seasons:
            print(f"\n=== SEASON PROJECTION {position} / season={season} / {cfg} ===")
            train_seasons = window_to_season_list(cfg["window"], season, available_seasons)
            if not train_seasons:
                logger.warning("Skipping %s/%s: no training seasons available", position, season)
                continue
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons,
                )
            except Exception as e:
                logger.warning("Fold %s/%s failed to load: %s", position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position].reset_index(drop=True)
            pos_test = test_df[test_df["position"] == position].copy()
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue

            feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
            if not feature_cols:
                logger.warning("Skipping %s/%s: no CAUSAL_FEATURES columns present", position, season)
                continue

            model = _architectures_for_fold()[cfg["architecture"]]
            X_train = pos_train[feature_cols]
            y_train = pos_train["fantasy_points"]
            model.fit(X_train, y_train)

            # Full player-season history (for carrying forward into synthetic rows)
            # includes this fold's train+test frame so mid-season history is present.
            full_history = pd.concat([pos_train, pos_test], ignore_index=True)

            for player_id, g in pos_test.groupby("player_id"):
                real_weeks = set(g["week"].astype(int))
                team = g.sort_values("week")["team"].iloc[0]  # first real game's team this season
                possible_weeks = possible_weeks_for_team(db, team, season)
                if not possible_weeks:
                    continue

                availability_rate = estimate_availability_rate(player_id, position, season, db)

                predicted_total = 0.0
                weeks_predicted = 0
                for week in possible_weeks:
                    if week in real_weeks:
                        row = g[g["week"] == week][feature_cols]
                    else:
                        synth = build_synthetic_week_row(
                            full_history, player_id, season, week, team, db, feature_engineer,
                        )
                        if synth is None:
                            continue
                        missing = [c for c in feature_cols if c not in synth.columns]
                        for c in missing:
                            synth[c] = np.nan
                        row = synth[feature_cols]
                    try:
                        pred = float(model.predict(row)[0])
                    except Exception as e:
                        logger.warning("Predict failed for %s week %s: %s", player_id, week, e)
                        continue
                    predicted_total += pred * availability_rate
                    weeks_predicted += 1

                if weeks_predicted == 0:
                    continue

                actual_total = float(g["fantasy_points"].sum())
                result_row = {
                    "player": player_id, "position": position, "season": season, "team": team,
                    "possible_weeks": len(possible_weeks), "weeks_predicted": weeks_predicted,
                    "games_actually_played": len(real_weeks),
                    "availability_rate": availability_rate,
                    "predicted_season_total": predicted_total,
                    "actual_season_total": actual_total,
                }
                all_rows.append(result_row)
                _append_df_to_csv(pd.DataFrame([result_row]), output_path)

    result = pd.DataFrame(all_rows)
    print(f"\n{len(result)} rows appended to {output_path}")
    return result
