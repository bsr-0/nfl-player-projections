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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .availability import POSITION_AVG_FALLBACK

logger = logging.getLogger(__name__)

# Absolute upper bound across all eras. Use this ONLY to size a loop or bound
# an assertion -- never to decide whether a week is regular season. That
# question is era-dependent and needs `regular_season_max_week(season)`.
REGULAR_SEASON_MAX_WEEK = 18

# Canonical definition lives in config.settings so the feature layer, the
# preseason models and the scripts can all reach it without importing this
# module. Re-exported here because several callers already import it from
# season_projection.
#
# Using a flat 18 admitted one full playoff round into every pre-2021 season
# total -- measured at 480 of 3,339 cold-start rows (14.4%) carrying a
# wild-card game worth 4,166 fantasy points, with mean bias -38.5 against
# -7.0 for clean rows in the same seasons. It hid from the obvious
# `games_played > possible_weeks` check because `possible_weeks_for_player`
# counts any week the player actually played, so the extra week inflated both
# sides at once.
from config.settings import (  # noqa: E402
    LAST_17_WEEK_SEASON, regular_season_max_week, regular_season_week_sql,
)

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
    """Regular-season weeks this team appears in the schedule — correctly
    excludes bye weeks (team absent that week) and playoffs (any week past
    `regular_season_max_week(season)`, which is 17 through 2020 and 18 from
    2021; a flat 18 counted the wild-card round as regular season).

    `season` is coerced to a plain int: sqlite3 does not bind numpy integers,
    so a numpy season silently matches no rows and returns [] — which reads
    downstream as "this team never played", producing zero synthetic weeks
    with no error. A caller iterating `groupby(["player_id", "season"])` gets
    numpy scalars without realising it.
    """
    sched = db.get_schedule(season=int(season), team=str(team))
    if sched is None or sched.empty:
        return []
    max_week = regular_season_max_week(season)
    weeks = sorted(int(w) for w in sched["week"].unique() if int(w) <= max_week)
    return weeks


_ACTIVE_ROSTER_CACHE: Dict[Tuple[str, int], set] = {}
_ROSTER_SEASON_COVERED: Dict[int, bool] = {}


def _season_has_roster_data(season: int) -> bool:
    """Does weekly_rosters cover this season at all? Distinguishes 'this
    player wasn't rostered' from 'we have no roster data to check'."""
    if season not in _ROSTER_SEASON_COVERED:
        import sqlite3
        from config.settings import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM weekly_rosters WHERE season = ? AND game_type = 'REG'",
                [int(season)]).fetchone()[0]
        finally:
            conn.close()
        _ROSTER_SEASON_COVERED[season] = n > 0
    return _ROSTER_SEASON_COVERED[season]


def active_roster_weeks(player_id: str, season: int) -> Optional[set]:
    """Regular-season weeks this player was on an ACTIVE roster.

    Returns None when the season has no roster coverage at all -- callers
    must then fall back to permitting the week rather than silently
    dropping every synthetic row.

    Only status 'ACT' counts. 'INA' (declared inactive for that game), 'DEV'
    (practice squad), 'RES' (IR/PUP/NFI), 'CUT' and 'RET' all mean the
    player could not have taken the field, so "what would he have scored"
    is not a question the data can answer.
    """
    season = int(season)
    if not _season_has_roster_data(season):
        return None
    key = (str(player_id), season)
    if key not in _ACTIVE_ROSTER_CACHE:
        import sqlite3
        from config.settings import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        try:
            rows = conn.execute(
                """SELECT DISTINCT week FROM weekly_rosters
                   WHERE player_id = ? AND season = ? AND game_type = 'REG'
                     AND status = 'ACT'""", [str(player_id), int(season)]).fetchall()
        finally:
            conn.close()
        _ACTIVE_ROSTER_CACHE[key] = {int(r[0]) for r in rows}
    return _ACTIVE_ROSTER_CACHE[key]


def possible_weeks_for_player(
    db, real_team_by_week: Dict[int, str], season: int,
    player_id: Optional[str] = None,
    skip_tracker: Optional["WeekSkipTracker"] = None,
    require_active_roster: bool = True,
) -> Tuple[List[int], Dict[int, str]]:
    """Weeks the player could have played, resolving their team PER WEEK.

    Synthetic candidate weeks are additionally filtered to weeks the player
    was on an ACTIVE roster. A roster audit of 644 pre-debut synthetic QB
    weeks found exactly ONE where the player was a listed starter; 45.7% of
    the manufactured fantasy points came from players who were on IR, on the
    practice squad, declared inactive, or not in the league at all --
    Marcus Mariota projected at ~15/wk for 12 weeks as QB2 behind a healthy
    starter, Kyler Murray at ~15/wk while rehabbing an ACL, and a retired
    Philip Rivers given 13 weeks before his week-15 2025 comeback
    (GAPS.md).

    Deliberately gated on ACTIVE ROSTER, not on being the starter. A
    rostered backup is a legitimate member of the forecast population --
    that he is a backup is for the model's features to express, not for the
    population definition to assume. Filtering to known starters would leak
    the outcome into the population.

    Callers used to pin the whole season to `g.sort_values("week")["team"]
    .iloc[0]` -- the player's first team. After a mid-season trade that
    projects them against their old team's schedule: wrong bye week, wrong
    opponents. Two 2023/2025 QBs came out with 18 weeks against a 17-week
    schedule because they played their FORMER team's bye week for their new
    one (GAPS.md).

    A player's team on a week they played is simply their own row's team.
    For a week they missed, the honest answer is the last team they were
    known to be on -- carried forward, or their first team for weeks before
    they ever appeared.

    Returns (playable weeks, week -> team) where a week is playable if the
    team the player was on THAT week has a game and -- for weeks he did not
    play -- he was on that team's active roster.
    """
    if not real_team_by_week:
        return [], {}

    known_weeks = sorted(real_team_by_week)
    first_team = real_team_by_week[known_weeks[0]]

    schedule_cache: Dict[str, List[int]] = {}

    def team_weeks(team: str) -> List[int]:
        if team not in schedule_cache:
            schedule_cache[team] = possible_weeks_for_team(db, team, season)
        return schedule_cache[team]

    active = (active_roster_weeks(player_id, season)
              if require_active_roster and player_id is not None else None)

    weeks: List[int] = []
    team_by_week: Dict[int, str] = {}
    # Era-aware bound: iterating to a flat 18 let the "he demonstrably played"
    # bypass below admit pre-2021 wild-card weeks even once the schedule
    # lookup excluded them, which is what inflated possible_weeks to 17
    # against a true 16-game schedule for playoff-team players.
    for week in range(1, regular_season_max_week(season) + 1):
        prior = [w for w in known_weeks if w <= week]
        team = real_team_by_week[prior[-1]] if prior else first_team
        team_by_week[week] = team

        # A week the player actually played is playable by definition, even
        # if the schedule lookup disagrees -- the game demonstrably happened.
        if week in real_team_by_week:
            weeks.append(week)
            continue
        if week not in team_weeks(team):
            continue  # bye or no game: never a candidate

        # Synthetic candidate: the team played and he didn't.
        if skip_tracker is not None:
            skip_tracker.count("candidate_weeks")
        if active is not None and week not in active:
            if skip_tracker is not None:
                skip_tracker.record(player_id, season, week,
                                    WeekSkipTracker.ROSTER_INELIGIBLE)
            continue
        if skip_tracker is not None:
            skip_tracker.count("roster_eligible")
        weeks.append(week)
    return weeks, team_by_week


def estimate_availability_rate(player_id: str, position: str, season: int, db,
                                position_avg_fallback: float = POSITION_AVG_FALLBACK) -> float:
    """Non-circular P(plays)-per-week estimate: the player's own games-played
    rate in seasons STRICTLY BEFORE `season` (leakage-safe — never uses the
    season being evaluated, since that would leak exactly what's being
    forecast). Falls back to a fixed position-average prior for players with
    no prior-season history (rookies) — same "be honest, don't force a
    number" spirit as tiers.py's 'rookie' tier fallback. The fallback value
    is shared with `availability.py` rather than duplicated, so the two can
    never drift apart when either is tuned.

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


def _lookup_depth_chart_rank_asof(gsis_id: str, target_season: int, target_week: int) -> Optional[int]:
    """Most recent depth_charts snapshot STRICTLY BEFORE (target_season,
    target_week) -- matches this module's existing "prior" convention
    (season < target_season, or same season with week < target_week), so
    a synthetic row never uses depth-chart information that wouldn't have
    been available yet at prediction time. Returns None (caller keeps the
    carried-forward value) when no such snapshot exists at all -- e.g. a
    player never in depth_charts, or 2018/2019 with no prior-season
    fallback available either. (2025 was in that list until its charts were
    loaded on 2026-08-19 -- see scripts/backfill_depth_charts.py.)

    Deliberately a STRICTER cutoff than `_add_depth_chart_rank`'s real-row
    lookup (`week <= target_week`, since that week's own snapshot is
    legitimate pre-game info for a real, already-happened game) -- these
    are two different, intentionally different cutoffs, not the same
    logic reused. Shares the same deduped as-of table
    (`_load_depth_chart_asof_table`) so both call sites agree on which
    value to use when the source data itself has duplicates (e.g. 2024's
    ~3x-inflated row count).
    """
    from src.features.feature_engineering import (
        _load_depth_chart_asof_table, DEPTH_CHART_MAX_STALENESS_SEASONS,
    )

    table = _load_depth_chart_asof_table()
    if table.empty:
        return None
    player_rows = table[table["gsis_id"] == gsis_id]
    if player_rows.empty:
        return None
    target_key = target_season * 100 + target_week - 1  # strict "<": search as of (target_key - 1)
    prior_rows = player_rows[player_rows["_key"] <= target_key]
    if prior_rows.empty:
        return None
    best_key = prior_rows["_key"].idxmax()
    # Same staleness bound as the real-row path (_add_depth_chart_rank) --
    # one shared policy, so a synthetic row and a real row never disagree
    # about whether a given snapshot is still trustworthy.
    if (target_season - prior_rows.loc[best_key, "_key"] // 100) > DEPTH_CHART_MAX_STALENESS_SEASONS:
        return None
    return int(prior_rows.loc[best_key, "depth_chart_rank"])


""" The three opponent-strength columns that actually reach the model
(identical across QB/RB/WR/TE in CAUSAL_FEATURES). All three are
season-to-date-through-week-N-1 measures of the season being predicted, so
all three need an August-legal replacement in preseason mode."""
_OPPONENT_STRENGTH_COLS = (
    "opp_fpts_allowed",
    "opp_fpts_allowed_s2d_lag1",
    "opp_fpts_allowed_dvoa_adjusted_lag1",
)


"""Features that survive onto a COLD-START row -- a player with no NFL history
at all. An explicit allowlist, not a pattern match: on a real rookie's row
every feature is populated (their week-8 `target_share_pct_roll3_mean` is a
genuine in-season value), so "what looks filled" cannot be used to classify
them. Anything absent here is blanked to NaN and left for LightGBM's
missing-aware splits.

Each entry is here because it is fixed before Week 1 of the target season:

  * draft capital and athleticism -- settled at the draft, in April;
  * college pedigree -- settled before that;
  * age -- birth date plus season;
  * destination-team context -- `dest_team_pos_*` is a shift(1) 3-season
    rolling team profile, and `team_prior_season_wins` / the coaching columns
    are prior-season facts;
  * `team_changed` is 0 by definition for a player who has never had a team.

Deliberately EXCLUDED even though a rookie's row carries values for them:
every `*_roll3_mean` / `*_lag*` / `*_accel`, `prev_season_ppg`,
`bayesian_prior_ppg`, `availability_3yr`, `career_year_flag`,
`fp_volatility_roll5`, `recv_drop_pct_season_prior`, `depth_chart_rank`,
`current_qb_epa_per_att`, and the per-week Vegas/weather columns. Those are
NFL history or in-season context, which is exactly what a rookie does not
have."""
COLD_START_KEEP_FEATURES = frozenset({
    # Draft capital / athleticism / pedigree
    "is_rookie", "is_undrafted", "draft_round", "draft_pick", "draft_pick_value",
    "rookie_draft_value", "rookie_bust_prob", "rookie_breakout_prob",
    "rookie_ceiling_ppg", "rookie_floor_ppg", "rookie_opportunity_score",
    "combine_score", "speed_score", "is_power5", "age_curve",
    # Destination-team context, all prior-season derived
    "dest_team_pos_tgt_pg", "dest_team_pos_carry_pg", "team_prior_season_wins",
    "coaching_change", "coaching_change_impact", "coaching_stability",
    "coaching_adaptation_score", "team_changed",
})

# Set on a cold-start row rather than left NaN: the conditional-on-playing
# assumption already used for synthetic weeks, since availability_rate carries
# the exposure discount separately.
COLD_START_NEUTRAL_INJURY_SCORE = 1.0


def build_cold_start_week_row(
    static_source: pd.DataFrame,
    feature_cols: List[str],
    target_season: int,
    target_week: int,
    team: str,
    db,
    feature_engineer,
) -> Optional[pd.DataFrame]:
    """Feature row for a player with NO prior NFL history (a true rookie).

    `build_synthetic_week_row` cannot help here: it works by carrying forward
    the most recent real game, and there is no such game. Without this, every
    rookie is dropped at row-construction time -- measured, 0 of 95 in 2025 --
    so no amount of missing-value handling downstream can reach them.

    `static_source` is any row of the player's own target-season data. Reading
    from it is NOT leakage: only `COLD_START_KEEP_FEATURES` is retained, and
    every column in that set is fixed before Week 1 (draft capital, combine,
    college, age, prior-season team context). All other features are set to
    NaN, which is the honest value -- a rookie has no prior-season target
    share, and 0 would assert that he had one and it was zero.
    """
    if static_source is None or static_source.empty:
        return None

    row = static_source.iloc[[0]].copy()
    row["season"] = target_season
    row["week"] = target_week
    row["team"] = team

    blanked = [c for c in feature_cols
               if c in row.columns and c not in COLD_START_KEEP_FEATURES]
    for col in blanked:
        row[col] = np.nan

    row = _apply_august_legal_matchup(
        row, team, target_season, target_week, db, feature_engineer,
    )
    if "injury_score" in row.columns:
        row["injury_score"] = COLD_START_NEUTRAL_INJURY_SCORE
    return row


def _apply_august_legal_matchup(row, team, target_season, target_week, db, feature_engineer):
    """Sets the target week's opponent from the published schedule, then
    grades that opponent using ONLY prior-season defensive performance.

    An August forecaster does know the schedule (released in May) and does
    know how each defense played last year. They do not know how it will
    play this year -- which is exactly what the live `opp_fpts_allowed*`
    features measure, since all three are season-to-date through week N-1
    of the season being predicted (see `_add_opp_fpts_allowed_s2d_lag1` /
    `_add_opp_fpts_allowed_dvoa_adjusted_lag1`).

    So the opponent comes from the target season's schedule, but the three
    strength columns are harvested from a probe row stamped to the END of
    the PRIOR season. That reuses `refresh_matchup_features` as-is rather
    than reimplementing its DVOA residual logic, and yields precisely "this
    defense's body of work last year" against the correct upcoming
    opponent.
    """
    from src.predict import get_schedule_map_for_week
    schedule_map = get_schedule_map_for_week(db, target_season, target_week)
    opponent, home_away = schedule_map.get(team, ("", "unknown"))
    row["opponent"] = opponent
    row["home_away"] = home_away
    if not opponent:
        return row

    probe = row.copy()
    probe["season"] = target_season - 1
    # Last regular-season week OF THE PRIOR SEASON -- that is the era that
    # matters here. A flat 18 probed the wild-card round for any prior season
    # through 2020, reading opponent strength off a playoff week.
    probe["week"] = regular_season_max_week(target_season - 1)
    probe = feature_engineer.refresh_matchup_features(probe)
    for col in _OPPONENT_STRENGTH_COLS:
        if col in probe.columns and col in row.columns:
            row[col] = probe[col].to_numpy()
    return row


def build_synthetic_week_row(
    player_history_df: pd.DataFrame,
    player_id: str,
    target_season: int,
    target_week: int,
    team: str,
    db,
    feature_engineer,
    preseason_mode: bool = False,
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
    if preseason_mode:
        # Matched-information mode: nothing from the target season at all,
        # not even earlier weeks of it.
        prior = player_history_df[
            (player_history_df["player_id"] == player_id)
            & (player_history_df["season"] < target_season)
        ].sort_values(["season", "week"])
    else:
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

    if preseason_mode:
        # August-legal: the schedule IS knowable (published in May), so the
        # opponent and home/away are set and the opponent is graded on last
        # season's defensive record. What is NOT knowable in August, and is
        # therefore deliberately left carried forward from the player's last
        # prior-season game rather than refreshed:
        #   * that week's Vegas line and weather (add_external_features) --
        #     week-by-week lines do not exist until the season is underway;
        #   * team rolling context (_compute_team_rolling_context) -- it
        #     averages the target season's preceding weeks;
        #   * depth chart as-of-week (_lookup_depth_chart_rank_asof) -- it
        #     reflects in-season promotions and demotions.
        # Known limitation, stated rather than hidden: `is_dome` is derived
        # inside add_external_features, so it stays at the carried-forward
        # value even though the stadium is August-knowable. That understates
        # Phase 7 slightly (is_dome is a high-importance RB feature) and is
        # a conservative error in this comparison, not a leak.
        row = _apply_august_legal_matchup(
            row, team, target_season, target_week, db, feature_engineer,
        )
        if "injury_score" in row.columns:
            row["injury_score"] = 1.0
        return row

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

    # 4. Depth chart rank — refresh from the real most-recent-prior
    # snapshot (see _lookup_depth_chart_rank_asof) instead of trusting the
    # carried-forward row's stale rank. This is the direct fix for a real,
    # confirmed bias: players requiring synthetic weeks showed wildly
    # positive season-total bias (e.g. QB +40 vs -28 for real-only-week
    # players) because a demoted/benched/replaced player's synthetic row
    # otherwise looks identical to how they looked as a healthy incumbent
    # starter. On a detected rank change, also rescale the usage-share
    # features left otherwise stale (depth chart rank and actual usage
    # aren't the same thing — a demoted player's old high snap-share value
    # would contradict the corrected rank rather than reinforce it), using
    # a simple empirical position+rank ratio from this fold's own history.
    if "depth_chart_rank" in row.columns:
        old_rank = row["depth_chart_rank"].iloc[0]
        new_rank = _lookup_depth_chart_rank_asof(player_id, target_season, target_week)
        if new_rank is not None and new_rank != old_rank:
            row["depth_chart_rank"] = new_rank
            position = row["position"].iloc[0] if "position" in row.columns else None
            if position is not None and "depth_chart_rank" in player_history_df.columns:
                hist = (
                    player_history_df[player_history_df["position"] == position]
                    if "position" in player_history_df.columns else player_history_df
                )
                usage_cols = [
                    c for c in row.columns
                    if c.endswith("_roll3_mean") and any(k in c for k in ("share", "target", "carry"))
                ]
                for col in usage_cols:
                    if col not in hist.columns:
                        continue
                    old_avg = hist.loc[hist["depth_chart_rank"] == old_rank, col].mean()
                    new_avg = hist.loc[hist["depth_chart_rank"] == new_rank, col].mean()
                    if pd.notna(old_avg) and pd.notna(new_avg) and old_avg > 0:
                        row[col] = row[col] * (new_avg / old_avg)

    return row


def resolve_week_source(
    week: int,
    real_weeks: set,
    data_source: Optional[str],
    exclude_pbp_confirmed_zeros: bool = False,
) -> bool:
    """True if `week` should be predicted from a real/inferred row (P(plays)=1,
    no availability_rate discount); False if it needs the synthetic
    carry-forward path (genuinely unknown week, availability_rate applies).

    `data_source` is the row's tag when `week in real_weeks`, else None.
    `exclude_pbp_confirmed_zeros` treats the weaker 2006-2017 tier
    (`inferred_pbp_confirmed_zero`) as absent, for sensitivity comparisons.
    """
    if week not in real_weeks:
        return False
    if exclude_pbp_confirmed_zeros and data_source == "inferred_pbp_confirmed_zero":
        return False
    return True


class WeekSkipTracker:
    """Player-weeks dropped from a season projection, made auditable.

    `FoldFailureTracker` does this one level up, for folds. The same failure
    mode exists per week: `compute_player_week_predictions` skips a week when
    the synthetic row can't be built (no prior history) or when predict()
    raises, and until this existed the first case logged nothing at all and
    the second logged a `logger.warning` that vanishes under `tail`.

    That silence is not cosmetic. A skipped week leaves the player looking
    like they had FEWER possible weeks than the schedule says, so
    `weeks_synthetic == 0` stops meaning "played every week" and starts
    quietly meaning "or we failed to project the ones they missed". In the
    v33 availability experiment that put 5 such players into the 39-player
    reference bucket and moved its bias from -30.6 to -28.8 -- a corrupted
    anchor for every gradient measured against it.
    """

    NO_PRIOR_HISTORY = "no_prior_history"
    PREDICT_FAILED = "predict_failed"
    ROSTER_INELIGIBLE = "roster_ineligible"

    # The synthetic-week funnel, so every stage is countable rather than
    # disappearing into a `continue`:
    #   candidate_weeks  team played, player didn't -- a week we COULD synthesize
    #   roster_eligible  ...and he was on the active roster that week
    #   row_constructed  ...and a synthetic feature row could be built
    #   predicted        ...and the model returned a prediction
    FUNNEL_STAGES = ("candidate_weeks", "roster_eligible", "row_constructed", "predicted")

    def __init__(self, phase: str):
        self.phase = phase
        self.skips: List[dict] = []
        self.funnel: Dict[str, int] = {k: 0 for k in self.FUNNEL_STAGES}

    def count(self, stage: str, n: int = 1) -> None:
        self.funnel[stage] = self.funnel.get(stage, 0) + n

    def record(self, player_id: str, season: int, week: int, reason: str,
               error: Optional[Exception] = None, **extra) -> None:
        self.skips.append({
            "phase": self.phase, "player_id": str(player_id), "season": int(season),
            "week": int(week), "reason": reason,
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
            **extra,
        })

    def report_funnel(self) -> None:
        """Every synthetic-week stage, so a drop can never be invisible."""
        if not any(self.funnel.values()):
            return
        print(f"\n[{self.phase}] synthetic-week funnel:")
        prev = None
        for stage in self.FUNNEL_STAGES:
            n = self.funnel.get(stage, 0)
            lost = "" if prev is None else f"   (-{prev - n})"
            print(f"    {stage:16s} {n:6d}{lost}")
            prev = n

    def report(self, output_path: Optional[Path] = None) -> None:
        self.report_funnel()
        if not self.skips:
            print(f"\n[{self.phase}] All player-weeks projected — no silently-dropped weeks.")
            return
        by_reason: Dict[str, int] = {}
        for s in self.skips:
            by_reason[s["reason"]] = by_reason.get(s["reason"], 0) + 1
        players = {s["player_id"] for s in self.skips}
        bar = "!" * 74
        print(f"\n{bar}")
        print(f"!! {self.phase}: {len(self.skips)} PLAYER-WEEK(S) SKIPPED "
              f"across {len(players)} player(s)")
        print(f"!! These weeks are absent from every season total below, so a player's")
        print(f"!! real+synthetic week count will NOT sum to their schedule. Do not read")
        print(f"!! 'zero synthetic weeks' as 'played every week' without checking here.")
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"!!   {reason}: {n}")
        print(bar)
        if output_path is not None:
            import json
            sidecar = Path(str(output_path) + ".weekskips.json")
            try:
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(self.skips, indent=2))
                print(f"!! Recorded to {sidecar}")
            except Exception as e:  # never let reporting break a completed run
                logger.warning("Could not write week-skip sidecar: %s", e)


def compute_player_week_predictions(
    player_id: str,
    g_by_week: Dict[int, pd.DataFrame],
    real_weeks: set,
    team_by_week: Dict[int, str],
    possible_weeks: List[int],
    model,
    feature_cols: List[str],
    full_history: pd.DataFrame,
    db,
    feature_engineer,
    season: int,
    exclude_pbp_confirmed_zeros: bool = False,
    capture_rows: bool = False,
    skip_tracker: Optional["WeekSkipTracker"] = None,
    preseason_mode: bool = False,
    cold_start: bool = False,
) -> List[dict]:
    """Per-week prediction list for one player's season -- the shared core
    reused by both `run_season_projection` (Phase 7, deterministic point
    total) and `season_simulation.py` (Phase 9, Monte Carlo quantiles), so
    the real/inferred/synthetic branching logic (`resolve_week_source`,
    `build_synthetic_week_row`) is defined exactly once.

    One dict per week actually predicted (weeks where prediction fails are
    silently skipped, matching the prior inline behavior): {week, is_real,
    data_source, actual_value (only set when is_real), point_prediction}.

    `capture_rows` additionally attaches the exact feature row the model was
    handed under "feature_row" -- for diagnostics that need to compare real
    against synthetic feature construction without re-deriving this logic.

    `team_by_week` is per week rather than one team for the season, so a
    player traded mid-season is projected against the schedule they were
    actually on -- build it with `possible_weeks_for_player`.
    """
    has_data_source = "data_source" in next(iter(g_by_week.values())).columns if g_by_week else False
    out: List[dict] = []
    for week in possible_weeks:
        data_source = (
            g_by_week[week]["data_source"].iloc[0]
            if week in real_weeks and has_data_source else None
        )
        # In preseason mode WHICH weeks he played is itself target-season
        # exposure information, so no week may be treated as known-played.
        treat_as_real = (not preseason_mode) and resolve_week_source(
            week, real_weeks, data_source, exclude_pbp_confirmed_zeros,
        )
        if treat_as_real:
            row = g_by_week[week][feature_cols]
            actual_value = float(g_by_week[week]["fantasy_points"].iloc[0])
        else:
            synth = build_synthetic_week_row(
                full_history, player_id, season, week, team_by_week[week],
                db, feature_engineer, preseason_mode=preseason_mode,
            )
            if synth is None and cold_start:
                # No prior NFL game to carry forward -- a true rookie. Build
                # from career-static attributes instead of dropping the player,
                # which is what excluded 100% of rookies before.
                synth = build_cold_start_week_row(
                    g_by_week.get(min(g_by_week)) if g_by_week else None,
                    feature_cols, season, week, team_by_week[week],
                    db, feature_engineer,
                )
            if synth is None:
                if skip_tracker is not None:
                    skip_tracker.record(player_id, season, week,
                                        WeekSkipTracker.NO_PRIOR_HISTORY)
                continue
            if skip_tracker is not None:
                skip_tracker.count("row_constructed")
            missing = [c for c in feature_cols if c not in synth.columns]
            for c in missing:
                synth[c] = np.nan
            row = synth[feature_cols]
            actual_value = None
        try:
            pred = float(model.predict(row)[0])
        except Exception as e:
            logger.warning("Predict failed for %s week %s: %s", player_id, week, e)
            if skip_tracker is not None:
                skip_tracker.record(player_id, season, week,
                                    WeekSkipTracker.PREDICT_FAILED, error=e)
            continue
        record = {
            "week": week,
            "is_real": treat_as_real,
            "data_source": data_source if treat_as_real else None,
            "actual_value": actual_value,
            "point_prediction": pred,
        }
        if capture_rows:
            record["feature_row"] = row.iloc[0].copy()
        if skip_tracker is not None and not treat_as_real:
            skip_tracker.count("predicted")
        out.append(record)
    return out


def run_season_projection(
    positions: Optional[Sequence[str]] = None,
    seasons: Optional[Sequence[int]] = None,
    output_path=None,
    exclude_pbp_confirmed_zeros: bool = False,
    architecture_override: Optional[dict] = None,
    week_output_path=None,
    feature_row_output_path=None,
    preseason_mode: bool = False,
    cold_start: bool = False,
) -> pd.DataFrame:
    """For each position's FINAL_CONFIG fold: fit once, then for every player
    with >=1 real game that season, sum E[PPR] across every possible week,
    and compare to the real season total.

    Three-way per-week branch (Complete Player-Game Panel-aware, see
    GAPS.md "SUPERSEDED -- 2026-08-10/11"):
      - A real `nflverse_stats` row exists -> predict off it directly.
        P(plays)=1 is already known, so `availability_rate` is NOT applied.
      - An inferred-zero row exists (`inferred_snap_verified_zero` or
        `inferred_pbp_confirmed_zero`) -> also predict off it directly, same
        reasoning: we have direct evidence the player took the field that
        week, so there's nothing left to discount.
      - No row of any kind -> genuinely unknown week. Build a synthetic
        carry-forward feature row and multiply the prediction by
        `estimate_availability_rate`'s P(plays) estimate, since this is the
        only case where the outcome is actually uncertain.

    Applying `availability_rate` uniformly to every week (the pre-2026-08-11
    behavior) double-discounted weeks we already had direct evidence for --
    real games, including true zero-PPR ones, were being scaled down by a
    play-rate prior even though P(plays)=1 was already known for them.

    `preseason_mode`: scores this as a TRUE preseason forecast, information-
    matched to the three arms in `scripts/walk_forward_preseason.py`
    (PreseasonProjector, the multi-year Ridge candidate, Step 8A), which see
    only prior-season player aggregates plus destination team. Four
    target-season leaks are closed together, since closing any subset still
    leaves the comparison unfair:

      1. no week is treated as known-played (see compute_player_week_
         predictions) -- which weeks he played is target-season exposure;
      2. carry-forward history is restricted to seasons strictly before the
         target, and `full_history` drops the test frame entirely;
      3. every per-week refresh is skipped (see build_synthetic_week_row);
      4. the active-roster gate is disabled and the team is pinned to the
         player's entering-season team, since weekly roster status and
         mid-season trades are both unknowable in August.

    `availability_rate` then applies to every week, making the result
    structurally comparable to Step 8A's E[games] x E[PPR/game].

    `exclude_pbp_confirmed_zeros`: sensitivity toggle. When True, treats
    `inferred_pbp_confirmed_zero` rows (the weaker, 2006-2017 tier -- see
    GAPS.md) as if they didn't exist, falling back to the synthetic path for
    those weeks instead. Off by default; mirrors the Population B-vs-C
    comparison used when the panel itself was validated.
    """
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.features.feature_engineering import PositionFeatureEngineer
    from src.models.single_week_ppr.evaluate import (
        DEFAULT_VALIDATION_SEASONS, run_fold, _architectures_for_fold, _append_df_to_csv,
        FoldFailureTracker,
    )
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.windows import window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    seasons = seasons or DEFAULT_VALIDATION_SEASONS
    output_path = output_path or Path("data/experiments/phase7_season_projection.csv")
    positions = list(positions) if positions else POSITIONS
    all_rows: List[dict] = []
    # Per-week detail, so a single run yields BOTH the weekly and the season
    # metrics on identical folds -- Phase 7C compares the two levels and they
    # must come from the same fit, not from separate runs.
    week_rows: Optional[List[dict]] = [] if week_output_path is not None else None
    # The exact feature rows the model scored. Needed because in cold-start
    # preseason mode every scored row is SYNTHETIC -- built here per week --
    # so a dose/stratum derived from real historical rows describes a
    # different population than the one being predicted, which is how the
    # 2026-08-26 strata came to misclassify 202 player-seasons as zero-dose.
    feature_rows: Optional[List[dict]] = [] if feature_row_output_path is not None else None
    db = DatabaseManager()

    tracker = FoldFailureTracker("Phase 7 (season projection)")
    week_skips = WeekSkipTracker("Phase 7 (season projection)")
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
                tracker.record(position, season, e)
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

            arch_name = (architecture_override or {}).get(position, cfg["architecture"])
            model = _architectures_for_fold()[arch_name]
            X_train = pos_train[feature_cols]
            y_train = pos_train["fantasy_points"]
            model.fit(X_train, y_train)

            # Full player-season history (for carrying forward into synthetic rows)
            # includes this fold's train+test frame so mid-season history is present.
            # Preseason mode must not see the target season at all, so the
            # test frame is excluded from the carry-forward pool -- otherwise
            # a synthetic week could carry forward from a real target-season
            # game even with treat_as_real forced off.
            full_history = (
                pos_train.reset_index(drop=True) if preseason_mode
                else pd.concat([pos_train, pos_test], ignore_index=True)
            )

            for player_id, g_all in pos_test.groupby("player_id"):
                # Regular season only -- player_weekly_stats also carries
                # playoff-week rows for players whose team made the playoffs,
                # which must never be folded into a "season" total: the model
                # only ever predicts regular-season weeks, so comparing against
                # an actual that silently includes bonus playoff production is
                # an apples-to-oranges inflation. Found via Phase 9 debugging.
                #
                # The cap is era-dependent. This originally used a flat 18,
                # which is right only from 2021 -- through 2020 the regular
                # season was 17 weeks, so week 18 IS the wild-card round and
                # one full playoff round was admitted for every pre-2021
                # season. See regular_season_max_week and GAPS.md.
                g = g_all[g_all["week"] <= regular_season_max_week(season)]
                if g.empty:
                    continue
                g_by_week = {int(w): sub for w, sub in g.groupby(g["week"].astype(int))}
                real_weeks = set(g_by_week.keys())
                real_team_by_week = {int(w): sub["team"].iloc[0] for w, sub in g_by_week.items()}
                if preseason_mode:
                    # Mid-season trades are not knowable in August, and the
                    # weekly active-roster status that normally gates
                    # synthetic weeks is itself in-season information. Pin to
                    # the entering-season team (the `dest_team` the other
                    # arms get) and let availability_rate carry the exposure
                    # discount for every week instead.
                    first_week = min(real_team_by_week)
                    lookup_team_by_week = {first_week: real_team_by_week[first_week]}
                else:
                    lookup_team_by_week = real_team_by_week
                possible_weeks, team_by_week = possible_weeks_for_player(
                    db, lookup_team_by_week, season, player_id=player_id,
                    skip_tracker=week_skips,
                    require_active_roster=not preseason_mode)
                if not possible_weeks:
                    continue

                availability_rate = estimate_availability_rate(player_id, position, season, db)

                week_predictions = compute_player_week_predictions(
                    player_id, g_by_week, real_weeks, team_by_week, possible_weeks, model,
                    feature_cols, full_history, db, feature_engineer, season,
                    exclude_pbp_confirmed_zeros, skip_tracker=week_skips,
                    preseason_mode=preseason_mode, cold_start=cold_start,
                    capture_rows=feature_rows is not None,
                )

                predicted_total = 0.0
                weeks_predicted = 0
                weeks_real_stats = 0
                weeks_inferred_snap_verified = 0
                weeks_inferred_pbp_confirmed = 0
                weeks_synthetic = 0
                for wp in week_predictions:
                    rate = 1.0 if wp["is_real"] else availability_rate
                    if feature_rows is not None and "feature_row" in wp:
                        fr = dict(wp["feature_row"])
                        fr.update({"player": player_id, "position": position,
                                   "season": season, "week": int(wp["week"])})
                        feature_rows.append(fr)
                    if week_rows is not None:
                        week_rows.append({
                            "player": player_id, "position": position, "season": season,
                            "week": int(wp["week"]), "prediction": wp["point_prediction"],
                            "rate_applied": rate,
                            "contribution": wp["point_prediction"] * rate,
                            "actual_ppr": wp["actual_value"], "is_real": wp["is_real"],
                            "data_source": wp.get("data_source"),
                            "architecture": arch_name,
                        })
                    if wp["is_real"]:
                        if wp["data_source"] == "inferred_snap_verified_zero":
                            weeks_inferred_snap_verified += 1
                        elif wp["data_source"] == "inferred_pbp_confirmed_zero":
                            weeks_inferred_pbp_confirmed += 1
                        else:
                            weeks_real_stats += 1
                    else:
                        weeks_synthetic += 1
                    predicted_total += wp["point_prediction"] * rate
                    weeks_predicted += 1

                if weeks_predicted == 0:
                    continue

                # Invariant, enforced rather than assumed: in preseason mode
                # no week may be known-played. Found the hard way -- the
                # `preseason_mode` argument was initially threaded through
                # every signature but not through THIS call site, so the flag
                # was silently inert and the run produced in-season numbers
                # under a preseason label. A wrong-but-plausible MAE is the
                # worst possible failure here, so it raises.
                if preseason_mode and (weeks_real_stats or weeks_inferred_snap_verified
                                       or weeks_inferred_pbp_confirmed):
                    raise RuntimeError(
                        f"preseason_mode invariant violated for {player_id} "
                        f"({position}/{season}): {weeks_real_stats} real + "
                        f"{weeks_inferred_snap_verified} snap-verified + "
                        f"{weeks_inferred_pbp_confirmed} pbp-confirmed weeks were "
                        f"treated as known-played. Every week must be synthetic."
                    )

                actual_total = float(g["fantasy_points"].sum())
                result_row = {
                    "player": player_id, "position": position, "season": season,
                    # Team is per-week now; report the one he last actually
                    # played for, which is what a season-level row means.
                    "team": real_team_by_week[max(real_team_by_week)],
                    "possible_weeks": len(possible_weeks), "weeks_predicted": weeks_predicted,
                    "games_actually_played": len(real_weeks),
                    "availability_rate": availability_rate,
                    "weeks_real_stats": weeks_real_stats,
                    "weeks_inferred_snap_verified": weeks_inferred_snap_verified,
                    "weeks_inferred_pbp_confirmed": weeks_inferred_pbp_confirmed,
                    "weeks_synthetic": weeks_synthetic,
                    "predicted_season_total": predicted_total,
                    "actual_season_total": actual_total,
                }
                all_rows.append(result_row)
                _append_df_to_csv(pd.DataFrame([result_row]), output_path)

    if feature_rows is not None:
        fdf = pd.DataFrame(feature_rows)
        fdf.to_parquet(feature_row_output_path, index=False)
        print(f"{len(fdf)} scored feature rows written to {feature_row_output_path}")
    if week_rows is not None:
        pd.DataFrame(week_rows).to_csv(week_output_path, index=False)
        print(f"{len(week_rows)} per-week rows written to {week_output_path}")

    result = pd.DataFrame(all_rows)
    print(f"\n{len(result)} rows appended to {output_path}")
    tracker.report(output_path)
    week_skips.report(output_path)
    return result
