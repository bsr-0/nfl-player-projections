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
    (week > 18).

    `season` is coerced to a plain int: sqlite3 does not bind numpy integers,
    so a numpy season silently matches no rows and returns [] — which reads
    downstream as "this team never played", producing zero synthetic weeks
    with no error. A caller iterating `groupby(["player_id", "season"])` gets
    numpy scalars without realising it.
    """
    sched = db.get_schedule(season=int(season), team=str(team))
    if sched is None or sched.empty:
        return []
    weeks = sorted(int(w) for w in sched["week"].unique() if int(w) <= REGULAR_SEASON_MAX_WEEK)
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
    for week in range(1, REGULAR_SEASON_MAX_WEEK + 1):
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


def _lookup_depth_chart_rank_asof(gsis_id: str, target_season: int, target_week: int) -> Optional[int]:
    """Most recent depth_charts snapshot STRICTLY BEFORE (target_season,
    target_week) -- matches this module's existing "prior" convention
    (season < target_season, or same season with week < target_week), so
    a synthetic row never uses depth-chart information that wouldn't have
    been available yet at prediction time. Returns None (caller keeps the
    carried-forward value) when no such snapshot exists at all -- e.g. a
    player never in depth_charts, or 2018/2019/2025 with no prior-season
    fallback available either.

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
        treat_as_real = resolve_week_source(
            week, real_weeks, data_source, exclude_pbp_confirmed_zeros,
        )
        if treat_as_real:
            row = g_by_week[week][feature_cols]
            actual_value = float(g_by_week[week]["fantasy_points"].iloc[0])
        else:
            synth = build_synthetic_week_row(
                full_history, player_id, season, week, team_by_week[week],
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

            model = _architectures_for_fold()[cfg["architecture"]]
            X_train = pos_train[feature_cols]
            y_train = pos_train["fantasy_points"]
            model.fit(X_train, y_train)

            # Full player-season history (for carrying forward into synthetic rows)
            # includes this fold's train+test frame so mid-season history is present.
            full_history = pd.concat([pos_train, pos_test], ignore_index=True)

            for player_id, g_all in pos_test.groupby("player_id"):
                # Regular season only -- player_weekly_stats also carries
                # playoff-week rows (week > 18) for players whose team made
                # the playoffs, which must never be folded into a "season"
                # total: the model only ever predicts regular-season weeks
                # (possible_weeks_for_team already caps at 18), so comparing
                # against an actual that silently includes bonus playoff
                # production is an apples-to-oranges inflation. Found via
                # Phase 9 debugging -- see GAPS.md.
                g = g_all[g_all["week"] <= REGULAR_SEASON_MAX_WEEK]
                if g.empty:
                    continue
                g_by_week = {int(w): sub for w, sub in g.groupby(g["week"].astype(int))}
                real_weeks = set(g_by_week.keys())
                real_team_by_week = {int(w): sub["team"].iloc[0] for w, sub in g_by_week.items()}
                possible_weeks, team_by_week = possible_weeks_for_player(
                    db, real_team_by_week, season, player_id=player_id,
                    skip_tracker=week_skips)
                if not possible_weeks:
                    continue

                availability_rate = estimate_availability_rate(player_id, position, season, db)

                week_predictions = compute_player_week_predictions(
                    player_id, g_by_week, real_weeks, team_by_week, possible_weeks, model,
                    feature_cols, full_history, db, feature_engineer, season,
                    exclude_pbp_confirmed_zeros, skip_tracker=week_skips,
                )

                predicted_total = 0.0
                weeks_predicted = 0
                weeks_real_stats = 0
                weeks_inferred_snap_verified = 0
                weeks_inferred_pbp_confirmed = 0
                weeks_synthetic = 0
                for wp in week_predictions:
                    rate = 1.0 if wp["is_real"] else availability_rate
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

                actual_total = float(g["fantasy_points"].sum())
                result_row = {
                    "player": player_id, "position": position, "season": season, "team": team,
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

    result = pd.DataFrame(all_rows)
    print(f"\n{len(result)} rows appended to {output_path}")
    tracker.report(output_path)
    week_skips.report(output_path)
    return result
