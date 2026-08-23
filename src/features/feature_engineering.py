"""Feature engineering pipeline for NFL player prediction.

Missing data root causes (and how we handle them):
- LEFT JOINs in get_all_players_for_training: team_stats, utilization_scores,
  team_defense_stats can be NULL when not populated. We backfill team_stats from
  player aggregates; utilization is computed in pipeline; defense defaults applied.
- Schedule/StrengthOfSchedule can fail or be missing: opponent_rating,
  matchup_difficulty, team_sos get neutral defaults (50.0).
- Rolling/lag features produce NaN for early rows per player: we impute in
  prepare_training_data and in a final _impute_missing step so model never sees NaN/inf.
- Injury/rookie: optional columns injury_score, is_injured, is_rookie get safe
  defaults (1.0, 0, 0) when not provided so they can act as utilization predictors.

Data quality (per requirements): max 5%% missing per feature is acceptable. We flag
features exceeding 5%% missing (logged); imputation strategy: column median, then 0.
Features with >5%% missing are still used but may reduce model reliability.
"""
import pandas as pd
import numpy as np
import os
import warnings
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    ROLLING_WINDOWS, LAG_WEEKS, POSITIONS, MOMENTUM_WEIGHTS,
    BOOM_BUST_THRESHOLDS, BOOM_BUST_DEFAULT,
    AGE_CURVE_PARAMS, AGE_CURVE_DEFAULT,
)
from src.utils.helpers import (
    rolling_average, exponential_weighted_average,
    create_lag_features, safe_divide
)
from src.features.feature_policy_registry import FeaturePolicyRegistry

# Rolling/aggregation on sparse early-season windows can emit this benign warning.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

# Process-level cache for _add_team_matchup_features()'s expensive lookup
# tables (prior-season averages, in-season rolling window, momentum score).
# These depend only on the full team_stats table, not on the current
# training window, but the walk-forward backtester calls create_features()
# fresh every week -- profiling showed this groupby/transform/apply block
# was ~430s of a ~1390s single-position backtest (~31%). Cached by row
# count + max date so a genuinely updated team_stats table busts it.
# (GAPS.md, 2026-08-06 perf fix.)
_team_matchup_lookup_cache: Dict[Any, Tuple] = {}

# Single-entry cache for the depth_charts as-of table (season/week/gsis_id
# -> rank) -- static per process like the team-matchup lookups above, and
# rebuilding it (a full-table read + merge_asof sort) on every
# _add_depth_chart_rank call would be wasteful across a backtest's many
# weekly calls.
_depth_chart_asof_cache: Dict[Any, pd.DataFrame] = {}

# Max seasons a depth-chart snapshot may be carried forward before it is
# treated as unknown. 1 = "last season's chart is acceptable, older is not".
DEPTH_CHART_MAX_STALENESS_SEASONS = 1

# Columns whose NaN belongs to the persisted snap-imputation step rather than
# to _impute_missing's generic median fill. Kept as a literal set (not an
# import) to avoid a circular import with utilization_score, which imports
# nothing from this module but is imported by it lazily elsewhere.
_SNAP_IMPUTATION_OWNED = frozenset({
    "snap_count", "team_snaps", "snap_share", "snap_share_pct",
    "snap_share_pct_roll3_mean",
})

# Columns whose missingness is STRUCTURAL -- the source does not exist for
# those seasons at all -- rather than random. Median-filling these invents a
# league-average value for an era that has no measurement, which is how
# team_motion_rate/team_play_action_rate became literal constants for
# 2006-2022 and real values from 2023: a train/test discontinuity, not a
# feature (GAPS.md 2026-08-20). FTN charting begins in 2022. LightGBM splits
# on NaN natively, so leaving them missing is both honest and usable.
_STRUCTURALLY_MISSING = frozenset({
    "team_motion_rate", "team_play_action_rate", "team_shotgun_rate",
})


# The NFL abolished the "Probable" designation after the 2015 season. It
# appears 2,772 / 2,607 / 2,702 times in 2013-2015 and exactly 0 from 2016.
PROBABLE_ABOLISHED_AFTER_SEASON = 2015


def _warn_on_probable_era_span(injuries: pd.DataFrame) -> None:
    """Warn when a training window straddles the "Probable" rule change.

    `Probable` scores 0.85, so `injury_score` reads as mildly unavailable for
    ~2,700 player-weeks a season before 2016 and for none after. That is a
    reporting-rule artifact, not players becoming healthier: `pct_injured`
    runs 14-16% pre-2016 against 4-6% after.

    NOT silently remapped to 1.0. That is a modelling decision, and this
    project has already seen one such "obviously correct" re-encoding fail
    its experiment -- unknown snap shares, where the semantically wrong value
    turned out to be the better predictor because the missingness was
    informative (GAPS.md 2026-08-19). The same question is open here: the
    ~2,700 vanished Probables were NOT absorbed by Questionable, which stayed
    near 1,300, so in the modern regime those players are simply unlisted at
    1.0 -- which argues for the remap but does not establish it.

    `TRAINING_START_YEAR_DEFAULT = 2018` keeps the default window clear of
    this. The warning exists so the "full" 2006+ preset cannot cross the
    boundary without saying so.
    """
    if "season" not in injuries.columns or "report_status" not in injuries.columns:
        return
    seasons = pd.to_numeric(injuries["season"], errors="coerce").dropna()
    if seasons.empty:
        return
    if seasons.min() > PROBABLE_ABOLISHED_AFTER_SEASON or seasons.max() <= PROBABLE_ABOLISHED_AFTER_SEASON:
        return  # entirely one side of the boundary

    n_probable = int((injuries["report_status"] == "Probable").sum())
    if not n_probable:
        return
    import warnings
    warnings.warn(
        f"injury_score spans the {PROBABLE_ABOLISHED_AFTER_SEASON}/"
        f"{PROBABLE_ABOLISHED_AFTER_SEASON + 1} 'Probable' rule change: "
        f"{n_probable:,} Probable rows score 0.85 in the earlier seasons and "
        f"the status does not exist later, so injury_score is not comparable "
        f"across the boundary. See GAPS.md.",
        RuntimeWarning, stacklevel=2,
    )


def _load_depth_chart_asof_table() -> pd.DataFrame:
    """Loads `depth_charts` into an as-of lookup table: one row per
    (gsis_id, season*100+week) with the deduped depth_team rank, sorted for
    `pd.merge_asof`. Excludes `season IS NULL` rows: those were 2025's feed
    written in nflverse's newer daily ESPN-style schema, which has no
    season/week/depth_team column at all, so a straight insert left them
    unmapped. They are now loaded properly by
    scripts/backfill_depth_charts_2025.py; the filter stays as a guard.
    Deduplicates per (season, week, gsis_id) via a deterministic MIN so a
    conflicting duplicate doesn't produce a nondeterministic result. (2024's
    row count was previously described here as "~3x-inflated" and in need of
    that dedup; it isn't. 2020-2023 are skill-position-only, while 2024+ hold
    every position -- the gap is a filter, not duplication. The MIN still
    matters for the genuinely conflicting listings that do exist: a player
    named at two depth slots in one week.)
    """
    cache_key = "depth_chart_asof_table"
    if cache_key in _depth_chart_asof_cache:
        return _depth_chart_asof_cache[cache_key]

    import sqlite3
    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))
    try:
        raw = pd.read_sql(
            """SELECT season, week, gsis_id, depth_team FROM depth_charts
               WHERE season IS NOT NULL AND week IS NOT NULL AND gsis_id IS NOT NULL""",
            conn,
        )
    finally:
        conn.close()
    raw = raw.dropna(subset=["season", "week", "gsis_id"])

    if raw.empty:
        _depth_chart_asof_cache[cache_key] = raw
        return raw

    raw["season"] = raw["season"].astype(int)
    raw["week"] = raw["week"].astype(int)
    raw["depth_chart_rank"] = pd.to_numeric(raw["depth_team"], errors="coerce").fillna(3).astype(int)
    raw["_key"] = raw["season"] * 100 + raw["week"]
    # MIN across duplicate (gsis_id, season, week) entries. The source data
    # genuinely lists some players in multiple slots for the same week (1,821
    # such conflicts in 2024 alone), e.g. a receiver listed deeper in the base
    # formation but first in a sub-package. MIN takes the most prominent
    # listed role, and is chosen over max/first mainly so the result is
    # DETERMINISTIC -- `first` would depend on row order and could differ
    # between runs. This is a judgment call, recorded here rather than left
    # implicit.
    table = (
        raw.groupby(["gsis_id", "_key"], as_index=False)["depth_chart_rank"].min()
        .sort_values("_key")
    )
    # Preserved so callers can measure how stale a matched snapshot is; the
    # left frame's own `_key` is what survives a merge_asof, so without this
    # the match's origin would be unrecoverable.
    table["_snap_key"] = table["_key"]
    _depth_chart_asof_cache[cache_key] = table
    return table


def _get_team_matchup_lookups(all_team_stats: pd.DataFrame, team_metrics: List[str]) -> Tuple:
    """Build (team_a_avgs, team_b_avgs, inseason_df, mom_df) from the full
    team_stats table, cached across calls within a process."""
    cache_key = (
        len(all_team_stats),
        tuple(sorted(all_team_stats.columns)),
        all_team_stats["season"].max() if "season" in all_team_stats.columns else None,
        all_team_stats["week"].max() if "week" in all_team_stats.columns else None,
    )
    cached = _team_matchup_lookup_cache.get(cache_key)
    if cached is not None:
        return cached

    team_avgs = all_team_stats.groupby(['team', 'season']).agg({
        col: 'mean' for col in team_metrics if col in all_team_stats.columns
    }).reset_index()
    # Shift season forward by 1 so that a row for season S uses season S-1 averages
    team_avgs['season'] = team_avgs['season'] + 1

    team_a_cols = {col: f'team_a_{col}' for col in team_metrics if col in team_avgs.columns}
    team_a_avgs = team_avgs.rename(columns=team_a_cols)

    team_b_cols = {col: f'team_b_{col}' for col in team_metrics if col in team_avgs.columns}
    team_b_avgs = team_avgs.rename(columns=team_b_cols)
    team_b_avgs = team_b_avgs.rename(columns={'team': 'opponent', 'season': 'season'})

    inseason_df = None
    if 'week' in all_team_stats.columns:
        avail_metrics = [m for m in team_metrics if m in all_team_stats.columns]
        if avail_metrics:
            ts_sorted = all_team_stats.sort_values(['team', 'season', 'week'])
            for metric in avail_metrics:
                ts_sorted[f'_inseason_{metric}'] = ts_sorted.groupby(
                    ['team', 'season']
                )[metric].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=2).mean()
                )
            inseason_cols = ['team', 'season', 'week'] + [f'_inseason_{m}' for m in avail_metrics]
            inseason_df = ts_sorted[inseason_cols].drop_duplicates()

    mom_df = None
    if 'week' in all_team_stats.columns and 'points_scored' in all_team_stats.columns:
        ts = all_team_stats.sort_values(['team', 'season', 'week'])

        def _momentum_60_30_10(grp: pd.Series) -> pd.Series:
            """Time-weighted momentum: 60% last 4w, 30% weeks 5-8, 10% weeks 9+."""
            out = pd.Series(index=grp.index, dtype=float)
            for i in range(len(grp)):
                hist = grp.iloc[:i]  # past weeks only (no leakage)
                if len(hist) == 0:
                    out.iloc[i] = np.nan
                    continue
                vals = hist.values[::-1]  # most recent first
                n = len(vals)
                w = np.zeros(n)
                w[: min(4, n)] = 0.6 / min(4, n)
                if n > 4:
                    w[4: min(8, n)] = 0.3 / min(4, n - 4)
                if n > 8:
                    w[8:] = 0.1 / (n - 8)
                out.iloc[i] = np.nansum(vals * w)
            return out

        ts["_mom_pts"] = ts.groupby(
            ['team', 'season'], group_keys=False
        )['points_scored'].transform(_momentum_60_30_10)

        composite_parts = [("_mom_pts", 0.50)]
        if 'passing_yards' in ts.columns:
            ts["_mom_pass"] = ts.groupby(
                ['team', 'season'], group_keys=False
            )['passing_yards'].transform(_momentum_60_30_10)
            composite_parts.append(("_mom_pass", 0.20))
        if 'rushing_yards' in ts.columns:
            ts["_mom_rush"] = ts.groupby(
                ['team', 'season'], group_keys=False
            )['rushing_yards'].transform(_momentum_60_30_10)
            composite_parts.append(("_mom_rush", 0.15))
        if 'turnovers' in ts.columns:
            ts["_mom_to"] = -ts.groupby(
                ['team', 'season'], group_keys=False
            )['turnovers'].transform(_momentum_60_30_10)
            composite_parts.append(("_mom_to", 0.15))

        for col, _ in composite_parts:
            exp_mean = ts.groupby(['team', 'season'])[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            exp_std = ts.groupby(['team', 'season'])[col].transform(
                lambda x: x.shift(1).expanding(min_periods=2).std()
            ).clip(lower=1e-6)
            ts[col + "_z"] = ((ts[col] - exp_mean) / exp_std).fillna(0.0)

        total_weight = sum(w for _, w in composite_parts)
        ts["offensive_momentum_score"] = sum(
            ts[col + "_z"] * (w / total_weight) for col, w in composite_parts
        )
        exp_mom_mean = ts.groupby(['team', 'season'])["offensive_momentum_score"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        exp_mom_std = ts.groupby(['team', 'season'])["offensive_momentum_score"].transform(
            lambda x: x.shift(1).expanding(min_periods=2).std()
        ).clip(lower=1e-6)
        ts["offensive_momentum_score"] = (
            22.0 + 8.0 * (ts["offensive_momentum_score"] - exp_mom_mean) / exp_mom_std
        ).fillna(22.0).clip(0, 44)

        temp_cols = [c for c in ts.columns if c.startswith("_mom_")]
        ts = ts.drop(columns=temp_cols, errors="ignore")

        mom_df = ts[['team', 'season', 'week', 'offensive_momentum_score']].drop_duplicates()

    result = (team_a_avgs, team_b_avgs, inseason_df, mom_df)
    _team_matchup_lookup_cache.clear()  # single-entry cache; avoid unbounded growth across seasons
    _team_matchup_lookup_cache[cache_key] = result
    return result


class FeatureEngineer:
    """Feature engineering for NFL player performance prediction."""

    def __init__(self, feature_mode: Optional[str] = None):
        from config.settings import FEATURE_MODE
        self.feature_mode = feature_mode or FEATURE_MODE
        self.rolling_windows = ROLLING_WINDOWS
        self.lag_weeks = LAG_WEEKS
        self.feature_columns = []
        self.policy_registry = FeaturePolicyRegistry.from_config()
        self.last_imputation_report: Dict[str, Any] = {}

    def create_features(self, df: pd.DataFrame,
                        include_target: bool = True) -> pd.DataFrame:
        """
        Create all features for model training/prediction.

        Args:
            df: DataFrame with player weekly stats
            include_target: Whether to include target variable

        Returns:
            DataFrame with engineered features
        """
        if self.feature_mode == "causal":
            return self.create_causal_features(df, include_target=include_target)

        df = df.copy()
        
        # Sort by player and time
        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        
        # Create base features
        df = self._create_base_features(df)

        # Create rolling features (historical averages)
        df = self._create_rolling_features(df)

        # Create lag features
        df = self._create_lag_features(df)

        # Create trend features
        df = self._create_trend_features(df)

        # Defragment after the heaviest column-adding phase to avoid PerformanceWarnings
        df = df.copy()

        # Create opponent features
        df = self._create_opponent_features(df)

        # Create situational features
        df = self._create_situational_features(df)

        # Team-change and scheme-fit features (proactive context adjustment)
        df = self._create_team_change_features(df)

        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Advanced requirement features (boom/bust, season phase, experience, classification, workload risk)
        df = self._create_advanced_requirement_features(df)

        # Return-from-injury production patterns (first 3 games back)
        df = self._create_return_from_injury_features(df)

        # Vegas game script predictors (spread, over/under, implied team total)
        df = self._create_vegas_game_script_features(df)

        # Team quality: prior-season wins (causal — known before season starts)
        df = self._add_prior_season_wins(df)
        # Prior-season QB efficiency (benefits WR/TE predictions)
        df = self._add_team_qb_efficiency(df)
        # QB-identity-aware efficiency: uses actual QB1 per team, roster-sourced for 2026+
        df = self._add_current_qb_efficiency(df)

        # Advanced analytics: sentiment, coaching changes, suspensions,
        # trade deadline, and playoff context features.
        df = self._create_advanced_analytics_features(df)

        # NGS stats (CPOE, separation, RYOE, time-to-throw)
        df = self._merge_ngs_data(df)
        df = self._create_ngs_features(df)

        # Draft capital (decayed pick value)
        df = self._merge_draft_capital(df)

        # Populate injury_score / is_injured from the local player_injuries
        # cache BEFORE the default-fallback in _ensure_injury_rookie_features
        # runs.  Without this call, injury_score pins to 1.0 (healthy) for
        # every row — see docs/PHASE_3_INJURY_FINDINGS.md.
        df = self._merge_injury_data_from_cache(df)

        # Optional injury/rookie predictors for utilization (defaults when missing)
        df = self._ensure_injury_rookie_features(df)
        
        # Outlier detection per requirements Section VI.C (>3 sigma flagged, not removed)
        df = self._flag_outliers(df, sigma_threshold=3.0)
        
        # Check missing rate per feature (requirement: max 5% acceptable); log exceedances
        self._check_missing_rate(df, threshold_pct=5.0)

        # Group policy-based imputations + indicator flags.
        fail_policy = os.getenv("NFL_FEATURE_POLICY_FAIL_ON_THRESHOLD", "0") == "1"
        policy_result = self.policy_registry.apply(
            df,
            context="feature_engineering",
            fail_on_threshold=fail_policy,
        )

        # Season-relative normalization to reduce train/test distribution shift
        df = self._normalize_season_relative(df)

        # Final imputation: no NaN/inf in numeric columns so pipelines are robust
        df = self._impute_missing(df)
        self.last_imputation_report = {
            "rates": policy_result.rates,
            "warn_features": policy_result.flagged_warn,
            "fail_features": policy_result.flagged_fail,
        }
        
        # Store feature column names
        self._update_feature_columns(df)
        
        return df

    def create_causal_features(self, df: pd.DataFrame,
                               include_target: bool = True) -> pd.DataFrame:
        """Create minimal causal feature set (9-11 per position).

        Council recommendation: opportunity share, snap %, short-window
        volume, one efficiency metric, and opponent/Vegas context only.
        Skips rolling windows > 3 weeks, lag features, trends, interactions,
        boom/bust, team change, and all other derived features.
        """
        df = df.copy()
        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

        # Base efficiency features (yards_per_carry, yards_per_target, etc.)
        df = self._create_base_features(df)

        # Rolling features — 3-week window only
        df = self._create_causal_rolling_features(df)

        # Opponent features (creates opp_fpts_allowed)
        df = self._create_opponent_features(df)

        # Vegas features (creates implied_team_total)
        df = self._create_vegas_game_script_features(df)

        # Weather (creates wind_speed_mph, is_dome, precipitation_flag, temperature_bucket)
        df = self._add_weather_features(df)

        # Team quality: prior-season wins (causal — known before season starts)
        df = self._add_prior_season_wins(df)
        # Prior-season QB efficiency (benefits WR/TE predictions)
        df = self._add_team_qb_efficiency(df)
        # QB-identity-aware efficiency: uses actual QB1 per team, roster-sourced for 2026+
        df = self._add_current_qb_efficiency(df)

        # NGS stats (CPOE, separation, RYOE) — merged + rolled for causal features
        df = self._merge_ngs_data(df)
        df = self._create_ngs_features(df)

        # Draft capital (decayed pick value) — available in full mode,
        # not in CAUSAL_FEATURES yet pending ablation
        df = self._merge_draft_capital(df)

        # Populate injury_score / is_injured from the local player_injuries
        # cache BEFORE the default-to-healthy fallback below.  See
        # docs/PHASE_3_INJURY_FINDINGS.md — without this call, the causal
        # path silently pins injury_score to 1.0 for every row and the
        # declared CAUSAL_FEATURES entry is dead weight.
        df = self._merge_injury_data_from_cache(df)

        # Ensure injury_score exists (1.0 = healthy when injury data not merged)
        if "injury_score" not in df.columns:
            df["injury_score"] = 1.0

        # Prior-season / season-to-date PPG proxy.  At week 1 of a
        # new season this evaluates to the prior season's PPG — the
        # canonical draft-time signal.  From week 2 onward it is a
        # season-to-date partial mean.  See _add_prev_season_ppg for
        # the exact semantics.
        df = self._add_prev_season_ppg(df)

        # v19 Exp 3: prior season weeks 14-17 avg PPG — cold-start form signal for QBs.
        df = self._add_prior_season_late_ppg(df)

        # v13 features: age, team change, availability, career year, Bayesian prior
        df = self._add_age_curve_feature(df)
        df = self._create_team_change_features(df)
        df = self._add_availability_3yr(df)
        df = self._add_career_year_flag(df)
        df = self._add_bayesian_prior_ppg(df)

        # v24: head-coach identity + coaching-change detection (GAPS.md §4.G).
        df = self._add_coaching_change_features(df)

        # v14: late-season momentum + contracts + depth chart + scheme + combine
        df = self._add_combine_features(df)
        df = self._add_scheme_tendencies(df)
        df = self._add_contract_features(df)
        df = self._add_depth_chart_rank(df)
        df = self._add_late_season_momentum(df)

        # v16: weekly PFR advanced stats (pressure rate, contact quality, drop rate)
        df = self._add_weekly_pfr_features(df)

        # v17: seasonal PFR prior-season stats (bad throw %, pocket time, broken tackles, drop rate)
        df = self._add_seasonal_pfr_features(df)

        # v30: team-level offensive-line quality (pass-block/run-block,
        # from weekly_pfr) + PBP-derived pass-play participation rate
        # (GAPS.md §11.1.C/D/H follow-up)
        df = self._add_team_ol_features(df)
        df = self._add_pbp_pass_participation_features(df)

        # Impute NaN/inf
        df = self._impute_missing(df)

        self._update_feature_columns(df)
        return df

    def _add_prev_season_ppg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a single ``prev_season_ppg`` column.

        Within each (player_id, season), the per-game expanding mean
        of ``fantasy_points`` (shift(1) to exclude the current week)
        is the "season-to-date PPG before this game".  Shifting that
        series by one row within each player maps week-1 of a new
        season onto the LAST row of the prior season — which is the
        full prior-season PPG.  From week 2 onward it degenerates to
        a within-season partial-PPG proxy.  Defined exactly this way
        in _create_rolling_features; this helper is the extracted
        standalone version so create_causal_features can use it
        without pulling in the rest of the full-pipeline rolling
        feature zoo.

        Rookie-prior fill: rows where prev_season_ppg is NaN AND the
        row's season equals the player's earliest season in the
        frame are rookies — they get the position + draft-round
        prior from ``data/rookie_priors.json`` (Phase 4C).  Non-
        rookie NaN cases (gaps, retirements, traded players without
        prior rows) fall through to the default 0-fill in
        ``_impute_missing``."""
        if "fantasy_points" not in df.columns or "season" not in df.columns:
            return df
        season_expanding_ppg = (
            df.groupby(["player_id", "season"])["fantasy_points"]
              .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )
        df["_tmp_season_ppg"] = season_expanding_ppg
        df["prev_season_ppg"] = df.groupby("player_id")["_tmp_season_ppg"].shift(1)
        df.drop(columns=["_tmp_season_ppg"], inplace=True, errors="ignore")
        df = self._apply_rookie_prior(df)
        return df

    def _add_prior_season_late_ppg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prior season weeks 14-17 avg PPG as a cold-start form signal for QBs.

        At week 1 of a new season there are no rolling in-season stats.  A QB
        who finished the prior season on a hot streak (wk 14-17) is a better
        boom candidate than one who faded.  Lagged by 1 season so it's causal.
        Falls back to prev_season_ppg when the DB query fails or returns no rows.
        """
        import logging
        logger = logging.getLogger(__name__)

        if "player_id" not in df.columns or "season" not in df.columns:
            df["prior_season_late_ppg"] = 0.0
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH
            conn = sqlite3.connect(str(DB_PATH))
            rows = conn.execute("""
                SELECT player_id, season,
                       AVG(fantasy_points) AS prior_season_late_ppg
                FROM player_weekly_stats
                WHERE week >= 14 AND fantasy_points IS NOT NULL
                GROUP BY player_id, season
            """).fetchall()
            conn.close()
        except Exception as e:
            logger.warning("prior_season_late_ppg DB query failed: %s", e)
            df["prior_season_late_ppg"] = df.get(
                "prev_season_ppg", pd.Series(0.0, index=df.index)
            )
            return df

        if not rows:
            df["prior_season_late_ppg"] = df.get(
                "prev_season_ppg", pd.Series(0.0, index=df.index)
            )
            return df

        late = pd.DataFrame(rows, columns=["player_id", "season", "prior_season_late_ppg"])
        # Shift: use season N's late-season games to predict season N+1
        late["season"] = late["season"] + 1

        df = df.merge(late[["player_id", "season", "prior_season_late_ppg"]],
                      on=["player_id", "season"], how="left")

        # Fall back to prev_season_ppg for rows with no late-season data
        fallback = (
            df["prev_season_ppg"]
            if "prev_season_ppg" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        df["prior_season_late_ppg"] = df["prior_season_late_ppg"].fillna(fallback).fillna(0.0)
        return df

    def _add_age_curve_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific age curve (quadratic decline from peak)."""
        from config.settings import AGE_CURVE_PARAMS
        default_peak, default_coeff = 27, 0.005

        # This branch used to carry its own birth_date join, dead-coded
        # behind `if "age" in df.columns` -- season_long_features had already
        # populated `age` with a position constant by the time it ran, so the
        # working path never executed (GAPS.md 2026-08-19). One derivation
        # now, shared with season_long_features.
        if "season" not in df.columns:
            df["age_curve"] = 1.0
            return df
        from src.features.player_age import derive_age
        age = derive_age(df)

        if "position" in df.columns:
            peak = df["position"].map(
                {p: c["peak"] for p, c in AGE_CURVE_PARAMS.items()}
            ).fillna(default_peak)
            coeff = df["position"].map(
                {p: c["coefficient"] for p, c in AGE_CURVE_PARAMS.items()}
            ).fillna(default_coeff)
        else:
            peak, coeff = default_peak, default_coeff

        df["age_curve"] = (1.0 - coeff * ((age - peak) ** 2)).clip(lower=0.3)
        return df

    def _add_availability_3yr(self, df: pd.DataFrame) -> pd.DataFrame:
        """3-year games-played availability rate (shifted to avoid leakage)."""
        if "player_id" not in df.columns or "season" not in df.columns:
            df["availability_3yr"] = 1.0
            return df

        gp = (
            df.groupby(["player_id", "season"])
            .size()
            .reset_index(name="gp")
        )
        gp["gp"] = gp["gp"].clip(upper=17)
        gp = gp.sort_values(["player_id", "season"])

        gp["gp_3yr"] = gp.groupby("player_id")["gp"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        )
        gp["seasons_3yr"] = gp.groupby("player_id")["gp"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).count()
        )
        gp["availability_3yr"] = (
            gp["gp_3yr"] / (gp["seasons_3yr"] * 17)
        ).clip(0, 1.0).fillna(1.0)

        avail_map = gp.set_index(["player_id", "season"])["availability_3yr"].to_dict()
        df["availability_3yr"] = df.apply(
            lambda r: avail_map.get((r.get("player_id"), r.get("season")), 1.0), axis=1
        )
        return df

    def _add_career_year_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary: 1 if player's prior season was 30%+ above career avg PPG."""
        if "prev_season_ppg" not in df.columns or "player_id" not in df.columns:
            df["career_year_flag"] = 0
            return df

        career_ppg = df.groupby("player_id")["prev_season_ppg"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        pct_above = (df["prev_season_ppg"] - career_ppg) / career_ppg.clip(lower=1.0)
        df["career_year_flag"] = (pct_above >= 0.30).astype(int).fillna(0)
        return df

    def _add_bayesian_prior_ppg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shrink player PPG toward position average for thin-data players.

        alpha = min(1.0, career_games / 34). Rookies (0 games) get pure
        position average; 2+ year vets get their own PPG unaltered.
        """
        if "prev_season_ppg" not in df.columns or "position" not in df.columns:
            df["bayesian_prior_ppg"] = 0.0
            return df

        pos_avg = df.groupby("position")["prev_season_ppg"].transform("mean")
        career_games = df.groupby("player_id").cumcount()
        alpha = (career_games / 34.0).clip(upper=1.0)

        df["bayesian_prior_ppg"] = (
            alpha * df["prev_season_ppg"] + (1 - alpha) * pos_avg
        ).fillna(pos_avg).fillna(0.0)
        return df

    def _add_combine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combine athleticism features (speed score, BMI).

        Speed score = (weight * 200) / (forty^4). Higher = better
        athlete for size. RBs above 100 historically outperform.
        """
        if "player_id" not in df.columns:
            df["speed_score"] = 0.0
            df["bmi"] = 0.0
            return df

        try:
            import sqlite3
            from pathlib import Path
            from config.settings import DB_PATH, PROJECT_ROOT
            c = sqlite3.connect(str(DB_PATH))
            combine = pd.read_sql(
                "SELECT pfr_id, pos, forty, wt, ht FROM combine_data_v2 "
                "WHERE pos IN ('QB','RB','WR','TE') AND forty IS NOT NULL AND wt IS NOT NULL",
                c,
            )
            c.close()
            # Match via draft picks parquet (pfr_id → gsis_id)
            dp_path = PROJECT_ROOT / "data" / "draft_picks.parquet"
            if dp_path.exists():
                dp = pd.read_parquet(dp_path, columns=["pfr_player_id", "gsis_id"])
                dp = dp.dropna(subset=["pfr_player_id", "gsis_id"])
                combine = combine.merge(dp, left_on="pfr_id", right_on="pfr_player_id", how="inner")
            else:
                df["speed_score"] = 0.0
                df["bmi"] = 0.0
                return df
        except Exception:
            df["speed_score"] = 0.0
            df["bmi"] = 0.0
            return df

        combine_map = {}
        for _, r in combine.iterrows():
            try:
                forty_f = float(r["forty"])
                wt_f = float(r["wt"])
                gsis_id = r["gsis_id"]
                ss = (wt_f * 200.0) / (forty_f ** 4) if forty_f > 0 else 0
                ht = r.get("ht", "")
                if ht and "-" in str(ht):
                    parts = str(ht).split("-")
                    inches = int(parts[0]) * 12 + int(parts[1])
                else:
                    inches = 72
                bmi = (wt_f * 703) / (inches ** 2) if inches > 0 else 0
                combine_map[gsis_id] = (round(ss, 1), round(bmi, 1))
            except (ValueError, TypeError):
                pass

        if combine_map:
            df["speed_score"] = df["player_id"].map(
                lambda pid: combine_map.get(pid, (0.0, 0.0))[0]
            )
            df["bmi"] = df["player_id"].map(
                lambda pid: combine_map.get(pid, (0.0, 0.0))[1]
            )
        else:
            df["speed_score"] = 0.0
            df["bmi"] = 0.0
        return df

    def _add_scheme_tendencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team scheme tendency features from FTN charting data.

        `team_scheme_tendencies` only covers 2022+ (FTN charting does not
        exist earlier), and the lookup is shifted a season, so every
        team-week through 2022 is unknown. These used to be filled with a
        constant 0.5/0.1/0.5, which made the columns literal constants for
        2006-2022 and real values from 2023 -- a train/test discontinuity
        that showed up as part of the 2025 fold anomaly (GAPS.md
        2026-08-20). Unknown is now NaN, which LightGBM splits on natively,
        the same rule the snap columns already follow.
        """
        if "team" not in df.columns or "season" not in df.columns:
            for col in ["team_motion_rate", "team_play_action_rate", "team_shotgun_rate"]:
                df[col] = np.nan
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH
            c = sqlite3.connect(str(DB_PATH))
            # Get season-level averages per team (shifted: use prior season for preseason)
            rows = c.execute("""
                SELECT team, season, AVG(motion_rate) as motion,
                       AVG(play_action_rate) as pa, AVG(shotgun_rate) as sg
                FROM team_scheme_tendencies
                GROUP BY team, season
            """).fetchall()
            c.close()
        except Exception:
            for col in ["team_motion_rate", "team_play_action_rate", "team_shotgun_rate"]:
                df[col] = np.nan
            return df

        # Build lookup shifted by 1 season (use prior year's scheme for prediction)
        scheme_map = {}
        for team, season, motion, pa, sg in rows:
            # A NULL from the table is unknown, not a league-average guess.
            # `x or default` also swallowed a genuine 0.0 rate.
            scheme_map[(team, int(season) + 1)] = tuple(
                np.nan if v is None else round(float(v), 3) for v in (motion, pa, sg)
            )

        unknown = (np.nan, np.nan, np.nan)
        motion_vals, pa_vals, sg_vals = [], [], []
        for _, row in df.iterrows():
            key = (row.get("team"), row.get("season"))
            vals = scheme_map.get(key, unknown)
            motion_vals.append(vals[0])
            pa_vals.append(vals[1])
            sg_vals.append(vals[2])

        df["team_motion_rate"] = motion_vals
        df["team_play_action_rate"] = pa_vals
        df["team_shotgun_rate"] = sg_vals
        return df

    def _add_coaching_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-coach identity + coaching-change detection (GAPS.md §4.G).

        head_coach is pre-game, publicly known information for every week
        of the season it's used in (not a same-week outcome stat), so
        merging it in directly and diffing week-over-week is leakage-safe.

        Merges from team_coaching_staff if ``head_coach`` isn't already on
        ``df`` (get_all_players_for_training joins it in for the main
        training path, but this method stays self-contained for other
        callers). CoachingChangeDetector.add_coaching_change_features()
        also emits ``scheme_fit_score``, which collides with the
        destination-team-aware column of the same name already produced by
        _create_team_change_features — that one is more specific (accounts
        for team changes), so it's preserved across this call rather than
        silently overwritten.
        """
        if "head_coach" not in df.columns:
            if not all(c in df.columns for c in ["team", "season", "week"]):
                return df
            try:
                import sqlite3
                from config.settings import DB_PATH
                conn = sqlite3.connect(str(DB_PATH))
                coach_df = pd.read_sql_query(
                    "SELECT team, season, week, head_coach FROM team_coaching_staff",
                    conn,
                )
                conn.close()
            except Exception:
                coach_df = pd.DataFrame()
            if not coach_df.empty:
                df = df.merge(coach_df, on=["team", "season", "week"], how="left")

        from src.features.advanced_analytics import CoachingChangeDetector
        prior_scheme_fit = df["scheme_fit_score"].copy() if "scheme_fit_score" in df.columns else None
        df = CoachingChangeDetector().add_coaching_change_features(df)
        if prior_scheme_fit is not None:
            df["scheme_fit_score"] = prior_scheme_fit
        return df

    def _add_snap_roll3_known(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """How much of `snap_share_pct_roll3_mean`'s window was actually known.

        The rolling mean skips NaN, so a window with one known observation and
        two unknowns produces a confident-looking number from a single game.
        Without this column the model cannot tell that apart from three known
        observations.

        Defined as  n_known / n_available  over the same shift(1).rolling(3)
        window the mean uses:

            n_available = prior rows that exist for this player (1-3; fewer
                          early in his history)
            n_known     = those whose snap_share_pct is not NaN

        Deliberately divided by n_available, NOT by the window size 3. A rookie
        in week 2 has one prior game; if it is known, his history is as
        complete as it can be. Dividing by 3 would score him 0.33 and conflate
        "new player" with "missing data" -- the exact conflation this whole
        exercise exists to remove.

        1.0 when every available prior game was measured, 0.0 when none were
        (in which case the mean itself is NaN). Never NaN itself, so it
        survives the blanket fillna(0) in
        UtilizationScoreCalculator.calculate_all_scores and reaches the model.
        """
        if "snap_share_pct" not in df.columns or "player_id" not in df.columns:
            return df

        def _count(series: pd.Series) -> pd.Series:
            return series.shift(1).rolling(window, min_periods=1).count()

        n_known = df.groupby("player_id")["snap_share_pct"].transform(_count)
        n_available = (
            pd.Series(1.0, index=df.index)
            .groupby(df["player_id"])
            .transform(_count)
        )
        known = (n_known / n_available).where(n_available > 0)
        df["snap_share_pct_roll3_known"] = known.fillna(0.0)
        return df

    def _add_depth_chart_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add official depth chart rank (1=starter, 2=backup, etc.), as of
        each row's OWN week -- not just the week-1 preseason designation.

        Previously always looked up `week=1`, so every row (including
        week 15, 16, 17...) got the SAME preseason rank regardless of real
        in-season promotions/demotions. Now uses the most recent
        depth_charts snapshot with week <= this row's own week (that
        week's own snapshot is legitimate pre-game information, same
        category as Vegas lines/weather for that game -- not leakage,
        unlike using a LATER week's snapshot would be). See
        season_projection.py's `_lookup_depth_chart_rank_asof` for the
        stricter "week < target_week" convention used for synthetic
        (not-yet-played) rows -- deliberately a different, stricter cutoff
        than this one, not the same logic copy-pasted.
        """
        if "player_id" not in df.columns or "season" not in df.columns or "week" not in df.columns:
            df["depth_chart_rank"] = 1
            return df

        table = _load_depth_chart_asof_table()
        if table.empty:
            df["depth_chart_rank"] = 3
            return df

        key = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int) * 100 \
            + pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
        left = pd.DataFrame({
            "_idx": df.index,
            "gsis_id": df["player_id"].astype(str).values,
            "_key": key.values,
        }).sort_values("_key")

        merged = pd.merge_asof(
            left, table, left_on="_key", right_on="_key", by="gsis_id", direction="backward",
        )
        merged = merged.set_index("_idx").reindex(df.index)

        # Bound how far back a snapshot may be carried. Without this the
        # as-of match is unbounded: a season with no depth-chart coverage
        # silently inherits the newest prior season's ranks, however old.
        # depth_charts covers 2020-2025. 2025 was previously believed
        # unbackfillable and carried stale 2024 ranks; in fact
        # import_depth_charts([2025]) succeeds and simply returns a different
        # (daily, ESPN-style) schema, loaded by
        # scripts/backfill_depth_charts_2025.py on 2026-08-19. The bound
        # still matters for 2018/2019 and for players absent from a given
        # season's charts: one season of staleness is acceptable (a
        # prior-season depth chart is real, if noisy, information -- exactly
        # what a preseason projection would legitimately have), but carrying
        # a rank forward several seasons is not. Beyond the bound we fall
        # back to the neutral default rather than pretend to know.
        stale_seasons = (merged["_key"] // 100) - (merged["_snap_key"] // 100)
        too_stale = stale_seasons > DEPTH_CHART_MAX_STALENESS_SEASONS
        ranks = merged["depth_chart_rank"].where(~too_stale)
        df["depth_chart_rank"] = ranks.fillna(3).astype(int).values

        n_stale = int(too_stale.sum())
        if n_stale:
            # print(), not logging: this module has no logger and reports via
            # print/warnings throughout (matching the surrounding pipeline's
            # visible-progress style).
            print(f"  depth_chart_rank: {n_stale}/{len(df)} rows exceeded the "
                  f"{DEPTH_CHART_MAX_STALENESS_SEASONS}-season staleness bound "
                  f"and fell back to the neutral default")
        return df

    # Class-level cache: the contracts table is static per process, but
    # this method used to re-query and rebuild its lookup from scratch on
    # every weekly call in a backtest (same `engineer` instance reused
    # across weeks). Cache holds the merge-ready per-player lookup table.
    _contract_lookup_cache: Optional[pd.DataFrame] = None

    @classmethod
    def _get_contract_lookup_table(cls) -> pd.DataFrame:
        if cls._contract_lookup_cache is not None:
            return cls._contract_lookup_cache

        try:
            import sqlite3
            from config.settings import DB_PATH
            c = sqlite3.connect(str(DB_PATH))
            contracts = c.execute("""
                SELECT gsis_id, position, year_signed, years, apy
                FROM contracts
                WHERE gsis_id IS NOT NULL AND years > 0
                  AND position IN ('QB','RB','WR','TE')
            """).fetchall()
            c.close()
        except Exception:
            cls._contract_lookup_cache = pd.DataFrame(
                columns=["player_id", "_final_year", "_apy_rank"]
            )
            return cls._contract_lookup_cache

        # Build lookup: gsis_id → (final_year, apy, position), keeping
        # the most recently signed contract per player.
        contract_map = {}
        for gsis, pos, yr_signed, yrs, apy in contracts:
            final_year = int(yr_signed + yrs - 1)
            existing = contract_map.get(gsis)
            if existing is None or yr_signed > existing[0]:
                contract_map[gsis] = (yr_signed, final_year, float(apy or 0), pos)

        # Compute positional APY percentiles.
        from collections import defaultdict
        pos_apys = defaultdict(list)
        for gsis, (_, _, apy, pos) in contract_map.items():
            if apy > 0:
                pos_apys[pos].append(apy)
        pos_apys_sorted = {pos: sorted(vals) for pos, vals in pos_apys.items()}

        def _apy_pctile(apy, pos):
            vals = pos_apys_sorted.get(pos, [])
            if not vals or apy <= 0:
                return 0.5
            rank = sum(1 for v in vals if v <= apy)
            return rank / len(vals)

        rows = [
            (pid, final_year, _apy_pctile(apy, pos))
            for pid, (_, final_year, apy, pos) in contract_map.items()
        ]
        cls._contract_lookup_cache = pd.DataFrame(
            rows, columns=["player_id", "_final_year", "_apy_rank"]
        )
        return cls._contract_lookup_cache

    def _add_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contract year flag and APY rank from contracts table."""
        if "player_id" not in df.columns or "season" not in df.columns:
            df["is_contract_year"] = 0
            df["contract_apy_rank"] = 0.5
            return df

        lookup = self._get_contract_lookup_table()
        if lookup.empty:
            df["is_contract_year"] = 0
            df["contract_apy_rank"] = 0.5
            return df

        merged = df[["player_id", "season"]].merge(lookup, on="player_id", how="left")
        df["is_contract_year"] = (
            (merged["season"] == merged["_final_year"]) & merged["_final_year"].notna()
        ).astype(int).to_numpy()
        df["contract_apy_rank"] = merged["_apy_rank"].fillna(0.5).to_numpy()
        return df

    def _add_team_ol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Team-level offensive-line quality: pass-block (sack/pressure
        rate allowed) and run-block (yards-before-contact) grades, derived
        by re-aggregating weekly_pfr (already used at the individual
        player level by _add_weekly_pfr_features) to team-week (GAPS.md
        §11.1.H follow-up).

        Pass-block: identifies each team-week's starting QB as the row
        with the most pass-pressure activity (times_sacked +
        times_pressured + times_hurried + times_hit -- weekly_pfr has no
        dropback-count column to rank by directly) and uses that row's
        own times_pressured_pct/times_sacked directly, rather than
        averaging across every QB who took a snap that week (which would
        dilute the signal with mop-up-duty backups).

        Run-block: unweighted mean of rushing_yards_before_contact_avg
        across that team's RB rows for the week. A deliberate
        simplification -- weekly_pfr carries no per-RB carry count to
        weight by.

        Both are same-week team OUTCOME stats (like team_pct_11_personnel
        elsewhere in this file) -- rolled here via shift(1).rolling(3) per
        player, same causal-safety pattern as _add_weekly_pfr_features,
        rather than exposed raw.
        """
        raw_cols = ["team_sack_rate_allowed", "team_run_block_ybc_avg"]
        output_cols = [f"{c}_roll3_mean" for c in raw_cols]
        if "team" not in df.columns or "season" not in df.columns or "week" not in df.columns:
            for col in output_cols:
                df[col] = 0.0
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH

            c = sqlite3.connect(str(DB_PATH))
            pfr = pd.read_sql("""
                SELECT team, season, week, stat_type,
                       times_pressured_pct, times_sacked, times_blitzed,
                       times_hurried, times_hit,
                       rushing_yards_before_contact_avg
                FROM weekly_pfr
            """, c)
            c.close()

            pass_pfr = pfr[pfr["stat_type"] == "pass"].copy()
            pass_pfr["_activity"] = (
                pass_pfr["times_sacked"].fillna(0)
                + pass_pfr["times_hurried"].fillna(0)
                + pass_pfr["times_hit"].fillna(0)
            )
            starter_idx = (
                pass_pfr.groupby(["team", "season", "week"])["_activity"].idxmax()
            )
            starters = pass_pfr.loc[starter_idx, ["team", "season", "week", "times_pressured_pct"]]
            starters = starters.rename(columns={"times_pressured_pct": "team_sack_rate_allowed"})

            rush_pfr = pfr[pfr["stat_type"] == "rush"]
            run_block = (
                rush_pfr.groupby(["team", "season", "week"])["rushing_yards_before_contact_avg"]
                .mean()
                .rename("team_run_block_ybc_avg")
                .reset_index()
            )

            df = df.merge(starters, on=["team", "season", "week"], how="left")
            df = df.merge(run_block, on=["team", "season", "week"], how="left")
            df = df.sort_values(["player_id", "season", "week"])
            for raw_col, out_col in zip(raw_cols, output_cols):
                df[out_col] = (
                    df.groupby("player_id")[raw_col]
                    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
                )
            df = df.drop(columns=raw_cols, errors="ignore")
            for col in output_cols:
                df[col] = df[col].fillna(0.0)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"team_ol features skipped: {e}")
            for col in output_cols:
                if col not in df.columns:
                    df[col] = 0.0

        return df

    def _add_pbp_pass_participation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-player pass-play participation rate, from the
        pbp_pass_participation table (GAPS.md §11.1.C/D follow-up).

        NOT true route participation -- see pbp_stats_aggregator.py's
        get_pass_play_participation_from_pbp docstring for why that's
        genuinely unavailable. This is real PBP-derived signal (fraction
        of a team's actual dropback pass plays a player was on the field
        for), distinct from raw snap_share_pct.

        Same-week outcome stat, rolled via shift(1).rolling(3) per player
        before exposure, same causal-safety pattern as the OL features
        above.
        """
        raw_col = "pbp_pass_play_participation_pct"
        output_col = f"{raw_col}_roll3_mean"
        if "player_id" not in df.columns or "season" not in df.columns or "week" not in df.columns:
            df[output_col] = 0.0
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH

            c = sqlite3.connect(str(DB_PATH))
            part = pd.read_sql("""
                SELECT player_id, season, week, pass_play_participation_pct
                FROM pbp_pass_participation
            """, c)
            c.close()
            part = part.rename(columns={"pass_play_participation_pct": raw_col})

            df = df.merge(part, on=["player_id", "season", "week"], how="left")
            df = df.sort_values(["player_id", "season", "week"])
            df[output_col] = (
                df.groupby("player_id")[raw_col]
                .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            )
            df = df.drop(columns=[raw_col], errors="ignore")
            df[output_col] = df[output_col].fillna(0.0)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"pbp_pass_participation features skipped: {e}")
            df[output_col] = 0.0

        return df

    def _add_weekly_pfr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling weekly PFR advanced stats: QB pressure/blitz/sack rates, RB contact quality, WR/TE drop rate."""
        output_cols = [
            "qb_pressure_pct_roll3_mean",
            "qb_blitz_rate_roll3_mean",
            "qb_hurry_rate_roll3_mean",
            "qb_hit_rate_roll3_mean",
            "qb_sack_rate_roll3_mean",
            "rb_ybc_avg_roll3_mean",
            "rb_yac_avg_roll3_mean",
            "recv_drop_pct_roll3_mean",
        ]
        if "player_id" not in df.columns or "season" not in df.columns:
            for col in output_cols:
                df[col] = 0.0
            return df

        try:
            import sqlite3
            import logging
            from config.settings import DB_PATH
            from src.data.nfl_data_loader import get_pfr_to_gsis_map

            logger = logging.getLogger(__name__)
            pfr_map = get_pfr_to_gsis_map()

            c = sqlite3.connect(str(DB_PATH))
            pfr = pd.read_sql("""
                SELECT pfr_player_id, season, week, stat_type,
                       times_pressured_pct,
                       times_blitzed, times_hurried, times_hit, times_sacked,
                       rushing_yards_before_contact_avg,
                       rushing_yards_after_contact_avg,
                       receiving_drop_pct
                FROM weekly_pfr
            """, c)
            c.close()

            pfr["player_id"] = pfr["pfr_player_id"].map(pfr_map)
            pfr = pfr.dropna(subset=["player_id"])

            pass_pfr = pfr[pfr["stat_type"] == "pass"][
                ["player_id", "season", "week", "times_pressured_pct",
                 "times_blitzed", "times_hurried", "times_hit", "times_sacked"]
            ]
            rush_pfr = pfr[pfr["stat_type"] == "rush"][
                ["player_id", "season", "week",
                 "rushing_yards_before_contact_avg", "rushing_yards_after_contact_avg"]
            ]
            recv_pfr = pfr[pfr["stat_type"] == "rec"][
                ["player_id", "season", "week", "receiving_drop_pct"]
            ]

            df = df.merge(pass_pfr, on=["player_id", "season", "week"], how="left")
            df = df.merge(rush_pfr, on=["player_id", "season", "week"], how="left")
            df = df.merge(recv_pfr, on=["player_id", "season", "week"], how="left")

            # Cast all merged PFR columns to float to avoid LossySetitemError
            # when assigning rolling means back to int64 columns
            int_pfr_cols = ["times_blitzed", "times_hurried", "times_hit", "times_sacked"]
            for col in int_pfr_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            df = df.sort_values(["player_id", "season", "week"])
            roll_map = {
                "times_pressured_pct":              "qb_pressure_pct_roll3_mean",
                "times_blitzed":                    "qb_blitz_rate_roll3_mean",
                "times_hurried":                    "qb_hurry_rate_roll3_mean",
                "times_hit":                        "qb_hit_rate_roll3_mean",
                "times_sacked":                     "qb_sack_rate_roll3_mean",
                "rushing_yards_before_contact_avg": "rb_ybc_avg_roll3_mean",
                "rushing_yards_after_contact_avg":  "rb_yac_avg_roll3_mean",
                "receiving_drop_pct":               "recv_drop_pct_roll3_mean",
            }
            for src, dst in roll_map.items():
                if src in df.columns:
                    df[dst] = (
                        df.groupby("player_id")[src]
                        .transform(lambda x: x.astype(float).shift(1).rolling(3, min_periods=1).mean())
                    )

            # Drop intermediate raw columns — only keep the _roll3_mean outputs
            raw_cols = list(roll_map.keys())
            df = df.drop(columns=[c for c in raw_cols if c in df.columns], errors="ignore")

            # Fill NaN with positional median per season
            pos_col = "position" if "position" in df.columns else None
            for dst in output_cols:
                if dst not in df.columns:
                    df[dst] = 0.0
                elif pos_col:
                    df[dst] = df.groupby([pos_col, "season"])[dst].transform(
                        lambda x: x.fillna(x.median())
                    )
                df[dst] = df[dst].fillna(0.0)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"weekly_pfr features skipped: {e}")
            for col in output_cols:
                if col not in df.columns:
                    df[col] = 0.0

        return df

    def _add_seasonal_pfr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prior-season PFR advanced stats: QB bad-throw/pocket-time, RB broken tackles, WR/TE drop rate."""
        output_cols = [
            "qb_bad_throw_pct_prior",
            "qb_pocket_time_prior",
            "rb_broken_tackles_prior",
            "recv_drop_pct_season_prior",
        ]
        if "player_id" not in df.columns or "season" not in df.columns:
            for col in output_cols:
                df[col] = 0.0
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH
            from src.data.nfl_data_loader import get_pfr_to_gsis_map

            pfr_map = get_pfr_to_gsis_map()

            c = sqlite3.connect(str(DB_PATH))
            pfr = pd.read_sql("""
                SELECT pfr_player_id, season, stat_type,
                       bad_throw_pct, pocket_time,
                       broken_tackles_per_att,
                       rec_drop_pct
                FROM seasonal_pfr
            """, c)
            c.close()

            pfr["player_id"] = pfr["pfr_player_id"].map(pfr_map)
            pfr = pfr.dropna(subset=["player_id"])
            # Shift by 1 season: season N stats predict season N+1
            pfr["season"] = pfr["season"] + 1

            pass_pfr = pfr[pfr["stat_type"] == "pass"][
                ["player_id", "season", "bad_throw_pct", "pocket_time"]
            ].rename(columns={
                "bad_throw_pct": "qb_bad_throw_pct_prior",
                "pocket_time":   "qb_pocket_time_prior",
            })
            rush_pfr = pfr[pfr["stat_type"] == "rush"][
                ["player_id", "season", "broken_tackles_per_att"]
            ].rename(columns={"broken_tackles_per_att": "rb_broken_tackles_prior"})
            recv_pfr = pfr[pfr["stat_type"] == "rec"][
                ["player_id", "season", "rec_drop_pct"]
            ].rename(columns={"rec_drop_pct": "recv_drop_pct_season_prior"})

            df = df.merge(pass_pfr, on=["player_id", "season"], how="left")
            df = df.merge(rush_pfr, on=["player_id", "season"], how="left")
            df = df.merge(recv_pfr, on=["player_id", "season"], how="left")

            # Fill NaN with positional median per season
            pos_col = "position" if "position" in df.columns else None
            for col in output_cols:
                if col not in df.columns:
                    df[col] = 0.0
                elif pos_col:
                    df[col] = df.groupby([pos_col, "season"])[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                df[col] = df[col].fillna(0.0)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"seasonal_pfr features skipped: {e}")
            for col in output_cols:
                if col not in df.columns:
                    df[col] = 0.0

        return df

    def _add_late_season_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Late-season FP momentum: avg FP weeks 13+ / season avg.

        Ratio > 1.0 = trending up entering offseason (positive draft signal).
        Ratio < 1.0 = trending down (negative signal).
        Shifted by 1 season: for Week 1 of season N, this reflects
        the late-season momentum from season N-1.
        """
        if "fantasy_points" not in df.columns or "season" not in df.columns:
            df["fp_late6_vs_season"] = 1.0
            return df

        # Per-player per-season: late-season (week 13+) avg vs. season avg,
        # shifted 1 season so it reflects the *prior* season's momentum.
        season_ratio = (
            df.groupby(["player_id", "season"])
            .agg(fp_late6=("fantasy_points", lambda x: x[df.loc[x.index, "week"] >= 13].mean()),
                 fp_avg=("fantasy_points", "mean"))
            .reset_index()
        )
        season_ratio["fp_late6_vs_season"] = (
            season_ratio["fp_late6"] / season_ratio["fp_avg"].clip(lower=1.0)
        ).fillna(1.0).clip(0.3, 2.5)

        # Shift: map season N ratio to season N+1 rows
        season_ratio["season"] = season_ratio["season"] + 1

        merged = df[["player_id", "season"]].merge(
            season_ratio[["player_id", "season", "fp_late6_vs_season"]],
            on=["player_id", "season"], how="left",
        )
        df["fp_late6_vs_season"] = merged["fp_late6_vs_season"].fillna(1.0).to_numpy()
        return df

    # Module-level caches for the rookie-prior fill.  Loaded on first
    # use to avoid DB hits / JSON parses per fold.
    _ROOKIE_PRIORS_CACHE: Optional[Dict[str, Dict[str, float]]] = None
    _DRAFT_ROUND_CACHE: Optional[Dict[str, int]] = None

    @classmethod
    def _load_rookie_priors(cls) -> Optional[Dict[str, Dict[str, float]]]:
        if cls._ROOKIE_PRIORS_CACHE is not None:
            return cls._ROOKIE_PRIORS_CACHE
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "data" / "rookie_priors.json"
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        # Strip metadata, keep only the position→bucket→ppg dicts.
        cls._ROOKIE_PRIORS_CACHE = {
            k: v for k, v in raw.items() if not k.startswith("_")
        }
        return cls._ROOKIE_PRIORS_CACHE

    @classmethod
    def _load_draft_rounds(cls) -> Dict[str, int]:
        if cls._DRAFT_ROUND_CACHE is not None:
            return cls._DRAFT_ROUND_CACHE
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).resolve().parents[2] / "data" / "nfl_data.db"
        if not db_path.exists():
            cls._DRAFT_ROUND_CACHE = {}
            return cls._DRAFT_ROUND_CACHE
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT player_id, draft_round FROM draft_picks "
                "WHERE player_id IS NOT NULL AND draft_round IS NOT NULL"
            ).fetchall()
        cls._DRAFT_ROUND_CACHE = {pid: int(r) for pid, r in rows}
        return cls._DRAFT_ROUND_CACHE

    @staticmethod
    def _round_bucket_for(round_num: Optional[int]) -> str:
        if round_num is None:
            return "UDFA"
        if round_num == 1:
            return "rd1"
        if round_num in (2, 3):
            return "rd2_3"
        if round_num in (4, 5, 6, 7):
            return "rd4_7"
        return "UDFA"

    def _apply_rookie_prior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill ``prev_season_ppg`` NaN rows for rookies with the
        fitted position+draft-round prior.  Non-rookie NaN rows are
        left alone (they get the default 0-fill later)."""
        if "prev_season_ppg" not in df.columns:
            return df
        if "position" not in df.columns or "player_id" not in df.columns:
            return df
        priors = self._load_rookie_priors()
        if priors is None:
            return df  # priors artifact missing → skip (logged once via _impute_missing)

        mask_nan = df["prev_season_ppg"].isna()
        if not mask_nan.any():
            return df
        earliest_season = df.groupby("player_id")["season"].transform("min")
        is_rookie = df["season"] == earliest_season
        if not (mask_nan & is_rookie).any():
            return df

        round_lookup = self._load_draft_rounds()
        # Compute per-row bucket once.
        buckets = df["player_id"].map(
            lambda pid: self._round_bucket_for(round_lookup.get(pid))
        )

        for pos in ("QB", "RB", "WR", "TE"):
            pos_priors = priors.get(pos)
            if not pos_priors:
                continue
            for bucket in ("rd1", "rd2_3", "rd4_7", "UDFA"):
                val = pos_priors.get(bucket)
                if val is None:
                    continue
                sel = (
                    mask_nan
                    & is_rookie
                    & (df["position"] == pos)
                    & (buckets == bucket)
                )
                if sel.any():
                    df.loc[sel, "prev_season_ppg"] = val
        return df

    def _create_causal_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling features with 3-week window only (causal mode).

        Uses shift(1) to exclude the current week, matching the full
        pipeline's leakage prevention.
        """
        from config.settings import CAUSAL_ROLLING_WINDOW
        window = CAUSAL_ROLLING_WINDOW
        roll_cols = [
            "rushing_attempts", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_tds",
            "passing_attempts", "passing_tds",
            "yards_per_carry", "yards_per_target", "yards_per_attempt",
            "completion_pct",
            # v31: raw catch_rate already existed (_create_base_features
            # above) but was never rolled — Phase 6 (next_focus.md) gap.
            "catch_rate",
            # QB efficiency signals (from PBP, 2018+)
            "pass_epa_per_play", "pass_success_rate",
            "pass_wpa_per_play", "rush_epa_per_play",
            "td_rate", "int_rate",
            # Receiving EPA (from PBP, 2018+) — declared in CAUSAL_FEATURES
            # for WR/TE but was only ever rolled in the full-mode pipeline
            # (_create_rolling_features), never here. Dead feature in causal
            # mode until now.
            "recv_epa_per_target",
            # Position shares — computed in _create_base_features from
            # raw totals; roll3 of a same-week share is the leakage-
            # safe draftable signal (prior three games' average share).
            "target_share_pct", "rush_share_pct",
            "snap_share_pct", "air_yards_share_pct",
            "redzone_target_share_pct", "goal_line_carry_share_pct",
            # v24: team position-target allocation, concentration, lead-back
            # share (GAPS.md §4.A) — same leakage pattern (this-week team
            # share -> roll3 of the player's own game history).
            "team_rb_target_share", "team_wr_target_share", "team_te_target_share",
            "team_target_concentration", "team_rb_lead_share",
            # v24: tempo/plays-per-game (GAPS.md §4.D) — already joined onto
            # the raw frame from team_stats, just needs rolling to be causal.
            "team_plays", "team_pace_sec_per_play",
            # GAPS.md §3.3/§11.1.E: offensive personnel grouping usage
            # (11/12/21 personnel %) — already joined onto the raw frame
            # from team_personnel_stats (this-week outcome stat, like
            # team_plays above), just needs rolling to be causal. Only
            # populated 2016+ (PBP offense_personnel coverage start); NaN
            # before that, which the roll3 mean/dropna handling downstream
            # already tolerates the same way as other partial-coverage
            # features (e.g. recv_epa_per_target, 2018+).
            "team_pct_11_personnel", "team_pct_12_personnel",
            "team_pct_21_personnel", "team_pct_13_personnel",
            # v32: team pass-rate-over-expected (GAPS.md Phase 6b follow-up)
            # — already joined onto the raw frame from team_stats (week - 1,
            # same as team_plays above), just needs rolling to be causal.
            "team_neutral_pass_rate_oe",
        ]
        for col in roll_cols:
            if col not in df.columns:
                continue
            col_name = f"{col}_roll3_mean"
            df[col_name] = (
                df.groupby("player_id")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

        df = self._add_snap_roll3_known(df, window)

        # WOPR (Weighted Opportunity Rating), built from the already-rolled,
        # leakage-safe shares rather than this-week raw shares.
        if {"target_share_pct_roll3_mean", "air_yards_share_pct_roll3_mean"} <= set(df.columns):
            df["wopr_roll3"] = (
                1.5 * (df["target_share_pct_roll3_mean"] / 100)
                + 0.7 * (df["air_yards_share_pct_roll3_mean"] / 100)
            )

        # v19 Exp 1: fp_volatility_roll5 — rolling std of actual PPR FP over 5 weeks (lag-1).
        # Captures boom/bust tendency; players with high std are ceiling candidates.
        if "fantasy_points" in df.columns:
            df["fp_volatility_roll5"] = (
                df.groupby("player_id")["fantasy_points"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
                .fillna(0)
            )

        # v19 Exp 2: target_share_accel and snap_share_accel (RB/WR/TE only).
        # Acceleration = last week's value minus rolling 4-week mean.
        # Positive = usage trending up (boom candidate); negative = usage fading (bust risk).
        if "target_share_pct" in df.columns:
            _ts_lag1 = df.groupby("player_id")["target_share_pct"].transform(
                lambda x: x.shift(1)
            )
            _ts_roll4 = df.groupby("player_id")["target_share_pct"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=2).mean()
            )
            df["target_share_accel"] = (_ts_lag1 - _ts_roll4).fillna(0)

        if "snap_share_pct" in df.columns:
            _ss_lag1 = df.groupby("player_id")["snap_share_pct"].transform(
                lambda x: x.shift(1)
            )
            _ss_roll4 = df.groupby("player_id")["snap_share_pct"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=2).mean()
            )
            df["snap_share_accel"] = (_ss_lag1 - _ss_roll4).fillna(0)

        return df

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features."""
        new_cols: dict = {}

        # Efficiency metrics
        new_cols["yards_per_carry"] = safe_divide(
            df["rushing_yards"], df["rushing_attempts"]
        )
        new_cols["yards_per_target"] = safe_divide(
            df["receiving_yards"], df["targets"]
        )
        new_cols["yards_per_reception"] = safe_divide(
            df["receiving_yards"], df["receptions"]
        )
        new_cols["catch_rate"] = safe_divide(df["receptions"], df["targets"]) * 100

        # Advanced PBP efficiency (EPA/WPA per opportunity)
        # Prefer pass_plays/rush_plays/recv_targets if populated; fall back to
        # passing_attempts/rushing_attempts/targets when the PBP columns are empty.
        pass_plays = df.get("pass_plays", pd.Series(0, index=df.index))
        if pass_plays.sum() == 0 and "passing_attempts" in df.columns:
            pass_plays = df["passing_attempts"]
        rush_plays = df.get("rush_plays", pd.Series(0, index=df.index))
        if rush_plays.sum() == 0 and "rushing_attempts" in df.columns:
            rush_plays = df["rushing_attempts"]
        recv_targets = df.get("recv_targets", pd.Series(0, index=df.index))
        if recv_targets.sum() == 0 and "targets" in df.columns:
            recv_targets = df["targets"]
        if "pass_epa" in df.columns:
            new_cols["pass_epa_per_play"] = safe_divide(df["pass_epa"], pass_plays)
        if "rush_epa" in df.columns:
            new_cols["rush_epa_per_play"] = safe_divide(df["rush_epa"], rush_plays)
        if "recv_epa" in df.columns:
            new_cols["recv_epa_per_target"] = safe_divide(df["recv_epa"], recv_targets)
        if "pass_wpa" in df.columns:
            new_cols["pass_wpa_per_play"] = safe_divide(df["pass_wpa"], pass_plays)
        if "rush_wpa" in df.columns:
            new_cols["rush_wpa_per_play"] = safe_divide(df["rush_wpa"], rush_plays)
        if "recv_wpa" in df.columns:
            new_cols["recv_wpa_per_target"] = safe_divide(df["recv_wpa"], recv_targets)

        # QB-specific (only if columns exist)
        if "passing_completions" in df.columns and "passing_attempts" in df.columns:
            new_cols["completion_pct"] = safe_divide(
                df["passing_completions"], df["passing_attempts"]
            ) * 100
            new_cols["yards_per_attempt"] = safe_divide(
                df.get("passing_yards", 0), df["passing_attempts"]
            )
            new_cols["td_rate"] = safe_divide(
                df.get("passing_tds", 0), df["passing_attempts"]
            ) * 100
            new_cols["int_rate"] = safe_divide(
                df.get("interceptions", 0), df["passing_attempts"]
            ) * 100

        # Volume metrics (with safe defaults)
        rushing_attempts = df.get("rushing_attempts", pd.Series(0, index=df.index))
        receptions = df.get("receptions", pd.Series(0, index=df.index))
        rushing_yards = df.get("rushing_yards", pd.Series(0, index=df.index))
        receiving_yards = df.get("receiving_yards", pd.Series(0, index=df.index))
        rushing_tds = df.get("rushing_tds", pd.Series(0, index=df.index))
        receiving_tds = df.get("receiving_tds", pd.Series(0, index=df.index))
        targets = df.get("targets", pd.Series(0, index=df.index))

        total_touches = rushing_attempts + receptions
        total_yards = rushing_yards + receiving_yards
        total_tds = rushing_tds + receiving_tds
        opportunities = rushing_attempts + targets

        new_cols["total_touches"] = total_touches
        new_cols["total_yards"] = total_yards
        new_cols["total_tds"] = total_tds
        new_cols["opportunities"] = opportunities
        new_cols["weighted_opportunities"] = rushing_attempts * 2 + targets
        new_cols["yards_per_td"] = safe_divide(total_yards, total_tds.replace(0, np.nan))

        # QB advanced: air yards per attempt, TD/INT ratio, deep ball attempts
        if "air_yards" in df.columns and "passing_attempts" in df.columns:
            new_cols["air_yards_per_attempt"] = safe_divide(df["air_yards"], df["passing_attempts"])
        if "passing_tds" in df.columns and "interceptions" in df.columns:
            new_cols["td_int_ratio"] = safe_divide(
                df["passing_tds"], df["interceptions"].replace(0, 0.5)
            )
        if "deep_pass_attempts" in df.columns:
            new_cols["deep_ball_pct"] = safe_divide(
                df["deep_pass_attempts"], df.get("passing_attempts", pd.Series(1, index=df.index))
            ) * 100

        # RB advanced: yards after contact, broken tackles
        if "yards_after_contact" in df.columns:
            new_cols["yac_per_carry"] = safe_divide(df["yards_after_contact"], df.get("rushing_attempts", pd.Series(1, index=df.index)))
        if "broken_tackles" in df.columns:
            new_cols["broken_tackle_rate"] = safe_divide(
                df["broken_tackles"], df.get("rushing_attempts", pd.Series(1, index=df.index))
            )

        # WR/TE advanced: average depth of target, yards after catch, contested catch rate
        if "average_depth_of_target" not in df.columns:
            if "air_yards" in df.columns and "targets" in df.columns:
                new_cols["average_depth_of_target"] = safe_divide(df["air_yards"], df["targets"])
        if "yards_after_catch" in df.columns and "receptions" in df.columns:
            new_cols["yac_per_reception"] = safe_divide(df["yards_after_catch"], df["receptions"])
        if "contested_catches" in df.columns and "contested_targets" in df.columns:
            new_cols["contested_catch_rate"] = safe_divide(df["contested_catches"], df["contested_targets"])
        if "slot_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["slot_pct"] = safe_divide(df["slot_snaps"], df["snap_count"]) * 100

        # Route participation rate for RB receiving work
        if "routes_run" in df.columns and "snap_count" in df.columns:
            new_cols["route_participation_rate"] = safe_divide(df["routes_run"], df["snap_count"])

        # Game script indicators
        new_cols["is_home"] = (df["home_away"] == "home").astype(int) if "home_away" in df.columns else 0

        # Season progress
        new_cols["season_week_pct"] = df["week"] / 18

        # -----------------------------------------------------------
        # Per-game position shares.  ``utilization_scores`` is empty
        # in this environment (see CAUSAL_FEATURES audit 2026-04-24),
        # so the ``*_pct`` columns were silently dropped.  Compute
        # them here from raw stats in player_weekly_stats: team
        # totals are derived per (team, season, week) groupby, then
        # each player's share = their count / team's count.
        #
        # These shares are THIS-WEEK values and therefore leakage-
        # prone as direct inputs.  The leakage-safe consumer is the
        # ``{col}_roll3_mean`` variant produced by
        # ``_create_causal_rolling_features`` / ``_create_rolling_features``,
        # which applies shift(1).rolling(3).mean().
        # -----------------------------------------------------------
        if all(c in df.columns for c in ["team", "season", "week"]):
            team_grp = df.groupby(["team", "season", "week"], sort=False)
            if "targets" in df.columns:
                new_cols["target_share_pct"] = (
                    safe_divide(df["targets"], team_grp["targets"].transform("sum"))
                    * 100
                )
            if "rushing_attempts" in df.columns:
                new_cols["rush_share_pct"] = (
                    safe_divide(
                        df["rushing_attempts"],
                        team_grp["rushing_attempts"].transform("sum"),
                    ) * 100
                )
            if "air_yards" in df.columns:
                new_cols["air_yards_share_pct"] = (
                    safe_divide(df["air_yards"], team_grp["air_yards"].transform("sum"))
                    * 100
                )
            if "redzone_targets" in df.columns:
                new_cols["redzone_target_share_pct"] = (
                    safe_divide(df["redzone_targets"], team_grp["redzone_targets"].transform("sum"))
                    * 100
                )
            if "rush_inside_5" in df.columns:
                new_cols["goal_line_carry_share_pct"] = (
                    safe_divide(df["rush_inside_5"], team_grp["rush_inside_5"].transform("sum"))
                    * 100
                )
            # GAPS.md §4.A: position-specific target allocation within team.
            # A 65% team pass rate means different things for a spread vs.
            # heavy-personnel team; these say WHO the targets go to.
            if "targets" in df.columns and "position" in df.columns:
                pos_target_sums = (
                    df.groupby(["team", "season", "week", "position"])["targets"]
                    .transform("sum")
                )
                team_target_totals = team_grp["targets"].transform("sum")
                for pos_code, col_name in [
                    ("RB", "team_rb_target_share"),
                    ("WR", "team_wr_target_share"),
                    ("TE", "team_te_target_share"),
                ]:
                    pos_mask = df["position"] == pos_code
                    pos_sum_this_pos = pos_target_sums.where(pos_mask)
                    # pos_target_sums is already scoped to each row's own
                    # position group, so restrict to rows of pos_code and
                    # broadcast that team-week's value to the whole team
                    # (including non-pos_code rows) via a team-week map.
                    per_team_week = (
                        pos_sum_this_pos.groupby(
                            [df["team"], df["season"], df["week"]]
                        ).transform("max")  # single value per team-week, NaN elsewhere
                    )
                    new_cols[col_name] = safe_divide(per_team_week, team_target_totals) * 100

                # Herfindahl index of target distribution (0-1): higher =
                # concentrated on fewer players, lower = spread across many.
                player_target_share = safe_divide(df["targets"], team_target_totals)
                new_cols["team_target_concentration"] = (
                    (player_target_share ** 2)
                    .groupby([df["team"], df["season"], df["week"]])
                    .transform("sum")
                )

            # Lead-back vs. committee: the top RB's share of the team's
            # RB rushing attempts that week (100 = true bell-cow, ~50 = even
            # committee split).
            if "rushing_attempts" in df.columns and "position" in df.columns:
                rb_mask = df["position"] == "RB"
                rb_rush_sum = (
                    df["rushing_attempts"].where(rb_mask)
                    .groupby([df["team"], df["season"], df["week"]])
                    .transform("sum")
                )
                rb_share = safe_divide(df["rushing_attempts"].where(rb_mask), rb_rush_sum)
                new_cols["team_rb_lead_share"] = (
                    (rb_share * 100)
                    .groupby([df["team"], df["season"], df["week"]])
                    .transform("max")
                )
        if "snap_count" in df.columns and "team_snaps" in df.columns:
            # Recomputed here even though UtilizationScoreCalculator already
            # produced it -- and it OVERWRITES that value, so it must honour
            # the same missingness mode. Via safe_divide it returned 0.0 for
            # an unknown snap share, discarding the NaN the calculator had
            # just preserved and leaving snap_share_pct_roll3_mean to be
            # averaged from fabricated zeros.
            from src.features.utilization_score import snap_share_pct as _snap_share_pct
            new_cols["snap_share_pct"] = _snap_share_pct(
                df["snap_count"], df["team_snaps"]
            )

        df = df.assign(**new_cols)
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling average features for key metrics."""
        rolling_cols = [
            "fantasy_points", "rushing_yards", "rushing_attempts", "rushing_tds",
            "receiving_yards", "receptions", "targets", "receiving_tds",
            "passing_yards", "passing_attempts", "passing_tds", "interceptions",
            "total_touches", "total_yards", "opportunities", "utilization_score",
            "yards_per_carry", "yards_per_target", "catch_rate",
            # Position shares (see _create_base_features) — roll-ed
            # variants are the leakage-safe consumers used by
            # CAUSAL_FEATURES; the raw _pct columns are this-week.
            "target_share_pct", "rush_share_pct",
            "snap_share_pct", "air_yards_share_pct",
            # PBP efficiency (computed in _create_base_features when
            # pass_epa / rush_epa columns are populated)
            "pass_epa_per_play", "rush_epa_per_play", "recv_epa_per_target",
            "pass_success_rate", "completion_pct",
        ]
        
        # Filter to columns that exist
        rolling_cols = [c for c in rolling_cols if c in df.columns]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_cols = {}
        
        for window in self.rolling_windows:
            for col in rolling_cols:
                # Rolling mean (shifted to avoid leakage)
                new_cols[f"{col}_roll{window}_mean"] = df.groupby("player_id")[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std for volatility (per requirements: utilization_score volatility too)
                if col in ["fantasy_points", "total_yards", "total_touches", "utilization_score"]:
                    new_cols[f"{col}_roll{window}_std"] = df.groupby("player_id")[col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
                    )
        
        # Exponential weighted averages (more weight on recent games)
        # Per requirements: heavy EWMA weighting for 4-week model; multiple spans
        ewm_cols = ["fantasy_points", "total_yards", "opportunities", "utilization_score"]
        ewm_cols = [c for c in ewm_cols if c in df.columns]
        
        for col in ewm_cols:
            for span in [3, 5, 8]:
                new_cols[f"{col}_ewm{span}"] = df.groupby("player_id")[col].transform(
                    lambda x, s=span: x.shift(1).ewm(span=s, adjust=False).mean()
                )
        
        # Regression-to-mean features for long-horizon (18-week) model
        # Per requirements: 18-week model should heavily weight regression to mean
        # IMPORTANT: Use expanding (causal) position-level aggregates to avoid lookahead bias.
        if "fantasy_points" in df.columns:
            if "position" in df.columns:
                # Bounded rolling position-level mean/std (window=200 ~3 seasons of
                # position data) to avoid sample-size growth leaking temporal position.
                pos_rolling_mean = df.groupby("position")["fantasy_points"].transform(
                    lambda x: x.shift(1).rolling(window=200, min_periods=1).mean()
                )
                player_ewm = df.groupby("player_id")["fantasy_points"].transform(
                    lambda x: x.shift(1).ewm(span=8, adjust=False).mean()
                )
                new_cols["fp_deviation_from_pos_mean"] = player_ewm - pos_rolling_mean
                pos_rolling_std = df.groupby("position")["fantasy_points"].transform(
                    lambda x: x.shift(1).rolling(window=200, min_periods=2).std()
                ).clip(lower=1.0)
                new_cols["fp_regression_to_mean_z"] = (player_ewm - pos_rolling_mean) / pos_rolling_std
            # Season-level mean for same player: use expanding mean within each
            # (player, season) group to avoid using future games within the season.
            if "season" in df.columns:
                season_expanding_ppg = df.groupby(["player_id", "season"])["fantasy_points"].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )
                df["_tmp_season_ppg"] = season_expanding_ppg
                new_cols["prev_season_ppg"] = df.groupby("player_id")["_tmp_season_ppg"].shift(1)
                df.drop(columns=["_tmp_season_ppg"], inplace=True, errors="ignore")
        
        if "utilization_score" in df.columns and "position" in df.columns:
            pos_util_expanding_mean = df.groupby("position")["utilization_score"].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            player_util_ewm = df.groupby("player_id")["utilization_score"].transform(
                lambda x: x.shift(1).ewm(span=8, adjust=False).mean()
            )
            new_cols["util_deviation_from_pos_mean"] = player_util_ewm - pos_util_expanding_mean
        
        # Add all new columns at once using pd.concat to avoid fragmentation
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for recent performance."""
        lag_cols = [
            "fantasy_points", "total_yards", "total_touches", "total_tds",
            "utilization_score", "snap_share"
        ]
        
        lag_cols = [c for c in lag_cols if c in df.columns]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_cols = {}
        for lag in self.lag_weeks:
            for col in lag_cols:
                new_cols[f"{col}_lag{lag}"] = df.groupby("player_id")[col].shift(lag)
        
        # Add all new columns at once
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend features to capture momentum."""
        trend_cols = ["fantasy_points", "total_yards", "utilization_score"]
        trend_cols = [c for c in trend_cols if c in df.columns]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_cols = {}
        
        for col in trend_cols:
            # Short-term trend (last 3 games slope)
            new_cols[f"{col}_trend3"] = df.groupby("player_id")[col].transform(
                lambda x: self._calculate_trend(x, 3)
            )
            
            # Medium-term trend (last 5 games slope)
            new_cols[f"{col}_trend5"] = df.groupby("player_id")[col].transform(
                lambda x: self._calculate_trend(x, 5)
            )
            
            # Long-term trend (last 8 games slope) per requirements III.A
            new_cols[f"{col}_trend8"] = df.groupby("player_id")[col].transform(
                lambda x: self._calculate_trend(x, 8)
            )
        
        # Week-over-week change (shift(1) to avoid leakage - use prior week's change only)
        for col in ["fantasy_points", "total_yards"]:
            if col in df.columns:
                shifted = df.groupby("player_id")[col].shift(1)
                new_cols[f"{col}_wow_change"] = shifted.diff()
                new_cols[f"{col}_wow_pct_change"] = shifted.pct_change(fill_method=None)
        
        # Add all new columns at once
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        return df
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend (slope) over a rolling window."""
        def slope(x):
            if len(x) < 2:
                return 0
            x_clean = x.dropna()
            if len(x_clean) < 2:
                return 0
            return np.polyfit(range(len(x_clean)), x_clean, 1)[0]
        
        return series.shift(1).rolling(window=window, min_periods=2).apply(slope, raw=False)
    
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on opponent strength."""
        # Fantasy points allowed by opponent (if available)
        opp_cols = [
            "fantasy_points_allowed_qb", "fantasy_points_allowed_rb",
            "fantasy_points_allowed_wr", "fantasy_points_allowed_te"
        ]
        
        for col in opp_cols:
            if col in df.columns:
                # Normalize to z-score using expanding (causal) mean/std.
                # shift(1) excludes the current row to prevent self-referential leakage.
                # Sort by (season, week) first so expanding windows respect temporal order.
                if "season" in df.columns and "week" in df.columns:
                    sorted_df = df.sort_values(["season", "week"])
                    shifted = sorted_df[col].shift(1)
                    expanding_mean = shifted.expanding(min_periods=1).mean()
                    expanding_std = shifted.expanding(min_periods=2).std().clip(lower=1e-6)
                    df[f"{col}_zscore"] = ((sorted_df[col] - expanding_mean) / expanding_std).reindex(df.index)
                else:
                    shifted = df[col].shift(1)
                    expanding_mean = shifted.expanding(min_periods=1).mean()
                    expanding_std = shifted.expanding(min_periods=2).std().clip(lower=1e-6)
                    df[f"{col}_zscore"] = (df[col] - expanding_mean) / expanding_std
        
        # Create position-specific opponent feature
        if "position" in df.columns:
            df["opp_fpts_allowed"] = np.nan

            for pos in POSITIONS:
                col = f"fantasy_points_allowed_{pos.lower()}"
                if col in df.columns:
                    mask = df["position"] == pos
                    # Convert values to float to avoid dtype incompatibility
                    values = pd.to_numeric(df.loc[mask, col], errors='coerce')
                    df.loc[mask, "opp_fpts_allowed"] = values.values

        # Season-to-date expanding lag-1 FP-allowed per opponent-position.
        # This is Phase 2 of the predictive-ceiling workstream
        # (docs/PREDICTIVE_CEILING_PLAN.md).  The existing opp_fpts_allowed
        # above is single-week-prior (noisy); the s2d version is the mean
        # of every prior week in the season, which the council spec'd as
        # the smoother signal.  Both features coexist so Ridge can weight
        # each independently.
        df = self._add_opp_fpts_allowed_s2d_lag1(df)

        # DVOA-style opponent-adjustment (GAPS.md §11.1.F): corrects the
        # s2d FPA-allowed figure for the strength of offenses actually
        # faced. Both features coexist, same rationale as above.
        df = self._add_opp_fpts_allowed_dvoa_adjusted_lag1(df)

        # Add comprehensive team-level features (TeamA = player's team, TeamB = opponent)
        df = self._add_team_matchup_features(df)

        return df

    def _add_opp_fpts_allowed_s2d_lag1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add season-to-date (through week N-1) FP-allowed at player's position.

        Queries the `team_defense_stats` table directly, computes
        ``shift(1).expanding(min_periods=1).mean()`` per (team, season) on each
        ``fantasy_points_allowed_{pos}`` column, then merges onto the player
        frame via (opponent, season, week) and resolves per-player position.

        On cache miss or insufficient coverage, the feature is filled with 0.0
        and a WARNING is logged if >10 % of rows defaulted — the silent-
        fallback trap documented in docs/PHASE_1_VEGAS_FINDINGS.md applies
        here too, and we want any downstream backtest log to surface it.
        """
        required = {"opponent", "season", "week", "position"}
        if not required.issubset(df.columns):
            df["opp_fpts_allowed_s2d_lag1"] = 0.0
            return df

        import logging
        logger = logging.getLogger(__name__)

        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique())
            if not seasons:
                raise ValueError("No seasons")
            season_list = ",".join(str(int(s)) for s in seasons)
            with db._get_connection() as conn:
                tds = pd.read_sql_query(
                    "SELECT team, season, week, "
                    "fantasy_points_allowed_qb, fantasy_points_allowed_rb, "
                    "fantasy_points_allowed_wr, fantasy_points_allowed_te "
                    f"FROM team_defense_stats WHERE season IN ({season_list}) "
                    "ORDER BY team, season, week",
                    conn,
                )
        except Exception as e:
            logger.warning(
                "team_defense_stats load failed (%s: %s); "
                "opp_fpts_allowed_s2d_lag1 will default to 0.0 for every row. "
                "Run DatabaseManager.ensure_team_defense_stats() to populate.",
                type(e).__name__, e,
            )
            df["opp_fpts_allowed_s2d_lag1"] = 0.0
            return df

        if tds.empty:
            logger.warning(
                "team_defense_stats returned zero rows for seasons %s; "
                "opp_fpts_allowed_s2d_lag1 will default to 0.0.",
                seasons,
            )
            df["opp_fpts_allowed_s2d_lag1"] = 0.0
            return df

        # Per-(team, season) expanding mean through week N-1.  Pattern lifted
        # from the canonical `season_expanding_ppg` at line ~381 but scoped
        # to (team, season) on the team_defense_stats frame rather than the
        # player frame.
        for pos in POSITIONS:
            col = f"fantasy_points_allowed_{pos.lower()}"
            if col not in tds.columns:
                continue
            tds[f"{col}_s2d_lag1"] = (
                tds.groupby(["team", "season"])[col]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            )

        # Merge onto player frame using (opponent, season, week).  Rename
        # `team` -> `opponent` on the team_defense_stats side for the join.
        s2d_cols = [f"fantasy_points_allowed_{p.lower()}_s2d_lag1" for p in POSITIONS]
        s2d_cols = [c for c in s2d_cols if c in tds.columns]
        merge_right = tds[["team", "season", "week"] + s2d_cols].rename(
            columns={"team": "opponent"}
        )

        before_cols = set(df.columns)
        df = df.merge(merge_right, on=["opponent", "season", "week"], how="left")

        # Resolve per-player position: opp_fpts_allowed_s2d_lag1 is the
        # merged value for the player's own position.
        df["opp_fpts_allowed_s2d_lag1"] = np.nan
        for pos in POSITIONS:
            src_col = f"fantasy_points_allowed_{pos.lower()}_s2d_lag1"
            if src_col not in df.columns:
                continue
            mask = df["position"] == pos
            df.loc[mask, "opp_fpts_allowed_s2d_lag1"] = pd.to_numeric(
                df.loc[mask, src_col], errors="coerce"
            )

        # Coverage check before the NaN fill — anything above 10 % default
        # rate is almost certainly a missing-backfill bug (per Step 1 of
        # Phase 2, the 2025 team_defense_stats needed a manual backfill).
        n_total = len(df)
        n_missing = int(df["opp_fpts_allowed_s2d_lag1"].isna().sum())
        if n_total and n_missing / n_total > 0.10:
            logger.warning(
                "opp_fpts_allowed_s2d_lag1 defaulted on %.1f%% of rows (>10%% "
                "threshold).  team_defense_stats may be missing weeks for the "
                "seasons in scope (%s).  Run "
                "DatabaseManager.ensure_team_defense_stats(season) for any "
                "missing season.",
                n_missing / n_total * 100, seasons,
            )

        df["opp_fpts_allowed_s2d_lag1"] = df["opp_fpts_allowed_s2d_lag1"].fillna(0.0)

        # Drop intermediate per-position columns added by the merge so they
        # don't leak into the feature set downstream.
        drop_cols = [c for c in df.columns if c not in before_cols and c != "opp_fpts_allowed_s2d_lag1"]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def _add_opp_fpts_allowed_dvoa_adjusted_lag1(self, df: pd.DataFrame) -> pd.DataFrame:
        """DVOA-style opponent-adjusted defensive FPA (GAPS.md §11.1.F).

        opp_fpts_allowed_s2d_lag1 (above) is contaminated by schedule
        strength: a defense that "allows" 25 PPG to RBs looks bad even if
        it faced Derrick Henry/Saquon Barkley/Josh Jacobs — that's expected
        given who it played, not a defensive weakness. This computes, for
        each defense-week, the residual between what it actually allowed
        and what an average-strength offense would have been expected to
        produce against it (based on that specific opponent's own
        season-to-date output through the prior week), then takes a
        causal expanding mean of that residual per defense — i.e. "does
        this defense over- or under-perform its raw FPA-allowed number,
        once you account for who it's actually played."

        Two-pass computation, entirely on team-week-deduplicated,
        (team, season, week)-sorted tables — NEVER on the per-player `df`
        directly. df is typically sorted by player_id first (see
        create_causal_features), so a groupby("team").shift() run directly
        on it would compare whichever rows happen to be adjacent for a
        team — i.e. two different players' unrelated weeks — instead of
        consecutive team-weeks. This exact bug was found and fixed in
        CoachingChangeDetector (advanced_analytics.py) earlier in this
        project; same risk class here, avoided by construction.

        Pass 1: each team's own offensive output by position, season-to-
        date through the prior week (off_s2d_lag1_{pos}), from
        team_offense_stats.
        Pass 2: for each defense-week, look up the opponent it actually
        faced (team_stats.opponent) and that opponent's own pre-game
        expected output (off_s2d_lag1_{pos}, as of that same week — no
        additional lag needed, Pass 1's shift(1) already makes it causal).
        residual = actual_fpts_allowed - opponent_expected_output.
        Expanding-mean the residual per defense, season (shift(1) first),
        merge onto df by (opponent, season, week), resolve per position —
        same pattern as opp_fpts_allowed_s2d_lag1 above.
        """
        required = {"opponent", "season", "week", "position"}
        if not required.issubset(df.columns):
            df["opp_fpts_allowed_dvoa_adjusted_lag1"] = 0.0
            return df

        import logging
        logger = logging.getLogger(__name__)

        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique())
            if not seasons:
                raise ValueError("No seasons")
            season_list = ",".join(str(int(s)) for s in seasons)
            with db._get_connection() as conn:
                off = pd.read_sql_query(
                    "SELECT team, season, week, "
                    "fantasy_points_produced_qb, fantasy_points_produced_rb, "
                    "fantasy_points_produced_wr, fantasy_points_produced_te "
                    f"FROM team_offense_stats WHERE season IN ({season_list}) "
                    "ORDER BY team, season, week",
                    conn,
                )
                deff = pd.read_sql_query(
                    "SELECT team, season, week, "
                    "fantasy_points_allowed_qb, fantasy_points_allowed_rb, "
                    "fantasy_points_allowed_wr, fantasy_points_allowed_te "
                    f"FROM team_defense_stats WHERE season IN ({season_list})",
                    conn,
                )
                opp_link = pd.read_sql_query(
                    "SELECT team, season, week, opponent "
                    f"FROM team_stats WHERE season IN ({season_list}) "
                    "AND opponent IS NOT NULL AND opponent != ''",
                    conn,
                )
        except Exception as e:
            logger.warning(
                "team_offense_stats/team_defense_stats/team_stats load failed "
                "(%s: %s); opp_fpts_allowed_dvoa_adjusted_lag1 will default "
                "to 0.0 for every row. Run DatabaseManager.ensure_team_offense_stats() "
                "and ensure_team_defense_stats() to populate.",
                type(e).__name__, e,
            )
            df["opp_fpts_allowed_dvoa_adjusted_lag1"] = 0.0
            return df

        if off.empty or deff.empty or opp_link.empty:
            logger.warning(
                "team_offense_stats/team_defense_stats/team_stats returned "
                "zero rows for seasons %s; opp_fpts_allowed_dvoa_adjusted_lag1 "
                "will default to 0.0.", seasons,
            )
            df["opp_fpts_allowed_dvoa_adjusted_lag1"] = 0.0
            return df

        # Pass 1: each team's own causal season-to-date offensive output.
        for pos in POSITIONS:
            col = f"fantasy_points_produced_{pos.lower()}"
            if col not in off.columns:
                continue
            off[f"{col}_s2d_lag1"] = (
                off.groupby(["team", "season"])[col]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            )

        # opp_link: one row per (defense-team, season, week) -> opponent
        # faced that week. Deduplicate defensively (team_stats is already
        # one row per team-week by construction, but don't assume it).
        opp_link = opp_link.drop_duplicates(subset=["team", "season", "week"])

        # Pass 2: attach the opponent's pre-game expected output to each
        # defense-week, compute the residual against what was actually
        # allowed, then causally expanding-mean the residual per defense.
        s2d_cols = [f"fantasy_points_produced_{p.lower()}_s2d_lag1" for p in POSITIONS]
        s2d_cols = [c for c in s2d_cols if c in off.columns]
        off_for_join = off[["team", "season", "week"] + s2d_cols].rename(
            columns={"team": "opponent_team"}
        )

        merged = opp_link.merge(
            off_for_join,
            left_on=["opponent", "season", "week"],
            right_on=["opponent_team", "season", "week"],
            how="left",
        )
        merged = merged.merge(
            deff, on=["team", "season", "week"], how="left",
        )
        merged = merged.sort_values(["team", "season", "week"])

        for pos in POSITIONS:
            allowed_col = f"fantasy_points_allowed_{pos.lower()}"
            expected_col = f"fantasy_points_produced_{pos.lower()}_s2d_lag1"
            if allowed_col not in merged.columns or expected_col not in merged.columns:
                continue
            residual_col = f"_residual_{pos.lower()}"
            merged[residual_col] = merged[allowed_col] - merged[expected_col]
            dvoa_col = f"opp_fpts_allowed_dvoa_adjusted_{pos.lower()}_lag1"
            merged[dvoa_col] = (
                merged.groupby(["team", "season"])[residual_col]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            )

        dvoa_cols = [f"opp_fpts_allowed_dvoa_adjusted_{p.lower()}_lag1" for p in POSITIONS]
        dvoa_cols = [c for c in dvoa_cols if c in merged.columns]
        merge_right = merged[["team", "season", "week"] + dvoa_cols].rename(
            columns={"team": "opponent"}
        )

        before_cols = set(df.columns)
        df = df.merge(merge_right, on=["opponent", "season", "week"], how="left")

        df["opp_fpts_allowed_dvoa_adjusted_lag1"] = np.nan
        for pos in POSITIONS:
            src_col = f"opp_fpts_allowed_dvoa_adjusted_{pos.lower()}_lag1"
            if src_col not in df.columns:
                continue
            mask = df["position"] == pos
            df.loc[mask, "opp_fpts_allowed_dvoa_adjusted_lag1"] = pd.to_numeric(
                df.loc[mask, src_col], errors="coerce"
            )

        n_total = len(df)
        n_missing = int(df["opp_fpts_allowed_dvoa_adjusted_lag1"].isna().sum())
        if n_total and n_missing / n_total > 0.10:
            logger.warning(
                "opp_fpts_allowed_dvoa_adjusted_lag1 defaulted on %.1f%% of "
                "rows (>10%% threshold). team_offense_stats/team_defense_stats "
                "may be missing weeks for the seasons in scope (%s).",
                n_missing / n_total * 100, seasons,
            )

        df["opp_fpts_allowed_dvoa_adjusted_lag1"] = df["opp_fpts_allowed_dvoa_adjusted_lag1"].fillna(0.0)

        drop_cols = [c for c in df.columns if c not in before_cols and c != "opp_fpts_allowed_dvoa_adjusted_lag1"]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def _add_opp_fpts_allowed_from_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Serving-path variant of the opp_fpts_allowed feature.

        The inline version in _create_opponent_features() (used for
        training) relies on fantasy_points_allowed_{pos} columns already
        joined onto df by get_all_players_for_training()'s bulk SQL join —
        efficient for training, but those columns reflect whatever
        `opponent` the row had AT JOIN TIME. refresh_matchup_features()
        overwrites `opponent` afterward (for an upcoming, not-yet-played
        game), so the inline version would silently keep serving the
        player's last-played opponent's defensive stats. This queries
        team_defense_stats directly by (opponent, season, week - 1) —
        same leakage-safe "prior week" semantic as the SQL join it
        mirrors — so it's correct to call after an opponent overwrite.
        """
        required = {"opponent", "season", "week", "position"}
        if not required.issubset(df.columns):
            df["opp_fpts_allowed"] = np.nan
            return df

        import logging
        logger = logging.getLogger(__name__)

        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique())
            if not seasons:
                raise ValueError("No seasons")
            season_list = ",".join(str(int(s)) for s in seasons)
            with db._get_connection() as conn:
                tds = pd.read_sql_query(
                    "SELECT team, season, week, "
                    "fantasy_points_allowed_qb, fantasy_points_allowed_rb, "
                    "fantasy_points_allowed_wr, fantasy_points_allowed_te "
                    f"FROM team_defense_stats WHERE season IN ({season_list})",
                    conn,
                )
        except Exception as e:
            logger.warning(
                "team_defense_stats load failed (%s: %s); opp_fpts_allowed "
                "will default to NaN in refresh_matchup_features(). Run "
                "DatabaseManager.ensure_team_defense_stats() to populate.",
                type(e).__name__, e,
            )
            df["opp_fpts_allowed"] = np.nan
            return df

        if tds.empty:
            df["opp_fpts_allowed"] = np.nan
            return df

        # Defense D's week-(W-1) stats apply to a game in week W — shift the
        # defense frame's week forward by 1 so it lands on the right join key.
        merge_right = tds.rename(columns={"team": "opponent"}).copy()
        merge_right["week"] = merge_right["week"] + 1
        fp_cols = [f"fantasy_points_allowed_{p.lower()}" for p in POSITIONS]
        fp_cols = [c for c in fp_cols if c in merge_right.columns]

        # df may already carry fantasy_points_allowed_{pos} columns from the
        # original (now-stale, pre-opponent-overwrite) bulk SQL join in
        # get_all_players_for_training() — drop them first so the merge
        # below actually lands on the unsuffixed column name instead of
        # silently becoming fantasy_points_allowed_{pos}_x/_y.
        df = df.drop(columns=[c for c in fp_cols if c in df.columns])

        before_cols = set(df.columns)
        df = df.merge(
            merge_right[["opponent", "season", "week"] + fp_cols],
            on=["opponent", "season", "week"], how="left",
        )

        df["opp_fpts_allowed"] = np.nan
        for pos in POSITIONS:
            col = f"fantasy_points_allowed_{pos.lower()}"
            if col not in df.columns:
                continue
            mask = df["position"] == pos
            df.loc[mask, "opp_fpts_allowed"] = pd.to_numeric(df.loc[mask, col], errors="coerce")

        drop_cols = [c for c in df.columns if c not in before_cols and c != "opp_fpts_allowed"]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def _add_team_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team-level features for both player's team (TeamA) and opponent (TeamB).
        
        This includes offensive/defensive metrics for game context prediction.
        """
        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            
            # Get all team stats
            all_team_stats = db.get_team_stats()
            if all_team_stats.empty:
                return df

            # Calculate rolling team averages (last 3 games)
            team_metrics = ['points_scored', 'points_allowed', 'total_yards',
                           'passing_yards', 'rushing_yards', 'turnovers',
                           'pass_attempts', 'rush_attempts', 'redzone_scores',
                           'neutral_pass_plays', 'neutral_run_plays',
                           'neutral_pass_rate', 'neutral_pass_rate_lg', 'neutral_pass_rate_oe',
                           'drive_count', 'drive_success_rate', 'avg_drive_epa',
                           'points_per_drive', 'pace_sec_per_play']

            # This function can run twice on the same df: once from
            # create_features(), again from refresh_matchup_features() to
            # recompute for the actual upcoming opponent (predict.py).
            # Dropping any columns from the first pass before re-merging
            # avoids pandas' silent _x/_y suffixing on the second pass,
            # which otherwise both leaves stale opponent data in place
            # and raises a caught-but-fatal KeyError on
            # 'offensive_momentum_score' below (GAPS.md, 2026-08-06).
            stale_cols = (
                [f'team_a_{m}' for m in team_metrics] + [f'team_b_{m}' for m in team_metrics] +
                ['matchup_scoring_edge', 'matchup_yards_diff', 'matchup_pass_diff',
                 'matchup_rush_diff', 'expected_game_total', 'expected_point_diff',
                 'team_a_plays_per_game', 'team_b_plays_per_game', 'team_a_pass_rate',
                 'offensive_momentum_score']
            )
            df = df.drop(columns=[c for c in stale_cols if c in df.columns])

            # These lookup tables depend only on the full team_stats table
            # (not on df / the current backtest window), so they're built
            # once and cached across the backtester's weekly calls.
            team_a_avgs, team_b_avgs, inseason_df, mom = _get_team_matchup_lookups(
                all_team_stats, team_metrics
            )

            # Merge TeamA prior-season stats
            if 'team' in df.columns and 'season' in df.columns:
                merge_cols = ['team', 'season']
                df = df.merge(
                    team_a_avgs,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_team_a')
                )

            # Merge TeamB prior-season stats (opponent)
            if 'opponent' in df.columns and 'season' in df.columns:
                merge_cols = ['opponent', 'season']
                df = df.merge(
                    team_b_avgs,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_team_b')
                )

            # In-season rolling team averages (4-week causal window) to supplement
            # stale prior-season averages.  shift(1) excludes current week.
            # Blend: 60% in-season / 40% prior-season when week >= 5, else prior-season only.
            if (inseason_df is not None and 'team' in df.columns and 'season' in df.columns
                    and 'week' in df.columns and 'week' in all_team_stats.columns):
                avail_metrics = [m for m in team_metrics if m in all_team_stats.columns]
                if avail_metrics:
                    df = df.merge(inseason_df, on=['team', 'season', 'week'], how='left')
                    # Blend in-season rolling with prior-season for TeamA columns
                    for metric in avail_metrics:
                        ta_col = f'team_a_{metric}'
                        is_col = f'_inseason_{metric}'
                        if ta_col in df.columns and is_col in df.columns:
                            has_inseason = df[is_col].notna() & (df['week'] >= 5)
                            df.loc[has_inseason, ta_col] = (
                                0.6 * df.loc[has_inseason, is_col] +
                                0.4 * df.loc[has_inseason, ta_col]
                            )
                    # Drop temporary in-season columns
                    df = df.drop(columns=[f'_inseason_{m}' for m in avail_metrics], errors='ignore')

            # Create matchup differential features
            matchup_pairs = [
                ('team_a_points_scored', 'team_b_points_allowed', 'matchup_scoring_edge'),
                ('team_a_total_yards', 'team_b_total_yards', 'matchup_yards_diff'),
                ('team_a_passing_yards', 'team_b_passing_yards', 'matchup_pass_diff'),
                ('team_a_rushing_yards', 'team_b_rushing_yards', 'matchup_rush_diff'),
            ]
            
            for col_a, col_b, new_col in matchup_pairs:
                if col_a in df.columns and col_b in df.columns:
                    df[new_col] = df[col_a] - df[col_b]
            
            # Game script prediction features
            if 'team_a_points_scored' in df.columns and 'team_b_points_scored' in df.columns:
                # Expected game total
                df['expected_game_total'] = df['team_a_points_scored'] + df['team_b_points_scored']
                # Expected point differential (positive = player's team favored)
                df['expected_point_diff'] = df['team_a_points_scored'] - df['team_b_points_allowed']
            
            # Pace features (plays per game)
            if 'team_a_pass_attempts' in df.columns and 'team_a_rush_attempts' in df.columns:
                df['team_a_plays_per_game'] = df['team_a_pass_attempts'] + df['team_a_rush_attempts']
            if 'team_b_pass_attempts' in df.columns and 'team_b_rush_attempts' in df.columns:
                df['team_b_plays_per_game'] = df['team_b_pass_attempts'] + df['team_b_rush_attempts']
            
            # Pass/rush tendency
            if 'team_a_pass_attempts' in df.columns and 'team_a_rush_attempts' in df.columns:
                total = df['team_a_pass_attempts'] + df['team_a_rush_attempts']
                df['team_a_pass_rate'] = df['team_a_pass_attempts'] / total.replace(0, 1)

            # Offensive momentum score per requirements III.B: weighted combination of
            # team offensive EPA trend (proxied by points_scored), pass/rush success rate
            # trends (passing_yards, rushing_yards), and scoring efficiency (turnovers).
            # Time-weighted: recent 4 weeks = 60%, weeks 5-8 = 30%, weeks 9+ = 10%.
            # (Computed once and cached in _get_team_matchup_lookups above.)
            if mom is not None and 'team' in df.columns and 'season' in df.columns and 'week' in df.columns:
                df = df.merge(mom, on=['team', 'season', 'week'], how='left')
                df['offensive_momentum_score'] = df['offensive_momentum_score'].fillna(22.0)
        except Exception as e:
            # Team features are optional - don't fail if unavailable, but log
            # so silent degradation is visible in pipeline output.
            print(f"  WARNING: Team matchup features unavailable ({type(e).__name__}: {e})")

        # --- Divisional game and prime-time game indicators (per requirements III.A) ---
        # Populate from nfl-data-py schedule data when available, otherwise keep defaults.
        if 'is_divisional' not in df.columns or 'is_primetime' not in df.columns:
            try:
                import nfl_data_py as nfl
                seasons = sorted(df["season"].unique()) if "season" in df.columns else []
                if seasons:
                    sched = nfl.import_schedules([int(s) for s in seasons])
                    if not sched.empty:
                        # Build a lookup: (season, week, team) -> (div_game, primetime)
                        # nfl-data-py schedules have div_game (bool) and gametime columns.
                        sched_rows = []
                        for _, row in sched.iterrows():
                            s = int(row.get("season", 0))
                            w = int(row.get("week", 0))
                            home = row.get("home_team", "")
                            away = row.get("away_team", "")
                            # Divisional: div_game column if present, else 0
                            is_div = int(row.get("div_game", 0)) if pd.notna(row.get("div_game")) else 0
                            # Prime-time: gametime 20:00+ (8pm+ starts), or game_type indicators
                            gametime = str(row.get("gametime", ""))
                            is_prime = 0
                            if gametime and gametime != "nan":
                                try:
                                    hour = int(gametime.split(":")[0])
                                    is_prime = 1 if hour >= 20 else 0
                                except (ValueError, IndexError):
                                    pass
                            # Also treat Thursday/Monday/Saturday night as primetime
                            gameday = str(row.get("gameday", ""))
                            weekday = str(row.get("weekday", row.get("day_of_week", "")))
                            if weekday.lower() in ("thursday", "monday"):
                                is_prime = 1
                            for team in [home, away]:
                                if team:
                                    sched_rows.append({
                                        "season": s, "week": w, "team": team,
                                        "_is_divisional": is_div, "_is_primetime": is_prime,
                                    })
                        if sched_rows:
                            sched_lookup = pd.DataFrame(sched_rows).drop_duplicates(
                                subset=["season", "week", "team"]
                            )
                            if "team" in df.columns and "season" in df.columns and "week" in df.columns:
                                df = df.merge(
                                    sched_lookup, on=["season", "week", "team"],
                                    how="left", suffixes=("", "_sched")
                                )
                                if "_is_divisional" in df.columns:
                                    if "is_divisional" not in df.columns:
                                        df["is_divisional"] = df["_is_divisional"].fillna(0).astype(int)
                                    else:
                                        df["is_divisional"] = df["is_divisional"].fillna(df["_is_divisional"]).fillna(0).astype(int)
                                    df = df.drop(columns=["_is_divisional"])
                                if "_is_primetime" in df.columns:
                                    if "is_primetime" not in df.columns:
                                        df["is_primetime"] = df["_is_primetime"].fillna(0).astype(int)
                                    else:
                                        df["is_primetime"] = df["is_primetime"].fillna(df["_is_primetime"]).fillna(0).astype(int)
                                    df = df.drop(columns=["_is_primetime"])
            except Exception as e:
                # Surface the silent-fallback failure per the 2026-04-22
                # re-council Step 2 audit.  Previously `except Exception:
                # pass` — same trap as the Vegas silent fallback fixed in
                # Phase 1 (docs/PHASE_1_VEGAS_FINDINGS.md).  These columns
                # aren't in CAUSAL_FEATURES today, but any future addition
                # would silently collapse to 0 without a warning here.
                import logging
                logging.getLogger(__name__).warning(
                    "Schedule fetch for is_divisional / is_primetime failed "
                    "(%s: %s); both columns default to 0 for every row that "
                    "didn't have a pre-merged value.  Run "
                    "scripts/backfill_vegas_lines.py (which uses the same "
                    "nflverse games.csv) or verify nfl_data_py reachability.",
                    type(e).__name__, e,
                )

        if 'offensive_momentum_score' not in df.columns:
            df['offensive_momentum_score'] = 22.0
        if 'is_divisional' not in df.columns:
            df['is_divisional'] = 0
        if 'is_primetime' not in df.columns:
            df['is_primetime'] = 0
        
        # Fill missing team features with league averages
        team_feature_defaults = {
            'team_a_points_scored': 22.0, 'team_a_points_allowed': 22.0,
            'team_a_total_yards': 340.0, 'team_a_passing_yards': 220.0,
            'team_a_rushing_yards': 120.0, 'team_a_turnovers': 1.5,
            'team_b_points_scored': 22.0, 'team_b_points_allowed': 22.0,
            'team_b_total_yards': 340.0, 'team_b_passing_yards': 220.0,
            'team_b_rushing_yards': 120.0, 'team_b_turnovers': 1.5,
            'matchup_scoring_edge': 0.0, 'matchup_yards_diff': 0.0,
            'matchup_pass_diff': 0.0, 'matchup_rush_diff': 0.0,
            'expected_game_total': 44.0, 'expected_point_diff': 0.0,
            'team_a_plays_per_game': 65.0, 'team_b_plays_per_game': 65.0,
            'team_a_pass_rate': 0.55,
            'team_a_neutral_pass_rate': 0.55, 'team_a_neutral_pass_rate_oe': 0.0,
            'team_b_neutral_pass_rate': 0.55, 'team_b_neutral_pass_rate_oe': 0.0,
            'team_a_drive_success_rate': 0.50, 'team_b_drive_success_rate': 0.50,
            'team_a_points_per_drive': 1.8, 'team_b_points_per_drive': 1.8,
            'team_a_avg_drive_epa': 0.0, 'team_b_avg_drive_epa': 0.0,
            'team_a_pace_sec_per_play': 28.0, 'team_b_pace_sec_per_play': 28.0,
        }
        
        for col, default_val in team_feature_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)
        
        return df
    
    def _create_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create situational/contextual features."""
        # Team offensive context
        if "team_plays" in df.columns:
            df["plays_share"] = safe_divide(
                df["opportunities"], df["team_plays"]
            )
        
        if "team_pass_attempts" in df.columns:
            df["team_pass_rate"] = safe_divide(
                df["team_pass_attempts"], 
                df.get("team_plays", df["team_pass_attempts"] + df.get("team_rush_attempts", 0))
            )

        # Situation-specific usage rates (advanced PBP)
        neutral_targets = df.get("neutral_targets", pd.Series(0, index=df.index))
        team_neutral_pass_plays = df.get("team_neutral_pass_plays", pd.Series(0, index=df.index))
        total_touches = df.get("rushing_attempts", pd.Series(0, index=df.index)) + df.get(
            "targets", pd.Series(0, index=df.index)
        )

        # Bye/rest timing — computed before the batch so interactions can reference them
        post_bye = df.groupby("player_id")["week"].transform(
            lambda x: (x.diff() > 1).astype(int)
        )
        days_since_last_game = df.groupby("player_id")["week"].transform(
            lambda x: (x.diff().fillna(1).clip(lower=1) * 7).astype(float)
        ).fillna(7.0)
        short_week = (days_since_last_game <= 4.0).astype(int)

        situational_cols: dict = {
            "neutral_target_share": safe_divide(neutral_targets, team_neutral_pass_plays),
            "third_down_target_rate": safe_divide(
                df.get("third_down_targets", pd.Series(0, index=df.index)),
                df.get("targets", pd.Series(0, index=df.index)),
            ),
            "short_yardage_touch_rate": safe_divide(
                df.get("short_yardage_rushes", pd.Series(0, index=df.index)),
                df.get("rushing_attempts", pd.Series(0, index=df.index)),
            ),
            "two_minute_target_rate": safe_divide(
                df.get("two_minute_targets", pd.Series(0, index=df.index)),
                df.get("targets", pd.Series(0, index=df.index)),
            ),
            "high_leverage_touch_rate": safe_divide(
                df.get("high_leverage_touches", pd.Series(0, index=df.index)),
                total_touches,
            ),
            "post_bye": post_bye,
            "days_since_last_game": days_since_last_game,
            "short_week": short_week,
            # Rest advantage: positive = more rest than typical 7 days, negative = short week
            "rest_advantage": (days_since_last_game - 7.0).clip(-4.0, 7.0),
        }

        if "fantasy_points_roll3_mean" in df.columns:
            fp_roll3 = df["fantasy_points_roll3_mean"].fillna(0)
            situational_cols["post_bye_x_recent_form"] = post_bye * fp_roll3
            situational_cols["short_week_x_recent_form"] = short_week * fp_roll3

        df = df.assign(**situational_cols)

        # Add schedule-based features if available
        df = self._add_schedule_features(df)
        
        # Add game script / garbage time adjustment
        df = self._add_game_script_adjustment(df)
        
        return df

    def _create_team_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features to proactively adjust for team changes and scheme fit.

        Captures:
        - team_changed: whether player changed teams since last appearance
        - weeks_on_team: number of weeks since joining current team (resets on change or season boundary)
        - team_change_pass_rate_delta: change in team pass rate vs previous team context
        - scheme_fit_score: positional fit to team pass rate (higher = better fit)
        - scheme_mismatch_on_change: mismatch severity when team_changed
        - team_change_recent_util: recent utilization prior to change (lagged)
        """
        if df.empty or "player_id" not in df.columns or "team" not in df.columns:
            return df

        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        grp = df.groupby("player_id", sort=False)
        prev_team = grp["team"].shift(1)
        prev_season = grp["season"].shift(1)

        season_changed = (df["season"] != prev_season) & prev_season.notna()
        team_changed = (df["team"] != prev_team) & prev_team.notna()

        # Weeks on current team (reset on team change or season boundary)
        change_flag = (team_changed | season_changed).astype(int)
        # Temporary helper columns needed for cumcount groupby — assigned individually
        # because they are consumed immediately and dropped before return.
        df["_team_change_flag"] = change_flag
        df["_team_stint_id"] = grp["_team_change_flag"].cumsum()
        weeks_on_team = df.groupby(["player_id", "_team_stint_id"]).cumcount() + 1

        # Team pass rate delta (use prior row for same player as a proxy)
        team_changed_int = team_changed.astype(int)
        if "team_a_pass_rate" in df.columns:
            prev_pass = grp["team_a_pass_rate"].shift(1)
            delta = (df["team_a_pass_rate"] - prev_pass).fillna(0.0)
            team_pass_rate_delta = delta
            team_change_pass_rate_delta = (delta * team_changed_int).fillna(0.0)
        else:
            team_pass_rate_delta = 0.0
            team_change_pass_rate_delta = 0.0

        # Scheme fit: position-specific preferred pass rate
        if "team_a_pass_rate" in df.columns and "position" in df.columns:
            pass_pref = df["position"].map({
                "QB": 0.60,
                "WR": 0.60,
                "TE": 0.58,
                "RB": 0.45,
            }).fillna(0.52)
            mismatch = (df["team_a_pass_rate"] - pass_pref).abs()
            # Normalize mismatch to [0, 1] with 0.6 as a wide max band
            mismatch_norm = (mismatch / 0.6).clip(0.0, 1.0)
            scheme_fit_score = (1.0 - mismatch_norm).fillna(0.5)
            scheme_mismatch = mismatch_norm.fillna(0.5)
            scheme_mismatch_on_change = (scheme_mismatch * team_changed_int).fillna(0.0)
        else:
            scheme_fit_score = 0.5
            scheme_mismatch = 0.5
            scheme_mismatch_on_change = 0.0

        # Recent utilization prior to change (lagged utilization only)
        util_col = None
        for cand in ["utilization_score_roll4_mean", "utilization_score_lag1", "utilization_score_roll3_mean"]:
            if cand in df.columns:
                util_col = cand
                break
        if util_col:
            team_change_recent_util = np.where(team_changed_int == 1,
                                               df[util_col].fillna(0.0), 0.0)
        else:
            team_change_recent_util = 0.0

        df = df.drop(columns=["_team_change_flag", "_team_stint_id"], errors="ignore")

        dest_tgt_pg, dest_carry_pg = self._add_dest_team_pos_profiles(df, team_changed_int)

        new_cols = {
            "team_changed": team_changed_int,
            "new_season": season_changed.astype(int),
            "weeks_on_team": weeks_on_team,
            "team_pass_rate_delta": team_pass_rate_delta,
            "team_change_pass_rate_delta": team_change_pass_rate_delta,
            "scheme_fit_score": scheme_fit_score,
            "scheme_mismatch": scheme_mismatch,
            "scheme_mismatch_on_change": scheme_mismatch_on_change,
            "team_change_recent_util": team_change_recent_util,
            "dest_team_pos_tgt_pg": dest_tgt_pg,
            "dest_team_pos_carry_pg": dest_carry_pg,
        }
        df = df.assign(**new_cols)
        return df

    def _add_dest_team_pos_profiles(
        self, df: pd.DataFrame, team_changed_int: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        For team-changers, return the destination team's historical positional
        targets/game and carries/game (3-season rolling, leakage-safe via shift(1)).
        Both series are zero for non-changers.
        Computed entirely from df — no additional DB query needed.
        """
        if "position" not in df.columns or "season" not in df.columns:
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)

        # Season-level team×position totals
        agg = (
            df.groupby(["team", "position", "season"])[["targets", "rushing_attempts", "week"]]
            .agg({"targets": "sum", "rushing_attempts": "sum", "week": "nunique"})
            .reset_index()
            .rename(columns={"week": "gp"})
        )
        agg["tgt_pg"] = agg["targets"] / agg["gp"].clip(lower=1)
        agg["carry_pg"] = agg["rushing_attempts"] / agg["gp"].clip(lower=1)

        # 3-season rolling mean with shift(1) — leakage-safe
        agg = agg.sort_values(["team", "position", "season"])
        for col in ["tgt_pg", "carry_pg"]:
            agg[f"hist_{col}"] = (
                agg.groupby(["team", "position"])[col]
                .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            )

        # Merge historical profile onto each row using the player's current team/position/season
        merged = df[["team", "position", "season"]].merge(
            agg[["team", "position", "season", "hist_tgt_pg", "hist_carry_pg"]],
            on=["team", "position", "season"],
            how="left",
        )

        # Zero out for non-changers
        dest_tgt_pg = (
            pd.Series(merged["hist_tgt_pg"].fillna(0.0).values, index=df.index)
            * team_changed_int.values
        )
        dest_carry_pg = (
            pd.Series(merged["hist_carry_pg"].fillna(0.0).values, index=df.index)
            * team_changed_int.values
        )

        return dest_tgt_pg, dest_carry_pg

    def _add_game_script_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add game script and garbage time adjustment features.
        
        Garbage time stats (when win probability is very low or very high) are
        less meaningful for projecting future performance. A backup RB getting
        15 carries in garbage time doesn't indicate elite usage going forward.
        
        We use score differential and game clock to estimate game script context
        and create adjustment factors for utilization metrics.
        
        Features created:
        - garbage_time_pct: Estimated % of stats accumulated in garbage time
        - game_script_factor: Adjustment factor (0.5-1.0) for utilization metrics
        - competitive_snaps_pct: % of snaps in competitive game situations
        """
        # Check if we have the necessary columns
        has_score_diff = 'score_differential' in df.columns or 'point_differential' in df.columns
        has_win_prob = 'win_probability' in df.columns or 'wp' in df.columns
        
        if has_score_diff:
            score_diff_col = 'score_differential' if 'score_differential' in df.columns else 'point_differential'
            df = self._calculate_garbage_time_from_score(df, score_diff_col)
        elif has_win_prob:
            wp_col = 'win_probability' if 'win_probability' in df.columns else 'wp'
            df = self._calculate_garbage_time_from_wp(df, wp_col)
        else:
            # Estimate from final score if available
            df = self._estimate_garbage_time_from_result(df)
        
        # Create game script adjustment factor and competitive snaps — batch together
        script_cols: dict = {}
        if 'garbage_time_pct' in df.columns:
            # Discount utilization by garbage time percentage
            # e.g., if 30% of stats came in garbage time, adjustment = 0.85
            script_cols['game_script_factor'] = (1.0 - 0.5 * df['garbage_time_pct']).clip(0.5, 1.0)
        else:
            script_cols['game_script_factor'] = 1.0

        if 'competitive_snaps' in df.columns and 'snap_count' in df.columns:
            script_cols['competitive_snaps_pct'] = safe_divide(df['competitive_snaps'], df['snap_count'])
        else:
            script_cols['competitive_snaps_pct'] = 1.0 - df.get('garbage_time_pct', 0)

        df = df.assign(**script_cols)
        return df
    
    def _calculate_garbage_time_from_score(
        self, 
        df: pd.DataFrame, 
        score_diff_col: str
    ) -> pd.DataFrame:
        """
        Calculate garbage time percentage from score differential.
        
        Garbage time is defined as:
        - Leading by 17+ points in 4th quarter
        - Trailing by 17+ points in 4th quarter
        - Leading by 24+ points in 3rd quarter
        - Trailing by 24+ points in 3rd quarter
        """
        # If we have quarter-level data
        if 'quarter' in df.columns:
            garbage_conditions = (
                # 4th quarter blowouts
                ((df['quarter'] == 4) & (df[score_diff_col].abs() >= 17)) |
                # 3rd quarter blowouts
                ((df['quarter'] == 3) & (df[score_diff_col].abs() >= 24))
            )
            is_garbage_time = garbage_conditions.astype(int)
            # groupby.transform needs the column in df, so assign both at once after computing
            df = df.assign(is_garbage_time=is_garbage_time)
            df = df.assign(
                garbage_time_pct=df.groupby(['player_id', 'season', 'week'])['is_garbage_time'].transform('mean')
            )
        else:
            # Estimate from final score differential
            # Games with 17+ point differential likely had significant garbage time
            df = df.assign(garbage_time_pct=np.where(
                df[score_diff_col].abs() >= 24, 0.30,
                np.where(
                    df[score_diff_col].abs() >= 17, 0.20,
                    np.where(
                        df[score_diff_col].abs() >= 10, 0.10,
                        0.0
                    )
                )
            ))

        return df
    
    def _calculate_garbage_time_from_wp(
        self, 
        df: pd.DataFrame, 
        wp_col: str
    ) -> pd.DataFrame:
        """
        Calculate garbage time percentage from win probability.
        
        Garbage time is when win probability is < 10% or > 90%.
        This is the most accurate method when play-by-play data is available.
        """
        # Binary garbage time indicator — assign before the groupby that reads it
        is_garbage_time = ((df[wp_col] < 0.10) | (df[wp_col] > 0.90)).astype(int)
        df = df.assign(is_garbage_time=is_garbage_time)
        df = df.assign(
            garbage_time_pct=df.groupby(['player_id', 'season', 'week'])['is_garbage_time'].transform('mean')
        )
        return df
    
    def _estimate_garbage_time_from_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate garbage time from final game result when detailed data unavailable.
        
        Uses final score differential to estimate how much garbage time likely occurred.
        This is a rough heuristic but better than ignoring game script entirely.
        """
        # Calculate or get final score differential
        if 'team_score' in df.columns and 'opponent_score' in df.columns:
            final_margin = df['team_score'] - df['opponent_score']
        elif 'margin' in df.columns:
            final_margin = df['margin']
        else:
            # No score data available
            return df.assign(garbage_time_pct=0.0)

        # Estimate garbage time based on final margin
        # Larger margins = more likely there was garbage time
        garbage_time_pct = np.select(
            [
                final_margin.abs() >= 28,  # Blowout: ~35% garbage time
                final_margin.abs() >= 21,  # Big win: ~25% garbage time
                final_margin.abs() >= 14,  # Comfortable: ~15% garbage time
                final_margin.abs() >= 7,   # Close-ish: ~5% garbage time
            ],
            [0.35, 0.25, 0.15, 0.05],
            default=0.0
        )
        df = df.assign(final_margin=final_margin, garbage_time_pct=garbage_time_pct)
        return df
    
    def create_adjusted_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game-script-adjusted utilization metrics.
        
        Multiplies raw utilization metrics by game_script_factor to discount
        stats accumulated in garbage time.
        
        Args:
            df: DataFrame with utilization metrics and game_script_factor
            
        Returns:
            DataFrame with adjusted utilization columns
        """
        util_cols = [col for col in df.columns if 'utilization' in col.lower() or 'share' in col.lower()]
        
        if 'game_script_factor' not in df.columns:
            return df
        
        for col in util_cols:
            # Create adjusted version
            adj_col = f"{col}_adj"
            df[adj_col] = df[col] * df['game_script_factor']
        
        return df
    
    def _add_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add schedule and strength of schedule features."""
        try:
            from src.utils.database import DatabaseManager
            from src.scrapers.schedule_scraper import StrengthOfScheduleCalculator
            
            db = DatabaseManager()
            
            # Get unique seasons in data
            seasons = df['season'].unique() if 'season' in df.columns else []
            
            for season in seasons:
                schedule = db.get_schedule(season=int(season))
                if schedule.empty:
                    continue
                
                # Get team stats for SOS calculation
                team_stats = db.get_team_stats(season=int(season) - 1)  # Prior year
                
                sos_calc = StrengthOfScheduleCalculator(team_stats)
                sos_calc.calculate_team_rankings(int(season))
                
                # Calculate SOS for each team
                all_sos = sos_calc.get_all_teams_sos(schedule)
                sos_map = dict(zip(all_sos['team'], all_sos['sos_rating']))
                
                # Add team SOS to player data
                season_mask = df['season'] == season
                if 'team' in df.columns:
                    df.loc[season_mask, 'team_sos'] = df.loc[season_mask, 'team'].map(sos_map)
                
                # Add weekly matchup difficulty
                for team in df.loc[season_mask, 'team'].unique():
                    if pd.isna(team):
                        continue
                    matchups = sos_calc.calculate_weekly_matchup_difficulty(schedule, team)
                    if not matchups.empty:
                        matchup_map = dict(zip(matchups['week'], matchups['matchup_difficulty']))
                        team_mask = season_mask & (df['team'] == team)
                        df.loc[team_mask, 'matchup_difficulty'] = df.loc[team_mask, 'week'].map(matchup_map)
                        
                        # Add opponent rating
                        opp_map = dict(zip(matchups['week'], matchups['opponent_rating']))
                        df.loc[team_mask, 'opponent_rating'] = df.loc[team_mask, 'week'].map(opp_map)
        except Exception as e:
            # Schedule features are optional - don't fail if unavailable
            print(f"  WARNING: Schedule features unavailable ({type(e).__name__}: {e})")
        
        # Fill missing schedule features with neutral values — batch together
        sos_defaults: dict = {
            col: df[col].fillna(50.0) if col in df.columns else 50.0
            for col in ('team_sos', 'matchup_difficulty', 'opponent_rating')
        }
        df = df.assign(**sos_defaults)
        return df
    
    def refresh_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute schedule- and opponent-dependent features on a dataframe that
        already has team, season, week, opponent, home_away set (e.g. prediction
        input for the upcoming week). Use after overwriting those columns so the
        model sees the correct matchup.
        """
        if df.empty or 'team' not in df.columns or 'season' not in df.columns:
            return df
        df = self._add_schedule_features(df)
        df = self._add_team_matchup_features(df)
        # opp_fpts_allowed / opp_fpts_allowed_s2d_lag1 are also
        # opponent-dependent and must be recomputed here — otherwise they
        # silently keep reflecting whichever opponent the row had before
        # `opponent` was overwritten for the upcoming game (a real bug
        # found and fixed 2026-08-04: see GAPS.md).
        if 'opponent' in df.columns and 'position' in df.columns:
            df = self._add_opp_fpts_allowed_from_db(df)
            df = self._add_opp_fpts_allowed_s2d_lag1(df)
            df = self._add_opp_fpts_allowed_dvoa_adjusted_lag1(df)
        # Ensure neutral defaults for any matchup columns the model might expect
        for col, default in [
            ('team_sos', 50.0), ('matchup_difficulty', 50.0), ('opponent_rating', 50.0),
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(default)
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key metrics."""
        # Utilization x Efficiency interactions
        # Use LAGGED utilization (not current-week) to avoid leakage.
        # The raw utilization_score is excluded from features, so derived
        # features from it would carry the same leakage risk.
        util_lagged = None
        if "utilization_score_lag_1" in df.columns:
            util_lagged = df["utilization_score_lag_1"]
        elif "utilization_score_roll3_mean" in df.columns:
            util_lagged = df["utilization_score_roll3_mean"]

        interaction_cols: dict = {}

        if util_lagged is not None:
            if "yards_per_carry" in df.columns:
                interaction_cols["util_x_ypc"] = util_lagged * df["yards_per_carry"]
            if "yards_per_target" in df.columns:
                interaction_cols["util_x_ypt"] = util_lagged * df["yards_per_target"]

        # Volume x Efficiency
        interaction_cols["touches_x_ypc"] = df["total_touches"] * df.get("yards_per_carry", 0)

        # Opportunity x TD rate
        if "total_tds" in df.columns and "opportunities" in df.columns:
            td_rate = safe_divide(df["total_tds"], df["opportunities"])
            interaction_cols["opp_x_td_rate"] = df["opportunities"] * td_rate

        if interaction_cols:
            df = df.assign(**interaction_cols)
        
        # --- Matchup Quality Indicator (requirements III.B) ---
        # Composite score combining opponent defense weakness, game script favorability,
        # and pace environment. Higher = better matchup for fantasy production.
        # Use expanding mean/std to avoid future data leakage in z-score components.
        mqi_components = []

        # Helper: compute causal z-scores using expanding mean/std (no future leakage).
        # shift(1) ensures the current row's value is excluded from its own z-score.
        def _causal_zscore(series: pd.Series) -> pd.Series:
            if "season" in df.columns and "week" in df.columns:
                sorted_vals = series.reindex(df.sort_values(["season", "week"]).index)
                shifted = sorted_vals.shift(1)
                exp_mean = shifted.expanding(min_periods=1).mean()
                exp_std = shifted.expanding(min_periods=2).std().clip(lower=0.1)
                z = ((sorted_vals - exp_mean) / exp_std).reindex(df.index)
            else:
                shifted = series.shift(1)
                exp_mean = shifted.expanding(min_periods=1).mean()
                exp_std = shifted.expanding(min_periods=2).std().clip(lower=0.1)
                z = (series - exp_mean) / exp_std
            return z

        # Component 1: Opponent points allowed (position-specific when available)
        if "opp_fpts_allowed" in df.columns:
            opp_z = _causal_zscore(df["opp_fpts_allowed"])
            mqi_components.append(opp_z.fillna(0) * 0.35)
        elif "matchup_difficulty" in df.columns:
            # matchup_difficulty: higher = harder opponent, so invert
            md_z = -(df["matchup_difficulty"] - 50.0) / 25.0
            mqi_components.append(md_z.fillna(0) * 0.35)

        # Component 2: Game script favorability (implied team total or expected point diff)
        if "implied_team_total" in df.columns:
            itt_z = _causal_zscore(df["implied_team_total"])
            mqi_components.append(itt_z.fillna(0) * 0.30)
        elif "expected_point_diff" in df.columns:
            epd_z = _causal_zscore(df["expected_point_diff"])
            mqi_components.append(epd_z.fillna(0) * 0.30)

        # Component 3: Pace environment (team plays per game)
        if "team_a_plays_per_game" in df.columns and "team_b_plays_per_game" in df.columns:
            combined_pace = df["team_a_plays_per_game"] + df["team_b_plays_per_game"]
            pace_z = _causal_zscore(combined_pace)
            mqi_components.append(pace_z.fillna(0) * 0.20)
        
        # Component 4: Home field advantage
        if "is_home" in df.columns:
            mqi_components.append(df["is_home"].fillna(0) * 0.15)
        
        if mqi_components:
            mqi_raw = sum(mqi_components)
            # Normalize to 0-100 scale using expanding min/max with shift(1)
            # to exclude current row from its own normalization bounds
            if "season" in df.columns and "week" in df.columns:
                # Sort by time FIRST so expanding windows are causal
                sort_order = df.sort_values(["season", "week"]).index
                mqi_sorted = mqi_raw.reindex(sort_order)
                shifted = mqi_sorted.shift(1)
                expanding_min = shifted.expanding(min_periods=1).min()
                expanding_max = shifted.expanding(min_periods=1).max()
                denom = (expanding_max - expanding_min).replace(0, np.nan)
                mqi_final = (((mqi_sorted - expanding_min) / denom) * 100).fillna(50.0).clip(0, 100).reindex(df.index)
            else:
                mqi_min, mqi_max = mqi_raw.min(), mqi_raw.max()
                if mqi_max > mqi_min:
                    mqi_final = ((mqi_raw - mqi_min) / (mqi_max - mqi_min) * 100).clip(0, 100)
                else:
                    mqi_final = 50.0
            df = df.assign(matchup_quality_indicator=mqi_final)
        else:
            df = df.assign(matchup_quality_indicator=50.0)
        
        return df
    
    def _create_advanced_requirement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features required by the comprehensive rubric but not yet present.

        Adds:
        - Boom/bust rates (% weeks >20 pts, <5 pts) per player (rolling)
        - Season phase indicators (early/mid/late season)
        - Divisional and prime-time game indicators (placeholders if not filled upstream)
        - NFL experience (years in league) and age-adjusted performance curves
        - Player usage classification (workhorse/committee RB, WR1/2/3 designation)
        - Contract year indicator (heuristic from NFL experience when no contract data)
        - Cumulative workload injury risk (per requirements Section II.C)
        """
        if df.empty:
            return df

        new_cols = {}

        # --- Boom / Bust rates (rolling window, shifted to avoid leakage) ---
        # Position-specific thresholds: QB scores higher so needs higher boom threshold, etc.
        if "fantasy_points" in df.columns:
            shifted_fp = df.groupby("player_id")["fantasy_points"].shift(1)
            if "position" in df.columns:
                boom_thresh = df["position"].map(
                    {p: t["boom"] for p, t in BOOM_BUST_THRESHOLDS.items()}
                ).fillna(BOOM_BUST_DEFAULT["boom"])
                bust_thresh = df["position"].map(
                    {p: t["bust"] for p, t in BOOM_BUST_THRESHOLDS.items()}
                ).fillna(BOOM_BUST_DEFAULT["bust"])
            else:
                boom_thresh = BOOM_BUST_DEFAULT["boom"]
                bust_thresh = BOOM_BUST_DEFAULT["bust"]
            boom_flag = (shifted_fp >= boom_thresh).astype(float)
            bust_flag = (shifted_fp < bust_thresh).astype(float)
            for window in [4, 8]:
                new_cols[f"boom_rate_roll{window}"] = boom_flag.groupby(
                    df["player_id"]
                ).transform(lambda x: x.rolling(window, min_periods=1).mean())
                new_cols[f"bust_rate_roll{window}"] = bust_flag.groupby(
                    df["player_id"]
                ).transform(lambda x: x.rolling(window, min_periods=1).mean())

        # --- Season phase indicators (early wk 1-6, mid 7-12, late 13-18) ---
        if "week" in df.columns:
            new_cols["is_early_season"] = (df["week"] <= 6).astype(int)
            new_cols["is_mid_season"] = ((df["week"] >= 7) & (df["week"] <= 12)).astype(int)
            new_cols["is_late_season"] = (df["week"] >= 13).astype(int)

        # --- NFL experience (years in league) ---
        if "season" in df.columns and "player_id" in df.columns:
            first_season = df.groupby("player_id")["season"].transform("min")
            new_cols["nfl_experience_years"] = (df["season"] - first_season).clip(lower=0)

        # --- Age-adjusted performance curve ---
        # Position-specific: RBs peak earlier (~25) and decline faster; QBs/TEs peak later (~28).
        if "age" in df.columns:
            age = df["age"].fillna(26)
        elif "season" in df.columns and "birth_year" in df.columns:
            age = df["season"] - df["birth_year"]
        else:
            age = None
        if age is not None:
            if "position" in df.columns:
                peak = df["position"].map(
                    {p: c["peak"] for p, c in AGE_CURVE_PARAMS.items()}
                ).fillna(AGE_CURVE_DEFAULT["peak"])
                coeff = df["position"].map(
                    {p: c["coefficient"] for p, c in AGE_CURVE_PARAMS.items()}
                ).fillna(AGE_CURVE_DEFAULT["coefficient"])
            else:
                peak = AGE_CURVE_DEFAULT["peak"]
                coeff = AGE_CURVE_DEFAULT["coefficient"]
            new_cols["age_curve"] = 1.0 - coeff * ((age - peak) ** 2)

        # --- Player usage classification (RB: workhorse/committee; WR: WR1/2/3) ---
        if "position" in df.columns and "total_touches" in df.columns:
            # RB workhorse: >15 touches/game rolling average = workhorse
            rb_mask = df["position"] == "RB"
            touches_roll4 = df.groupby("player_id")["total_touches"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            )
            new_cols["is_workhorse_rb"] = ((touches_roll4 >= 15) & rb_mask).astype(int)
            new_cols["is_committee_rb"] = ((touches_roll4 < 15) & (touches_roll4 >= 5) & rb_mask).astype(int)

        if "position" in df.columns and "targets" in df.columns:
            # WR designation based on rolling target share within team
            tgt_roll4 = df.groupby("player_id")["targets"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            )
            wr_mask = df["position"] == "WR"
            new_cols["is_wr1"] = ((tgt_roll4 >= 7) & wr_mask).astype(int)
            new_cols["is_wr2"] = ((tgt_roll4 >= 4) & (tgt_roll4 < 7) & wr_mask).astype(int)
            new_cols["is_wr3"] = ((tgt_roll4 < 4) & (tgt_roll4 >= 1) & wr_mask).astype(int)

            # TE: red zone specialist indicator
            te_mask = df["position"] == "TE"
            if "receiving_tds" in df.columns:
                td_roll4 = df.groupby("player_id")["receiving_tds"].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )
                new_cols["is_rz_specialist_te"] = ((td_roll4 >= 0.3) & te_mask).astype(int)

        # --- Three-down back indicator (high snap share + receiving work) ---
        if "snap_share" in df.columns and "receptions" in df.columns:
            snap_roll = df.groupby("player_id")["snap_share"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(0)
            rec_roll = df.groupby("player_id")["receptions"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(0)
            rb_mask2 = df["position"] == "RB" if "position" in df.columns else pd.Series(False, index=df.index)
            new_cols["is_three_down_back"] = ((snap_roll >= 0.5) & (rec_roll >= 1.5) & rb_mask2).astype(int)

        # --- Contract year indicator (per requirements III.C) ---
        # When an explicit contract_year column is available, use it directly.
        # Otherwise, use NFL experience as a heuristic: players in years 4-5 are
        # typically in or approaching the end of their rookie contract; players in
        # years 8-9 are often approaching a second contract expiration. This is an
        # imperfect proxy but captures the known "contract year bump" effect.
        if "contract_year" in df.columns:
            new_cols["is_contract_year"] = df["contract_year"].fillna(0).astype(int)
        elif "nfl_experience_years" in new_cols:
            exp = new_cols["nfl_experience_years"]
            new_cols["is_contract_year"] = (
                ((exp == 3) | (exp == 4) | (exp == 7) | (exp == 8))
            ).astype(int)
        elif "season" in df.columns and "player_id" in df.columns:
            first_season = df.groupby("player_id")["season"].transform("min")
            exp = (df["season"] - first_season).clip(lower=0)
            new_cols["is_contract_year"] = (
                ((exp == 3) | (exp == 4) | (exp == 7) | (exp == 8))
            ).astype(int)
        else:
            new_cols["is_contract_year"] = 0

        # --- Cumulative workload injury risk (per requirements Section II.C) ---
        # Higher cumulative recent touches = higher injury probability for RBs
        # Age-adjusted: older players have higher risk at same workload level
        # Position-specific thresholds: RB (high risk at 80+ touches/4w),
        # WR/TE (high risk at 40+ touches/4w), QB (sack-based, separate below)
        if "total_touches" in df.columns:
            cum_touches_3w = df.groupby("player_id")["total_touches"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).sum()
            )
            cum_touches_4w = df.groupby("player_id")["total_touches"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).sum()
            )
            new_cols["cumulative_workload_3w"] = cum_touches_3w.fillna(0)
            new_cols["cumulative_workload_4w"] = cum_touches_4w.fillna(0)

            # Position-specific workload thresholds (touches per 4 weeks)
            pos_thresholds = {"RB": 80.0, "WR": 40.0, "TE": 35.0, "QB": 120.0}
            base_risk = cum_touches_4w.fillna(0).copy()
            if "position" in df.columns:
                for pos, threshold in pos_thresholds.items():
                    pos_mask = df["position"] == pos
                    base_risk[pos_mask] = (cum_touches_4w[pos_mask].fillna(0) / threshold).clip(0, 1)
            else:
                base_risk = (cum_touches_4w.fillna(0) / 80.0).clip(0, 1)

            # Age multiplier: risk increases ~3% per year above age 27
            # (peak athletic years); younger players get slight discount
            age_multiplier = pd.Series(1.0, index=df.index)
            if "age" in df.columns:
                age = df["age"].fillna(26)
                age_multiplier = (1.0 + 0.03 * (age - 27).clip(lower=-3)).clip(0.9, 1.5)
            elif "age_curve" in new_cols:
                # Invert age_curve: lower curve = older = higher risk
                age_multiplier = (2.0 - new_cols["age_curve"]).clip(0.9, 1.5)

            new_cols["workload_injury_risk"] = (base_risk * age_multiplier).clip(0, 1)
            # Raw (non-age-adjusted) for comparison
            new_cols["workload_injury_risk_raw"] = (cum_touches_4w.fillna(0) / 100.0).clip(0, 1)

        # --- QB-specific: sack rate based injury risk ---
        if "sacks" in df.columns and "passing_attempts" in df.columns:
            sack_roll = df.groupby("player_id")["sacks"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(0)
            new_cols["qb_sack_injury_risk"] = (sack_roll / 5.0).clip(0, 1)

        # Assign all new columns at once
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)

        return df

    def _create_return_from_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create return-from-injury production pattern features.

        Per requirements: track first 3 games back from injury.
        Features:
        - games_since_injury: 0 = just returned, NaN = no recent injury
        - is_first_3_games_back: binary flag for first 3 games after missing time
        - return_from_injury_discount: performance discount factor (0.7-1.0)
        """
        if df.empty or "player_id" not in df.columns:
            return df.assign(games_since_injury=np.nan, is_first_3_games_back=0, return_from_injury_discount=1.0)

        new_cols = {"games_since_injury": pd.Series(np.nan, index=df.index),
                    "is_first_3_games_back": pd.Series(0, index=df.index, dtype=int),
                    "return_from_injury_discount": pd.Series(1.0, index=df.index)}

        # Detect missed weeks per player (gap > 1 week between consecutive rows)
        for pid, grp in df.groupby("player_id"):
            if len(grp) < 2 or "week" not in grp.columns:
                continue
            idx = grp.index
            weeks = grp["week"].values
            gaps = np.diff(weeks)
            games_since = 999
            for i in range(1, len(idx)):
                if gaps[i - 1] > 1:
                    # Player missed at least one week
                    games_since = 0
                elif games_since < 999:
                    games_since += 1
                if games_since <= 2:
                    row_label = idx[i]
                    new_cols["games_since_injury"].at[row_label] = games_since
                    new_cols["is_first_3_games_back"].at[row_label] = 1
                    # Discount: 0.70 first game back, 0.85 second, 0.95 third
                    discount = [0.70, 0.85, 0.95][min(games_since, 2)]
                    new_cols["return_from_injury_discount"].at[row_label] = discount

        new_cols["games_since_injury"] = new_cols["games_since_injury"].fillna(99.0)
        df = df.assign(**new_cols)
        return df

    def _create_vegas_game_script_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Vegas game script predictors from schedule data.

        Per requirements: spread, over/under, implied team total, win probability.
        These come from nfl-data-py schedule data which has spread_line and total_line.
        """
        # Check if vegas features already added by external_data.py
        if "spread" in df.columns and "game_total" in df.columns:
            early_cols: dict = {}
            if "implied_team_total" not in df.columns:
                early_cols["implied_team_total"] = (df["game_total"] + df["spread"]) / 2
            if "win_probability" not in df.columns:
                # Rough conversion: spread of -7 ~ 70% win probability
                early_cols["win_probability"] = (
                    (0.5 + df["spread"].clip(-14, 14) / 28.0 * (-1)).clip(0.05, 0.95)
                )
            if early_cols:
                df = df.assign(**early_cols)
            return df

        # Try to load from schedule data
        try:
            seasons = sorted(df["season"].unique()) if "season" in df.columns else []
            if not seasons:
                raise ValueError("No seasons")

            schedules = None

            # Prefer the local `schedule` table — populated by
            # scripts/backfill_vegas_lines.py from the nflverse/nfldata
            # games.csv.  This is host-agnostic (no live call to
            # habitatring.com, which is blocked in some environments) and
            # lets the backtest run deterministically offline.  Fall back
            # to nfl_data_py only when the cache is empty or missing
            # spread_line coverage for the requested seasons.
            try:
                from src.utils.database import DatabaseManager
                db = DatabaseManager()
                with db._get_connection() as conn:
                    season_list = ",".join(str(int(s)) for s in seasons)
                    cached = pd.read_sql_query(
                        "SELECT season, week, home_team, away_team, "
                        "spread_line, total_line "
                        f"FROM schedule WHERE season IN ({season_list})",
                        conn,
                    )
                if not cached.empty and cached["spread_line"].notna().any():
                    schedules = cached.rename(columns={"total_line": "total_line"})
                    import logging
                    logging.getLogger(__name__).info(
                        "Vegas features loaded from local schedule cache: "
                        "%d rows across seasons %s (spread_line coverage: %.1f%%).",
                        len(cached), seasons,
                        cached["spread_line"].notna().mean() * 100,
                    )
            except Exception as cache_err:
                import logging
                logging.getLogger(__name__).debug(
                    "Schedule cache unavailable (%s: %s); falling back to nfl_data_py.",
                    type(cache_err).__name__, cache_err,
                )
                schedules = None

            if schedules is None or schedules.empty:
                import nfl_data_py as nfl
                schedules = nfl.import_schedules([int(s) for s in seasons])

            if schedules.empty or "spread_line" not in schedules.columns:
                raise ValueError("No spread data in schedules")

            # Build lookup: home_team + away_team + season + week -> spread, total
            sched = schedules.copy()
            sched = sched.rename(columns={"gameday": "game_date"}, errors="ignore")
            # Use total_line (Vegas over/under) not total (actual game score)
            if "total_line" in sched.columns:
                sched["vegas_total"] = sched["total_line"]
            elif "total" in sched.columns:
                sched["vegas_total"] = sched["total"]
            else:
                sched["vegas_total"] = 46.0

            # Create home and away lookups
            home_lookup = sched[["season", "week", "home_team", "spread_line", "vegas_total"]].copy()
            home_lookup = home_lookup.rename(columns={"home_team": "team"})
            home_lookup["spread"] = -home_lookup["spread_line"]  # negative spread = home favored
            home_lookup["game_total"] = home_lookup["vegas_total"]
            home_lookup["implied_team_total"] = (home_lookup["game_total"] - home_lookup["spread"]) / 2

            away_lookup = sched[["season", "week", "away_team", "spread_line", "vegas_total"]].copy()
            away_lookup = away_lookup.rename(columns={"away_team": "team"})
            away_lookup["spread"] = away_lookup["spread_line"]  # positive spread = away underdog
            away_lookup["game_total"] = away_lookup["vegas_total"]
            away_lookup["implied_team_total"] = (away_lookup["game_total"] + away_lookup["spread"]) / 2

            vegas = pd.concat([home_lookup, away_lookup], ignore_index=True)
            vegas = vegas[["season", "week", "team", "spread", "game_total", "implied_team_total"]].drop_duplicates()

            # Merge
            before_len = len(df)
            df = df.merge(vegas, on=["season", "week", "team"], how="left", suffixes=("", "_vegas"))
            # Prefer existing columns if already present
            for col in ["spread", "game_total", "implied_team_total"]:
                vegas_col = f"{col}_vegas"
                if vegas_col in df.columns:
                    if col not in df.columns:
                        df[col] = df[vegas_col]
                    else:
                        df[col] = df[col].fillna(df[vegas_col])
                    df = df.drop(columns=[vegas_col])

        except Exception as e:
            # Surface the silent-fallback failure so silently-degraded Vegas
            # features can't masquerade as live model inputs.  Previously this
            # was an unconditional `pass`, which let nfl_data_py outages
            # silently collapse implied_team_total/spread to constants and
            # train Ridge on dead inputs.  See docs/PHASE_1_VEGAS_FINDINGS.md.
            import logging
            logging.getLogger(__name__).warning(
                "Vegas-line load from nfl_data_py failed (%s: %s); "
                "implied_team_total/spread will fall back to constant defaults "
                "(23.0 / 0.0) for any rows lacking pre-merged Vegas columns. "
                "Run scripts/check_vegas_features.py to diagnose.",
                type(e).__name__, e,
            )

        # Defaults for missing values — compute all new columns before assigning.
        # Track how many rows received a constant fill so the downstream test
        # in tests/test_backtest_validation.py can flag silent degradation.
        default_cols: dict = {}
        n_filled_with_default = 0
        for col, default in [("spread", 0.0), ("game_total", 46.0), ("implied_team_total", 23.0)]:
            if col not in df.columns:
                default_cols[col] = default
                n_filled_with_default += len(df)
            else:
                missing_mask = df[col].isna()
                missing_count = int(missing_mask.sum())
                if missing_count:
                    n_filled_with_default += missing_count
                filled = df[col].fillna(default)
                if filled is not df[col]:
                    default_cols[col] = filled
        if n_filled_with_default and len(df) > 0:
            frac = n_filled_with_default / (len(df) * 3)  # 3 columns counted
            if frac > 0.5:
                import logging
                logging.getLogger(__name__).warning(
                    "Vegas features defaulted on %.0f%% of rows (>50%% threshold). "
                    "implied_team_total/spread are effectively dead inputs for this "
                    "training set.  Verify nfl_data_py.import_schedules works in this "
                    "environment.",
                    frac * 100,
                )
        if default_cols:
            df = df.assign(**default_cols)

        # Win probability from spread and is_favorite — batch together
        derived_cols: dict = {}
        if "win_probability" not in df.columns:
            derived_cols["win_probability"] = (
                (0.5 + df["spread"].clip(-14, 14) / 28.0 * (-1)).clip(0.05, 0.95)
            )
        if "is_favorite" not in df.columns:
            derived_cols["is_favorite"] = (df["spread"] < 0).astype(int)
        if derived_cols:
            df = df.assign(**derived_cols)

        return df

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge per-game weather from the ``game_weather`` table.

        Pre-game weather is known before kickoff, so it's treated like the
        Vegas game-script features (unrolled, current-week values) rather
        than rolled/lagged. Dome games have no recorded wind/precip/temp —
        those rows get the "no weather effect" defaults, not the outdoor
        column median.
        """
        if not all(c in df.columns for c in ["season", "week", "team"]):
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH
            seasons = sorted(df["season"].dropna().unique().tolist())
            if not seasons:
                raise ValueError("No seasons")
            season_list = ",".join(str(int(s)) for s in seasons)
            conn = sqlite3.connect(str(DB_PATH))
            weather = pd.read_sql_query(
                "SELECT season, week, home_team, away_team, is_dome, "
                "temp_f, wind_mph, precip_mm FROM game_weather "
                f"WHERE season IN ({season_list})",
                conn,
            )
            conn.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Weather load from game_weather failed (%s: %s); "
                "wind_speed_mph/is_dome/precipitation_flag/temperature_bucket "
                "will default to 'no weather effect' for every row.",
                type(e).__name__, e,
            )
            weather = pd.DataFrame(columns=["season", "week", "team", "is_dome", "temp_f", "wind_mph", "precip_mm"])

        if not weather.empty:
            home = weather.rename(columns={"home_team": "team"}).drop(columns=["away_team"])
            away = weather.rename(columns={"away_team": "team"}).drop(columns=["home_team"])
            by_team = pd.concat([home, away], ignore_index=True).drop_duplicates(
                subset=["season", "week", "team"]
            )
            df = df.merge(by_team, on=["season", "week", "team"], how="left", suffixes=("", "_wx"))

        is_dome = df.get("is_dome", pd.Series(0, index=df.index)).fillna(0).astype(int)
        wind_speed_mph = df.get("wind_mph", pd.Series(np.nan, index=df.index))
        precip_mm = df.get("precip_mm", pd.Series(np.nan, index=df.index))
        temp_f = df.get("temp_f", pd.Series(np.nan, index=df.index))

        # Dome games: no wind/precip, and temperature is climate-controlled
        # (bucketed as mild regardless of a missing/placeholder temp_f).
        weather_cols = {
            "wind_speed_mph": wind_speed_mph.where(is_dome == 0, 0.0).fillna(0.0),
            "precipitation_flag": ((precip_mm.fillna(0.0) > 0) & (is_dome == 0)).astype(int),
            "temperature_bucket": pd.cut(
                temp_f.where(is_dome == 0, 65.0).fillna(65.0),
                bins=[-100, 20, 32, 50, 200],
                labels=[3, 2, 1, 0],
            ).astype(int),
            "is_dome": is_dome,
        }
        df = df.assign(**weather_cols)
        df = df.drop(columns=[c for c in ["temp_f", "wind_mph", "precip_mm"] if c in df.columns])
        return df

    # ------------------------------------------------------------------
    # NGS (Next Gen Stats) features
    # ------------------------------------------------------------------

    _ngs_cache: Optional[pd.DataFrame] = None

    def _load_ngs_data(self) -> pd.DataFrame:
        """Load NGS data from the database (cached per instance)."""
        if self._ngs_cache is not None:
            return self._ngs_cache
        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            self._ngs_cache = db.get_ngs_data()
            if self._ngs_cache is not None and not self._ngs_cache.empty:
                print(f"  NGS data loaded: {len(self._ngs_cache)} rows")
            else:
                self._ngs_cache = pd.DataFrame()
        except Exception as e:
            print(f"  WARNING: Could not load NGS data: {e}")
            self._ngs_cache = pd.DataFrame()
        return self._ngs_cache

    def _merge_ngs_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Left-join NGS stats onto training data by (player_id, season, week)."""
        ngs = self._load_ngs_data()
        if ngs.empty:
            return df
        join_keys = ["player_id", "season", "week"]
        ngs_cols = [c for c in ngs.columns if c.startswith("ngs_")]
        # Drop any existing ngs_ columns to avoid duplication on re-runs
        existing = [c for c in ngs_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
        merged = df.merge(ngs[join_keys + ngs_cols], on=join_keys, how="left")
        for c in ngs_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
        return merged

    def _create_ngs_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling-3 averages of key NGS metrics per position.

        Shift by 1 to prevent same-week leakage.
        """
        ngs_rolling_cols = [c for c in df.columns if c.startswith("ngs_")]
        if not ngs_rolling_cols:
            return df
        for col in ngs_rolling_cols:
            roll_col = f"{col}_roll3_mean"
            df[roll_col] = (
                df.groupby("player_id")[col]
                .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            )
            df[roll_col] = df[roll_col].fillna(0.0)
        return df

    # ------------------------------------------------------------------
    # Draft capital features
    # ------------------------------------------------------------------

    def _merge_draft_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add draft_capital_value (decayed by years since draft) to the DataFrame."""
        if "draft_capital_value" in df.columns:
            return df
        try:
            from src.utils.database import DatabaseManager
            from src.features.advanced_rookie_injury import draft_capital_value_from_pick
            db = DatabaseManager()
            with db._get_connection() as conn:
                draft_df = pd.read_sql_query(
                    "SELECT player_id, draft_season, draft_pick FROM draft_picks_v2 "
                    "WHERE player_id LIKE '00-%' AND draft_pick IS NOT NULL",
                    conn,
                )
            if draft_df.empty:
                df["draft_capital_value"] = 0.05
                return df
            draft_df["raw_capital"] = draft_df["draft_pick"].apply(draft_capital_value_from_pick)
            draft_df = draft_df.drop_duplicates(subset=["player_id"], keep="first")
            # Use existing draft_season if already present (from season_long_features),
            # otherwise merge it from draft_df to avoid _x/_y column conflicts.
            merge_cols = ["player_id", "raw_capital"]
            if "draft_season" not in df.columns:
                merge_cols.insert(1, "draft_season")
            df = df.merge(draft_df[merge_cols], on="player_id", how="left")
            # Decay: full value year 1, zero by year 6
            draft_season_col = df["draft_season"] if "draft_season" in df.columns else df["season"]
            years_since = df["season"] - draft_season_col.fillna(df["season"])
            df["draft_capital_value"] = (
                df["raw_capital"].fillna(0.05) * (1.0 - 0.2 * years_since).clip(lower=0.0)
            )
            df = df.drop(columns=["raw_capital"], errors="ignore")
        except Exception as e:
            print(f"  WARNING: Could not merge draft capital: {e}")
            df["draft_capital_value"] = 0.05
        return df

    def _add_prior_season_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team_prior_season_wins: the team's actual wins in the prior season.

        Computed from the schedule table (count wins per team-season) and lagged
        by 1 season so it's strictly causal — known before the current season starts.
        Defaults to 8.5 (half of 17, league average) when unavailable.
        """
        if "team" not in df.columns or "season" not in df.columns:
            df["team_prior_season_wins"] = 8.5
            return df
        try:
            import sqlite3
            from config.settings import DB_PATH
            conn = sqlite3.connect(str(DB_PATH))
            rows = conn.execute("""
                SELECT
                    home_team AS team, season,
                    SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) AS wins
                FROM schedule
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL AND week <= 18
                GROUP BY home_team, season
                UNION ALL
                SELECT
                    away_team AS team, season,
                    SUM(CASE WHEN away_score > home_score THEN 1 ELSE 0 END) AS wins
                FROM schedule
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL AND week <= 18
                GROUP BY away_team, season
            """).fetchall()
            conn.close()
        except Exception:
            df["team_prior_season_wins"] = 8.5
            return df

        import pandas as pd
        wins_df = pd.DataFrame(rows, columns=["team", "season", "wins"])
        wins_df = wins_df.groupby(["team", "season"], as_index=False)["wins"].sum()
        # Lag by 1 season: prior_season_wins for season S comes from season S-1
        wins_df["season"] = wins_df["season"] + 1
        wins_df = wins_df.rename(columns={"wins": "team_prior_season_wins"})

        df = df.merge(wins_df, on=["team", "season"], how="left")
        df["team_prior_season_wins"] = df["team_prior_season_wins"].fillna(8.5)
        return df

    def _add_team_qb_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team_qb_pass_epa_per_att: prior-season QB passing EPA per attempt.

        A WR/TE on a high-EPA QB team should project higher controlling for
        target share. Lagged by 1 season so it's strictly causal.
        Defaults to 0.0 (league average) when unavailable.
        """
        if "team" not in df.columns or "season" not in df.columns:
            df["team_qb_pass_epa_per_att"] = 0.0
            return df
        try:
            import sqlite3
            from config.settings import DB_PATH
            conn = sqlite3.connect(str(DB_PATH))
            rows = conn.execute("""
                SELECT pws.team, pws.season,
                       SUM(pws.pass_epa) AS total_epa,
                       SUM(pws.passing_attempts) AS total_att
                FROM player_weekly_stats pws
                JOIN players p ON pws.player_id = p.player_id
                WHERE p.position = 'QB'
                  AND pws.passing_attempts > 0
                GROUP BY pws.team, pws.season
                HAVING SUM(pws.passing_attempts) > 50
            """).fetchall()
            conn.close()
        except Exception:
            df["team_qb_pass_epa_per_att"] = 0.0
            return df

        import pandas as pd
        qb_df = pd.DataFrame(rows, columns=["team", "season", "total_epa", "total_att"])
        qb_df["team_qb_pass_epa_per_att"] = qb_df["total_epa"] / qb_df["total_att"]
        qb_df = qb_df[["team", "season", "team_qb_pass_epa_per_att"]]
        # Lag by 1 season: prior QB quality predicts this season's WR/TE output
        qb_df["season"] = qb_df["season"] + 1

        df = df.merge(qb_df, on=["team", "season"], how="left")
        df["team_qb_pass_epa_per_att"] = df["team_qb_pass_epa_per_att"].fillna(0.0)
        return df

    def _add_current_qb_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add current_qb_epa_per_att: EPA/att for the actual QB1 assigned to each team.

        Unlike team_qb_pass_epa_per_att (which lags the prior-year team aggregate),
        this identifies the specific QB by player identity and uses their rolling
        3-season career EPA. For 2026+, QB1 is sourced from nfl-data-py current
        rosters rather than inferred from prior-year stats, capturing free agency
        moves and trades before any games are played.

        Causality: historical QB1 = QB with most attempts in season-1 for that team.
        Career EPA = sum over the 3 seasons strictly before the prediction season.
        """
        if "team" not in df.columns or "season" not in df.columns:
            df["current_qb_epa_per_att"] = 0.0
            return df

        try:
            import sqlite3
            from config.settings import DB_PATH
            conn = sqlite3.connect(str(DB_PATH))
            rows = conn.execute("""
                SELECT pws.player_id, pws.season, pws.team,
                       SUM(pws.pass_epa)          AS total_epa,
                       SUM(pws.passing_attempts)  AS total_att
                FROM player_weekly_stats pws
                JOIN players p ON pws.player_id = p.player_id
                WHERE p.position = 'QB' AND pws.passing_attempts > 0
                GROUP BY pws.player_id, pws.season, pws.team
                HAVING SUM(pws.passing_attempts) > 50
            """).fetchall()
            conn.close()
        except Exception:
            df["current_qb_epa_per_att"] = 0.0
            return df

        qb_seasons = pd.DataFrame(rows, columns=["player_id", "season", "team", "total_epa", "total_att"])

        # QB1 per team per season (causal): QB with most attempts in the PRIOR season.
        # Advance season by 1 so the lookup key is the prediction season.
        qb1_prior = (
            qb_seasons.sort_values("total_att", ascending=False)
            .groupby(["team", "season"])[["player_id"]]
            .first()
            .reset_index()
            .assign(target_season=lambda x: x["season"] + 1)
            [["team", "target_season", "player_id"]]
        )

        # Career EPA per att for each QB going into each target season:
        # sum over the 3 seasons strictly before target_season.
        target_seasons = sorted(df["season"].unique())
        qb_prior_epa_rows = []
        for ts in target_seasons:
            prior = qb_seasons[(qb_seasons["season"] >= ts - 3) & (qb_seasons["season"] < ts)]
            if prior.empty:
                continue
            agg = prior.groupby("player_id").agg(
                epa=("total_epa", "sum"), att=("total_att", "sum")
            ).reset_index()
            agg = agg[agg["att"] >= 50]
            agg["current_qb_epa_per_att"] = agg["epa"] / agg["att"]
            agg["target_season"] = ts
            qb_prior_epa_rows.append(agg[["player_id", "target_season", "current_qb_epa_per_att"]])

        if not qb_prior_epa_rows:
            df["current_qb_epa_per_att"] = 0.0
            return df

        qb_prior_epa_df = pd.concat(qb_prior_epa_rows, ignore_index=True)

        # For 2026+ seasons, override QB1 lookup with live roster data.
        upcoming = [s for s in target_seasons if s >= 2026]
        if upcoming:
            try:
                import nfl_data_py as nfl
                rosters = nfl.import_seasonal_rosters(upcoming)
                rosters = rosters[
                    (rosters["position"] == "QB") & (rosters["status"] == "ACT")
                ].copy()
                rosters["season"] = rosters["season"].astype(int)
                # Rank QBs by career attempts so QB1 = most experienced (likely starter)
                qb_career_att = qb_seasons.groupby("player_id")["total_att"].sum().reset_index()
                qb_career_att.columns = ["player_id", "career_att"]
                rosters = rosters.merge(qb_career_att, on="player_id", how="left")
                rosters["career_att"] = rosters["career_att"].fillna(0)
                qb1_roster = (
                    rosters.sort_values("career_att", ascending=False)
                    .groupby(["team", "season"])
                    .first()
                    .reset_index()[["team", "season", "player_id"]]
                    .rename(columns={"season": "target_season"})
                )
                # Replace historical prior-season QB1 entries for upcoming seasons
                qb1_prior = qb1_prior[~qb1_prior["target_season"].isin(upcoming)]
                qb1_prior = pd.concat([qb1_prior, qb1_roster], ignore_index=True)
            except Exception:
                pass  # Fall back to prior-season QB1 if roster fetch fails

        # Join QB1 identity → their career EPA
        team_qb_epa = qb1_prior.merge(
            qb_prior_epa_df, on=["player_id", "target_season"], how="left"
        )[["team", "target_season", "current_qb_epa_per_att"]]
        team_qb_epa = team_qb_epa.rename(columns={"target_season": "season"})

        df = df.merge(team_qb_epa, on=["team", "season"], how="left")
        df["current_qb_epa_per_att"] = df["current_qb_epa_per_att"].fillna(0.0)
        return df

    def _create_advanced_analytics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced analytics features: sentiment, coaching, suspension, trade, playoff."""
        from src.features.advanced_analytics import AdvancedAnalyticsEngine
        engine = AdvancedAnalyticsEngine()
        return engine.add_all_features(df)

    def _merge_injury_data_from_cache(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject `injury_score` and `is_injured` from the local player_injuries
        cache.  Runs BEFORE ``_ensure_injury_rookie_features``.

        Without this, the only prior path that populated `injury_score` lived
        in ``src/data/external_data.py::add_external_features`` and was wired
        into the old ``ModelBacktester`` path only — the walk-forward
        ``TimeSeriesBacktester`` never called it, so injury_score defaulted
        to 1.0 (healthy) for every row and the declared feature was silently
        dead.  Phase 3 fix: query the `player_injuries` table populated by
        scripts/backfill_injuries.py and merge per (player_id, season, week).

        Mapping: the existing ``InjuryDataLoader.INJURY_STATUS_SCORES`` is
        re-used so the semantic matches the legacy ``add_external_features``
        path (0.0 = Out / IR, 0.15 = Doubtful, 0.50 = Questionable, 0.85 =
        Probable, 1.0 = no report / healthy).
        """
        required = {"player_id", "season", "week"}
        if df.empty or not required.issubset(df.columns):
            return df

        import logging
        logger = logging.getLogger(__name__)

        # Lazy import to avoid pulling external_data (and its heavy deps) in
        # contexts where the merge would no-op anyway.
        try:
            from src.data.external_data import InjuryDataLoader
            status_map = InjuryDataLoader.INJURY_STATUS_SCORES
        except Exception:
            status_map = {
                "Out": 0.0, "Doubtful": 0.15, "Questionable": 0.50,
                "Probable": 0.85, "IR": 0.0, "IR-R": 0.0,
            }

        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique())
            if not seasons:
                return df
            season_list = ",".join(str(int(s)) for s in seasons)
            with db._get_connection() as conn:
                cached = pd.read_sql_query(
                    "SELECT player_id, season, week, report_status "
                    f"FROM player_injuries WHERE season IN ({season_list})",
                    conn,
                )
        except Exception as e:
            logger.warning(
                "player_injuries cache load failed (%s: %s); injury_score "
                "will default to 1.0 on every row.  Run "
                "scripts/backfill_injuries.py to populate.",
                type(e).__name__, e,
            )
            return df

        if cached.empty:
            logger.warning(
                "player_injuries cache is empty for seasons %s; injury_score "
                "will default to 1.0 on every row.  Run "
                "scripts/backfill_injuries.py.",
                seasons,
            )
            return df

        # Map report_status -> score.  Unknown / empty -> 1.0 (healthy).
        def _score(s):
            if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
                return 1.0
            return float(status_map.get(s, 1.0))

        cached["_injury_score"] = cached["report_status"].map(_score)
        cached["_is_injured"] = (cached["_injury_score"] < 1.0).astype(int)
        _warn_on_probable_era_span(cached)

        merged = df.merge(
            cached[["player_id", "season", "week", "_injury_score", "_is_injured"]],
            on=["player_id", "season", "week"],
            how="left",
        )

        # Coverage report — how many rows found a matching injury row?
        # (Matched with report_status = None -> 1.0 counts as a match; a null
        # here means the player-week simply isn't in the injury report at
        # all, which is the common case for starters.  Not a silent fallback.)
        n_total = len(merged)
        n_matched = int(merged["_injury_score"].notna().sum())
        if n_total:
            logger.info(
                "Injury cache merged: %d/%d rows matched (%.1f%%); "
                "unmatched rows treated as healthy.",
                n_matched, n_total, n_matched / n_total * 100,
            )

        # Resolve: prefer cached when present, fall back to existing column
        # (from src/data/external_data.py's older path if both fired), else
        # 1.0 / 0.
        existing_score = (
            pd.to_numeric(df["injury_score"], errors="coerce")
            if "injury_score" in df.columns else pd.Series(np.nan, index=df.index)
        )
        existing_injured = (
            pd.to_numeric(df["is_injured"], errors="coerce")
            if "is_injured" in df.columns else pd.Series(np.nan, index=df.index)
        )

        merged["injury_score"] = (
            merged["_injury_score"]
            .combine_first(existing_score.reindex(merged.index))
            .fillna(1.0)
            .clip(0.0, 1.0)
        )
        merged["is_injured"] = (
            merged["_is_injured"]
            .combine_first(existing_injured.reindex(merged.index))
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

        return merged.drop(columns=[c for c in ("_injury_score", "_is_injured") if c in merged.columns])

    def _ensure_injury_rookie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure injury and rookie predictor columns exist with safe defaults.
        Used as predictors for utilization: injury_score (0-1 availability),
        is_injured (0/1), is_rookie (limited sample = higher uncertainty).
        When external injury data is absent, defaults assume full availability.
        """
        if df.empty:
            return df

        injury_cols: dict = {}

        # Injury: from external_data when available; else full availability
        injury_cols["injury_score"] = (
            df["injury_score"].fillna(1.0).clip(0.0, 1.0)
            if "injury_score" in df.columns else 1.0
        )
        injury_cols["is_injured"] = (
            df["is_injured"].fillna(0).astype(int).clip(0, 1)
            if "is_injured" in df.columns else 0
        )

        # `is_rookie` deliberately NOT set here. This function used to define it
        # as `games_count <= 8`, which is not rookie status at all -- it labels
        # any veteran who missed half a season as a rookie. It was harmless only
        # because advanced_rookie_injury.py overwrites the column afterwards
        # with the real definition (season == first_season). But that module was
        # wrapped in a bare `except` until FEATURE_VERSION 34, so on any failure
        # this wrong definition silently survived into training, attached to
        # rookie_* priors that are 0.0 for veterans. One owner now:
        # advanced_rookie_injury.py.

        df = df.assign(**injury_cols)
        return df
    
    def _flag_outliers(self, df: pd.DataFrame, sigma_threshold: float = 3.0) -> pd.DataFrame:
        """
        Flag statistical outliers (>3 standard deviations) per requirements Section VI.C.
        
        Legitimate outliers (record-breaking performances) are kept but flagged.
        Injury-impacted games get special handling via is_outlier_injury flag.
        Creates 'is_statistical_outlier' column (0/1) for model awareness.
        """
        if df.empty:
            return df
        key_cols = ["fantasy_points", "total_yards", "total_touches", "utilization_score"]
        key_cols = [c for c in key_cols if c in df.columns]
        if not key_cols:
            return df.assign(is_statistical_outlier=0)

        outlier_mask = pd.Series(False, index=df.index)

        # Use expanding mean/std to avoid future data leakage in outlier thresholds
        has_temporal = "season" in df.columns and "week" in df.columns
        if has_temporal:
            sort_idx = df.sort_values(["season", "week"]).index
        for col in key_cols:
            if col not in df.columns:
                continue
            if has_temporal:
                col_sorted = df[col].reindex(sort_idx)
                shifted = col_sorted.shift(1)
                mean_val = shifted.expanding(min_periods=10).mean().reindex(df.index)
                std_val = shifted.expanding(min_periods=10).std().reindex(df.index)
            else:
                mean_val = df[col].mean()
                std_val = df[col].std()
            if isinstance(std_val, (int, float)) and (pd.isna(std_val) or std_val == 0):
                continue
            col_outlier = (df[col] - mean_val).abs() > sigma_threshold * std_val
            outlier_mask = outlier_mask | col_outlier.fillna(False)

        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  Outlier detection: {n_outliers} rows flagged (>{sigma_threshold}σ in {key_cols})")

        # Injury-impacted outlier flag: low performance + injured (expanding stats)
        if "injury_score" in df.columns and "fantasy_points" in df.columns:
            if has_temporal:
                fp_sorted = df["fantasy_points"].reindex(sort_idx)
                fp_shifted = fp_sorted.shift(1)
                fp_mean = fp_shifted.expanding(min_periods=10).mean().reindex(df.index)
                fp_std = fp_shifted.expanding(min_periods=10).std().reindex(df.index)
            else:
                fp_mean = df["fantasy_points"].mean()
                fp_std = df["fantasy_points"].std()
            if isinstance(fp_std, (int, float)) and (pd.isna(fp_std) or fp_std == 0):
                is_outlier_injury = 0
            else:
                low_perf = df["fantasy_points"] < (fp_mean - 2 * fp_std)
                injured = df["injury_score"].fillna(1.0) < 0.7
                is_outlier_injury = (low_perf & injured).fillna(False).astype(int)
        else:
            is_outlier_injury = 0

        # Assign both outlier flags at once
        df = df.assign(is_statistical_outlier=outlier_mask.astype(int), is_outlier_injury=is_outlier_injury)
        return df

    def _check_missing_rate(self, df: pd.DataFrame, threshold_pct: float = 5.0) -> None:
        """
        Log features with missing rate above threshold (requirement: max 5% per feature).
        Does not drop columns; call before _impute_missing for visibility.
        """
        if df.empty or len(df) == 0:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_rows = len(df)
        over = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            missing = df[col].isna().sum()
            if missing == 0:
                continue
            pct = 100.0 * missing / n_rows
            if pct > threshold_pct:
                over.append((col, round(pct, 1)))
        if over:
            over.sort(key=lambda x: -x[1])
            # In test runs this warning is noisy and expected due to synthetic/sparse fixtures.
            if os.getenv("PYTEST_CURRENT_TEST"):
                return
            if os.getenv("NFL_FEATURE_WARN_MISSINGNESS", "1") != "1":
                return
            import warnings
            warnings.warn(
                f"Feature engineering: {len(over)} features exceed {threshold_pct}% missing "
                f"(requirement guideline). Consider reviewing data or dropping: {[x[0] for x in over[:10]]}"
                + (f" ... and {len(over) - 10} more" if len(over) > 10 else ""),
                UserWarning,
                stacklevel=2,
            )

    def _normalize_season_relative(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce temporal leakage by season-relative normalization.

        Addresses adversarial-validation AUC ≈ 0.957 by preventing features
        from encoding absolute temporal position (era, season count, stat
        magnitude drift across NFL rule changes).

        Uses expanding backward-looking statistics (shift + expanding) so each
        row only sees strictly prior weeks within the same season/position,
        avoiding within-season lookahead bias.
        """
        # 1. Cap nfl_experience_years to reduce long-tail temporal signal
        if "nfl_experience_years" in df.columns:
            df["nfl_experience_years"] = df["nfl_experience_years"].clip(upper=12)

        # 2. Season-normalize rolling stat means to remove era-level magnitude drift
        if "season" in df.columns and "position" in df.columns:
            roll_mean_cols = [c for c in df.columns
                             if "_roll" in c and "_mean" in c]
            if roll_mean_cols:
                df = df.sort_values(["season", "position", "week"]).reset_index(drop=True)
                # Ensure float dtype so z-score values can be stored
                for col in roll_mean_cols:
                    df[col] = df[col].astype(float)
                for col in roll_mean_cols:
                    grp = df.groupby(["season", "position"])[col]
                    # shift(1) + expanding so each row sees only prior weeks
                    expanding_mean = grp.transform(
                        lambda x: x.shift(1).expanding(min_periods=3).mean()
                    )
                    expanding_std = grp.transform(
                        lambda x: x.shift(1).expanding(min_periods=3).std()
                    ).clip(lower=0.01)
                    valid = expanding_mean.notna()
                    df.loc[valid, col] = (
                        (df.loc[valid, col] - expanding_mean[valid]) / expanding_std[valid]
                    )
                    # Rows without enough prior data (weeks 1-3) are left unnormalized

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace inf with nan and impute NaN in numeric columns so model never sees missing/inf.
        Root cause: LEFT JOINs (team_stats, utilization, defense), rolling/lag NaNs, failed schedule.

        Imputation strategy (missingness-aware):
        1. For rolling/lag features with >5% missing, add a binary ``_missing`` indicator
           column so the model can distinguish "no prior data" from "low prior performance."
        2. Impute with column median (avoids distorting distribution), fallback 0.

        Per requirements, features with >5% missing are flagged in _check_missing_rate;
        we still impute so pipelines run.
        """
        if df.empty:
            return df
        # Replace inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Add binary missing indicators for rolling/lag features with meaningful missingness.
        # These let the model learn that early-season NaN != zero performance.
        n_rows = len(df)
        missing_indicator_cols = {}
        rolling_lag_tokens = ("_roll", "_lag", "_ewm", "_trend")
        for col in numeric_cols:
            if col not in df.columns:
                continue
            if not any(tok in col for tok in rolling_lag_tokens):
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            miss_pct = n_missing / n_rows
            # Only add indicator when missingness is structurally meaningful (>2%)
            if miss_pct > 0.02:
                indicator_name = f"{col}_missing"
                if indicator_name not in df.columns:
                    missing_indicator_cols[indicator_name] = df[col].isna().astype(np.int8)

        if missing_indicator_cols:
            indicator_df = pd.DataFrame(missing_indicator_cols, index=df.index)
            df = pd.concat([df, indicator_df], axis=1)

        # Fill NaN: position-aware median per column to avoid cross-position
        # contamination (e.g., filling RB completion_pct with QB median).
        # Re-fetch numeric cols since we may have added indicator columns.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        qb_specific_tokens = (
            "completion", "passer", "passing", "yards_per_attempt",
            "td_rate", "int_rate", "qb_", "sack_rate", "air_yards_per",
        )
        has_position = "position" in df.columns
        # Snap features are owned by the persisted snap-imputation step
        # (utilization_score.apply_snap_imputation), which fills them from
        # train-fitted position x era medians. Filling them here would use a
        # median of whatever frame is in hand -- including test rows at
        # backtest time -- and would consume the missingness before the
        # owning step ever sees it. The `_missing` indicators above are still
        # emitted for them; only the fill is deferred.
        for col in numeric_cols:
            if col not in df.columns or not df[col].isna().any():
                continue
            if col in _SNAP_IMPUTATION_OWNED or col in _STRUCTURALLY_MISSING:
                continue
            is_qb_col = any(tok in col.lower() for tok in qb_specific_tokens)
            if is_qb_col and has_position:
                # QB-specific columns: fill with QB median for QBs, 0 for others
                qb_med = df.loc[df["position"] == "QB", col].median()
                if pd.isna(qb_med):
                    qb_med = 0.0
                df.loc[df["position"] == "QB", col] = df.loc[df["position"] == "QB", col].fillna(qb_med)
                df[col] = df[col].fillna(0.0)
            else:
                med = df[col].median()
                if pd.isna(med):
                    med = 0.0
                df[col] = df[col].fillna(med)
        return df
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Update list of feature columns.
        
        Excludes identifiers, raw targets, and leakage-prone columns. This
        guard is intentionally conservative to prevent model-output or target
        leakage in downstream training/evaluation pipelines.
        """
        exclude_cols = {
            "player_id", "name", "season", "week", "team", "opponent",
            "home_away", "position", "fantasy_points", "id", "created_at",
            "games_played",
        }
        # Also exclude any target columns that might have been added before
        # feature selection (e.g. during training pipeline)
        exclude_prefixes = ("target_", "actual_for_backtest", "predicted_", "baseline_")
        
        self.feature_columns = [
            col for col in df.columns 
            if col not in exclude_cols
            and not any(col.startswith(p) for p in exclude_prefixes)
            and df[col].dtype in [np.float64, np.int64, float, int]
        ]
        from src.utils.leakage import filter_feature_columns
        # Deliberately NOT wrapped in try/except (GAPS.md §9 audit,
        # 2026-08-05 follow-up): a failure here would leave
        # self.feature_columns completely UNFILTERED (leakage columns like
        # target_1w/utilization_score could remain), and this function is
        # simple/stable enough that a failure means something is
        # seriously wrong -- letting it raise is safer than training on
        # unfiltered features with only a printed warning that could be
        # missed. run_weekly_retrain() already catches and durably records
        # any exception from train_models() in RETRAIN_STATUS_FILE, so
        # raising here doesn't silently crash unattended automation.
        self.feature_columns = filter_feature_columns(self.feature_columns)
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_columns
    
    def prepare_training_data(self, df: pd.DataFrame, 
                              target_weeks: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with features
            target_weeks: Number of weeks ahead to predict (1-18)
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = df.copy()
        
        # Create target: fantasy points N weeks ahead
        if target_weeks == 1:
            df["target"] = df.groupby("player_id")["fantasy_points"].shift(-1)
        else:
            # For multi-week prediction, use sum of next N weeks
            df["target"] = df.groupby("player_id")["fantasy_points"].transform(
                lambda x: x.shift(-1).rolling(window=target_weeks, min_periods=1).sum()
            )
        
        # Remove rows without target
        df = df.dropna(subset=["target"])
        
        # Get features
        feature_cols = self.get_feature_columns()
        available_features = [c for c in feature_cols if c in df.columns]
        
        X = df[available_features].copy()
        y = df["target"]
        
        # Clean inf values and fill NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X, y
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for making predictions.
        
        Args:
            df: DataFrame with player stats
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Create features
        df = self.create_features(df, include_target=False)
        
        # Get most recent row per player
        latest = df.groupby("player_id").last().reset_index()
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        available_features = [c for c in feature_cols if c in latest.columns]
        
        return latest[["player_id", "name", "position", "team"] + available_features]


class PositionFeatureEngineer(FeatureEngineer):
    """Position-specific feature engineering."""

    def __init__(self, position: str, feature_mode: Optional[str] = None):
        super().__init__(feature_mode=feature_mode)
        self.position = position
    
    def create_features(self, df: pd.DataFrame, 
                        include_target: bool = True) -> pd.DataFrame:
        """Create position-specific features."""
        # Filter to position
        df = df[df["position"] == self.position].copy()
        
        # Create base features
        df = super().create_features(df, include_target)
        
        # Add position-specific features
        if self.position == "QB":
            df = self._create_qb_features(df)
        elif self.position == "RB":
            df = self._create_rb_features(df)
        elif self.position == "WR":
            df = self._create_wr_features(df)
        elif self.position == "TE":
            df = self._create_te_features(df)
        
        return df
    
    def _create_qb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """QB-specific features per requirements: pass attempts, completion %, air yards,
        TD/INT ratio, rushing, sacks, time in pocket, deep ball, red zone efficiency."""
        new_cols = {}
        new_cols["rush_pct_of_plays"] = safe_divide(
            df["rushing_attempts"],
            df["passing_attempts"] + df["rushing_attempts"]
        ) * 100
        new_cols["yards_per_completion"] = safe_divide(
            df["passing_yards"], df["passing_completions"]
        )
        if "sacks" in df.columns:
            adj = df["passing_attempts"] + df["sacks"]
            new_cols["adj_pass_attempts"] = adj
            new_cols["sack_rate"] = safe_divide(df["sacks"], adj) * 100
        # Time in pocket (when available from PBP data)
        if "time_in_pocket" in df.columns:
            new_cols["avg_time_in_pocket"] = df["time_in_pocket"]
        # Red zone efficiency
        if "redzone_attempts" in df.columns and "redzone_completions" in df.columns:
            new_cols["rz_completion_pct"] = safe_divide(
                df["redzone_completions"], df["redzone_attempts"]
            ) * 100
        # Deep ball attempts (20+ yards)
        if "deep_pass_attempts" in df.columns:
            new_cols["deep_pass_rate"] = safe_divide(
                df["deep_pass_attempts"], df["passing_attempts"]
            ) * 100
            if "deep_pass_completions" in df.columns:
                new_cols["deep_pass_accuracy"] = safe_divide(
                    df["deep_pass_completions"], df["deep_pass_attempts"]
                ) * 100
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
    
    def _create_rb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RB-specific features per requirements: rush attempts, receptions, red zone/goal line,
        snap share, route participation, yards after contact, broken tackles."""
        new_cols = {}
        new_cols["receiving_pct"] = safe_divide(
            df["receptions"], df["total_touches"]
        ) * 100
        new_cols["td_per_touch"] = safe_divide(
            df["total_tds"], df["total_touches"]
        )
        if "total_touches_roll5_mean" in df.columns:
            new_cols["workload_trend"] = df["total_touches"] - df["total_touches_roll5_mean"]
        # Goal line carries (inside 5-yard line)
        if "goal_line_carries" in df.columns:
            new_cols["goal_line_carry_rate"] = safe_divide(
                df["goal_line_carries"], df["rushing_attempts"]
            )
        elif "rush_inside_10" in df.columns:
            new_cols["goal_line_carry_rate"] = safe_divide(
                df["rush_inside_10"], df["rushing_attempts"]
            )
        # Red zone carries and targets
        if "redzone_carries" in df.columns:
            new_cols["rz_carry_rate"] = safe_divide(df["redzone_carries"], df["rushing_attempts"])
        if "redzone_targets" in df.columns:
            new_cols["rz_target_rate"] = safe_divide(df["redzone_targets"], df["targets"])
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
    
    def _create_wr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """WR-specific features per requirements: targets, aDOT, YAC, target share,
        red zone targets, contested catch rate, route diversity, slot vs outside."""
        new_cols = {}
        # Target quality
        new_cols["yards_per_route"] = safe_divide(
            df["receiving_yards"], df.get("routes_run", df.get("snap_count", pd.Series(1, index=df.index)))
        )

        # Contested catch proxy (low catch rate but high yards)
        new_cols["contested_proxy"] = safe_divide(
            df["receiving_yards"], df["targets"]
        ) * (1 - df["catch_rate"] / 100)

        # Deep threat indicator
        new_cols["yards_per_catch"] = safe_divide(
            df["receiving_yards"], df["receptions"]
        )

        # Red zone target rate
        if "redzone_targets" in df.columns and "targets" in df.columns:
            new_cols["rz_target_rate"] = safe_divide(df["redzone_targets"], df["targets"])

        # Slot vs outside alignment
        if "slot_pct" not in df.columns and "slot_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["slot_pct"] = safe_divide(df["slot_snaps"], df["snap_count"]) * 100
        if "outside_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["outside_pct"] = safe_divide(df["outside_snaps"], df["snap_count"]) * 100

        # Route tree diversity: number of different route types (if available)
        if "route_diversity_score" in df.columns:
            new_cols["route_tree_diversity"] = df["route_diversity_score"]

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
    
    def _create_te_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """TE-specific features per requirements: targets, receptions, receiving yards, TDs,
        air yards, aDOT, YAC, target share, red zone targets, contested catch rate,
        route tree diversity, slot vs. outside alignment."""
        new_cols = {}

        # Red zone specialist (high TD rate relative to volume)
        new_cols["rz_specialist_score"] = safe_divide(
            df["receiving_tds"], df["targets"]
        ) * 100

        # Seam threat (yards per target)
        new_cols["seam_threat"] = df.get("yards_per_target", pd.Series(0, index=df.index))

        # Red zone target rate
        if "redzone_targets" in df.columns and "targets" in df.columns:
            new_cols["te_rz_target_rate"] = safe_divide(df["redzone_targets"], df["targets"])

        # Contested catch proxy (low catch rate but high yards - same approach as WR)
        if "catch_rate" in df.columns and "receiving_yards" in df.columns and "targets" in df.columns:
            new_cols["te_contested_proxy"] = safe_divide(
                df["receiving_yards"], df["targets"]
            ) * (1 - df["catch_rate"].fillna(0) / 100)

        # aDOT / air yards per target (depth of target)
        if "air_yards" in df.columns and "targets" in df.columns:
            new_cols["te_adot"] = safe_divide(df["air_yards"], df["targets"])
        elif "air_yards_share" in df.columns:
            new_cols["te_adot"] = df["air_yards_share"]

        # Yards after catch (YAC)
        if "yards_after_catch" in df.columns:
            new_cols["te_yac"] = df["yards_after_catch"]
        elif "receiving_yards" in df.columns and "air_yards" in df.columns:
            new_cols["te_yac"] = (df["receiving_yards"] - df["air_yards"]).clip(lower=0)

        # Yards per route (efficiency per snap involvement)
        new_cols["te_yards_per_route"] = safe_divide(
            df["receiving_yards"],
            df.get("routes_run", df.get("snap_count", pd.Series(1, index=df.index)))
        )

        # Route participation rate
        if "routes_run" in df.columns and "snap_count" in df.columns:
            new_cols["te_route_participation"] = safe_divide(df["routes_run"], df["snap_count"]) * 100
        elif "route_participation" in df.columns:
            new_cols["te_route_participation"] = df["route_participation"]

        # Inline blocking rate proxy: snaps NOT running routes as % of total snaps
        if "routes_run" in df.columns and "snap_count" in df.columns:
            new_cols["te_inline_block_rate"] = (
                1.0 - safe_divide(df["routes_run"], df["snap_count"])
            ).clip(0, 1) * 100

        # Slot vs. outside alignment
        if "slot_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["te_slot_pct"] = safe_divide(df["slot_snaps"], df["snap_count"]) * 100

        # Route tree diversity (when available)
        if "route_diversity_score" in df.columns:
            new_cols["te_route_diversity"] = df["route_diversity_score"]

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df


# NOTE: safe_divide is imported from src.utils.helpers - do not redefine here
