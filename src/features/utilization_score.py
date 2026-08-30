"""
Utilization Score Calculator for NFL Fantasy Football.

The Utilization Score ranges from 0-100 and measures player opportunity/usage.
Higher scores correlate with better fantasy production.

Components (per requirements): snap share, target/touch share, red zone involvement,
and high-value touch rate (rushes inside 10-yard line, targets 15+ air yards) when
PBP-derived data is available. The high_value_touch component is weighted in
UTILIZATION_WEIGHTS and computed by _add_high_value_touch_rate().

Position-specific benchmarks (PPR scoring):
- RB 60-69: ~12.2 PPG, 70%+ finish as RB2/RB3
- RB 70-79: ~15.1 PPG, strong RB2 upside
- RB 80+: Elite usage, RB1 potential

The methodology weights different opportunity metrics by position.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import UTILIZATION_WEIGHTS
from src.utils.helpers import safe_divide

# "preserve" keeps an unknown snap share as NaN so the persisted imputation
# below can fill it and snap_share_pct_roll3_known can flag it. "zero" is the
# pre-2026-08-19 behaviour, retained for A/B and ablation reruns: it routes
# through safe_divide, which collapses unknown to 0.0 -- telling the model
# the player took 0% of snaps.
#
# DEFAULT IS "zero" because the pre-registered A/B REJECTED the alternative.
# With "preserve" plus position x era median imputation, the 1,518 rows with
# incomplete snap history got WORSE (+0.036 MAE, p=6.9e-04) while the 20,279
# complete rows improved -- the inverse of a genuine mechanism fix. Unknown-
# snap players are 55-65% of a known player's usage, so a median overstates
# them; the fabricated zero is wrong in principle but a closer predictive
# proxy for this particular missingness. See GAPS.md 2026-08-19.
#
# "preserve" is retained as working, tested infrastructure, not as a
# deprecated branch. Flip via set_snap_missingness_mode().
SNAP_MISSINGNESS_MODE = "zero"

# First season with any snap data at all: `snap_counts` starts here, and
# player_weekly_stats carries NULL (previously a fabricated 0.0) before it.
#
# This boundary is deliberately NOT governed by SNAP_MISSINGNESS_MODE, because
# the two kinds of missingness are different populations and the A/B above only
# settles one of them. For 2013+, a missing snap row is INFORMATIVE -- the
# player is a low-usage player, 55-65% of a known player's usage -- which is
# why the fabricated zero beat median imputation and why the mode stays "zero".
# Before 2013 the sensor simply did not exist, so missingness is STRUCTURAL and
# carries no signal about the player: zero understates every starter in seven
# seasons and a median invents a measurement for an era that has none. Those
# rows are held as NaN for LightGBM's missing-aware splits regardless of mode.
SNAP_DATA_START_SEASON = 2013


def set_snap_missingness_mode(mode: str) -> None:
    global SNAP_MISSINGNESS_MODE
    if mode not in ("zero", "preserve"):
        raise ValueError(f"mode must be 'zero' or 'preserve', got {mode!r}")
    SNAP_MISSINGNESS_MODE = mode


def snap_share_pct(snap_count, team_snaps):
    """Percentage of team snaps, honouring SNAP_MISSINGNESS_MODE.

    `safe_divide` returns its default (0.0) when either operand is NaN, which
    is the single upstream source of every fabricated zero in
    snap_share_pct_roll3_mean and snap_share_accel.
    """
    if SNAP_MISSINGNESS_MODE == "zero":
        return safe_divide(snap_count, team_snaps) * 100
    numerator = pd.to_numeric(snap_count, errors="coerce")
    denominator = pd.to_numeric(team_snaps, errors="coerce")
    return (numerator / denominator * 100).where(denominator > 0)

logger = logging.getLogger(__name__)


def _bounds_key_to_str(key: Tuple[str, str]) -> str:
    """Serialize (position, component) for JSON."""
    return f"{key[0]}|{key[1]}"


def _bounds_str_to_key(s: str) -> Tuple[str, str]:
    """Deserialize JSON key to (position, component)."""
    a, b = s.split("|", 1)
    return (a, b)


# --------------------------------------------------------------------------
# Snap-share imputation: a learned, train-only feature transformation.
#
# Deliberately mirrors the percentile-bounds lifecycle below (fit on train ->
# persist with train_seasons -> validate on load -> apply unchanged) rather
# than living inside ComponentPredictor. It is feature semantics, not model
# preprocessing: an inference path that bypasses the estimator must still get
# the same transformation.
#
# Kept as a SIBLING artifact (snap_imputation.json) rather than a section of
# the bounds file -- same contract, unrelated transformation, and it cannot
# invalidate the bounds artifact's own metadata check.
# --------------------------------------------------------------------------

SNAP_ROLL3_COL = "snap_share_pct_roll3_mean"
SNAP_KNOWN_COL = "snap_share_pct_roll3_known"
SNAP_IMPUTATION_FILENAME = "snap_imputation.json"

# Columns whose NaN means "unknown" rather than "zero", and which must
# therefore survive blanket fills. A zero here is a factual claim -- that the
# player took no snaps, or that his team ran no plays -- and both are
# distinguishable from missing data in the source table.
MISSINGNESS_PRESERVED_COLS = frozenset({
    "snap_count", "team_snaps", "snap_share", "snap_share_pct",
})

# Same argument, applied to team personnel usage: a NaN in team_pct_12_personnel
# means the personnel table has no row for that team-week, NOT that the team
# lined up in 12 personnel on 0% of snaps. Filling it zero is a factual claim
# the data does not support, and it propagates: the model-facing features are
# the roll3 means derived FROM these columns
# (team_pct_{11,12,21}_personnel_roll3_mean), so the fabricated zeros are
# averaged into two live features per position.
#
# Held behind PRESERVE_PERSONNEL_MISSINGNESS (default OFF) rather than switched
# on directly. Every prior result -- Phase 2 architecture selection, Phase 3
# windows, FINAL_CONFIG, the 11-fold walk-forward -- was produced with these
# filled, and flipping the default silently would invalidate all of it while
# blending the effect into the pending v34/v35 re-baseline. Turning it on is a
# config change, so it can run as its own attributable arm.
PERSONNEL_MISSINGNESS_COLS = frozenset({
    "team_pct_11_personnel", "team_pct_12_personnel",
    "team_pct_13_personnel", "team_pct_21_personnel",
})

# Columns joined from the PRIOR week, which therefore cannot exist in week 1.
#
# get_all_players_for_training joins team and opponent stats on
# `ts.week = pws.week - 1` to avoid leaking the current week's outcome. There is
# no week 0, so week 1 always yields NULL -- and the blanket fill below turned
# that into a measured zero for 100% of week-1 rows.
#
# Measured 2026-08-29: `opp_fpts_allowed` is a DIRECT causal feature for all
# four positions with a mid-season median of 21.6, and it read 0.0 for every
# week-1 row. That states the opponent allows zero fantasy points -- the best
# defence possible, identical for all 32 teams -- in precisely the week where
# form data is weakest and matchup should carry the most weight.
#
# The zero also propagates: it sits inside the 3-game rolling means for weeks
# 2-4, where 25-33% of team_pace_sec_per_play_roll3_mean rows came out below
# 20 s/play against a true p1 of 24.75.
#
# DEFAULT ON, unlike PERSONNEL_MISSINGNESS_COLS above. That set is gated off to
# stay comparable with earlier results, and 0% personnel usage is at least an
# arguable reading. Neither applies here: "allows 0.0 fantasy points" is a join
# miss with no defensible reading, and the results it would stay comparable with
# were already superseded by the label, horizon and backfill fixes of the same
# day. Set PRESERVE_PRIOR_WEEK_MISSINGNESS=0 to restore the old behaviour.
PRIOR_WEEK_JOIN_COLS = frozenset({
    "opp_fpts_allowed",
    "team_pace_sec_per_play",
    "team_neutral_pass_rate",
    "team_neutral_pass_rate_oe",
    "team_plays",
    # The SOURCE columns behind opp_fpts_allowed. It is assembled in
    # feature_engineering (`df.loc[mask, "opp_fpts_allowed"] = ...` per
    # position) from these four, so exempting only the assembled name left the
    # inputs to be zero-filled and the output was still 0 on 100% of week-1
    # rows. Verified: straight out of SQL these are 100% NaN at week 1, so the
    # zeros were entirely the blanket fill's doing.
    #
    # This is the scoping trap in this codebase: a column can feed a causal
    # feature under a DIFFERENT NAME, so checking CAUSAL_FEATURES membership by
    # name (or by `<name>_*` prefix) misses it. Same shape as the
    # team_pct_12_personnel -> ..._roll3_mean case already recorded above.
    "fantasy_points_allowed_qb",
    "fantasy_points_allowed_rb",
    "fantasy_points_allowed_wr",
    "fantasy_points_allowed_te",
})


def missingness_preserved_cols() -> frozenset:
    """Columns exempt from the blanket numeric fill, resolved at CALL time.

    Read live rather than captured at import: GAPS.md 7.7/7.8 records that
    monkeypatching module-level constants in this codebase is unreliable, so a
    module-level union would make the toggle untestable and easy to mis-set.
    """
    from config.settings import (
        PRESERVE_PERSONNEL_MISSINGNESS,
        PRESERVE_PRIOR_WEEK_MISSINGNESS,
    )

    out = MISSINGNESS_PRESERVED_COLS
    if PRESERVE_PERSONNEL_MISSINGNESS:
        out = out | PERSONNEL_MISSINGNESS_COLS
    if PRESERVE_PRIOR_WEEK_MISSINGNESS:
        out = out | PRIOR_WEEK_JOIN_COLS
    return out

# Fallback keys, used when a fold's training data has no rows for a given
# position/era combination.
_ALL_ERAS = "all"
_GLOBAL = "__global__"


def snap_era(seasons) -> "pd.Series":
    """The snap-data regime boundary, not arbitrary year buckets.

    Pre-2018 is the era whose snap coverage was reconstructed from the PFR
    feed and therefore carries most of the residual missingness.
    """
    return pd.Series(
        np.where(pd.to_numeric(seasons, errors="coerce") < 2018, "pre2018", "post2018"),
        index=getattr(seasons, "index", None),
    )


def fit_snap_imputation(train_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Position x era medians of the rolling snap feature, from TRAIN rows only.

    Also stores per-position and global fallbacks so a fold whose training
    data lacks an era still resolves to a real number rather than silently
    reintroducing a zero.
    """
    values: Dict[Tuple[str, str], float] = {}
    if train_df.empty or SNAP_ROLL3_COL not in train_df.columns:
        return values

    df = train_df[[c for c in ("position", "season", SNAP_ROLL3_COL)
                   if c in train_df.columns]].copy()
    df["_v"] = pd.to_numeric(df[SNAP_ROLL3_COL], errors="coerce")
    df = df.dropna(subset=["_v"])
    if df.empty:
        return values

    if "season" in df.columns:
        df["_era"] = snap_era(df["season"]).values
    else:
        df["_era"] = _ALL_ERAS

    if "position" in df.columns:
        for (pos, era), grp in df.groupby(["position", "_era"]):
            values[(str(pos), str(era))] = float(grp["_v"].median())
        for pos, grp in df.groupby("position"):
            values[(str(pos), _ALL_ERAS)] = float(grp["_v"].median())

    # Global-per-era as well as global-overall, so era granularity survives
    # even on a frame carrying no position column.
    for era, grp in df.groupby("_era"):
        values[(_GLOBAL, str(era))] = float(grp["_v"].median())
    values[(_GLOBAL, _ALL_ERAS)] = float(df["_v"].median())
    return values


def save_snap_imputation(values: Dict[Tuple[str, str], float], path: Path,
                         metadata: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {_bounds_key_to_str(k): float(v) for k, v in values.items()}
    if metadata:
        out["__meta__"] = metadata
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_snap_imputation(path: Path, return_meta: bool = False):
    path = Path(path)
    if not path.exists():
        return ({}, {}) if return_meta else {}
    with open(path) as f:
        raw = json.load(f)
    meta = raw.get("__meta__", {}) if isinstance(raw, dict) else {}
    values = {_bounds_str_to_key(k): float(v)
              for k, v in raw.items() if k != "__meta__"}
    return (values, meta) if return_meta else values


def validate_snap_imputation_meta(metadata: Dict,
                                  expected_train_seasons: Optional[list]) -> bool:
    """Same contract as validate_percentile_bounds_meta: an artifact fitted on
    different seasons than the current training set must be refused."""
    if not expected_train_seasons:
        return True
    if not metadata:
        return False
    seasons = metadata.get("train_seasons")
    if not isinstance(seasons, list) or not seasons:
        return False
    return set(seasons) == set(expected_train_seasons)


def apply_snap_imputation(df: pd.DataFrame,
                          values: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    """Fill unknown rolling snap shares from PERSISTED values.

    Never computes a statistic from `df` -- that is the whole point. Passing
    an empty mapping leaves the frame untouched, so a missing artifact fails
    visibly upstream instead of quietly imputing from the inference
    population.
    """
    if df.empty or not values or SNAP_ROLL3_COL not in df.columns:
        return df

    df = df.copy()
    current = pd.to_numeric(df[SNAP_ROLL3_COL], errors="coerce")
    if current.notna().all():
        return df

    # Pre-2013 rows are exempt: there is no snap measurement for that era, so
    # filling them from a position x era median invents one -- and `snap_era`
    # splits at 2018, so the "pre2018" bucket would blend real 2013-2017
    # behaviour into seasons that have none. Left NaN for the model to split on.
    if "season" in df.columns:
        pre_snap_era = pd.to_numeric(df["season"], errors="coerce") < SNAP_DATA_START_SEASON
    else:
        pre_snap_era = pd.Series(False, index=df.index)

    era = (snap_era(df["season"]) if "season" in df.columns
           else pd.Series(_ALL_ERAS, index=df.index))
    era.index = df.index
    position = (df["position"].astype(str) if "position" in df.columns
                else pd.Series(_GLOBAL, index=df.index))

    # Fallback chain, most specific first: position+era -> position -> era ->
    # global. Each step widens the population rather than jumping straight to
    # a single number.
    global_default = values.get((_GLOBAL, _ALL_ERAS))
    fill = [
        values.get(
            (p, e),
            values.get(
                (p, _ALL_ERAS),
                values.get((_GLOBAL, e), global_default),
            ),
        )
        for p, e in zip(position, era)
    ]
    filled = current.fillna(pd.Series(fill, index=df.index))
    df[SNAP_ROLL3_COL] = filled.where(~pre_snap_era, current)
    return df


def save_percentile_bounds(
    position_percentiles: Dict[Tuple[str, str], Tuple[float, float]],
    path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """Persist percentile bounds (train-only) for use at test/serve.

    Keys (position, col) -> (lo, hi). Optional metadata (e.g., train seasons)
    is stored under "__meta__" for leakage auditing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {_bounds_key_to_str(k): list(v) for k, v in position_percentiles.items()}
    if metadata:
        out["__meta__"] = metadata
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_percentile_bounds(path: Path, return_meta: bool = False):
    """Load percentile bounds from file.

    Returns dict (position, col) -> (lo, hi). If return_meta=True, returns
    (bounds, metadata).
    """
    path = Path(path)
    if not path.exists():
        return ({}, {}) if return_meta else {}
    with open(path) as f:
        raw = json.load(f)
    meta = raw.get("__meta__", {}) if isinstance(raw, dict) else {}
    bounds = {
        _bounds_str_to_key(k): (float(v[0]), float(v[1]))
        for k, v in raw.items()
        if k != "__meta__"
    }
    return (bounds, meta) if return_meta else bounds


def validate_percentile_bounds_meta(metadata: Dict, expected_train_seasons: Optional[list]) -> bool:
    """Validate bounds metadata against expected training seasons."""
    if not expected_train_seasons:
        return True
    if not metadata:
        return False
    seasons = metadata.get("train_seasons")
    if not isinstance(seasons, list) or not seasons:
        return False
    return set(seasons) == set(expected_train_seasons)


class UtilizationScoreCalculator:
    """Calculate Utilization Scores for NFL players by position."""
    
    def __init__(self, weights: Optional[Dict] = None, position_percentiles: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None):
        self.weights = weights if weights is not None else UTILIZATION_WEIGHTS
        self.position_percentiles = dict(position_percentiles) if position_percentiles is not None else {}
    
    def calculate_all_scores(self, player_df: pd.DataFrame, 
                             team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate utilization scores for all players.
        
        Args:
            player_df: DataFrame with player weekly stats
            team_df: DataFrame with team weekly stats
            
        Returns:
            DataFrame with utilization scores added
        """
        # Handle empty DataFrame
        if player_df.empty or "position" not in player_df.columns:
            return player_df
        
        # Merge player and team data
        merged = self._merge_player_team_data(player_df, team_df)

        # High-value touch rate (optional): requires goal-line rushes and/or deep targets.
        # This is computed once here so RB/WR/TE can consume it if weights include it.
        merged = self._add_high_value_touch_rate(merged)
        
        # Calculate position-specific utilization scores
        result_dfs = []
        
        for position in ["QB", "RB", "WR", "TE"]:
            pos_df = merged[merged["position"] == position].copy()
            if len(pos_df) == 0:
                continue
            
            if position == "RB":
                pos_df = self._calculate_rb_utilization(pos_df)
            elif position == "WR":
                pos_df = self._calculate_wr_utilization(pos_df)
            elif position == "TE":
                pos_df = self._calculate_te_utilization(pos_df)
            elif position == "QB":
                pos_df = self._calculate_qb_utilization(pos_df)
            
            result_dfs.append(pos_df)
        
        if not result_dfs:
            return player_df
        
        # Filter out empty DataFrames and ensure consistent dtypes before concat
        result_dfs = [df for df in result_dfs if not df.empty and len(df) > 0]
        
        if not result_dfs:
            return player_df
        
        # Concatenate with explicit handling to avoid FutureWarning
        result = pd.concat(result_dfs, ignore_index=True, sort=False)
        
        # Add _missing indicators for rolling/lag columns before filling
        rolling_lag_cols = [c for c in result.columns
                            if ('_rolling_' in c or '_lag_' in c
                                or c in ('rolling_volatility', 'rolling_consistency'))
                            and result[c].dtype.kind in 'fc']
        for col in rolling_lag_cols:
            result[f'{col}_missing'] = result[col].isna().astype(np.int8)

        # Fill any remaining NaN values in numeric columns, EXCEPT those whose
        # NaN carries meaning. This method runs before feature engineering
        # builds snap_share_pct_roll3_mean, so zeroing snap_share_pct here
        # would erase the missingness before the rolling feature and its
        # `known` indicator are derived from it -- the imputation artifact
        # would then have nothing to fill and `known` would read 1.0
        # everywhere. Caught by the end-to-end trace test; see
        # tests/test_snap_missingness_end_to_end.py.
        preserved = missingness_preserved_cols()
        numeric_cols = [c for c in result.select_dtypes(include=[np.number]).columns
                        if c not in preserved]
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        return result

    def _add_high_value_touch_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add high_value_touch_rate (0-100) when source columns are available."""
        if df.empty:
            return df
        df = df.copy()

        rush_inside_10 = df.get("rush_inside_10", pd.Series(0, index=df.index)).fillna(0)
        targets_15_plus = df.get("targets_15_plus", pd.Series(0, index=df.index)).fillna(0)
        rush_att = df.get("rushing_attempts", pd.Series(0, index=df.index)).fillna(0)
        targets = df.get("targets", pd.Series(0, index=df.index)).fillna(0)

        denom = (rush_att + targets).replace(0, np.nan)
        rate = safe_divide((rush_inside_10 + targets_15_plus), denom) * 100
        df["high_value_touch_rate"] = rate.replace([np.inf, -np.inf], 0).fillna(0).clip(0, 100)
        return df
    
    def _merge_player_team_data(self, player_df: pd.DataFrame, 
                                 team_df: pd.DataFrame) -> pd.DataFrame:
        """Merge player stats with team totals for share calculations."""
        # Ensure we have the necessary team columns
        team_cols = ["team", "season", "week", "pass_attempts", "rush_attempts", 
                     "total_plays", "redzone_attempts"]
        
        available_cols = [c for c in team_cols if c in team_df.columns]
        
        if len(available_cols) < 3:
            # If team data is minimal, calculate from player data
            return self._calculate_team_totals_from_players(player_df)
        
        merged = player_df.merge(
            team_df[available_cols],
            on=["team", "season", "week"],
            how="left",
            suffixes=("", "_team")
        )
        
        return merged
    
    def _calculate_team_totals_from_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team totals from aggregated player data.

        team_snaps and team_rush_attempts are true team-level values already
        present on `df` whenever it's sourced from
        DatabaseManager.get_all_players_for_training() (player_weekly_stats.team_snaps
        and the team_stats-joined team_rush_attempts respectively) -- both reflect
        the whole team for that game, independent of which player rows happen to
        be in `df` after upstream filtering (min_games, filter_to_eligible_players,
        or a position-scoped query). Recomputing them from df's own rows would
        (a) under-count whenever df isn't the complete roster for that game, and
        (b) for snaps specifically, be wrong even with a complete roster: every
        snap is one play that ~11 offensive players simultaneously share credit
        for, so summing individual snap_count double/triple/N-counts the same
        play instead of reconstructing the team's real play count. Previously
        this function always recomputed both via groupby-sum and merged with no
        `suffixes=`, so when a same-named column already existed on `df` (as it
        always does in live serving/training), pandas silently split it into
        `team_snaps_x`/`team_snaps_y`, and every per-position share formula's
        `df.get("team_snaps", snap_count)` fallback then silently self-divided
        (snap_share_pct = snap_count/snap_count = 100%) instead of erroring --
        see GAPS.md, 2026-08-08. So: preserve any pre-existing team_snaps/
        team_rush_attempts untouched, and only fall back to recomputing (using
        max() for snap_count, not sum(), for the reason above) when truly absent.
        """
        # Check which columns exist for grouping and aggregation
        group_cols = []
        for col in ["team", "season", "week"]:
            if col in df.columns:
                group_cols.append(col)

        if not group_cols:
            # No grouping possible -- set team totals to NaN so that
            # downstream share calculations produce NaN rather than
            # the misleading 100% share that results from team == individual.
            import warnings
            warnings.warn(
                "Cannot compute team totals: missing team/season/week columns. "
                "Share-based utilization components will be NaN."
            )
            df = df.copy()
            for col in ["team_targets", "team_rush_attempts", "team_snaps"]:
                if col not in df.columns:
                    df[col] = np.nan
            return df

        # team_snaps/team_rush_attempts: preserve if already present (true
        # team-level values, not a player-row sum -- see docstring). targets/
        # receptions have no pre-existing source and are correctly summed
        # (each target/reception is attributable to exactly one player, unlike
        # a shared snap), subject to the same roster-completeness caveat.
        preserve_if_present = {"team_snaps", "team_rush_attempts"}
        col_mapping = {
            "targets": "team_targets",
            "receptions": "team_receptions",
            "rushing_attempts": "team_rush_attempts",
            "snap_count": "team_snaps",
        }
        fallback_agg = {
            "snap_count": "max",
            "rushing_attempts": "sum",
            "targets": "sum",
            "receptions": "sum",
        }

        # Build aggregation dict based on available columns
        agg_dict = {}
        for col, new_name in col_mapping.items():
            if new_name in preserve_if_present and new_name in df.columns:
                continue
            if col in df.columns:
                agg_dict[col] = fallback_agg[col]

        if not agg_dict:
            # Nothing to recompute -- all target columns already present.
            return df

        team_totals = df.groupby(group_cols).agg(agg_dict).reset_index()

        # Rename columns
        rename_dict = {col: col_mapping[col] for col in agg_dict.keys()}
        team_totals = team_totals.rename(columns=rename_dict)

        # Defense-in-depth: if a target column unexpectedly already exists
        # despite the preserve-if-present check above (e.g. a future schema
        # change), fail loudly via an obviously-wrong suffixed column that
        # gets dropped-with-warning, instead of silently splitting into a
        # same-looking column that downstream code can't find by name.
        merged = df.merge(team_totals, on=group_cols, how="left",
                           suffixes=("", "_recompute_collision"))
        collision_cols = [c for c in merged.columns if c.endswith("_recompute_collision")]
        if collision_cols:
            import warnings
            warnings.warn(
                f"Unexpected team-total column collision: {collision_cols}; "
                "dropping recomputed duplicates and keeping the pre-existing values."
            )
            merged = merged.drop(columns=collision_cols)

        return merged
    
    def _calculate_rb_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RB Utilization Score.
        
        Components (Fantasy Life aligned: snap share, targets, touches):
        - Snap share, rush share, target share, red zone share, touch share (carries + receptions) / team touches.
        """
        weights = self.weights["RB"]
        
        snap_count = df.get("snap_count", pd.Series(0, index=df.index))
        rushing_attempts = df.get("rushing_attempts", pd.Series(0, index=df.index))
        targets = df.get("targets", pd.Series(0, index=df.index))
        receptions = df.get("receptions", pd.Series(0, index=df.index))
        rushing_tds = df.get("rushing_tds", pd.Series(0, index=df.index))
        receiving_tds = df.get("receiving_tds", pd.Series(0, index=df.index))
        
        team_snaps = df.get("team_snaps", snap_count)
        team_rush = df.get("team_rush_attempts", rushing_attempts)
        team_targets = df.get("team_targets", targets)
        team_receptions = df.get("team_receptions", receptions)
        
        df["snap_share_pct"] = snap_share_pct(snap_count, team_snaps)
        df["rush_share_pct"] = safe_divide(rushing_attempts, team_rush) * 100
        df["target_share_pct"] = safe_divide(targets, team_targets) * 100
        
        if "redzone_attempts" in df.columns:
            df["redzone_share_pct"] = safe_divide(
                df.get("redzone_touches", rushing_tds + receiving_tds),
                df["redzone_attempts"]
            ) * 100
        else:
            df["redzone_share_pct"] = ((rushing_tds + receiving_tds) * 10).clip(0, 100)
        
        # Touch share (Fantasy Life): (carries + receptions) / team touches
        player_touches = rushing_attempts + receptions
        team_touches = team_rush + team_receptions
        df["touch_share_pct"] = safe_divide(player_touches, team_touches) * 100
        
        df["snap_share_norm"] = self._percentile_normalize(df["snap_share_pct"], position="RB", component_key="snap_share_pct")
        df["rush_share_norm"] = self._percentile_normalize(df["rush_share_pct"], position="RB", component_key="rush_share_pct")
        df["target_share_norm"] = self._percentile_normalize(df["target_share_pct"], position="RB", component_key="target_share_pct")
        df["redzone_share_norm"] = self._percentile_normalize(df["redzone_share_pct"], position="RB", component_key="redzone_share_pct")
        df["touch_share_norm"] = self._percentile_normalize(df["touch_share_pct"], position="RB", component_key="touch_share_pct")
        
        w = weights
        score = (
            df["snap_share_norm"] * w.get("snap_share", 0.20) +
            df["rush_share_norm"] * w.get("rush_share", 0.25) +
            df["target_share_norm"] * w.get("target_share", 0.20) +
            df["redzone_share_norm"] * w.get("redzone_share", 0.20) +
            df["touch_share_norm"] * w.get("touch_share", 0.15)
        )
        # Optional: high-value touch (rushes inside 10, targets 15+ air yards) when data and weight available
        if w.get("high_value_touch", 0) > 0 and "high_value_touch_rate" in df.columns:
            hv_norm = self._percentile_normalize(df["high_value_touch_rate"], position="RB", component_key="high_value_touch_rate")
            score = score + hv_norm * w["high_value_touch"]
        df["utilization_score"] = score.clip(0, 100)
        
        df["util_snap_share"] = df["snap_share_pct"]
        df["util_rush_share"] = df["rush_share_pct"]
        df["util_target_share"] = df["target_share_pct"]
        df["util_redzone_share"] = df["redzone_share_pct"]
        
        return df
    
    def _calculate_wr_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WR Utilization Score.
        
        Components:
        - Target share (30%): % of team targets
        - Air yards share (25%): % of team air yards
        - Snap share (15%): % of team offensive snaps
        - Red zone targets (20%): Red zone target involvement
        - Route participation (10%): Routes run / team pass plays
        """
        weights = self.weights["WR"]
        
        # Calculate component metrics
        df["target_share_pct"] = safe_divide(
            df["targets"], df.get("team_targets", df["targets"])
        ) * 100
        
        # Air yards share (estimate from receiving yards if not available)
        if "air_yards" in df.columns:
            df["air_yards_share_pct"] = safe_divide(
                df["air_yards"], df.get("team_air_yards", df["air_yards"])
            ) * 100
        else:
            # Estimate from yards per target
            yards_per_target = safe_divide(df["receiving_yards"], df["targets"])
            df["air_yards_share_pct"] = df["target_share_pct"] * (yards_per_target / 10)
        
        df["snap_share_pct"] = snap_share_pct(
            df["snap_count"], df.get("team_snaps", df["snap_count"])
        )
        
        # Red zone targets: use PBP-derived data when available, else TD-based proxy
        if "redzone_targets" in df.columns and "team_redzone_targets" in df.columns:
            df["redzone_targets_pct"] = safe_divide(
                df["redzone_targets"], df["team_redzone_targets"]
            ).clip(0, 1) * 100
        elif "redzone_targets" in df.columns:
            # Normalize against a reasonable max (e.g., 3 RZ targets/game is high for WR)
            df["redzone_targets_pct"] = (df["redzone_targets"] / 3.0 * 100).clip(0, 100)
        else:
            # Fallback: TD-based proxy (documented limitation)
            df["redzone_targets_pct"] = (df["receiving_tds"] * 15).clip(0, 100)
        
        # Route participation: use actual routes when available (Fantasy Life,
        # never populated by any current loader -- see GAPS.md §11.1.C/D),
        # else the PBP-derived pass-play participation rate when available
        # (real data, but NOT true route participation -- can't separate
        # route-runners from in-line pass-blockers on the same play; see
        # get_pass_play_participation_from_pbp docstring), else the flat
        # snap-share proxy.
        if "routes_run" in df.columns and "team_routes" in df.columns:
            df["route_participation_pct"] = safe_divide(df["routes_run"], df["team_routes"]) * 100
        elif "routes_run" in df.columns and "team_snaps" in df.columns:
            df["route_participation_pct"] = safe_divide(df["routes_run"], df["team_snaps"]) * 100
        elif "pbp_pass_play_participation_pct_roll3_mean" in df.columns:
            df["route_participation_pct"] = (df["pbp_pass_play_participation_pct_roll3_mean"] * 100).clip(0, 100)
        elif "pbp_pass_play_participation_pct" in df.columns:
            df["route_participation_pct"] = (df["pbp_pass_play_participation_pct"] * 100).clip(0, 100)
        else:
            df["route_participation_pct"] = (df["snap_share_pct"] * 0.8).clip(0, 100)
        
        # Normalize components (use fitted bounds when set to avoid leakage)
        df["target_share_norm"] = self._percentile_normalize(df["target_share_pct"], position="WR", component_key="target_share_pct")
        df["air_yards_norm"] = self._percentile_normalize(df["air_yards_share_pct"], position="WR", component_key="air_yards_share_pct")
        df["snap_share_norm"] = self._percentile_normalize(df["snap_share_pct"], position="WR", component_key="snap_share_pct")
        df["redzone_targets_norm"] = self._percentile_normalize(df["redzone_targets_pct"], position="WR", component_key="redzone_targets_pct")
        df["route_part_norm"] = self._percentile_normalize(df["route_participation_pct"], position="WR", component_key="route_participation_pct")
        
        # Final utilization score; optional high_value_touch (targets 15+ air yards) when weight > 0
        score = (
            df["target_share_norm"] * weights["target_share"] +
            df["air_yards_norm"] * weights["air_yards_share"] +
            df["snap_share_norm"] * weights["snap_share"] +
            df["redzone_targets_norm"] * weights["redzone_targets"] +
            df["route_part_norm"] * weights["route_participation"]
        )
        if weights.get("high_value_touch", 0) > 0 and "high_value_touch_rate" in df.columns:
            hv_norm = self._percentile_normalize(df["high_value_touch_rate"], position="WR", component_key="high_value_touch_rate")
            score = score + hv_norm * weights["high_value_touch"]
        df["utilization_score"] = score.clip(0, 100)
        
        # Store component values
        df["util_target_share"] = df["target_share_pct"]
        df["util_air_yards_share"] = df["air_yards_share_pct"]
        df["util_snap_share"] = df["snap_share_pct"]
        
        return df
    
    def _calculate_te_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TE Utilization Score.
        
        Components:
        - Target share (30%): % of team targets
        - Snap share (20%): % of team offensive snaps
        - Red zone targets (25%): Red zone involvement
        - Air yards share (15%): % of team air yards
        - Inline rate (10%): Usage as inline blocker vs slot
        """
        weights = self.weights["TE"]
        
        # Calculate component metrics
        df["target_share_pct"] = safe_divide(
            df["targets"], df.get("team_targets", df["targets"])
        ) * 100
        
        df["snap_share_pct"] = snap_share_pct(
            df["snap_count"], df.get("team_snaps", df["snap_count"])
        )
        
        # Red zone targets: use PBP-derived data when available, else TD-based proxy
        if "redzone_targets" in df.columns and "team_redzone_targets" in df.columns:
            df["redzone_targets_pct"] = safe_divide(
                df["redzone_targets"], df["team_redzone_targets"]
            ).clip(0, 1) * 100
        elif "redzone_targets" in df.columns:
            # TEs typically see 1-2 RZ targets/game at most
            df["redzone_targets_pct"] = (df["redzone_targets"] / 2.0 * 100).clip(0, 100)
        else:
            # Fallback: TD-based proxy (documented limitation; TEs are valuable in RZ)
            df["redzone_targets_pct"] = (df["receiving_tds"] * 20).clip(0, 100)
        
        # Air yards share
        if "air_yards" in df.columns:
            df["air_yards_share_pct"] = safe_divide(
                df["air_yards"], df.get("team_air_yards", df["air_yards"])
            ) * 100
        else:
            yards_per_target = safe_divide(df["receiving_yards"], df["targets"])
            df["air_yards_share_pct"] = df["target_share_pct"] * (yards_per_target / 8)
        
        # Inline rate (estimate - higher snap share with lower target share = more blocking)
        snap_to_target_ratio = safe_divide(df["snap_share_pct"], df["target_share_pct"] + 1)
        df["inline_rate_pct"] = (100 - snap_to_target_ratio * 10).clip(0, 100)
        
        # Normalize components (use fitted bounds when set to avoid leakage)
        df["target_share_norm"] = self._percentile_normalize(df["target_share_pct"], position="TE", component_key="target_share_pct")
        df["snap_share_norm"] = self._percentile_normalize(df["snap_share_pct"], position="TE", component_key="snap_share_pct")
        df["redzone_targets_norm"] = self._percentile_normalize(df["redzone_targets_pct"], position="TE", component_key="redzone_targets_pct")
        df["air_yards_norm"] = self._percentile_normalize(df["air_yards_share_pct"], position="TE", component_key="air_yards_share_pct")
        df["inline_rate_norm"] = self._percentile_normalize(df["inline_rate_pct"], position="TE", component_key="inline_rate_pct")
        
        # Final utilization score; optional high_value_touch when weight > 0
        score = (
            df["target_share_norm"] * weights["target_share"] +
            df["snap_share_norm"] * weights["snap_share"] +
            df["redzone_targets_norm"] * weights["redzone_targets"] +
            df["air_yards_norm"] * weights["air_yards_share"] +
            df["inline_rate_norm"] * weights["inline_rate"]
        )
        if weights.get("high_value_touch", 0) > 0 and "high_value_touch_rate" in df.columns:
            hv_norm = self._percentile_normalize(df["high_value_touch_rate"], position="TE", component_key="high_value_touch_rate")
            score = score + hv_norm * weights["high_value_touch"]
        df["utilization_score"] = score.clip(0, 100)
        
        # Store component values
        df["util_target_share"] = df["target_share_pct"]
        df["util_snap_share"] = df["snap_share_pct"]
        
        return df
    
    def _calculate_qb_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate QB Utilization Score.
        
        Components:
        - Dropback rate (25%): Pass attempts relative to team plays
        - Rush attempt share (20%): Designed runs and scrambles
        - Red zone opportunity (25%): Red zone play involvement
        - Play volume (30%): Total plays (pass + rush)
        """
        weights = self.weights["QB"]
        
        # Calculate component metrics
        total_plays = df["passing_attempts"] + df["rushing_attempts"]
        team_plays = df.get("team_plays", total_plays)
        
        df["dropback_rate_pct"] = safe_divide(
            df["passing_attempts"], team_plays
        ) * 100
        
        df["rush_share_pct"] = safe_divide(
            df["rushing_attempts"], df["rushing_attempts"] + 5  # Normalize for QBs
        ) * 100
        
        # Red zone opportunity (from TDs)
        total_tds = df["passing_tds"] + df["rushing_tds"]
        df["redzone_opp_pct"] = (total_tds * 12).clip(0, 100)
        
        # Play volume (total plays normalized)
        df["play_volume_pct"] = (total_plays / 50 * 100).clip(0, 100)
        
        # Normalize components (use fitted bounds when set to avoid leakage)
        df["dropback_norm"] = self._percentile_normalize(df["dropback_rate_pct"], position="QB", component_key="dropback_rate_pct")
        df["rush_share_norm"] = self._percentile_normalize(df["rush_share_pct"], position="QB", component_key="rush_share_pct")
        df["redzone_opp_norm"] = self._percentile_normalize(df["redzone_opp_pct"], position="QB", component_key="redzone_opp_pct")
        df["play_volume_norm"] = self._percentile_normalize(df["play_volume_pct"], position="QB", component_key="play_volume_pct")
        
        # Calculate final utilization score
        df["utilization_score"] = (
            df["dropback_norm"] * weights["dropback_rate"] +
            df["rush_share_norm"] * weights["rush_attempt_share"] +
            df["redzone_opp_norm"] * weights["redzone_opportunity"] +
            df["play_volume_norm"] * weights["play_volume"]
        )
        
        # Store component values
        df["util_dropback_rate"] = df["dropback_rate_pct"]
        df["util_rush_share"] = df["rush_share_pct"]
        
        return df
    
    # NOTE: safe_divide is imported from src.utils.helpers
    
    def _percentile_normalize(self, series: pd.Series, position: str = None, component_key: str = None) -> pd.Series:
        """
        Normalize a series to 0-100. If bounds were fitted (fit_percentile_bounds), use them to avoid leakage.
        Otherwise use rank-based percentile within current data (legacy).
        """
        if series.isna().all() or len(series) == 0:
            return series
        # Auto-load persisted bounds if none in memory
        self._ensure_bounds_loaded()
        key = (position, component_key) if (position and component_key) else None
        bounds = self.position_percentiles.get(key) if key else None
        if bounds is not None and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lo, hi = bounds
            if hi > lo:
                return ((series - lo) / (hi - lo) * 100).clip(0, 100)
            # Zero-width bounds: training data was constant (likely all zeros
            # or missing data). Return neutral score; do NOT rank within
            # current data — that leaks test-set information.
            return pd.Series(50.0, index=series.index)
        return series.rank(pct=True, na_option="bottom") * 100

    _BOUNDS_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "utilization_percentile_bounds.json"

    def fit_percentile_bounds(self, train_df: pd.DataFrame, position: str, component_columns: list,
                               persist: bool = True, metadata: Optional[Dict] = None) -> None:
        """
        Fit min/max (or 1st/99th percentile) per component on train data for consistent apply at serve.
        Store in self.position_percentiles keyed by (position, col).
        
        When persist=True (default), auto-saves bounds to disk so that the
        prediction pipeline can load them without retraining.
        """
        pos_df = train_df[train_df["position"] == position]
        if pos_df.empty:
            return
        for col in component_columns:
            if col not in pos_df.columns:
                continue
            s = pos_df[col].dropna()
            if len(s) < 10:
                continue
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            if lo == hi:
                if col.endswith("_pct") or "rate" in col:
                    logger.warning(
                        "Zero-width percentile bounds for %s|%s: lo=hi=%.4f "
                        "(likely missing data). Expanding to default range [0.0, 100.0].",
                        position, col, lo,
                    )
                    lo, hi = 0.0, 100.0
                else:
                    logger.warning(
                        "Zero-width percentile bounds for %s|%s: lo=hi=%.4f "
                        "(likely missing data).",
                        position, col, lo,
                    )
            self.position_percentiles[(position, col)] = (float(lo), float(hi))
        
        if persist:
            save_percentile_bounds(self.position_percentiles, self._BOUNDS_DEFAULT_PATH, metadata=metadata)

    def _ensure_bounds_loaded(self) -> None:
        """Auto-load persisted percentile bounds if none are in memory."""
        if not self.position_percentiles and self._BOUNDS_DEFAULT_PATH.exists():
            self.position_percentiles = load_percentile_bounds(self._BOUNDS_DEFAULT_PATH)
            # Warn about zero-width bounds so issues are visible in logs
            for (pos, col), (lo, hi) in self.position_percentiles.items():
                if lo == hi:
                    logger.warning(
                        "Loaded zero-width percentile bounds for %s|%s: "
                        "lo=hi=%.4f. Neutral score (50.0) will be used.",
                        pos, col, lo,
                    )
    
    def get_utilization_tier(self, score: float, position: str) -> str:
        """
        Get the utilization tier description for a score.
        
        Returns tier like "Elite", "Strong", "Average", "Below Average", "Low"
        """
        if score >= 80:
            return "Elite"
        elif score >= 70:
            return "Strong"
        elif score >= 60:
            return "Average"
        elif score >= 50:
            return "Below Average"
        else:
            return "Low"
    
    def get_expected_ppg_range(self, score: float, position: str) -> Dict[str, float]:
        """
        Get expected PPG range based on utilization score and position.
        
        Based on historical data:
        - RB 60-69: ~12.2 PPG
        - RB 70-79: ~15.1 PPG
        - RB 80+: ~18+ PPG
        """
        ppg_ranges = {
            "RB": {
                (0, 50): {"min": 3.0, "avg": 6.5, "max": 10.0},
                (50, 60): {"min": 6.0, "avg": 9.5, "max": 13.0},
                (60, 70): {"min": 9.0, "avg": 12.2, "max": 16.0},
                (70, 80): {"min": 12.0, "avg": 15.1, "max": 20.0},
                (80, 100): {"min": 15.0, "avg": 18.5, "max": 28.0},
            },
            "WR": {
                (0, 50): {"min": 2.0, "avg": 5.0, "max": 9.0},
                (50, 60): {"min": 5.0, "avg": 8.0, "max": 12.0},
                (60, 70): {"min": 8.0, "avg": 11.0, "max": 15.0},
                (70, 80): {"min": 11.0, "avg": 14.5, "max": 19.0},
                (80, 100): {"min": 14.0, "avg": 18.0, "max": 26.0},
            },
            "TE": {
                (0, 50): {"min": 1.5, "avg": 4.0, "max": 7.0},
                (50, 60): {"min": 4.0, "avg": 6.5, "max": 10.0},
                (60, 70): {"min": 6.0, "avg": 9.0, "max": 13.0},
                (70, 80): {"min": 9.0, "avg": 12.0, "max": 17.0},
                (80, 100): {"min": 12.0, "avg": 16.0, "max": 24.0},
            },
            "QB": {
                (0, 50): {"min": 8.0, "avg": 12.0, "max": 16.0},
                (50, 60): {"min": 12.0, "avg": 15.0, "max": 19.0},
                (60, 70): {"min": 15.0, "avg": 18.0, "max": 23.0},
                (70, 80): {"min": 18.0, "avg": 21.0, "max": 27.0},
                (80, 100): {"min": 21.0, "avg": 25.0, "max": 35.0},
            },
        }
        
        position_ranges = ppg_ranges.get(position, ppg_ranges["RB"])
        
        for (low, high), ppg in position_ranges.items():
            if low <= score < high:
                return ppg
        
        return position_ranges[(80, 100)]  # Default to highest tier


def calculate_utilization_scores(player_df: pd.DataFrame, 
                                  team_df: pd.DataFrame = None,
                                  weights: Optional[Dict] = None,
                                  percentile_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    Convenience function to calculate utilization scores.
    
    Args:
        player_df: DataFrame with player weekly stats
        team_df: Optional DataFrame with team stats
        weights: Optional position -> component -> weight dict (from utilization_weight_optimizer)
        percentile_bounds: Optional (position, component_col) -> (lo, hi) from train (avoids leakage at test/serve)
        
    Returns:
        DataFrame with utilization_score column added
    """
    calculator = UtilizationScoreCalculator(weights=weights, position_percentiles=percentile_bounds)
    
    if team_df is None:
        team_df = pd.DataFrame()
    
    return calculator.calculate_all_scores(player_df, team_df)


def compute_raw_utilization_score(df: pd.DataFrame,
                                   weights: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """
    Compute utilization_score_raw from raw _pct columns (no percentile normalization).

    Used for target derivation to decouple targets from normalization parameters.
    Same weight keys as recalculate_utilization_with_weights but maps to _pct columns.
    """
    from config.settings import UTILIZATION_WEIGHTS

    result = df.copy()
    weights = weights or UTILIZATION_WEIGHTS
    pct_to_key = {
        "RB": {"snap_share_pct": "snap_share", "rush_share_pct": "rush_share",
               "target_share_pct": "target_share", "redzone_share_pct": "redzone_share",
               "touch_share_pct": "touch_share"},
        "WR": {"target_share_pct": "target_share", "air_yards_share_pct": "air_yards_share",
               "snap_share_pct": "snap_share", "redzone_targets_pct": "redzone_targets",
               "route_participation_pct": "route_participation"},
        "TE": {"target_share_pct": "target_share", "snap_share_pct": "snap_share",
               "redzone_targets_pct": "redzone_targets", "air_yards_share_pct": "air_yards_share",
               "inline_rate_pct": "inline_rate"},
        "QB": {"dropback_rate_pct": "dropback_rate", "rush_share_pct": "rush_attempt_share",
               "redzone_opp_pct": "redzone_opportunity", "play_volume_pct": "play_volume"},
    }
    for position in ["QB", "RB", "WR", "TE"]:
        mask = result["position"] == position
        if not mask.any():
            continue
        pos_weights = weights.get(position, UTILIZATION_WEIGHTS.get(position, {}))
        mapping = pct_to_key.get(position, {})
        score = pd.Series(0.0, index=result.index)
        for pct_col, key in mapping.items():
            if pct_col in result.columns and key in pos_weights:
                score = score + result[pct_col].fillna(0) * pos_weights[key]
        result.loc[mask, "utilization_score_raw"] = score[mask].clip(0, 100)
    return result


def recalculate_utilization_with_weights(df: pd.DataFrame,
                                         weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Recompute utilization_score from existing _norm columns using new weights.
    Use when components exist but weights were optimized from data.
    """
    from config.settings import UTILIZATION_WEIGHTS
    
    result = df.copy()
    norm_to_key = {
        "RB": {"snap_share_norm": "snap_share", "rush_share_norm": "rush_share",
               "target_share_norm": "target_share", "redzone_share_norm": "redzone_share",
               "touch_share_norm": "touch_share"},
        "WR": {"target_share_norm": "target_share", "air_yards_norm": "air_yards_share",
               "snap_share_norm": "snap_share", "redzone_targets_norm": "redzone_targets",
               "route_part_norm": "route_participation"},
        "TE": {"target_share_norm": "target_share", "snap_share_norm": "snap_share",
               "redzone_targets_norm": "redzone_targets", "air_yards_norm": "air_yards_share",
               "inline_rate_norm": "inline_rate"},
        "QB": {"dropback_norm": "dropback_rate", "rush_share_norm": "rush_attempt_share",
               "redzone_opp_norm": "redzone_opportunity", "play_volume_norm": "play_volume"},
    }
    for position in ["QB", "RB", "WR", "TE"]:
        mask = result["position"] == position
        if not mask.any():
            continue
        pos_weights = weights.get(position, UTILIZATION_WEIGHTS.get(position, {}))
        mapping = norm_to_key.get(position, {})
        score = pd.Series(0.0, index=result.index)
        for norm_col, key in mapping.items():
            if norm_col in result.columns and key in pos_weights:
                score = score + result[norm_col].fillna(0) * pos_weights[key]
        result.loc[mask, "utilization_score"] = score[mask].clip(0, 100)
    return result
