"""Configuration settings for NFL predictor."""
import os
from pathlib import Path

# ── The Odds API ──────────────────────────────────────────────────────────────
# Loaded from .env — never hardcode.  Rotate via .env only.
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# ESPN fantasy-league ("pool") data: the user's own roster, lineup and
# matchups. NOT NFL data, and the published site is NFL projections only.
#
# The location is the safeguard. docs/ is the GitHub Pages root, and the UI
# fetches relative paths beneath it, so a browser cannot reach anything outside
# docs/ no matter what the page markup says. Keeping this directory under
# data/ makes displaying league data impossible rather than merely discouraged.
# The predecessor script wrote docs/data/my_roster.json -- inside the published
# tree -- which is exactly the mistake this constant exists to prevent.
#
# Anything written here is gitignored. Do not move it under docs/.
ESPN_PRIVATE_DIR = DATA_DIR / "espn_private"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ESPN_PRIVATE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "nfl_data.db"

# Scraping settings
SCRAPER_DELAY = 2.0  # Seconds between requests
SCRAPER_DELAY_JITTER = 1.5  # Max random jitter added to delay (seconds)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6; rv:122.0) Gecko/20100101 Firefox/122.0",
]
SCRAPER_CACHE_TTL_HOURS = 24  # Reuse cached HTTP responses within this window
SCRAPER_CACHE_SUBDIR = "http_cache"
from datetime import datetime

def _current_nfl_season():
    """Current NFL season (Sept-Feb); single source from nfl_calendar."""
    from src.utils.nfl_calendar import get_current_nfl_season
    return get_current_nfl_season()

# -----------------------------------------------------------------------------
# YEAR PARAMETERS (single source of truth for workflow and plan criteria)
# -----------------------------------------------------------------------------
# Earliest season to load/scrape.
# Requirements: 18-week model needs min 8, optimal 10+ seasons.
# PBP data available back to 1999; weekly player data reliable from ~2014.
# Set to 2006 to balance data volume and quality (provides ~19 seasons).
MIN_HISTORICAL_YEAR = 2006
# Earliest season nfl-data-py weekly data is considered complete (used for availability checks).
AVAILABLE_SEASONS_START_YEAR = 2016
# Current NFL season (Sept-Feb): Jan-Aug = previous year, Sept-Dec = current year.
CURRENT_YEAR = datetime.now().year
CURRENT_NFL_SEASON = _current_nfl_season()
# Default range for scraping/loading: MIN_HISTORICAL_YEAR through current NFL season (inclusive).
SEASONS_TO_SCRAPE = list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1))

# -----------------------------------------------------------------------------
# Regular-season length. The NFL played 17 weeks through 2020 and 18 from 2021,
# so week 18 is the WILD CARD ROUND in the earlier era, not a regular-season
# week. A flat 18 therefore folds one full playoff round into anything that
# means "the season" -- season totals, availability denominators, prior-season
# win counts. Measured cost when this was flat: 480 of 3,339 pre-2021
# cold-start rows carried a wild-card game (4,166 fantasy points), and 62 of
# 480 pre-2021 team-seasons had an inflated win total.
#
# Lives here rather than in season_projection.py because feature_engineering,
# the preseason models and several scripts all need it, and importing the
# projection module into the feature layer would invert the dependency.
LAST_17_WEEK_SEASON = 2020


def regular_season_max_week(season) -> int:
    """Last regular-season week for `season`: 17 through 2020, 18 from 2021."""
    return 17 if int(season) <= LAST_17_WEEK_SEASON else 18


def regular_season_week_sql(week_col: str = "week", season_col: str = "season") -> str:
    """SQL predicate for "this row is a regular-season week".

    Written as a fragment rather than a parameter so callers that aggregate
    across many seasons in one query stay correct per-row. Interpolating table
    aliases is safe here: both arguments are column names supplied by the
    calling module, never user input.
    """
    return (f"{week_col} <= (CASE WHEN {season_col} <= {LAST_17_WEEK_SEASON} "
            f"THEN 17 ELSE 18 END)")

# -----------------------------------------------------------------------------
# PBP ADVANCED FEATURE SETTINGS
# -----------------------------------------------------------------------------
# Enable advanced PBP-derived features (EPA/WPA/success, neutral pass rate, drive metrics).
PBP_ADVANCED_FEATURES_ENABLED = True
# Seasons to compute advanced PBP features for (all since MIN_HISTORICAL_YEAR).
PBP_ADVANCED_SEASONS = list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1))
# Neutral script threshold (score diff)
NEUTRAL_SCORE_DIFF = 7
# Short-yardage threshold
SHORT_YARDAGE_YDSTOGO = 2
# Red zone and goal line yardline_100 thresholds
RED_ZONE_YARDLINE = 20
GOAL_LINE_YARDLINE = 5
# Two-minute warning window (seconds)
TWO_MINUTE_SECONDS = 120
# Fallback league neutral pass rate when insufficient data
PROE_FALLBACK_LG_NEUTRAL_RATE = 0.56

# Manual position overrides: {player_id: correct_position}
# For players misclassified by nfl-data-py (e.g. rookies missing from weekly_rosters).
POSITION_OVERRIDES = {
    "00-0040126": "TE",  # Colston Loveland (CHI) — classified as WR by nfl-data-py
}

# Positions (offensive skill positions only)
POSITIONS = ["QB", "RB", "WR", "TE"]
# Offensive skill positions used by the utilization-based ML pipeline
OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE"]

# Fantasy scoring (PPR - primary)
SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "interceptions": -2,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "receptions": 1,  # PPR
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "fumbles_lost": -2,
    "two_point_conversions": 2,
}

# Half-PPR scoring (per requirements: support PPR, Half-PPR, Standard)
SCORING_HALF_PPR = {
    **SCORING,
    "receptions": 0.5,
}

# Standard (non-PPR) scoring
SCORING_STANDARD = {
    **SCORING,
    "receptions": 0,
}

# All supported scoring formats
SCORING_FORMATS = {
    "ppr": SCORING,
    "half_ppr": SCORING_HALF_PPR,
    "standard": SCORING_STANDARD,
}

# ESPN integration slot mappings
ESPN_SLOT_MAP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "RB/WR/TE": "FLEX", "RB/WR": "FLEX", "WR/TE": "FLEX",
    "OP": "SUPERFLEX", "BE": "BENCH", "IR": "IR", "K": "K", "D/ST": "DST",
}
ESPN_DEFAULT_ROSTER_SLOTS = {
    "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 6,
}

# Kicker scoring
SCORING_KICKER = {
    "fg_0_39": 3,        # FG made 0-39 yards
    "fg_40_49": 4,       # FG made 40-49 yards
    "fg_50_plus": 5,     # FG made 50+ yards
    "xp_made": 1,        # Extra point made
    "fg_missed": -1,     # FG missed
    "xp_missed": -1,     # XP missed
}

# DST scoring
SCORING_DST = {
    "sack": 1,
    "interception": 2,
    "fumble_recovery": 2,
    "safety": 2,
    "defensive_td": 6,
    "special_teams_td": 6,
    "blocked_kick": 2,
    # Points allowed brackets (bonus/penalty)
    "pa_0": 10,          # shutout
    "pa_1_6": 7,
    "pa_7_13": 4,
    "pa_14_20": 1,
    "pa_21_27": 0,
    "pa_28_34": -1,
    "pa_35_plus": -4,
}

# Utilization Score weights by position
# When goal-line and aDOT/air-yards data are available, set high_value_touch weight and
# compute high_value_touch_rate (rushes inside 10, targets 15+ air yards) in utilization_score.
UTILIZATION_WEIGHTS = {
    "RB": {
        "snap_share": 0.20,
        "rush_share": 0.25,
        "target_share": 0.20,
        "redzone_share": 0.20,
        "touch_share": 0.10,  # (carries + receptions) / team touches, Fantasy Life aligned
        "high_value_touch": 0.05,  # rushes inside 10, high-value target share
    },
    "WR": {
        "target_share": 0.30,
        "air_yards_share": 0.25,
        "snap_share": 0.15,
        "redzone_targets": 0.20,
        "route_participation": 0.05,
        "high_value_touch": 0.05,  # targets 15+ air yards
    },
    "TE": {
        "target_share": 0.30,
        "snap_share": 0.20,
        "redzone_targets": 0.25,
        "air_yards_share": 0.15,
        "inline_rate": 0.05,
        "high_value_touch": 0.05,  # high-value target share
    },
    "QB": {
        "dropback_rate": 0.25,
        "rush_attempt_share": 0.20,
        "redzone_opportunity": 0.25,
        "play_volume": 0.30,
    },
}

# Huber loss delta: controls the transition point between quadratic (MSE) and
# linear loss.  Residuals below this threshold are penalized quadratically;
# above it, linearly.  A delta of 1.0 treats any error >1 point as an outlier,
# which over-suppresses predictions for boom/bust games.  Setting delta to 5.0
# (roughly one standard deviation of weekly fantasy points) preserves outlier
# robustness while giving the model meaningful gradient signal on big games.
HUBER_DELTA = 5.0

# Model settings
MODEL_CONFIG = {
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "n_optuna_trials": 100,
    "early_stopping_rounds": 25,
    "validation_pct": 0.2,       # Fraction of training data for ensemble weight optimization
    "n_features_per_position": 15,  # Max features after selection (per position); reduced from 50 to combat overfitting with ~250 samples/position
    "correlation_threshold": 0.92,  # Drop one of pair if correlation exceeds this
    "vif_threshold": 10,  # Iteratively drop features with VIF above this
    "adaptive_feature_count": True,  # Scale n_features_per_position by sqrt(n_samples)
    "recency_decay_halflife": 1.5,  # Seasons: weight halves every 1.5 seasons (tightened from 2.0 for 2027 focus)
    # Horizon-aware recency: longer horizons should decay more slowly.
    # Defaults: 1w=1.5 seasons, 4w=2.5 seasons, 18w=3.5 seasons.
    # If a horizon is missing, falls back to recency_decay_halflife.
    "horizon_recency_halflife": {1: 1.5, 4: 2.5, 18: 3.5},
    "cv_gap_seasons": 1,  # Gap between train and val for purged CV (1 = purge last season before test)
    # Per-position target override: "fp" trains directly on fantasy points (no util conversion),
    # "util" trains on utilization score then converts to FP (original two-stage approach).
    # Target type per position: "fp" (direct), "util" (two-stage), "component" (predict stats, assemble FP).
    #
    # SWITCHED TO "fp" 2026-08-29. Component mode was measured WORSE at every
    # position, in 12 of 12 folds, on identical frames:
    #
    #     mean weekly MAE   component   direct(fp)   direct better by
    #       QB                 6.530       6.091          0.439
    #       RB                 4.474       4.391          0.083
    #       TE                 3.147       2.831          0.316
    #       WR                 4.373       4.103          0.270
    #
    # and the comparison favoured component (its artifact was trained on the
    # full set; the direct arm got 3 seasons per fold) and it still lost.
    #
    # Component mode was also the root of three defects found the same week:
    # the train/serve feature skew that produced 808-point weekly predictions
    # masked by a [0,55] clamp; `predicted_utilization` holding fantasy points,
    # which made every util_tier read "Low"; and the notna row gate that trained
    # it on 4-11% of available rows until 2026-08-28.
    #
    # The original rationale in component_predictor.py -- "GBT produced negative
    # R2 for WR/TE/RB" on direct FP -- was true when written and is superseded,
    # not contradicted: that predates the data corrections, the row-gate fix,
    # causal features and C_gbm_mae. See GAPS.md 2026-08-29.
    "position_target_type": {"QB": "fp", "RB": "fp", "WR": "fp", "TE": "fp"},
    # Horizon-specific models (per requirements): 4w LSTM+ARIMA, 18w deep feedforward
    "use_4w_hybrid": True,   # Use Hybrid4WeekModel for n_weeks in 4w band when TF available
    # 18w RETIRED 2026-08-29. Two independent reasons:
    #
    # 1. The target is undefined on 90.7% of rows. An 18-game forward sum needs
    #    18 future games; a season is 17, so only the first weeks of a season
    #    can carry a real label. True coverage is 9.3% (2,806 rows across all
    #    four positions). It read as 100% only because _impute_missing was
    #    median-filling the label -- see GAPS.md 2026-08-29.
    # 2. Nothing consumes it. All 620 rows on the shipped draft board come from
    #    the season model (projection_model == "step8" for every one).
    #
    # The season model already answers "how will this player do over a long
    # stretch", and answers it without this data problem. Long horizons are
    # otherwise obtained by predicting N single weeks and summing them, which
    # needs no separately trained long-horizon model.
    "use_18w_deep": False,
    "horizon_4w_weeks": (4, 5, 6, 7, 8),   # n_weeks that use 4-week hybrid model
    "horizon_long_threshold": 9,   # n_weeks >= this use 18-week deep model when available
    
    # 4-week LSTM hyperparameters (Section IV.A of requirements)
    "lstm_sequence_length": 10,        # Sequence length 8-12 weeks
    "lstm_units": 256,                 # First LSTM layer units (128-256)
    "lstm_dropout": 0.25,              # Dropout between LSTM layers (0.2-0.3)
    "lstm_learning_rate": 0.001,       # Adam optimizer learning rate
    "lstm_epochs": 80,                 # Training epochs (50-100)
    "lstm_batch_size": 32,             # Batch size (32-64)
    "lstm_weight": 0.6,               # LSTM component weight in hybrid (60%)
    "arima_weight": 0.4,              # ARIMA component weight in hybrid (40%)
    "lstm_optuna_trials": 15,         # Optuna trials for LSTM hyperparameter tuning
    "arima_order": (2, 1, 2),         # ARIMA (p, d, q) order
    
    # 18-week residual feedforward hyperparameters
    "deep_n_features": 150,            # Expected input features (150-200)
    "deep_hidden_units": None,         # None = auto-generate 2-stage residual net (256→64)
    "deep_dropout": 0.35,              # Dropout per layer (0.3-0.5)
    "deep_learning_rate": 0.0005,      # Adam learning rate (0.0001-0.01)
    "deep_epochs": 100,                # Training epochs
    "deep_batch_size": 64,             # Batch size (16-128)
    "deep_blend_traditional": 0.3,     # 30% traditional + 70% deep
    "deep_optuna_trials": 15,          # Optuna trials for deep model hyperparameter tuning
    # Training gate policy: when True, fail-fast on requirement minimums
    # (training seasons and per-position player counts) instead of warning only.
    "strict_requirements_default": False,
    # Stability selection bootstrap iterations (per position)
    "stability_n_bootstrap": 30,
    # SHAP/PDP explainability (can be slow on large models)
    "enable_shap_pdp": True,
}

# Fast training overrides: ~8-10x faster with minimal accuracy loss.
# Applied via `python -m src.models.train --fast`.
# Reduces Optuna trials, CV folds, stability bootstrap, horizon model
# epochs, and skips expensive optional steps (SHAP, robust CV, QB dual-target).
FAST_MODEL_CONFIG = {
    "n_optuna_trials": 15,          # 15 vs 100 (6.7x fewer)
    "cv_folds": 3,                  # 3 vs 5 (1.7x fewer OOF folds)
    "stability_n_bootstrap": 10,    # 10 vs 30 (3x fewer)
    "lstm_optuna_trials": 0,        # 0 vs 15 (skip LSTM tuning in fast mode for stability)
    "lstm_epochs": 40,              # 40 vs 80
    "deep_optuna_trials": 5,        # 5 vs 15
    "deep_epochs": 50,              # 50 vs 100
    "enable_shap_pdp": False,       # Skip SHAP/PDP
}

# LOYO (Leave-One-Year-Out) walk-forward backtest configuration.
# Trains a fresh model per fold; tests on each season in the range.
LOYO_CONFIG = {
    "default_test_seasons_start": 2020,  # Earliest valid test season (need >=6 prior seasons: 2014-2019)
    "default_test_seasons_end": None,    # None = latest available - 1 (exclude current)
    "min_train_seasons": 5,              # Minimum training seasons per fold (matches 18w requirement)
    "purge_gap": 1,                      # Seasons to exclude before test fold
}

# =============================================================================
# FEATURE MODE: "full" (400+ features) or "causal" (9-11 per position)
# =============================================================================
# Council recommendation (2026-04-01): strip to 5-10 causal features with
# demonstrated causal relationships to production. Causal mode is the default
# per council recommendation; full mode available via NFL_FEATURE_MODE=full.
FEATURE_MODE = os.environ.get("NFL_FEATURE_MODE", "causal")

CAUSAL_ROLLING_WINDOW = 3  # Only rolling window used in causal mode

# Per-position causal feature lists: opportunity shares, short-window volume,
# one efficiency metric, opponent context, and Vegas implied total.
# v9: snap_share_pct restored (PFR→GSIS merge fix); NGS features added.
# draft_capital_value available in full mode but not promoted to causal yet.
CAUSAL_FEATURES = {
    "RB": [
        "snap_share_pct_roll3_mean",
        "rush_share_pct_roll3_mean", "target_share_pct_roll3_mean",
        "rushing_attempts_roll3_mean", "targets_roll3_mean",
        "rushing_tds_roll3_mean", "yards_per_carry_roll3_mean",
        "ngs_rush_yards_over_expected_per_att_roll3_mean",
        # v31: receiving-efficiency signal for pass-catching RBs
        "catch_rate_roll3_mean",
        # PBP efficiency (2018+) — EPA captures play value in context
        "rush_epa_per_play_roll3_mean",
        "opp_fpts_allowed", "opp_fpts_allowed_s2d_lag1", "opp_fpts_allowed_dvoa_adjusted_lag1",
        "implied_team_total", "spread",
        "team_prior_season_wins",
        "injury_score", "prev_season_ppg",
        "age_curve", "team_changed", "availability_3yr",
        "career_year_flag", "bayesian_prior_ppg",
        "fp_late6_vs_season",
        "is_contract_year", "contract_apy_rank", "depth_chart_rank",
        "speed_score", "team_motion_rate", "team_play_action_rate",
        # v27: rookie identity/draft-capital/combine (real target-leakage
        # bug fixed in advanced_rookie_injury.py first, see GAPS.md §7.2)
        "is_rookie", "rookie_draft_value", "rookie_bust_prob", "combine_score",
        # v34: real draft capital. Until FEATURE_VERSION 34 these columns were
        # absent from the training frame, so advanced_rookie_injury.py fell
        # through to its round-5/pick-150 defaults for EVERY rookie and five of
        # the six rookie_* features were constant. See GAPS.md, "The rookie
        # feature set is NOMINAL". is_undrafted is separate on purpose: ~39% of
        # players who reach the league are undrafted, and that is signal, not a
        # missing value.
        "draft_round", "draft_pick", "draft_pick_value", "is_undrafted",
        # v35: college pedigree, era-aware. Conference is resolved for the
        # season BEFORE the draft, so realignment cannot backdate a fact --
        # a 2011 Texas A&M pick is Big 12, a 2013 one is SEC. Only the binary
        # is exposed to the models; college_conference stays a string for
        # inspection. See src/features/college_conference.py.
        "is_power5",
        # v28: rookie opportunity/breakout/ceiling/floor — held back at v27
        # pending two separate leakage bugs in calculate_opportunity_score
        # (full-season sum, no time filter) and add_advanced_injury_features'
        # same-week workload; both fixed, see GAPS.md §7.2 follow-up
        "rookie_opportunity_score", "rookie_breakout_prob",
        "rookie_ceiling_ppg", "rookie_floor_ppg",
        # PFR contact quality (2018+) — OL quality vs individual elusiveness
        "rb_ybc_avg_roll3_mean", "rb_yac_avg_roll3_mean",
        # Seasonal PFR prior-season broken tackles — individual elusiveness baseline
        "rb_broken_tackles_prior",
        # v19: boom/ceiling features
        "fp_volatility_roll5", "target_share_accel", "snap_share_accel",
        # v22: destination team positional usage (zero for non-changers)
        "dest_team_pos_tgt_pg", "dest_team_pos_carry_pg",
        # v23: red zone / goal-line per-player allocation, weather
        "redzone_target_share_pct_roll3_mean", "goal_line_carry_share_pct_roll3_mean",
        "wind_speed_mph", "is_dome", "precipitation_flag", "temperature_bucket",
        # v24: team position-target allocation, lead-back share, tempo
        "team_rb_target_share_roll3_mean", "team_rb_lead_share_roll3_mean",
        "team_plays_roll3_mean", "team_pace_sec_per_play_roll3_mean",
        # v24: head-coach identity + coaching-change detection
        "coaching_change", "coaching_adaptation_score", "coaching_stability",
        "coaching_change_impact",
        # v29: offensive personnel grouping usage (GAPS.md §3.3/§11.1.E).
        # 12/21 personnel = extra blocker/2-back sets -> more RB opportunity.
        "team_pct_12_personnel_roll3_mean", "team_pct_21_personnel_roll3_mean",
        # v30: offensive-line run-block quality (GAPS.md §11.1.H).
        "team_run_block_ybc_avg_roll3_mean",
        # v32: team pass-rate-over-expected (GAPS.md Phase 6b follow-up).
        "team_neutral_pass_rate_oe_roll3_mean",
    ],
    "WR": [
        "snap_share_pct_roll3_mean",
        "target_share_pct_roll3_mean", "air_yards_share_pct_roll3_mean",
        "targets_roll3_mean", "receptions_roll3_mean",
        "receiving_tds_roll3_mean", "yards_per_target_roll3_mean",
        # v31: hands reliability / target-to-catch conversion
        "catch_rate_roll3_mean",
        "ngs_avg_separation_roll3_mean",
        # NGS target depth + YAC ability (2018+) — route type and elusiveness signal
        "ngs_avg_intended_air_yards_roll3_mean",
        "ngs_avg_yac_above_expectation_roll3_mean",
        # PBP efficiency (2018+) — receiving EPA per target
        "recv_epa_per_target_roll3_mean",
        "opp_fpts_allowed", "opp_fpts_allowed_s2d_lag1", "opp_fpts_allowed_dvoa_adjusted_lag1",
        "implied_team_total", "spread",
        "team_prior_season_wins",
        # QB identity EPA: actual QB1 assigned to team (roster-sourced for 2026+)
        "current_qb_epa_per_att",
        "injury_score", "prev_season_ppg",
        "age_curve", "team_changed", "availability_3yr",
        "career_year_flag", "bayesian_prior_ppg",
        "fp_late6_vs_season",
        "is_contract_year", "contract_apy_rank", "depth_chart_rank",
        "speed_score", "team_motion_rate", "team_play_action_rate",
        # v27: rookie identity/draft-capital/combine (real target-leakage
        # bug fixed in advanced_rookie_injury.py first, see GAPS.md §7.2)
        "is_rookie", "rookie_draft_value", "rookie_bust_prob", "combine_score",
        # v34: real draft capital. Until FEATURE_VERSION 34 these columns were
        # absent from the training frame, so advanced_rookie_injury.py fell
        # through to its round-5/pick-150 defaults for EVERY rookie and five of
        # the six rookie_* features were constant. See GAPS.md, "The rookie
        # feature set is NOMINAL". is_undrafted is separate on purpose: ~39% of
        # players who reach the league are undrafted, and that is signal, not a
        # missing value.
        "draft_round", "draft_pick", "draft_pick_value", "is_undrafted",
        # v35: college pedigree, era-aware. Conference is resolved for the
        # season BEFORE the draft, so realignment cannot backdate a fact --
        # a 2011 Texas A&M pick is Big 12, a 2013 one is SEC. Only the binary
        # is exposed to the models; college_conference stays a string for
        # inspection. See src/features/college_conference.py.
        "is_power5",
        # v28: rookie opportunity/breakout/ceiling/floor — held back at v27
        # pending two separate leakage bugs in calculate_opportunity_score
        # (full-season sum, no time filter) and add_advanced_injury_features'
        # same-week workload; both fixed, see GAPS.md §7.2 follow-up
        "rookie_opportunity_score", "rookie_breakout_prob",
        "rookie_ceiling_ppg", "rookie_floor_ppg",
        # PFR drop rate (2018+) — target quality / hands reliability
        "recv_drop_pct_roll3_mean",
        # Seasonal PFR prior-season drop rate — full-season hands baseline
        "recv_drop_pct_season_prior",
        # v19: boom/ceiling features
        "fp_volatility_roll5", "target_share_accel", "snap_share_accel",
        # v22: destination team positional target volume (zero for non-changers)
        "dest_team_pos_tgt_pg",
        # v23: red zone target share, WOPR, weather
        "redzone_target_share_pct_roll3_mean", "wopr_roll3",
        "wind_speed_mph", "is_dome", "precipitation_flag", "temperature_bucket",
        # v24: team position-target allocation/concentration, tempo
        "team_wr_target_share_roll3_mean", "team_target_concentration_roll3_mean",
        "team_plays_roll3_mean", "team_pace_sec_per_play_roll3_mean",
        # v24: head-coach identity + coaching-change detection
        "coaching_change", "coaching_adaptation_score", "coaching_stability",
        "coaching_change_impact",
        # v29: offensive personnel grouping usage (GAPS.md §3.3/§11.1.E).
        # 11 personnel (3 WR) is a spread-offense signal, ceiling for WR3+.
        "team_pct_11_personnel_roll3_mean", "team_pct_12_personnel_roll3_mean",
        # v30: offensive-line pass-block quality (route development time)
        # and PBP-derived pass-play participation rate (GAPS.md
        # §11.1.C/D/H follow-up). NOT true route participation -- see
        # get_pass_play_participation_from_pbp docstring; it's a real but
        # imperfect proxy (can't separate route-runners from in-line
        # pass-blockers on the same play).
        "team_sack_rate_allowed_roll3_mean", "pbp_pass_play_participation_pct_roll3_mean",
        # v32: rushing usage (jet sweeps/end-arounds) + team pass-rate-over-
        # expected (GAPS.md Phase 6b follow-up).
        "rush_share_pct_roll3_mean", "team_neutral_pass_rate_oe_roll3_mean",
    ],
    "TE": [
        "snap_share_pct_roll3_mean",
        "target_share_pct_roll3_mean",
        "targets_roll3_mean", "receptions_roll3_mean",
        "receiving_tds_roll3_mean", "yards_per_target_roll3_mean",
        # v31: hands reliability / target-to-catch conversion
        "catch_rate_roll3_mean",
        "ngs_avg_separation_roll3_mean",
        # NGS target depth + YAC ability (2018+) — route type and elusiveness signal
        "ngs_avg_intended_air_yards_roll3_mean",
        "ngs_avg_yac_above_expectation_roll3_mean",
        # PBP efficiency (2018+) — receiving EPA per target
        "recv_epa_per_target_roll3_mean",
        "opp_fpts_allowed", "opp_fpts_allowed_s2d_lag1", "opp_fpts_allowed_dvoa_adjusted_lag1",
        "implied_team_total", "spread",
        "team_prior_season_wins",
        # QB identity EPA: actual QB1 assigned to team (roster-sourced for 2026+)
        "current_qb_epa_per_att",
        "injury_score", "prev_season_ppg",
        "age_curve", "team_changed", "availability_3yr",
        "career_year_flag", "bayesian_prior_ppg",
        "fp_late6_vs_season",
        "is_contract_year", "contract_apy_rank", "depth_chart_rank",
        "speed_score", "team_motion_rate", "team_play_action_rate",
        # v27: rookie identity/draft-capital/combine (real target-leakage
        # bug fixed in advanced_rookie_injury.py first, see GAPS.md §7.2)
        "is_rookie", "rookie_draft_value", "rookie_bust_prob", "combine_score",
        # v34: real draft capital. Until FEATURE_VERSION 34 these columns were
        # absent from the training frame, so advanced_rookie_injury.py fell
        # through to its round-5/pick-150 defaults for EVERY rookie and five of
        # the six rookie_* features were constant. See GAPS.md, "The rookie
        # feature set is NOMINAL". is_undrafted is separate on purpose: ~39% of
        # players who reach the league are undrafted, and that is signal, not a
        # missing value.
        "draft_round", "draft_pick", "draft_pick_value", "is_undrafted",
        # v35: college pedigree, era-aware. Conference is resolved for the
        # season BEFORE the draft, so realignment cannot backdate a fact --
        # a 2011 Texas A&M pick is Big 12, a 2013 one is SEC. Only the binary
        # is exposed to the models; college_conference stays a string for
        # inspection. See src/features/college_conference.py.
        "is_power5",
        # v28: rookie opportunity/breakout/ceiling/floor — held back at v27
        # pending two separate leakage bugs in calculate_opportunity_score
        # (full-season sum, no time filter) and add_advanced_injury_features'
        # same-week workload; both fixed, see GAPS.md §7.2 follow-up
        "rookie_opportunity_score", "rookie_breakout_prob",
        "rookie_ceiling_ppg", "rookie_floor_ppg",
        # PFR drop rate (2018+) — target quality / hands reliability
        "recv_drop_pct_roll3_mean",
        # Seasonal PFR prior-season drop rate — full-season hands baseline
        "recv_drop_pct_season_prior",
        # v19: boom/ceiling features
        "fp_volatility_roll5", "target_share_accel", "snap_share_accel",
        # v22: destination team positional target volume (zero for non-changers)
        "dest_team_pos_tgt_pg",
        # v23: red zone target share, WOPR, weather
        "redzone_target_share_pct_roll3_mean", "wopr_roll3",
        "wind_speed_mph", "is_dome", "precipitation_flag", "temperature_bucket",
        # v24: team position-target allocation/concentration, tempo
        "team_te_target_share_roll3_mean", "team_target_concentration_roll3_mean",
        "team_plays_roll3_mean", "team_pace_sec_per_play_roll3_mean",
        # v24: head-coach identity + coaching-change detection
        "coaching_change", "coaching_adaptation_score", "coaching_stability",
        "coaching_change_impact",
        # v29: offensive personnel grouping usage (GAPS.md §3.3/§11.1.E).
        # 12/13 personnel (2-3 TE sets) directly determines TE snap/route
        # ceiling versus a 3-WR spread offense.
        "team_pct_12_personnel_roll3_mean", "team_pct_13_personnel_roll3_mean",
        # v30: offensive-line pass-block quality + PBP-derived pass-play
        # participation rate (GAPS.md §11.1.C/D/H follow-up; see WR list
        # above for the "not true route participation" caveat).
        "team_sack_rate_allowed_roll3_mean", "pbp_pass_play_participation_pct_roll3_mean",
        # v32: team pass-rate-over-expected (GAPS.md Phase 6b follow-up).
        "team_neutral_pass_rate_oe_roll3_mean",
    ],
    "QB": [
        # Volume (3-week rolling)
        "passing_attempts_roll3_mean", "passing_tds_roll3_mean",
        "rushing_attempts_roll3_mean", "rushing_yards_roll3_mean",
        "rushing_tds_roll3_mean",
        # Efficiency (3-week rolling)
        "yards_per_attempt_roll3_mean", "completion_pct_roll3_mean",
        # PBP advanced (3-week rolling, 2018+)
        "pass_epa_per_play_roll3_mean", "pass_success_rate_roll3_mean",
        # NGS (3-week rolling, 2018+)
        "ngs_completion_percentage_above_expectation_roll3_mean",
        "ngs_avg_time_to_throw_roll3_mean",
        # Game context
        "opp_fpts_allowed", "opp_fpts_allowed_s2d_lag1", "opp_fpts_allowed_dvoa_adjusted_lag1",
        "implied_team_total", "spread",
        "team_prior_season_wins",
        # Player context
        "injury_score", "prev_season_ppg",
        "age_curve", "team_changed", "availability_3yr",
        "career_year_flag", "bayesian_prior_ppg",
        "fp_late6_vs_season",
        "is_contract_year", "contract_apy_rank", "depth_chart_rank",
        "speed_score", "team_motion_rate", "team_play_action_rate",
        # v27: rookie identity/draft-capital/combine (real target-leakage
        # bug fixed in advanced_rookie_injury.py first, see GAPS.md §7.2)
        "is_rookie", "rookie_draft_value", "rookie_bust_prob", "combine_score",
        # v34: real draft capital. Until FEATURE_VERSION 34 these columns were
        # absent from the training frame, so advanced_rookie_injury.py fell
        # through to its round-5/pick-150 defaults for EVERY rookie and five of
        # the six rookie_* features were constant. See GAPS.md, "The rookie
        # feature set is NOMINAL". is_undrafted is separate on purpose: ~39% of
        # players who reach the league are undrafted, and that is signal, not a
        # missing value.
        "draft_round", "draft_pick", "draft_pick_value", "is_undrafted",
        # v35: college pedigree, era-aware. Conference is resolved for the
        # season BEFORE the draft, so realignment cannot backdate a fact --
        # a 2011 Texas A&M pick is Big 12, a 2013 one is SEC. Only the binary
        # is exposed to the models; college_conference stays a string for
        # inspection. See src/features/college_conference.py.
        "is_power5",
        # v28: rookie opportunity/breakout/ceiling/floor — held back at v27
        # pending two separate leakage bugs in calculate_opportunity_score
        # (full-season sum, no time filter) and add_advanced_injury_features'
        # same-week workload; both fixed, see GAPS.md §7.2 follow-up
        "rookie_opportunity_score", "rookie_breakout_prob",
        "rookie_ceiling_ppg", "rookie_floor_ppg",
        # PFR pressure rate (2018+) — pressured pct as true rate (causal, not count-based)
        "qb_pressure_pct_roll3_mean",
        # Seasonal PFR prior-season accuracy + pocket time — decision quality baseline
        "qb_bad_throw_pct_prior", "qb_pocket_time_prior",
        # NGS decision quality (2018+) — aggressiveness and depth of target
        "ngs_aggressiveness_roll3_mean", "ngs_avg_air_yards_to_sticks_roll3_mean",
        # v19: ceiling volatility signal
        "fp_volatility_roll5",
        # v23: weather
        "wind_speed_mph", "is_dome", "precipitation_flag", "temperature_bucket",
        # v24: tempo — more plays/game means more dropbacks and rush attempts
        "team_plays_roll3_mean", "team_pace_sec_per_play_roll3_mean",
        # v24: head-coach identity + coaching-change detection
        "coaching_change", "coaching_adaptation_score", "coaching_stability",
        "coaching_change_impact",
        # prior_season_late_ppg removed in v20 — caused +3.6pt upward bias for
        # benched QBs whose high late-2019 scores don't reflect 2020 backup role.
        # prev_season_ppg + depth_chart_rank already cover prior quality.
        # v30: offensive-line pass-block quality (GAPS.md §11.1.H) — most
        # direct beneficiary, affects time to throw.
        "team_sack_rate_allowed_roll3_mean",
        # v32: team pass-rate-over-expected (GAPS.md Phase 6b follow-up).
        "team_neutral_pass_rate_oe_roll3_mean",
    ],
}

# =============================================================================
# COMPONENT PREDICTION (predict stat lines, assemble fantasy points)
# =============================================================================
# Council Phase 2: predict stable components separately (targets, receptions,
# yards, TDs) then assemble FP from those predictions.  Each component has
# higher autocorrelation and lower touchdown contamination than raw FP.

# PBP/NGS features that are only available 2018+.  For pre-2018 rows these
# should be NaN (not 0) so HistGradientBoosting can treat them as missing.
QB_PBP_FEATURES = [
    "pass_epa_per_play_roll3_mean", "pass_success_rate_roll3_mean",
    "pass_wpa_per_play_roll3_mean", "rush_epa_per_play_roll3_mean",
    "ngs_completion_percentage_above_expectation_roll3_mean",
    "ngs_avg_time_to_throw_roll3_mean",
    "ngs_aggressiveness_roll3_mean",
    "ngs_avg_air_yards_to_sticks_roll3_mean",
]

# PPR scoring weights: stat_column -> points_per_unit
PPR_SCORING_WEIGHTS = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -2.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "receptions": 1.0,
    "fumbles_lost": -2.0,
}

# Which stat components to predict per position (only the meaningful ones)
COMPONENT_TARGETS = {
    "QB": ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds"],
    "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"],
    "WR": ["receptions", "receiving_yards", "receiving_tds"],
    "TE": ["receptions", "receiving_yards", "receiving_tds"],
}

# =============================================================================
# TRAINING DATA WINDOW
# =============================================================================
#
# The NFL has evolved significantly over time:
#   - 2000-2010: Run-heavy offenses, fewer spread concepts
#   - 2011-2019: Pass-first revolution, RPO emergence
#   - 2020+: RPO explosion, increased passing efficiency
#
# Training on older data (pre-2018) teaches outdated patterns from a
# fundamentally different era of NFL football (council recommendation:
# drop pre-2018 data).  The modern passing game, RPO schemes, and rule
# changes make pre-2018 data actively harmful for generalization.
TRAINING_START_YEAR_DEFAULT = 2018   # Modern NFL era (council: drop pre-2018)
TRAINING_END_YEAR_DEFAULT = CURRENT_NFL_SEASON   # Latest season (same as CURRENT_NFL_SEASON)
TRAINING_YEARS = {
    "start_year": TRAINING_START_YEAR_DEFAULT,
    "end_year": TRAINING_END_YEAR_DEFAULT,
    "test_years": [TRAINING_END_YEAR_DEFAULT],   # Latest season held out for testing
    "min_years": 3,
}

# Requirement-derived minimum training seasons per horizon (see docs/fantasy requirements)
MIN_TRAINING_SEASONS_1W = 3   # 1-week model: min 3, optimal 5+
MIN_TRAINING_SEASONS_18W = 5  # 18-week model: min 5 (adjusted for 2018+ window)

# Horizons the weekly model is TRAINED on. Multi-week projections are built by
# predicting N single weeks and summing them, so a horizon only belongs here if
# it is trained directly rather than derived.
#
# 18 was removed 2026-08-29: its label is undefined on 90.7% of rows (an
# 18-game forward sum needs more future games than a 17-game season leaves) and
# nothing consumed its output. See MODEL_CONFIG["use_18w_deep"].
TRAINING_HORIZONS = [1, 4]
MIN_TRAINING_SEASONS_4W = 4   # 4-week horizon (LSTM+ARIMA): min 4
# Per-position minimum players for training (requirements: ~30 QB, 60 RB, 70 WR, 30 TE)
MIN_PLAYERS_PER_POSITION = {"QB": 30, "RB": 60, "WR": 70, "TE": 30}

# Alternative training windows (end_year = CURRENT_NFL_SEASON; start_year explicit)
TRAINING_WINDOW_PRESETS = {
    "modern": {"start_year": TRAINING_START_YEAR_DEFAULT, "end_year": TRAINING_END_YEAR_DEFAULT},
    "extended": {"start_year": 2014, "end_year": TRAINING_END_YEAR_DEFAULT},
    "full": {"start_year": MIN_HISTORICAL_YEAR, "end_year": TRAINING_END_YEAR_DEFAULT},
}

# Feature engineering rolling windows (rubric-required windows included).
# Keep required 3,4,5,8 and include 12 for longer-term trends.
ROLLING_WINDOWS = [3, 4, 5, 8, 12]
LAG_WEEKS = [1, 2, 3, 4]  # Lag features

# Prediction settings
MAX_PREDICTION_WEEKS = 18
MIN_GAMES_FOR_PREDICTION = 4  # Minimum historical games needed

# =============================================================================
# PLAYER ELIGIBILITY FILTERING
# =============================================================================
# Players must appear on a roster in at least one of the ELIGIBLE_SEASONS to be
# included in predictions. This prevents retired players (e.g. Gronkowski, who
# retired in 2022) from appearing in projections for future seasons.
# By default, require a roster entry in the most recent 2 seasons.
ELIGIBLE_SEASONS_LOOKBACK = 2  # Number of recent seasons to check for roster presence

# Production retraining/monitoring configuration
# Used by scripts/production_retrain_and_monitor.py and train drift checks.
RETRAINING_CONFIG = {
    # --- Schedule ---
    "auto_retrain": True,
    "retrain_day": "Tuesday",                  # Weekly retrain day (in-season)
    "in_season_cadence_days": 7,               # Retrain every 7 days during NFL season
    "off_season_cadence_days": 30,             # Retrain monthly during off-season
    "retrain_hour_utc": 6,                     # Preferred retrain hour (UTC) for cron scheduling
    "retrain_sla_seconds": 24 * 3600,          # Max allowed wall-clock time for a retrain cycle

    # --- Drift detection ---
    "degradation_threshold_pct": 20.0,         # Flag drift if RMSE degrades >20% vs previous run
    "drift_auto_rollback": True,               # Auto-rollback to previous model on drift detection
    "drift_check_after_retrain": True,          # Run drift check immediately after each retrain
    "drift_position_threshold_pct": 25.0,      # Per-position RMSE drift threshold (stricter per-pos)

    # --- Data freshness ---
    "max_data_staleness_hours": 168,           # 7 days: alert if latest data is older than this
    "require_current_season_data": True,       # Block retrain if current-season PBP data is missing
    "min_new_weeks_for_retrain": 1,            # Require at least 1 new week of data before retraining

    # --- Rollback policy ---
    "max_rollback_versions": 5,                # Keep last 5 model versions for rollback
    "rollback_on_test_regression": True,       # Rollback if test-set metrics regress beyond threshold
    "rollback_rmse_increase_pct": 15.0,        # Rollback if RMSE increases >15% on test set

    # --- Feature version enforcement ---
    "block_retrain_on_version_mismatch": False, # If True, refuse to serve stale-feature models
    "warn_on_version_mismatch": True,           # Print warning when feature version differs

    # --- Monitoring hooks ---
    "enable_drift_status_file": True,          # Write drift_status.json after each check
    "enable_retrain_status_file": True,        # Write retrain_status.json after each retrain
    "alert_on_drift": True,                    # Log WARNING-level alert on drift detection
    "alert_on_sla_breach": True,               # Log WARNING if retrain exceeds SLA
}

# QB target selection (util vs future fantasy points): metadata file written after training
QB_TARGET_CHOICE_FILENAME = "qb_target_choice.json"

# Feature set version: bump when feature_engineering adds/removes/renames model features.
# Saved when training; checked when loading models. Mismatch triggers a retrain warning.
FEATURE_VERSION = "35"  # v35: is_power5 from an era-aware college->conference map (realignment-correct: Texas A&M is Big 12 pre-2012, SEC after; USC is Pac-12 until the 2024 Big Ten move). v34: real draft capital + birth_date. See GAPS.md.
# v32: team pass-rate-over-expected (team_neutral_pass_rate_oe_roll3_mean, all 4 positions) + WR rushing usage (rush_share_pct_roll3_mean) — GAPS.md Phase 6b follow-up, two real gaps missed by the v31/Phase 6 audit — see FEATURE_VERSION history below
# v31: rolling catch rate (catch_rate_roll3_mean, RB/WR/TE) — GAPS.md §8.1 Phase 6 (next_focus.md) audit; also corrects 10 stale "missing" items in that section, actually resolved v24-v30
# v30: team OL quality (team_sack_rate_allowed/team_run_block_ybc_avg_roll3_mean, from weekly_pfr) + PBP-derived pass-play participation rate (pbp_pass_play_participation_pct_roll3_mean, NOT true route participation, GAPS.md §11.1.C/D/H)
# v29: offensive personnel grouping usage (team_pct_11/12/21/13_personnel_roll3_mean, GAPS.md §3.3/§11.1.E) — parsed from PBP offense_personnel, 2016+ coverage
# Preserve NaN in team personnel columns instead of blanket-filling them to 0.
# OFF by default deliberately: every existing result was produced with them
# filled, so this is staged as an attributable arm for the pending v34/v35
# re-baseline rather than a silent default flip. See
# src/features/utilization_score.py PERSONNEL_MISSINGNESS_COLS and GAPS.md.
# Preserve NaN in rolling/lag HISTORY features instead of median-filling them.
# A player-season's first week has no preceding weeks, so those features are
# genuinely undefined (8.9% of training rows). Filling them teaches the model
# that a rookie's week 1 looks like an average veteran, and leaves LightGBM
# with no missing-direction to use when a real cold-start row arrives.
#
# DEFAULT FLIPPED TO ON, 2026-08-28, on a pre-registered 11-fold paired
# experiment. Full write-up:
#   data/experiments/phase7_histnan_v3_20260827/RESULTS.md
#   PRE_REGISTRATION.md (v2) in data/experiments/phase7_histnan_20260826/
#
#   falsification  placebo (OFF vs OFF) 538/538 identical -- noise floor zero
#   primary        9 of 11 folds favour ON; mean paired dMAE -0.658, t = -3.03
#   rule           >=8/11 folds AND mean <= -0.25, fixed before the run: MET
#
# READ THE EFFECT CORRECTLY. Pooled it is -1.55% of a 43-point MAE, and ON is
# closer on exactly 50.0% of individual player-seasons -- a coin flip per
# player. The entire gain is ROOKIES:
#
#   rookie   n=1188  dMAE -3.2596  10/11 folds  MAE 44.4 -> 41.1  (-7.3%)
#   veteran  n=5107  dMAE -0.0649   8/11 folds  MAE 42.7 -> 42.6  (nil)
#
# So this is "~7% better rookie season projections, no veteran effect", not a
# general accuracy improvement. Describing it as the latter overstates it by
# roughly 5x, which is the dilution factor of a 19%-rookie population.
#
# Mechanism caveat: 96.5% of scored feature rows are byte-identical between
# arms, so the gain is MODEL-level (the flag also changes the training matrix)
# rather than coming from debut rows being scored differently. The design
# cannot separate those two channels and does not claim to.
PRESERVE_HISTORY_MISSINGNESS = os.getenv("NFL_PRESERVE_HISTORY_MISSINGNESS", "1") == "1"

PRESERVE_PERSONNEL_MISSINGNESS = os.getenv("NFL_PRESERVE_PERSONNEL_MISSINGNESS", "0") == "1"

# Prior-week team/opponent joins cannot resolve in week 1 (no week 0), and the
# blanket numeric fill turned that NULL into a measured 0 for 100% of week-1
# rows. Default ON: `opp_fpts_allowed = 0` claims the opponent allows zero
# fantasy points, uniformly for all 32 teams, in the week where matchup matters
# most. See PRIOR_WEEK_JOIN_COLS in src/features/utilization_score.py and
# GAPS.md 2026-08-29.
PRESERVE_PRIOR_WEEK_MISSINGNESS = os.getenv("NFL_PRESERVE_PRIOR_WEEK_MISSINGNESS", "1") == "1"

# How the log1p target transform is handled. Training on log1p(y) and inverting
# with expm1 estimates a conditional GEOMETRIC mean, which sits below the
# arithmetic mean by Jensen's inequality -- so predictions come out low, and
# most so where outcomes are most volatile.
#
# Measured on the 2026-08-30 serving backtest: QB -19% relative bias, RB -37%,
# WR -40%, TE -39%. QB is the only position whose 1-week model does NOT use the
# transform, and it has roughly half the bias of the others.
#
#   "on"        transform, invert with expm1 (historical behaviour)
#   "smearing"  transform, invert with expm1 x Duan smearing factor fitted on
#               out-of-fold residuals -- corrects the retransformation bias
#               without assuming a residual distribution
#   "off"       never transform
TARGET_TRANSFORM_MODE = os.getenv("NFL_TARGET_TRANSFORM_MODE", "smearing")

# Which positions get the log1p target transform, PINNED rather than re-decided
# from data on every training run.
#
# TargetTransformer originally activated on measured target skewness
# (|skew| >= 0.5). That is data-dependent, so the decision flipped between
# training windows: QB 1w was transformed when trained on 2018-2024 and not on
# 2018-2025. Since the transform (and therefore the smearing correction that
# offsets its bias) changes prediction scale, a position could silently gain or
# lose a ~20% calibration correction from one retrain to the next, with nothing
# reporting it.
#
# Values below are what the validated 2026-08-31 production build settled on,
# consistent across both windows for the SAVED models:
#     QB inactive (both horizons), RB/WR/TE active (both horizons)
#
# Set to None to restore the old behaviour of deciding from skewness. A
# position absent from this dict also falls back to the skew check.
TARGET_TRANSFORM_POSITIONS = {"QB": False, "RB": True, "WR": True, "TE": True}

# Number of prediction-quantile strata used to fit the Duan smearing factor.
# 1 = a single global factor (the original Duan estimator).
#
# Duan assumes residuals are homoscedastic in log space. They are not here --
# variance grows with the prediction -- so one factor under-corrects the busy
# end. Measured 2026-09-01: after global smearing the models still predicted
# only 0.67-0.77x their own training target mean.
# MEASURED 2026-09-01 and REJECTED as a default. Stratifying made RB and WR
# worse (RB bias -1.62 -> -1.83, WR -1.45 -> -1.54) on a held-out 2025 backtest.
#
# The hypothesis was backwards. Fitted factors DECREASE with the prediction
# (RB [1.283, 1.243, 1.285, 1.202, 1.120]) because in LOG space the low-usage
# players are the volatile ones -- a 0-point week vs a 5-point week is a huge
# log swing, while a starter moving 12 -> 18 is a small one. Stratification
# correctly learned to lift the top end less, and the top end is where the
# points are. A uniform factor is accidentally better.
#
# Kept configurable rather than deleted: the code path is measured and
# documented, and >1 is the right tool if residuals ever become
# heteroscedastic in the other direction.
SMEARING_STRATA = int(os.getenv("NFL_SMEARING_STRATA", "1"))

FEATURE_VERSION_FILENAME = "feature_version.txt"

# =============================================================================
# ADVANCED ANALYTICS SETTINGS
# =============================================================================
# Enable/disable individual advanced analytics modules.
ADVANCED_ANALYTICS_CONFIG = {
    "enable_news_sentiment": True,      # NLP sentiment scoring from news text
    "enable_coaching_change": True,      # Coaching change detection & impact
    "enable_suspension_risk": True,      # Suspension history & risk scoring
    "enable_trade_deadline": True,       # Trade deadline proximity features
    "enable_playoff_features": True,     # Playoff context & rest risk
    "trade_deadline_week": 8,            # NFL trade deadline (week 8)
    "coaching_adaptation_weeks": 6,      # Weeks for players to adapt to new coach
    "suspension_base_risk": 0.03,        # Baseline annualized suspension probability
    "suspension_recidivism_factor": 2.5, # Risk multiplier per prior suspension
}

# =============================================================================
# PERFORMANCE TARGETS (from requirements)
# =============================================================================
# Position-specific RMSE targets by horizon
RMSE_TARGETS_1W = {"QB": 7.5, "RB": 8.5, "WR": 8.0, "TE": 7.0}
RMSE_TARGETS_4W = {"QB": 10.0, "RB": 11.0, "WR": 10.0, "TE": 9.0}
RMSE_TARGETS_18W = {"QB": 15.0, "RB": 15.0, "WR": 15.0, "TE": 15.0}

# MAPE targets by horizon
MAPE_TARGETS = {"1w": 25.0, "4w": 35.0, "18w": 45.0}

# R² targets by horizon
R2_TARGETS = {"1w": 0.50, "4w": 0.40, "18w": 0.30}

# Cash head-to-head decision-quality defaults — consumed by the walk-forward
# backtester (see src/evaluation/ts_backtester.py) and by the lineup-decision
# reporting in src/evaluation/backtester.py::backtest_lineup_decisions.
# 1.8× is the typical DraftKings/FanDuel cash H2H payout after ~20% rake;
# break-even ≈ 55.6% win rate.
DECISION_QUALITY = {
    "payout_multiplier": 1.8,
    "break_even_win_rate": 1.0 / 1.8,
}

# Default Ridge regularization for the walk-forward backtester.  Lowered
# back to 1.0 on 2026-08-06 (GAPS.md §7.4 follow-up: comprehensive
# skeptical audit of the 2026-04-20 alpha=10,000 change). The doc this
# constant used to cite (docs/ALPHA_SWEEP_20260419.md) doesn't exist
# anywhere in this repo, and no artifact backing the specific claim
# ("29-14 vs 27-16 over 43 weeks, p=0.016") survived either. What does
# exist and is reproducible: (1) the real alpha-sweep data
# (data/backtest_results/alpha_sweep_summary.json) shows correlation
# declining at every position from α=10,000 onward, not improving; (2) a
# direct, same-day, same-code reproduction on the 2025 season showed
# α=1.0 beating α=10,000 on BOTH correlation (0.351 vs 0.350, a wash) AND
# hindsight win rate (72.7%/16-6/p=0.026 vs 63.6%/14-8/p=0.143,
# not significant) -- the reproduction contradicts the original claim's
# direction, not just its magnitude. Also note: this constant has zero
# effect on real production regardless of its value -- train.py/
# component_predictor.py/position_models.py/ensemble.py never import it;
# production's ComponentPredictor hardcodes alpha=1.0 independently. It
# only affects src/evaluation/ts_backtester.py's evaluation tooling. 1.0
# was chosen over an untested intermediate value (the sweep's correlation
# peak was around 100-1,000 for some positions) because 1.0 is what was
# actually validated on decision-quality (the metric that matters) and
# matches production's own hardcoded value -- picking a different,
# decision-quality-untested number here would just be a new unvalidated
# guess of the same kind this change is trying to get away from.
RIDGE_DEFAULT_ALPHA = 1.0

# Per-position exponent for scaling a single-week prediction std into an
# n-week-total std (GAPS.md §7.4 follow-up: "n_weeks**0.4 multi-week
# CI-scaling exponent"), used only by ensemble.py's single_week_models
# fallback path (no per-horizon-trained model to draw real calibration
# from there). Derived via scripts/derive_multiweek_ci_exponent.py: fits
# std(N) = std(1) * N**alpha by log-log regression of the realized std of
# real forward-looking N-week fantasy-point sums (player_weekly_stats,
# 2006-2025) against N, for N in [1,2,4,8,13,18]. R² > 0.998 for every
# position. All four values come out well above the independent-weeks
# baseline of 0.5, confirming real positive week-to-week autocorrelation
# (hot/cold stretches, role entrenchment) -- and well above the previous
# hardcoded guess of 0.4, which was wrong in direction (below even the
# i.i.d. baseline, under-widening multi-week CIs).
MULTI_WEEK_CI_SCALE_EXPONENT = {
    "QB": 0.713,
    "RB": 0.810,
    "WR": 0.780,
    "TE": 0.798,
}
# Fallback for any position not in the dict above (mean of the fitted values).
MULTI_WEEK_CI_SCALE_EXPONENT_DEFAULT = 0.775

# Success criteria thresholds (from requirements Section VII)
SUCCESS_CRITERIA = {
    "spearman_rho_min": 0.65,        # Ranking accuracy target
    "within_10_pts_pct_min": 80.0,   # 80%+ within 10 points
    "within_7_pts_pct_min": 70.0,    # 70%+ within 7 points
    "beat_naive_baseline_pct": 25.0, # Beat all baselines by >25%
    "beat_expert_pct_qb": 10.0,      # Beat expert projections by 8-12%
    "beat_expert_pct_rb": 12.5,      # Beat expert projections by 10-15%
    "beat_expert_pct_wr": 10.0,      # Beat expert projections by 8-12%
    "beat_expert_pct_te": 15.0,      # Beat expert projections by 12-18%
    "tier_accuracy_min": 0.75,       # >75% correct tier classification
    "max_weekly_degradation_pct": 20.0,  # No >20% accuracy degradation across season
    "confidence_band_coverage": 88.2,    # % of players within 10-point CI
}

# Prediction speed requirement
MAX_PREDICTION_TIME_PER_PLAYER_SECONDS = 5.0

# Offensive Momentum Score time weights (per requirements)
MOMENTUM_WEIGHTS = {
    "recent_4w": 0.60,   # Recent 4 weeks = 60%
    "mid_5_8w": 0.30,    # Weeks 5-8 = 30%
    "early_9_plus": 0.10, # Weeks 9+ = 10%
}

# Position-specific boom/bust thresholds (fantasy points)
# QB scores higher on average, so boom/bust thresholds are higher.
# TE scores lower, so thresholds are lower.
BOOM_BUST_THRESHOLDS = {
    "QB": {"boom": 25, "bust": 10},
    "RB": {"boom": 20, "bust": 5},
    "WR": {"boom": 20, "bust": 5},
    "TE": {"boom": 15, "bust": 3},
}
BOOM_BUST_DEFAULT = {"boom": 20, "bust": 5}

# Position-specific age curve parameters
# RBs peak earlier and decline faster; QBs/TEs peak later with gentler decline.
AGE_CURVE_PARAMS = {
    "QB": {"peak": 28, "coefficient": 0.003},
    "RB": {"peak": 25, "coefficient": 0.008},
    "WR": {"peak": 27, "coefficient": 0.005},
    "TE": {"peak": 28, "coefficient": 0.004},
}
AGE_CURVE_DEFAULT = {"peak": 27, "coefficient": 0.005}

# Minimum samples required to enable converter hyperparameter tuning
CONVERTER_TUNING_MIN_SAMPLES = 200
