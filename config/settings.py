"""Configuration settings for NFL predictor."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "nfl_data.db"

# Scraping settings
SCRAPER_DELAY = 2.0  # Seconds between requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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

# Model settings
MODEL_CONFIG = {
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "n_optuna_trials": 100,
    "early_stopping_rounds": 25,
    "validation_pct": 0.2,       # Fraction of training data for ensemble weight optimization
    "n_features_per_position": 50,  # Max features after selection (per position)
    "correlation_threshold": 0.92,  # Drop one of pair if correlation exceeds this
    "vif_threshold": 10,  # Iteratively drop features with VIF above this
    "adaptive_feature_count": True,  # Scale n_features_per_position by sqrt(n_samples)
    "recency_decay_halflife": 2.0,  # Seasons: weight halves every 2 seasons (None = no weighting)
    "cv_gap_seasons": 1,  # Gap between train and val for purged CV (1 = purge last season before test)
    # Horizon-specific models (per requirements): 4w LSTM+ARIMA, 18w deep feedforward
    "use_4w_hybrid": True,   # Use Hybrid4WeekModel for n_weeks in 4w band when TF available
    "use_18w_deep": True,   # Use DeepSeasonLongModel for long horizon when TF available
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
# Training on older data (pre-2011) may teach outdated patterns.
# Default training window: end_year = current NFL season (single source: CURRENT_NFL_SEASON).
# start_year defaults are explicit so training windows are coherent and overridable.
TRAINING_START_YEAR_DEFAULT = 2014   # Default first year for training (10+ seasons for 18w model)
TRAINING_END_YEAR_DEFAULT = CURRENT_NFL_SEASON   # Latest season (same as CURRENT_NFL_SEASON)
TRAINING_YEARS = {
    "start_year": TRAINING_START_YEAR_DEFAULT,
    "end_year": TRAINING_END_YEAR_DEFAULT,
    "test_years": [TRAINING_END_YEAR_DEFAULT],   # Latest season held out for testing
    "min_years": 5,
}

# Requirement-derived minimum training seasons per horizon (see docs/fantasy requirements)
MIN_TRAINING_SEASONS_1W = 3   # 1-week model: min 3, optimal 5+
MIN_TRAINING_SEASONS_18W = 8  # 18-week model: min 8, optimal 10+
MIN_TRAINING_SEASONS_4W = 5  # 4-week horizon (LSTM+ARIMA): min 5, optimal 8+
# Per-position minimum players for training (requirements: ~30 QB, 60 RB, 70 WR, 30 TE)
MIN_PLAYERS_PER_POSITION = {"QB": 30, "RB": 60, "WR": 70, "TE": 30}

# Alternative training windows (end_year = CURRENT_NFL_SEASON; start_year explicit)
TRAINING_WINDOW_PRESETS = {
    "modern": {"start_year": MIN_HISTORICAL_YEAR, "end_year": TRAINING_END_YEAR_DEFAULT},
    "balanced": {"start_year": TRAINING_START_YEAR_DEFAULT, "end_year": TRAINING_END_YEAR_DEFAULT},
    "extended": {"start_year": 2010, "end_year": TRAINING_END_YEAR_DEFAULT},
    "full": {"start_year": 2000, "end_year": TRAINING_END_YEAR_DEFAULT},
}

# Feature engineering rolling windows (rubric-required windows included).
# Keep required 3,4,5,8 and include 12 for longer-term trends.
ROLLING_WINDOWS = [3, 4, 5, 8, 12]
LAG_WEEKS = [1, 2, 3, 4]  # Lag features

# Prediction settings
MAX_PREDICTION_WEEKS = 18
MIN_GAMES_FOR_PREDICTION = 4  # Minimum historical games needed

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
FEATURE_VERSION = "7"  # v7: Residual deep net, Huber loss, target transforms, stability selection, bye/short-week features, learned blend weights, isotonic calibration, player embeddings
FEATURE_VERSION_FILENAME = "feature_version.txt"

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
