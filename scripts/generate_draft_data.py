#!/usr/bin/env python3
"""Generate static JSON data files for the fantasy draft web app.

Produces two categories of data:

1. **Model Performance** (``model_performance.json``):
   Previous season's out-of-sample predictions alongside actual results,
   demonstrating how the model performed on truly unseen data.

2. **Upcoming Season Projections** (``players_{POS}.json``):
   ML model predictions for the upcoming season.  When the schedule has
   not been released yet the projection fields are set to ``null`` so the
   frontend can display a "pending" state instead of extrapolating.

No extrapolation is ever performed — all numbers come from the ML model
or from real game results.

Usage:
    python scripts/generate_draft_data.py
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.utils.player_names import board_name

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest_results"

# Floor/ceiling spread formula for _resolve_projection() (GAPS.md §7.4
# follow-up, 2026-08-05).
#
# History: the original formula was `spread = 1.5 * fp_std * sqrt(17)`,
# where fp_std is a player's own prior-season week-to-week volatility.
# Checked against 3,060 real player-seasons (2019-2025, real production
# PreseasonProjector predictions vs real season totals): only 42.2%
# empirical coverage against the ~86.6% the 1.5 constant implied (a
# Gaussian z=1.5). A first fix (bumping the constant to empirically-
# derived z=4.0) corrected the AVERAGE coverage but left a second,
# deeper problem: fp_std turned out to be a weak, backwards-tilted
# confidence signal -- once you control for the SIZE of the projection
# (pred_total), fp_std explains almost nothing about how wrong a
# prediction turns out to be (partial correlation ~0.05, noise), and
# coverage grouped by fp_std quintile ranged from 76% (low fp_std,
# under-covered) to 96% (high fp_std, over-covered) even after the z=4.0
# fix -- calibrated on average, badly miscalibrated player-by-player.
#
# What actually predicts relative error well: log(pred_total) itself
# (bigger/more-established projections are proportionally more accurate,
# Spearman rho=-0.47) and the model's own confidence_score (rho=-0.26 in
# a quantile-regression fit, right-signed and significant, p<1e-5).
# fp_std was dropped entirely -- it added no independent signal once
# these two were included.
#
# Coefficients below are from a quantile regression (q=0.866, matching
# the original z=1.5-implied target) of relative error
# (|actual-pred_total|/pred_total) on log(pred_total) + confidence_score,
# fit on all 3,060 available player-seasons. Validated on a genuine
# holdout split (fit 2019-2022, tested on unseen 2023-2025): coverage
# 89.1% (vs 87.9% for the old fp_std z=4.0 formula on the same holdout),
# and critically, coverage-by-fp_std-quintile spread dropped from 22
# points (73%-95%) to 4 points (87%-91%) -- the miscalibration-by-tier
# problem is actually fixed, not just re-averaged. Spread-vs-actual-error
# correlation also improved slightly (0.36 vs 0.32).
#
# Follow-up: checked coverage by position for the pooled (position-blind)
# fit above and found a real, if smaller, residual gap -- QB 82.5%, RB
# 83.5%, WR 85.9%, TE 92.1% (TE over-covered, QB/RB under-covered).
# Refit with position as a covariate (QB as the reference level): closed
# the gap to QB 84.6% / RB 86.3% / TE 86.6% / WR 86.2%, all within ~2pp
# of the 86.6% target, with fp_std-quintile coverage unchanged (still a
# tight ~5pp range). Cheap to add and a clear improvement, so shipped.
# --- Asymmetric floor/ceiling (GAPS.md §11.2.C follow-up, 2026-08-06) ---
#
# The formula above (still kept, see FLOOR_CEILING_REL_SPREAD_* below) was
# symmetric: one relative spread applied equally above and below the point
# total. Checked whether that's the right shape by testing real 2025
# backtest residuals for bimodality (Pfister's coefficient, all 4
# positions) -- not classically bimodal (all well below the 0.555
# threshold), but genuinely right-skewed and heavy-tailed (positive skew
# 0.48-1.37, real excess kurtosis on RB/WR/TE). A symmetric interval on a
# skewed distribution is structurally mis-shaped.
#
# Confirmed directly: refit the same real PreseasonProjector-vs-actual
# dataset (2,034 player-seasons, 2019-2025, `scripts/calibrate_floor_
# ceiling.py`) as two one-sided quantile regressions (q=0.067 for floor,
# q=0.933 for ceiling, matching the same 86.6% target) instead of one
# symmetric one. On a genuine holdout (fit 2019-2022, test 2023-2025):
# the symmetric formula's floor was almost never actually breached (2.5%
# vs. the 6.7% it was supposed to be -- needlessly conservative) while
# its ceiling was slightly too tight (7.6% vs 6.7%). The asymmetric fit
# gets both sides close to target (7.5% / 7.5%), a 3x reduction in
# per-side miscalibration (sum of |actual-target| gap: 0.051 -> 0.016).
# Overall coverage dropped slightly (89.9% -> 85.0%, vs. an 86.6% target)
# -- expected and acceptable, since the symmetric formula's 89.9% was
# itself an artifact of over-covering on the floor side while under-
# covering on the ceiling side, not a real calibration win.
# --- Refit to a 50% interval, 2026-08-28 ---
#
# The 86.6% bands were correctly calibrated (89.1% holdout coverage) and
# simultaneously useless: measured on the live 2026 board their median width
# was 206% of the projection, p90 326% -- e.g. Stafford proj 330, floor 123,
# ceiling 490. A band spanning 4x contains the outcome precisely because it
# says almost nothing.
#
# Narrowed to a 50% two-sided interval (floor q=0.25, ceiling q=0.75) via
# `python scripts/calibrate_floor_ceiling.py --target-coverage 0.50`.
# Holdout (fit 2019-2022, test 2023-2025): floor breached 26.6%, ceiling
# breached 23.8% against a 25% target; per-side miscalibration 0.029.
#
# This does NOT make the projection more certain -- it reports a narrower
# quantile of the same distribution. The band now means "half the time the
# outcome lands in here", and the old 86.6% width remains the honest picture
# of how uncertain a season-total projection actually is (MAE ~43 on a mean
# actual near 100). The previous coefficients are preserved below.
FLOOR_ASYM_COEF = {
    "const": -1.464328, "log_pred": 0.172331, "confidence_score": 0.347607,
    "pos_RB": -0.073935, "pos_WR": -0.050796, "pos_TE": -0.032941,
}
CEILING_ASYM_COEF = {
    "const": 1.493868, "log_pred": -0.238032, "confidence_score": 0.187379,
    "pos_RB": -0.179821, "pos_WR": -0.122886, "pos_TE": -0.274811,
}
# Superseded 86.6%-coverage coefficients, kept so the wider interval can be
# restored or compared without re-running the fit.
FLOOR_ASYM_COEF_866 = {
    "const": -1.307583, "log_pred": 0.108491, "confidence_score": 0.072483,
    "pos_RB": -0.071201, "pos_WR": -0.017547, "pos_TE": 0.033451,
}
CEILING_ASYM_COEF_866 = {
    "const": 4.383422, "log_pred": -0.662733, "confidence_score": -0.081698,
    "pos_RB": -0.126647, "pos_WR": -0.306856, "pos_TE": -0.369531,
}
FLOOR_CEILING_DEFAULT_CONFIDENCE = 0.7
# Sanity clamps on the fitted relative error -- guards against extreme
# extrapolation for pred_total values far outside the fit data's range.
ASYM_REL_MIN, ASYM_REL_MAX = -0.95, 5.0


def _asym_rel_error(coef: dict, log_total: float, conf: float, position: str) -> float:
    rel = coef["const"] + coef["log_pred"] * log_total + coef["confidence_score"] * conf
    rel += coef.get(f"pos_{position}", 0.0)
    return max(ASYM_REL_MIN, min(ASYM_REL_MAX, rel))


def _floor_ceiling(total: float, confidence, position: str = None) -> tuple:
    """Asymmetric floor/ceiling for a season-total projection, given
    (optionally) a per-player model confidence score, and position.
    Returns (floor, ceiling). See the fit methodology and rationale in
    the comment above FLOOR_ASYM_COEF."""
    conf = (
        FLOOR_CEILING_DEFAULT_CONFIDENCE
        if confidence is None or pd.isna(confidence)
        else float(confidence)
    )
    total = float(total)
    log_total = np.log(max(total, 1.0))
    floor_rel = _asym_rel_error(FLOOR_ASYM_COEF, log_total, conf, position)
    ceiling_rel = _asym_rel_error(CEILING_ASYM_COEF, log_total, conf, position)
    floor = max(0.0, total * (1 + floor_rel))
    ceiling = max(total, total * (1 + ceiling_rel))
    floor = min(floor, total)
    return floor, ceiling


# --- Legacy symmetric formula, kept for reference/rollback only; no
# longer called anywhere in this file as of the asymmetric fix above. ---
FLOOR_CEILING_REL_SPREAD_INTERCEPT = 2.620
FLOOR_CEILING_REL_SPREAD_LOG_PRED_COEF = -0.335
FLOOR_CEILING_REL_SPREAD_CONF_COEF = 0.022
FLOOR_CEILING_REL_SPREAD_POSITION_COEF = {"QB": 0.0, "RB": -0.121, "WR": -0.172, "TE": -0.295}
FLOOR_CEILING_REL_SPREAD_MIN = 0.15
FLOOR_CEILING_REL_SPREAD_MAX = 3.0


def _floor_ceiling_spread(total: float, confidence, position: str = None) -> float:
    """Absolute spread for floor/ceiling, given a season-total projection,
    (optionally) a per-player model confidence score, and position. See
    FLOOR_CEILING_REL_SPREAD_* above for how these coefficients were derived."""
    conf = (
        FLOOR_CEILING_DEFAULT_CONFIDENCE
        if confidence is None or pd.isna(confidence)
        else float(confidence)
    )
    pos_coef = FLOOR_CEILING_REL_SPREAD_POSITION_COEF.get(position, 0.0)
    log_total = np.log(max(float(total), 1.0))
    rel_spread = (
        FLOOR_CEILING_REL_SPREAD_INTERCEPT
        + FLOOR_CEILING_REL_SPREAD_LOG_PRED_COEF * log_total
        + FLOOR_CEILING_REL_SPREAD_CONF_COEF * conf
        + pos_coef
    )
    rel_spread = max(FLOOR_CEILING_REL_SPREAD_MIN, min(FLOOR_CEILING_REL_SPREAD_MAX, rel_spread))
    return rel_spread * float(total)


def _load_authoritative_position_map():
    """Load authoritative position labels from roster snapshots."""
    try:
        from src.utils.database import DatabaseManager
        return DatabaseManager().get_authoritative_player_positions()
    except Exception:
        return {}


def _apply_authoritative_positions(df: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    """Overwrite noisy position labels with authoritative roster labels."""
    if df.empty or not pos_map or "player_id" not in df.columns:
        return df
    out = df.copy()
    authoritative = out["player_id"].astype(str).map(pos_map)
    if "position" in out.columns:
        out["position"] = authoritative.where(authoritative.notna(), out["position"])
    else:
        out["position"] = authoritative
    return out


# ---------------------------------------------------------------------------
# 1. Model Performance: previous season out-of-sample predictions vs actuals
# ---------------------------------------------------------------------------

def _load_latest_backtest_json(season: int = None):
    """Load the latest backtest JSON for a given (or most recent) season."""
    if not BACKTEST_DIR.exists():
        return None
    files = sorted(BACKTEST_DIR.glob("backtest_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        try:
            with open(p) as f:
                data = json.load(f)
            if season is None or data.get("season") == season:
                return data
        except Exception:
            continue
    return None


def _load_ts_backtest_predictions(season: int = None):
    """Load per-player per-week ts-backtest predictions CSV."""
    if not BACKTEST_DIR.exists():
        return None
    pattern = f"ts_backtest_{season}_*_predictions.csv" if season else "ts_backtest_*_predictions.csv"
    files = sorted(BACKTEST_DIR.glob(pattern),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return pd.read_csv(files[0])
    except Exception:
        return None


def _auto_generate_ts_backtest(season: int):
    """Auto-generate ts_backtest predictions for a season if none exist.

    Runs the expanding-window time-series backtester which:
    - Trains on all seasons before ``season`` (zero data leakage)
    - Predicts each week of ``season`` independently
    - Saves predictions CSV and metrics JSON to backtest_results/
    """
    print(f"  Auto-generating ts_backtest for {season} season...")
    print(f"  This trains on all seasons before {season} (zero data leakage).")
    try:
        from src.evaluation.ts_backtester import run_ts_backtest
        pred_df, results = run_ts_backtest(
            season=season,
            model_type="ridge",
            positions=None,
            verbose=True,
        )
        # Verify leakage safety from results
        diag = results.get("diagnostics", {})
        if diag.get("leakage_check_passed"):
            print(f"  Verified: {season} out-of-sample predictions "
                  f"(leakage check passed, scaling fit on train only)")
        print(f"  Generated {len(pred_df)} predictions for {season} season")
        return True
    except Exception as e:
        print(f"  WARNING: Auto-generation of ts_backtest failed: {e}")
        print(f"  You can run manually: python scripts/run_ts_backtest.py --season {season}")
        return False


def generate_model_performance():
    """Create model_performance.json showing previous season predictions vs actuals."""
    from src.utils.nfl_calendar import get_current_nfl_season
    current_season = get_current_nfl_season()
    # Previous completed season
    prev_season = current_season

    # Try ts-backtest predictions (per-player, per-week granularity)
    ts_preds = _load_ts_backtest_predictions(prev_season)
    if ts_preds is None:
        # Auto-generate ts_backtest for the previous season if missing
        print(f"  No ts_backtest predictions found for {prev_season} season.")
        if _auto_generate_ts_backtest(prev_season):
            ts_preds = _load_ts_backtest_predictions(prev_season)

    if ts_preds is None:
        # Fall back to one season earlier
        ts_preds = _load_ts_backtest_predictions(prev_season - 1)
        if ts_preds is not None:
            prev_season = prev_season - 1
    pos_map = _load_authoritative_position_map()
    if ts_preds is not None and not ts_preds.empty:
        ts_preds = _apply_authoritative_positions(ts_preds, pos_map)

    # Also load aggregate backtest metrics — prefer ts_backtest JSON over generic backtest
    backtest_json = None
    if BACKTEST_DIR.exists():
        ts_jsons = sorted(
            BACKTEST_DIR.glob(f"ts_backtest_{prev_season}_*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        for p in ts_jsons:
            try:
                with open(p) as f:
                    backtest_json = json.load(f)
                break
            except Exception:
                continue
    if backtest_json is None:
        backtest_json = _load_latest_backtest_json(prev_season)
    if backtest_json is None:
        backtest_json = _load_latest_backtest_json()  # any season

    payload = {
        "season": prev_season,
        "has_per_player_predictions": ts_preds is not None and len(ts_preds) > 0,
        "aggregate_metrics": {},
        "by_position": {},
        "top_performers": {},
        "per_player_season_totals": [],
        "leakage_verification": {
            "status": "verified",
            "training_seasons": f"all seasons before {prev_season}",
            "test_season": prev_season,
            "methodology": "expanding-window weekly refit with leakage checks per fold",
            "note": f"Model was trained exclusively on data before {prev_season}. "
                    f"No {prev_season} data was used in training, feature engineering, or scaling.",
        },
    }

    if backtest_json:
        payload["aggregate_metrics"] = backtest_json.get("metrics", {})
        payload["by_position"] = backtest_json.get("by_position", {})
        tp = backtest_json.get("top_performers", {})
        for pos in ["QB", "RB", "WR", "TE"]:
            if pos in tp and "top_10_actual" in tp[pos]:
                payload["top_performers"][pos] = tp[pos]["top_10_actual"]

    # Build per-player season aggregates from ts-backtest predictions
    if ts_preds is not None and not ts_preds.empty:
        required_cols = {"player_id", "name", "position", "predicted", "actual"}
        if required_cols.issubset(set(ts_preds.columns)):
            ts_preds = ts_preds[ts_preds["position"].isin(["QB", "RB", "WR", "TE"])]
            agg = ts_preds.groupby(["player_id", "name", "position"]).agg(
                predicted_total=("predicted", "sum"),
                actual_total=("actual", "sum"),
                games=("actual", "count"),
                predicted_ppg=("predicted", "mean"),
                actual_ppg=("actual", "mean"),
            ).reset_index()

            if "team" in ts_preds.columns:
                team_map = ts_preds.groupby("player_id")["team"].last().to_dict()
                agg["team"] = agg["player_id"].map(team_map).fillna("")
            else:
                agg["team"] = ""

            agg["error"] = (agg["predicted_total"] - agg["actual_total"]).round(1)
            agg["abs_error"] = agg["error"].abs()

            # Sort by actual total (best performers first)
            agg = agg.sort_values("actual_total", ascending=False)

            records = []
            for _, row in agg.head(200).iterrows():
                records.append({
                    "player_id": str(row["player_id"]),
                    "name": row["name"],
                    "position": row["position"],
                    "team": row.get("team", ""),
                    "predicted_total": round(float(row["predicted_total"]), 1),
                    "actual_total": round(float(row["actual_total"]), 1),
                    "predicted_ppg": round(float(row["predicted_ppg"]), 1),
                    "actual_ppg": round(float(row["actual_ppg"]), 1),
                    "games": int(row["games"]),
                    "error": round(float(row["error"]), 1),
                })
            payload["per_player_season_totals"] = records

    out_path = DATA_DIR / "model_performance.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)
    print(f"  Wrote model_performance.json ({len(payload.get('per_player_season_totals', []))} players, season {prev_season})")


# ---------------------------------------------------------------------------
# 2. Upcoming season projections (ML predictions, never extrapolated)
# ---------------------------------------------------------------------------

def load_season_data(season: int):
    """Load regular season data (weeks 1-18) from parquet for a given season."""
    parquet_path = DATA_DIR / "daily_predictions.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    df = _apply_authoritative_positions(df, _load_authoritative_position_map())
    mask = (
        (df["season"] == season)
        & (df["week"] <= 18)
        & (df["position"].isin(["QB", "RB", "WR", "TE"]))
    )
    return df[mask]


def aggregate_player_stats(season_df):
    """Aggregate per-player season totals from weekly data."""
    agg_dict = {
        "total_fp": ("fantasy_points", "sum"),
        "games_played": ("fantasy_points", "count"),
        "ppg": ("fantasy_points", "mean"),
        "fp_std": ("fantasy_points", "std"),
        "util_mean": ("utilization_score", "mean"),
    }

    optional_cols = {
        "vol_mean": ("weekly_volatility", "mean"),
        "consistency_mean": ("consistency_score", "mean"),
        "confidence_mean": ("confidence_score", "mean"),
        "cv_mean": ("coefficient_of_variation", "mean"),
    }
    for key, (col, func) in optional_cols.items():
        if col in season_df.columns:
            agg_dict[key] = (col, func)

    agg = season_df.groupby(["player_id", "name", "team", "position"]).agg(
        **agg_dict
    ).reset_index()

    agg["fp_std"] = agg["fp_std"].fillna(0)

    for col in ["vol_mean", "consistency_mean", "confidence_mean", "cv_mean"]:
        if col not in agg.columns:
            agg[col] = 0.0

    return agg


def compute_risk_scores(agg):
    """Risk score 0-100 where higher = more risky."""
    agg["risk_vol"] = 0.0
    agg["risk_cv"] = 0.0
    agg["risk_consistency"] = 0.0
    agg["risk_games"] = 0.0

    for pos in ["QB", "RB", "WR", "TE"]:
        mask = agg["position"] == pos
        subset = agg.loc[mask]
        if len(subset) == 0:
            continue

        vol_min, vol_max = subset["vol_mean"].min(), subset["vol_mean"].max()
        vol_range = vol_max - vol_min if vol_max > vol_min else 1
        agg.loc[mask, "risk_vol"] = (subset["vol_mean"] - vol_min) / vol_range

        cv_min, cv_max = subset["cv_mean"].min(), subset["cv_mean"].max()
        cv_range = cv_max - cv_min if cv_max > cv_min else 1
        agg.loc[mask, "risk_cv"] = (subset["cv_mean"] - cv_min) / cv_range

        cons_max = subset["consistency_mean"].max()
        if cons_max > 0:
            agg.loc[mask, "risk_consistency"] = 1 - (
                subset["consistency_mean"] / cons_max
            )

        gp_max = subset["games_played"].max()
        if gp_max > 0:
            agg.loc[mask, "risk_games"] = 1 - (subset["games_played"] / gp_max)

    agg["risk_score"] = (
        agg["risk_vol"] * 30
        + agg["risk_cv"] * 25
        + agg["risk_consistency"] * 25
        + agg["risk_games"] * 20
    ).round(0).clip(0, 100).astype(int)

    return agg


def add_feature_importance(agg):
    """Attach top features from top10_features_per_position.json."""
    features_path = MODELS_DIR / "top10_features_per_position.json"
    if not features_path.exists():
        agg["key_features"] = [[] for _ in range(len(agg))]
        agg["feature_importance_rank"] = [{} for _ in range(len(agg))]
        return agg

    with open(features_path) as f:
        features = json.load(f)

    def get_key_features(pos):
        return [feat["feature"] for feat in features.get(pos, [])][:5]

    def get_importance_dict(pos):
        return {
            feat["feature"]: round(feat["importance"], 4)
            for feat in features.get(pos, [])
        }

    agg["key_features"] = agg["position"].map(get_key_features)
    agg["feature_importance_rank"] = agg["position"].map(get_importance_dict)
    return agg


def _check_schedule_available(upcoming_season: int) -> bool:
    """Check if the NFL schedule is available for the upcoming season."""
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        return db.has_schedule_for_season(upcoming_season)
    except Exception:
        return False


def _load_ml_predictions(upcoming_season: int):
    """Try to load ML predictions from daily_predictions.parquet for the upcoming season."""
    parquet_path = DATA_DIR / "daily_predictions.parquet"
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        df = _apply_authoritative_positions(df, _load_authoritative_position_map())
        upcoming = df[
            (df["season"] == upcoming_season)
            & (df["position"].isin(["QB", "RB", "WR", "TE"]))
        ]
        if upcoming.empty:
            return None
        # Check if these are actual ML predictions (have projection_18w column with non-null values)
        proj_cols = [c for c in ["projection_1w", "projection_4w", "projection_18w"] if c in upcoming.columns]
        if not proj_cols:
            return None
        has_predictions = False
        for col in proj_cols:
            if upcoming[col].notna().any():
                has_predictions = True
                break
        if not has_predictions:
            return None
        return upcoming
    except Exception:
        return None


PRESEASON_MODEL = os.getenv("NFL_PRESEASON_MODEL", "step8")


def _load_step8_projections(upcoming_season: int):
    """Step 8 season totals for `upcoming_season`, with board metadata.

    Step 8 is the production season model as of 2026-08-28: first of four arms
    on the corrected 11-fold walk-forward (mean MAE rank 1.25 vs candidate
    1.75, phase7 3.00, production 4.00) and best on rookies (39.81 vs 53.29 for
    PreseasonProjector, which is what this replaced).

    Uses the SAME feature builder as training via `inference_season=`, rather
    than the parallel hand-written query PreseasonProjector needs. Returns the
    identical shape the caller already consumes -- player_id, pred_total,
    confidence_score, support_class -- so the swap is contained here.
    """
    import pandas as pd
    from src.utils.database import DatabaseManager
    from src.models.preseason_features import build_multiyear_season_pairs
    from src.models.single_week_ppr.season_availability import load_player_seasons
    from src.models.season_step8 import (
        Step8SeasonModel, possible_games_for_players, with_board_metadata,
    )

    db = DatabaseManager()
    panel = load_player_seasons()
    seasons = list(range(2019, upcoming_season))
    pairs = build_multiyear_season_pairs(db, seasons, inference_season=upcoming_season)
    train = pairs[pairs["target_season"] < upcoming_season]
    infer = pairs[pairs["target_season"] == upcoming_season].copy()
    if train.empty or infer.empty:
        return None

    model = Step8SeasonModel().fit(train, panel, before_season=upcoming_season)
    pg = possible_games_for_players(infer, upcoming_season)
    preds = model.predict(infer, possible_games=pg)
    out = with_board_metadata(infer, preds)
    # The games half of `games x rate`, which the model computes and used to
    # throw away. A consumer projecting a SINGLE week needs it: the season
    # total carries the games a player is expected to miss, and dividing by 17
    # spreads that discount across a week he is being started in.
    out["expected_games"] = (model.availability.predict_rate(infer)
                             * np.asarray(pg, dtype=float))
    return out.rename(columns={"predicted_total": "pred_total"})[
        ["player_id", "pred_total", "confidence_score", "support_class",
         "expected_games"]
    ]


def _load_preseason_projections(upcoming_season: int, prev_season: int):
    """Season-total projections for the draft board.

    Dispatches on PRESEASON_MODEL (env NFL_PRESEASON_MODEL, default "step8").
    The PreseasonProjector path below is retained for comparison and rollback;
    it is DEMOTED and last of four arms -- see that module's docstring.
    """
    if PRESEASON_MODEL == "step8":
        try:
            df = _load_step8_projections(upcoming_season)
        except Exception:
            return None
        return df if df is not None and not df.empty else None
    return _load_preseason_projector_projections(upcoming_season, prev_season)


def _load_preseason_projector_projections(upcoming_season: int, prev_season: int):
    """Load PreseasonProjector season-total predictions for upcoming_season.

    DEMOTED 2026-08-28 -- retained for rollback/comparison only. Reachable via
    NFL_PRESEASON_MODEL=preseason_projector.

    Reuses scripts/snake_draft_sim.py's load_preseason_projections() rather
    than re-deriving the DB query it depends on — that function's feature
    query exactly mirrors PreseasonProjector's training-time query on
    purpose (see the comment above it): a simplified/partial query silently
    produces severely under-estimated projections (missing features
    zero-fill, which after StandardScaler centering skews everything low).

    Returns a DataFrame indexed by player_id with pred_total (and
    confidence_score/support_class when available), or None if the
    projector/DB path fails for any reason — callers should treat that as
    "fall back to the existing projection_18w tier", not an error.
    """
    from scripts.snake_draft_sim import load_preseason_projections

    # season - 1 is queried internally as the "prior" (completed) season, so
    # passing prev_season + 1 guarantees it queries prev_season itself, even
    # when main()'s own prev_season fallback (prev_season - 1) has fired
    # because the naive prev_season had no weekly data yet.
    #
    # projection_mode="ml" (not "auto") is deliberate: "auto" silently
    # degrades to ppg*17 internally when the model file is missing/broken,
    # and its return value doesn't distinguish that from a real ML
    # prediction — which would make every player in this file look like it
    # came from "preseason_model" even when the model was never loaded.
    # "ml" raises instead, so a missing/broken model surfaces here and this
    # function can correctly report "no preseason projections" and let the
    # caller fall through to the projection_18w/ppg tiers.
    try:
        df = load_preseason_projections(
            season=prev_season + 1, adp_df=None, projection_mode="ml",
        )
    except Exception:
        return None
    if df is None or df.empty or "pred_total" not in df.columns:
        return None
    return df


def _load_oos_prediction_map():
    """Load OOS predictions from model_performance.json for enriching player files."""
    perf_path = DATA_DIR / "model_performance.json"
    if not perf_path.exists():
        return {}
    try:
        with open(perf_path) as f:
            perf = json.load(f)
        return {
            p["player_id"]: p
            for p in perf.get("per_player_season_totals", [])
        }
    except Exception:
        return {}


def _resolve_projection(row, has_preseason_projection: bool, has_ml_predictions: bool,
                        schedule_available: bool):
    """Pick a season-total projection for one player, preferring the
    PreseasonProjector season-total model over the projection_18w fallback.

    Returns (proj_total, proj_ppg, proj_floor, proj_ceiling, source_label).

    PreseasonProjector consumes no schedule/matchup data (it's trained on
    prior-season aggregate stats only), so unlike projection_18w it is NOT
    gated behind schedule_available — that's a deliberate behavior change:
    players get a real number earlier in the offseason, before the
    schedule drops, instead of waiting on a tier that doesn't actually need
    the schedule to begin with.
    """
    if has_preseason_projection:
        preseason_total = row.get("preseason_projection_total")
        if pd.notna(preseason_total):
            total = float(preseason_total)
            confidence = row.get("preseason_confidence")
            proj_total = round(total, 1)
            proj_ppg = round(total / 17, 1)
            floor, ceiling = _floor_ceiling(total, confidence, row.get("position"))
            proj_floor = round(floor, 1)
            proj_ceiling = round(ceiling, 1)
            return proj_total, proj_ppg, proj_floor, proj_ceiling, "preseason_model"

    if has_ml_predictions and schedule_available:
        p18 = row.get("projection_18w")
        if pd.notna(p18):
            total = float(p18)
            proj_total = round(total, 1)
            proj_ppg = round(total / 17, 1)
            # No per-player confidence_score available on this path (that's
            # specific to PreseasonProjector's predict_with_details) --
            # _floor_ceiling falls back to FLOOR_CEILING_DEFAULT_CONFIDENCE.
            floor, ceiling = _floor_ceiling(total, None, row.get("position"))
            proj_floor = round(floor, 1)
            proj_ceiling = round(ceiling, 1)
            return proj_total, proj_ppg, proj_floor, proj_ceiling, "weekly_18w"

    return None, None, None, None, None


def output_position_files(agg, upcoming_season: int, schedule_available: bool,
                          has_ml_predictions: bool, prev_season: int,
                          has_preseason_projection: bool = False):
    """Write per-position JSON files.

    Season-total projections prefer the PreseasonProjector season-total
    model (see _resolve_projection); projection_18w (a single-week
    prediction scaled by 18, not a season-level model) is the fallback when
    the preseason projector is unavailable for a player/position. When no
    schedule is available for the upcoming season, projection_18w-sourced
    fields are null so the frontend shows a "pending" state — the preseason
    model's fields are not schedule-gated (see _resolve_projection).

    During the off-season, enriches each player with out-of-sample prediction
    data from the previous season (model_predicted_total, actual_total, error).
    """
    # Load OOS prediction data for off-season enrichment
    oos_map = _load_oos_prediction_map()

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = agg[agg["position"] == pos].copy()

        # Sort: prefer preseason-model projection, then projection_18w, then previous-season PPG
        pos_has_preseason = (
            has_preseason_projection
            and "preseason_projection_total" in pos_df.columns
            and pos_df["preseason_projection_total"].notna().any()
        )
        if pos_has_preseason:
            sort_col = "preseason_projection_total"
            pos_df = pos_df.sort_values(sort_col, ascending=False, na_position="last")
        elif has_ml_predictions and "projection_18w" in pos_df.columns:
            sort_col = "projection_18w"
            pos_df = pos_df.sort_values(sort_col, ascending=False, na_position="last")
        else:
            pos_df = pos_df.sort_values("ppg", ascending=False, na_position="last")

        players = []
        for rank, (_, row) in enumerate(pos_df.iterrows(), 1):
            proj_total, proj_ppg, proj_floor, proj_ceiling, projection_source = (
                _resolve_projection(row, has_preseason_projection, has_ml_predictions,
                                   schedule_available)
            )

            # OOS prediction data from ts_backtest (for off-season display)
            player_id = str(row["player_id"])
            oos = oos_map.get(player_id, {})

            players.append({
                "player_id": player_id,
                "name": row["name"],
                "team": row["team"],
                "position": row["position"],
                "bye_week": None,
                "adp": rank,
                "projection_points_total": proj_total,
                "projection_points_per_game": proj_ppg,
                "projection_floor": proj_floor,
                "projection_ceiling": proj_ceiling,
                "projection_source": projection_source,
                "support_class": row.get("preseason_support_class") or None,
                "expected_games": (round(float(row["expected_games"]), 2)
                                   if pd.notna(row.get("expected_games"))
                                   else None),
                "projection_model": PRESEASON_MODEL,
                "risk_score": int(row["risk_score"]) if pd.notna(row.get("risk_score")) else None,
                "injury_flag": False,
                "age": None,
                "key_features": row.get("key_features", []),
                "feature_importance_rank": row.get("feature_importance_rank", {}),
                "uses_schedule": schedule_available,
                "prev_season": prev_season,
                "prev_season_ppg": round(float(row["ppg"]), 1) if pd.notna(row.get("ppg")) else None,
                "prev_season_total_fp": round(float(row["total_fp"]), 1) if pd.notna(row.get("total_fp")) else None,
                "prev_season_games": int(row["games_played"]) if pd.notna(row.get("games_played")) else None,
                "has_ml_prediction": proj_total is not None,
                # OOS prediction data (previous season model accuracy)
                "model_predicted_total_prev_season": oos.get("predicted_total"),
                "actual_total_prev_season": oos.get("actual_total"),
                "prediction_error_prev_season": oos.get("error"),
            })
        out_path = DATA_DIR / f"players_{pos}.json"
        with open(out_path, "w") as f:
            json.dump(_json_safe(players), f, indent=2, allow_nan=False)
        n_preseason = sum(1 for p in players if p["projection_source"] == "preseason_model")
        n_weekly = sum(1 for p in players if p["projection_source"] == "weekly_18w")
        print(f"  Wrote {len(players)} players to {out_path.name}"
              f" (preseason model: {n_preseason}, weekly_18w fallback: {n_weekly},"
              f" pending: {len(players) - n_preseason - n_weekly})")


def generate_schedule_impact(upcoming_season: int, schedule_available: bool):
    """Generate schedule_impact.json."""
    if schedule_available:
        payload = {
            "schedule_incorporated": True,
            "reason": f"The {upcoming_season} NFL schedule has been incorporated.",
            "season": upcoming_season,
            "schedule_available": True,
        }
    else:
        payload = {
            "schedule_incorporated": False,
            "reason": (
                f"The {upcoming_season} NFL schedule has not been released. "
                f"Projections will be available once the schedule is out. "
                f"No extrapolations are used."
            ),
            "season": upcoming_season,
            "schedule_available": False,
        }
    out_path = DATA_DIR / "schedule_impact.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)
    print(f"  Wrote {out_path.name}")


def generate_model_metadata_frontend(upcoming_season: int, prev_season: int,
                                     schedule_available: bool, has_ml_predictions: bool):
    """Create draft_model_metadata.json for the frontend methodology section."""
    meta_path = MODELS_DIR / "model_metadata.json"
    backtest_path = DATA_DIR / "advanced_model_results.json"
    features_path = MODELS_DIR / "top10_features_per_position.json"

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    backtest = {}
    if backtest_path.exists():
        with open(backtest_path) as f:
            backtest = json.load(f)

    features = {}
    if features_path.exists():
        with open(features_path) as f:
            features = json.load(f)

    train_seasons = meta.get("train_seasons", [])
    training_window = ""
    if train_seasons:
        training_window = f"{train_seasons[0]}-{train_seasons[-1]}"

    if has_ml_predictions and schedule_available:
        target_def = f"PPR fantasy points — ML model predictions for {upcoming_season} season"
        data_basis = (
            f"All projections are ML model outputs for the {upcoming_season} season. "
            f"Model performance on the {prev_season} held-out season is shown in the "
            f"Model Performance tab."
        )
    else:
        target_def = f"PPR fantasy points — awaiting {upcoming_season} NFL schedule"
        data_basis = (
            f"The {upcoming_season} NFL schedule has not been released. "
            f"Projections will appear once the schedule is available. "
            f"No extrapolations are used. The Model Performance tab shows how "
            f"the model performed on the {prev_season} season (out-of-sample)."
        )

    payload = {
        "target_definition": target_def,
        "training_data_range": training_window or "2006-2024",
        "positions": ["QB", "RB", "WR", "TE"],
        "schedule_incorporated": schedule_available,
        "upcoming_season": upcoming_season,
        "prev_season": prev_season,
        "has_ml_predictions": has_ml_predictions and schedule_available,
        "version": "v2.0.0",
        "last_updated": meta.get("training_date", ""),
        "training_date": meta.get("training_date"),
        "test_season": meta.get("test_season"),
        "n_features_per_position": meta.get("n_features_per_position", {}),
        "training_metrics": meta.get("training_metrics", {}),
        "backtest_results": backtest.get("backtest_results", {}),
        "top_features": features,
        "methodology": {
            "model_type": "LightGBM ensemble with XGBoost and Ridge regression",
            "training_window": training_window or "2006-2024",
            "test_season": str(meta.get("test_season", prev_season)),
            "scoring_format": "PPR (1 point per reception)",
            "features_description": (
                "50 features per position including utilization scores, "
                "rolling averages (3/5/8 week windows), lag features, "
                "team context, matchup quality, and advanced play-by-play metrics"
            ),
            "overfitting_prevention": [
                "Time-series cross-validation with gap seasons",
                "Recency decay weighting (half-life 2-4 seasons)",
                "Feature selection via stability bootstrap (30 iterations)",
                "Correlation threshold filtering (r > 0.92)",
                "VIF multicollinearity checks (VIF > 10)",
                "Early stopping with 25 rounds patience",
            ],
            "horizons": ["1-week", "4-week", "Full season (18-week)"],
        },
        "data_basis_note": data_basis,
    }
    out_path = DATA_DIR / "draft_model_metadata.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)
    print(f"  Wrote {out_path.name}")


def schedule_transition_check(upcoming_season: int, schedule_available: bool):
    """Detect if the schedule just became available and guide the user.

    Reads the previous ``schedule_impact.json`` to see if the schedule status
    changed from unavailable to available.  When a transition is detected,
    prints instructions for retraining models and regenerating predictions.
    """
    prev_impact_path = DATA_DIR / "schedule_impact.json"
    was_unavailable = True
    if prev_impact_path.exists():
        try:
            with open(prev_impact_path) as f:
                prev = json.load(f)
            was_unavailable = not prev.get("schedule_incorporated", False)
        except Exception:
            pass

    if schedule_available and was_unavailable:
        print()
        print("=" * 60)
        print(f"SCHEDULE DETECTED for {upcoming_season}!")
        print("=" * 60)
        print("The NFL schedule has become available. Next steps:")
        print(f"  1. Retrain models: python -m src.models.train")
        print(f"  2. Generate predictions: python scripts/generate_app_data.py --parquet")
        print(f"  3. Regenerate web data: python scripts/generate_draft_data.py")
        print()

        # Check if models need retraining
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                test_season = meta.get("test_season")
                if test_season and test_season != upcoming_season:
                    print(f"  NOTE: Models were last trained with test_season={test_season}.")
                    print(f"  Retraining is needed to use {upcoming_season} as the prediction target.")
            except Exception:
                pass
        print()



def _json_safe(obj):
    """Replace NaN/Inf with None so the payload is VALID JSON.

    Python's json.dump emits bare `NaN`/`Infinity` by default. That is a
    non-standard extension: Python can read it back, but a browser's
    JSON.parse rejects it outright. docs/data/players_*.json shipped with
    13-58 NaN tokens each, so docs/draft.html failed to parse them and
    rendered "Failed to load" for every visitor (found 2026-08-31).
    """
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# Draft tables carry PFR's team codes; the board carries nflverse's. Only the
# codes that actually differ are listed.
PFR_TEAM_CODES = {"LVR": "LV", "NOR": "NO", "NWE": "NE", "GNB": "GB",
                  "KAN": "KC", "SFO": "SF", "TAM": "TB", "LAR": "LA"}


def _name_from_cfb_slug(slug):
    """"carnell-tate-1" -> "Carnell Tate". Last resort only."""
    if not isinstance(slug, str) or not slug:
        return None
    parts = [w for w in slug.split("-") if not w.isdigit()]
    return " ".join(w.capitalize() for w in parts) or None


def _rookie_identities(draft: pd.DataFrame, ids: pd.DataFrame,
                       combine: pd.DataFrame) -> pd.DataFrame:
    """Name, board id and team for each drafted player.

    draft_picks_v2 carries no name at all, and its player_id is an nflverse
    stub (MEN516487) while every other row on the board is keyed by GSIS id.
    nflverse's id map supplies both; combine data covers the handful it misses;
    the college-reference slug is the last resort. A pick that none of the
    three can name is dropped -- a board row reading "MEN516487" is worse than
    no row.

    Returns `draft_id` (the key the projections are indexed by) alongside
    `player_id` (GSIS where known), because those are not the same key.
    """
    out = draft.copy()
    out["name"] = None

    if ids is not None and not ids.empty:
        lookup = ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
        out = out.merge(lookup[["pfr_id", "name", "gsis_id"]].rename(
            columns={"name": "nflverse_name"}),
            left_on="pfr_player_id", right_on="pfr_id", how="left")
        out["name"] = out["name"].fillna(out["nflverse_name"])
    if "gsis_id" not in out.columns:
        out["gsis_id"] = None

    if combine is not None and not combine.empty:
        lookup = combine.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
        out = out.merge(lookup[["pfr_id", "player_name"]].rename(
            columns={"player_name": "combine_name"}),
            left_on="pfr_player_id", right_on="pfr_id", how="left",
            suffixes=("", "_combine"))
        out["name"] = out["name"].fillna(out["combine_name"])

    out["name"] = out["name"].fillna(
        out["cfb_player_id"].map(_name_from_cfb_slug))
    out["draft_id"] = out["player_id"]
    out["player_id"] = out["gsis_id"].fillna(out["draft_id"])
    out["team"] = out["draft_team"].replace(PFR_TEAM_CODES)
    named = out[out["name"].notna()].copy()
    named["name"] = named["name"].map(board_name)
    return named[["draft_id", "player_id", "name", "team", "position"]]


def _load_rookie_identities(upcoming_season: int) -> pd.DataFrame:
    """Read the draft class and resolve who each pick is."""
    import sqlite3
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    try:
        draft = pd.read_sql(
            "SELECT player_id, position, draft_team, pfr_player_id, "
            "cfb_player_id FROM draft_picks_v2 WHERE draft_season = ? "
            "AND position IN ('QB','RB','WR','TE') AND player_id IS NOT NULL",
            conn, params=[int(upcoming_season)])
        combine = pd.read_sql(
            "SELECT pfr_id, player_name FROM combine_data_v2 "
            "WHERE pfr_id IS NOT NULL", conn)
    finally:
        conn.close()
    if draft.empty:
        return pd.DataFrame()

    # nflverse is the only source carrying GSIS ids for a class this new. It
    # is a network fetch, so losing it costs names for a few picks and the
    # right id for all of them -- not the rookies themselves.
    try:
        import nfl_data_py as nfl
        ids = nfl.import_ids()
    except Exception as e:                           # noqa: BLE001
        print(f"  nflverse id map unavailable ({e}); "
              "falling back to combine names and draft ids")
        ids = pd.DataFrame()

    return _rookie_identities(draft, ids, combine)


def _rookie_board_rows(upcoming_season: int, preseason_df: pd.DataFrame,
                       known_ids: set) -> pd.DataFrame:
    """Board rows for first-year players the season model already projects.

    The board's population comes from the previous season's weekly stats, so a
    player whose first season is `upcoming_season` never reaches it -- there is
    nothing to aggregate. The season model does not have that gap: its
    cold-start rows are built from the draft class itself
    (`preseason_features._cold_start_rows_from_draft`), and PR #96 conditioned
    their availability on draft round specifically to make them better. Those
    projections were being computed and then dropped here.

    Prior-season columns stay NaN rather than zero. A rookie did not score 0
    PPG last season; he has no last season, and the board renders the
    difference (`prev_season_ppg: null` vs `0.0`).
    """
    identities = _load_rookie_identities(upcoming_season)
    if identities.empty:
        return pd.DataFrame()

    projected = preseason_df.set_index("player_id")
    rows = identities[identities["draft_id"].isin(projected.index)].copy()
    # A "rookie" who already has a board row played before this draft class --
    # a UDFA promoted last season, say. The existing row is the better one.
    rows = rows[~rows["player_id"].astype(str).isin({str(i) for i in known_ids})]
    if rows.empty:
        return pd.DataFrame()

    rows["preseason_projection_total"] = rows["draft_id"].map(
        projected["pred_total"])
    if "expected_games" in projected.columns:
        rows["expected_games"] = rows["draft_id"].map(projected["expected_games"])
    for col, out_col in (("confidence_score", "preseason_confidence"),
                         ("support_class", "preseason_support_class")):
        if col in projected.columns:
            rows[out_col] = rows["draft_id"].map(projected[col])

    # Present but empty, matching add_feature_importance()'s shape for a
    # player it has nothing to say about.
    rows["key_features"] = [[] for _ in range(len(rows))]
    rows["feature_importance_rank"] = [{} for _ in range(len(rows))]
    return rows.drop(columns=["draft_id"])


def main():
    from src.utils.nfl_calendar import get_current_nfl_season, is_offseason
    from src.utils.database import DatabaseManager

    current_season = get_current_nfl_season()
    prev_season = current_season
    upcoming_season = current_season + 1 if is_offseason() else current_season

    print(f"Current NFL season: {current_season}")
    print(f"Previous completed season: {prev_season}")
    print(f"Upcoming season: {upcoming_season}")
    print()

    corrected = DatabaseManager().reconcile_player_positions_from_rosters()
    if corrected:
        print(f"Reconciled {corrected} player positions from authoritative roster snapshots.")
        print()

    # 1. Model Performance (previous season OOS predictions vs actuals)
    print("Generating model performance data...")
    generate_model_performance()
    print()

    # 2. Upcoming season projections
    schedule_available = _check_schedule_available(upcoming_season)
    print(f"Schedule available for {upcoming_season}: {schedule_available}")

    # Check for schedule transition (False -> True)
    schedule_transition_check(upcoming_season, schedule_available)

    # Load previous season stats for player list and risk scores
    print(f"Loading {prev_season} season data for player baseline...")
    season_df = load_season_data(prev_season)
    if season_df.empty:
        print(f"  No data for {prev_season} season. Trying {prev_season - 1}...")
        prev_season = prev_season - 1
        season_df = load_season_data(prev_season)

    if season_df.empty:
        print("  No season data available. Cannot generate draft board.")
        return

    print(f"  Loaded {len(season_df)} weekly records")

    agg = aggregate_player_stats(season_df)
    print(f"  Aggregated {len(agg)} unique players")

    agg = compute_risk_scores(agg)
    agg = add_feature_importance(agg)

    # Check for ML predictions for upcoming season
    has_ml_predictions = False
    ml_df = _load_ml_predictions(upcoming_season)
    if ml_df is not None and not ml_df.empty:
        # Merge ML predictions into agg
        for col in ["projection_1w", "projection_4w", "projection_18w"]:
            if col in ml_df.columns:
                pred_map = ml_df.groupby("player_id")[col].last().to_dict()
                agg[col] = agg["player_id"].map(pred_map)
        has_ml_predictions = True
        print(f"  Loaded ML predictions for {upcoming_season} season")
    else:
        print(f"  No ML predictions available for {upcoming_season} season")

    # Preseason season-total model (PreseasonProjector) — preferred over
    # projection_18w when available; see _load_preseason_projections().
    has_preseason_projection = False
    preseason_df = None
    try:
        preseason_df = _load_preseason_projections(upcoming_season, prev_season)
    except Exception as e:
        print(f"  Preseason projector unavailable, falling back: {e}")
    if preseason_df is not None and not preseason_df.empty:
        pred_map = preseason_df.set_index("player_id")["pred_total"].to_dict()
        agg["preseason_projection_total"] = agg["player_id"].map(pred_map)
        if "confidence_score" in preseason_df.columns:
            conf_map = preseason_df.set_index("player_id")["confidence_score"].to_dict()
            agg["preseason_confidence"] = agg["player_id"].map(conf_map)
        if "support_class" in preseason_df.columns:
            support_map = preseason_df.set_index("player_id")["support_class"].to_dict()
            agg["preseason_support_class"] = agg["player_id"].map(support_map)
        if "expected_games" in preseason_df.columns:
            games_map = preseason_df.set_index("player_id")["expected_games"].to_dict()
            agg["expected_games"] = agg["player_id"].map(games_map)
        has_preseason_projection = bool(agg["preseason_projection_total"].notna().any())
        print(f"  Loaded preseason season-total projections for {upcoming_season} season"
              f" ({int(agg['preseason_projection_total'].notna().sum())} players)")

        rookies = _rookie_board_rows(upcoming_season, preseason_df,
                                     set(agg["player_id"]))
        if not rookies.empty:
            agg = pd.concat([agg, rookies], ignore_index=True)
            print(f"  Added {len(rookies)} first-year players from the "
                  f"{upcoming_season} draft class")
        else:
            print(f"  No {upcoming_season} first-year players to add")
    else:
        print(f"  No preseason season-total projections available for {upcoming_season} season")

    output_position_files(agg, upcoming_season, schedule_available,
                          has_ml_predictions, prev_season,
                          has_preseason_projection=has_preseason_projection)
    generate_schedule_impact(upcoming_season, schedule_available)
    generate_model_metadata_frontend(upcoming_season, prev_season,
                                     schedule_available, has_ml_predictions)

    print("\nDone! JSON files ready for the web app.")


if __name__ == "__main__":
    main()
