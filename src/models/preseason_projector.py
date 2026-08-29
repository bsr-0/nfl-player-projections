"""Season-total preseason projector. DEMOTED 2026-08-28 -- NOT the production
model. Superseded by `src.models.season_step8.Step8SeasonModel`.

    DO NOT ADOPT THIS FOR NEW WORK. It is retained as a comparison baseline
    and because scripts/generate_draft_data.py still consumes it pending the
    UI cutover. It is LAST of four arms, at every position.

Corrected 11-fold walk-forward, 2026-08-28 (data/experiments/four_arm_20260828,
4,630 player-seasons scored identically by all four arms):

    mean MAE rank   step8 1.25 | candidate 1.75 | phase7 3.00 | production 4.00

    season MAE      QB      RB      WR      TE
      step8       68.5    51.9    43.7    29.2
      candidate   68.8    51.7    44.2    30.3
      phase7      69.8    52.8    45.2    30.6
      production  76.2    57.5    49.5    33.5     <- this module

    rookies (n=1012)  step8 39.81 | candidate 40.56 | phase7 44.96
                      production 53.29             <- this module, worst by 13.5

That is 4.3-7.7 MAE worse than step8 at every position, and 33% worse than
step8 on rookies. The gap is not attributable to the training-data defects
fixed 2026-08-25/26: this comparison was run AFTER all of them, including the
component-mode fix that had been training production on 4-11% of available
rows. Production was expected to improve and did not.

Kept, not deleted, for three reasons: it is the fourth arm in the standing
comparison; `predict_with_details()` supplies confidence_score/support_class
that scripts/generate_draft_data.py uses for floor/ceiling sizing and which
Step8SeasonModel does not yet reproduce; and deleting a measured baseline
makes future regressions harder to detect, not easier.

See GAPS.md 2026-08-28 and data/experiments/four_arm_20260828/.

A position-specific Ridge model trained on prior-season aggregate stat
rates plus age/experience/support-role context. No calibration layer:
GAPS.md/TRACKING.md (2026-08) measured the upstream calibrator's real
effect as negligible (ΔR² ~0.000-0.002) on the deployed model, and the
legacy veteran-elite/fragile-role cohort patches were already inactive
in the live artifact -- so this was simplified down to the one base
model that actually does the work.

    projector = PreseasonProjector.train(seasons=range(2018, 2026))[0]
    preds = projector.predict(prior_season_df, position="WR")

`predict_with_details()` also returns `confidence_score`/`support_class`
per player -- these feed the asymmetric floor/ceiling sizing in
scripts/generate_draft_data.py and must stay even though they're not
used by the base prediction itself.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

POSITION_PEAK_AGES = {"QB": 29, "RB": 24, "WR": 27, "TE": 28}
POSITION_PRIME_WINDOWS = {
    "QB": (26, 34),
    "RB": (22, 27),
    "WR": (24, 30),
    "TE": (25, 31),
}
POSITION_VETERAN_AGES = {"QB": 35, "RB": 29, "WR": 30, "TE": 31}
POSITION_FALLBACK_AGES = {"QB": 28.0, "RB": 25.0, "WR": 26.0, "TE": 27.0}

MIN_GAMES = 6
MIN_SAMPLES = 20
MODEL_SCHEMA_VERSION = 3

BASE_FEATURES_COMMON = [
    "ppg",
    "games_played",
    "snap_share",
    "age",
    "years_from_peak",
    "is_in_prime",
    "veteran_flag",
    "post_peak_ppg",
    "years_exp",
    "rookie_or_low_experience",
]

BASE_FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": BASE_FEATURES_COMMON
    + [
        "passing_yards_pg",
        "passing_tds_pg",
        "interceptions_pg",
        "rushing_yards_pg",
        "completion_pct",
        "ppg_x_passing_yards_pg",
        "games_played_x_passing_yards_pg",
    ],
    "RB": BASE_FEATURES_COMMON
    + [
        "carries_pg",
        "targets_pg",
        "receptions_pg",
        "rushing_yards_pg",
        "receiving_yards_pg",
        "rush_share",
        "target_share",
        "ppg_x_carries_pg",
        "ppg_x_snap_share",
        "games_played_x_carries_pg",
        "low_volume_efficiency_flag",
    ],
    "WR": BASE_FEATURES_COMMON
    + [
        "targets_pg",
        "receptions_pg",
        "receiving_yards_pg",
        "air_yards_pg",
        "target_share",
        "ppg_x_targets_pg",
        "targets_pg_x_snap_share",
        "low_target_efficiency_flag",
    ],
    "TE": BASE_FEATURES_COMMON
    + [
        "targets_pg",
        "receptions_pg",
        "receiving_yards_pg",
        "target_share",
        "ppg_x_targets_pg",
        "games_played_x_snap_share",
        "low_target_efficiency_flag",
    ],
}

# A feature missing on at least this share of rows gets a companion indicator.
MIN_MISSING_FOR_INDICATOR = 0.01

RIDGE_ALPHA_BY_POSITION = {"QB": 14.0, "RB": 28.0, "WR": 24.0, "TE": 18.0}
SUPPORT_CLASS_ORDER = ("starter", "committee", "backup", "rotational")


class PreseasonProjector:
    """Predict full-season fantasy points from prior-season aggregate signals."""

    def __init__(self):
        self.models: Dict[str, Ridge] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.imputers: Dict[str, dict] = {}
        self.audit_report: Dict[str, Any] = {}
        self.is_fitted = False

    @staticmethod
    def _season_start(season: int) -> datetime:
        return datetime(int(season), 9, 1)

    @classmethod
    def _compute_age(cls, birth_date: object, season: int) -> float:
        if birth_date is None or pd.isna(birth_date) or birth_date == "":
            return np.nan
        try:
            birth_dt = pd.to_datetime(birth_date)
        except Exception:
            return np.nan
        return (cls._season_start(season) - birth_dt.to_pydatetime()).days / 365.25

    @staticmethod
    def _coerce_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(frame.get(column, default), errors="coerce").fillna(default)

    @classmethod
    def _assign_support_class(cls, result: pd.DataFrame) -> pd.Series:
        position = result["position"].fillna("")
        snap = cls._coerce_numeric(result, "snap_share")
        carries = cls._coerce_numeric(result, "carries_pg")
        targets = cls._coerce_numeric(result, "targets_pg")
        passing = cls._coerce_numeric(result, "passing_yards_pg")
        ppg = cls._coerce_numeric(result, "ppg")

        support = pd.Series("rotational", index=result.index, dtype=object)

        qb_starter = (position == "QB") & ((passing >= 220.0) | (ppg >= 16.0))
        qb_backup = (position == "QB") & (passing < 150.0) & (ppg < 12.0)
        rb_starter = (position == "RB") & ((carries >= 15.0) | (snap >= 0.60))
        rb_committee = (position == "RB") & ~rb_starter & (
            (carries >= 8.0) | (targets >= 3.0) | (snap >= 0.38)
        )
        rb_backup = (position == "RB") & (carries < 6.0) & (snap < 0.30) & (targets < 2.5)
        wr_starter = (position == "WR") & ((targets >= 7.0) | (snap >= 0.78))
        wr_committee = (position == "WR") & ~wr_starter & ((targets >= 5.0) | (snap >= 0.60))
        wr_backup = (position == "WR") & (targets < 3.5) & (snap < 0.45)
        te_starter = (position == "TE") & ((targets >= 6.0) | (snap >= 0.72))
        te_committee = (position == "TE") & ~te_starter & ((targets >= 4.0) | (snap >= 0.55))
        te_backup = (position == "TE") & (targets < 2.8) & (snap < 0.45)

        support.loc[qb_starter | rb_starter | wr_starter | te_starter] = "starter"
        support.loc[rb_committee | wr_committee | te_committee] = "committee"
        support.loc[qb_backup | rb_backup | wr_backup | te_backup] = "backup"
        return support

    @classmethod
    def _prepare_feature_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add causal age, interaction, support, and confidence features."""
        result = df.copy()

        if "projection_season" in result.columns:
            projection_season = pd.to_numeric(result["projection_season"], errors="coerce")
        elif "curr_season" in result.columns:
            projection_season = pd.to_numeric(result["curr_season"], errors="coerce")
        elif "prior_season" in result.columns:
            projection_season = pd.to_numeric(result["prior_season"], errors="coerce") + 1
        else:
            projection_season = pd.Series(np.nan, index=result.index)

        if "position" not in result.columns:
            result["position"] = ""

        for col in [
            "ppg",
            "games_played",
            "snap_share",
            "passing_yards_pg",
            "passing_tds_pg",
            "interceptions_pg",
            "rushing_yards_pg",
            "carries_pg",
            "targets_pg",
            "receptions_pg",
            "receiving_yards_pg",
            "air_yards_pg",
            "completion_pct",
            "rush_share",
            "target_share",
            "years_exp",
        ]:
            if col not in result.columns:
                result[col] = 0.0
            result[col] = pd.to_numeric(result[col], errors="coerce")

        if "age" not in result.columns:
            result["age"] = np.nan
        age_missing = result["age"].isna()
        if "birth_date" in result.columns:
            computed_age = [
                cls._compute_age(birth_date, season) if np.isfinite(season) else np.nan
                for birth_date, season in zip(result["birth_date"], projection_season)
            ]
            result.loc[age_missing, "age"] = computed_age
            age_missing = result["age"].isna()
        if age_missing.any() and "years_exp" in result.columns:
            result.loc[age_missing, "age"] = 22.0 + pd.to_numeric(
                result.loc[age_missing, "years_exp"], errors="coerce"
            )
            age_missing = result["age"].isna()
        if age_missing.any():
            result.loc[age_missing, "age"] = result.loc[age_missing, "position"].map(
                POSITION_FALLBACK_AGES
            ).fillna(26.0)

        result["age"] = pd.to_numeric(result["age"], errors="coerce").fillna(26.0)
        result["ppg"] = cls._coerce_numeric(result, "ppg")
        result["games_played"] = cls._coerce_numeric(result, "games_played")
        result["snap_share"] = cls._coerce_numeric(result, "snap_share").clip(0.0, 1.0)
        result["years_exp"] = cls._coerce_numeric(result, "years_exp").clip(lower=0.0)

        peak_ages = result["position"].map(POSITION_PEAK_AGES).fillna(27.0)
        result["years_from_peak"] = result["age"] - peak_ages
        prime_start = result["position"].map(
            {pos: bounds[0] for pos, bounds in POSITION_PRIME_WINDOWS.items()}
        ).fillna(24.0)
        prime_end = result["position"].map(
            {pos: bounds[1] for pos, bounds in POSITION_PRIME_WINDOWS.items()}
        ).fillna(30.0)
        result["is_in_prime"] = (
            (result["age"] >= prime_start) & (result["age"] <= prime_end)
        ).astype(float)
        veteran_thresholds = result["position"].map(POSITION_VETERAN_AGES).fillna(30.0)
        result["veteran_flag"] = (result["age"] >= veteran_thresholds).astype(float)
        result["post_peak_ppg"] = result["ppg"] * result["years_from_peak"].clip(lower=0.0)
        result["rookie_or_low_experience"] = (result["years_exp"] <= 1.0).astype(float)

        result["ppg_x_carries_pg"] = result["ppg"] * cls._coerce_numeric(result, "carries_pg")
        result["ppg_x_snap_share"] = result["ppg"] * result["snap_share"]
        result["games_played_x_carries_pg"] = result["games_played"] * cls._coerce_numeric(
            result, "carries_pg"
        )
        result["ppg_x_targets_pg"] = result["ppg"] * cls._coerce_numeric(result, "targets_pg")
        result["targets_pg_x_snap_share"] = cls._coerce_numeric(result, "targets_pg") * result[
            "snap_share"
        ]
        result["games_played_x_snap_share"] = result["games_played"] * result["snap_share"]
        result["ppg_x_passing_yards_pg"] = result["ppg"] * cls._coerce_numeric(
            result, "passing_yards_pg"
        )
        result["games_played_x_passing_yards_pg"] = result["games_played"] * cls._coerce_numeric(
            result, "passing_yards_pg"
        )

        result["low_volume_efficiency_flag"] = (
            (result["position"] == "RB")
            & (result["ppg"] >= 12.0)
            & (
                (cls._coerce_numeric(result, "carries_pg") < 10.0)
                | (result["snap_share"] < 0.40)
            )
        ).astype(float)
        result["low_target_efficiency_flag"] = (
            result["position"].isin(["WR", "TE"])
            & (result["ppg"] >= 11.0)
            & (cls._coerce_numeric(result, "targets_pg") < 6.0)
        ).astype(float)

        result["support_class"] = cls._assign_support_class(result)
        for support_class in SUPPORT_CLASS_ORDER:
            result[f"support_class_{support_class}"] = (
                result["support_class"] == support_class
            ).astype(float)

        # confidence_score feeds scripts/generate_draft_data.py's asymmetric
        # floor/ceiling sizing (FLOOR_CEILING_COEF) -- kept even though the
        # base model itself doesn't use it as a feature.
        workload_norm = np.where(
            result["position"].eq("QB"),
            cls._coerce_numeric(result, "passing_yards_pg").clip(0.0, 300.0) / 300.0,
            np.where(
                result["position"].eq("RB"),
                cls._coerce_numeric(result, "carries_pg").clip(0.0, 20.0) / 20.0,
                cls._coerce_numeric(result, "targets_pg").clip(0.0, 10.0) / 10.0,
            ),
        )
        experience_norm = cls._coerce_numeric(result, "years_exp").clip(0.0, 5.0) / 5.0
        support_bonus = (
            0.20 * result["support_class_starter"]
            + 0.10 * result["support_class_committee"]
            + 0.02 * result["support_class_rotational"]
        )
        result["confidence_score"] = np.clip(
            0.30 * (result["games_played"].clip(0.0, 17.0) / 17.0)
            + 0.25 * result["snap_share"]
            + 0.20 * workload_norm
            + 0.15 * experience_norm
            + support_bonus,
            0.05,
            1.0,
        )
        return result

    @staticmethod
    def _fit_linear_model(
        pos_df: pd.DataFrame,
        features: List[str],
        alpha: float,
    ) -> Tuple[Ridge, StandardScaler]:
        raw = pos_df[features].replace([np.inf, -np.inf], np.nan)
        miss_rate = raw.isna().mean()
        indicators = sorted(miss_rate[miss_rate >= MIN_MISSING_FOR_INDICATOR].index)
        medians = raw.median().fillna(0.0)
        imputer = {"medians": medians, "indicators": indicators}

        X = PreseasonProjector._apply_imputer(raw, imputer).to_numpy(dtype=float)
        y = pos_df["season_total"].to_numpy(dtype=float)
        # Rows are dropped ONLY for a missing TARGET. The previous version also
        # required every FEATURE to be finite, which after adding rookies would
        # have silently discarded exactly the population being added -- the
        # model would then have been scored on an easier, veteran-only set and
        # looked better for it.
        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        return model, scaler, imputer

    @staticmethod
    def _apply_imputer(raw: pd.DataFrame, imputer: dict) -> pd.DataFrame:
        """Median-impute plus 0/1 indicators, using TRAIN-fitted values.

        The indicator is what lets Ridge separate "no prior season" from "an
        average prior season"; imputing alone would tell the model a rookie was
        a median veteran.
        """
        out = raw.copy()
        for col in imputer["indicators"]:
            out[f"{col}__isna"] = out[col].isna().astype(float)
        out = out.fillna(imputer["medians"])
        return out.fillna(0.0)

    @classmethod
    def _predict_base(
        cls,
        frame: pd.DataFrame,
        features: List[str],
        scaler: StandardScaler,
        model: Ridge,
        imputer: Optional[dict] = None,
    ) -> np.ndarray:
        raw = frame.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
        if imputer is None:
            # Legacy callers / models persisted before FEATURE_VERSION 35 have
            # no imputer. Fall back to the old zero-fill rather than raising,
            # but do NOT pretend it is equivalent: zero-filling a rookie's
            # prior season asserts he scored zero.
            X = raw.fillna(0.0).to_numpy(dtype=float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X = PreseasonProjector._apply_imputer(raw, imputer).to_numpy(dtype=float)
        pred = model.predict(scaler.transform(X))
        return np.maximum(pred, 0.0)

    @classmethod
    def _top150_actual_mean(cls, df: pd.DataFrame, pred_col: str) -> Optional[float]:
        if df.empty or "curr_season" not in df.columns:
            return None
        values: List[float] = []
        for season in sorted(df["curr_season"].dropna().unique()):
            top = df[df["curr_season"] == season].sort_values(pred_col, ascending=False).head(150)
            if top.empty:
                continue
            values.append(float(pd.to_numeric(top["season_total"], errors="coerce").mean()))
        if not values:
            return None
        return float(np.mean(values))

    @classmethod
    def _top_actual_mean_by_position(
        cls,
        df: pd.DataFrame,
        pred_col: str,
        position: str,
        limit: int,
    ) -> Optional[float]:
        if df.empty or "curr_season" not in df.columns:
            return None
        values: List[float] = []
        for season in sorted(df["curr_season"].dropna().unique()):
            top = (
                df[(df["curr_season"] == season) & (df["position"] == position)]
                .sort_values(pred_col, ascending=False)
                .head(limit)
            )
            if top.empty:
                continue
            values.append(float(pd.to_numeric(top["season_total"], errors="coerce").mean()))
        if not values:
            return None
        return float(np.mean(values))

    @classmethod
    def _summarize_predictions(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Honest holdout diagnostics (MAE/bias/top-N actual means) for the
        audit report. Reporting only -- not used to select or gate anything."""
        report: Dict[str, Any] = {"overall": {}, "by_position": {}}
        if df.empty:
            return report

        actual = pd.to_numeric(df["season_total"], errors="coerce")
        pred = pd.to_numeric(df["pred"], errors="coerce")
        report["overall"] = {
            "mae": round(float(np.mean(np.abs(pred - actual))), 4),
            "bias": round(float(np.mean(pred - actual)), 4),
        }

        top150 = cls._top150_actual_mean(df, "pred")
        if top150 is not None:
            report["overall"]["top150_actual_mean"] = round(top150, 4)
        rb24 = cls._top_actual_mean_by_position(df, "pred", "RB", 24)
        wr24 = cls._top_actual_mean_by_position(df, "pred", "WR", 24)
        te12 = cls._top_actual_mean_by_position(df, "pred", "TE", 12)
        if rb24 is not None:
            report["overall"]["top24_rb_actual_mean"] = round(rb24, 4)
        if wr24 is not None:
            report["overall"]["top24_wr_actual_mean"] = round(wr24, 4)
        if te12 is not None:
            report["overall"]["top12_te_actual_mean"] = round(te12, 4)

        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = df[df["position"] == pos]
            if pos_df.empty:
                continue
            pos_actual = pd.to_numeric(pos_df["season_total"], errors="coerce")
            pos_pred = pd.to_numeric(pos_df["pred"], errors="coerce")
            report["by_position"][pos] = {
                "n": int(len(pos_df)),
                "mae": round(float(np.mean(np.abs(pos_pred - pos_actual))), 4),
                "bias": round(float(np.mean(pos_pred - pos_actual)), 4),
            }
        return report

    @classmethod
    def _holdout_audit(cls, prepared_pairs: pd.DataFrame) -> Dict[str, Any]:
        """Fit on all-but-last season, evaluate on the last season, purely
        for a real reported MAE/bias -- not used to pick anything."""
        if "curr_season" not in prepared_pairs.columns or prepared_pairs["curr_season"].nunique() < 2:
            return {}
        holdout_season = prepared_pairs["curr_season"].max()
        rows: List[pd.DataFrame] = []
        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = prepared_pairs[prepared_pairs["position"] == pos]
            train_df = pos_df[pos_df["curr_season"] != holdout_season]
            test_df = pos_df[pos_df["curr_season"] == holdout_season]
            features = [f for f in BASE_FEATURES_BY_POSITION[pos] if f in train_df.columns]
            if len(train_df) < MIN_SAMPLES or not features or test_df.empty:
                continue
            model, scaler, imputer = cls._fit_linear_model(
                train_df, features, RIDGE_ALPHA_BY_POSITION[pos])
            scored = test_df.copy()
            # Imputer fitted on train_df only, applied to the holdout -- fitting
            # it across both would leak the holdout's distribution.
            scored["pred"] = cls._predict_base(scored, features, scaler, model, imputer)
            rows.append(scored)
        if not rows:
            return {}
        predictions = pd.concat(rows, ignore_index=True)
        summary = cls._summarize_predictions(predictions)
        summary["holdout_season"] = int(holdout_season)
        return summary

    @staticmethod
    def _safe_round(value: Any, digits: int = 4) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        try:
            if not np.isfinite(value):
                return None
            return round(float(value), digits)
        except Exception:
            return value

    @classmethod
    def _sanitize_jsonable(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): cls._sanitize_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._sanitize_jsonable(v) for v in obj]
        return cls._safe_round(obj)

    @staticmethod
    def _deep_get(payload: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    # ------------------------------------------------------------------
    # Training data assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_season_pairs(db, seasons: List[int]) -> pd.DataFrame:
        """Build (prior season features, current season target) pairs."""
        from config.settings import regular_season_week_sql

        frames = []
        season_list = sorted(seasons)
        for i in range(len(season_list) - 1):
            prior = season_list[i]
            curr = season_list[i + 1]

            with db._get_connection() as conn:
                prior_df = pd.read_sql_query(
                    f"""
                    SELECT
                        pws.player_id,
                        p.name AS player_name,
                        p.position,
                        COUNT(*) AS games_played,
                        AVG(pws.fantasy_points) AS ppg,
                        AVG(COALESCE(pws.passing_yards, 0)) AS passing_yards_pg,
                        AVG(COALESCE(pws.passing_tds, 0)) AS passing_tds_pg,
                        AVG(COALESCE(pws.interceptions, 0)) AS interceptions_pg,
                        AVG(COALESCE(pws.rushing_yards, 0)) AS rushing_yards_pg,
                        AVG(COALESCE(pws.rushing_attempts, 0)) AS carries_pg,
                        AVG(COALESCE(pws.targets, 0)) AS targets_pg,
                        AVG(COALESCE(pws.receptions, 0)) AS receptions_pg,
                        AVG(COALESCE(pws.receiving_yards, 0)) AS receiving_yards_pg,
                        AVG(COALESCE(pws.air_yards, 0)) AS air_yards_pg,
                        AVG(
                            CASE
                                WHEN COALESCE(pws.passing_attempts, 0) > 0
                                THEN 100.0 * pws.passing_completions / pws.passing_attempts
                                ELSE 0
                            END
                        ) AS completion_pct,
                        AVG(COALESCE(us.snap_share, 0)) AS snap_share,
                        AVG(COALESCE(us.target_share, 0)) AS target_share,
                        AVG(COALESCE(us.rush_share, 0)) AS rush_share,
                        p.birth_date AS birth_date,
                        COALESCE(r.years_exp, 0) AS years_exp
                    FROM player_weekly_stats pws
                    JOIN players p ON pws.player_id = p.player_id
                    LEFT JOIN utilization_scores us
                      ON pws.player_id = us.player_id
                     AND pws.season = us.season
                     AND pws.week = us.week
                    LEFT JOIN rosters r
                      ON pws.player_id = r.player_id
                     AND r.season = ?
                    WHERE pws.season = ?
                      AND p.position IN ('QB', 'RB', 'WR', 'TE')
                      -- Era-aware: 17 through 2020, 18 from 2021. A flat 18
                      -- folded the wild-card round into these prior-season
                      -- per-game rates and into the COUNT(*) >= MIN_GAMES
                      -- eligibility test below.
                      AND {regular_season_week_sql('pws.week', 'pws.season')}
                    GROUP BY pws.player_id, p.name, p.position, p.birth_date, r.years_exp
                    HAVING COUNT(*) >= ?
                    """,
                    conn,
                    params=(prior, prior, MIN_GAMES),
                )
                curr_df = pd.read_sql_query(
                    """
                    SELECT pws.player_id, p.name AS curr_name,
                           p.position AS curr_position, p.birth_date AS curr_birth_date,
                           SUM(pws.fantasy_points) AS season_total
                    FROM player_weekly_stats pws
                    JOIN players p ON pws.player_id = p.player_id
                    WHERE pws.season = ?
                    GROUP BY pws.player_id, p.name, p.position, p.birth_date
                    HAVING COUNT(*) >= 4
                    """,
                    conn,
                    params=(curr,),
                )

            if prior_df.empty or curr_df.empty:
                continue

            # LEFT from the CURRENT season, not an inner join. The inner
            # version required a prior-season row, so a player whose first NFL
            # season is `curr` never formed a pair at all -- measured, 0 of the
            # true 2025 rookies reached this frame. Their prior-season columns
            # arrive NaN, which _fit_linear_model now imputes explicitly rather
            # than dropping.
            merged = curr_df.merge(prior_df, on="player_id", how="left")
            # Identity comes from prior_df for veterans and is NaN for rookies,
            # who have no prior row. Without this backfill their `position` is
            # NaN and fit()'s per-position loop drops every one of them -- the
            # rows would exist and still never be trained on.
            for col, src in (("player_name", "curr_name"),
                             ("position", "curr_position"),
                             ("birth_date", "curr_birth_date")):
                if col in merged.columns:
                    merged[col] = merged[col].fillna(merged[src])
                else:
                    merged[col] = merged[src]
            merged = merged.drop(columns=["curr_name", "curr_position", "curr_birth_date"])
            merged["prior_season"] = prior
            merged["curr_season"] = curr
            merged["projection_season"] = curr

            frames.append(merged)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, pairs_df: pd.DataFrame) -> "PreseasonProjector":
        prepared_pairs = self._prepare_feature_frame(pairs_df)

        audit = self._holdout_audit(prepared_pairs)

        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = prepared_pairs[prepared_pairs["position"] == pos]
            features = [f for f in BASE_FEATURES_BY_POSITION[pos] if f in pos_df.columns]
            if len(pos_df) < MIN_SAMPLES or not features:
                continue
            model, scaler, imputer = self._fit_linear_model(
                pos_df, features, RIDGE_ALPHA_BY_POSITION[pos])
            self.models[pos] = model
            self.scalers[pos] = scaler
            self.feature_names[pos] = features
            self.imputers[pos] = imputer

        self.audit_report = self._sanitize_jsonable(audit)
        self.is_fitted = len(self.models) > 0
        return self

    def _prepare_for_position(self, prior_season_df: pd.DataFrame, position: str) -> pd.DataFrame:
        frame = prior_season_df.copy()
        if "position" not in frame.columns:
            frame["position"] = position
        return self._prepare_feature_frame(frame)

    def predict_with_details(self, prior_season_df: pd.DataFrame, position: str) -> pd.DataFrame:
        if position not in self.models:
            raise ValueError(f"PreseasonProjector not fitted for {position}")
        prepared = self._prepare_for_position(prior_season_df, position)
        pred = self._predict_base(
            prepared,
            self.feature_names[position],
            self.scalers[position],
            self.models[position],
            self.imputers.get(position),
        )
        return pd.DataFrame(
            {
                "pred": pred,
                "confidence_score": prepared.get(
                    "confidence_score", pd.Series(np.nan, index=prepared.index)
                ),
                "support_class": prepared.get(
                    "support_class", pd.Series("", index=prepared.index, dtype=object)
                ),
            },
            index=prepared.index,
        )

    def predict(self, prior_season_df: pd.DataFrame, position: str) -> np.ndarray:
        details = self.predict_with_details(prior_season_df, position)
        return details["pred"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        data: Dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "audit_report": self.audit_report,
            "positions": {},
        }
        for pos, model in self.models.items():
            scaler = self.scalers[pos]
            data["positions"][pos] = {
                "features": self.feature_names[pos],
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "alpha": float(getattr(model, "alpha", 0.0)),
            }

        Path(path).write_text(json.dumps(self._sanitize_jsonable(data), indent=2))
        logger.info("PreseasonProjector saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PreseasonProjector":
        data = json.loads(Path(path).read_text())
        proj = cls()
        proj.audit_report = data.get("audit_report", {})

        # schema_version 3: flat per-position payload. Older artifacts
        # (pre-2026-08-07, back when this had a calibration layer) nested
        # the base model under positions[pos]["base_outcome_model"] --
        # accept either shape, ignore any calibration keys either way.
        for pos, payload in data.get("positions", {}).items():
            base_payload = payload.get("base_outcome_model", payload)
            model = Ridge(alpha=float(base_payload.get("alpha", 0.0)))
            model.coef_ = np.asarray(base_payload["coef"], dtype=float)
            model.intercept_ = float(base_payload["intercept"])
            model.n_features_in_ = len(model.coef_)

            scaler = StandardScaler()
            scaler.mean_ = np.asarray(base_payload["scaler_mean"], dtype=float)
            scaler.scale_ = np.asarray(base_payload["scaler_scale"], dtype=float)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)

            proj.models[pos] = model
            proj.scalers[pos] = scaler
            proj.feature_names[pos] = base_payload["features"]

        proj.is_fitted = len(proj.models) > 0
        return proj

    def get_bias_report(self) -> Dict[str, Dict[str, Any]]:
        return self._deep_get(self.audit_report, ("by_position",), {})

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, seasons: List[int], db=None) -> Tuple["PreseasonProjector", pd.DataFrame]:
        from src.utils.database import DatabaseManager

        db = db or DatabaseManager()
        pairs_df = cls._build_season_pairs(db, seasons)
        if pairs_df.empty:
            raise ValueError("No season pairs found — check database has 2+ seasons of data")
        proj = cls()
        proj.fit(pairs_df)
        return proj, pairs_df
