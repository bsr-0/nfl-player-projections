"""Season-total preseason projector.

The projector is an upstream two-stage model:
1. A position-specific base outcome model trained only on prior-season
   football features.
2. A bounded upstream calibration layer that uses confidence and role
   features (no market/ADP data — pure ML).

The outward contract stays the same for callers:

    projector = PreseasonProjector.train(seasons=range(2018, 2026))[0]
    preds = projector.predict(prior_season_df, position="WR")
"""
from __future__ import annotations

import copy
import json
import importlib.util
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass
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
MIN_CALIBRATION_SAMPLES = 18
LARGE_DIVERGENCE_MIN_POINTS = {"QB": 16.0, "RB": 14.0, "WR": 12.0, "TE": 10.0}
MODEL_SCHEMA_VERSION = 2

# Historical tolerance gates for model selection.
OUTCOME_MAE_TOLERANCE = 3.0
OUTCOME_BIAS_TOLERANCE = 2.0
TOP150_ACTUAL_MEAN_TOLERANCE = 1.0
RB_TOP24_ACTUAL_MEAN_TOLERANCE = 4.0
DRAFT_SIM_REGRESSION_PATH = Path(__file__).resolve().parents[2] / "data" / "draft_sim_results" / "mode_sweep_preseason_compare_2024_2025.json"
LIVE_PROJECTOR_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "preseason_projector.json"
DRAFT_SIM_MEAN_RANK_TOLERANCE = 0.15
DRAFT_SIM_LIFT_TOLERANCE = 5.0
DRAFT_SIM_MODEL_POINTS_TOLERANCE = 10.0


@dataclass
class VeteranEliteCalibration:
    """Legacy calibration retained for artifact compatibility."""

    position: str
    factor: float
    age_threshold: float
    elite_ppg_threshold: float
    sample_size: int
    mean_error_before: float
    mean_error_after: float
    mae_before: float
    mae_after: float
    median_actual_to_pred_ratio: float

    def applies_to(self, frame: pd.DataFrame) -> pd.Series:
        age = pd.to_numeric(frame.get("age"), errors="coerce").fillna(0.0)
        ppg = pd.to_numeric(frame.get("ppg"), errors="coerce").fillna(0.0)
        return (age >= self.age_threshold) & (ppg >= self.elite_ppg_threshold)


@dataclass
class FragileRoleCalibration:
    """Legacy calibration retained for artifact compatibility."""

    position: str
    factor: float
    ppg_threshold: float
    low_feature: str
    low_feature_threshold: float
    sample_size: int
    mean_error_before: float
    mean_error_after: float
    mae_before: float
    mae_after: float

    def applies_to(self, frame: pd.DataFrame) -> pd.Series:
        ppg = pd.to_numeric(frame.get("ppg"), errors="coerce").fillna(0.0)
        feature = pd.to_numeric(frame.get(self.low_feature), errors="coerce").fillna(0.0)
        return (ppg >= self.ppg_threshold) & (feature <= self.low_feature_threshold)


LEGACY_FEATURES_COMMON = [
    "ppg",
    "games_played",
    "snap_share",
    "age",
    "years_from_peak",
    "is_in_prime",
    "veteran_flag",
    "post_peak_ppg",
]

LEGACY_FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": LEGACY_FEATURES_COMMON
    + [
        "passing_yards_pg",
        "passing_tds_pg",
        "interceptions_pg",
        "rushing_yards_pg",
        "completion_pct",
    ],
    "RB": LEGACY_FEATURES_COMMON
    + [
        "carries_pg",
        "targets_pg",
        "receptions_pg",
        "rushing_yards_pg",
        "receiving_yards_pg",
        "rush_share",
        "target_share",
    ],
    "WR": LEGACY_FEATURES_COMMON
    + [
        "targets_pg",
        "receptions_pg",
        "receiving_yards_pg",
        "air_yards_pg",
        "target_share",
    ],
    "TE": LEGACY_FEATURES_COMMON
    + [
        "targets_pg",
        "receptions_pg",
        "receiving_yards_pg",
        "target_share",
    ],
}

BASE_FEATURES_COMMON = LEGACY_FEATURES_COMMON + ["years_exp", "rookie_or_low_experience"]

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

RB_CONSTRUCTION_FEATURES = BASE_FEATURES_BY_POSITION["RB"] + [
    "starter_workhorse_flag",
    "veteran_starter_flag",
    "support_class_starter_x_carries_pg",
    "ppg_x_veteran_flag",
    "carries_pg_x_target_share",
    "games_played_x_starter_flag",
]

RB_CONSTRUCTION_FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": BASE_FEATURES_BY_POSITION["QB"],
    "RB": RB_CONSTRUCTION_FEATURES,
    "WR": BASE_FEATURES_BY_POSITION["WR"],
    "TE": BASE_FEATURES_BY_POSITION["TE"],
}

HYBRID_LEGACY_RB_FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": BASE_FEATURES_BY_POSITION["QB"],
    "RB": LEGACY_FEATURES_BY_POSITION["RB"],
    "WR": BASE_FEATURES_BY_POSITION["WR"],
    "TE": BASE_FEATURES_BY_POSITION["TE"],
}

CALIBRATION_FEATURES_COMMON = [
    "raw_pred",
    "confidence_score",
    "low_information_score",
    "rookie_or_low_experience",
    "support_class_starter",
    "support_class_committee",
    "support_class_backup",
    "support_class_rotational",
    "raw_pred_x_confidence",
]

CALIBRATION_FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": CALIBRATION_FEATURES_COMMON
    + [
        "games_played",
        "snap_share",
        "ppg",
        "passing_yards_pg",
    ],
    "RB": CALIBRATION_FEATURES_COMMON
    + [
        "games_played",
        "snap_share",
        "carries_pg",
        "targets_pg",
        "ppg_x_carries_pg",
        "low_volume_efficiency_flag",
    ],
    "WR": CALIBRATION_FEATURES_COMMON
    + [
        "games_played",
        "snap_share",
        "targets_pg",
        "target_share",
        "ppg_x_targets_pg",
        "low_target_efficiency_flag",
    ],
    "TE": CALIBRATION_FEATURES_COMMON
    + [
        "games_played",
        "snap_share",
        "targets_pg",
        "target_share",
        "ppg_x_targets_pg",
    ],
}

RIDGE_ALPHA_BY_POSITION = {"QB": 14.0, "RB": 28.0, "WR": 24.0, "TE": 18.0}
CALIBRATOR_ALPHA_BY_POSITION = {"QB": 24.0, "RB": 36.0, "WR": 34.0, "TE": 28.0}
CALIBRATOR_MAX_ADJUSTMENT_SHARE = {"QB": 0.20, "RB": 0.30, "WR": 0.28, "TE": 0.24}
SUPPORT_CLASS_ORDER = ("starter", "committee", "backup", "rotational")


@dataclass
class VariantSpec:
    name: str
    feature_map: Dict[str, List[str]]
    alpha_by_position: Dict[str, float]
    use_calibrator: bool


@dataclass
class UpstreamCalibrator:
    position: str
    features: List[str]
    coef: List[float]
    intercept: float
    scaler_mean: List[float]
    scaler_scale: List[float]
    max_adjustment_share: float
    sample_size: int
    train_mae_before: float
    train_mae_after: float
    confidence_power: float = 1.0
    min_confidence_multiplier: float = 0.20
    starter_adjustment_multiplier: float = 1.0

    def _make_scaler(self) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(self.scaler_mean, dtype=float)
        scaler.scale_ = np.asarray(self.scaler_scale, dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(self.features)
        return scaler

    def _make_model(self) -> Ridge:
        model = Ridge()
        model.coef_ = np.asarray(self.coef, dtype=float)
        model.intercept_ = float(self.intercept)
        model.n_features_in_ = len(self.features)
        return model

    def calibrate(self, prepared: pd.DataFrame, raw_pred: np.ndarray) -> np.ndarray:
        feature_frame = PreseasonProjector._build_calibration_feature_frame(
            self.position, prepared, raw_pred
        )
        X = feature_frame.reindex(columns=self.features, fill_value=0.0).fillna(0.0).to_numpy(
            dtype=float
        )
        if X.size == 0:
            return np.maximum(raw_pred, 0.0)

        scaler = self._make_scaler()
        model = self._make_model()
        candidate = model.predict(scaler.transform(X))
        confidence = pd.to_numeric(prepared.get("confidence_score"), errors="coerce").fillna(0.5)
        confidence = confidence.clip(0.05, 1.0).to_numpy(dtype=float)
        starter_mask = pd.to_numeric(prepared.get("support_class_starter"), errors="coerce").fillna(0.0)
        starter_multiplier = np.where(
            starter_mask.to_numpy(dtype=float) > 0.5,
            self.starter_adjustment_multiplier,
            1.0,
        )
        floor = LARGE_DIVERGENCE_MIN_POINTS.get(self.position, 10.0)
        scale = np.maximum(np.maximum(raw_pred, 0.0), floor)
        confidence_decay = np.power(1.0 - confidence, self.confidence_power)
        dynamic_share = self.max_adjustment_share * (
            self.min_confidence_multiplier
            + (1.0 - self.min_confidence_multiplier) * confidence_decay
        )
        dynamic_share = dynamic_share * starter_multiplier
        bounded_delta = np.clip(candidate - raw_pred, -dynamic_share * scale, dynamic_share * scale)
        pred = raw_pred + bounded_delta

        return np.maximum(pred, 0.0)

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "UpstreamCalibrator":
        return cls(**payload)


class PreseasonProjector:
    """Predict full-season fantasy points from prior-season aggregate signals."""

    def __init__(self):
        self.models: Dict[str, Ridge] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.upstream_calibrators: Dict[str, UpstreamCalibrator] = {}
        self.selection_report: Dict[str, Any] = {}
        self.audit_report: Dict[str, Any] = {}
        self.variant_name: Optional[str] = None
        self.legacy_veteran_elite_calibration: Dict[str, VeteranEliteCalibration] = {}
        self.legacy_fragile_role_calibration: Dict[str, List[FragileRoleCalibration]] = {}
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
        result["starter_workhorse_flag"] = (
            (result["position"] == "RB")
            & (result["support_class"] == "starter")
            & (cls._coerce_numeric(result, "carries_pg") >= 14.0)
        ).astype(float)
        result["veteran_starter_flag"] = (
            (result["position"] == "RB")
            & (result["support_class"] == "starter")
            & (result["veteran_flag"] > 0.5)
        ).astype(float)
        result["support_class_starter_x_carries_pg"] = result["support_class_starter"] * cls._coerce_numeric(
            result, "carries_pg"
        )
        result["ppg_x_veteran_flag"] = result["ppg"] * result["veteran_flag"]
        result["carries_pg_x_target_share"] = cls._coerce_numeric(result, "carries_pg") * cls._coerce_numeric(
            result, "target_share"
        )
        result["games_played_x_starter_flag"] = result["games_played"] * result["support_class_starter"]

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
        result["low_information_score"] = 1.0 - result["confidence_score"]
        return result

    @staticmethod
    def _fit_linear_model(
        pos_df: pd.DataFrame,
        features: List[str],
        alpha: float,
    ) -> Tuple[Ridge, StandardScaler]:
        X = pos_df[features].fillna(0.0).to_numpy(dtype=float)
        y = pos_df["season_total"].to_numpy(dtype=float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[valid], y[valid]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        return model, scaler

    @classmethod
    def _build_calibration_feature_frame(
        cls,
        position: str,
        prepared: pd.DataFrame,
        raw_pred: np.ndarray,
    ) -> pd.DataFrame:
        frame = prepared.copy()
        frame["raw_pred"] = np.asarray(raw_pred, dtype=float)
        frame["raw_pred_x_confidence"] = frame["raw_pred"] * cls._coerce_numeric(
            frame, "confidence_score", 0.5
        )
        return frame

    @classmethod
    def _fit_upstream_calibrator(
        cls,
        position: str,
        calibration_df: pd.DataFrame,
    ) -> Tuple[Optional[UpstreamCalibrator], Dict[str, Any]]:
        audit: Dict[str, Any] = {}
        if calibration_df.empty or len(calibration_df) < MIN_CALIBRATION_SAMPLES:
            return None, audit

        calibration_features = [
            feature for feature in CALIBRATION_FEATURES_BY_POSITION[position] if feature in calibration_df.columns
        ]
        if not calibration_features:
            return None, audit

        X = calibration_df[calibration_features].fillna(0.0).to_numpy(dtype=float)
        y = pd.to_numeric(calibration_df["season_total"], errors="coerce").to_numpy(dtype=float)
        raw = pd.to_numeric(calibration_df["raw_pred"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(raw)
        if valid.sum() < MIN_CALIBRATION_SAMPLES:
            return None, audit

        X = X[valid]
        y = y[valid]
        raw = raw[valid]
        fit_frame = calibration_df.loc[valid].copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=CALIBRATOR_ALPHA_BY_POSITION[position])
        model.fit(X_scaled, y)

        calibrator = UpstreamCalibrator(
            position=position,
            features=calibration_features,
            coef=model.coef_.tolist(),
            intercept=float(model.intercept_),
            scaler_mean=scaler.mean_.tolist(),
            scaler_scale=scaler.scale_.tolist(),
            max_adjustment_share=CALIBRATOR_MAX_ADJUSTMENT_SHARE[position],
            sample_size=int(len(y)),
            train_mae_before=float(np.mean(np.abs(raw - y))),
            train_mae_after=0.0,
        )
        bounded_pred = calibrator.calibrate(fit_frame, raw)
        mae_before = float(np.mean(np.abs(raw - y)))
        mae_after = float(np.mean(np.abs(bounded_pred - y)))
        bias_before = float(np.mean(raw - y))
        bias_after = float(np.mean(bounded_pred - y))
        calibrator.train_mae_after = mae_after
        audit = {
            "sample_size": int(len(y)),
            "features": calibration_features,
            "mae_before": round(mae_before, 4),
            "mae_after": round(mae_after, 4),
            "bias_before": round(bias_before, 4),
            "bias_after": round(bias_after, 4),
            "max_adjustment_share": round(calibrator.max_adjustment_share, 4),
        }
        if mae_after >= mae_before - 1e-9 and abs(bias_after) >= abs(bias_before) - 1e-9:
            return None, audit
        return calibrator, audit

    @staticmethod
    def _variant_specs() -> List[VariantSpec]:
        return [
            VariantSpec(
                name="ridge_baseline",
                feature_map=LEGACY_FEATURES_BY_POSITION,
                alpha_by_position={pos: 10.0 for pos in ("QB", "RB", "WR", "TE")},
                use_calibrator=False,
            ),
            VariantSpec(
                name="position_specific_ridge",
                feature_map=BASE_FEATURES_BY_POSITION,
                alpha_by_position=RIDGE_ALPHA_BY_POSITION,
                use_calibrator=False,
            ),
            VariantSpec(
                name="position_specific_ridge_rb_construction",
                feature_map=RB_CONSTRUCTION_FEATURES_BY_POSITION,
                alpha_by_position=RIDGE_ALPHA_BY_POSITION,
                use_calibrator=False,
            ),
            VariantSpec(
                name="hybrid_legacy_rb_position_specific",
                feature_map=HYBRID_LEGACY_RB_FEATURES_BY_POSITION,
                alpha_by_position=RIDGE_ALPHA_BY_POSITION,
                use_calibrator=False,
            ),
            VariantSpec(
                name="position_specific_ridge_plus_calibrator",
                feature_map=BASE_FEATURES_BY_POSITION,
                alpha_by_position=RIDGE_ALPHA_BY_POSITION,
                use_calibrator=True,
            ),
        ]

    @staticmethod
    def _load_draft_sim_module():
        module_name = "draft_sim_mode_sweep_runtime"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing
        module_path = Path(__file__).resolve().parents[2] / "scripts" / "draft_sim_mode_sweep.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load draft sim module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _load_draft_sim_baseline(cls, path: Path = DRAFT_SIM_REGRESSION_PATH) -> Dict[int, Dict[str, Any]]:
        if not path.exists():
            return {}
        rows = json.loads(path.read_text())
        baseline: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            if row.get("ranking") == "preseason_ml":
                baseline[int(row["season"])] = row
        return baseline

    @classmethod
    def _summarize_draft_sim_rows(cls, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        aggregate = {
            "mean_rank_avg": round(float(np.mean([row["mean_rank"] for row in rows])), 4),
            "lift_vs_adp_mean_avg": round(float(np.mean([row["lift_vs_adp_mean"] for row in rows])), 4),
            "mean_model_actual_avg": round(float(np.mean([row["mean_model_actual"] for row in rows])), 4),
            "wins_vs_adp_mean_slots_total": int(sum(row["wins_vs_adp_mean_slots"] for row in rows)),
        }
        by_season = {str(int(row["season"])): row for row in rows}
        return {"aggregate": aggregate, "by_season": by_season}

    @classmethod
    def _evaluate_draft_sim_for_projector(
        cls,
        projector: "PreseasonProjector",
        seasons: Iterable[int],
    ) -> Dict[str, Any]:
        sweep = cls._load_draft_sim_module()
        rows: List[Dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="preseason-projector-") as tmpdir:
            projector_path = Path(tmpdir) / "preseason_projector.json"
            projector.save(projector_path)
            for season in seasons:
                csv_path = sweep.find_predictions_csv(int(season))
                row = sweep.run_slot_sweep(
                    int(season),
                    "preseason_ml",
                    csv_path,
                    projector_path=projector_path,
                )
                rows.append(row)
        return cls._summarize_draft_sim_rows(rows)

    @classmethod
    def _draft_sim_passes_baseline(
        cls,
        baseline_summary: Dict[str, Any],
        candidate_summary: Dict[str, Any],
    ) -> bool:
        if not baseline_summary or not candidate_summary:
            return False
        base_agg = baseline_summary.get("aggregate", {})
        cand_agg = candidate_summary.get("aggregate", {})
        if not base_agg or not cand_agg:
            return False
        if cand_agg.get("mean_rank_avg", float("inf")) > base_agg.get("mean_rank_avg", float("inf")) + DRAFT_SIM_MEAN_RANK_TOLERANCE:
            return False
        if cand_agg.get("lift_vs_adp_mean_avg", float("-inf")) < base_agg.get("lift_vs_adp_mean_avg", float("-inf")) - DRAFT_SIM_LIFT_TOLERANCE:
            return False
        if cand_agg.get("mean_model_actual_avg", float("-inf")) < base_agg.get("mean_model_actual_avg", float("-inf")) - DRAFT_SIM_MODEL_POINTS_TOLERANCE:
            return False
        return True

    @classmethod
    def _fit_base_components(
        cls,
        position: str,
        pos_df: pd.DataFrame,
        spec: VariantSpec,
    ) -> Optional[Tuple[Ridge, StandardScaler, List[str]]]:
        feature_candidates = spec.feature_map[position]
        features = [feature for feature in feature_candidates if feature in pos_df.columns]
        if len(pos_df) < MIN_SAMPLES or not features:
            return None
        model, scaler = cls._fit_linear_model(pos_df, features, spec.alpha_by_position[position])
        return model, scaler, features

    @classmethod
    def _fit_projector_variant(
        cls,
        prepared_pairs: pd.DataFrame,
        spec: VariantSpec,
    ) -> "PreseasonProjector":
        projector = cls()
        projector.variant_name = spec.name
        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = prepared_pairs[prepared_pairs["position"] == pos].copy()
            components = cls._fit_base_components(pos, pos_df, spec)
            if components is None:
                continue
            model, scaler, features = components
            projector.models[pos] = model
            projector.scalers[pos] = scaler
            projector.feature_names[pos] = features
            if spec.use_calibrator:
                oof = cls._rolling_oof_predictions(pos_df, spec, pos)
                if not oof.empty:
                    calibrator, _ = cls._fit_upstream_calibrator(pos, oof)
                    if calibrator is not None:
                        projector.upstream_calibrators[pos] = calibrator
        projector.is_fitted = len(projector.models) > 0
        return projector

    @classmethod
    def _predict_base(
        cls,
        frame: pd.DataFrame,
        features: List[str],
        scaler: StandardScaler,
        model: Ridge,
    ) -> np.ndarray:
        X = frame.reindex(columns=features, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pred = model.predict(scaler.transform(X))
        return np.maximum(pred, 0.0)

    @classmethod
    def _rolling_oof_predictions(
        cls,
        pos_df: pd.DataFrame,
        spec: VariantSpec,
        position: str,
    ) -> pd.DataFrame:
        if "curr_season" not in pos_df.columns or pos_df["curr_season"].nunique() < 2:
            return pd.DataFrame()
        folds: List[pd.DataFrame] = []
        for holdout_season in sorted(pos_df["curr_season"].dropna().unique()):
            train_df = pos_df[pos_df["curr_season"] != holdout_season]
            test_df = pos_df[pos_df["curr_season"] == holdout_season]
            components = cls._fit_base_components(position, train_df, spec)
            if components is None or test_df.empty:
                continue
            model, scaler, features = components
            fold = test_df.copy()
            fold["raw_pred"] = cls._predict_base(fold, features, scaler, model)
            folds.append(fold)
        if not folds:
            return pd.DataFrame()
        return pd.concat(folds, ignore_index=True)

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
        report: Dict[str, Any] = {
            "overall": {},
            "by_position": {},
            "cohort_market_error": {},
        }
        if df.empty:
            return report

        actual = pd.to_numeric(df["season_total"], errors="coerce")
        base_pred = pd.to_numeric(df["base_pred"], errors="coerce")
        pred = pd.to_numeric(df["pred"], errors="coerce")
        report["overall"] = {
            "base_mae": round(float(np.mean(np.abs(base_pred - actual))), 4),
            "pred_mae": round(float(np.mean(np.abs(pred - actual))), 4),
            "base_bias": round(float(np.mean(base_pred - actual)), 4),
            "pred_bias": round(float(np.mean(pred - actual)), 4),
        }

        top150_base = cls._top150_actual_mean(df, "base_pred")
        top150_pred = cls._top150_actual_mean(df, "pred")
        if top150_base is not None:
            report["overall"]["base_top150_actual_mean"] = round(top150_base, 4)
        if top150_pred is not None:
            report["overall"]["pred_top150_actual_mean"] = round(top150_pred, 4)
        rb24_base = cls._top_actual_mean_by_position(df, "base_pred", "RB", 24)
        rb24_pred = cls._top_actual_mean_by_position(df, "pred", "RB", 24)
        wr24_base = cls._top_actual_mean_by_position(df, "base_pred", "WR", 24)
        wr24_pred = cls._top_actual_mean_by_position(df, "pred", "WR", 24)
        te12_base = cls._top_actual_mean_by_position(df, "base_pred", "TE", 12)
        te12_pred = cls._top_actual_mean_by_position(df, "pred", "TE", 12)
        if rb24_base is not None:
            report["overall"]["base_top24_rb_actual_mean"] = round(rb24_base, 4)
        if rb24_pred is not None:
            report["overall"]["pred_top24_rb_actual_mean"] = round(rb24_pred, 4)
        if wr24_base is not None:
            report["overall"]["base_top24_wr_actual_mean"] = round(wr24_base, 4)
        if wr24_pred is not None:
            report["overall"]["pred_top24_wr_actual_mean"] = round(wr24_pred, 4)
        if te12_base is not None:
            report["overall"]["base_top12_te_actual_mean"] = round(te12_base, 4)
        if te12_pred is not None:
            report["overall"]["pred_top12_te_actual_mean"] = round(te12_pred, 4)

        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = df[df["position"] == pos]
            if pos_df.empty:
                continue
            pos_actual = pd.to_numeric(pos_df["season_total"], errors="coerce")
            pos_base = pd.to_numeric(pos_df["base_pred"], errors="coerce")
            pos_pred = pd.to_numeric(pos_df["pred"], errors="coerce")
            pos_report: Dict[str, Any] = {
                "n": int(len(pos_df)),
                "base_mae": round(float(np.mean(np.abs(pos_base - pos_actual))), 4),
                "pred_mae": round(float(np.mean(np.abs(pos_pred - pos_actual))), 4),
                "base_bias": round(float(np.mean(pos_base - pos_actual)), 4),
                "pred_bias": round(float(np.mean(pos_pred - pos_actual)), 4),
            }
            report["by_position"][pos] = pos_report

        return report

    @classmethod
    def _evaluate_variant(cls, prepared_pairs: pd.DataFrame, spec: VariantSpec) -> Dict[str, Any]:
        evaluation_rows: List[pd.DataFrame] = []
        calibrator_audits: Dict[str, Any] = {}
        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = prepared_pairs[prepared_pairs["position"] == pos].copy()
            if pos_df.empty or "curr_season" not in pos_df.columns:
                continue
            for holdout_season in sorted(pos_df["curr_season"].dropna().unique()):
                train_df = pos_df[pos_df["curr_season"] != holdout_season].copy()
                test_df = pos_df[pos_df["curr_season"] == holdout_season].copy()
                components = cls._fit_base_components(pos, train_df, spec)
                if components is None or test_df.empty:
                    continue
                model, scaler, features = components
                scored_test = test_df.copy()
                base_pred = cls._predict_base(scored_test, features, scaler, model)

                pred = base_pred.copy()
                if spec.use_calibrator:
                    calibration_df = cls._rolling_oof_predictions(train_df, spec, pos)
                    if not calibration_df.empty:
                        calibrator, audit = cls._fit_upstream_calibrator(pos, calibration_df)
                        if audit:
                            calibrator_audits.setdefault(pos, []).append(
                                {"holdout_season": int(holdout_season), **audit}
                            )
                        if calibrator is not None:
                            pred = calibrator.calibrate(scored_test, base_pred)

                fold = scored_test.copy()
                fold["base_pred"] = base_pred
                fold["pred"] = pred
                evaluation_rows.append(fold)

        if not evaluation_rows:
            return {"summary": {}, "predictions": pd.DataFrame(), "calibrator_audits": calibrator_audits}

        predictions = pd.concat(evaluation_rows, ignore_index=True)
        summary = cls._summarize_predictions(predictions)
        summary["variant"] = spec.name
        summary["calibrator_audits"] = calibrator_audits
        return {
            "summary": summary,
            "predictions": predictions,
            "calibrator_audits": calibrator_audits,
        }

    @classmethod
    def _evaluate_projector_on_pairs(
        cls,
        projector: "PreseasonProjector",
        prepared_pairs: pd.DataFrame,
        variant_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows: List[pd.DataFrame] = []
        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = prepared_pairs[prepared_pairs["position"] == pos].copy()
            if pos_df.empty or pos not in projector.models:
                continue
            details = projector.predict_with_details(pos_df, pos)
            scored = pos_df.copy()
            for column in details.columns:
                scored[column] = details[column].to_numpy()
            rows.append(scored)
        predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        summary = cls._summarize_predictions(predictions)
        summary["variant"] = variant_name or projector.variant_name
        return {"summary": summary, "predictions": predictions, "calibrator_audits": {}}

    @classmethod
    def _fit_incumbent_calibrator_candidate(
        cls,
        incumbent_projector: "PreseasonProjector",
        prepared_pairs: pd.DataFrame,
        candidate_name: str = "incumbent_plus_upstream_calibrator",
        positions: Tuple[str, ...] = ("RB", "WR", "TE"),
        adjustment_scale: float = 1.0,
        confidence_power: float = 1.0,
        min_confidence_multiplier: float = 0.20,
        starter_adjustment_multiplier: float = 1.0,
    ) -> "PreseasonProjector":
        candidate = copy.deepcopy(incumbent_projector)
        candidate.variant_name = candidate_name
        candidate.upstream_calibrators = dict(candidate.upstream_calibrators)
        for pos in positions:
            pos_df = prepared_pairs[prepared_pairs["position"] == pos].copy()
            if pos_df.empty or pos not in candidate.models:
                continue
            details = candidate.predict_with_details(pos_df, pos)
            calibration_df = pos_df.copy()
            calibration_df["raw_pred"] = pd.to_numeric(details["pred"], errors="coerce").to_numpy(dtype=float)
            calibrator, _ = cls._fit_upstream_calibrator(pos, calibration_df)
            if calibrator is not None:
                calibrator.max_adjustment_share *= adjustment_scale
                calibrator.confidence_power = confidence_power
                calibrator.min_confidence_multiplier = min_confidence_multiplier
                calibrator.starter_adjustment_multiplier = starter_adjustment_multiplier
                candidate.upstream_calibrators[pos] = calibrator
        candidate.is_fitted = len(candidate.models) > 0
        return candidate

    @classmethod
    def _candidate_passes_gate(
        cls,
        baseline_summary: Dict[str, Any],
        candidate_summary: Dict[str, Any],
    ) -> bool:
        baseline = baseline_summary.get("overall", {})
        candidate = candidate_summary.get("overall", {})
        if not baseline or not candidate:
            return False
        if candidate.get("pred_mae", float("inf")) > baseline.get("pred_mae", float("inf")) + OUTCOME_MAE_TOLERANCE:
            return False
        if abs(candidate.get("pred_bias", 0.0)) > abs(baseline.get("pred_bias", 0.0)) + OUTCOME_BIAS_TOLERANCE:
            return False
        if (
            candidate.get("pred_top150_actual_mean") is not None
            and baseline.get("pred_top150_actual_mean") is not None
            and candidate["pred_top150_actual_mean"] < baseline["pred_top150_actual_mean"] - TOP150_ACTUAL_MEAN_TOLERANCE
        ):
            return False
        if (
            candidate.get("pred_top24_rb_actual_mean") is not None
            and baseline.get("pred_top24_rb_actual_mean") is not None
            and candidate["pred_top24_rb_actual_mean"] < baseline["pred_top24_rb_actual_mean"] - RB_TOP24_ACTUAL_MEAN_TOLERANCE
        ):
            return False
        return True

    @classmethod
    def _market_objective_score(cls, summary: Dict[str, Any]) -> float:
        overall = summary.get("overall", {})
        market_mae = float(overall.get("pred_market_mae", 1e9))
        large_div = float(overall.get("pred_large_divergence_share", 1.0))
        gap_excess = abs(float(overall.get("pred_rb_wr_gap_excess", 1e9)))
        return market_mae + 40.0 * large_div + gap_excess

    @classmethod
    def _draft_sim_candidate_score(cls, summary: Dict[str, Any]) -> float:
        aggregate = summary.get("aggregate", {})
        if not aggregate:
            return float("inf")
        return float(aggregate.get("mean_rank_avg", 1e9)) - (
            float(aggregate.get("lift_vs_adp_mean_avg", -1e9)) / 100.0
        )

    @classmethod
    def _select_variant(
        cls,
        prepared_pairs: pd.DataFrame,
        include_draft_sim_gate: bool = False,
        incumbent_projector: Optional["PreseasonProjector"] = None,
    ) -> Tuple["PreseasonProjector", Dict[str, Any], Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        specs = cls._variant_specs()
        for spec in specs:
            results[spec.name] = cls._evaluate_variant(prepared_pairs, spec)

        baseline = results["ridge_baseline"]["summary"]
        incumbent_summary = (
            cls._evaluate_projector_on_pairs(
                incumbent_projector,
                prepared_pairs,
                variant_name=incumbent_projector.variant_name,
            )["summary"]
            if incumbent_projector is not None
            else {}
        )
        gated: List[Tuple[VariantSpec, Dict[str, Any], bool, float]] = []
        for spec in specs:
            summary = results[spec.name]["summary"]
            passes_gate = spec.name == "ridge_baseline" or cls._candidate_passes_gate(baseline, summary)
            score = cls._market_objective_score(summary) if passes_gate else float("inf")
            gated.append((spec, summary, passes_gate, score))

        gated_valid = [row for row in gated if row[2]]
        candidate_projectors: Dict[str, PreseasonProjector] = {}
        baseline_rows = cls._load_draft_sim_baseline() if include_draft_sim_gate else {}
        draft_sim_seasons = sorted(int(season) for season in baseline_rows.keys()) or [2024, 2025]
        if include_draft_sim_gate:
            if incumbent_projector is not None:
                draft_sim_baseline = cls._evaluate_draft_sim_for_projector(
                    incumbent_projector,
                    draft_sim_seasons,
                )
            else:
                draft_sim_baseline = cls._summarize_draft_sim_rows(list(baseline_rows.values()))
        else:
            draft_sim_baseline = {}
        incumbent_variant = incumbent_projector.variant_name if incumbent_projector is not None else None

        if include_draft_sim_gate and draft_sim_baseline:
            for spec, _, passes_gate, _ in gated:
                if not passes_gate:
                    continue
                projector = cls._fit_projector_variant(prepared_pairs, spec)
                candidate_projectors[spec.name] = projector
                draft_sim_summary = cls._evaluate_draft_sim_for_projector(
                    projector,
                    draft_sim_seasons,
                )
                results[spec.name]["draft_sim_summary"] = draft_sim_summary
                results[spec.name]["draft_sim_passes_gate"] = cls._draft_sim_passes_baseline(
                    draft_sim_baseline,
                    draft_sim_summary,
                )
                results[spec.name]["draft_sim_candidate_score"] = cls._draft_sim_candidate_score(
                    draft_sim_summary
                )
            if incumbent_projector is not None:
                incumbent_candidates = [
                    cls._fit_incumbent_calibrator_candidate(
                        incumbent_projector,
                        prepared_pairs,
                    ),
                    cls._fit_incumbent_calibrator_candidate(
                        incumbent_projector,
                        prepared_pairs,
                        candidate_name="incumbent_plus_conservative_upstream_calibrator",
                        adjustment_scale=0.55,
                        confidence_power=2.0,
                        min_confidence_multiplier=0.0,
                        starter_adjustment_multiplier=0.0,
                    ),
                ]
                for incumbent_candidate in incumbent_candidates:
                    candidate_projectors[incumbent_candidate.variant_name] = incumbent_candidate
                    incumbent_candidate_eval = cls._evaluate_projector_on_pairs(
                        incumbent_candidate,
                        prepared_pairs,
                        variant_name=incumbent_candidate.variant_name,
                    )
                    incumbent_candidate_summary = incumbent_candidate_eval["summary"]
                    incumbent_candidate_passes = cls._candidate_passes_gate(
                        incumbent_summary or baseline,
                        incumbent_candidate_summary,
                    )
                    incumbent_candidate_draft = cls._evaluate_draft_sim_for_projector(
                        incumbent_candidate,
                        draft_sim_seasons,
                    )
                    results[incumbent_candidate.variant_name] = {
                        "summary": incumbent_candidate_summary,
                        "predictions": incumbent_candidate_eval["predictions"],
                        "calibrator_audits": {},
                        "draft_sim_summary": incumbent_candidate_draft,
                        "draft_sim_passes_gate": incumbent_candidate_passes
                        and cls._draft_sim_passes_baseline(
                            draft_sim_baseline, incumbent_candidate_draft
                        ),
                        "draft_sim_candidate_score": cls._draft_sim_candidate_score(
                            incumbent_candidate_draft
                        ),
                        "passes_gate": incumbent_candidate_passes,
                    }
        else:
            for spec, _, passes_gate, _ in gated:
                if passes_gate:
                    results[spec.name]["draft_sim_summary"] = {}
                    results[spec.name]["draft_sim_passes_gate"] = None
                    results[spec.name]["draft_sim_candidate_score"] = None

        promoted_projector: PreseasonProjector
        promoted_summary: Dict[str, Any]
        candidate_variant_name = min(gated_valid, key=lambda row: row[3])[0].name if gated_valid else specs[0].name

        passing_names = [
            spec.name
            for spec, _, passes_gate, _ in gated
            if passes_gate and results[spec.name].get("draft_sim_passes_gate") is True
        ]
        incumbent_candidate_names = [
            "incumbent_plus_upstream_calibrator",
            "incumbent_plus_conservative_upstream_calibrator",
        ]
        for incumbent_candidate_name in incumbent_candidate_names:
            if results.get(incumbent_candidate_name, {}).get("draft_sim_passes_gate") is True:
                passing_names.append(incumbent_candidate_name)

        if include_draft_sim_gate and draft_sim_baseline and passing_names:
            selected_name = min(
                passing_names,
                key=lambda name: (
                    results[name].get("draft_sim_candidate_score", float("inf")),
                    cls._market_objective_score(results[name]["summary"]),
                ),
            )
            promoted_projector = candidate_projectors[selected_name]
            promoted_summary = results[selected_name]["summary"]
            promotion_decision = "promote_candidate"
            candidate_variant_name = selected_name
        else:
            selected_spec = min(gated_valid, key=lambda row: row[3])[0] if gated_valid else specs[0]
            promoted_summary = results[selected_spec.name]["summary"]
            promoted_projector = candidate_projectors.get(
                selected_spec.name,
                cls._fit_projector_variant(prepared_pairs, selected_spec),
            )
            promotion_decision = "promote_candidate"
            if include_draft_sim_gate and draft_sim_baseline and incumbent_projector is not None:
                promotion_decision = "hold_incumbent"
                promoted_projector = incumbent_projector
                promoted_summary = incumbent_projector.get_upstream_audit_report() or incumbent_summary or {}

        selection_report = {
            "selected_variant": promoted_projector.variant_name or candidate_variant_name,
            "candidate_variant": candidate_variant_name,
            "baseline_variant": "ridge_baseline",
            "incumbent_variant": incumbent_variant,
            "promotion_decision": promotion_decision,
            "historical_tolerance": {
                "outcome_mae_tolerance": OUTCOME_MAE_TOLERANCE,
                "outcome_bias_tolerance": OUTCOME_BIAS_TOLERANCE,
                "top150_actual_mean_tolerance": TOP150_ACTUAL_MEAN_TOLERANCE,
                "top24_rb_actual_mean_tolerance": RB_TOP24_ACTUAL_MEAN_TOLERANCE,
            },
            "draft_sim_tolerance": {
                "mean_rank_tolerance": DRAFT_SIM_MEAN_RANK_TOLERANCE,
                "lift_tolerance": DRAFT_SIM_LIFT_TOLERANCE,
                "mean_model_points_tolerance": DRAFT_SIM_MODEL_POINTS_TOLERANCE,
            },
            "draft_sim_baseline": draft_sim_baseline,
            "variants": {
                **{
                    spec.name: {
                        "passes_gate": passes_gate,
                        "market_objective_score": None if not np.isfinite(score) else round(score, 4),
                        "draft_sim_passes_gate": results[spec.name].get("draft_sim_passes_gate"),
                        "draft_sim_candidate_score": cls._safe_round(
                            results[spec.name].get("draft_sim_candidate_score")
                        ),
                        "draft_sim_summary": results[spec.name].get("draft_sim_summary", {}),
                        "summary": results[spec.name]["summary"],
                    }
                    for spec, _, passes_gate, score in gated
                },
                **(
                    {
                        name: {
                            "passes_gate": results[name].get("passes_gate"),
                            "market_objective_score": cls._safe_round(
                                cls._market_objective_score(results[name]["summary"])
                            ),
                            "draft_sim_passes_gate": results[name].get("draft_sim_passes_gate"),
                            "draft_sim_candidate_score": cls._safe_round(
                                results[name].get("draft_sim_candidate_score")
                            ),
                            "draft_sim_summary": results[name].get("draft_sim_summary", {}),
                            "summary": results[name]["summary"],
                        }
                        for name in incumbent_candidate_names
                        if name in results
                    }
                    if any(name in results for name in incumbent_candidate_names)
                    else {}
                ),
            },
        }
        if promotion_decision == "hold_incumbent" and incumbent_projector is not None:
            promoted_projector.selection_report = cls._sanitize_jsonable(selection_report)
            if not promoted_projector.audit_report:
                promoted_projector.audit_report = {}
        return promoted_projector, promoted_summary, selection_report

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
        if isinstance(obj, list):
            return [cls._sanitize_jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return [cls._sanitize_jsonable(v) for v in obj]
        return cls._safe_round(obj)

    @staticmethod
    def _top_errors(df: pd.DataFrame, pred_col: str, top_n: int = 10) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        out = df.copy()
        out["error"] = pd.to_numeric(out[pred_col], errors="coerce") - pd.to_numeric(
            out["season_total"], errors="coerce"
        )
        cols = [
            "player_id",
            "player_name",
            "prior_season",
            "curr_season",
            "position",
            "age",
            "ppg",
            "games_played",
            pred_col,
            "season_total",
            "error",
        ]
        present = [col for col in cols if col in out.columns]
        return (
            out.sort_values("error", ascending=False)
            .head(top_n)[present]
            .round(4)
            .to_dict(orient="records")
        )

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
        frames = []
        season_list = sorted(seasons)
        for i in range(len(season_list) - 1):
            prior = season_list[i]
            curr = season_list[i + 1]

            with db._get_connection() as conn:
                prior_df = pd.read_sql_query(
                    """
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
                      AND pws.week <= 18
                    GROUP BY pws.player_id, p.name, p.position, p.birth_date, r.years_exp
                    HAVING COUNT(*) >= ?
                    """,
                    conn,
                    params=(prior, prior, MIN_GAMES),
                )
                curr_df = pd.read_sql_query(
                    """
                    SELECT player_id, SUM(fantasy_points) AS season_total
                    FROM player_weekly_stats
                    WHERE season = ?
                    GROUP BY player_id
                    HAVING COUNT(*) >= 4
                    """,
                    conn,
                    params=(curr,),
                )

            if prior_df.empty or curr_df.empty:
                continue

            merged = prior_df.merge(curr_df, on="player_id")
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

    def fit(
        self,
        pairs_df: pd.DataFrame,
        include_draft_sim_gate: bool = False,
        incumbent_projector: Optional["PreseasonProjector"] = None,
    ) -> "PreseasonProjector":
        prepared_pairs = self._prepare_feature_frame(pairs_df)
        selected_projector, selected_summary, selection_report = self._select_variant(
            prepared_pairs,
            include_draft_sim_gate=include_draft_sim_gate,
            incumbent_projector=incumbent_projector,
        )

        self.models = selected_projector.models
        self.scalers = selected_projector.scalers
        self.feature_names = selected_projector.feature_names
        self.upstream_calibrators = selected_projector.upstream_calibrators
        self.legacy_veteran_elite_calibration = selected_projector.legacy_veteran_elite_calibration
        self.legacy_fragile_role_calibration = selected_projector.legacy_fragile_role_calibration
        self.variant_name = selected_projector.variant_name
        self.selection_report = self._sanitize_jsonable(selection_report)

        audit = dict(selected_summary) if selected_summary else dict(selected_projector.audit_report or {})
        audit["variant_name"] = self.variant_name
        audit["selected_variant"] = self.variant_name
        audit["selection_report"] = self.selection_report
        self.audit_report = self._sanitize_jsonable(audit)
        self.is_fitted = len(self.models) > 0
        return self

    def _prepare_for_position(self, prior_season_df: pd.DataFrame, position: str) -> pd.DataFrame:
        frame = prior_season_df.copy()
        if "position" not in frame.columns:
            frame["position"] = position
        prepared = self._prepare_feature_frame(frame)
        return prepared

    def predict_with_details(self, prior_season_df: pd.DataFrame, position: str) -> pd.DataFrame:
        if position not in self.models:
            raise ValueError(f"PreseasonProjector not fitted for {position}")
        prepared = self._prepare_for_position(prior_season_df, position)
        base_pred = self._predict_base(
            prepared,
            self.feature_names[position],
            self.scalers[position],
            self.models[position],
        )
        pred = base_pred.copy()
        if position in self.legacy_veteran_elite_calibration:
            calibration = self.legacy_veteran_elite_calibration[position]
            mask = calibration.applies_to(prepared).to_numpy()
            pred = pred.copy()
            pred[mask] = pred[mask] * calibration.factor
        if position in self.legacy_fragile_role_calibration:
            pred = pred.copy()
            for calibration in self.legacy_fragile_role_calibration[position]:
                mask = calibration.applies_to(prepared).to_numpy()
                pred[mask] = pred[mask] * calibration.factor
        calibrator = self.upstream_calibrators.get(position)
        if calibrator is not None:
            pred = calibrator.calibrate(prepared, pred)

        return pd.DataFrame(
            {
                "base_pred": np.maximum(base_pred, 0.0),
                "pred": np.maximum(pred, 0.0),
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
            "variant_name": self.variant_name,
            "selection_report": self.selection_report,
            "audit_report": self.audit_report,
            "positions": {},
            "legacy_veteran_elite_calibration": {
                pos: asdict(calibration)
                for pos, calibration in self.legacy_veteran_elite_calibration.items()
            },
            "legacy_fragile_role_calibration": {
                pos: [asdict(calibration) for calibration in calibrations]
                for pos, calibrations in self.legacy_fragile_role_calibration.items()
            },
        }
        for pos, model in self.models.items():
            scaler = self.scalers[pos]
            payload: Dict[str, Any] = {
                "base_outcome_model": {
                    "features": self.feature_names[pos],
                    "coef": model.coef_.tolist(),
                    "intercept": float(model.intercept_),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "alpha": float(getattr(model, "alpha", 0.0)),
                },
                "upstream_calibrator": None,
            }
            if pos in self.upstream_calibrators:
                payload["upstream_calibrator"] = self.upstream_calibrators[pos].to_payload()
            data["positions"][pos] = payload

        Path(path).write_text(json.dumps(self._sanitize_jsonable(data), indent=2))
        logger.info("PreseasonProjector saved to %s", path)

    @classmethod
    def _load_legacy_schema(cls, data: Dict[str, Any]) -> "PreseasonProjector":
        proj = cls()
        for pos, payload in data.get("positions", {}).items():
            model = Ridge()
            model.coef_ = np.asarray(payload["coef"], dtype=float)
            model.intercept_ = float(payload["intercept"])
            model.n_features_in_ = len(model.coef_)
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=float)
            scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            proj.models[pos] = model
            proj.scalers[pos] = scaler
            proj.feature_names[pos] = payload["features"]

        for pos, payload in data.get("veteran_elite_calibration", {}).items():
            proj.legacy_veteran_elite_calibration[pos] = VeteranEliteCalibration(**payload)
        for pos, payload in data.get("fragile_role_calibration", {}).items():
            if isinstance(payload, list):
                proj.legacy_fragile_role_calibration[pos] = [
                    FragileRoleCalibration(**item) for item in payload
                ]
            else:
                proj.legacy_fragile_role_calibration[pos] = [FragileRoleCalibration(**payload)]

        proj.variant_name = "legacy_ridge_with_cohort_patches"
        proj.selection_report = {
            "selected_variant": proj.variant_name,
            "baseline_variant": proj.variant_name,
        }
        proj.audit_report = {}
        proj.is_fitted = len(proj.models) > 0
        return proj

    @classmethod
    def load(cls, path: Path) -> "PreseasonProjector":
        data = json.loads(Path(path).read_text())
        if not data.get("schema_version"):
            return cls._load_legacy_schema(data)

        proj = cls()
        proj.variant_name = data.get("variant_name")
        proj.selection_report = data.get("selection_report", {})
        proj.audit_report = data.get("audit_report", {})

        for pos, payload in data.get("positions", {}).items():
            base_payload = payload.get("base_outcome_model", {})
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

            calibrator_payload = payload.get("upstream_calibrator")
            if calibrator_payload:
                # Strip legacy market fields when loading old artifacts
                for legacy_field in ("market_weight_cap", "market_confidence_power"):
                    calibrator_payload.pop(legacy_field, None)
                proj.upstream_calibrators[pos] = UpstreamCalibrator.from_payload(calibrator_payload)

        for pos, payload in data.get("legacy_veteran_elite_calibration", {}).items():
            proj.legacy_veteran_elite_calibration[pos] = VeteranEliteCalibration(**payload)
        for pos, payload in data.get("legacy_fragile_role_calibration", {}).items():
            proj.legacy_fragile_role_calibration[pos] = [
                FragileRoleCalibration(**item) for item in payload
            ]

        proj.is_fitted = len(proj.models) > 0
        return proj

    def get_bias_report(self) -> Dict[str, Dict[str, Any]]:
        return self._deep_get(self.audit_report, ("by_position",), {})

    def get_fragile_role_report(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._deep_get(self.audit_report, ("calibrator_audits",), {})

    def get_upstream_audit_report(self) -> Dict[str, Any]:
        return self.audit_report

    def get_selection_report(self) -> Dict[str, Any]:
        return self.selection_report

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        seasons: List[int],
        db=None,
        include_draft_sim_gate: bool = True,
        incumbent_path: Path = LIVE_PROJECTOR_PATH,
    ) -> Tuple["PreseasonProjector", pd.DataFrame]:
        from src.utils.database import DatabaseManager

        db = db or DatabaseManager()
        pairs_df = cls._build_season_pairs(db, seasons)
        if pairs_df.empty:
            raise ValueError("No season pairs found — check database has 2+ seasons of data")
        incumbent_projector = None
        if include_draft_sim_gate and incumbent_path.exists():
            try:
                incumbent_projector = cls.load(incumbent_path)
            except Exception as exc:
                logger.warning("Could not load incumbent projector from %s: %s", incumbent_path, exc)
        proj = cls()
        proj.fit(
            pairs_df,
            include_draft_sim_gate=include_draft_sim_gate,
            incumbent_projector=incumbent_projector,
        )
        return proj, pairs_df
