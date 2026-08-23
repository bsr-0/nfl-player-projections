"""Feature preparation pipeline functions for NFL model training."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import POSITIONS, MODELS_DIR, MODEL_CONFIG
from src.features.feature_engineering import FeatureEngineer, PositionFeatureEngineer
from src.features.utilization_score import (
    calculate_utilization_scores,
    recalculate_utilization_with_weights,
    UtilizationScoreCalculator,
    save_percentile_bounds,
    load_percentile_bounds,
    validate_percentile_bounds_meta,
    SNAP_IMPUTATION_FILENAME,
    fit_snap_imputation,
    save_snap_imputation,
    load_snap_imputation,
    validate_snap_imputation_meta,
    apply_snap_imputation,
)
from src.features.utilization_weight_optimizer import fit_utilization_weights, UTIL_COMPONENTS
from src.models.ensemble import ModelTrainer
from src.data.lineage import (
    get_artifact_id,
    persist_dataframe_artifact,
    set_artifact_id,
)
from src.models.utilization_to_fp import train_utilization_to_fp_per_position

logger = logging.getLogger(__name__)


def _create_horizon_targets(df: pd.DataFrame, n_weeks: List[int] = None) -> pd.DataFrame:
    """Create causal horizon targets within each player-season boundary."""
    if df.empty:
        return df
    n_weeks = n_weeks or [1, 4, 18]
    out = df.copy()
    group_cols = ["player_id", "season"] if "season" in out.columns else ["player_id"]

    def _forward_window(series: pd.Series, window: int, agg: str) -> pd.Series:
        """
        Aggregate future values x[t+1:t+window] for each row t.
        Uses reverse rolling to keep strict forward-looking targets.

        Strict min_periods to avoid noisy/biased targets:
        - For sum targets (fantasy points): require 75% of window to prevent
          systematic underestimation for late-season rows. An 18-week sum
          with only 9 games would be half the expected magnitude, teaching
          the model that late-season == low production.
        - For mean targets (utilization): require 60% of window since means
          are scale-invariant, but very short windows produce high-variance
          estimates.
        - 1-week targets always require exactly 1 future game (no change).
        Rows with insufficient future data become NaN and are excluded during
        training.
        """
        shifted = series.shift(-1)
        rev = shifted.iloc[::-1]
        if window <= 1:
            min_p = 1
        elif agg == "sum":
            # Require 75% of window for sums to prevent scale bias
            min_p = max(int(np.ceil(window * 0.75)), 2)
        else:
            # Require 60% of window for means (less sensitive to count)
            min_p = max(int(np.ceil(window * 0.60)), 2)
        if agg == "sum":
            return rev.rolling(window=window, min_periods=min_p).sum().iloc[::-1]
        return rev.rolling(window=window, min_periods=min_p).mean().iloc[::-1]

    for nw in n_weeks:
        out[f"target_{nw}w"] = out.groupby(group_cols)["fantasy_points"].transform(
            lambda x, w=nw: _forward_window(x, window=w, agg="sum")
        )
    # Use raw (unnormalized) utilization score for targets when available
    # to decouple targets from percentile normalization parameters.
    util_col = "utilization_score_raw" if "utilization_score_raw" in out.columns else "utilization_score"
    if util_col in out.columns:
        out["target_util_1w"] = out.groupby(group_cols)[util_col].shift(-1)
        for nw in [w for w in n_weeks if w != 1]:
            out[f"target_util_{nw}w"] = out.groupby(group_cols)[util_col].transform(
                lambda x, w=nw: _forward_window(x, window=w, agg="mean")
            )
    return out


def _apply_with_temporal_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transform_fn,
    label: str,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a transformation so test rows can use historical context from prior
    train seasons, but train rows are NEVER influenced by test data.

    Strategy: transform train alone first, then transform train+test combined
    and keep only the test rows from the combined result. This ensures:
    - Train features are computed using train data only (no test leakage)
    - Test features can see train-season history (expanding/rolling windows)
    """
    if train_df.empty and test_df.empty:
        return train_df, test_df

    # Step 1: Transform train data alone — train features see only train data
    train_out = transform_fn(train_df.copy(), **kwargs)

    # Step 2: Transform combined (train + test) — test rows benefit from
    # train-season historical context in expanding/rolling windows
    if test_df.empty:
        test_out = test_df.copy()
    else:
        split_col = "__split_context_marker__"
        train_in = train_df.copy()
        test_in = test_df.copy()
        train_in[split_col] = 0
        test_in[split_col] = 1
        combined = pd.concat([train_in, test_in], ignore_index=True, sort=False)

        sort_cols = [c for c in ["season", "week", "player_id"] if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)

        transformed = transform_fn(combined, **kwargs)
        if split_col not in transformed.columns:
            raise ValueError(f"{label}: split marker missing after transform")

        # Only keep test rows from the combined result
        test_out = transformed[transformed[split_col] == 1].drop(columns=[split_col]).reset_index(drop=True)

    print(f"  Applied {label} (train-only features, test with context): train={len(train_out)}, test={len(test_out)}")
    return train_out, test_out


def add_utilization_scores(data: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Add utilization scores only (for two-phase prep when fitting weights from data)."""
    return calculate_utilization_scores(data, weights=weights)


def add_engineered_features(data: pd.DataFrame, position: str = None) -> pd.DataFrame:
    """Add feature engineering (rolling, lag, trend, etc.) - assumes utilization_score exists."""
    if position:
        engineer = PositionFeatureEngineer(position)
        return engineer.create_features(data)
    engineer = FeatureEngineer()
    return engineer.create_features(data)


def add_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add advanced rookie/injury/combine features.

    Raises rather than swallowing, as of FEATURE_VERSION 34. This used to
    catch bare `Exception`, print a line, and return the frame unchanged --
    which is how the entire module (rookie profiles, combine scores, injury
    predictions) silently failed to run for every v22-v26 training run, on a
    KeyError nobody saw. See GAPS.md, "The rookie feature set is NOMINAL".

    Swallowing is no longer defensible: this module owns 9+ columns that are
    declared in CAUSAL_FEATURES, and `is_rookie` itself. Downstream fold code
    filters features to those actually present, so a silent skip does not
    error -- it just trains a quietly weaker model on a quietly different
    feature set, which is the worst available outcome.
    """
    from src.features.advanced_rookie_injury import add_advanced_rookie_injury_features
    return add_advanced_rookie_injury_features(data)


def prepare_features(data: pd.DataFrame, position: str = None,
                     utilization_weights: dict = None) -> pd.DataFrame:
    """Prepare features for training (utilization + engineered features + advanced features)."""
    # Conference FIRST, on the raw frame, because it reads `draft_college` and
    # `draft_season` and both are destroyed downstream by blanket fillers:
    # add_utilization_scores fills draft_season NaN -> 0, and
    # add_advanced_rookie_injury_features mode-fills draft_college, which
    # replaced every undrafted player's missing college with "Ohio St." (1,416
    # rows -> 12,157) and handed them a fabricated Big Ten pedigree. Resolving
    # here means the feature is derived from what the database actually said.
    from src.features.college_conference import add_conference_features
    data = add_conference_features(data)
    print("Calculating utilization scores...")
    data = add_utilization_scores(data, weights=utilization_weights)
    print("Engineering features...")
    data = add_engineered_features(data, position=position)
    return add_advanced_features(data)


def _infer_bounded_columns(df: pd.DataFrame) -> List[str]:
    """
    Select bounded/percentage-like columns for explicit MinMax scaling.
    Keeps scaling policy deterministic and train/serve consistent.
    """
    if df.empty:
        return []
    candidates: List[str] = []
    # "prob"/"probability" deliberately excluded: those columns (e.g.
    # injury_prob_advanced/combined, rookie_breakout_prob, rookie_bust_prob)
    # are already-calibrated model probabilities, often with a narrow
    # natural range by design (injury risk is capped at 0.25 in
    # AdvancedInjuryPredictor.predict_injury_probability). MinMax-stretching
    # a narrow range like [0.10, 0.25] to fill [0, 1] destroys its calibrated
    # meaning -- a real 18% weekly injury risk was getting rescaled to ~55%,
    # crushing predict.py's `availability = 1 - injury_prob_combined`
    # multiplier for nearly every player (GAPS.md, 2026-08-07).
    bounded_tokens = ("pct", "rate", "share", "percentage")
    for col in df.columns:
        if col.startswith("target_util_") or (
            col.startswith("target_") and (col.endswith("w") or col[7:8].isdigit())
        ):
            continue
        if col in {"fantasy_points", "predicted_points", "predicted_utilization"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lower_col = col.lower()
        if not any(t in lower_col for t in bounded_tokens):
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 3:
            continue
        lo = float(series.quantile(0.01))
        hi = float(series.quantile(0.99))
        # Accept common bounded ranges like 0-1 and 0-100.
        if lo >= -1e-6 and hi <= 1.5:
            candidates.append(col)
        elif lo >= -1e-6 and hi <= 100.5:
            candidates.append(col)
    return sorted(set(candidates))


def _apply_bounded_scaling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: Path,
) -> Dict[str, Any]:
    """
    Fit MinMax scaler on bounded columns using train only, apply to train/test, persist artifact.
    """
    cols = _infer_bounded_columns(train_df)
    # Train/test column reconciliation. The missing-indicator step in
    # feature_engineering builds `<col>_missing` columns CONDITIONALLY per
    # frame (only when that frame's missingness exceeds 2%), and train/test
    # are engineered separately — so a column can legitimately exist in one
    # and not the other. Without this guard, `test_df[col]` below raises
    # KeyError and (because callers like run_window_comparison wrap fold
    # loading in a broad try/except) the entire fold is silently dropped
    # from results. Observed for real: QB/2025/'all' vanished from a Phase 3
    # grid with no visible error. Indicator columns absent from test mean
    # "no missingness here", so 0 is the semantically correct fill; any
    # OTHER absent column is dropped from scaling rather than invented.
    if not test_df.empty:
        absent = [c for c in cols if c not in test_df.columns]
        for c in absent:
            if c.endswith("_missing"):
                test_df[c] = 0
        cols = [c for c in cols if c in test_df.columns]
    artifact: Dict[str, Any] = {"columns": cols, "scaler": None}
    if not cols:
        return artifact
    scaler = MinMaxScaler()
    # Cast to float first — some bounded columns may be int64 (e.g., count-based
    # rate columns from SQLite) and pandas will reject float assignment into int cols.
    for col in cols:
        if pd.api.types.is_integer_dtype(train_df[col]):
            train_df[col] = train_df[col].astype(float)
        if not test_df.empty and pd.api.types.is_integer_dtype(test_df[col]):
            test_df[col] = test_df[col].astype(float)
    # NaN must survive scaling. This used to fillna(0.0) first, which turned
    # "unknown" into a real value at the bottom of the column's range -- the
    # same filler bug already fixed for the snap columns, and the reason
    # marking a feature unknown upstream had no effect (GAPS.md 2026-08-20).
    # MinMaxScaler cannot consume NaN, so scale the observed rows and write
    # the missing ones back as NaN; LightGBM splits on missing natively.
    # MinMaxScaler is a per-column affine map fixed by that column's min and
    # max, so the placeholder used for NaN must not sit outside the observed
    # range. 0.0 could, and would drag the fitted min down; the column median
    # cannot, by construction. Masked entries are discarded either way.
    def _scale(frame: pd.DataFrame, fit: bool) -> None:
        vals = frame[cols].replace([np.inf, -np.inf], np.nan)
        mask = vals.notna().values
        filled = vals.fillna(vals.median()).fillna(0.0).values
        scaled = scaler.fit_transform(filled) if fit else scaler.transform(filled)
        frame.loc[:, cols] = np.where(mask, scaled, np.nan)

    _scale(train_df, fit=True)
    if not test_df.empty:
        _scale(test_df, fit=False)
    artifact["scaler"] = scaler
    try:
        import joblib
        joblib.dump(artifact, path)
        print(f"  Saved bounded feature scaler artifact: {path.name} ({len(cols)} columns)")
    except Exception as e:
        print(f"  Bounded scaler artifact save skipped: {e}")
    return artifact


def _prepare_training_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    positions: list,
    tune_hyperparameters: bool,
    n_trials: int,
    fast: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, "ModelTrainer"]:
    """Shared preprocessing pipeline used by both train_models() and _run_one_fold().

    Handles: DVP, external features, season-long features, utilization scores,
    horizon targets, util weight optimization, feature engineering, bounded scaling,
    winsorization, model training, and util-to-fp conversion.

    When fast=True, skips QB dual-target selection by not passing test_data to
    the model trainer (QB defaults to utilization or FP fallback path).

    Returns (train_data, test_data, trainer).
    """
    from config.settings import MODELS_DIR

    # DVP
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        db.ensure_team_defense_stats()
    except Exception as e:
        logger.warning("Team defense stats (DVP) skipped: %s", e)

    # Team offensive output, for DVOA-style opponent-adjustment of the
    # above (GAPS.md §11.1.F).
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        db.ensure_team_offense_stats()
    except Exception as e:
        logger.warning("Team offense stats (DVOA adjustment) skipped: %s", e)

    # Offensive personnel grouping usage (GAPS.md §3.3/§11.1.E). Only
    # fetches seasons not already present, so this is cheap after the
    # initial backfill; new weeks in the current season get picked up here.
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        db.ensure_team_personnel_stats()
    except Exception as e:
        logger.warning("Team personnel grouping stats skipped: %s", e)

    # External (Vegas, injury, weather) with shared train/test temporal context.
    try:
        from src.data.external_data import add_external_features
        all_seasons = sorted(set(train_data["season"].dropna().astype(int)) | set(test_data["season"].dropna().astype(int)))
        train_data, test_data = _apply_with_temporal_context(
            train_data, test_data, add_external_features, "external features",
            seasons=all_seasons,
        )
    except Exception as e:
        logger.warning("External features (weather/injury/Vegas) skipped: %s", e)

    # Season-long draft/rookie context with shared temporal context.
    try:
        from src.features.season_long_features import add_season_long_features
        train_data, test_data = _apply_with_temporal_context(
            train_data, test_data, add_season_long_features, "season-long features",
        )
    except Exception as e:
        logger.warning("Season-long features skipped: %s", e)

    # Utilization scores with train-only percentile bounds
    team_df = pd.DataFrame()
    util_calc = UtilizationScoreCalculator(weights=None)
    train_seasons_list = []
    if "season" in train_data.columns:
        train_seasons_list = sorted({int(s) for s in train_data["season"].dropna().unique()})
    bounds_meta = {
        "train_seasons": train_seasons_list,
        "min_season": min(train_seasons_list) if train_seasons_list else None,
        "max_season": max(train_seasons_list) if train_seasons_list else None,
        "created_at": datetime.now().isoformat(),
    }
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    for pos in POSITIONS:
        util_calc.fit_percentile_bounds(
            train_data, pos, UTIL_COMPONENTS.get(pos, []), metadata=bounds_meta
        )
    bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
    save_percentile_bounds(util_calc.position_percentiles, bounds_path, metadata=bounds_meta)
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    loaded_bounds, loaded_meta = load_percentile_bounds(bounds_path, return_meta=True)
    if not validate_percentile_bounds_meta(loaded_meta, train_seasons_list):
        raise ValueError(
            "Utilization percentile bounds metadata mismatch; "
            "refusing to use bounds not fit on the current training seasons."
        )
    test_data = calculate_utilization_scores(test_data, team_df=team_df, weights=None, percentile_bounds=loaded_bounds)

    # Compute raw (unnormalized) utilization scores for target derivation
    # to decouple targets from percentile normalization parameters.
    from src.features.utilization_score import compute_raw_utilization_score
    train_data = compute_raw_utilization_score(train_data)
    test_data = compute_raw_utilization_score(test_data)

    # Horizon targets (season-bounded) — uses utilization_score_raw when available
    train_data = _create_horizon_targets(train_data, n_weeks=[1, 4, 18])
    test_data = _create_horizon_targets(test_data, n_weeks=[1, 4, 18])

    # Data-driven utilization weight optimization
    util_weights = fit_utilization_weights(
        train_data,
        target_col="target_util_1w" if "target_util_1w" in train_data.columns else "target_1w",
        tune_alpha_cv=True,
    )
    train_data = recalculate_utilization_with_weights(train_data, util_weights)
    test_data = recalculate_utilization_with_weights(test_data, util_weights)
    # Recompute raw scores with optimized weights and rebuild targets
    train_data = compute_raw_utilization_score(train_data, weights=util_weights)
    test_data = compute_raw_utilization_score(test_data, weights=util_weights)
    train_data = _create_horizon_targets(train_data, n_weeks=[1, 4, 18])
    test_data = _create_horizon_targets(test_data, n_weeks=[1, 4, 18])
    with open(MODELS_DIR / "utilization_weights.json", "w") as f:
        json.dump(util_weights, f, indent=2)

    # Feature engineering
    train_data, test_data = _apply_with_temporal_context(
        train_data, test_data,
        lambda d: add_advanced_features(add_engineered_features(d)),
        "feature engineering",
    )

    # Snap-share imputation. Placed here because feature engineering is where
    # snap_share_pct_roll3_mean is created, and deliberately shaped like the
    # percentile-bounds block above: fit on train rows, persist with the same
    # train_seasons metadata, reload, validate, then apply the LOADED values
    # to both halves. Recomputing at apply time would make the transformation
    # depend on the population it is applied to -- the exact defect this
    # replaces, where an unknown snap share silently became 0.0.
    # Inert under the default mode: the A/B rejected median imputation, so
    # snap_share_pct carries no NaN and there would be nothing to fill. Gated
    # rather than left running so no artifact is written that implies a
    # production behaviour we do not have.
    from src.features.utilization_score import SNAP_MISSINGNESS_MODE
    if SNAP_MISSINGNESS_MODE != "zero":
        snap_path = MODELS_DIR / SNAP_IMPUTATION_FILENAME
        save_snap_imputation(fit_snap_imputation(train_data), snap_path,
                             metadata=bounds_meta)
        loaded_snap, loaded_snap_meta = load_snap_imputation(snap_path, return_meta=True)
        if not validate_snap_imputation_meta(loaded_snap_meta, train_seasons_list):
            raise ValueError(
                "Snap imputation metadata mismatch; refusing to use values not "
                "fit on the current training seasons."
            )
        train_data = apply_snap_imputation(train_data, loaded_snap)
        test_data = apply_snap_imputation(test_data, loaded_snap)

    parent_silver_ids = [i for i in [get_artifact_id(train_data), get_artifact_id(test_data)] if i]
    silver_train_meta = persist_dataframe_artifact(
        train_data.copy(),
        layer="silver",
        table="training_features",
        run_id=f"trainprep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        metadata={
            "source": "feature_engineering_pipeline",
            "seasons": sorted({int(s) for s in train_data["season"].dropna().unique()}) if "season" in train_data.columns else [],
            "normalization": "post temporal-context feature engineering + utilization/targets",
        },
        parent_artifact_ids=parent_silver_ids,
    )
    train_data = set_artifact_id(train_data, silver_train_meta["artifact_id"])
    _apply_bounded_scaling(
        train_data, test_data, MODELS_DIR / "feature_scaler_bounded.joblib",
    )

    # Player embeddings: PCA-based dense representations from aggregated train stats
    try:
        from src.models.advanced_techniques import PlayerEmbeddings
        print("Computing player embeddings (PCA on aggregated stats)...")
        emb = PlayerEmbeddings(embedding_dim=8)
        emb.fit(train_data)  # Fit on train only to avoid leakage
        # Add embedding columns to both train and test
        for df_ref, label in [(train_data, "train"), (test_data, "test")]:
            if "player_id" not in df_ref.columns:
                continue
            emb_matrix = np.array([emb.get_embedding(pid) for pid in df_ref["player_id"]])
            for i in range(emb_matrix.shape[1]):
                df_ref[f"player_emb_{i}"] = emb_matrix[:, i]
        print(f"  Added {emb.embedding_dim} player embedding features")
    except Exception as e:
        logger.warning("Player embeddings skipped: %s", e)

    # Winsorize targets at 1st/99th percentile per position.
    # Bounds are derived from train data and applied symmetrically to both
    # train and test to avoid distribution mismatch during evaluation.
    winsor_bounds = {}  # (pos, col) -> (lo, hi)
    for pos in ["QB", "RB", "WR", "TE"]:
        mask = train_data["position"] == pos
        target_cols = [f"target_{n}w" for n in [1, 4, 18]] + [
            "target_util_1w", "target_util_4w", "target_util_18w"]
        for col in target_cols:
            if col not in train_data.columns:
                continue
            valid = train_data.loc[mask, col].dropna()
            if len(valid) < 20:
                continue
            lo, hi = valid.quantile(0.01), valid.quantile(0.99)
            winsor_bounds[(pos, col)] = (lo, hi)
            train_data.loc[mask, col] = train_data.loc[mask, col].clip(lo, hi)

    # Apply the same train-derived bounds to test targets
    for (pos, col), (lo, hi) in winsor_bounds.items():
        if col in test_data.columns:
            test_mask = test_data["position"] == pos
            test_data.loc[test_mask, col] = test_data.loc[test_mask, col].clip(lo, hi)

    # Train models (fast mode: skip QB dual-target comparison by withholding test_data)
    trainer = ModelTrainer()
    trainer.train_all_positions(
        train_data, positions=positions, tune_hyperparameters=tune_hyperparameters,
        n_weeks_list=[1, 4, 18], test_data=None if fast else test_data,
    )

    # Utilization -> FP conversion (only for positions trained on util targets)
    pos_target_cfg = MODEL_CONFIG.get("position_target_type", {})
    converter_positions = [
        pos for pos in ["RB", "WR", "TE", "QB"]
        if pos_target_cfg.get(pos, "util") != "fp"
    ]
    if converter_positions:
        try:
            train_utilization_to_fp_per_position(train_data, positions=converter_positions, fast=fast)
        except Exception as e:
            logger.warning("Utilization-to-FP conversion training skipped: %s", e)

    return train_data, test_data, trainer
