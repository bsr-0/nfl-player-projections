"""Phase 2 + Phase 3 + Phase 4 (next_focus.md) walk-forward evaluation for
the single-week PPR model.

Phase 2: naive baselines vs. six new architectures vs. the existing
production methodology, on raw `fantasy_points` (run_comparison).
Phase 3: training-window size x recency-weighting optimization for the
best Phase 2 architectures per position (run_window_comparison).
Phase 4: row-level predictions for each position's FINAL_CONFIG (chosen
architecture/window/weighting from Phases 2-3), enabling bucket/tier/
calibration breakdowns in analysis.py (run_final_validation).

Reuses the existing leakage-safe feature pipeline (`_prepare_training_data`
in src/models/feature_preparation.py) rather than re-implementing feature
engineering. Must not modify any production file or overwrite production
model artifacts. `settings.MODELS_DIR` is redirected to a temp directory per
fold (mirroring src/models/train.py's own walk-forward loop at
train.py:1053-1061), but that redirection alone is NOT sufficient: some
modules (e.g. src/models/utilization_to_fp.py:16) do
`from config.settings import MODELS_DIR` at import time, so reassigning
`settings.MODELS_DIR` later doesn't affect their already-bound writes.
`_protect_data_dir()` below is the real safety net — it snapshots an
allowlist of known-leaky model-artifact paths before the fold and restores
any touched files afterward via git (tracked) or deletion (newly-created
untracked). See GAPS.md §7.7/§7.8 for the incidents that made this
necessary — including why it's an allowlist, not a denylist over all of
data/ (a denylist version deleted the production database once already).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from src.models.single_week_ppr.architectures import (
    GBMRegressor,
    HurdleModel,
    QuantileGBM,
    YeoJohnsonHuber,
    naive_baselines,
)

logger = logging.getLogger(__name__)


# ALLOWLIST, not a denylist over all of data/ — see incident note below.
# `data/nfl_data.db` (and *.db-shm/*.db-wal) must NEVER be in this set: it's
# gitignored (untracked) and is legitimately, intentionally written by
# auto_refresh_data() inside load_training_data() as part of normal
# operation. An earlier denylist version of this function treated that
# legitimate write as "pollution" and deleted the production database
# (110K+ rows) because it was untracked — see GAPS.md §7.8. Only add a path
# here after confirming (like these two) that it's a stray model-training
# side effect that ignores MODELS_DIR redirection, never a live data asset.
_PROTECTED_PATHS = (
    Path("data/models"),
    Path("data/data_availability_cache.json"),
    Path("data/utilization_percentile_bounds.json"),
)


@contextmanager
def _protect_data_dir():
    """Snapshot the allowlisted model-artifact paths and restore any file
    the wrapped block touches, regardless of MODELS_DIR redirection.

    `load_training_data()` (called by run_fold before _prepare_training_data)
    writes to data/data_availability_cache.json and data/models/ via
    auto_refresh_data(); utilization weight fitting writes
    data/utilization_percentile_bounds.json — none of these respect a
    redirected MODELS_DIR. This must wrap the WHOLE fold, not just the
    _prepare_training_data call. See module docstring / GAPS.md §7.7-7.8.
    """
    def _snapshot() -> Dict[Path, float]:
        snap = {}
        for target in _PROTECTED_PATHS:
            if target.is_file():
                snap[target] = target.stat().st_mtime
            elif target.is_dir():
                for root, _dirs, files in os.walk(target):
                    for f in files:
                        p = Path(root) / f
                        snap[p] = p.stat().st_mtime
        return snap

    before = _snapshot()
    try:
        yield
    finally:
        after = _snapshot()
        touched = [p for p in after if p not in before or after[p] != before[p]]
        if touched:
            logger.warning(
                "%d allowlisted model-artifact file(s) were written despite "
                "MODELS_DIR redirection — restoring via git: %s",
                len(touched), [str(p) for p in touched],
            )
            tracked, untracked = [], []
            for p in touched:
                rel = p.relative_to(Path.cwd()) if p.is_absolute() else p
                result = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", str(rel)],
                    capture_output=True, cwd=Path.cwd(),
                )
                (tracked if result.returncode == 0 else untracked).append(rel)
            if tracked:
                subprocess.run(["git", "checkout", "--", *[str(p) for p in tracked]],
                                check=True, cwd=Path.cwd())
            for p in untracked:
                (Path.cwd() / p).unlink(missing_ok=True)

DEFAULT_VALIDATION_SEASONS = (2023, 2024, 2025)
COMPARISON_OUTPUT_PATH = Path("data/experiments/phase2_single_week_comparison.csv")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """MAE (primary) + secondary metrics (next_focus.md Phase 2 §7)."""
    aligned = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if aligned.empty:
        return {"mae": np.nan, "rmse": np.nan, "medae": np.nan, "r2": np.nan,
                "spearman": np.nan, "bias": np.nan, "n": 0}
    yt, yp = aligned["y_true"].to_numpy(), aligned["y_pred"].to_numpy()
    return {
        "mae": mean_absolute_error(yt, yp),
        "rmse": mean_squared_error(yt, yp) ** 0.5,
        "medae": median_absolute_error(yt, yp),
        "r2": r2_score(yt, yp) if len(yt) > 1 else np.nan,
        "spearman": spearmanr(yt, yp).correlation if len(yt) > 1 else np.nan,
        "bias": float(np.mean(yp - yt)),
        "n": len(yt),
    }


def compute_quantile_metrics(y_true: pd.Series, quantile_preds: pd.DataFrame) -> dict:
    """Pinball loss + empirical coverage per quantile (architecture E)."""
    aligned = quantile_preds.join(y_true.rename("y_true"), how="inner").dropna()
    out: Dict[str, float] = {}
    for col, q in {"p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9}.items():
        if col not in aligned.columns or aligned.empty:
            continue
        err = aligned["y_true"] - aligned[col]
        out[f"{col}_pinball"] = float(np.mean(np.maximum(q * err, (q - 1) * err)))
        out[f"{col}_coverage"] = float((aligned["y_true"] <= aligned[col]).mean())
    return out


def _existing_methodology_predictions(trainer, test_df: pd.DataFrame, position: str) -> pd.Series:
    """Replicates the prediction-extraction logic in
    src/models/train.py:_run_one_fold (lines ~844-880) for one position,
    without re-running _prepare_training_data a second time. Returns a
    Series of predicted_points aligned to test_df.index (NaN where the
    existing production path has no prediction for a row).
    """
    from config.settings import MODEL_CONFIG
    from src.models.train import _load_qb_target_choice

    preds_out = pd.Series(np.nan, index=test_df.index, dtype=float)
    if position not in trainer.trained_models:
        return preds_out

    converters = {}
    try:
        from src.models.utilization_to_fp import UtilizationToFPConverter
        c = UtilizationToFPConverter.load(position)
        if getattr(c, "is_fitted", False):
            converters[position] = c
    except Exception as e:
        logger.warning("Converter load for %s skipped: %s", position, e)

    qb_target = _load_qb_target_choice()
    multi_model = trainer.trained_models[position]
    pos_mask = test_df["position"] == position
    if pos_mask.sum() < 5:
        return preds_out

    if multi_model is None:
        comp = trainer.component_predictors.get(position)
        if comp is None:
            return preds_out
        preds = comp.predict(test_df.loc[pos_mask].copy())
    else:
        base = multi_model.models.get(1) or list(multi_model.models.values())[0]
        medians = getattr(base, "feature_medians", {})
        for fn in getattr(base, "feature_names", []):
            if fn not in test_df.columns:
                test_df.loc[pos_mask, fn] = medians.get(fn, 0)
        preds = multi_model.predict(test_df.loc[pos_mask].copy(), n_weeks=1)

    preds_out.loc[pos_mask] = preds

    ptc = MODEL_CONFIG.get("position_target_type", {})
    should_convert = (
        position in converters
        and ptc.get(position, "util") == "util"
        and (position != "QB" or qb_target == "util")
    )
    if should_convert:
        eff_df = test_df.loc[pos_mask].copy()
        eff_df["utilization_score"] = preds
        try:
            fp_pred = converters[position].predict(preds, efficiency_df=eff_df)
            preds_out.loc[pos_mask] = fp_pred
        except Exception as e:
            logger.warning("FP conversion for %s skipped: %s", position, e)

    return preds_out


def run_fold(
    position: str,
    test_season: int,
    tune_hyperparameters: bool = False,
    n_trials: int = 0,
    train_seasons_override: Optional[Sequence[int]] = None,
):
    """Loads one (train_seasons, test_season) fold, feature-engineers it via
    the existing leakage-safe pipeline, and extracts the existing-methodology
    prediction for `position` for later comparison.

    train_seasons_override (Phase 3): when given, bypasses load_training_data
    /get_train_test_seasons entirely — which hard-floor training data at
    TRAINING_START_YEAR_DEFAULT (2018) with no override parameter — and
    instead loads raw rows directly via DatabaseManager.get_all_players_for_
    training() and splits by this explicit season list. _prepare_training_data
    doesn't care how the caller split train/test, so this is a safe swap that
    doesn't touch data_manager.py or monkeypatch any config constant (see
    GAPS.md §7.7/§7.8 for why monkeypatching module-level constants here is
    unreliable). Without an override, behavior is identical to Phase 2.

    Redirects config.settings.MODELS_DIR to a temp dir for the duration of
    the call, AND wraps the whole fold (including load_training_data, which
    also writes under data/) in _protect_data_dir() as the real safety net
    (the redirection alone is not reliable — see module docstring).

    Returns (train_df, test_df, existing_methodology_pred, train_seasons).
    """
    import config.settings as settings
    from src.models.feature_preparation import _prepare_training_data

    old_models_dir = settings.MODELS_DIR
    with _protect_data_dir(), tempfile.TemporaryDirectory() as tmp:
        settings.MODELS_DIR = Path(tmp)
        try:
            if train_seasons_override is not None:
                from src.utils.database import DatabaseManager
                train_seasons = list(train_seasons_override)
                combined = DatabaseManager().get_all_players_for_training(position=position)
                train_data = combined[combined["season"].isin(train_seasons)]
                test_data = combined[combined["season"] == test_season]
            else:
                from src.models.data_loading import load_training_data
                train_data, test_data, train_seasons, _ = load_training_data(
                    [position], test_season=test_season, optimize_training_years=False,
                )

            if len(test_data) < 20:
                raise ValueError(f"Not enough test rows for {position} season {test_season}: {len(test_data)}")
            if len(train_data) < 20:
                raise ValueError(f"Not enough train rows for {position} seasons {train_seasons}: {len(train_data)}")

            train_df, test_df, trainer = _prepare_training_data(
                train_data, test_data, [position], tune_hyperparameters, n_trials, fast=True,
            )
            existing_pred = _existing_methodology_predictions(trainer, test_df, position)
        finally:
            settings.MODELS_DIR = old_models_dir

    return train_df, test_df, existing_pred, train_seasons


def _architectures_for_fold() -> Dict[str, object]:
    return {
        "A_gbm_mse": GBMRegressor(objective="regression"),
        "B_gbm_huber": GBMRegressor(objective="huber"),
        "C_gbm_mae": GBMRegressor(objective="regression_l1"),
        "D_hurdle": HurdleModel(threshold=0.0),
        "D_hurdle_t5": HurdleModel(threshold=5.0),
        "F_yeojohnson_huber": YeoJohnsonHuber(),
    }


def _build_feature_matrices(pos_train: pd.DataFrame, pos_test: pd.DataFrame, feature_cols: List[str]):
    """X/y split for train/test, given a feature-column list already run
    through filter_feature_columns.

    Phase 3 note: only fillna(0) for the sklearn fallback path — LightGBM
    handles NaN natively via missing-aware splits, which is the CORRECT
    behavior once pre-2018 rows (structurally missing NGS/EPA/modern
    snap-share columns, not randomly missing) enter the training set via
    window_to_season_list's floor bypass. fillna(0) would make "no data"
    indistinguishable from "zero usage," a real distortion for those rows.
    Phase 2 used unconditional fillna(0), reasonable when all data was
    already 2018+; kept for the sklearn-fallback case only, since sklearn's
    GradientBoostingRegressor can't handle NaN at all.
    """
    from src.models.single_week_ppr.architectures import HAS_LIGHTGBM

    X_train = pos_train[feature_cols]
    X_test = pos_test[feature_cols]
    if not HAS_LIGHTGBM:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    y_train = pos_train["fantasy_points"]
    y_test = pos_test["fantasy_points"]
    return X_train, y_train, X_test, y_test


def run_comparison(
    positions: Optional[Sequence[str]] = None,
    seasons: Sequence[int] = DEFAULT_VALIDATION_SEASONS,
    tune_hyperparameters: bool = False,
    output_path: Path = COMPARISON_OUTPUT_PATH,
) -> pd.DataFrame:
    """Runs the full Phase 2 comparison: baselines + existing methodology +
    architectures A-F, per position per validation season. Saves a tidy
    comparison table and returns it.
    """
    from config.settings import POSITIONS, CAUSAL_FEATURES
    from src.utils.leakage import filter_feature_columns

    positions = list(positions) if positions else POSITIONS
    rows: List[dict] = []

    for position in positions:
        for season in seasons:
            print(f"\n=== {position} / test_season={season} ===")
            try:
                train_df, test_df, existing_pred, _ = run_fold(position, season, tune_hyperparameters)
            except Exception as e:
                logger.warning("Fold %s/%s failed to load: %s", position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position]
            pos_test = test_df[test_df["position"] == position]
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue

            feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
            if not feature_cols:
                logger.warning("Skipping %s/%s: no CAUSAL_FEATURES columns present", position, season)
                continue

            X_train, y_train, X_test, y_test = _build_feature_matrices(pos_train, pos_test, feature_cols)

            # pos_train/pos_test come from separate DB-query DataFrames, so their
            # integer indices overlap (not globally unique) — concat with
            # ignore_index=True and track test rows by position, since
            # concat preserves row order (train rows first, then test rows).
            combined = pd.concat([pos_train, pos_test], ignore_index=True)
            test_uid_index = combined.index[-len(pos_test):]
            for name, series in naive_baselines(combined).items():
                pred = series.loc[test_uid_index]
                pred.index = pos_test.index
                rows.append({"position": position, "season": season, "model": name,
                             **compute_metrics(y_test, pred)})

            rows.append({"position": position, "season": season, "model": "existing_methodology",
                         **compute_metrics(y_test, existing_pred.reindex(pos_test.index))})

            for name, model in _architectures_for_fold().items():
                try:
                    model.fit(X_train, y_train)
                    pred = pd.Series(model.predict(X_test), index=X_test.index)
                    rows.append({"position": position, "season": season, "model": name,
                                 **compute_metrics(y_test, pred)})
                except Exception as e:
                    logger.warning("Architecture %s failed for %s/%s: %s", name, position, season, e)

            try:
                qmodel = QuantileGBM().fit(X_train, y_train)
                qpred = qmodel.predict(X_test)
                rows.append({
                    "position": position, "season": season, "model": "E_quantile_gbm",
                    **compute_metrics(y_test, qpred["p50"]),
                    **compute_quantile_metrics(y_test, qpred),
                })
            except Exception as e:
                logger.warning("QuantileGBM failed for %s/%s: %s", position, season, e)

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved comparison table to {output_path} ({len(result)} rows)")
    return result


# ---------------------------------------------------------------------------
# Phase 3: training-window / recency-weighting comparison
# ---------------------------------------------------------------------------

WINDOW_OUTPUT_PATH = Path("data/experiments/phase3_training_window_comparison.csv")

# Per position: test only B (Huber, the plan's stated primary candidate) +
# that position's Phase 2 MAE winner (data/experiments/
# phase2_single_week_comparison.csv). See Phase 3 plan — weighting/
# architecture refits are cheap once a fold is loaded; the window reload
# (4 positions x 5 windows x 3 seasons = 60 fold-loads) is the expensive
# axis, so architecture count doesn't need to be minimized further.
#
# Updated 2026-08-10 after the Complete Player-Game Panel fix (GAPS.md) —
# every position's Phase 2 winner changed once trained on the corrected,
# non-selection-biased data (data/experiments/
# phase2_single_week_comparison_v2_corrected.csv). RB's literal best-MAE
# was E_quantile_gbm's median, but E is a supplementary floor/median/
# ceiling tool per next_focus.md's own framing, not a competing point-
# estimate architecture -- kept the "winner" selection restricted to the
# point-estimate architectures (A/B/C/D/F), matching original Phase 2
# logic; RB's best among those is C_gbm_mae.
PHASE3_WINNER_ARCHITECTURE = {
    "QB": "F_yeojohnson_huber",
    "RB": "C_gbm_mae",
    "WR": "C_gbm_mae",
    "TE": "B_gbm_huber",
}


def _architectures_for_position(position: str) -> Dict[str, object]:
    all_arch = _architectures_for_fold()
    winner_name = PHASE3_WINNER_ARCHITECTURE[position]
    selected = {"B_gbm_huber": all_arch["B_gbm_huber"]}
    if winner_name != "B_gbm_huber":
        selected[winner_name] = all_arch[winner_name]
    return selected


def _append_row_to_csv(row: dict, output_path: Path) -> None:
    """Appends one row immediately rather than batching a single final
    write — a killed/timed-out run (Phase 2 nearly lost a full 12-fold run
    this way) keeps whatever completed so far.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not output_path.exists()
    df_row.to_csv(output_path, mode="a", header=write_header, index=False)


def run_window_comparison(
    positions: Optional[Sequence[str]] = None,
    seasons: Sequence[int] = DEFAULT_VALIDATION_SEASONS,
    windows: Optional[Sequence[str]] = None,
    weightings: Optional[Sequence[str]] = None,
    tune_hyperparameters: bool = False,
    output_path: Path = WINDOW_OUTPUT_PATH,
) -> pd.DataFrame:
    """Phase 3 (next_focus.md): position x architecture x training-window x
    recency-weighting, using only rolling (walk-forward) validation.

    Deliberately bypasses the codebase's 2018+ training floor via
    run_fold(train_seasons_override=...) so "10y"/"all" windows are actually
    distinct from "5y" for our 2023-2025 validation seasons — see Phase 3
    plan. Naive baselines are NOT recomputed here (they don't depend on
    training window at all — that's Phase 2's job); existing_methodology is
    included once per fold for continuity since it's a free side effect of
    run_fold's _prepare_training_data call.
    """
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.models.single_week_ppr.windows import (
        WINDOW_CANDIDATES, WEIGHTING_SCHEMES, window_to_season_list, compute_recency_weights,
    )
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    positions = list(positions) if positions else POSITIONS
    windows = list(windows) if windows else list(WINDOW_CANDIDATES)
    weightings = list(weightings) if weightings else list(WEIGHTING_SCHEMES)
    rows: List[dict] = []

    for position in positions:
        architectures = _architectures_for_position(position)
        available_seasons = sorted(
            DatabaseManager().get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )

        for season in seasons:
            for window in windows:
                print(f"\n=== {position} / test_season={season} / window={window} ===")
                train_seasons = window_to_season_list(window, season, available_seasons)
                if not train_seasons:
                    logger.warning("Skipping %s/%s/%s: no training seasons available", position, season, window)
                    continue

                try:
                    train_df, test_df, existing_pred, _ = run_fold(
                        position, season, tune_hyperparameters, train_seasons_override=train_seasons,
                    )
                except Exception as e:
                    logger.warning("Fold %s/%s/%s failed to load: %s", position, season, window, e)
                    continue

                pos_train = train_df[train_df["position"] == position]
                pos_test = test_df[test_df["position"] == position]
                if len(pos_test) < 20:
                    logger.warning("Skipping %s/%s/%s: only %d test rows", position, season, window, len(pos_test))
                    continue

                feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
                feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
                if not feature_cols:
                    logger.warning("Skipping %s/%s/%s: no CAUSAL_FEATURES columns present", position, season, window)
                    continue

                X_train, y_train, X_test, y_test = _build_feature_matrices(pos_train, pos_test, feature_cols)
                base_row = {
                    "position": position, "season": season, "window": window,
                    "actual_years": len(train_seasons), "n_train_rows": len(pos_train),
                }

                existing_row = {**base_row, "weighting": "n/a", "model": "existing_methodology",
                                 **compute_metrics(y_test, existing_pred.reindex(pos_test.index))}
                rows.append(existing_row)
                _append_row_to_csv(existing_row, output_path)

                for weighting in weightings:
                    sample_weight = compute_recency_weights(pos_train["season"], weighting)
                    for name, model in architectures.items():
                        try:
                            model.fit(X_train, y_train, sample_weight=sample_weight)
                            pred = pd.Series(model.predict(X_test), index=X_test.index)
                            row = {**base_row, "weighting": weighting, "model": name,
                                   **compute_metrics(y_test, pred)}
                            rows.append(row)
                            _append_row_to_csv(row, output_path)
                        except Exception as e:
                            logger.warning("Architecture %s (%s weighting) failed for %s/%s/%s: %s",
                                            name, weighting, position, season, window, e)

    result = pd.DataFrame(rows)
    print(f"\n{len(result)} rows appended to {output_path}")
    return result


# ---------------------------------------------------------------------------
# Phase 4: row-level predictions for FINAL_CONFIG, for bucket/tier/
# calibration breakdowns (analysis.py)
# ---------------------------------------------------------------------------

ROW_LEVEL_OUTPUT_PATH = Path("data/experiments/phase4_row_level_predictions.csv")

# Column order matches next_focus.md Phase 10's "Save Out-of-Sample
# Predictions" spec (player/position/season/week/actual_ppr/prediction/
# model/fold/training_window/weighting_strategy + p25/p50/p75/p90) so this
# artifact is reusable there rather than rebuilt.
ROW_LEVEL_COLUMNS = [
    "player", "position", "season", "week", "actual_ppr", "prediction", "model",
    "fold", "training_window", "weighting_strategy", "p25", "p50", "p75", "p90",
]


def _append_df_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False)


def _build_row_level_frame(
    pos_test: pd.DataFrame,
    y_test: pd.Series,
    pred_series: Optional[pd.Series],
    model_name: str,
    base_meta: dict,
    extra_cols: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=pos_test.index)
    out["player"] = pos_test["player_id"] if "player_id" in pos_test.columns else None
    out["week"] = pos_test["week"]
    out["actual_ppr"] = y_test
    out["prediction"] = pred_series.reindex(pos_test.index) if pred_series is not None else np.nan
    out["model"] = model_name
    for k, v in base_meta.items():
        out[k] = v
    for col in ("p25", "p50", "p75", "p90"):
        out[col] = extra_cols[col].reindex(pos_test.index) if extra_cols and col in extra_cols else np.nan
    return out[ROW_LEVEL_COLUMNS].reset_index(drop=True)


def run_final_validation(
    positions: Optional[Sequence[str]] = None,
    seasons: Sequence[int] = DEFAULT_VALIDATION_SEASONS,
    tune_hyperparameters: bool = False,
    output_path: Path = ROW_LEVEL_OUTPUT_PATH,
) -> pd.DataFrame:
    """Phase 4: for each position's FINAL_CONFIG (chosen architecture/window/
    weighting from Phases 2-3), saves ROW-LEVEL predicted-vs-actual pairs
    (not aggregated metrics) for baselines + existing_methodology + the
    winning architecture + QuantileGBM. analysis.py consumes this for
    bucket/tier/calibration breakdowns.
    """
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.windows import window_to_season_list, compute_recency_weights
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    positions = list(positions) if positions else POSITIONS
    all_frames: List[pd.DataFrame] = []
    all_arch = _architectures_for_fold()

    for position in positions:
        cfg = FINAL_CONFIG[position]
        available_seasons = sorted(
            DatabaseManager().get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )

        for season in seasons:
            print(f"\n=== FINAL VALIDATION {position} / test_season={season} / {cfg} ===")
            train_seasons = window_to_season_list(cfg["window"], season, available_seasons)
            if not train_seasons:
                logger.warning("Skipping %s/%s: no training seasons available", position, season)
                continue
            try:
                train_df, test_df, existing_pred, _ = run_fold(
                    position, season, tune_hyperparameters, train_seasons_override=train_seasons,
                )
            except Exception as e:
                logger.warning("Fold %s/%s failed to load: %s", position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position]
            pos_test = test_df[test_df["position"] == position]
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue

            feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
            if not feature_cols:
                logger.warning("Skipping %s/%s: no CAUSAL_FEATURES columns present", position, season)
                continue

            X_train, y_train, X_test, y_test = _build_feature_matrices(pos_train, pos_test, feature_cols)
            sample_weight = compute_recency_weights(pos_train["season"], cfg["weighting"])
            base_meta = {
                "position": position, "season": season, "fold": season,
                "training_window": cfg["window"], "weighting_strategy": cfg["weighting"],
            }

            # Baselines (free — same combined-index trick as run_comparison).
            combined = pd.concat([pos_train, pos_test], ignore_index=True)
            test_uid_index = combined.index[-len(pos_test):]
            for name, series in naive_baselines(combined).items():
                pred = series.loc[test_uid_index]
                pred.index = pos_test.index
                frame = _build_row_level_frame(pos_test, y_test, pred, name, base_meta)
                all_frames.append(frame)
                _append_df_to_csv(frame, output_path)

            # Existing methodology (free — already computed by run_fold).
            frame = _build_row_level_frame(
                pos_test, y_test, existing_pred.reindex(pos_test.index), "existing_methodology", base_meta,
            )
            all_frames.append(frame)
            _append_df_to_csv(frame, output_path)

            # Winning architecture for this position.
            try:
                model = all_arch[cfg["architecture"]]
                model.fit(X_train, y_train, sample_weight=sample_weight)
                pred = pd.Series(model.predict(X_test), index=X_test.index)
                frame = _build_row_level_frame(pos_test, y_test, pred, cfg["architecture"], base_meta)
                all_frames.append(frame)
                _append_df_to_csv(frame, output_path)
            except Exception as e:
                logger.warning("Winning architecture %s failed for %s/%s: %s",
                                cfg["architecture"], position, season, e)

            # Quantile model, for calibration analysis.
            try:
                qmodel = QuantileGBM().fit(X_train, y_train, sample_weight=sample_weight)
                qpred = qmodel.predict(X_test)
                frame = _build_row_level_frame(
                    pos_test, y_test, qpred["p50"], "E_quantile_gbm", base_meta,
                    extra_cols={c: qpred[c] for c in ("p25", "p50", "p75", "p90")},
                )
                all_frames.append(frame)
                _append_df_to_csv(frame, output_path)
            except Exception as e:
                logger.warning("QuantileGBM failed for %s/%s: %s", position, season, e)

    result = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=ROW_LEVEL_COLUMNS)
    print(f"\n{len(result)} row-level predictions appended to {output_path}")
    return result


# ---------------------------------------------------------------------------
# Phase 5: hyperparameter tuning for FINAL_CONFIG architectures
# ---------------------------------------------------------------------------

TUNED_OUTPUT_PATH = Path("data/experiments/phase5_tuned_predictions.csv")
TUNED_PARAMS_OUTPUT_PATH = Path("data/experiments/phase5_tuned_hyperparameters.csv")


def run_tuned_validation(
    positions: Optional[Sequence[str]] = None,
    seasons: Sequence[int] = DEFAULT_VALIDATION_SEASONS,
    n_trials: int = 100,
    output_path: Path = TUNED_OUTPUT_PATH,
    params_output_path: Path = TUNED_PARAMS_OUTPUT_PATH,
) -> pd.DataFrame:
    """Phase 5: nested-CV hyperparameter tuning for each position's
    FINAL_CONFIG architecture. For each outer (position, test_season) fold,
    tunes on an INNER walk-forward split within that fold's training
    seasons only (tuning.tune_hyperparameters_for_fold — never touches the
    outer test season), then refits on the full outer training set with
    the tuned params and evaluates once on the true outer test season.

    Saves row-level predictions in the same schema as Phase 4's CSV (for a
    direct tuned-vs-default comparison) plus the tuned hyperparameters
    themselves for transparency.
    """
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.tuning import tune_hyperparameters_for_fold, _build_model
    from src.models.single_week_ppr.windows import window_to_season_list, compute_recency_weights
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    positions = list(positions) if positions else POSITIONS
    all_frames: List[pd.DataFrame] = []
    param_rows: List[dict] = []

    for position in positions:
        cfg = FINAL_CONFIG[position]
        available_seasons = sorted(
            DatabaseManager().get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )

        for season in seasons:
            print(f"\n=== TUNING {position} / test_season={season} / {cfg} ===")
            train_seasons = window_to_season_list(cfg["window"], season, available_seasons)
            if not train_seasons:
                logger.warning("Skipping %s/%s: no training seasons available", position, season)
                continue
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons,
                )
            except Exception as e:
                logger.warning("Fold %s/%s failed to load: %s", position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position].reset_index(drop=True)
            pos_test = test_df[test_df["position"] == position]
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue

            feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
            if not feature_cols:
                logger.warning("Skipping %s/%s: no CAUSAL_FEATURES columns present", position, season)
                continue

            X_train, y_train, X_test, y_test = _build_feature_matrices(pos_train, pos_test, feature_cols)
            sample_weight = compute_recency_weights(pos_train["season"], cfg["weighting"])
            seasons_train = pos_train["season"].to_numpy()

            try:
                best_params = tune_hyperparameters_for_fold(
                    cfg["architecture"], X_train, y_train, seasons_train,
                    sample_weight=sample_weight, n_trials=n_trials,
                )
            except Exception as e:
                logger.warning("Tuning failed for %s/%s: %s", position, season, e)
                continue

            param_rows.append({"position": position, "season": season, **best_params})

            try:
                model = _build_model(cfg["architecture"], best_params)
                model.fit(X_train, y_train, sample_weight=sample_weight)
                pred = pd.Series(model.predict(X_test), index=X_test.index)
                base_meta = {
                    "position": position, "season": season, "fold": season,
                    "training_window": cfg["window"], "weighting_strategy": cfg["weighting"],
                }
                frame = _build_row_level_frame(pos_test, y_test, pred, cfg["architecture"] + "_tuned", base_meta)
                all_frames.append(frame)
                _append_df_to_csv(frame, output_path)
            except Exception as e:
                logger.warning("Tuned refit failed for %s/%s: %s", position, season, e)

    if param_rows:
        params_df = pd.DataFrame(param_rows)
        params_output_path.parent.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(params_output_path, index=False)
        print(f"\nSaved tuned hyperparameters to {params_output_path}")

    result = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=ROW_LEVEL_COLUMNS)
    print(f"\n{len(result)} tuned row-level predictions appended to {output_path}")
    return result
