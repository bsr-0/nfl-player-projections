"""Phase 2 (next_focus.md) walk-forward comparison: naive baselines vs. six
new architectures vs. the existing production methodology, on raw
`fantasy_points`.

Reuses the existing leakage-safe feature pipeline (`_prepare_training_data`
in src/models/feature_preparation.py) rather than re-implementing feature
engineering — see the Phase 2 plan for why. Must not modify any production
file or overwrite production model artifacts. `settings.MODELS_DIR` is
redirected to a temp directory per fold (mirroring src/models/train.py's own
walk-forward loop at train.py:1053-1061), but that redirection alone is NOT
sufficient: some modules (e.g. src/models/utilization_to_fp.py:16) do
`from config.settings import MODELS_DIR` at import time, so reassigning
`settings.MODELS_DIR` later doesn't affect their already-bound writes.
`_protect_models_dir()` below is the real safety net — it snapshots
data/models/ before the fold and restores any touched files afterward via
git (tracked files) or deletion (newly-created untracked files). See
GAPS.md Phase 2 notes for the incident that made this necessary.
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


def run_fold(position: str, test_season: int, tune_hyperparameters: bool = False, n_trials: int = 0):
    """Loads one (train_seasons, test_season) fold, feature-engineers it via
    the existing leakage-safe pipeline, and extracts the existing-methodology
    prediction for `position` for later comparison.

    Redirects config.settings.MODELS_DIR to a temp dir for the duration of
    the call, AND wraps the whole fold (including load_training_data, which
    also writes under data/) in _protect_data_dir() as the real safety net
    (the redirection alone is not reliable — see module docstring).

    Returns (train_df, test_df, existing_methodology_pred, train_seasons).
    """
    import config.settings as settings
    from src.models.data_loading import load_training_data
    from src.models.feature_preparation import _prepare_training_data

    old_models_dir = settings.MODELS_DIR
    with _protect_data_dir(), tempfile.TemporaryDirectory() as tmp:
        settings.MODELS_DIR = Path(tmp)
        try:
            train_data, test_data, train_seasons, _ = load_training_data(
                [position], test_season=test_season, optimize_training_years=False,
            )
            if len(test_data) < 20:
                raise ValueError(f"Not enough test rows for {position} season {test_season}: {len(test_data)}")

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

            X_train = pos_train[feature_cols].fillna(0)
            y_train = pos_train["fantasy_points"]
            X_test = pos_test[feature_cols].fillna(0)
            y_test = pos_test["fantasy_points"]

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
