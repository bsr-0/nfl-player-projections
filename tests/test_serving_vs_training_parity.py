"""Serving-path feature prep must match training's test-side prep.

Same rows (an as_of frame), two preps: NFLPredictor._prepare_features and a
replica of the sequence _prepare_training_data applies to its held-out
frame using the persisted artifacts. Every model feature should agree, and
the persisted ensemble should produce the same predictions from both.

This is the check that would have caught the 5x week-1 blow-up (scaler
silently skipped) and any future stage that drifts between the two paths.
It needs the local DB and trained models (~3 min).
"""
import json

import numpy as np
import pandas as pd
import pytest

from config.settings import DB_PATH, MODELS_DIR

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists() or not (MODELS_DIR / "multiweek_wr.joblib").exists(),
    reason="needs the local database and trained models")

AS_OF = (2025, 10)


@pytest.fixture(scope="module")
def preps():
    import warnings
    warnings.filterwarnings("ignore")
    from src.predict import NFLPredictor
    from src.features.utilization_score import (
        calculate_utilization_scores, recalculate_utilization_with_weights,
        load_percentile_bounds, load_snap_imputation, apply_snap_imputation, SNAP_IMPUTATION_FILENAME)
    from src.models.feature_preparation import (
        add_engineered_features, add_advanced_features, apply_bounded_scaler_artifact)
    from src.features.college_conference import add_conference_features
    from src.data.external_data import add_external_features
    from src.features.season_long_features import add_season_long_features

    p = NFLPredictor()
    assert p.initialize()
    base = p._load_player_data(None, min_games=1)
    s, w = pd.to_numeric(base["season"]), pd.to_numeric(base["week"])
    base = base[(s < AS_OF[0]) | ((s == AS_OF[0]) & (w < AS_OF[1]))].copy()

    serving = p._prepare_features(base.copy())

    t = add_conference_features(base.copy())
    t = add_external_features(t, seasons=sorted(t["season"].dropna().astype(int).unique()))
    t = add_season_long_features(t)
    bounds = load_percentile_bounds(MODELS_DIR / "utilization_percentile_bounds.json")
    t = calculate_utilization_scores(t, team_df=pd.DataFrame(), weights=None, percentile_bounds=bounds)
    t = recalculate_utilization_with_weights(t, json.load(open(MODELS_DIR / "utilization_weights.json")))
    t = add_advanced_features(add_engineered_features(t))
    t = apply_snap_imputation(t, load_snap_imputation(MODELS_DIR / SNAP_IMPUTATION_FILENAME))
    training_like = apply_bounded_scaler_artifact(t, p.bounded_scaler_artifact)

    def latest(df):
        return df.sort_values(["player_id", "season", "week"]).groupby("player_id").last()
    a, b = latest(serving), latest(training_like)
    common = sorted(set(a.index) & set(b.index))
    return p, a.loc[common], b.loc[common]


def _model_features(p):
    feats = set()
    for mw in p.predictor.position_models.values():
        feats |= set(getattr(mw.models.get(1), "feature_names", []))
    return feats


def test_model_features_agree_between_paths(preps):
    p, a, b = preps
    feats = [f for f in _model_features(p) if f in a.columns and f in b.columns]
    assert len(feats) > 300
    disagreeing = []
    for f in feats:
        x, y = pd.to_numeric(a[f], errors="coerce"), pd.to_numeric(b[f], errors="coerce")
        both = x.notna() & y.notna()
        nan_mismatch = int((x.isna() != y.isna()).sum())
        scale = max(float(y[both].abs().mean()), 1e-9) if both.any() else 1.0
        rel = float((x[both] - y[both]).abs().mean()) / scale if both.any() else 0.0
        if rel > 1e-3 or nan_mismatch > 0.01 * len(x):
            disagreeing.append((f, round(rel, 4), nan_mismatch))
    assert len(disagreeing) <= 0.03 * len(feats), disagreeing[:15]


def test_scaler_columns_are_scaled_on_the_serving_path(preps):
    p, a, _ = preps
    cols = [c for c in p.bounded_scaler_artifact["columns"] if c in a.columns]
    vals = a[cols].replace([np.inf, -np.inf], np.nan)
    frac_in_unit = float(((vals >= -0.01) & (vals <= 1.01)).mean().mean())
    assert frac_in_unit > 0.9, f"only {frac_in_unit:.1%} of scaler-column values in [0,1]: scaler not applied"


def test_persisted_ensemble_predicts_the_same_from_both_preps(preps):
    p, a, b = preps
    a = a.reset_index(); b = b.reset_index()
    for L in (a, b):
        L["season"], L["week"] = AS_OF
    pa = p.predictor.predict(a, n_weeks=1).set_index("player_id")["predicted_points"]
    pb = p.predictor.predict(b, n_weeks=1).set_index("player_id")["predicted_points"]
    common = pa.index.intersection(pb.index)
    assert len(common) > 500
    assert abs(pa[common].mean() - pb[common].mean()) < 0.02 * pb[common].mean()
    assert np.corrcoef(pa[common], pb[common])[0, 1] > 0.99
