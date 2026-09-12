"""The injury-risk model: fit, persist, load, predict, and how
add_advanced_injury_features uses it (model-first, heuristic as a warned
fallback only).

Replaces both a heuristic that measurably overpredicted risk ~5.5x (Brier
0.057 vs 0.033 for "always predict the base rate") and a class_weight=
'balanced' RandomForestClassifier that was coded but never actually wired to
run in production, and would have been far worse had it been (raw
predict_proba averaged 48.5% against ~3.4% true incidence, Brier 0.267).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.features.advanced_rookie_injury import (
    AdvancedInjuryPredictor,
    INJURY_RISK_NUMERIC_FEATURES,
    _build_injury_risk_matrix,
    _load_injury_risk_model,
    fit_injury_risk_model,
    predict_injury_risk,
)


def _synthetic_training_frame(n=800, seed=0):
    rng = np.random.RandomState(seed)
    age = rng.randint(21, 34, n)
    week = rng.randint(1, 18, n)
    weekly_workload = rng.uniform(0, 30, n)
    season_workload = weekly_workload * rng.uniform(1, 10, n)
    prior_injuries = rng.poisson(0.5, n)
    position = rng.choice(["QB", "RB", "WR", "TE"], n)
    # True risk rises with age, workload, and prior injuries -- give the
    # model something real to recover, at a realistic ~3-5% prevalence.
    logit = (-4.0 + 0.05 * (age - 27) + 0.03 * weekly_workload + 0.3 * prior_injuries)
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({
        "age": age, "week": week, "weekly_workload": weekly_workload,
        "season_workload": season_workload, "prior_injuries": prior_injuries,
        "position": position, "is_injured_next_week": y,
    })


@pytest.fixture
def artifact(tmp_path):
    df = _synthetic_training_frame()
    return fit_injury_risk_model(df, path=tmp_path / "injury_risk_model.joblib")


def test_fit_persists_a_loadable_artifact(tmp_path):
    df = _synthetic_training_frame()
    path = tmp_path / "injury_risk_model.joblib"
    fit_injury_risk_model(df, path=path)
    assert path.exists()
    loaded = _load_injury_risk_model(path=path)
    assert loaded is not None
    assert set(loaded["numeric_features"]) == set(INJURY_RISK_NUMERIC_FEATURES)
    assert set(loaded["position_categories"]) == {"QB", "RB", "WR", "TE"}


def test_missing_artifact_path_returns_none(tmp_path):
    assert _load_injury_risk_model(path=tmp_path / "does_not_exist.joblib") is None


def test_predictions_are_on_a_realistic_probability_scale(artifact):
    df = _synthetic_training_frame(n=300, seed=1)
    p = predict_injury_risk(df, artifact)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # A rare weekly event: no well-specified model should average anywhere
    # near the ~19%/~48% this replaced.
    assert p.mean() < 0.15


def test_higher_workload_and_prior_injuries_predict_higher_risk(artifact):
    """Sanity on direction, not just scale: the model should have actually
    learned the real relationship, not just landed on a low constant."""
    base = pd.DataFrame({
        "age": [27], "week": [8], "weekly_workload": [5.0],
        "season_workload": [40.0], "prior_injuries": [0], "position": ["RB"],
    })
    high_risk = base.assign(weekly_workload=28.0, season_workload=200.0, prior_injuries=3)
    assert predict_injury_risk(high_risk, artifact)[0] > predict_injury_risk(base, artifact)[0]


def test_unknown_position_does_not_raise(artifact):
    df = pd.DataFrame({
        "age": [27], "week": [8], "weekly_workload": [10.0],
        "season_workload": [50.0], "prior_injuries": [0], "position": ["K"],
    })
    p = predict_injury_risk(df, artifact)
    assert np.isfinite(p[0])


def test_feature_matrix_is_identical_between_fit_and_predict_paths(artifact):
    """The exact bug class this session kept finding: two call sites building
    'the same' feature matrix slightly differently. Pin it directly."""
    df = _synthetic_training_frame(n=5, seed=2)
    a = _build_injury_risk_matrix(df, artifact["numeric_features"], artifact["position_categories"])
    b = _build_injury_risk_matrix(df, artifact["numeric_features"], artifact["position_categories"])
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == artifact["feature_names"]


class TestAddAdvancedInjuryFeaturesUsesTheModel:
    def _base_df(self):
        return pd.DataFrame({
            "player_id": ["P1", "P1", "P2"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "position": ["RB", "RB", "WR"],
            "age": [24, 24, 29],
            "rushing_attempts": [10, 12, 0],
            "targets": [2, 3, 8],
            "is_injured": [0, 0, 0],
        })

    def test_uses_persisted_model_when_available(self, tmp_path, monkeypatch):
        import src.features.advanced_rookie_injury as m
        monkeypatch.setattr(m, "MODELS_DIR", tmp_path)
        fit_injury_risk_model(_synthetic_training_frame(), path=tmp_path / m.INJURY_RISK_MODEL_FILENAME)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = AdvancedInjuryPredictor().add_advanced_injury_features(self._base_df())
        assert not any("falling back to the unvalidated heuristic" in str(x.message) for x in w)
        assert out["injury_prob_combined"].equals(out["injury_prob_advanced"])
        assert out["injury_prob_ml"].isna().all()
        assert out["injury_prob_advanced"].between(0, 1).all()

    def test_warns_and_falls_back_when_no_model_is_persisted(self, tmp_path, monkeypatch):
        import src.features.advanced_rookie_injury as m
        monkeypatch.setattr(m, "MODELS_DIR", tmp_path)   # empty dir, no artifact

        with pytest.warns(RuntimeWarning, match="falling back to the unvalidated heuristic"):
            out = AdvancedInjuryPredictor().add_advanced_injury_features(self._base_df())
        assert out["injury_prob_advanced"].between(0, 1).all()
