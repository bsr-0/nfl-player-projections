"""Tests for reconstruction-aware Plan A candidate helpers."""
import numpy as np
import pandas as pd

from src.evaluation.team_reconstruction_candidates import _best_volume_alpha, _safe_volume_weights, _team_feature_columns


def test_volume_weights_are_finite_and_neutral_for_missing_totals():
    weights = _safe_volume_weights(np.array([200.0, 400.0, np.nan, 0.0]), 1.0)
    assert np.isfinite(weights).all()
    assert weights[2] == 1.0
    assert weights[3] == 1.0
    assert weights[1] > weights[0]


def test_volume_alpha_prefers_model_when_it_improves_reconstructed_volume():
    team_total = np.full(40, 400.0)
    baseline = np.full(40, 0.10)
    model = np.full(40, 0.20)
    actual = np.full(40, 0.20)
    assert _best_volume_alpha(actual * team_total, model, baseline, team_total) == 1.0


def test_team_feature_columns_exclude_current_week_totals():
    df = pd.DataFrame(columns=[
        "team_targets", "team_targets_roll3", "team_rushing_yards",
        "team_rushing_yards_roll3", "share_of_team_targets_roll3",
    ])
    cols = _team_feature_columns(df, "rushing_yards")
    assert "team_targets" not in cols
    assert "team_rushing_yards" not in cols
    assert "team_targets_roll3" in cols
    assert "team_rushing_yards_roll3" in cols
