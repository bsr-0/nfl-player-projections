import pandas as pd
import pytest

from src.models.single_week_ppr.participation_integration import (
    _feature_matrix,
    attach_phase2_oof,
    deduplicate_ppr_player_weeks,
    paired_bootstrap_mae_delta,
    validate_phase2_oof,
)


def _oof():
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p1"], "season": [2023, 2023, 2024], "week": [1, 1, 1],
        "model": ["hist_gbm", "hist_gbm", "logistic"],
        "participation_probability": [.8, .2, .9], "expected_snap_share": [.5, .1, .6],
        "phase2_test_season": [2023, 2023, 2024], "phase2_train_max_season": [2022, 2022, 2023],
    })


def test_oof_selects_one_model_and_preserves_provenance():
    out = validate_phase2_oof(_oof())
    assert len(out) == 2
    assert (out.phase2_train_max_season < out.season).all()


def test_oof_rejects_same_season_training():
    oof = _oof()
    oof.loc[0, "phase2_train_max_season"] = 2023
    with pytest.raises(ValueError, match="own/future"):
        validate_phase2_oof(oof)


def test_oof_rejects_duplicate_selected_model_keys():
    oof = pd.concat([_oof(), _oof().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_phase2_oof(oof)


def test_attach_uses_only_identical_matched_population():
    ppr = pd.DataFrame({
        "player_id": ["p1", "p2", "missing"], "season": [2023, 2023, 2023], "week": [1, 1, 1],
        "fantasy_points": [10., 2., 7.],
    })
    out = attach_phase2_oof(ppr, validate_phase2_oof(_oof()))
    assert out.player_id.tolist() == ["p1", "p2"]


def test_attach_deduplicates_identical_target_player_week_using_most_complete_row():
    ppr = pd.DataFrame({
        "player_id": ["p1", "p1"], "season": [2023, 2023], "week": [1, 1],
        "fantasy_points": [10., 10.], "sparse_feature": [None, 1.],
    })
    out = attach_phase2_oof(ppr, validate_phase2_oof(_oof()))
    assert len(out) == 1
    assert out.sparse_feature.iloc[0] == 1.


def test_duplicate_ppr_player_week_with_conflicting_target_fails_loudly():
    ppr = pd.DataFrame({
        "player_id": ["p1", "p1"], "season": [2023, 2023], "week": [1, 1],
        "fantasy_points": [10., 11.],
    })
    with pytest.raises(ValueError, match="disagree"):
        deduplicate_ppr_player_weeks(ppr)


def test_feature_matrix_keeps_rows_with_production_handled_missing_features():
    frame = pd.DataFrame({
        "fantasy_points": [10., 5.], "causal_feature": [None, 1.],
    })
    X, y = _feature_matrix(frame, ["causal_feature"])
    assert len(X) == len(y) == 2
    assert X.causal_feature.isna().sum() == 1


def test_paired_bootstrap_detects_better_augmented_prediction():
    rows = pd.DataFrame({
        "player_id": ["a", "a", "b", "b"],
        "actual": [0., 10., 20., 5.],
        "baseline_prediction": [4., 5., 15., 0.],
        "augmented_prediction": [1., 9., 19., 5.],
    })
    result = paired_bootstrap_mae_delta(rows, n_bootstrap=400, seed=1)
    assert result["mae_delta"] < 0
    assert result["ci95_high"] < 0


def test_feature_only_fold_does_not_fit_or_predict_legacy_models(monkeypatch):
    from contextlib import nullcontext
    from src.models.single_week_ppr import evaluate as folds
    from src.models import feature_preparation as preparation
    from src.utils.database import DatabaseManager
    data = pd.DataFrame({
        'player_id': [f'p{i}' for i in range(30)] * 3,
        'season': [2021]*30 + [2022]*30 + [2023]*30,
        'week': [1]*90, 'position': ['QB']*90,
    })
    seen = {}

    def prepare(train, test, positions, tune, trials, fast=False, context_data=None,
                fit_models=True):
        seen['fit_models'] = fit_models
        seen['context_seasons'] = set(context_data.season)
        return train.assign(prepared=True), test.assign(prepared=True), None

    def unexpected(*args, **kwargs):
        raise AssertionError('feature-only fold tried to predict legacy models')

    monkeypatch.setattr(DatabaseManager, 'get_all_players_for_training', lambda *a, **k: data)
    monkeypatch.setattr(folds, '_protect_data_dir', nullcontext)
    monkeypatch.setattr(preparation, '_prepare_training_data', prepare)
    monkeypatch.setattr(folds, '_existing_methodology_predictions', unexpected)
    train, test, predictions, seasons = folds.run_fold(
        'QB', 2023, train_seasons_override=[2022], fit_existing_models=False)
    assert seen == {'fit_models': False, 'context_seasons': {2021}}
    assert train.prepared.all() and test.prepared.all()
    assert predictions is None
    assert seasons == [2022]
    assert set(train.season) == {2022} and set(test.season) == {2023}
