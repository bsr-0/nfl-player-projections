import pandas as pd
import pytest

from src.models.single_week_ppr.participation_integration import (
    attach_phase2_oof,
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


def test_paired_bootstrap_detects_better_augmented_prediction():
    rows = pd.DataFrame({
        "actual": [0., 10., 20., 5.],
        "baseline_prediction": [4., 5., 15., 0.],
        "augmented_prediction": [1., 9., 19., 5.],
    })
    result = paired_bootstrap_mae_delta(rows, n_bootstrap=400, seed=1)
    assert result["mae_delta"] < 0
    assert result["ci95_high"] < 0
