"""Serving must apply the bounded scaler exactly as training applied it.

Training fits a MinMax scaler on the bounded columns of the train frame and
transforms its test frame with it (`_apply_bounded_scaling`), reconciling
absent `*_missing` indicators as 0 and preserving NaN. Serving loads that
artifact and used to (a) zero-fill NaN before scaling and (b) silently skip
the whole transform on any column mismatch -- which served week-1
predictions 5-7x too high on 2026-09-11 because 18 indicator columns are
absent from an as_of frame with no current-season rows.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.feature_preparation import (
    BoundedScalerMismatch,
    _apply_bounded_scaling,
    apply_bounded_scaler_artifact,
)


def _frames():
    rng = np.random.RandomState(0)
    train = pd.DataFrame({
        "snap_share_pct_roll3_mean": rng.uniform(0, 100, 60),
        "targets_roll3_mean": rng.uniform(0, 15, 60),
        "targets_roll3_mean_missing": (rng.uniform(size=60) < 0.1).astype(int),
        "player_id": [f"P{i}" for i in range(60)],
    })
    test = pd.DataFrame({
        "snap_share_pct_roll3_mean": [10.0, np.nan, 250.0, np.inf],
        "targets_roll3_mean": [3.0, 4.0, np.nan, 1.0],
        "player_id": ["A", "B", "C", "D"],
    })                                       # no *_missing column: indicator not emitted
    return train, test


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    train, test = _frames()
    monkeypatch.setattr("src.models.feature_preparation._infer_bounded_columns",
                        lambda df: ["snap_share_pct_roll3_mean", "targets_roll3_mean",
                                    "targets_roll3_mean_missing"])
    art = _apply_bounded_scaling(train, test, tmp_path / "scaler.joblib")
    return art, test          # `test` is now the training-time transformed test frame


def test_serving_transform_matches_training_transform(artifact):
    art, train_side = artifact
    _, raw_test = _frames()
    serve_side = apply_bounded_scaler_artifact(raw_test.copy(), art)

    for col in art["columns"]:
        pd.testing.assert_series_equal(serve_side[col], train_side[col], check_names=False)
    # NaN survives scaling on both sides (PositionModel.predict median-fills later, identically).
    assert serve_side["snap_share_pct_roll3_mean"].isna().tolist() == [False, True, False, True]
    assert serve_side["targets_roll3_mean_missing"].tolist() == [0, 0, 0, 0]


def test_values_are_scaled_not_raw(artifact):
    art, _ = artifact
    _, raw_test = _frames()
    out = apply_bounded_scaler_artifact(raw_test.copy(), art)
    assert out.loc[0, "snap_share_pct_roll3_mean"] < 1.0          # 10 of ~100 -> ~0.1, not 10
    assert out.loc[2, "snap_share_pct_roll3_mean"] > 1.0          # 250 exceeds the train range: MinMax extrapolates


def test_absent_real_column_raises_instead_of_skipping(artifact):
    art, _ = artifact
    _, raw_test = _frames()
    with pytest.raises(BoundedScalerMismatch, match="targets_roll3_mean"):
        apply_bounded_scaler_artifact(raw_test.drop(columns=["targets_roll3_mean"]), art)


def test_empty_artifact_is_a_noop():
    df = pd.DataFrame({"a": [1.0]})
    assert apply_bounded_scaler_artifact(df, {"columns": [], "scaler": None}) is df
