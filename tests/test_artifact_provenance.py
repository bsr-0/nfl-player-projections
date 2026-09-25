"""Saved model artifacts must record where they came from.

Until 2026-09-25 they recorded nothing. That is what made the 2026-09-24
incident expensive: a walk-forward fold's models were sitting in
`data/models` and there was no way to distinguish them from a production
retrain, so the state had to be reconstructed from file mtimes and shell
history. `destination_dir` is the field that settles it -- a validation fold
writes into a sandbox temp directory, a real retrain writes into
`data/models` -- so a stamped artifact claiming a temp directory is provably
a fold model no matter where it later ends up.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import FEATURE_VERSION
from src.models.position_models import (
    MultiWeekModel,
    PositionModel,
    _training_provenance,
    read_artifact_provenance,
)
from src.utils.models_dir import redirect_models_dir

EXPECTED_KEYS = {"saved_at", "destination_dir", "feature_version", "git_commit"}


def test_provenance_records_the_four_fields():
    stamp = _training_provenance(Path("/tmp/somewhere/model_qb_1w.joblib"))
    assert set(stamp) == EXPECTED_KEYS
    assert stamp["destination_dir"] == "/tmp/somewhere"
    assert stamp["feature_version"] == str(FEATURE_VERSION).strip()
    assert stamp["saved_at"].endswith("+00:00"), "timestamp must be tz-aware UTC"


def test_destination_dir_distinguishes_a_sandbox_from_production(tmp_path):
    """The whole point: a fold and a retrain are told apart by where they
    wrote, which survives in the artifact even if the file is moved.
    """
    fold = _training_provenance(tmp_path / "fold" / "model_qb_1w.joblib")
    real = _training_provenance(Path("/repo/data/models/model_qb_1w.joblib"))
    assert fold["destination_dir"] != real["destination_dir"]
    assert real["destination_dir"].endswith("data/models")


def test_git_commit_is_present_or_explicitly_none():
    """Recorded honestly rather than omitted, so "not collected" and "not in
    a checkout" are distinguishable.
    """
    stamp = _training_provenance(Path("x/y.joblib"))
    assert "git_commit" in stamp
    assert stamp["git_commit"] is None or isinstance(stamp["git_commit"], str)


def _fitted_position_model():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2 * X["a"] + rng.normal(scale=0.1, size=n) + 10)
    model = PositionModel(position="QB", n_weeks=1)
    model.fit(X, y, tune_hyperparameters=False)
    return model


def test_position_model_save_stamps_provenance(tmp_path):
    model = _fitted_position_model()
    path = tmp_path / "model_qb_1w.joblib"
    model.save(path)

    stamp = read_artifact_provenance(path)
    assert stamp is not None and set(stamp) == EXPECTED_KEYS
    assert stamp["destination_dir"] == str(tmp_path)


def test_stamping_does_not_break_round_trip_load(tmp_path):
    """The extra key must not disturb the load path."""
    model = _fitted_position_model()
    path = tmp_path / "model_qb_1w.joblib"
    model.save(path)

    loaded = PositionModel.load("QB", 1, filepath=path)
    assert loaded.position == "QB" and loaded.n_weeks == 1
    assert loaded.feature_names == model.feature_names


def test_unstamped_legacy_artifact_reads_as_none(tmp_path):
    """Artifacts written before stamping existed must not raise."""
    path = tmp_path / "legacy.joblib"
    joblib.dump({"position": "QB", "n_weeks": 1, "models": {}}, path)
    assert read_artifact_provenance(path) is None


def test_a_sandboxed_save_is_identifiable_afterwards(tmp_path):
    """End-to-end: save through a redirected MODELS_DIR (what a fold does),
    then confirm the artifact admits it came from the sandbox.
    """
    model = _fitted_position_model()
    sandbox = tmp_path / "fold"
    with redirect_models_dir(sandbox):
        import src.models.position_models as pm
        model.save(pm.MODELS_DIR / "model_qb_1w.joblib")

    stamp = read_artifact_provenance(sandbox / "model_qb_1w.joblib")
    assert stamp["destination_dir"] == str(sandbox)
    assert "data/models" not in stamp["destination_dir"]


def test_multiweek_save_is_stamped_too(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    multi = MultiWeekModel(position="QB")
    multi.fit(X, {1: pd.Series(2 * X["a"] + 10)}, tune_hyperparameters=False)
    path = tmp_path / "multiweek_qb.joblib"
    multi.save(path)

    stamp = read_artifact_provenance(path)
    assert stamp is not None and stamp["destination_dir"] == str(tmp_path)
    # The pre-existing version_metadata block is preserved alongside it.
    assert "version_metadata" in joblib.load(path)
