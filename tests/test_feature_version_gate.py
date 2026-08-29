"""The feature-version gate must refuse, not warn.

Until 2026-08-29 a mismatch printed a warning and served predictions anyway.
Measured cost: artifacts trained at FEATURE_VERSION 33 were fed v35 features
and `predict()` returned a median of 55.00 for WR week predictions against
actuals averaging 6.12 (MAE 43.24, vs 4.07 for the same model on a proper
evaluation). The output saturated at a cap rather than predicting, and nothing
failed loudly. It went unnoticed because predict_next_week() has no consumer.
"""
import os
import pytest

import config.settings as settings
import src.models.ensemble as ens
from src.models.ensemble import FeatureVersionMismatch, _warn_if_feature_version_mismatch


@pytest.fixture
def version_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ens, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("NFL_ALLOW_FEATURE_VERSION_MISMATCH", raising=False)

    def write(v):
        (tmp_path / settings.FEATURE_VERSION_FILENAME).write_text(str(v))
    return write


def test_mismatch_raises(version_file, monkeypatch):
    monkeypatch.setattr(ens, "FEATURE_VERSION", "35")
    version_file("33")
    with pytest.raises(FeatureVersionMismatch, match="33"):
        _warn_if_feature_version_mismatch()


def test_match_passes(version_file, monkeypatch):
    monkeypatch.setattr(ens, "FEATURE_VERSION", "35")
    version_file("35")
    _warn_if_feature_version_mismatch()


def test_missing_version_file_raises(tmp_path, monkeypatch):
    """No version file means the artifacts cannot be matched to the feature
    set at all -- strictly less information than a mismatch, so it cannot be
    the more permissive case."""
    monkeypatch.setattr(ens, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("NFL_ALLOW_FEATURE_VERSION_MISMATCH", raising=False)
    with pytest.raises(FeatureVersionMismatch):
        _warn_if_feature_version_mismatch()


def test_override_downgrades_to_warning(version_file, monkeypatch, capsys):
    """The escape hatch exists for inspecting old artifacts. It must be
    explicit and must say so loudly."""
    monkeypatch.setattr(ens, "FEATURE_VERSION", "35")
    version_file("33")
    monkeypatch.setenv("NFL_ALLOW_FEATURE_VERSION_MISMATCH", "1")
    _warn_if_feature_version_mismatch()
    assert "WARNING (override set)" in capsys.readouterr().out
