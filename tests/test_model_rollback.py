"""Rollback must archive weights, not just claim to.

`model_metadata.json` reported `rollback_available: true` whenever a previous
training *summary* existed. Nothing archived weights, so the flag was true
for years with no recoverable model behind it -- and on 2026-09-24 a
walk-forward run overwrote all four positions' served artifacts with nothing
to restore from (GAPS.md, that date).
"""
import shutil

import pytest

from src.utils.model_rollback import (
    BOOKKEEPING_FILES,
    ROLLBACK_DIRNAME,
    available_rollbacks,
    restore_models,
    snapshot_models,
)


def _served(models_dir, *, marker="v1"):
    models_dir.mkdir(parents=True, exist_ok=True)
    for position in ("qb", "rb", "wr", "te"):
        (models_dir / f"model_{position}_1w.joblib").write_text(f"{marker}-{position}-1w")
        (models_dir / f"multiweek_{position}.joblib").write_text(f"{marker}-{position}-multi")
    (models_dir / "model_metadata.json").write_text(f'{{"training_date": "{marker}"}}')
    (models_dir / "feature_version.txt").write_text("36")
    return models_dir


def test_snapshot_captures_served_artifacts_and_bookkeeping(tmp_path):
    models = _served(tmp_path / "models")
    snapshot = snapshot_models(models)

    names = {f.name for f in snapshot.iterdir()}
    assert len([n for n in names if n.endswith(".joblib")]) == 8
    for bookkeeping in BOOKKEEPING_FILES:
        assert bookkeeping in names, "weights without their metadata recreate the mismatch"


def test_snapshot_is_taken_before_overwrite_and_survives_it(tmp_path):
    """The property that was missing: after a run replaces the artifacts,
    the previous ones are still recoverable.
    """
    models = _served(tmp_path / "models", marker="old")
    snapshot_models(models)
    _served(models, marker="new")  # a training run overwrites everything

    assert (models / "model_qb_1w.joblib").read_text() == "new-qb-1w"
    restore_models(available_rollbacks(models)[-1], models)
    assert (models / "model_qb_1w.joblib").read_text() == "old-qb-1w"
    assert (models / "multiweek_te.joblib").read_text() == "old-te-multi"


def test_first_ever_run_has_nothing_to_archive(tmp_path):
    empty = tmp_path / "models"
    empty.mkdir()
    assert snapshot_models(empty) is None
    assert available_rollbacks(empty) == []


def test_retention_prunes_oldest_first(tmp_path):
    models = _served(tmp_path / "models")
    made = []
    for i in range(4):
        _served(models, marker=f"v{i}")
        snap = snapshot_models(models, keep=2)
        # Names are second-resolution; force distinct dirs for the test.
        renamed = snap.parent / f"2026010{i}T000000Z"
        snap.rename(renamed)
        made.append(renamed)

    kept = available_rollbacks(models)
    assert len(kept) <= 3  # 2 retained + the just-renamed one
    assert kept[-1].name == made[-1].name


def test_partial_snapshot_is_not_listed_as_restorable(tmp_path):
    """An interrupted copy must never look like a usable snapshot."""
    models = _served(tmp_path / "models")
    snapshot_models(models)
    (models / ROLLBACK_DIRNAME / "20260101T000000Z.partial").mkdir()

    listed = available_rollbacks(models)
    assert all(not d.name.endswith(".partial") for d in listed)
    assert len(listed) == 1


def test_failed_snapshot_leaves_no_staging_directory(tmp_path, monkeypatch):
    models = _served(tmp_path / "models")

    def _boom(src, dst, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", _boom)
    with pytest.raises(OSError):
        snapshot_models(models)

    root = models / ROLLBACK_DIRNAME
    assert not any(root.iterdir()), "staging dir must be cleaned up on failure"


def test_restore_rejects_a_missing_snapshot(tmp_path):
    models = _served(tmp_path / "models")
    with pytest.raises(FileNotFoundError):
        restore_models(models / ROLLBACK_DIRNAME / "nope", models)


def test_restore_rejects_an_empty_snapshot(tmp_path):
    models = _served(tmp_path / "models")
    empty = models / ROLLBACK_DIRNAME / "20260101T000000Z"
    empty.mkdir(parents=True)
    with pytest.raises(ValueError):
        restore_models(empty, models)


def test_snapshots_do_not_shadow_served_artifacts(tmp_path):
    """The rollback dir lives inside data/models; its copies must not be
    picked up as served artifacts by the next snapshot.
    """
    models = _served(tmp_path / "models")
    snapshot_models(models)
    second = snapshot_models(models)
    assert len([f for f in second.iterdir() if f.name.endswith(".joblib")]) == 8
