"""Validation folds must not be able to reach the real model artifacts.

The bug this pins (found 2026-09-24): `train.py`'s walk-forward loop, the
LOYO backtest, and single_week_ppr's fold runner all tried to sandbox their
throwaway folds with `settings.MODELS_DIR = Path(tmp)`. Nineteen modules
bind the directory with `from config.settings import MODELS_DIR`, which
copies the value at import time, so that assignment reached none of them.
`position_models.save()` writes `MODELS_DIR / model_{pos}_{n}w.joblib` --
exactly what `EnsemblePredictor.load_models()` serves -- so every fold
silently replaced the production models with fold models. A walk-forward
run overwrote all four positions' served artifacts that day.
"""
import tempfile
from pathlib import Path

import config.settings as settings
from src.utils.models_dir import redirect_models_dir


def test_plain_settings_assignment_does_not_reach_imported_bindings():
    """Pins the defect itself, so nobody 'simplifies' the helper away.

    If this ever starts failing, `from config.settings import MODELS_DIR`
    has stopped copying the value and the helper may be unnecessary --
    but until then, the naive assignment is not a sandbox.
    """
    import src.models.position_models as pm

    original = pm.MODELS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        settings.MODELS_DIR = Path(tmp)
        try:
            assert pm.MODELS_DIR == original, (
                "position_models now tracks settings.MODELS_DIR; re-evaluate "
                "whether redirect_models_dir is still needed")
        finally:
            settings.MODELS_DIR = original


def test_redirect_reaches_modules_that_imported_by_value():
    import src.models.position_models as pm

    original = pm.MODELS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        with redirect_models_dir(tmp):
            assert pm.MODELS_DIR == Path(tmp)
            assert settings.MODELS_DIR == Path(tmp)
        assert pm.MODELS_DIR == original
        assert settings.MODELS_DIR == original


def test_the_actual_save_path_lands_in_the_sandbox(tmp_path):
    """The end-to-end property that matters: the filename expression
    `position_models` uses for the served artifact must resolve inside the
    temp directory, not into data/models.
    """
    import src.models.position_models as pm

    real = pm.MODELS_DIR
    sandbox = tmp_path / "fold"
    with redirect_models_dir(sandbox):
        # Same expression as position_models.save() (lines ~1450 / ~1665).
        single = pm.MODELS_DIR / "model_qb_1w.joblib"
        multi = pm.MODELS_DIR / "multiweek_qb.joblib"
        # Assert containment BEFORE writing. Writing first and checking
        # afterwards is how this test destroyed the real QB artifacts while
        # it was being developed: with the redirect broken, these paths are
        # the live ones, and the write lands on production.
        assert single.parent == sandbox and multi.parent == sandbox
        assert real not in single.parents and real not in multi.parents
        single.write_bytes(b"fold-artifact")
        multi.write_bytes(b"fold-artifact")
        assert single.read_bytes() == b"fold-artifact"
    assert pm.MODELS_DIR == real


def test_bindings_are_restored_after_an_exception():
    import src.models.position_models as pm

    original = pm.MODELS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with redirect_models_dir(tmp):
                raise RuntimeError("fold blew up")
        except RuntimeError:
            pass
    assert pm.MODELS_DIR == original
    assert settings.MODELS_DIR == original


def test_module_imported_inside_the_block_is_restored_too(tmp_path):
    """A module first imported *during* a fold binds the sandbox path. If it
    isn't restored on exit it keeps writing to a deleted temp directory.
    """
    import sys

    name = "src.models.bayesian_models"
    sys.modules.pop(name, None)
    with redirect_models_dir(tmp_path / "fold"):
        module = __import__(name, fromlist=["MODELS_DIR"])
        assert module.MODELS_DIR == tmp_path / "fold"
    assert module.MODELS_DIR == settings.MODELS_DIR
    assert "fold" not in str(module.MODELS_DIR)


def test_nested_redirects_restore_in_order(tmp_path):
    import src.models.position_models as pm

    original = pm.MODELS_DIR
    with redirect_models_dir(tmp_path / "outer"):
        assert pm.MODELS_DIR == tmp_path / "outer"
        with redirect_models_dir(tmp_path / "inner"):
            assert pm.MODELS_DIR == tmp_path / "inner"
        assert pm.MODELS_DIR == tmp_path / "outer"
    assert pm.MODELS_DIR == original


def test_target_directory_is_created(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    with redirect_models_dir(target):
        assert target.is_dir()
