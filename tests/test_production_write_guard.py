"""Only a full production retrain may rewrite the served artifacts.

The bounded scaler is one MinMaxScaler fit jointly on every training row, and
utilization weights / percentile bounds are per position but share one file
each. A QB-only "restore" retrain on 2026-09-28 therefore left RB/WR/TE
serving on a QB-fit scaler, default utilization weights and no percentile
bounds, while their own model files were untouched (GAPS.md, that date).
"""
import pandas as pd
import pytest

import config.settings as settings
import src.models.feature_preparation as fp
import src.models.train as train_module
from src.utils.models_dir import (
    ProductionArtifactWriteError,
    assert_safe_models_dir_write,
    redirect_models_dir,
)

ALL = ["QB", "RB", "WR", "TE"]


def _frame():
    return pd.DataFrame([{"player_id": "P", "season": 2024, "week": 1,
                          "team": "AAA", "position": "QB"}])


def test_production_constant_survives_a_redirect(tmp_path):
    production = settings.PRODUCTION_MODELS_DIR
    with redirect_models_dir(tmp_path):
        assert settings.MODELS_DIR == tmp_path
        assert settings.PRODUCTION_MODELS_DIR == production


@pytest.mark.parametrize("kwargs", [
    dict(positions=["QB"], fit_models=True, production_run=True),
    dict(positions=ALL, fit_models=False, production_run=True),
    dict(positions=ALL, fit_models=True, production_run=False),
])
def test_anything_but_a_full_production_retrain_is_refused(kwargs):
    positions = kwargs.pop("positions")
    with pytest.raises(ProductionArtifactWriteError):
        assert_safe_models_dir_write(settings.PRODUCTION_MODELS_DIR, positions, **kwargs)


def test_full_production_retrain_is_allowed_in_any_order():
    assert_safe_models_dir_write(settings.PRODUCTION_MODELS_DIR, list(reversed(ALL)),
                                 fit_models=True, production_run=True)


def test_sandboxed_runs_are_unrestricted(tmp_path):
    for positions in (["QB"], ALL):
        for fit_models in (True, False):
            assert_safe_models_dir_write(tmp_path, positions, fit_models=fit_models,
                                         production_run=False)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("pipeline ran past the guard")


@pytest.mark.parametrize("production_run, fit_models", [(True, True), (False, False)])
def test_prepare_training_data_refuses_before_doing_anything(monkeypatch, production_run,
                                                             fit_models):
    """The two incidents: a QB-only production retrain, and a direct
    fit_models=False call against the live dir to 'regenerate' the scaler."""
    monkeypatch.setattr("src.features.college_conference.add_conference_features",
                        _fail_if_called)
    with pytest.raises(ProductionArtifactWriteError):
        fp._prepare_training_data(_frame(), _frame(), ["QB"], tune_hyperparameters=False,
                                  n_trials=1, fit_models=fit_models,
                                  production_run=production_run)


def test_train_models_refuses_a_subset_before_loading_or_snapshotting(monkeypatch):
    """Refusing only inside _prepare_training_data would come after
    snapshot_models(), whose two-version retention could evict a good
    snapshot on the way to the error."""
    for name in ("validate_training_cache_integrity", "run_db_quality_gates",
                 "load_training_data", "snapshot_models", "_prepare_training_data"):
        monkeypatch.setattr(train_module, name, _fail_if_called)
    with pytest.raises(ProductionArtifactWriteError, match="RB"):
        train_module.train_models(positions=["QB"], tune_hyperparameters=False)


def test_walk_forward_with_no_folds_does_not_fall_through_to_production(monkeypatch, tmp_path):
    """With every fold skipped, walk-forward used to continue into the
    production path: snapshot, then a real retrain of `positions`."""
    class _Tracker:
        def start_run(self, **kwargs):
            return "test-run"

        def log_params(self, *args, **kwargs):
            pass

    monkeypatch.setattr("src.evaluation.experiment_tracker.ExperimentTracker", _Tracker)
    frame = pd.DataFrame({"player_id": ["P"] * 3, "season": [2021, 2022, 2023],
                          "week": [1, 1, 1], "position": ["QB"] * 3,
                          "fantasy_points": [1.0, 2.0, 3.0]})

    def _load(positions, test_season=None, **kwargs):
        train, test = frame.iloc[:2], frame.iloc[2:]  # 1 test row: every fold skips
        if kwargs.get("return_context"):
            return train, test, [2021, 2022], 2023, None
        return train, test, [2021, 2022], test_season

    monkeypatch.setattr(train_module, "load_training_data", _load)
    monkeypatch.setattr(train_module, "find_artifact_ids", lambda **kwargs: [])
    monkeypatch.setattr(train_module, "snapshot_models", _fail_if_called)
    monkeypatch.setattr(train_module, "_prepare_training_data", _fail_if_called)
    with redirect_models_dir(tmp_path), pytest.raises(RuntimeError, match="no fold metrics"):
        train_module.train_models(positions=["QB"], walk_forward=True,
                                  tune_hyperparameters=False, skip_cache_check=True,
                                  skip_quality_gate=True)


def test_ablation_trains_in_a_sandbox_and_still_writes_its_report(monkeypatch, tmp_path):
    """Every ModelTrainer in the study saves into MODELS_DIR; unsandboxed, the
    study left production serving its last (no_rank_no_util) variant."""
    import src.evaluation.ablation as ablation
    import src.models.position_models as position_models

    seen = {}

    def _study(positions, fast, test_season):
        seen["models_dir"] = position_models.MODELS_DIR
        return {"summary": {}}

    monkeypatch.setattr(ablation, "_run_ablation_study", _study)
    monkeypatch.setattr(ablation, "format_ablation_report", lambda results, positions: "")
    with redirect_models_dir(tmp_path):  # stands in for data/models in this test
        ablation.run_ablation_study(positions=ALL)

    assert seen["models_dir"] not in (tmp_path, settings.PRODUCTION_MODELS_DIR)
    assert (tmp_path / "ablation_results.json").is_file()
