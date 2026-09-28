"""The OOF smoke run exercises the real end-of-run path, in isolation.

Every failed walk-forward so far died after the folds had trained: the
per-fold metrics JSON write, coverage accounting, game context, the panel
verifier. Here training, data loading and the database are stubbed, and
everything from fold capture to verification is the real code -- so this is
the path a real run takes once Optuna is done.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import smoke_test_oof  # noqa: E402

import config.settings as settings
import src.models.train as train_module
from src.evaluation.backtester import ModelBacktester

SEASONS = list(range(2019, 2025))
POSITIONS = ["QB", "RB", "WR", "TE"]
PAIRS = [("T0", "T1"), ("T2", "T3"), ("T4", "T5"), ("T6", "T7")]
OPPONENT = {**{h: a for h, a in PAIRS}, **{a: h for h, a in PAIRS}}


def _season_rows(season):
    rng = np.random.default_rng(season)
    rows = [{"player_id": f"{pos}{k}", "season": season, "week": week, "team": f"T{k}",
             "opponent": OPPONENT[f"T{k}"], "position": pos,
             "fantasy_points": float(rng.gamma(2.0, 5.0))}
            for pos in POSITIONS for k in range(6) for week in range(1, 5)]
    frame = pd.DataFrame(rows)
    frame["target_1w"] = frame.groupby(["player_id", "season"])["fantasy_points"].shift(-1)
    return frame


ALL_ROWS = pd.concat([_season_rows(s) for s in SEASONS], ignore_index=True)


def _load(positions, test_season=None, **kwargs):
    test_season = test_season or SEASONS[-1]
    train_seasons = [s for s in SEASONS if s < test_season]
    train = ALL_ROWS[ALL_ROWS["season"].isin(train_seasons)]
    test = ALL_ROWS[ALL_ROWS["season"] == test_season]
    if kwargs.get("return_context"):
        return train, test, train_seasons, test_season, ALL_ROWS.iloc[0:0]
    return train, test, train_seasons, test_season


class _Model:
    models = None
    feature_names = []
    feature_medians = {}

    def __init__(self):
        self.models = {1: self}

    def predict(self, frame, n_weeks=1):
        return frame["fantasy_points"].to_numpy() * 0.5 + 4.0


class _Trainer:
    trained_models = {pos: _Model() for pos in POSITIONS}
    training_metrics = {}


def _backtest(trainer, test_data, train_seasons, season, train_data=None, publish=True):
    assert publish is False, "a walk-forward fold must never publish its backtest"
    # The real metric function, so the fold-metrics write sees real types.
    scored = test_data.dropna(subset=["actual_for_backtest"])
    return {"by_position": {
        pos: ModelBacktester()._calculate_metrics(part["actual_for_backtest"], part["predicted_points"])
        for pos, part in scored.groupby("position")}}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    db_path = tmp_path / "nfl.db"
    with sqlite3.connect(db_path) as conn:
        pd.DataFrame([{"season": s, "week": w, "home_team": h, "away_team": a,
                       "game_id": f"{s}_{w}_{h}_{a}"}
                      for s in SEASONS for w in range(1, 5) for h, a in PAIRS]
                     ).to_sql("schedule", conn, index=False)
        ALL_ROWS[["player_id", "season", "week"]].to_sql("player_weekly_stats", conn, index=False)

    class _Db:
        def __init__(self, path=None):
            self.db_path = db_path

    class _Tracker:
        def start_run(self, **kwargs):
            return "smoke-test"

        def log_params(self, *args, **kwargs):
            pass

    monkeypatch.setattr("src.utils.database.DatabaseManager", _Db)
    monkeypatch.setattr("src.evaluation.experiment_tracker.ExperimentTracker", _Tracker)
    monkeypatch.setattr(train_module, "load_training_data", _load)
    monkeypatch.setattr(train_module, "find_artifact_ids", lambda **kwargs: [])
    monkeypatch.setattr(train_module, "_prepare_training_data",
                        lambda train, test, positions, *a, **k: (train, test, _Trainer()))
    monkeypatch.setattr(train_module, "_run_backtest_after_training", _backtest)
    # train_models(fast=True) rewrites MODEL_CONFIG in place; undo it after.
    for key in settings.FAST_MODEL_CONFIG:
        monkeypatch.setitem(settings.MODEL_CONFIG, key, settings.MODEL_CONFIG.get(key))


def _mtimes(paths):
    return {p: p.stat().st_mtime_ns for p in paths if p.exists()}


def test_smoke_reaches_a_verified_panel_without_touching_real_outputs(stubbed, tmp_path):
    real = [settings.PRODUCTION_MODELS_DIR / "scoring_environment_report.json",
            settings.DATA_DIR / "experiments" / "walk_forward_fold_metrics.json",
            settings.DATA_DIR / "experiments" / "walk_forward_oof_predictions.parquet"]
    before = _mtimes(real)

    out_dir = tmp_path / "smoke"
    verification = smoke_test_oof.run_smoke(out_dir, skip_cache_check=True, skip_quality_gate=True)

    assert verification["status"] == "verified"
    assert verification["seasons"] == [2021, 2022, 2023, 2024]
    assert verification["positions"] == sorted(POSITIONS)
    assert (out_dir / "walk_forward_fold_metrics.json").is_file()
    assert (out_dir / "walk_forward_oof_predictions.parquet").is_file()
    assert _mtimes(real) == before, "the smoke run wrote outside its own directories"


def test_smoke_reports_a_blocked_run_as_a_failure(stubbed, monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "train_models", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="blocked"):
        smoke_test_oof.run_smoke(tmp_path / "smoke")


def test_smoke_refuses_to_reuse_an_output_directory(tmp_path):
    (tmp_path / "smoke").mkdir()
    with pytest.raises(FileExistsError):
        smoke_test_oof.run_smoke(tmp_path / "smoke")


# --------------------------------------------------------------------------
# isolation is verified at runtime, not assumed
# --------------------------------------------------------------------------

@pytest.fixture
def protected_sandbox(stubbed, monkeypatch, tmp_path):
    """Point the tripwire at a tmp data dir + models dir holding one of each
    protected kind, and let the test decide what the 'run' writes."""
    data = tmp_path / "data"
    models = data / "models"
    (data / "backtest_results").mkdir(parents=True)
    models.mkdir()
    (models / "model_qb_1w.joblib").write_bytes(b"served-weights")
    (models / "feature_scaler_bounded.joblib").write_bytes(b"served-scaler")
    (data / "advanced_model_results.json").write_text('{"test_season": 2025}')
    (data / "utilization_percentile_bounds.json").write_text("{}")
    (data / "backtest_results" / "backtest_2025_20260917.json").write_text("{}")
    monkeypatch.setattr(smoke_test_oof, "DATA_DIR", data)
    monkeypatch.setattr(settings, "PRODUCTION_MODELS_DIR", models)
    return data


def _run_writing(monkeypatch, write):
    real = train_module.train_models

    def _train(**kwargs):
        write()
        return real(**kwargs)

    monkeypatch.setattr(train_module, "train_models", _train)


@pytest.mark.parametrize("relative, content", [
    ("advanced_model_results.json", '{"test_season": 2023}'),
    ("utilization_percentile_bounds.json", '{"QB|x": [0, 1]}'),
    ("backtest_results/backtest_2025_20260926.json", "{}"),          # a NEW published file
    ("models/feature_scaler_bounded.joblib", b"fold-scaler"),
    ("models/model_qb_1w.joblib", b"fold-weights"),
])
def test_smoke_fails_when_the_run_modifies_a_protected_file(
        protected_sandbox, monkeypatch, tmp_path, relative, content):
    target = protected_sandbox / relative
    writer = target.write_bytes if isinstance(content, bytes) else target.write_text
    _run_writing(monkeypatch, lambda: writer(content))

    with pytest.raises(RuntimeError, match="modified 1 protected file"):
        smoke_test_oof.run_smoke(tmp_path / "smoke", skip_cache_check=True, skip_quality_gate=True)


def test_smoke_fails_when_the_run_deletes_a_protected_file(protected_sandbox, monkeypatch, tmp_path):
    _run_writing(monkeypatch, (protected_sandbox / "advanced_model_results.json").unlink)
    with pytest.raises(RuntimeError, match="modified 1 protected file"):
        smoke_test_oof.run_smoke(tmp_path / "smoke", skip_cache_check=True, skip_quality_gate=True)


def test_smoke_passes_when_only_unprotected_files_change(protected_sandbox, monkeypatch, tmp_path):
    """Gate reports and caches are legitimately rewritten by data refresh."""
    _run_writing(monkeypatch, lambda: (protected_sandbox / "models" / "data_quality_gate_report.json"
                                       ).write_text("{}"))
    verification = smoke_test_oof.run_smoke(tmp_path / "smoke", skip_cache_check=True,
                                            skip_quality_gate=True)
    assert verification["status"] == "verified"
