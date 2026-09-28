"""Validation folds must not publish, and serving bounds must be redirect-aware.

2026-09-26: the main walk-forward's folds (inside redirect_models_dir) still
overwrote three things outside data/models, all committed in abb7d2e:

  * data/advanced_model_results.json -- the app's "authoritative single source
    of truth" -- became a 2023 fold's results (test_season 2023);
  * data/backtest_results/backtest_2025_20260926.json, labelled
    model_source=production_ensemble, became what the results page publishes
    as the served ensemble's 2025 backtest;
  * data/utilization_percentile_bounds.json, the file serving auto-loads
    (predict.py builds UtilizationScoreCalculator without bounds), got a
    fold's bounds (train seasons through 2023) while the models were trained
    with production's -- a hard-coded path redirect_models_dir cannot reach.
"""
import numpy as np
import pandas as pd
import pytest

import src.evaluation.backtester as backtester_module
import src.models.train as train_module
from src.features.utilization_score import (
    UtilizationScoreCalculator,
    load_percentile_bounds,
    save_percentile_bounds,
)
from src.features.utilization_weight_optimizer import UTIL_COMPONENTS
from src.models.feature_preparation import _fit_utilization_bounds
from src.utils.models_dir import redirect_models_dir

QB_COMPONENT = UTIL_COMPONENTS["QB"][0]


def _qb_frame(n=40, scale=1.0):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"position": "QB", **{c: rng.uniform(0, 100, n) * scale
                                              for c in UTIL_COMPONENTS["QB"]}})


# --------------------------------------------------------------------------
# utilization bounds: one canonical, redirect-aware path
# --------------------------------------------------------------------------

def test_fit_does_not_persist_unless_asked(tmp_path):
    with redirect_models_dir(tmp_path):
        UtilizationScoreCalculator().fit_percentile_bounds(_qb_frame(), "QB", [QB_COMPONENT])
    assert list(tmp_path.iterdir()) == []


def test_persist_goes_to_the_redirected_models_dir_only(tmp_path):
    import config.settings as settings

    stray = settings.DATA_DIR / "utilization_percentile_bounds.json"
    before = stray.stat().st_mtime_ns if stray.exists() else None
    with redirect_models_dir(tmp_path):
        UtilizationScoreCalculator().fit_percentile_bounds(
            _qb_frame(), "QB", [QB_COMPONENT], persist=True)
    assert (tmp_path / "utilization_percentile_bounds.json").is_file()
    assert (stray.stat().st_mtime_ns if stray.exists() else None) == before


def test_serving_calculator_loads_the_file_training_writes(tmp_path):
    """predict.py builds the calculator with no bounds; it must land on the
    same MODELS_DIR file the training pipeline applies to its held-out data."""
    save_percentile_bounds({("QB", QB_COMPONENT): (0.0, 200.0)},
                           tmp_path / "utilization_percentile_bounds.json")
    with redirect_models_dir(tmp_path):
        scaled = UtilizationScoreCalculator()._percentile_normalize(
            pd.Series([100.0, 50.0]), "QB", QB_COMPONENT)
    assert scaled.tolist() == [50.0, 25.0]


def test_fit_drops_bounds_seeded_from_a_previous_run(tmp_path):
    """The calculator auto-loads persisted bounds on first use; anything the
    new fit does not replace used to be re-saved under the NEW run's
    train_seasons metadata."""
    path = tmp_path / "utilization_percentile_bounds.json"
    save_percentile_bounds({("RB", "rush_share_pct"): (1.0, 2.0),
                            ("QB", QB_COMPONENT): (-999.0, 999.0)}, path)
    meta = {"train_seasons": [2022, 2023], "min_season": 2022, "max_season": 2023}
    with redirect_models_dir(tmp_path):
        calc = UtilizationScoreCalculator()
        calc._ensure_bounds_loaded()
        assert ("RB", "rush_share_pct") in calc.position_percentiles  # seeded
        _fit_utilization_bounds(calc, _qb_frame(), path, meta)

    bounds, saved_meta = load_percentile_bounds(path, return_meta=True)
    assert ("RB", "rush_share_pct") not in bounds
    assert {pos for pos, _ in bounds} == {"QB"}
    assert bounds[("QB", QB_COMPONENT)] != (-999.0, 999.0)
    assert saved_meta["train_seasons"] == [2022, 2023]


# --------------------------------------------------------------------------
# fold backtests do not publish
# --------------------------------------------------------------------------

class _PositionModel:
    """Shaped like PositionModel: base learners keyed by name."""
    models = {"ridge": object()}
    meta_learner = None
    feature_names = []
    feature_medians = {}


class _Model:
    """Shaped like MultiWeekModel: one PositionModel per horizon."""
    def __init__(self):
        self.models = {1: _PositionModel()}

    def predict(self, frame, n_weeks=1):
        return frame["fantasy_points"].to_numpy() * 0.5 + 5.0


class _Trainer:
    trained_models = {pos: _Model() for pos in ("QB", "RB", "WR", "TE")}


def _test_frame(n_per_position=100, season=2024):
    rng = np.random.default_rng(1)
    rows = []
    for position in ("QB", "RB", "WR", "TE"):
        for i in range(n_per_position):
            points = float(rng.gamma(2.0, 5.0))
            rows.append({"player_id": f"{position}{i % 25}", "name": f"{position}{i % 25}",
                         "season": season, "week": 1 + i // 25, "team": "AAA", "opponent": "BBB",
                         "position": position, "fantasy_points": points,
                         "target_1w": float(rng.gamma(2.0, 5.0))})
    return pd.DataFrame(rows)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Real backtester and real _run_backtest_after_training; only the data
    directories are pointed at tmp, with a stale file the publish step deletes."""
    monkeypatch.setattr(train_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(backtester_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(train_module, "_load_qb_target_choice", lambda: "fp")
    (tmp_path / "ml_evaluation_results.json").write_text("{}")
    return tmp_path


def _run(publish):
    return train_module._run_backtest_after_training(
        _Trainer(), _test_frame(), [2022, 2023], 2024, publish=publish)


def test_publishing_run_writes_the_app_results_and_artifact(sandbox):
    """Control: without this the next test would pass by never reaching the
    publish step at all."""
    results = _run(publish=True)
    assert results["trust"]["trusted"], results["trust"]
    assert (sandbox / "advanced_model_results.json").is_file()
    assert list((sandbox / "backtest_results").glob("backtest_2024_*.json"))
    assert not (sandbox / "ml_evaluation_results.json").exists()


def test_fold_backtest_publishes_nothing_but_keeps_the_trust_verdict(sandbox):
    results = _run(publish=False)

    assert results["trust"]["trusted"], results["trust"]
    assert results["model_source"] == "production_ensemble"  # why it must not be saved
    assert not (sandbox / "advanced_model_results.json").exists()
    assert not list((sandbox / "backtest_results").glob("backtest_*.json"))
    assert (sandbox / "ml_evaluation_results.json").exists(), "a fold deleted a data file"


def test_run_one_fold_never_publishes(monkeypatch):
    seen = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(train_module, "_prepare_training_data",
                        lambda train, test, positions, *a, **k: (train, test, _Trainer()))
    monkeypatch.setattr(train_module, "_run_backtest_after_training", _spy)
    monkeypatch.setattr(train_module, "_load_qb_target_choice", lambda: "fp")
    frame = _test_frame()
    train_module._run_one_fold(frame, frame.assign(season=2025), [2024], 2025,
                               ["QB"], False, None)
    assert seen["publish"] is False
