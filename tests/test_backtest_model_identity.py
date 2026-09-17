"""Backtest artifacts must say which model produced them, and the results
page must publish the SERVED model's numbers.

Before this, backtester/train.py artifacts recorded no model identity and
scripts/generate_results_page.py took the newest ts_backtest_2025_*.json
regardless of model -- a Ridge walk-forward's accuracy published as if it
were the stacked GBM ensemble's (AUDIT_REPORT.md #16).
"""
import json

import pytest

from src.evaluation.backtester import describe_model_type


class _Fitted:
    def predict(self, X):
        return X


class _PositionModel:
    def __init__(self, bases, meta):
        self.models = {b: _Fitted() for b in bases}
        self.models["lightgbm"] = None            # absent learner must be skipped
        self.meta_learner = _Fitted() if meta else None


class _MultiWeek:
    def __init__(self, bases, meta, horizons=(1, 2, 3, 4)):
        pm = _PositionModel(bases, meta)
        self.models = {h: pm for h in horizons}


def test_describe_model_type_is_derived_from_fitted_objects(tmp_path, monkeypatch):
    import src.evaluation.backtester as bt
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)
    (tmp_path / "feature_version.txt").write_text("35")

    out = describe_model_type(
        {"QB": _MultiWeek(["xgboost", "ridge"], meta=True),
         "RB": _MultiWeek(["random_forest"], meta=False, horizons=(1,))},
    )
    assert out["model_type"] == "per_position_stacked_ensemble"
    assert out["model_type_by_position"]["QB"] == "stacked(RidgeCV meta)[ridge+xgboost] horizons=[1, 2, 3, 4]"
    assert out["model_type_by_position"]["RB"] == "weighted_blend[random_forest] horizons=[1]"
    assert out["models_feature_version"] == "35"
    assert out["feature_version"]  # the code's, always present


def test_missing_models_version_file_is_none_not_error(tmp_path, monkeypatch):
    import src.evaluation.backtester as bt
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)
    assert describe_model_type({})["models_feature_version"] is None


@pytest.fixture
def results_page(tmp_path, monkeypatch):
    import scripts.generate_results_page as g
    monkeypatch.setattr(g, "BACKTEST_DIR", tmp_path)
    return g, tmp_path


def _write(dir_, name, payload):
    (dir_ / name).write_text(json.dumps(payload))


def test_results_page_publishes_served_model_not_newest_ridge(results_page):
    g, d = results_page
    _write(d, "ts_backtest_2025_20260901_000000.json", {
        "season": 2025, "backtest_date": "2026-09-01", "model_type": "ridge(alpha=1.0)",
        "metrics": {"rmse": 1.0}, "decision_quality": {"n_weeks": 3, "weekly_results": []},
    })
    _write(d, "backtest_2025_20260801.json", {          # older served-model artifact
        "season": 2025, "backtest_date": "2026-08-01", "model_source": "production_ensemble",
        "metrics": {"rmse": 9.0},
    })
    _write(d, "backtest_2025_20260903.json", {          # newest served-model artifact
        "season": 2025, "backtest_date": "2026-09-03", "model_source": "production_ensemble",
        "model_type": "per_position_stacked_ensemble", "feature_version": "36",
        "metrics": {"rmse": 6.0}, "strong_baseline_comparison": {"season_avg": {"rmse_improvement_pct": 4.0}},
    })
    _write(d, "backtest_2025_20260905.json", {          # newest overall, but a baseline fallback run
        "season": 2025, "backtest_date": "2026-09-05", "model_source": "baseline", "metrics": {"rmse": 0.1},
    })

    s = g.build_backtest_summary()

    assert s["source_file"] == "backtest_2025_20260903.json"
    assert s["is_served_model"] is True
    assert s["model_source"] == "production_ensemble"
    assert s["model_type"] == "per_position_stacked_ensemble"
    assert s["overall"] == {"rmse": 6.0}
    assert s["strong_baseline_comparison"]["season_avg"]["rmse_improvement_pct"] == 4.0
    # Lineup/decision analysis still comes from the Ridge run, and says so.
    assert s["decision_quality_source"]["source_file"] == "ts_backtest_2025_20260901_000000.json"
    assert s["decision_quality_source"]["model_type"] == "ridge(alpha=1.0)"
    assert s["decision_quality"]["n_weeks"] == 3


def test_results_page_falls_back_to_ridge_when_no_served_artifact(results_page):
    g, d = results_page
    _write(d, "ts_backtest_2025_20260901_000000.json", {
        "season": 2025, "backtest_date": "2026-09-01", "metrics": {"rmse": 1.0}, "decision_quality": {},
    })
    s = g.build_backtest_summary()
    assert s["is_served_model"] is False
    assert s["source_file"] == "ts_backtest_2025_20260901_000000.json"
    assert s["model_type"] == "unknown"        # an old Ridge artifact never recorded it
