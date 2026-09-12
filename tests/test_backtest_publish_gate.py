"""An off-scale backtest artifact must never be published as model accuracy.

2026-09-10: the standalone backtester fed unscaled features to models trained
on bounded-scaled ones. Predictions came out ~5x too large (avg 40 vs actual
7, MAPE 687%), every success criterion was False -- and the artifact was
still saved under the normal name, advanced_model_results.json was still
written, and the results page would have published it. The verdict existed;
nothing consumed it.
"""
import json

import pytest

from src.evaluation.backtester import ModelBacktester, assess_artifact_trust


def _results(**metrics):
    base = {"season": 2025, "n_predictions": 6000,
            "metrics": {"rmse": 6.0, "r2": 0.4, "mape": 40.0, "avg_actual": 7.0, "avg_predicted": 7.2}}
    base["metrics"].update(metrics)
    return base


def test_sane_artifact_is_trusted():
    assert assess_artifact_trust(_results()) == {"trusted": True, "reasons": []}


def test_off_scale_predictions_are_untrusted():
    verdict = assess_artifact_trust(_results(avg_predicted=40.06, mape=687.1, r2=-23.3))
    assert verdict["trusted"] is False
    assert any("off-scale" in r for r in verdict["reasons"])
    assert any("MAPE" in r for r in verdict["reasons"])


def test_losing_to_a_baseline_is_still_a_valid_measurement():
    """Quality is reported elsewhere; the gate is about scale, not merit."""
    assert assess_artifact_trust(_results(r2=-0.2, mape=60.0))["trusted"] is True


def test_tiny_sample_is_untrusted():
    r = _results(); r["n_predictions"] = 30
    assert assess_artifact_trust(r)["trusted"] is False


def test_save_results_stamps_verdict_and_renames_untrusted(tmp_path):
    bt = ModelBacktester()
    bt.results_dir = tmp_path

    good = _results()
    path = bt.save_results(good)
    assert path.name.startswith("backtest_2025_") and "UNTRUSTED" not in path.name
    assert json.loads(path.read_text())["trust"]["trusted"] is True

    bad = _results(avg_predicted=40.0, mape=687.0)
    path = bt.save_results(bad)
    assert path.name.endswith("_UNTRUSTED.json")
    saved = json.loads(path.read_text())
    assert saved["trust"]["trusted"] is False and saved["trust"]["reasons"]


@pytest.fixture
def results_page(tmp_path, monkeypatch):
    import scripts.generate_results_page as g
    monkeypatch.setattr(g, "BACKTEST_DIR", tmp_path)
    return g, tmp_path


def test_results_page_skips_untrusted_served_artifacts(results_page):
    g, d = results_page
    (d / "ts_backtest_2025_20260901_000000.json").write_text(json.dumps(
        {"season": 2025, "metrics": {"rmse": 1.0}, "decision_quality": {}}))
    (d / "backtest_2025_20260903.json").write_text(json.dumps(
        {"season": 2025, "model_source": "production_ensemble", "metrics": {"rmse": 6.0}}))   # pre-gate, no verdict
    (d / "backtest_2025_20260910.json").write_text(json.dumps(
        {"season": 2025, "model_source": "production_ensemble", "metrics": {"rmse": 38.6},
         "trust": {"trusted": False, "reasons": ["off-scale"]}}))
    (d / "backtest_2025_20260911_UNTRUSTED.json").write_text(json.dumps(
        {"season": 2025, "model_source": "production_ensemble", "metrics": {"rmse": 40.0}}))
    # A --weeks quick check: trusted, newest, but only part of the season.
    (d / "backtest_2025_20260912_PARTIAL.json").write_text(json.dumps(
        {"season": 2025, "model_source": "production_ensemble", "metrics": {"rmse": 5.0},
         "trust": {"trusted": True, "reasons": []}, "partial_season": True}))
    (d / "backtest_2025_20260913.json").write_text(json.dumps(
        {"season": 2025, "model_source": "production_ensemble", "metrics": {"rmse": 5.0},
         "trust": {"trusted": True, "reasons": []}, "partial_season": True}))   # flag without suffix

    s = g.build_backtest_summary()
    assert s["source_file"] == "backtest_2025_20260903.json"
    assert s["overall"] == {"rmse": 6.0}
