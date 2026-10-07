"""The weekly page's accuracy blocks must describe the run of the code it serves.

2026-09-17 produced two trusted, full-season serving-path walk-forwards of the
same models: `backtest_2025_20260917_v5prod.json` at 07:47, before the pace
blend shipped, and `backtest_2025_20260917.json` at 12:00, of the blended code
the page serves. The selector took the last FILE NAME, and '_' sorts after '.',
so the page published the pre-blend run: bias +0.9/+0.6/+0.5 for QB/RB/WR
where the served model runs -0.6/-0.2/-0.0, and "loses to a blended
heuristic" where it beats it.
"""
import json

import pytest


def _artifact(run_at, qb_bias, blended_pct, **extra):
    raw = {
        "season": 2025,
        "backtest_date": run_at,
        "backtest_path": "serving_path_as_of_walk_forward",
        "partial_season": False,
        "weeks_evaluated": list(range(1, 19)),
        "trust": {"trusted": True, "reasons": []},
        "model_metadata": {"training_date": "2026-09-17T03:32:56"},
        "by_position": {"QB": {"mae": 6.7, "avg_predicted": 17.0 + qb_bias, "avg_actual": 17.0}},
        "strong_baseline_comparison": {
            "blended_heuristic": {"rmse_improvement_pct": blended_pct},
            "trailing_3g_avg": {"rmse_improvement_pct": 7.0},
        },
        # Legacy artifacts hold numpy bools serialized as strings.
        "success_criteria": {"model_has_real_edge": "False",
                             "beat_all_baselines_by_20_pct": False},
    }
    raw.update(extra)
    return raw


@pytest.fixture
def weekly(tmp_path, monkeypatch):
    import scripts.generate_weekly_data as g
    monkeypatch.setattr(g, "BACKTEST_DIR", tmp_path)
    return g, tmp_path


def _write(d, name, raw):
    (d / name).write_text(json.dumps(raw))


def test_latest_run_wins_not_last_file_name(weekly):
    g, d = weekly
    _write(d, "backtest_2025_20260917_v5prod.json", _artifact("2026-09-17T07:47:30", +0.94, -0.42))
    _write(d, "backtest_2025_20260917.json", _artifact("2026-09-17T12:00:30", -0.60, +0.86))

    latest = g.latest_serving_backtest()
    assert latest[0] == "backtest_2025_20260917.json"

    measured = g.measured_accuracy(latest)
    assert measured["QB"]["bias"] == pytest.approx(-0.60)
    assert measured["_source"]["file"] == "backtest_2025_20260917.json"
    assert measured["_source"]["backtest_date"] == "2026-09-17T12:00:30"
    assert measured["_source"]["models_trained_at"] == "2026-09-17T03:32:56"

    standing = g.baseline_standing(latest)
    assert standing["rmse_improvement_pct"]["blended_heuristic"] == pytest.approx(0.86)
    assert standing["beats_every_baseline"] is True


def test_success_flags_are_real_booleans(weekly):
    g, d = weekly
    _write(d, "backtest_2025_20260917.json", _artifact("2026-09-17T12:00:30", 0.0, 1.0))
    standing = g.baseline_standing(g.latest_serving_backtest())
    assert standing["model_has_real_edge"] is False          # was the truthy string "False"
    assert standing["beat_all_baselines_by_20_pct"] is False
    assert g._flag("True") is True and g._flag(None) is None and g._flag("maybe") is None


def test_ineligible_artifacts_never_win(weekly):
    g, d = weekly
    _write(d, "backtest_2025_20260917.json", _artifact("2026-09-17T12:00:30", -0.6, 0.86))
    later = "2026-09-20T09:00:00"
    _write(d, "backtest_2025_20260920_UNTRUSTED.json", _artifact(later, 9.0, -9.0))
    _write(d, "backtest_2025_20260920_PARTIAL.json", _artifact(later, 9.0, -9.0))
    _write(d, "backtest_2025_20260921.json", _artifact(later, 9.0, -9.0, partial_season=True))
    _write(d, "backtest_2025_20260922.json", _artifact(later, 9.0, -9.0, backtest_path=None))
    _write(d, "backtest_2025_20260923.json",
           _artifact(later, 9.0, -9.0, trust={"trusted": False, "reasons": ["off-scale"]}))
    undated = _artifact(later, 9.0, -9.0)
    del undated["backtest_date"]
    _write(d, "backtest_2025_20260924.json", undated)

    assert g.latest_serving_backtest()[0] == "backtest_2025_20260917.json"


def test_latest_season_before_latest_run(weekly):
    g, d = weekly
    _write(d, "backtest_2025_20260917.json", _artifact("2026-09-17T12:00:30", -0.6, 0.86))
    older_season = _artifact("2026-09-25T12:00:00", 9.0, -9.0, season=2024)
    _write(d, "backtest_2024_20260925.json", older_season)
    assert g.latest_serving_backtest()[0] == "backtest_2025_20260917.json"


def test_both_blocks_come_from_one_artifact(weekly):
    """The newest run lacking a baseline block must not borrow an older run's."""
    g, d = weekly
    _write(d, "backtest_2025_20260917.json", _artifact("2026-09-17T12:00:30", -0.6, 0.86))
    newest = _artifact("2026-09-18T12:00:00", -0.5, 0.0)
    newest["strong_baseline_comparison"] = {}
    _write(d, "backtest_2025_20260918.json", newest)

    latest = g.latest_serving_backtest()
    assert latest[0] == "backtest_2025_20260918.json"
    assert g.baseline_standing(latest) is None


def test_no_artifact_publishes_nothing(weekly):
    g, _ = weekly
    assert g.latest_serving_backtest() is None
    assert g.measured_accuracy(None) is None and g.baseline_standing(None) is None
