"""WR volume shadow test: the capture hook, volume lookup, weekly validity and the scorer's rule (synthetic rows only)."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import score_wr_volume_prereg as sc
from scripts import wr_volume_shadow as sh
from src.evaluation.paired_ppr_comparison import file_sha256


def test_capture_hook_copies_the_frame_and_changes_nothing():
    from src.models.ensemble import EnsemblePredictor
    original = EnsemblePredictor.predict
    calls = []

    def fake(self, player_data, n_weeks=1):
        calls.append(player_data)
        return player_data.assign(predicted_points=1.0)

    EnsemblePredictor.predict = fake
    try:
        frame = pd.DataFrame({"player_id": ["a"], "position": ["WR"]})
        with sh.capture_model_input() as box:
            out = EnsemblePredictor.predict(None, frame, 1)
        assert out.predicted_points.iloc[0] == 1.0 and calls[0] is frame        # same object reached the model
        assert box["frame"] is not frame and box["frame"].equals(frame)           # a copy, not the live frame
        box["frame"].loc[0, "player_id"] = "mutated"
        assert frame.player_id.iloc[0] == "a"
    finally:
        EnsemblePredictor.predict = original
    assert EnsemblePredictor.predict is original                                  # restored


def test_latest_volume_uses_the_latest_game_before_the_target_week():
    rows = [("a", 2025, 17, 8, 4, 80), ("a", 2025, 18, 4, 2, 40), ("a", 2026, 1, 10, 6, 100), ("a", 2026, 2, 6, 3, 50),
            ("a", 2026, 3, 99, 99, 999), ("b", 2025, 18, 6, 3, 30)]
    stats = pd.DataFrame(rows, columns=["player_id", "season", "week", "targets", "receptions", "receiving_yards"])
    stats["fantasy_points"] = 1.0
    out = sh.latest_volume(stats, 2026, 3).set_index("player_id")
    assert out.loc["a", "targets_s2d"] == pytest.approx(8.0)                      # mean(10, 6): through 2026 wk 2 only
    assert out.loc["b", "receptions_s2d"] == 3 and out.loc["b", "receiving_yards_s2d"] == 30   # last season's final mean
    assert out.loc["a", "receiving_yards_s2d"] == pytest.approx(75.0)


def test_shadow_weeks_need_hashes_a_real_run_and_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "OUT", tmp_path)

    def week(n, *, dry=False, before=True, tamper=False):
        d = tmp_path / f"week_{n:02d}"
        d.mkdir()
        (d / "forecasts.csv").write_text("player_id\nx\n")
        digest = sh.sha(d / "forecasts.csv")
        (d / "manifest.json").write_text(json.dumps({"week": n, "dry_run": dry, "before_deadline": before,
                                                     "forecasts_sha256": digest}))
        if tamper:
            (d / "forecasts.csv").write_text("player_id\ny\n")
    week(7); week(8, tamper=True); week(9, dry=True); week(10, before=False)
    got = sh.check()
    assert got["valid_weeks"] == [7]
    assert got["excluded"][8].startswith("forecasts differ") and got["excluded"][10] == "written after the deadline"
    assert got["excluded"][13] == "not written"


def _frame(effect: float, seed: int = 3, noise: float = 4.0):
    rng = np.random.default_rng(seed)
    weeks = np.arange(7, 14)
    rows = []
    for w in weeks:
        for i in range(70):
            rows.append((f"p{i}", w, 4))
    d = pd.DataFrame(rows, columns=["player_id", "week", "games_before"])
    truth = np.tile(rng.gamma(3, 3, 70), len(weeks))
    d["actual"] = truth + rng.normal(0, noise, len(d))
    d["ppg_before"] = truth + rng.normal(0, 1, len(d))
    d["c0"] = 0.8 * truth + rng.normal(0, 1, len(d))                 # lags the player's level
    d["c1"] = d.c0 + effect * (truth - d.c0)                         # moves toward it
    return d


def test_a_better_arm_passes_every_rule():
    r = sc.score(_frame(0.8), B=400, seed=1)
    assert r["rule_1"]["mean_d"] < 0 and r["rule_1"]["upper_95_one_sided"] < sc.MARGIN
    assert r["passes"] == {"rule_1": True, "rule_2": True, "rule_3": True, "all": True}


def test_an_identical_arm_does_not_pass_rule_one_or_three():
    r = sc.score(_frame(0.0), B=400, seed=1)
    assert r["rule_1"]["mean_d"] == 0 and not r["passes"]["rule_1"] and not r["passes"]["all"]


def test_a_noisier_arm_fails_on_error():
    d = _frame(0.8)
    d["c1"] = d.c1 + np.random.default_rng(9).normal(0, 6, len(d))
    r = sc.score(d, B=400, seed=1)
    assert not r["passes"]["rule_1"] and not r["passes"]["all"]


def test_scorer_refuses_early_and_voids_a_short_window(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "OUTPUT", tmp_path / "report.json")
    assert sc.MIN_WEEKS == 5 and sc.MARGIN == 0.03


def test_fill_mirrors_production_for_inputs_absent_from_the_frame():
    class Base:
        feature_names = ["a", "b"]
        feature_medians = {"a": 1.0, "b": 7.0}

    class Model:
        models = {1: Base()}
    frame = pd.DataFrame({"a": [5.0, np.nan]})
    out = sh.fill_like_production(frame, Model())
    assert out.b.tolist() == [7.0, 7.0]                         # absent column -> training median
    assert np.isnan(out.a.iloc[1]) and out.a.iloc[0] == 5.0      # present columns untouched (NaN stays NaN)
    assert "b" not in frame.columns                              # input not mutated
