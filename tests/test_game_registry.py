import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.models.game_outcome import registry as R


def _frames(n_seasons=(2024, 2025, 2026), weeks=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in n_seasons:
        for w in weeks:
            for g in range(8):
                rows.append({"season": s, "week": w, "home_team": f"H{g}", "away_team": f"A{g}",
                             "f1": rng.normal(), "f2": rng.normal(), "spread_line": rng.normal(0, 3),
                             "total_line": 44 + rng.normal(0, 3)})
    base = pd.DataFrame(rows)
    outcome = base.assign(home_win=(base["f1"] + rng.normal(0, 1, len(base)) > 0).astype(int))
    outcome = outcome[["season", "week", "home_team", "away_team", "home_win", "f1", "f2", "spread_line", "total_line"]]
    margin = base.assign(home_margin=base["f1"] * 6 + rng.normal(0, 5, len(base)),
                         game_total=44 + base["f2"] * 3 + rng.normal(0, 3, len(base)))
    margin = margin[["season", "week", "home_team", "away_team", "home_margin", "game_total",
                     "f1", "f2", "spread_line", "total_line"]]
    return outcome, margin


@pytest.fixture
def stub(monkeypatch):
    state = {"frames": _frames()}
    monkeypatch.setattr(R, "build_game_outcome_rows", lambda con=None, **k: state["frames"][0].copy())
    monkeypatch.setattr(R, "build_margin_total_rows", lambda con=None, **k: state["frames"][1].copy())
    return state


def test_fit_set_saves_nine_models_and_a_manifest(stub, tmp_path):
    entry = R.fit_set("production", con=None, models_dir=tmp_path)
    assert sorted(entry["artifacts_sha256"]) == sorted(a[2] for a in R.ARTIFACTS) and len(entry["artifacts_sha256"]) == 9
    assert entry["trained_through"] == {"season": 2026, "week": 3} and entry["n_games"] == 72
    assert json.loads((tmp_path / R.MANIFEST_NAME).read_text())["sets"]["production"]["n_games"] == 72
    models = R.load_set("production", tmp_path)
    X = stub["frames"][0][["f1", "f2", "spread_line", "total_line"]].head(5)
    out = R.predict_set(models, X)
    assert set(out.columns) == {a[4] for a in R.ARTIFACTS} and len(out) == 5
    assert out["home_win_prob_logistic"].between(0, 1).all()


def test_walkforward_set_never_sees_its_own_season(stub, tmp_path):
    entry = R.fit_set(R.walkforward_name(2026), con=None, models_dir=tmp_path, before_season=2026)
    assert entry["trained_before_season"] == 2026 and entry["n_games"] == 48
    assert entry["trained_through"] == {"season": 2025, "week": 3}
    assert (tmp_path / R.WALKFORWARD_DIR / "2026" / "game_outcome_rf.joblib").exists()


def test_production_goes_stale_when_a_new_week_completes_and_refit_clears_it(stub, tmp_path):
    R.fit_set("production", con=None, models_dir=tmp_path)
    ok, why = R.check_set("production", con=None, models_dir=tmp_path)
    assert ok, why
    stub["frames"] = _frames(weeks=(1, 2, 3, 4))
    ok, why = R.check_set("production", con=None, models_dir=tmp_path)
    assert not ok and "wk3" in why and "wk4" in why
    entry, note = R.ensure_set("production", con=None, models_dir=tmp_path)
    assert note.startswith("refit") and entry["trained_through"]["week"] == 4
    assert R.check_set("production", con=None, models_dir=tmp_path)[0]


def test_no_refit_flag_reports_stale_instead_of_fitting(stub, tmp_path):
    R.fit_set("production", con=None, models_dir=tmp_path)
    stub["frames"] = _frames(weeks=(1, 2, 3, 4))
    entry, note = R.ensure_set("production", con=None, models_dir=tmp_path, refit=False)
    assert note.startswith("STALE") and entry["trained_through"]["week"] == 3


def test_walkforward_set_ignores_in_season_results(stub, tmp_path):
    name = R.walkforward_name(2026)
    R.fit_set(name, con=None, models_dir=tmp_path, before_season=2026)
    more = _frames(weeks=(1, 2, 3, 4))             # 2026 gains a week; seasons before 2026 gain nothing
    stub["frames"] = tuple(f[(f["season"] == 2026) | (f["week"] <= 3)].reset_index(drop=True) for f in more)
    assert R.check_set(name, con=None, models_dir=tmp_path, before_season=2026)[0]
    ok, why = R.check_set(name, con=None, models_dir=tmp_path, before_season=2027)
    assert not ok and "before season 2026" in why


def test_a_changed_artifact_or_feature_list_is_not_current(stub, tmp_path):
    R.fit_set("production", con=None, models_dir=tmp_path)
    (tmp_path / "game_total_rf.joblib").write_bytes(b"tampered")
    ok, why = R.check_set("production", con=None, models_dir=tmp_path)
    assert not ok and "game_total_rf.joblib" in why
    R.fit_set("production", con=None, models_dir=tmp_path)
    frames = _frames()
    stub["frames"] = (frames[0].assign(f3=1.0), frames[1].assign(f3=1.0))
    ok, why = R.check_set("production", con=None, models_dir=tmp_path)
    assert not ok and "feature list" in why


def test_check_without_a_manifest_entry_is_stale(tmp_path, stub):
    ok, why = R.check_set("production", con=None, models_dir=tmp_path)
    assert not ok and "no manifest entry" in why


def test_tuned_params_come_from_the_sidecars(tmp_path):
    (tmp_path / "game_outcome_model_metadata.json").write_text(json.dumps({"tuned_params": {"logistic": {"C": 3.0}}}))
    got = R.tuned_params(tmp_path)
    assert got["outcome"] == {"logistic": {"C": 3.0}} and got["margin"] == {} and got["total"] == {}


def test_retuned_hyperparameters_make_both_sets_stale(stub, tmp_path):
    R.fit_set("production", con=None, models_dir=tmp_path)
    R.fit_set(R.walkforward_name(2026), con=None, models_dir=tmp_path, before_season=2026)
    (tmp_path / "game_outcome_model_metadata.json").write_text(json.dumps({"tuned_params": {"logistic": {"C": 3.0}}}))
    for name, before in (("production", None), (R.walkforward_name(2026), 2026)):
        ok, why = R.check_set(name, con=None, models_dir=tmp_path, before_season=before)
        assert not ok and "hyperparameters" in why
    entry, note = R.ensure_set(R.walkforward_name(2026), con=None, models_dir=tmp_path, before_season=2026)
    assert note.startswith("refit")
    assert R.check_set(R.walkforward_name(2026), con=None, models_dir=tmp_path, before_season=2026)[0]
