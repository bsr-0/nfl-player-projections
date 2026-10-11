import numpy as np
import pandas as pd
import pytest

from src.evaluation import game_backtest as B


def _games():
    rows = []
    for w, final in ((1, True), (2, True), (3, False)):
        for g in range(2):
            rows.append({"week": w, "home_team": f"H{g}", "away_team": f"A{g}",
                         "kickoff": pd.Timestamp("2026-09-10", tz="UTC") + pd.Timedelta(days=7 * w),
                         "home_score": 24.0 if final else np.nan, "away_score": 20.0 if final else np.nan, "final": final})
    return pd.DataFrame(rows)


def test_complete_weeks_needs_every_game_final():
    g = _games()
    assert B.complete_weeks(g) == [1, 2]
    g.loc[(g.week == 2) & (g.home_team == "H1"), "final"] = False
    assert B.complete_weeks(g) == [1]


class _Stub:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([1 - self.p * np.ones(len(X)), self.p * np.ones(len(X))])

    def predict(self, X):
        return self.p * np.ones(len(X))


def test_season_backtest_scores_complete_weeks_with_the_walkforward_set(monkeypatch):
    seen = {}
    rows = pd.DataFrame([{"season": 2026, "week": w, "home_team": f"H{g}", "away_team": f"A{g}", "home_win": 1,
                          "f1": 0.0, "spread_line": 3.0, "total_line": 44.0}
                         for w in (1, 2, 3) for g in range(2)])
    monkeypatch.setattr(B, "build_game_outcome_rows", lambda con=None, **k: rows)
    monkeypatch.setattr(B, "kickoff_times", lambda season, con: _games())

    def ensure(name, **k):
        seen["ensure"] = (name, k["before_season"])
        return ({"trained_before_season": 2026, "trained_through": {"season": 2025, "week": 22},
                 "n_games": 5417, "fitted_at": "x"}, "fitted through 2025 wk22")
    monkeypatch.setattr(B.registry, "ensure_set", ensure)
    monkeypatch.setattr(B.registry, "load_set", lambda name, models_dir=None: {
        "home_win_prob_logistic": (_Stub(0.7), "proba"), "predicted_margin_ridge": (_Stub(4.0), "point"),
        "predicted_total_ridge": (_Stub(46.0), "point")})
    out, info = B.season_backtest(2026, None)
    assert seen["ensure"] == ("walkforward_2026", 2026)
    assert sorted(out["week"].unique()) == [1, 2] and len(out) == 4          # week 3 is not complete
    assert info["weeks"] == [1, 2] and info["fitted_before_season"] == 2026
    s = B.backtest_summary(out)
    assert s["season"]["n"] == 4 and s["season"]["win_loss"]["logistic"]["correct"] == 4
    assert set(s["weeks"]) == {"1", "2"}
    # ridge margin 4 vs home win by 4 against a spread of 3: pick wins; market miss is 1
    assert s["season"]["margin"]["ridge"]["mae"] == 0.0 and s["season"]["margin"]["market_line"]["mae"] == 1.0


def test_stale_walkforward_models_are_refused(monkeypatch):
    monkeypatch.setattr(B.registry, "ensure_set", lambda name, **k: ({"x": 1}, "STALE: files missing"))
    with pytest.raises(RuntimeError):
        B.season_backtest(2026, None, refit=False)


def test_unpublished_rows_keep_only_games_without_a_published_prediction():
    bt = pd.DataFrame({"week": [2, 2, 3], "home_team": ["BUF", "KC", "KC"], "away_team": ["DET", "DEN", "LV"]})
    out = B.unpublished_rows(bt, {(2, "KC", "DEN"), (3, "KC", "LV")})
    assert out[["home_team", "away_team"]].values.tolist() == [["BUF", "DET"]]
    assert (out["source"] == "backtest").all()
