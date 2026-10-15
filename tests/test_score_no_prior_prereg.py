"""The no-prior pre-registration scorer, on synthetic rows only (no window week is ever read)."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import score_no_prior_prereg as s
from src.evaluation.paired_ppr_comparison import file_sha256

B = 300


def _rows(unblended_noise: float, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = {"QB": 30, "RB": 60, "WR": 90, "TE": 45}
    players = pd.DataFrame([(f"{p}{i:03d}", p) for p, n in counts.items() for i in range(n)],
                           columns=["player_id", "position"])
    players["skill"] = rng.gamma(3.0, 3.0, len(players))
    players["ahead"] = rng.random(len(players)) < 0.5         # running 3+ PPG above a stale prior
    rows = players.merge(pd.DataFrame({"week": list(s.WINDOW)}), how="cross")
    rows["season"] = s.SEASON
    n, skill = len(rows), rows.skill.to_numpy()
    prior = skill - 3.0 * rows.ahead.to_numpy()
    rows["actual"] = skill + rng.normal(0, 5, n)
    rows["rolling3"] = skill + rng.normal(0, 3, n)
    rows["incumbent_live_blend"] = 0.6 * skill + 0.4 * prior + rng.normal(0, 1, n)
    rows["served_unblended"] = (skill + rng.normal(0, unblended_noise, n)) / s.K
    rows["games_before"] = rows.week - 3
    rows["form_gap"] = skill - prior + rng.normal(0, 0.5, n)
    return s.candidate_forecasts(rows).rename(columns={"incumbent_live_blend": "incumbent"})


@pytest.fixture(scope="module")
def good():
    return s.score(_rows(1.0), B=B, seed=1)


def test_candidates_are_the_registered_transforms():
    fc = pd.DataFrame({"position": ["QB", "WR"], "incumbent_live_blend": [20.0, 9.0], "served_unblended": [22.0, 10.0]})
    out = s.candidate_forecasts(fc)
    assert out.no_prior_scaled.tolist() == pytest.approx([22.0 * 0.9218, 10.0 * 0.9218])
    assert out.no_prior_scaled_qb_blend.tolist() == pytest.approx([20.0, 10.0 * 0.9218])


def test_a_candidate_without_the_lag_is_noninferior_and_leans_less(good):
    for c in s.CANDIDATES:
        r = good["candidates"][c]
        assert r["P1"]["holm_rejects"] and r["guardrail_failures"] == []
        assert r["S1"]["passes"] and r["S1"]["abs_lean_cut"] > 0.5 and r["eligible"]
    assert good["decision"] == "no_prior_scaled"                # both eligible: the simpler one


def test_a_noisier_candidate_is_not_noninferior():
    r = s.score(_rows(6.0), B=B, seed=1)
    assert not r["candidates"]["no_prior_scaled"]["P1"]["holm_rejects"]
    assert not r["candidates"]["no_prior_scaled"]["eligible"]


def test_p1_uses_the_same_draws_as_the_rule_gates(good):
    for c in s.CANDIDATES:
        p1, g1 = good["candidates"][c]["P1"], good["candidates"][c]["report_only"]["superiority_G1_report"]
        assert p1["point_difference"] == pytest.approx(g1["point_difference"])
        assert p1["upper_bound_95"] == pytest.approx(g1["upper_bound_95"])


def test_window_weeks_keep_only_hashed_real_weeks(tmp_path):
    def week(n, *, dry=False, before=True, tamper=False):
        d = tmp_path / f"week_{n:02d}"
        d.mkdir()
        (d / "forecasts.csv").write_text("player_id\nx\n")
        digest = file_sha256(d / "forecasts.csv")
        (d / "manifest.json").write_text(json.dumps({"season": s.SEASON, "week": n, "dry_run": dry,
                                                     "before_deadline": before, "forecasts_sha256": digest}))
        if tamper:
            (d / "forecasts.csv").write_text("player_id\ny\n")
    week(6); week(7, tamper=True); week(8, dry=True); week(9, before=False)
    valid, excluded = s.window_weeks(tmp_path)
    assert list(valid) == [6]
    assert excluded[7].startswith("forecasts differ") and "dry run" in excluded[8]
    assert excluded[9] == "written after the deadline" and excluded[13] == "not written"


def test_actual_points_score_ten_components_and_check_the_stored_total():
    st = pd.DataFrame({"player_id": ["a"], "season": [2026], "week": [6], **{c: [0.0] for c in s.COMPONENTS}})
    st.loc[0, ["passing_yards", "passing_tds", "fumbles_lost", "two_point_conversions", "receptions"]] = [250, 2, 1, 1, 3]
    st["fantasy_points"] = 10 + 8 - 2 + 2 + 3
    assert s.actual_points(st).actual.iloc[0] == pytest.approx(21.0)
    with pytest.raises(ValueError, match="beyond rounding"):
        s.actual_points(st.assign(fantasy_points=19.0))
    with pytest.raises(ValueError, match="more than one"):
        s.actual_points(pd.concat([st, st]))


def test_form_context_counts_only_earlier_games():
    stats = pd.DataFrame({"player_id": ["a", "a", "a"], "week": [3, 5, 6], "fantasy_points": [10.0, 20.0, 99.0]})
    pace = pd.DataFrame({"player_id": ["a"], "step8_pace": [12.0]})
    out = s.form_context(pd.DataFrame({"player_id": ["a", "a"], "week": [6, 1]}), stats, pace)
    assert out.games_before.tolist() == [2, 0]
    assert out.form_gap.iloc[0] == pytest.approx(15.0 - 12.0) and np.isnan(out.form_gap.iloc[1])
