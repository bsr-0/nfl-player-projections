"""Synthetic rows with known verdicts: each gate must pass a clean candidate and fail its own regression."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.selection_gates import evaluate, holm

B = 400


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    counts = {"QB": 40, "RB": 80, "WR": 120, "TE": 60}
    players = pd.DataFrame([(f"{p}{i:03d}", p) for p, n in counts.items() for i in range(n)],
                           columns=["player_id", "position"])
    players["skill"] = rng.gamma(3.0, 3.0, len(players))
    rows = players.merge(pd.DataFrame({"week": range(1, 18)}), how="cross")
    rows["season"] = 2025
    n = len(rows)
    skill = rows.skill.to_numpy()
    rows["actual"] = skill + rng.normal(0, 5, n)
    rows["rolling3"] = skill + rng.normal(0, 3, n)
    rows["incumbent"] = skill + rng.normal(0, 3, n)
    good = skill + rng.normal(0, 1.5, n)
    rows["better"] = good
    rows["equal"] = skill + rng.normal(0, 3, n)
    rows["qb_worse"] = np.where(rows.position.eq("QB"), skill + rng.normal(0, 9, n), good)
    rows["biased"] = good + 3.0
    rows["heavy_tail"] = good + np.where(rng.random(n) < 0.02, 40.0, 0.0)
    rows["scrambled"] = skill + rng.normal(0, 6, n)
    # Top-12 by rolling-3 within position-week: worse there only.
    rank = rows.groupby(["week", "position"]).rolling3.rank(ascending=False, method="first")
    rows["top_worse"] = np.where(rank <= 12, skill + rng.normal(0, 9, n), good)
    # Exactly 0.07 closer on almost every row: significant but below the 0.10 minimum effect.
    eb = rows.incumbent - rows.actual
    rows["tiny_gain"] = rows.incumbent - np.sign(eb) * np.minimum(eb.abs(), 0.07)
    rows["good_lo"], rows["good_hi"] = good - 6.4, good + 6.4
    rows["narrow_lo"], rows["narrow_hi"] = good - 1.0, good + 1.0
    return rows


ALL = ["better", "equal", "qb_worse", "biased", "heavy_tail", "scrambled", "top_worse", "tiny_gain"]


@pytest.fixture(scope="module")
def report(frame):
    return evaluate(frame, ALL, B=B, seed=1,
                    intervals={"better": ("good_lo", "good_hi"), "biased": ("narrow_lo", "narrow_hi")})


def test_clean_candidate_passes_every_gate(report):
    r = report["candidates"]["better"]
    assert r["G1"]["passes"] and r["G9"]["passes"] and r["G7"]["passes"]
    assert r["guardrail_failures"] == [] and r["stage_passes"]


def test_g1_point_matches_direct_mae(frame, report):
    r = report["candidates"]["better"]
    direct = (frame.better - frame.actual).abs().mean() - (frame.incumbent - frame.actual).abs().mean()
    assert r["G1"]["point_difference"] == pytest.approx(direct, abs=1e-12)
    assert r["G1"]["upper_bound_95"] < 0


def test_equal_candidate_fails_g1(report):
    r = report["candidates"]["equal"]
    assert not r["G1"]["holm_rejects"] and not r["stage_passes"]


def test_minimum_effect_blocks_a_significant_but_small_gain(report):
    r = report["candidates"]["tiny_gain"]
    assert r["G1"]["holm_rejects"]
    assert -r["G1"]["point_difference"] == pytest.approx(0.07, abs=0.005)
    assert not r["G1"]["passes"]


@pytest.mark.parametrize("candidate,cell", [
    ("qb_worse", "G3 MAE QB"),
    ("top_worse", "G4 MAE tier_rolling3 1-12"),
    ("biased", "G5 |bias| overall"),
    ("heavy_tail", "G2 RMSE"),
    ("scrambled", "G6 Spearman"),
])
def test_each_guardrail_fails_its_regression(report, candidate, cell):
    r = report["candidates"][candidate]
    assert cell in r["guardrail_failures"]
    assert not r["guardrails_pass"] and not r["stage_passes"]


def test_g9_and_g7_fail(report):
    assert not report["candidates"]["scrambled"]["G9"]["passes"]
    assert report["candidates"]["biased"]["G7"]["passes"] is False


def test_holm_step_down():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.2, "b": 0.01}) == {"a": False, "b": True}


def test_same_seed_is_deterministic(frame):
    a = evaluate(frame, ["better"], B=200, seed=3)
    b = evaluate(frame, ["better"], B=200, seed=3)
    assert a["candidates"]["better"]["G1"] == b["candidates"]["better"]["G1"]


def test_missing_prediction_or_duplicate_key_fails_loudly(frame):
    gap = frame.copy()
    gap.loc[5, "better"] = np.nan
    with pytest.raises(ValueError, match="every model must predict"):
        evaluate(gap, ["better"], B=50)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(pd.concat([frame, frame.iloc[:1]]), ["better"], B=50)


def test_partial_holm_family_is_refused(frame):
    with pytest.raises(ValueError, match="Holm"):
        evaluate(frame, ["better"], B=50, holm_family=["better", "equal"])


def test_g1_bound_covers_a_known_true_difference():
    """Errors are e_model - e_actual with normal noise, so the true MAE gap is analytic."""
    sd_y, sd_inc, sd_cand = 5.0, 3.0, 2.5
    true = np.sqrt(2 / np.pi) * (np.hypot(sd_cand, sd_y) - np.hypot(sd_inc, sd_y))
    rng = np.random.default_rng(11)
    covered, recovered, reps = 0, [], 60
    for rep in range(reps):
        rows = pd.DataFrame([(f"P{i:03d}", ("QB", "RB", "WR", "TE")[i % 4]) for i in range(160)],
                            columns=["player_id", "position"])
        rows = rows.merge(pd.DataFrame({"week": range(1, 18)}), how="cross").assign(season=2025)
        n = len(rows)
        skill = np.repeat(rng.gamma(3.0, 3.0, 160), 17)
        rows["actual"] = skill + rng.normal(0, sd_y, n)
        rows["incumbent"] = skill + rng.normal(0, sd_inc, n)
        rows["rolling3"] = skill + rng.normal(0, 3, n)
        rows["cand"] = skill + rng.normal(0, sd_cand, n)
        g1 = evaluate(rows, ["cand"], B=300, seed=rep)["candidates"]["cand"]["G1"]
        covered += true <= g1["upper_bound_95"]
        recovered.append(g1["point_difference"])
    assert np.mean(recovered) == pytest.approx(true, abs=0.02)
    assert covered / reps >= 0.90


def test_empty_guardrail_cell_fails_loudly(frame):
    with pytest.raises(ValueError, match="no rows"):
        evaluate(frame[frame.position != "TE"], ["better"], B=50)
