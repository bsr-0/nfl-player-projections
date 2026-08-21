"""E[PPR/game | played]: target definition, weighting, and leakage.

The target is each player's OBSERVED season rate. Games played is an
observation WEIGHT, never part of constructing the target and never a
feature -- target-season games_played is not knowable pre-season, so
admitting it as an input would be direct leakage.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.single_week_ppr.season_conditional_production import (
    TARGET, attach_conditional_target, decompose_errors, feature_columns,
    fit_conditional_production, predict_conditional_production,
)


def _panel():
    return pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["RB"] * 3,
        "season": [2022] * 3,
        "games_played": [3, 17, 10],
        "ppr": [30.0, 170.0, 60.0],
        "ppr_per_game": [10.0, 10.0, 6.0],
        "possible_games": [17, 17, 17],
    })


def _pairs():
    return pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["RB"] * 3,
        "target_season": [2022] * 3,
        "ppg_y1": [9.0, 10.5, 6.5],
        "games_played_y1": [12.0, 16.0, 11.0],
        "years_of_history": [2, 5, 3],
    })


# --- target definition ----------------------------------------------------

def test_target_is_the_observed_season_rate():
    out = attach_conditional_target(_pairs(), _panel())
    assert out[TARGET].tolist() == [10.0, 10.0, 6.0]


def test_target_comes_from_the_contract_panel_not_a_raw_row_count():
    """A 3-game and a 17-game season at the same rate are the SAME target
    value; they differ only in how much they are trusted."""
    out = attach_conditional_target(_pairs(), _panel())
    a = out[out.player_id == "a"].iloc[0]
    b = out[out.player_id == "b"].iloc[0]
    assert a[TARGET] == b[TARGET] == 10.0
    assert a["games_played"] != b["games_played"]


def test_rows_without_a_contract_season_are_dropped():
    pairs = _pairs()
    pairs.loc[len(pairs)] = {"player_id": "ghost", "position": "RB",
                             "target_season": 2022, "ppg_y1": 5.0,
                             "games_played_y1": 8.0, "years_of_history": 1}
    out = attach_conditional_target(pairs, _panel())
    assert "ghost" not in set(out["player_id"])


# --- leakage --------------------------------------------------------------

def test_target_season_games_played_is_not_a_feature():
    """Target-season games are unknowable pre-season. Prior-season games
    (games_played_y1) ARE legitimate and must survive."""
    out = attach_conditional_target(_pairs(), _panel())
    feats = feature_columns(out)
    assert "games_played" not in feats
    assert "ppr" not in feats
    assert TARGET not in feats
    assert "games_played_y1" in feats, "prior-season games are legitimate features"


def test_no_target_season_exposure_column_is_ever_a_feature():
    """THE CONTRACT: target-season exposure may set weights and the target,
    never an input. Checked against the panel schema, so adding a panel
    column cannot silently open a leak."""
    from src.models.single_week_ppr.season_conditional_production import TARGET_SEASON_COLUMNS
    out = attach_conditional_target(_pairs(), _panel())
    feats = set(feature_columns(out))
    assert feats.isdisjoint(TARGET_SEASON_COLUMNS)


def test_prior_season_exposure_lags_survive_the_filter():
    """The other half of the contract: a filter broad enough to catch
    'games_played' by name would destroy legitimate prior-season signal."""
    out = attach_conditional_target(_pairs(), _panel())
    out["games_played_y2"] = 14.0
    feats = set(feature_columns(out))
    assert {"games_played_y1", "games_played_y2"} <= feats


def test_prediction_is_invariant_to_every_target_season_exposure_column():
    """Behavioural form of the contract, generalised beyond games_played.
    Permanent regression guard: this is exactly the leak a future
    feature-selection refactor could reopen."""
    from src.models.single_week_ppr.season_conditional_production import TARGET_SEASON_COLUMNS
    train = pd.concat([attach_conditional_target(_pairs(), _panel())] * 12, ignore_index=True)
    feats = feature_columns(train)
    model = fit_conditional_production(train, feats)
    test = attach_conditional_target(_pairs(), _panel())
    baseline = predict_conditional_production(model, test, feats).tolist()
    for col in TARGET_SEASON_COLUMNS:
        tampered = test.copy()
        tampered[col] = 999.0
        assert predict_conditional_production(model, tampered, feats).tolist() == baseline, \
            f"prediction moved when target-season {col} changed"


def test_prediction_does_not_depend_on_target_season_games():
    """Weighting is a training-time choice only. Changing a test row's
    games_played must not move its prediction."""
    train = attach_conditional_target(_pairs(), _panel())
    train = pd.concat([train] * 12, ignore_index=True)
    feats = feature_columns(train)
    model = fit_conditional_production(train, feats)

    test = attach_conditional_target(_pairs(), _panel())
    baseline = predict_conditional_production(model, test, feats)
    tampered = test.copy()
    tampered["games_played"] = 1
    assert predict_conditional_production(model, tampered, feats).tolist() == baseline.tolist()


def test_weighting_shifts_the_fit_toward_high_game_seasons():
    rng = np.random.RandomState(0)
    n = 120
    heavy = pd.DataFrame({"x": 1.0, "ppr_per_game": 10.0, "games_played": 17.0}, index=range(n))
    light = pd.DataFrame({"x": 1.0, "ppr_per_game": 2.0, "games_played": 1.0}, index=range(n))
    train = pd.concat([heavy, light], ignore_index=True)
    weighted = fit_conditional_production(train, ["x"]).predict(train[["x"]]).mean()
    from src.models.single_week_ppr.architectures import GBMRegressor
    unweighted = GBMRegressor(objective="regression").fit(
        train[["x"]], train["ppr_per_game"]).predict(train[["x"]]).mean()
    assert weighted > unweighted


# --- prediction bounds ----------------------------------------------------

def test_predicted_rate_is_floored_at_zero():
    train = pd.DataFrame({"x": [1.0, 2.0] * 30, "ppr_per_game": [-5.0, -6.0] * 30,
                          "games_played": [10.0] * 60})
    model = fit_conditional_production(train, ["x"])
    assert (predict_conditional_production(model, train, ["x"]) >= 0).all()


# --- decomposition --------------------------------------------------------

def test_decomposition_sums_exactly_to_the_season_bias():
    rng = np.random.RandomState(1)
    n = 200
    ga = pd.Series(rng.uniform(1, 17, n))
    ra = pd.Series(rng.uniform(0, 20, n))
    gp = ga + rng.normal(0, 2, n)
    rp = ra + rng.normal(0, 3, n)
    d = decompose_errors(gp, ga, rp, ra, ga * ra)
    total = d["contrib_games"] + d["contrib_rate"] + d["contrib_interaction"]
    assert total == pytest.approx(float((gp * rp - ga * ra).mean()), abs=1e-9)


def test_decomposition_isolates_an_exposure_failure():
    ga, ra = pd.Series([16.0] * 50), pd.Series([10.0] * 50)
    gp, rp = pd.Series([10.0] * 50), pd.Series([10.0] * 50)   # games wrong, rate right
    d = decompose_errors(gp, ga, rp, ra, ga * ra)
    assert d["contrib_rate"] == pytest.approx(0.0)
    assert d["contrib_games"] < -50


def test_decomposition_isolates_a_production_failure():
    ga, ra = pd.Series([16.0] * 50), pd.Series([10.0] * 50)
    gp, rp = pd.Series([16.0] * 50), pd.Series([6.0] * 50)    # games right, rate wrong
    d = decompose_errors(gp, ga, rp, ra, ga * ra)
    assert d["contrib_games"] == pytest.approx(0.0)
    assert d["contrib_rate"] < -50
