"""Correctness for src/models/team_hierarchical/models.py's MixedEffectsShareModel.

Verifies the design correction documented in that module: predict() must
actually apply a known player's random effect on top of the fixed-effects
prediction (statsmodels' own .predict() does not, by default -- see the
module docstring), a genuinely new player must fall back to fixed-effects-
only, an unseen slot must fall back sensibly rather than crash, and a
degenerate fold must fail unless a fallback was explicitly requested.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.team_hierarchical.models import MixedEffectsFitError, MixedEffectsShareModel


def _multi_team_data(n_teams=6, n_weeks=12, seed=0):
    """Several players share each slot label (one per team), so `slot`
    alone does not fully determine player identity -- the scenario the
    random effect is actually meant to explain. Each player has a
    persistent bias; the fixed effect should capture the slot-level
    population average, the random effect each player's deviation from it.
    """
    rng = np.random.RandomState(seed)
    rows = []
    player_bias = {}
    for team_i in range(n_teams):
        for slot in ("RB1", "RB2"):
            pid = f"team{team_i}_{slot}"
            player_bias[pid] = rng.normal(0, 0.08)
            for week in range(1, n_weeks + 1):
                x = rng.normal()
                base = 0.35 if slot == "RB1" else 0.15
                share = base + 0.05 * x + player_bias[pid] + rng.normal(scale=0.01)
                rows.append({"player_id": pid, "slot": slot, "x": x, "week": week, "share": share})
    return pd.DataFrame(rows), player_bias


def _train_test_split(df, train_weeks, test_weeks):
    train = df[df["week"].isin(train_weeks)]
    test = df[df["week"].isin(test_weeks)]
    return train, test


def test_fit_converges_on_reasonable_synthetic_data():
    df, _ = _multi_team_data()
    train, _ = _train_test_split(df, range(1, 10), range(10, 13))
    model = MixedEffectsShareModel(target_col="share").fit(
        train[["player_id", "slot", "x"]], train["share"].to_numpy()
    )
    assert model.converged_ is True
    assert model.result_ is not None


def test_known_player_random_effect_beats_fixed_effects_only():
    """A known player's random-effect-adjusted prediction must be closer to
    their true persistent bias than the fixed-effects-only prediction --
    proving predict() actually applies the random effect (statsmodels'
    raw .predict() does not, by default)."""
    df, player_bias = _multi_team_data()
    train, test = _train_test_split(df, range(1, 10), range(10, 13))
    model = MixedEffectsShareModel(target_col="share").fit(
        train[["player_id", "slot", "x"]], train["share"].to_numpy()
    )

    with_re = model.predict(test[["player_id", "slot", "x"]])
    fixed_only = model.result_.predict(
        exog=test.assign(slot=pd.Categorical(test["slot"], categories=model._slot_categories))
    ).to_numpy()

    mae_with_re = np.abs(with_re - test["share"].to_numpy()).mean()
    mae_fixed_only = np.abs(fixed_only - test["share"].to_numpy()).mean()
    assert mae_with_re < mae_fixed_only


def test_cold_start_player_falls_back_to_fixed_effects_only():
    df, _ = _multi_team_data()
    train, _ = _train_test_split(df, range(1, 10), range(10, 13))
    model = MixedEffectsShareModel(target_col="share").fit(
        train[["player_id", "slot", "x"]], train["share"].to_numpy()
    )

    new_player = pd.DataFrame({"player_id": ["never_seen"], "slot": ["RB1"], "x": [0.0]})
    pred = model.predict(new_player)
    fixed_only = model.result_.predict(
        exog=new_player.assign(slot=pd.Categorical(new_player["slot"], categories=model._slot_categories))
    ).to_numpy()
    np.testing.assert_allclose(pred, fixed_only)


def test_unseen_slot_falls_back_to_rank1_same_position():
    df, _ = _multi_team_data()  # only RB1/RB2 exist in training
    train, _ = _train_test_split(df, range(1, 10), range(10, 13))
    model = MixedEffectsShareModel(target_col="share").fit(
        train[["player_id", "slot", "x"]], train["share"].to_numpy()
    )

    unseen = pd.DataFrame({"player_id": ["team0_RB1"], "slot": ["RB4"], "x": [0.0]})
    rb1_equiv = pd.DataFrame({"player_id": ["team0_RB1"], "slot": ["RB1"], "x": [0.0]})
    pred_unseen = model.predict(unseen)
    pred_rb1 = model.predict(rb1_equiv)
    np.testing.assert_allclose(pred_unseen, pred_rb1)


def test_unseen_slot_with_no_same_position_fallback_uses_constant():
    df, _ = _multi_team_data()  # no WR/TE at all in training
    train, _ = _train_test_split(df, range(1, 10), range(10, 13))
    model = MixedEffectsShareModel(target_col="share").fit(
        train[["player_id", "slot", "x"]], train["share"].to_numpy()
    )
    unseen = pd.DataFrame({"player_id": ["nobody"], "slot": ["WR3"], "x": [0.0]})
    pred = model.predict(unseen)
    assert pred[0] == pytest.approx(model._fallback_mean_)


def test_degenerate_fold_falls_back_to_constant_without_crashing():
    tiny = pd.DataFrame({
        "player_id": ["a"], "slot": ["RB1"], "x": [0.0],
    })
    model = MixedEffectsShareModel(target_col="share", failure_policy="mean").fit(tiny, np.array([0.4]))
    assert model.converged_ is False
    assert model.result_ is None
    pred = model.predict(pd.DataFrame({"player_id": ["a", "b"], "slot": ["RB1", "RB2"], "x": [0.0, 0.0]}))
    np.testing.assert_allclose(pred, [0.4, 0.4])
    assert model.fit_diagnostics_["fallback"] is True


def test_degenerate_fold_fails_by_default():
    model = MixedEffectsShareModel()
    with pytest.raises(MixedEffectsFitError, match="insufficient"):
        model.fit(pd.DataFrame({"player_id": ["a"], "slot": ["RB1"]}), np.array([0.4]))
    with pytest.raises(MixedEffectsFitError, match="previous fit failed"):
        model.predict(pd.DataFrame({"player_id": ["a"], "slot": ["RB1"]}))


def test_training_only_imputation_and_redundant_slot_rank():
    df, _ = _multi_team_data()
    X = df[["player_id", "slot", "x"]].copy()
    X.loc[::5, "x"] = np.nan
    X["slot_rank"] = X.slot.str[-1].astype(int)
    X["all_missing"] = np.nan
    model = MixedEffectsShareModel().fit(X, df.share.to_numpy())
    assert model.fit_diagnostics_["used_training_rows"] == len(X)
    assert {"slot_rank", "all_missing"} <= set(model.fit_diagnostics_["dropped_redundant_columns"])
    median = model._medians.copy()
    test = X.iloc[:3].copy()
    test.index = [0, 0, 0]
    test["x"] = [np.nan, 100, np.nan]
    pred = model.predict(test)
    imputed = test.fillna(median)
    np.testing.assert_allclose(pred, model.predict(imputed))
    pd.testing.assert_series_equal(median, model._medians)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_labels_rejected(bad):
    X = pd.DataFrame({"player_id": ["a", "b"], "slot": ["RB1", "RB2"]})
    with pytest.raises(ValueError, match="labels must be finite"):
        MixedEffectsShareModel().fit(X, np.array([bad, 0.1]))


def test_numeric_optimizer_failure_is_visible(monkeypatch):
    class FailingFit:
        def fit(self, **kwargs):
            raise np.linalg.LinAlgError("synthetic singular covariance")
    monkeypatch.setattr("src.models.team_hierarchical.models.smf.mixedlm", lambda *a, **kw: FailingFit())
    df, _ = _multi_team_data()
    model = MixedEffectsShareModel()
    with pytest.raises(MixedEffectsFitError, match="all mixed-effects optimizer"):
        model.fit(df[["player_id", "slot", "x"]], df.share.to_numpy())
    assert len(model.fit_diagnostics_["attempts"]) == 2
    assert all("synthetic singular" in a["error"] for a in model.fit_diagnostics_["attempts"])


def test_fit_requires_player_id_and_slot_columns():
    with pytest.raises(ValueError, match="player_id"):
        MixedEffectsShareModel(target_col="share").fit(
            pd.DataFrame({"x": [0.1, 0.2]}), np.array([0.1, 0.2])
        )


def test_constant_fixed_effect_column_is_dropped_not_fatal():
    """A numeric fixed-effect column with zero variance in a fold (e.g.
    every training row cold-start, is_cold_start always 1) would make the
    design matrix singular -- must be dropped, not crash the fit."""
    df, _ = _multi_team_data()
    train, _ = _train_test_split(df, range(1, 10), range(10, 13))
    train = train[["player_id", "slot", "x"]].copy()
    train["always_one"] = 1.0
    model = MixedEffectsShareModel(target_col="share").fit(train, df.loc[train.index, "share"].to_numpy())
    assert "always_one" not in model._fixed_effect_cols
    assert model.converged_ is True
