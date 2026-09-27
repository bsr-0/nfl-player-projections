"""Checks for the smooth-absolute-error Plan B joint allocation arm."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plan_b_joint_mae_backtester import run_backtest
from src.models.team_hierarchical.joint_other_mae import JointOtherMAEModel


def _panel():
    rows = []
    for season in range(2018, 2024):
        for week in range(1, 5):
            for team in ("A", "B"):
                for rank, position in enumerate(("RB", "WR", "TE"), 1):
                    actual = (0.30, 0.35, 0.00)[rank - 1]
                    rows.append({"player_id": f"{team}{rank}", "team": team,
                                 "position": position, "season": season, "week": week,
                                 "slot": f"{position}1", "depth_chart_rank": rank,
                                 "roster_snap_share_s2d": 0.6 - rank * 0.1,
                                 "is_cold_start": int(week == 1),
                                 "share_of_team_targets": actual,
                                 "share_of_team_targets_roll3": (0.28, 0.34, 0.01)[rank - 1],
                                 "share_of_team_targets_s2d": (0.29, 0.33, 0.01)[rank - 1],
                                 "team_targets": 30.0,
                                 "team_targets_roll3": 29.0,
                                 "team_targets_s2d": 28.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("other_weight", [0.0, 0.3, 1.0, 2.0])
def test_analytic_gradient_matches_central_difference(other_weight):
    panel = _panel()
    model = JointOtherMAEModel("targets", other_weight=other_weight)
    prepared = model._prepare(panel, fit=True)
    y = panel[model.label].to_numpy(float)
    group, n_groups = model._groups(panel)
    other_y = 1 - np.bincount(group, weights=y, minlength=n_groups)
    rng = np.random.default_rng(13)
    coef = rng.normal(0, 0.15, prepared[1].shape[1] + prepared[2].shape[1])
    loss, gradient = model._objective(prepared, y, other_y, coef)
    assert np.isfinite(loss)
    numerical = np.zeros_like(coef)
    for i in range(len(coef)):
        shifted = coef.copy()
        shifted[i] += 1e-6
        higher = model._objective(prepared, y, other_y, shifted)[0]
        shifted[i] -= 2e-6
        lower = model._objective(prepared, y, other_y, shifted)[0]
        numerical[i] = (higher - lower) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)


@pytest.mark.parametrize("other_weight", [-1.0, float("nan"), float("inf")])
def test_bad_other_weight_fails(other_weight):
    with pytest.raises(ValueError):
        JointOtherMAEModel("targets", other_weight=other_weight)


def test_other_weight_scales_the_other_bucket_loss_term():
    """The other-bucket smooth-MAE term must scale exactly by other_weight, holding
    predictions fixed -- the player-row term must not change with other_weight at all."""
    panel = _panel()
    baseline = JointOtherMAEModel("targets", other_weight=1.0, penalty=0.0)
    prepared = baseline._prepare(panel, fit=True)
    y = panel[baseline.label].to_numpy(float)
    group, n_groups = baseline._groups(panel)
    other_y = 1 - np.bincount(group, weights=y, minlength=n_groups)
    rng = np.random.default_rng(7)
    coef = rng.normal(0, 0.15, prepared[1].shape[1] + prepared[2].shape[1])
    player_p, other_p, _, _ = baseline._predict_components(prepared, coef)
    player_term = float(np.sqrt((player_p - y) ** 2 + baseline.smooth_width ** 2).mean()
                        - baseline.smooth_width)
    other_term = float(np.sqrt((other_p - other_y) ** 2 + baseline.smooth_width ** 2).mean()
                       - baseline.smooth_width)
    for other_weight in (0.0, 0.3, 2.0):
        weighted = JointOtherMAEModel("targets", other_weight=other_weight, penalty=0.0)
        prepared_w = weighted._prepare(panel, fit=True)
        loss, _ = weighted._objective(prepared_w, y, other_y, coef)
        expected = player_term + other_weight * other_term
        np.testing.assert_allclose(loss, expected, atol=1e-12)


def test_higher_other_weight_improves_mass_calibration():
    """Raising other_weight should pull the predicted omitted-bucket mass closer to actual."""
    panel = _panel()
    train, test = panel[panel.season < 2023], panel[panel.season == 2023]
    group, n_groups = JointOtherMAEModel._groups(test)
    actual_other = 1 - np.bincount(
        group, weights=test[JointOtherMAEModel("targets").label].to_numpy(float), minlength=n_groups)
    errors = {}
    for other_weight in (0.0, 1.0):
        model = JointOtherMAEModel("targets", other_weight=other_weight).fit(train)
        _, _, other = model.predict(test)
        errors[other_weight] = float(np.abs(other - actual_other).mean())
    assert errors[1.0] < errors[0.0]


def test_mass_conservation_and_zero_player_rows():
    panel = _panel()
    model = JointOtherMAEModel("targets").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023]
    player, _, other = model.predict(test)
    group, n_groups = model._groups(test)
    np.testing.assert_allclose(np.bincount(group, weights=player, minlength=n_groups) + other, 1)
    assert model.fit_diagnostics_["converged"]
    assert (player >= 0).all() and (player <= 1).all()


def test_future_labels_cannot_change_earlier_fold():
    panel = _panel()
    before, _ = run_backtest(panel, "targets", [2022, 2023], n_bootstrap=100)
    changed = panel.copy()
    changed.loc[changed.season == 2023, "share_of_team_targets"] = 0.1
    after, _ = run_backtest(changed, "targets", [2022, 2023], n_bootstrap=100)
    early = before.season.eq(2022)
    for arm in ("rolling3", "normalized_prior", "joint_other_mae"):
        np.testing.assert_allclose(before.loc[early, arm], after.loc[early, arm], atol=1e-12)


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_bad_smoothing_width_fails(width):
    with pytest.raises(ValueError):
        JointOtherMAEModel("targets", smooth_width=width)


def _qb_only_panel():
    """passing_yards is QB-only (src/models/team_allocation/features.py's
    TARGET_POPULATIONS) -- exercise that population end to end through the
    model, not just through filter_population, which the model itself never
    calls (it just fits whatever frame it's given)."""
    rows = []
    for season in range(2018, 2024):
        for week in range(1, 5):
            for team in ("A", "B"):
                for rank, actual in ((1, 0.85), (2, 0.05)):
                    rows.append({"player_id": f"{team}_qb{rank}", "team": team,
                                 "position": "QB", "season": season, "week": week,
                                 "slot": f"QB{rank}", "depth_chart_rank": rank,
                                 "roster_snap_share_s2d": 0.9 - rank * 0.3,
                                 "is_cold_start": int(week == 1),
                                 "share_of_team_passing_yards": actual,
                                 "share_of_team_passing_yards_roll3": (0.80, 0.06)[rank - 1],
                                 "share_of_team_passing_yards_s2d": (0.82, 0.05)[rank - 1],
                                 "team_passing_yards": 260.0,
                                 "team_passing_yards_roll3": 255.0,
                                 "team_passing_yards_s2d": 250.0})
    return pd.DataFrame(rows)


def _receptions_panel():
    """receptions is QB-excluded, like targets/receiving_yards -- reuse the
    same RB/WR/TE panel shape as _panel(), just for the other new target."""
    rows = []
    for season in range(2018, 2024):
        for week in range(1, 5):
            for team in ("A", "B"):
                for rank, position in enumerate(("RB", "WR", "TE"), 1):
                    actual = (0.15, 0.55, 0.05)[rank - 1]
                    rows.append({"player_id": f"{team}{rank}", "team": team,
                                 "position": position, "season": season, "week": week,
                                 "slot": f"{position}1", "depth_chart_rank": rank,
                                 "roster_snap_share_s2d": 0.6 - rank * 0.1,
                                 "is_cold_start": int(week == 1),
                                 "share_of_team_receptions": actual,
                                 "share_of_team_receptions_roll3": (0.14, 0.53, 0.06)[rank - 1],
                                 "share_of_team_receptions_s2d": (0.16, 0.54, 0.05)[rank - 1],
                                 "team_receptions": 24.0,
                                 "team_receptions_roll3": 23.0,
                                 "team_receptions_s2d": 22.0})
    return pd.DataFrame(rows)


def test_receptions_end_to_end():
    """Regression test for extending Plan B's joint architecture to the
    second new target, receptions (QB-excluded, like targets/receiving_yards)."""
    panel = _receptions_panel()
    model = JointOtherMAEModel("receptions").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023]
    player, prior, other = model.predict(test)
    group, n_groups = model._groups(test)
    np.testing.assert_allclose(np.bincount(group, weights=player, minlength=n_groups) + other, 1)
    assert model.fit_diagnostics_["converged"]
    assert np.isfinite(player).all() and (player >= 0).all() and (player <= 1).all()


def test_passing_yards_qb_only_population_end_to_end():
    """Regression test for extending Plan B's joint architecture beyond the
    original 4 VOLUME_COLS (2026-09-25) -- passing_yards is architecturally
    identical in shape but exercises the QB-only population path, never
    covered by any prior Plan B test."""
    panel = _qb_only_panel()
    model = JointOtherMAEModel("passing_yards").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023]
    player, prior, other = model.predict(test)
    group, n_groups = model._groups(test)
    np.testing.assert_allclose(np.bincount(group, weights=player, minlength=n_groups) + other, 1)
    assert model.fit_diagnostics_["converged"]
    assert np.isfinite(player).all() and (player >= 0).all() and (player <= 1).all()
