"""Checks for reshaping Plan B's joint-MAE predictions into Plan A's
arm-agnostic full-PPR schema, and the population/label validation gate
before concatenating a new candidate arm."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plan_b_arm_adapter import (
    KEY,
    attach_plan_b_arm,
    reshape_plan_b_predictions,
    season_to_fold_map,
)
from src.evaluation.joint_ppr_selector import eligible_allocation_arms
from src.models.position_models import SeasonAwareTimeSeriesSplit


def test_season_to_fold_map_pins_the_real_config():
    """seasons 2006-2025, 3 test seasons, gap 0 -- the exact config
    scripts/evaluate_joint_ppr_selector.py uses by default, and what the
    real full_ppr_safe_blend_event_mass_20260924 predictions CSVs actually
    contain (verified directly against that data: fold 0/1/2 = 2023/24/25)."""
    mapping = season_to_fold_map(list(range(2006, 2026)), n_test_seasons=3, gap_seasons=0)
    assert mapping == {2023: 0, 2024: 1, 2025: 2}


@pytest.mark.parametrize("seasons,n_test,gap", [
    (list(range(2018, 2024)), 2, 1),
    (list(range(2015, 2026)), 4, 0),
])
def test_season_to_fold_map_matches_the_splitter_directly(seasons, n_test, gap):
    """Not just pinned constants -- must agree with actually running
    SeasonAwareTimeSeriesSplit for other configs too, so a future change to
    --seasons/--n-test-seasons/cv_gap_seasons can't silently desync this
    from Plan A's real fold assignment."""
    unique_seasons = np.array(sorted(set(seasons)))
    splitter = SeasonAwareTimeSeriesSplit(n_splits=n_test, seasons=unique_seasons,
                                          gap_seasons=gap, strict=True)
    expected = {}
    for fold, (_, test_idx) in enumerate(splitter.split(np.empty(len(unique_seasons)))):
        expected[int(unique_seasons[test_idx][0])] = fold
    assert season_to_fold_map(seasons, n_test, gap) == expected


def _plan_b_wide_predictions():
    rows = []
    for season in (2023, 2024):
        for week in (1, 2):
            for team in ("A", "B"):
                for pid in (f"{team}1", f"{team}2"):
                    rows.append({"player_id": pid, "season": season, "week": week, "team": team,
                                "position": "RB", "slot": "RB1", "is_cold_start": 0,
                                "train_end_season": season - 1,
                                "actual_share": 0.4, "rolling3": 0.38,
                                "normalized_prior": 0.39, "joint_other_mae": 0.41,
                                "predicted_other_mass": 0.1})
    return pd.DataFrame(rows)


def test_reshape_produces_the_exact_long_schema():
    wide = _plan_b_wide_predictions()
    fold_map = {2023: 0, 2024: 1}
    reshaped = reshape_plan_b_predictions(wide, "joint_other_mae", fold_map)
    assert list(reshaped.columns) == [
        "player_id", "season", "week", "team", "position", "is_cold_start",
        "fold", "arm", "actual_share", "predicted_share",
    ]
    assert (reshaped.arm == "joint_other_mae").all()
    assert (reshaped.loc[reshaped.season.eq(2023), "fold"] == 0).all()
    assert (reshaped.loc[reshaped.season.eq(2024), "fold"] == 1).all()
    np.testing.assert_allclose(reshaped.predicted_share, wide.joint_other_mae)


def test_reshape_raises_on_a_season_missing_from_the_fold_map():
    wide = _plan_b_wide_predictions()
    with pytest.raises(ValueError, match="have no fold"):
        reshape_plan_b_predictions(wide, "joint_other_mae", {2023: 0})  # 2024 missing


def test_reshape_raises_on_missing_required_column():
    wide = _plan_b_wide_predictions().drop(columns=["is_cold_start"])
    with pytest.raises(ValueError, match="missing required columns"):
        reshape_plan_b_predictions(wide, "joint_other_mae", {2023: 0, 2024: 1})


def _allocation_result_and_truth():
    """A minimal synthetic allocation_result (existing rolling3 arm) and
    matching truth frame for two folds, mirroring ppr_truth.py's KEY shape."""
    key_rows = []
    for season, fold in ((2023, 0), (2024, 1)):
        for week in (1, 2):
            for team in ("A", "B"):
                for pid in (f"{team}1", f"{team}2"):
                    key_rows.append({"player_id": pid, "season": season, "week": week,
                                     "team": team, "position": "RB", "fold": fold})
    keys = pd.DataFrame(key_rows)
    rolling3 = keys.copy()
    rolling3["arm"] = "rolling3"
    rolling3["actual_share"] = 0.4
    rolling3["predicted_share"] = 0.38
    truth = keys.copy()
    truth["rushing_yards"] = 20.0
    allocation_result = {"predictions": rolling3}
    return allocation_result, truth


def test_attach_plan_b_arm_succeeds_and_is_consumable_by_the_real_selector():
    allocation_result, truth = _allocation_result_and_truth()
    wide = _plan_b_wide_predictions()
    fold_map = {2023: 0, 2024: 1}
    reshaped = reshape_plan_b_predictions(wide, "joint_other_mae", fold_map)
    updated = attach_plan_b_arm(allocation_result, reshaped, "rushing_yards", truth, folds={0, 1})
    assert set(updated["predictions"].arm.unique()) == {"rolling3", "joint_other_mae"}
    # Must be directly usable by joint_ppr_selector's real gating logic, unmodified.
    allowed, decisions = eligible_allocation_arms(updated["predictions"], prior_folds={0})
    assert "joint_other_mae" in decisions


def test_attach_plan_b_arm_rejects_a_population_mismatch():
    allocation_result, truth = _allocation_result_and_truth()
    wide = _plan_b_wide_predictions()
    reshaped = reshape_plan_b_predictions(wide, "joint_other_mae", {2023: 0, 2024: 1})
    dropped = reshaped.iloc[1:].reset_index(drop=True)  # missing one row vs. truth's population
    with pytest.raises(ValueError, match="population differs"):
        attach_plan_b_arm(allocation_result, dropped, "rushing_yards", truth, folds={0, 1})


def test_attach_plan_b_arm_rejects_a_label_mismatch():
    allocation_result, truth = _allocation_result_and_truth()
    wide = _plan_b_wide_predictions()
    reshaped = reshape_plan_b_predictions(wide, "joint_other_mae", {2023: 0, 2024: 1})
    reshaped.loc[0, "actual_share"] = 0.99  # disagrees with the existing rolling3 row's label
    with pytest.raises(ValueError, match="actual_share disagrees"):
        attach_plan_b_arm(allocation_result, reshaped, "rushing_yards", truth, folds={0, 1})
