import numpy as np
import pandas as pd

from src.models.team_allocation.hierarchical import (
    EmpiricalBayesShareModel,
    _fit_event_gate,
    _team_event_probabilities,
    _eligibility_mask,
    add_role_tier,
    hierarchical_feature_columns,
    predict_residual_over_opportunity,
    predict_multinomial_count_allocation,
)


def _rows(n_weeks=8):
    rows = []
    for week in range(1, n_weeks + 1):
        for pid, snap, target in (("a", 0.9, 1), ("b", 0.4, 0)):
            rows.append({
                "player_id": pid, "position": "RB", "team": "A", "season": 2024,
                "week": week, "snap_share_roll3": snap,
                "share_of_team_targets_roll3": snap,
                "is_cold_start": 0, "team_rushing_tds": 1,
                "team_rushing_tds_roll3": 1, "team_rushing_tds_s2d": 1,
                "share_of_team_rushing_tds": target,
                "redzone_targets_roll3": snap,
            })
    return pd.DataFrame(rows)


def test_role_tier_uses_lagged_snap_history():
    df = _rows(1)
    out = add_role_tier(df)
    assert out.loc[out.player_id.eq("a"), "role_tier"].iloc[0] == "RB1"
    assert out.loc[out.player_id.eq("b"), "role_tier"].iloc[0] == "RB2"


def test_hierarchical_features_are_lagged_only():
    cols = hierarchical_feature_columns(_rows(), "receiving_tds")
    assert "redzone_targets_roll3" in cols
    assert "share_of_team_rushing_tds" not in cols


def test_event_feature_families_are_explicit_and_lagged():
    df = _rows()
    df["team_redzone_targets_roll3"] = 2.0
    player_only = hierarchical_feature_columns(
        df, "receiving_tds", families={"player_event_role"}
    )
    team_only = hierarchical_feature_columns(
        df, "receiving_tds", families={"team_event_context"}
    )
    assert "redzone_targets_roll3" in player_only
    assert "team_redzone_targets_roll3" not in player_only
    assert "team_redzone_targets_roll3" in team_only
    assert all(col == "is_cold_start" or col.endswith(("_roll3", "_s2d")) for col in team_only)


def test_event_gate_is_temporally_calibrated_and_usable():
    df = pd.concat([_rows(8), _rows(8).assign(season=2025)], ignore_index=True)
    df.loc[df.week % 2 == 0, "team_rushing_tds"] = 0
    fitted, threshold = _fit_event_gate(df, "rushing_tds")
    assert fitted is not None
    assert 0.2 <= threshold <= 0.8
    _, _, calibration = fitted
    assert 0.0 <= calibration["weight"] <= 1.0
    probability = _team_event_probabilities(fitted, df, "rushing_tds")
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_empirical_bayes_player_effects_shrink_and_carry_forward():
    train = _rows(8)
    train = add_role_tier(train)
    model = EmpiricalBayesShareModel().fit(
        train[["player_id", "role_tier"]],
        train["share_of_team_rushing_tds"].to_numpy(float),
    )
    test = train.iloc[[0, 1]].copy()
    pred = model.predict(test[["player_id", "role_tier"]])
    assert np.isfinite(pred).all()
    assert set(model.player_effects_) == {"a", "b"}


def test_residual_opportunity_arm_is_bounded():
    train = add_role_tier(_rows(8))
    test = add_role_tier(_rows(2).assign(season=2025))
    for frame in (train, test):
        frame["team_receiving_tds"] = 1
        frame["share_of_team_receiving_tds"] = frame["share_of_team_rushing_tds"]
    pred = predict_residual_over_opportunity(train, test, "receiving_tds")
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()


def test_event_eligibility_excludes_cold_roster_rows():
    df = add_role_tier(_rows(1))
    df["redzone_targets_roll3"] = [0.0, 0.0]
    df["recv_targets_roll3"] = [0.0, 0.0]
    df["share_of_team_targets_roll3"] = [0.0, 0.0]
    df["air_yards_roll3"] = [0.0, 0.0]
    df["snap_share_roll3"] = [0.0, 0.0]
    assert not _eligibility_mask(df, "receiving_tds").any()


def test_multinomial_count_arm_is_bounded_and_grouped():
    train = add_role_tier(pd.concat([_rows(8), _rows(8).assign(season=2025)], ignore_index=True))
    test = add_role_tier(_rows(2).assign(season=2026))
    for frame in (train, test):
        frame["receiving_tds"] = frame["share_of_team_rushing_tds"]
        frame["team_receiving_tds"] = 1
        frame["team_receiving_tds_roll3"] = 1
        frame["team_receiving_tds_s2d"] = 1
        frame["share_of_team_receiving_tds"] = frame["share_of_team_rushing_tds"]
        frame["recv_targets_roll3"] = frame["snap_share_roll3"]
    pred = predict_multinomial_count_allocation(train, test, "receiving_tds")
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()
    # The event gate can down-weight an event-free team-week; allocation
    # remains compositional up to that calibrated probability.
    assert (pred.reshape(-1, 2).sum(axis=1) <= 1.0 + 1e-6).all()
    assert (pred.reshape(-1, 2).sum(axis=1) > 0.0).all()
