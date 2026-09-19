import numpy as np
import pandas as pd
import pytest

from src.models.participation_opportunity import (
    FEATURES, add_targets, build_causal_features, expanding_season_folds,
    normalize_snap_share, target_name, validate_causal_contract,
    TEAM_COMPETITION_FEATURES, predict_preseason,
    predict_asof_week,
)
from scripts.run_phase2_participation import evaluate, position_platt_gbm_probability, summarize


def _panel():
    rows = []
    for season in range(2021, 2026):
        for week in (1, 2, 3):
            for player, snaps, pct, status in (
                ("lead", 50, 80.0, "ACT"),
                ("depth", 0 if week == 1 else 8, 0.0 if week == 1 else 12.0, "ACT"),
            ):
                rows.append({
                    "player_id": player, "season": season, "week": week,
                    "team": "KC", "opponent": "LV", "position": "RB",
                    "home_away": "home", "status": status,
                    "participation_state": "confirmed_played" if snaps else "confirmed_zero_snaps",
                    "offense_snaps": snaps, "offense_pct": pct,
                    "fantasy_points": 10.0 if snaps else pd.NA,
                })
    rows.append({
        "player_id": "unknown", "season": 2025, "week": 1, "team": "KC",
        "opponent": "LV", "position": "WR", "home_away": "away", "status": "ACT",
        "participation_state": "unknown", "offense_snaps": pd.NA,
        "offense_pct": pd.NA, "fantasy_points": pd.NA,
    })
    return pd.DataFrame(rows)


def test_percentage_and_fraction_snap_share_are_normalized():
    assert normalize_snap_share(pd.Series([0, 50, 100, np.nan])).tolist()[:3] == [0, .5, 1]
    assert normalize_snap_share(pd.Series([0, .5, 1])).tolist() == [0, .5, 1]
    assert normalize_snap_share(pd.Series([.5, 50])).tolist() == [.5, .5]


def test_unknown_is_unlabeled_not_negative():
    out = add_targets(_panel())
    row = out[out.player_id.eq("unknown")].iloc[0]
    assert row.label_observed == 0
    assert pd.isna(row[target_name(0)])
    assert pd.isna(row[target_name(.10)])


def test_threshold_boundaries_are_objective():
    out = add_targets(_panel())
    zero = out[(out.player_id == "depth") & (out.week == 1)].iloc[0]
    used = out[(out.player_id == "depth") & (out.week == 2)].iloc[0]
    assert zero[target_name(0)] == 0
    assert used[target_name(0)] == 1
    assert used[target_name(.10)] == 1
    assert used[target_name(.25)] == 0


def test_features_never_include_current_outcome_columns():
    frame = build_causal_features(_panel())
    validate_causal_contract(frame)
    assert not ({"offense_snaps", "snap_share", "fantasy_points", "status"} & set(FEATURES))


def test_injury_score_requires_explicit_kickoff_filtered_opt_in():
    panel = _panel()
    panel["injury_score"] = 0.0
    neutral = build_causal_features(panel)
    opted_in = build_causal_features(panel, include_pregame_injury=True)
    assert neutral.injury_score.eq(1.0).all()
    assert opted_in.injury_score.eq(0.0).all()


def test_first_player_row_is_cold_start_and_has_no_usage_history():
    frame = build_causal_features(_panel())
    first = frame[frame.player_id.eq("lead")].sort_values(["season", "week"]).iloc[0]
    assert first.cold_start == 1
    assert first.prior_games_observed == 0
    assert pd.isna(first.snap_share_lag1)


def test_team_competition_features_exclude_focal_player_and_use_lagged_history():
    frame = build_causal_features(_panel())
    lead = frame[(frame.player_id.eq("lead")) & (frame.season.eq(2021)) & (frame.week.eq(2))].iloc[0]
    depth = frame[(frame.player_id.eq("depth")) & (frame.season.eq(2021)) & (frame.week.eq(2))].iloc[0]
    assert set(TEAM_COMPETITION_FEATURES) <= set(frame)
    # At week 2, lead's only peer had a 0% Week-1 share; depth's had 80%.
    assert lead.peer_history_count == 1
    assert lead.peer_snap_share_lag1_sum == 0
    assert depth.peer_history_count == 1
    assert depth.peer_snap_share_lag1_sum == pytest.approx(.8)


def test_teammate_current_outcome_never_enters_same_week_competition_features():
    panel = _panel()
    before = build_causal_features(panel)
    current = panel.player_id.eq("depth") & panel.season.eq(2025) & panel.week.eq(3)
    panel.loc[current, ["offense_snaps", "offense_pct"]] = [50, 90.]
    after = build_causal_features(panel)
    lead_week = before.player_id.eq("lead") & before.season.eq(2025) & before.week.eq(3)
    pd.testing.assert_frame_equal(
        before.loc[lead_week, TEAM_COMPETITION_FEATURES],
        after.loc[lead_week, TEAM_COMPETITION_FEATURES],
    )


def test_future_outcome_change_cannot_change_past_features():
    panel = _panel()
    before = build_causal_features(panel)
    mask = (panel.player_id == "lead") & (panel.season == 2025) & (panel.week == 3)
    panel.loc[mask, ["offense_snaps", "offense_pct"]] = [1, 1.0]
    after = build_causal_features(panel)
    past = (before.season < 2025) | ((before.season == 2025) & (before.week < 3))
    pd.testing.assert_frame_equal(before.loc[past, FEATURES], after.loc[past, FEATURES])


def test_walk_forward_has_strict_temporal_boundary():
    frame = build_causal_features(_panel())
    observed = frame[frame.label_observed.eq(1)]
    folds = list(expanding_season_folds(observed, min_train_seasons=2))
    assert folds
    for fold in folds:
        assert observed.loc[fold.train_index, "season"].max() < fold.test_season
        assert observed.loc[fold.test_index, "season"].eq(fold.test_season).all()


def test_duplicate_keys_fail_before_feature_generation():
    panel = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_causal_features(panel)


def test_observed_row_without_share_fails_loudly():
    panel = _panel()
    panel.loc[0, "offense_pct"] = pd.NA
    with pytest.raises(ValueError, match="no usable offense_pct"):
        add_targets(panel)


def test_evaluation_harness_produces_strict_oof_metrics():
    frame = build_causal_features(_panel())
    predictions, report = evaluate(frame, min_train_seasons=2)
    assert not predictions.empty
    assert "unknown" not in set(predictions.player_id)
    assert predictions.season.min() >= 2023
    assert (predictions.phase2_train_max_season < predictions.season).all()
    assert predictions.phase2_test_season.equals(predictions.season)
    summary = summarize(report)
    assert any(
        row["threshold"] == .10 and row["model"] == "logistic" and row["segment"] == "all"
        for row in summary["classification"]
    )
    assert any(
        row["threshold"] == .10 and row["model"] == "hist_gbm_platt_position" and row["segment"] == "all"
        for row in summary["classification"]
    )
    assert any(
        row["threshold"] == .10 and row["model"] == "hist_gbm_team_competition" and row["segment"] == "all"
        for row in summary["classification"]
    )


def test_position_platt_calibration_reserves_latest_training_season():
    frame = build_causal_features(_panel())
    observed = frame[frame.label_observed.eq(1)]
    fold = list(expanding_season_folds(observed, min_train_seasons=3))[0]
    train, test = observed.loc[fold.train_index], observed.loc[fold.test_index]
    target = target_name(.10)
    probability, modes = position_platt_gbm_probability(train, test, target)
    assert len(probability) == len(test)
    assert np.isfinite(probability).all()
    assert set(modes) == {"RB"}
    # The calibration season is the latest *training* season, never held out.
    assert modes["RB"] in {
        f"platt_holdout_{int(train.season.max())}",
        "raw_fallback_insufficient_calibration_classes",
    }


def test_preseason_prediction_excludes_target_season_from_training():
    panel = _panel()
    # Keep unique player-week keys while making this a viable opportunity fit.
    panel = pd.concat([
        panel.assign(player_id=panel.player_id + f"_{copy}")
        for copy in range(10)
    ], ignore_index=True)
    frame = build_causal_features(panel)
    prediction = predict_preseason(frame, 2025)
    assert prediction.season.eq(2025).all()
    assert prediction.week.eq(1).all()
    assert {"participation_probability", "conditional_snap_share", "expected_snap_share"} <= set(prediction)
    assert prediction.participation_probability.between(0, 1).all()
    assert prediction.expected_snap_share.between(0, 1).all()


def test_asof_week_prediction_excludes_target_week_outcome():
    panel = _panel()
    panel = pd.concat([panel.assign(player_id=panel.player_id + f"_{copy}") for copy in range(10)], ignore_index=True)
    before = build_causal_features(panel)
    prediction_before = predict_asof_week(before, 2025, 2)
    mask = (panel.season.eq(2025) & panel.week.eq(2))
    panel.loc[mask, ["offense_snaps", "offense_pct"]] = [0, 0]
    after = build_causal_features(panel)
    prediction_after = predict_asof_week(after, 2025, 2)
    pd.testing.assert_frame_equal(prediction_before, prediction_after)


def test_percent_strings_preserve_one_percent():
    assert normalize_snap_share(pd.Series(["1%", "0.5%", "100%", 1.0])).tolist() == [.01, .005, 1., 1.]


def test_baselines_use_history_of_the_scored_threshold():
    from scripts.run_phase2_participation import baseline_probability
    panel = _panel()
    panel.loc[panel.player_id.eq("lead"), "offense_pct"] = 5.0
    frame = build_causal_features(panel)
    train = frame[frame.season.lt(2025) & frame.label_observed.eq(1)]
    test = frame[frame.player_id.eq("lead") & frame.season.eq(2025)]
    for mode in ("previous_game", "rolling3"):
        assert (baseline_probability(train, test, target_name(.10), mode) == 0).all()
        assert (baseline_probability(train, test, target_name(0), mode) == 1).all()


def test_unidentified_player_is_not_silently_dropped_from_features():
    panel = _panel()
    panel.loc[0, 'player_id'] = None
    with pytest.raises(ValueError, match='identity'):
        build_causal_features(panel)


def test_calibration_error_distinguishes_calibrated_and_overconfident_predictions():
    from scripts.run_phase2_participation import classification_metrics
    assert classification_metrics(pd.Series([0, 1]), np.array([.5, .5]))['ece'] == 0
    assert classification_metrics(pd.Series([0, 0]), np.array([1., 1.]))['ece'] == 1


def test_opportunity_reports_matched_cold_start_and_position_baselines():
    panel = _panel()
    panel = pd.concat([panel.assign(player_id=panel.player_id + f'_{copy}')
                       for copy in range(10)], ignore_index=True)
    debut = panel[panel.season.eq(2025) & panel.player_id.str.startswith('lead')].copy()
    debut['player_id'] = 'debut_' + debut.player_id
    frame = build_causal_features(pd.concat([panel, debut], ignore_index=True))
    _, report = evaluate(frame, min_train_seasons=4)
    opportunity = pd.DataFrame(report['opportunity'])
    for segment in ['all', 'position:RB', 'cold_start', 'has_history']:
        rows = opportunity[opportunity.segment.eq(segment)]
        assert set(rows.model) == {'rolling3', 'hist_gbm_mae', 'hist_gbm_team_competition_mae'}
        assert rows.n.nunique() == 1
        assert rows.n.iloc[0] > 0
