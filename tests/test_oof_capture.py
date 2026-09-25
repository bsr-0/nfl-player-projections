"""Row-level OOF capture must be genuinely out-of-fold.

A contaminated panel makes every downstream segment number look better than
it is, and does so silently -- the failure mode that GAPS.md has recorded
repeatedly in this repo. The leakage conditions are therefore enforced in
code and pinned here, not left to inspection of the calling site.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.oof_capture import (
    ACTUAL_COLUMN,
    OOFLeakageError,
    add_experience_segments,
    build_panel,
    capture_fold_rows,
    segment_report,
    write_panel,
)


def _fold_frame(season, n=6, players=None, predicted=None, actual=None):
    players = players or [f"p{i}" for i in range(n)]
    return pd.DataFrame({
        "player_id": players,
        "season": season,
        "week": [1 + i % 3 for i in range(len(players))],
        "team": ["A", "B"] * (len(players) // 2) + ["A"] * (len(players) % 2),
        "opponent": ["B", "A"] * (len(players) // 2) + ["B"] * (len(players) % 2),
        "position": [["QB", "RB", "WR", "TE"][i % 4] for i in range(len(players))],
        "predicted_points": predicted if predicted is not None else np.linspace(5, 20, len(players)),
        ACTUAL_COLUMN: actual if actual is not None else np.linspace(6, 18, len(players)),
    })


# --------------------------------------------------------------------------
# Leakage guarantees
# --------------------------------------------------------------------------

def test_rejects_a_test_season_that_is_also_a_training_season():
    with pytest.raises(OOFLeakageError, match="in-sample"):
        capture_fold_rows(_fold_frame(2024), train_seasons=[2022, 2023, 2024], test_season=2024)


def test_rejects_a_fold_frame_holding_other_seasons():
    """If the test frame carries rows outside the held-out season, some of
    those rows were seen in training.
    """
    contaminated = pd.concat([_fold_frame(2024), _fold_frame(2023)], ignore_index=True)
    with pytest.raises(OOFLeakageError, match="carries rows from"):
        capture_fold_rows(contaminated, train_seasons=[2021, 2022, 2023], test_season=2024)


def test_rejects_overlapping_folds_at_panel_build():
    """Two folds predicting the same player-week would double-count it and
    silently reweight every aggregate computed from the panel.
    """
    fold = capture_fold_rows(_fold_frame(2024), train_seasons=[2022, 2023], test_season=2024)
    with pytest.raises(OOFLeakageError, match="more than one"):
        build_panel([fold, fold])


def test_accepts_a_clean_fold():
    rows = capture_fold_rows(_fold_frame(2024), train_seasons=[2022, 2023], test_season=2024)
    assert len(rows) == 6
    assert rows["train_seasons"].iloc[0] == "2022,2023"
    assert rows["n_train_seasons"].iloc[0] == 2


# --------------------------------------------------------------------------
# Row handling
# --------------------------------------------------------------------------

def test_unscored_rows_are_dropped_not_zero_filled():
    """A row the model could not score is not evidence about the model.
    Zero-filling it would understate error, the same silent-zero defect
    already fixed once in reconstruct_partial_fantasy_points.
    """
    frame = _fold_frame(2024)
    frame.loc[0, "predicted_points"] = np.nan
    frame.loc[1, ACTUAL_COLUMN] = np.nan

    rows = capture_fold_rows(frame, train_seasons=[2023], test_season=2024)
    assert len(rows) == 4
    assert not (rows["predicted_points"] == 0).any()


def test_residual_sign_convention_is_predicted_minus_actual():
    """Bias reporting depends on this: positive means over-prediction."""
    frame = _fold_frame(2024, n=2, predicted=[12.0, 8.0], actual=[10.0, 10.0])
    rows = capture_fold_rows(frame, train_seasons=[2023], test_season=2024)
    assert rows["residual"].tolist() == [2.0, -2.0]


def test_missing_identity_columns_raise():
    frame = _fold_frame(2024).drop(columns=["opponent"])
    with pytest.raises(ValueError, match="identity columns"):
        capture_fold_rows(frame, train_seasons=[2023], test_season=2024)


# --------------------------------------------------------------------------
# Experience segmentation -- the "new players without prior starts" split
# --------------------------------------------------------------------------

def test_cold_start_is_a_players_first_appearance_only():
    early = _fold_frame(2023, players=["a", "b"])
    late = _fold_frame(2024, players=["a", "c"])
    panel = build_panel([
        capture_fold_rows(early, train_seasons=[2022], test_season=2023),
        capture_fold_rows(late, train_seasons=[2022, 2023], test_season=2024),
    ])

    by_player = panel.set_index(["player_id", "season"])["is_cold_start"]
    assert by_player[("a", 2023)] and not by_player[("a", 2024)]
    assert by_player[("b", 2023)] and by_player[("c", 2024)]


def test_prior_weeks_never_counts_the_row_itself():
    """A strict shift: a row must not be informed by its own existence."""
    frames = [_fold_frame(s, players=["a"]) for s in (2022, 2023, 2024)]
    panel = build_panel([
        capture_fold_rows(f, train_seasons=[s - 1], test_season=s)
        for f, s in zip(frames, (2022, 2023, 2024))
    ])
    assert panel.sort_values("season")["prior_weeks_in_panel"].tolist() == [0, 1, 2]


def test_experience_segments_are_ordered_by_season_then_week():
    """Out-of-order input must not make a later week look like a debut."""
    frame = pd.concat([
        _fold_frame(2024, players=["a"]).assign(week=9),
        _fold_frame(2024, players=["a"]).assign(week=2),
    ], ignore_index=True)
    segmented = add_experience_segments(frame.assign(residual=0.0))
    debut = segmented.loc[segmented["is_cold_start"], "week"].tolist()
    assert debut == [2]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_segment_report_splits_by_position_and_cold_start():
    panel = build_panel([
        capture_fold_rows(_fold_frame(2023), train_seasons=[2022], test_season=2023),
        capture_fold_rows(_fold_frame(2024), train_seasons=[2022, 2023], test_season=2024),
    ])
    report = segment_report(panel, by=("position", "is_cold_start"))
    assert {"position", "is_cold_start", "n", "mae", "rmse", "bias"} <= set(report.columns)
    assert report["n"].sum() == len(panel)


def test_segment_report_surfaces_bias_that_mae_hides():
    """Two segments with identical MAE but opposite bias must be
    distinguishable -- the whole reason bias is reported.
    """
    over = _fold_frame(2023, n=4, predicted=[12.0] * 4, actual=[10.0] * 4)
    under = _fold_frame(2024, n=4, predicted=[8.0] * 4, actual=[10.0] * 4)
    panel = build_panel([
        capture_fold_rows(over, train_seasons=[2022], test_season=2023),
        capture_fold_rows(under, train_seasons=[2022, 2023], test_season=2024),
    ])
    report = segment_report(panel, by=("season",))
    assert report["mae"].tolist() == [2.0, 2.0]
    assert report["bias"].tolist() == [2.0, -2.0]


def test_empty_inputs_do_not_raise():
    assert build_panel([]).empty
    assert segment_report(pd.DataFrame(columns=["residual", "position"])).empty


def test_panel_round_trips_through_parquet(tmp_path):
    panel = build_panel([capture_fold_rows(_fold_frame(2024),
                                           train_seasons=[2023], test_season=2024)])
    path = write_panel(panel, tmp_path / "oof.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(path), panel)
