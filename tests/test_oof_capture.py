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
    add_game_context,
    build_panel,
    capture_fold_rows,
    fold_coverage,
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


def test_game_context_uses_exact_scheduled_home_away_pair(tmp_path):
    import sqlite3
    db_path = tmp_path / "schedule.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE schedule (season INTEGER, week INTEGER, home_team TEXT, away_team TEXT, game_id TEXT)")
    conn.execute("INSERT INTO schedule VALUES (2024, 1, 'H', 'A', 'game-id')")
    conn.commit()
    conn.close()
    # Origin week 1 (vs X), target week 2 (the scheduled H-A game).
    frame = pd.DataFrame({
        "player_id": ["h", "h", "a", "a"], "season": 2024, "week": [1, 2, 1, 2],
        "team": ["H", "H", "A", "A"], "opponent": ["X", "A", "Y", "H"],
        "position": "WR", "predicted_points": 10.0, ACTUAL_COLUMN: [8.0, np.nan, 9.0, np.nan],
    })
    panel = build_panel([capture_fold_rows(frame, train_seasons=[2023], test_season=2024)])
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO schedule VALUES (2024, 2, 'H', 'A', 'game-2')")
    conn.commit()
    conn.close()
    context = add_game_context(panel, db_path=db_path)
    assert set(context["game_id"]) == {"game-2"}
    assert set(context["home_team"]) == {"H"} and set(context["away_team"]) == {"A"}


# --------------------------------------------------------------------------
# Target-game identity (actual_points is the NEXT observed game's outcome)
# --------------------------------------------------------------------------

def _season_frame():
    """Player p: weeks 1, 2, 5 (gap), traded B->C after week 2."""
    fantasy = [3.0, 7.0, 11.0, 4.0, 6.0]
    frame = pd.DataFrame({
        "player_id": ["p", "p", "p", "q", "q"], "season": 2024,
        "week": [1, 2, 5, 1, 2], "team": ["B", "B", "C", "D", "D"],
        "opponent": ["E", "F", "G", "H", "I"], "position": "WR",
        "predicted_points": 5.0, "fantasy_points": fantasy,
    })
    frame["target_1w"] = frame.groupby(["player_id", "season"])["fantasy_points"].shift(-1)
    frame[ACTUAL_COLUMN] = frame["target_1w"]
    return frame


def test_capture_names_the_next_observed_game_not_the_origin():
    rows = capture_fold_rows(_season_frame().sample(frac=1, random_state=3),
                             train_seasons=[2023], test_season=2024)
    got = rows.set_index(["player_id", "week"]).sort_index()
    assert got.loc[("p", 2), "target_week"] == 5          # gap, not week + 1
    assert got.loc[("p", 2), "target_team"] == "C"        # post-trade team
    assert got.loc[("p", 2), "target_opponent"] == "G"
    assert got.loc[("p", 2), "actual_points"] == 11.0     # week-5 outcome
    assert got.loc[("q", 1), "target_opponent"] == "I"
    assert len(rows) == 3                                  # last games have no target


def test_capture_refuses_a_frame_that_no_longer_reproduces_target_1w():
    # Player r widens the WR target_1w range. The check allows for
    # winsorized targets (target_1w == clip(next fantasy_points) within each
    # position's observed range), so a lost row is caught when the shifted
    # value lands inside that range -- in _season_frame alone, 11 vs a stated
    # 7 at the range's top is indistinguishable from a clip at 7.
    wide = pd.DataFrame({"player_id": "r", "season": 2024, "week": [1, 2, 3], "team": "D",
                         "opponent": ["H", "I", "J"], "position": "WR", "predicted_points": 5.0,
                         "fantasy_points": [1.0, 20.0, 2.0]})
    wide["target_1w"] = wide["fantasy_points"].shift(-1)
    wide[ACTUAL_COLUMN] = wide["target_1w"]
    frame = pd.concat([_season_frame(), wide], ignore_index=True)
    frame = frame.drop(index=1)  # a row lost after targets were built
    with pytest.raises(ValueError, match="does not reproduce target_1w"):
        capture_fold_rows(frame, train_seasons=[2023], test_season=2024)


def test_target_game_panel_rekeys_and_refuses_origin_only_panels():
    from src.models.oof_capture import target_game_panel
    panel = build_panel([capture_fold_rows(_season_frame(), train_seasons=[2023], test_season=2024)])
    with pytest.raises(ValueError, match="predates target-game capture"):
        target_game_panel(panel)
    with_context = panel.assign(game_id="g", home_team=panel["target_team"],
                                away_team=panel["target_opponent"])
    rekeyed = target_game_panel(with_context).set_index(["player_id", "origin_week"])
    assert rekeyed.loc[("p", 2), "week"] == 5 and rekeyed.loc[("p", 2), "team"] == "C"
    assert rekeyed.loc[("p", 2), "origin_team"] == "B"


def test_capture_accepts_targets_winsorized_like_prepare_training_data():
    """_prepare_training_data clips test target_1w to per-position training
    quantiles and leaves fantasy_points alone; comparing them unclipped
    failed every real fold."""
    from src.models.feature_preparation import _create_horizon_targets

    rng = np.random.default_rng(0)
    frame = _create_horizon_targets(pd.DataFrame(
        [{"player_id": f"p{i}", "season": 2024, "week": w, "team": "A", "opponent": "B",
          "position": "WR" if i % 2 else "RB", "predicted_points": 10.0,
          "fantasy_points": float(rng.gamma(2, 5)) - (3.0 if w == 2 else 0.0)}
         for i in range(40) for w in range(1, 6)]), n_weeks=[1])
    for position, (lo, hi) in {"WR": (0.0, 25.0), "RB": (-0.5, 18.0)}.items():
        mask = frame["position"] == position
        frame.loc[mask, "target_1w"] = frame.loc[mask, "target_1w"].clip(lo, hi)
    frame[ACTUAL_COLUMN] = frame["target_1w"]
    assert (frame["target_1w"] != frame.groupby("player_id")["fantasy_points"].shift(-1)).any()

    rows = capture_fold_rows(frame, train_seasons=[2023], test_season=2024)
    assert len(rows) == 160  # every row with a next game


def test_self_paired_target_game_is_excluded_and_counted():
    """A target game recorded as a team playing itself matches no schedule
    entry, so add_game_context would reject the whole panel."""
    frame = _season_frame()
    frame.loc[2, "opponent"] = frame.loc[2, "team"]  # p's week-5 game: C vs C
    rows = capture_fold_rows(frame, train_seasons=[2023], test_season=2024)
    assert not ((rows["player_id"] == "p") & (rows["week"] == 2)).any()
    assert len(rows) == 2

    coverage = fold_coverage(frame, rows, test_season=2024).iloc[0]
    assert coverage["n_invalid_target_game"] == 1
    assert coverage["n_no_target_game"] == 2
    assert coverage["n_missing_prediction_or_actual"] == 0
