"""Fixes from the 2026-09-25 methodology review of the OOF panel.

One lens of an adversarial review completed before the reviewing session
hit a rate limit; its claims were independently re-verified against
data/nfl_data.db (composition percentages, zero-rates, and skew all
replicated within rounding) before being acted on here. Five problems, five
groups of tests:

1. `is_cold_start` (panel-relative) is confounded with early-season weeks
   and with the shortest-training-window fold -- `week_bucket` and
   `add_career_experience_segments` address it.
2. `segment_report` had no uncertainty on non-independent rows -- the
   `cluster` argument.
3. No artifact recorded which run produced a panel -- `write_run_panel`.
4. Dropped rows were logged, not persisted -- `fold_coverage`.
5. MAE/RMSE/bias alone understate a zero-inflated, right-skewed outcome --
   `zero_rate`, `mae_positive`, `spearman` in `segment_report`.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.models.oof_capture import (
    ACTUAL_COLUMN,
    OOF_PANEL_FILENAME,
    add_career_experience_segments,
    add_experience_segments,
    build_panel,
    capture_fold_rows,
    fold_coverage,
    segment_report,
    write_run_panel,
)


def _fold_frame(season, n=6, players=None, predicted=None, actual=None, positions=None):
    players = players or [f"p{i}" for i in range(n)]
    return pd.DataFrame({
        "player_id": players,
        "season": season,
        "week": [1 + i % 3 for i in range(len(players))],
        "team": ["A", "B"] * (len(players) // 2) + ["A"] * (len(players) % 2),
        "opponent": ["B", "A"] * (len(players) // 2) + ["B"] * (len(players) % 2),
        "position": positions or [["QB", "RB", "WR", "TE"][i % 4] for i in range(len(players))],
        "predicted_points": predicted if predicted is not None else np.linspace(5, 20, len(players)),
        ACTUAL_COLUMN: actual if actual is not None else np.linspace(6, 18, len(players)),
    })


# --------------------------------------------------------------------------
# 1. week_bucket -- surfacing the cold-start / early-week confound
# --------------------------------------------------------------------------

def test_week_bucket_splits_early_from_mid_late():
    panel = pd.DataFrame({"week": [1, 2, 3, 9]})
    out = add_experience_segments(pd.concat([panel, pd.DataFrame({
        "player_id": ["a", "b", "c", "d"], "season": 2024, "team": "A", "opponent": "B",
        "position": "QB", "residual": 0.0,
    })], axis=1))
    assert out.set_index("week")["week_bucket"].to_dict() == {
        1: "early", 2: "early", 3: "mid_late", 9: "mid_late"}


def test_cold_start_and_week_bucket_together_reveal_the_confound():
    """The scenario the review measured: a fold's cold-start cell is
    dominated by week-1 rows, while returning players' rows spread across
    the season. Grouping by is_cold_start alone hides this; adding
    week_bucket does not.
    """
    # An earlier fold establishes p6/p7 as "returning" by the 2023 fold.
    earlier = _fold_frame(2022, n=2, players=["p6", "p7"], positions=["QB", "QB"])
    earlier["week"] = [1, 1]
    # 2023: 6 brand-new players debut in week 1 (cold-start); p6/p7 return in week 9.
    later = _fold_frame(2023, n=8, players=[f"p{i}" for i in range(6)] + ["p6", "p7"],
                        positions=["QB"] * 8)
    later["week"] = [1, 1, 1, 1, 1, 1, 9, 9]
    panel = build_panel([
        capture_fold_rows(earlier, train_seasons=[2021], test_season=2022),
        capture_fold_rows(later, train_seasons=[2021, 2022], test_season=2023),
    ])
    # Restrict to the 2023 cell under test: p6/p7's OWN 2022 rows are also
    # legitimately cold-start (that fold's own first appearance), which is
    # correct, not the confound this test is isolating.
    panel = panel[panel["season"] == 2023]

    by_cold_start_only = segment_report(panel, by=("is_cold_start",))
    cold_row = by_cold_start_only[by_cold_start_only["is_cold_start"]]
    assert cold_row["n"].iloc[0] == 6  # true, but doesn't say WHY they're a distinct group

    by_both = segment_report(panel, by=("is_cold_start", "week_bucket"))
    early_cold = by_both.query("is_cold_start and week_bucket == 'early'")
    assert early_cold["n"].iloc[0] == 6, "the confound must be visible once week_bucket is added"


# --------------------------------------------------------------------------
# 2. career-relative cold-start
# --------------------------------------------------------------------------

def test_career_cold_start_unmasks_a_returning_veteran(tmp_path, monkeypatch):
    """The measured failure mode: a veteran whose first PANEL row is the
    panel's earliest season is mislabelled cold-start. Career history says
    otherwise and must override it.
    """
    import sqlite3
    db_path = tmp_path / "t.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE player_weekly_stats (player_id TEXT, season INTEGER, week INTEGER)")
    # "veteran" played in 2019-2022, long before the panel starts in 2023.
    conn.executemany("INSERT INTO player_weekly_stats VALUES (?, ?, ?)",
                     [("veteran", 2021, w) for w in range(1, 5)])
    conn.commit()
    conn.close()

    class _FakeDB:
        def __init__(self, path=None):
            self.db_path = path or db_path

    monkeypatch.setattr("src.utils.database.DatabaseManager", _FakeDB)

    panel = build_panel([capture_fold_rows(
        _fold_frame(2023, n=2, players=["veteran", "rookie"], positions=["QB", "QB"]),
        train_seasons=[2022], test_season=2023)])
    out = add_career_experience_segments(panel, db_path=db_path)

    by_player = out.set_index("player_id")
    assert by_player.loc["veteran", "is_cold_start"] == True  # panel-relative: still true
    assert by_player.loc["veteran", "is_cold_start_career"] == False  # career-relative: corrected
    assert by_player.loc["rookie", "is_cold_start_career"] == True


def test_career_cold_start_fails_open_when_db_unavailable(monkeypatch):
    """This is a reporting augmentation, not a leakage guarantee -- unlike
    capture_fold_rows, it must degrade gracefully rather than raise.
    """
    def _boom(*a, **k):
        raise RuntimeError("no db here")
    monkeypatch.setattr("src.utils.database.DatabaseManager", _boom)

    panel = build_panel([capture_fold_rows(_fold_frame(2023), train_seasons=[2022], test_season=2023)])
    out = add_career_experience_segments(panel)
    assert out["is_cold_start_career"].isna().all()


def test_career_cold_start_on_empty_panel_does_not_raise():
    empty = pd.DataFrame(columns=["player_id", "season", "week", "is_cold_start"])
    out = add_career_experience_segments(empty)
    assert out.empty


# --------------------------------------------------------------------------
# 3. cluster-aware uncertainty
# --------------------------------------------------------------------------

def test_segment_report_without_cluster_has_no_ci_columns():
    panel = build_panel([capture_fold_rows(_fold_frame(2023), train_seasons=[2022], test_season=2023)])
    report = segment_report(panel, by=("position",))
    assert "mae_ci_lo" not in report.columns, "must be absent, not silently wrong, when unrequested"


def test_segment_report_with_cluster_adds_a_ci_that_widens_with_correlated_residuals():
    """The scenario cluster-awareness exists for: residuals correlated
    WITHIN a cluster (a real player effect, as the review measured -- ICC
    0.08-0.11 within player_id on real data), not merely "many small
    clusters vs few big ones" in the abstract. Give few_players a strong,
    consistent per-player bias (high within-cluster correlation) and
    many_players near-independent per-row noise (~zero within-cluster
    correlation, since each row is its own cluster) -- the clustered case
    must show a wider CI, because there are effectively only 3 independent
    pieces of information instead of 40.
    """
    n = 40
    # many_players: 40 singleton clusters, small independent-looking noise.
    many_wiggle = [((i % 5) - 2) * 0.5 for i in range(n)]  # -1..+1, cycling
    many_players = _fold_frame(2023, n=n, players=[f"p{i}" for i in range(n)],
                               positions=["QB"] * n,
                               predicted=[10.0 + w for w in many_wiggle],
                               actual=[10.0] * n)

    # few_players: 3 clusters, each with a CONSTANT bias -- residual is
    # determined almost entirely by which cluster a row belongs to.
    bias_by_player = {0: 8.0, 1: -8.0, 2: 0.0}
    few_players = _fold_frame(2023, n=n, players=[f"p{i % 3}" for i in range(n)],
                              positions=["QB"] * n,
                              predicted=[10.0 + bias_by_player[i % 3] for i in range(n)],
                              actual=[10.0] * n)
    # Distinct week per repetition of the same player_id -- the fixture's
    # default week formula (1 + i % 3) collides with a 3-player cycle and
    # would otherwise produce duplicate (player, week) rows.
    few_players["week"] = [i // 3 + 1 for i in range(n)]

    wide_panel = build_panel([capture_fold_rows(many_players, train_seasons=[2022], test_season=2023)])
    narrow_panel = build_panel([capture_fold_rows(few_players, train_seasons=[2022], test_season=2023)])

    wide = segment_report(wide_panel, by=("position",), cluster="player_id", n_boot=2000, seed=1)
    narrow = segment_report(narrow_panel, by=("position",), cluster="player_id", n_boot=2000, seed=1)

    wide_width = wide["mae_ci_hi"].iloc[0] - wide["mae_ci_lo"].iloc[0]
    narrow_width = narrow["mae_ci_hi"].iloc[0] - narrow["mae_ci_lo"].iloc[0]
    assert narrow_width > wide_width, (
        "3 correlated clusters must produce a WIDER CI than 40 near-"
        "independent ones -- that would mean the bootstrap is resampling "
        "rows and ignoring the clustering structure entirely")


def test_a_single_cluster_returns_nan_ci_rather_than_a_fabricated_one():
    frame = _fold_frame(2023, n=4, players=["only_one"] * 4, positions=["QB"] * 4)
    frame["week"] = [1, 2, 3, 4]  # distinct weeks: same player, not a duplicate player-week
    panel = build_panel([capture_fold_rows(frame, train_seasons=[2022], test_season=2023)])
    report = segment_report(panel, by=("position",), cluster="player_id", n_boot=200)
    assert np.isnan(report["mae_ci_lo"].iloc[0])


# --------------------------------------------------------------------------
# 4. coverage tracking for dropped rows
# --------------------------------------------------------------------------

def test_fold_coverage_records_offered_captured_and_dropped():
    offered = _fold_frame(2023, n=4, positions=["QB", "QB", "RB", "RB"])
    offered.loc[0, "predicted_points"] = np.nan  # one QB row unscored
    captured = capture_fold_rows(offered, train_seasons=[2022], test_season=2023)

    coverage = fold_coverage(offered, captured, test_season=2023)
    qb_row = coverage.set_index("position").loc["QB"]
    assert qb_row["n_offered"] == 2 and qb_row["n_captured"] == 1 and qb_row["n_dropped"] == 1
    rb_row = coverage.set_index("position").loc["RB"]
    assert rb_row["n_dropped"] == 0


def test_fold_coverage_handles_a_fully_dropped_position():
    offered = _fold_frame(2023, n=2, positions=["TE", "TE"])
    offered["predicted_points"] = np.nan
    captured = capture_fold_rows(offered, train_seasons=[2022], test_season=2023)
    assert captured.empty

    coverage = fold_coverage(offered, captured, test_season=2023)
    te_row = coverage.set_index("position").loc["TE"]
    assert te_row["n_offered"] == 2 and te_row["n_captured"] == 0 and te_row["n_dropped"] == 2


# --------------------------------------------------------------------------
# 5. zero-inflation-aware reporting
# --------------------------------------------------------------------------

def test_zero_rate_and_mae_positive_are_reported():
    frame = _fold_frame(2023, n=4, predicted=[0.0, 0.0, 10.0, 14.0], actual=[0.0, 0.0, 8.0, 10.0])
    panel = build_panel([capture_fold_rows(frame, train_seasons=[2022], test_season=2023)])
    report = segment_report(panel, by=("season",))
    assert report["zero_rate"].iloc[0] == pytest.approx(0.5)
    # mae over ALL 4 rows would include the two perfect zero predictions;
    # mae_positive must only average the two nonzero-actual rows (|2|, |4|).
    assert report["mae"].iloc[0] == pytest.approx((0 + 0 + 2 + 4) / 4)
    assert report["mae_positive"].iloc[0] == pytest.approx((2 + 4) / 2)


def test_spearman_is_reported_and_nan_when_undefined():
    frame = _fold_frame(2023, n=4, predicted=[1, 2, 3, 4], actual=[10, 20, 30, 40])
    panel = build_panel([capture_fold_rows(frame, train_seasons=[2022], test_season=2023)])
    report = segment_report(panel, by=("season",))
    assert report["spearman"].iloc[0] == pytest.approx(1.0)

    single_row = build_panel([capture_fold_rows(
        _fold_frame(2024, n=1, players=["only"], positions=["QB"]),
        train_seasons=[2023], test_season=2024)])
    single_report = segment_report(single_row, by=("season",))
    assert np.isnan(single_report["spearman"].iloc[0])


# --------------------------------------------------------------------------
# write_run_panel: provenance + no silent overwrite between runs
# --------------------------------------------------------------------------

def test_two_runs_do_not_overwrite_each_others_immutable_copy(tmp_path):
    panel_a = build_panel([capture_fold_rows(_fold_frame(2023), train_seasons=[2022], test_season=2023)])
    panel_b = build_panel([capture_fold_rows(_fold_frame(2024), train_seasons=[2022, 2023], test_season=2024)])

    written_a = write_run_panel(panel_a, tmp_path, label="arm-a")
    written_b = write_run_panel(panel_b, tmp_path, label="arm-b")

    assert written_a["run_dir"] != written_b["run_dir"]
    assert written_a["panel_path"].exists(), "first run's immutable copy must survive the second run"
    restored_a = pd.read_parquet(written_a["panel_path"])
    assert set(restored_a["season"]) == {2023}, "first run's panel must be unchanged by the second"


def test_manifest_records_provenance_and_label(tmp_path):
    panel = build_panel([capture_fold_rows(_fold_frame(2023), train_seasons=[2022], test_season=2023)])
    written = write_run_panel(panel, tmp_path, label="pre-vegas-fix")

    manifest = json.loads(written["manifest_path"].read_text())
    assert manifest["label"] == "pre-vegas-fix"
    assert manifest["seasons"] == [2023]
    assert "generated_at" in manifest and "git_commit" in manifest


def test_coverage_is_written_alongside_the_panel_when_provided(tmp_path):
    offered = _fold_frame(2023, n=2, positions=["QB", "QB"])
    offered.loc[0, "predicted_points"] = np.nan
    captured = capture_fold_rows(offered, train_seasons=[2022], test_season=2023)
    coverage = fold_coverage(offered, captured, test_season=2023)
    panel = build_panel([captured])

    written = write_run_panel(panel, tmp_path, coverage=coverage, label="x")
    assert written["coverage_path"] is not None
    records = json.loads(written["coverage_path"].read_text())
    assert records[0]["n_dropped"] == 1


def test_latest_pointer_is_overwritten_but_labelled_as_such(tmp_path):
    panel_a = build_panel([capture_fold_rows(_fold_frame(2023), train_seasons=[2022], test_season=2023)])
    panel_b = build_panel([capture_fold_rows(_fold_frame(2024), train_seasons=[2022, 2023], test_season=2024)])
    write_run_panel(panel_a, tmp_path, label="first")
    written_b = write_run_panel(panel_b, tmp_path, label="second")

    latest = pd.read_parquet(tmp_path / OOF_PANEL_FILENAME)
    assert set(latest["season"]) == {2024}, "the flat pointer is whichever run wrote last, by design"
    assert written_b["latest_path"] == tmp_path / OOF_PANEL_FILENAME


def test_old_run_panels_are_pruned_but_recent_ones_survive(tmp_path):
    for i in range(8):
        panel = build_panel([capture_fold_rows(
            _fold_frame(2020 + i, n=2, players=[f"p{i}"]),
            train_seasons=[2019 + i], test_season=2020 + i)])
        write_run_panel(panel, tmp_path, label=f"run{i}", keep=3)

    remaining = sorted((tmp_path / "oof_panels").iterdir())
    assert len(remaining) == 3
    assert remaining[-1].name.endswith("run7")
