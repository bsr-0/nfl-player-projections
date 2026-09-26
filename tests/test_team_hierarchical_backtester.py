"""Synthetic integration and no-future-label tests for the future Plan B run."""
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_plan_b_shares import main
from src.evaluation.team_hierarchical_backtester import ARMS, KEYS, run_backtest, validate_panel
from src.models.team_hierarchical.models import MixedEffectsFitError


def panel_fixture():
    rng = np.random.default_rng(71)
    rows = []
    for team in range(4):
        for rank in (1, 2):
            bias = rng.normal(0, 0.04)
            for season in (2021, 2022, 2023):
                for week in range(1, 7):
                    lagged = rng.uniform(0.1, 0.5)
                    rows.append(dict(player_id=f"p{team}_{rank}", team=f"T{team}", position="RB",
                                     season=season, week=week, slot=f"RB{rank}", slot_rank=rank,
                                     depth_chart_rank=rank, roster_snap_share_s2d=rng.uniform(0.2, 0.7),
                                     is_cold_start=int(week == 1),
                                     share_of_team_rushing_yards=np.clip(0.2 + 0.2 * lagged + bias + rng.normal(0, 0.015), 0, 1),
                                     share_of_team_rushing_yards_roll3=lagged if week > 1 else np.nan,
                                     share_of_team_rushing_yards_s2d=lagged,
                                     team_rushing_yards_roll3=rng.uniform(90, 110),
                                     team_rushing_yards_s2d=rng.uniform(90, 110)))
    return pd.DataFrame(rows)


def test_integration_exact_rows_training_cutoff_and_metrics():
    panel = panel_fixture()
    rows, report = run_backtest(panel, "rushing_yards", [2022, 2023], n_bootstrap=100)
    assert len(rows) == 96
    assert all(report["pooled"][arm]["n"] == 96 for arm in ARMS)
    assert (rows.train_end_season < rows.season).all()
    pd.testing.assert_frame_equal(rows[KEYS].sort_values(KEYS).reset_index(drop=True),
                                  panel.loc[panel.season > 2021, KEYS].sort_values(KEYS).reset_index(drop=True))
    for arm in ARMS:
        assert report["pooled"][arm]["mae"] == pytest.approx(np.abs(rows[arm] - rows.actual_share).mean())
    assert report["rolling3_null_to_zero_rows"] == 16
    assert (rows.loc[rows.week == 1, "rolling3"] == 0).all()
    assert not report["team_renormalization"]
    assert all(f["fit"]["used_training_rows"] == f["n_train"] for f in report["folds"])
    assert report["comparisons"]["mixed_effects_minus_rolling3"]["status"] == "ok"


def test_future_fold_labels_cannot_change_earlier_predictions():
    panel = panel_fixture()
    rows, _ = run_backtest(panel, "rushing_yards", [2022, 2023], n_bootstrap=100)
    changed = panel.copy()
    changed.loc[changed.season == 2023, "share_of_team_rushing_yards"] = 0.9
    new_rows, _ = run_backtest(changed, "rushing_yards", [2022, 2023], n_bootstrap=100)
    # No model/preprocessor/optimizer choice uses either held-out fold's labels.
    np.testing.assert_array_equal(rows[ARMS], new_rows[ARMS])


@pytest.mark.parametrize("problem", ["duplicate", "null_truth", "inf_feature", "missing_column", "empty_fold"])
def test_invalid_panel_fails_before_fit(problem):
    panel = panel_fixture()
    if problem == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    elif problem == "null_truth":
        panel.loc[0, "share_of_team_rushing_yards"] = np.nan
    elif problem == "inf_feature":
        panel.loc[0, "team_rushing_yards_roll3"] = np.inf
    elif problem == "missing_column":
        panel = panel.drop(columns="slot")
    else:
        panel = panel[panel.season != 2022]
    with pytest.raises(ValueError):
        validate_panel(panel, "rushing_yards", [2022, 2023])


def _input_files(tmp_path):
    panel = panel_fixture()
    slotcols = ["player_id", "season", "week", "team", "slot", "slot_rank", "depth_chart_rank", "roster_snap_share_s2d"]
    slots = panel[slotcols].rename(columns={"roster_snap_share_s2d": "snap_share_s2d"})
    shares = panel.drop(columns=["slot", "slot_rank", "depth_chart_rank", "roster_snap_share_s2d"])
    db = tmp_path / "input.db"
    with sqlite3.connect(db) as conn:
        shares.to_sql("team_week_player_shares", conn, index=False)
    csv = tmp_path / "slots.csv"
    slots.to_csv(csv, index=False)
    return db, csv


def _args(db, csv, output):
    return ["--db", str(db), "--slots-csv", str(csv), "--target", "rushing_yards",
            "--seasons", "2021", "2023", "--test-seasons", "2022", "2023",
            "--output-dir", str(output), "--n-bootstrap", "100"]


def test_cli_default_preflight_never_trains_or_changes_inputs(tmp_path, monkeypatch):
    db, csv = _input_files(tmp_path)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (db, csv)]
    def forbidden(*a, **kw):
        raise AssertionError("preflight must not train")
    monkeypatch.setattr("scripts.evaluate_plan_b_shares.run_backtest", forbidden)
    out = tmp_path / "preflight"
    assert main(_args(db, csv, out)) == 0
    assert json.loads((out / "preflight.json").read_text())["status"] == "ready_to_attempt_fit"
    assert not (out / "predictions.csv").exists()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (db, csv)]
    with pytest.raises(SystemExit):
        main(_args(db, csv, out))


def test_cli_run_verifies_saved_metrics(tmp_path):
    db, csv = _input_files(tmp_path)
    preflight = tmp_path / "preflight"
    assert main(_args(db, csv, preflight)) == 0
    out = tmp_path / "run"
    assert main(_args(db, csv, out) + ["--run", "--preflight-manifest", str(preflight / "manifest.json")]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["saved_row_mae_verified"]
    saved = pd.read_csv(out / "predictions.csv")
    assert report["pooled"]["mixed_effects"]["mae"] == pytest.approx(np.abs(saved.mixed_effects - saved.actual_share).mean())


def test_cli_fit_failure_does_not_publish_pooled_result(tmp_path, monkeypatch):
    db, csv = _input_files(tmp_path)
    preflight = tmp_path / "preflight"
    assert main(_args(db, csv, preflight)) == 0
    def fail(*a, **kw):
        raise MixedEffectsFitError("synthetic nonconvergence")
    monkeypatch.setattr("scripts.evaluate_plan_b_shares.run_backtest", fail)
    out = tmp_path / "failure"
    assert main(_args(db, csv, out) + ["--run", "--preflight-manifest", str(preflight / "manifest.json")]) == 2
    assert "nonconvergence" in json.loads((out / "failure.json").read_text())["error"]
    assert not (out / "report.json").exists()


@pytest.mark.parametrize("change", ["slots", "panel"])
def test_cli_refuses_inputs_changed_after_preflight(tmp_path, change):
    db, csv = _input_files(tmp_path)
    preflight = tmp_path / "preflight"
    assert main(_args(db, csv, preflight)) == 0
    if change == "slots":
        slots = pd.read_csv(csv)
        slots.loc[0, "snap_share_s2d"] += 0.01
        slots.to_csv(csv, index=False)
    else:
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE team_week_player_shares SET share_of_team_rushing_yards=0.8 WHERE player_id='p0_1' AND season=2021 AND week=1")
    out = tmp_path / "changed"
    assert main(_args(db, csv, out) + ["--run", "--preflight-manifest", str(preflight / "manifest.json")]) == 2
    assert not (out / "report.json").exists()
    assert "differs from the passed preflight" in json.loads((out / "failure.json").read_text())["error"]


def test_cli_missing_slots_stops_cleanly(tmp_path):
    db, csv = _input_files(tmp_path)
    args = _args(db, csv, tmp_path / "missing")
    args = args[:2] + args[4:]
    assert main(args) == 2
    failure = json.loads((tmp_path / "missing" / "failure.json").read_text())
    assert "team_week_roster_slots table not found" in failure["error"]
