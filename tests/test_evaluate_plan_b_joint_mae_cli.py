"""CLI-level checks for scripts/evaluate_plan_b_joint_mae.py's provenance
guard: any 2023+ confirmation must either verify against a prior run
(matched rows/labels) or explicitly declare --first-confirmation-run
(population-completeness verification against its own preflight coverage)."""
import json
import sqlite3

import pytest

from scripts.evaluate_plan_b_joint_mae import main
from scripts.evaluate_plan_b_shares import main as preflight_main
from tests.test_team_hierarchical_backtester import _input_files


def _input_files_with_team_total(tmp_path):
    """panel_fixture()/_input_files() was built for the mixed-effects
    backtester, which never needs a same-week team total. The joint-MAE
    backtester does (run_backtest requires team_{target}) -- add it here
    rather than touching the shared fixture other tests depend on."""
    db, csv = _input_files(tmp_path)
    with sqlite3.connect(db) as conn:
        # A fixed constant, not a per-row copy of the noisy roll3 column --
        # run_backtest requires the same-week team total to agree across
        # every player on the same team-week, which panel_fixture()'s
        # independently-randomized roll3 values do not.
        conn.execute("ALTER TABLE team_week_player_shares ADD COLUMN team_rushing_yards REAL")
        conn.execute("UPDATE team_week_player_shares SET team_rushing_yards = 100.0")
        conn.commit()
    return db, csv


def _preflight(tmp_path, db, csv):
    out = tmp_path / "preflight"
    assert preflight_main([
        "--db", str(db), "--slots-csv", str(csv), "--target", "rushing_yards",
        "--seasons", "2021", "2023", "--test-seasons", "2023", "--output-dir", str(out),
    ]) == 0
    return out


def _run_args(preflight, output, extra):
    return ["--target", "rushing_yards", "--preflight-dir", str(preflight),
            "--test-seasons", "2023", "--output-dir", str(output), "--n-bootstrap", "100", *extra]


def test_2023_confirmation_requires_prior_run_or_first_confirmation_flag(tmp_path):
    db, csv = _input_files_with_team_total(tmp_path)
    preflight = _preflight(tmp_path, db, csv)
    with pytest.raises(SystemExit):
        main(_run_args(preflight, tmp_path / "out", []))


def test_prior_run_predictions_and_first_confirmation_run_are_mutually_exclusive(tmp_path):
    db, csv = _input_files_with_team_total(tmp_path)
    preflight = _preflight(tmp_path, db, csv)
    fake_prior = tmp_path / "fake_prior.csv"
    fake_prior.write_text("player_id,season,week,team,position,actual_share,rolling3\n")
    with pytest.raises(SystemExit):
        main(_run_args(preflight, tmp_path / "out",
                       ["--prior-run-predictions", str(fake_prior), "--first-confirmation-run"]))


def test_first_confirmation_run_succeeds_on_a_population_complete_preflight(tmp_path):
    db, csv = _input_files_with_team_total(tmp_path)
    preflight = _preflight(tmp_path, db, csv)
    coverage = json.loads((preflight / "coverage.json").read_text())
    assert coverage["excluded_rows"] == 0  # this fixture's slots cover every eligible row
    output = tmp_path / "out"
    assert main(_run_args(preflight, output, ["--first-confirmation-run"])) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["first_confirmation_run"]["excluded_rows"] == 0


def test_first_confirmation_run_rejects_an_incomplete_population(tmp_path):
    """The population-completeness substitute check must actually gate on
    coverage.json, not just require the flag to be present."""
    db, csv = _input_files_with_team_total(tmp_path)
    preflight = _preflight(tmp_path, db, csv)
    coverage_path = preflight / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["excluded_rows"] = 5  # simulate a capped/incomplete population
    coverage_path.write_text(json.dumps(coverage))
    output = tmp_path / "out"
    assert main(_run_args(preflight, output, ["--first-confirmation-run"])) == 2
    failure = json.loads((output / "failure.json").read_text())
    assert "population-complete" in failure["error"]
