"""The comparator must reject misleading coverage, labels, and forecast dates."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from config.settings import SCORING
from scripts.compare_ppr_predictions import run_comparison
from src.evaluation.paired_ppr_comparison import (
    KEY, SCORING_DEFINITION, check_manifest, comparison_report, file_sha256,
    load_predictions, pair_predictions, paired_week_interval,
)
from src.evaluation.ppr_truth import TARGETS
from src.utils.helpers import calculate_fantasy_points_df


@pytest.fixture
def panel():
    raw = pd.DataFrame([
        ["001", 2024, 1, "A", "QB"],
        ["002", 2024, 1, "A", "WR"],
        ["001", 2024, 3, "A", "QB"],
        ["002", 2024, 3, "B", "WR"],  # target team, after a trade
        ["003", 2024, 3, "B", "RB"],
    ], columns=KEY)
    for col in TARGETS:
        raw[col] = 0.0
    raw.loc[0, ["rushing_yards", "receiving_yards", "receptions"]] = [-12, -3, 1]
    raw.loc[1, ["passing_yards", "passing_tds", "interceptions"]] = [25, 1, 1]
    raw.loc[2, "receiving_tds"] = 1
    actual = calculate_fantasy_points_df(raw[TARGETS])
    base = raw[KEY].assign(recorded_actual=actual, prediction=actual + [2, 4, -2, 2, 0])
    cand = raw[KEY].assign(recorded_actual=actual, prediction=actual + [1, 1, 1, 0, 0])
    return raw, base, cand


def metadata(path, frame, label="test"):
    return {"schema_version": 1, "label": label, "key_semantics": "target_game",
            "scoring_definition": SCORING_DEFINITION,
            "scoring_weights": {target: SCORING[target] for target in TARGETS},
            "rows_file": path.name, "rows_sha256": file_sha256(path),
            "prediction_column": "prediction", "actual_column": "recorded_actual",
            "folds": [{"test_season": int(s), "trained_through_season": int(s) - 1,
                       "selection_through_season": None} for s in sorted(frame.season.unique())],
            "provenance": {"test_fixture": True}}


def export(tmp_path, frame, label):
    path = tmp_path / f"{label}.csv"
    frame.to_csv(path, index=False)
    manifest = tmp_path / f"{label}.manifest.json"
    manifest.write_text(json.dumps(metadata(path, frame, label)))
    return manifest


def test_direct_raw_scoring_and_row_order(panel):
    raw, base, cand = panel
    rows, audit, excluded = pair_predictions(base.iloc[::-1], cand.sample(frac=1, random_state=1), raw.iloc[::-1])
    got = rows.set_index(["player_id", "week"]).actual_ppr
    assert got["001", 1] == -.5  # negative rushing AND receiving yards + a reception
    assert got["002", 1] == 3.0  # non-QB passing and interception
    assert got["001", 3] == 6.0  # QB receiving touchdown
    assert got["003", 3] == 0.0
    assert rows.negative_yards.sum() == 1
    assert rows.off_position_stat.sum() == 3
    assert audit["matched_rows"] == 5
    assert all(frame.empty for frame in excluded.values())
    report = comparison_report(rows, 100)
    assert report["pooled"]["baseline_mae"] == 2.0
    assert report["pooled"]["candidate_mae"] == .6
    assert report["pooled"]["delta_candidate_minus_baseline"] == -1.4
    assert set(report["by_position"]) == {"QB", "WR", "RB"}
    assert report["by_position"]["RB"]["paired_interval"]["status"] == "insufficient_week_blocks"


@pytest.mark.parametrize("arm", ["raw", "baseline", "candidate"])
@pytest.mark.parametrize("mutation", ["duplicate", "null_key", "blank_key", "nonfinite", "fractional_week"])
def test_invalid_inputs_fail(panel, arm, mutation):
    frames = dict(zip(["raw", "baseline", "candidate"], panel))
    frame = frames[arm].copy()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "null_key":
        frame.loc[0, "team"] = None
    elif mutation == "blank_key":
        frame.loc[0, "player_id"] = " "
    elif mutation == "fractional_week":
        frame["week"] = frame.week.astype(float)
        frame.loc[0, "week"] = 1.5
    else:
        frame.loc[0, "rushing_yards" if arm == "raw" else "prediction"] = np.inf
    frames[arm] = frame
    with pytest.raises(ValueError):
        pair_predictions(frames["baseline"], frames["candidate"], frames["raw"])


@pytest.mark.parametrize("arm", ["raw", "baseline", "candidate"])
def test_missing_row_fails(panel, arm):
    raw, base, cand = [frame.iloc[1:] if name == arm else frame
                       for name, frame in zip(["raw", "baseline", "candidate"], panel)]
    with pytest.raises(ValueError):
        pair_predictions(base, cand, raw)


def test_wrong_labels_and_team_fail(panel):
    raw, base, cand = panel
    bad = cand.copy()
    bad.loc[0, "recorded_actual"] = 0  # former floored-share label
    with pytest.raises(ValueError, match="recorded actual labels"):
        pair_predictions(base, bad, raw)
    bad = cand.copy()
    bad.loc[3, "team"] = "A"  # origin team cannot substitute for target team
    with pytest.raises(ValueError, match="key sets differ"):
        pair_predictions(base, bad, raw)


def test_explicit_population_counts_exclusions_and_requires_all_keys(panel):
    raw, base, cand = panel
    population = raw.iloc[2:][KEY]
    rows, audit, excluded = pair_predictions(base, cand.iloc[1:], raw, population)
    assert len(rows) == 3
    assert audit["baseline"]["excluded_rows"] == 2
    assert audit["candidate"]["excluded_rows"] == 1
    assert audit["baseline"]["excluded_by_season"] == {"2024": 2}
    assert len(excluded["baseline"]) == 2
    with pytest.raises(ValueError, match="missing 1 expected"):
        pair_predictions(base, cand.iloc[:-1], raw, population)
    with pytest.raises(ValueError, match="duplicate"):
        pair_predictions(base, cand, raw, pd.concat([population, population.iloc[:1]]))


def test_week_interval_uses_block_sums_and_counts(panel):
    raw, base, cand = panel
    rows, _, _ = pair_predictions(base, cand, raw)
    interval = paired_week_interval(rows, 100, 8)
    # Week 1: delta sum -4, n=2. Week 3: delta sum -3, n=3.
    draws = np.random.default_rng(8).integers(0, 2, size=(100, 2))
    delta = np.array([-4, -3])[draws].sum(axis=1) / np.array([2, 3])[draws].sum(axis=1)
    assert interval["ci_low"] == np.quantile(delta, .025)
    assert interval["ci_high"] == np.quantile(delta, .975)
    assert interval["significant_improvement"]
    assert paired_week_interval(rows.iloc[::-1], 100, 8) == interval
    rows.delta_abs_error = 0.0
    assert not paired_week_interval(rows, 100)["significant_improvement"]


@pytest.mark.parametrize("field,value", [
    ("key_semantics", "origin_week"), ("scoring_definition", "full_ppr"),
    ("scoring_weights", {}), ("provenance", {}), ("folds", []),
])
def test_incompatible_manifest_rejected(tmp_path, panel, field, value):
    _, base, _ = panel
    manifest_path = export(tmp_path, base, "base")
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_predictions(manifest_path)


@pytest.mark.parametrize("field", ["trained_through_season", "selection_through_season"])
def test_future_cutoffs_rejected(tmp_path, panel, field):
    _, base, _ = panel
    manifest_path = export(tmp_path, base, "base")
    manifest = json.loads(manifest_path.read_text())
    manifest["folds"][0][field] = 2024
    with pytest.raises(ValueError, match="held-out season"):
        check_manifest(manifest, base, "base")


def test_origin_and_hash_are_checked(tmp_path, panel):
    _, base, _ = panel
    manifest_path = export(tmp_path, base, "base")
    manifest = json.loads(manifest_path.read_text())
    bad = base.assign(origin_season=base.season, origin_week=base.week)
    with pytest.raises(ValueError, match="origin must precede"):
        check_manifest(manifest, bad, "base")
    # A bye/missing weekly row can make the next observed target week skip a number.
    good = base.iloc[2:].assign(origin_season=2024, origin_week=1)
    check_manifest(manifest, good, "base")
    (tmp_path / "base.csv").write_text("changed while running")
    with pytest.raises(ValueError, match="hash"):
        load_predictions(manifest_path)


def test_completed_run_preserves_inputs_and_recomputes_saved_mae(tmp_path, panel):
    raw, base, cand = panel
    bm, cm = export(tmp_path, base, "base"), export(tmp_path, cand, "candidate")
    raw_path = tmp_path / "truth.csv"
    raw.to_csv(raw_path, index=False)
    hashes = {path: file_sha256(path) for path in tmp_path.iterdir()}
    output = tmp_path / "result"
    report = run_comparison(bm, cm, raw_path, output, n_bootstrap=100)
    assert report["output"]["saved_mae_recomputed"]
    assert report["pooled"]["candidate_mae"] == .6
    assert (output / "report.json").exists()
    assert {path: file_sha256(path) for path in hashes} == hashes
    with pytest.raises(ValueError, match="already exists"):
        run_comparison(bm, cm, raw_path, output, n_bootstrap=100)


def test_parquet_export_uses_same_contract(tmp_path, panel):
    pytest.importorskip("pyarrow")
    _, base, _ = panel
    path = tmp_path / "predictions.parquet"
    base.to_parquet(path, index=False)
    manifest = tmp_path / "prediction.manifest.json"
    manifest.write_text(json.dumps(metadata(path, base)))
    loaded, _, _ = load_predictions(manifest)
    assert loaded.player_id.tolist() == base.player_id.tolist()
    assert loaded.prediction.tolist() == base.prediction.tolist()


def test_mismatch_writes_no_report(tmp_path, panel):
    raw, base, cand = panel
    bm, cm = export(tmp_path, base, "base"), export(tmp_path, cand.iloc[1:], "candidate")
    raw_path = tmp_path / "truth.csv"
    raw.to_csv(raw_path, index=False)
    with pytest.raises(ValueError, match="key sets differ"):
        run_comparison(bm, cm, raw_path, tmp_path / "result", n_bootstrap=100)
    assert not (tmp_path / "result").exists()


def test_shared_input_cannot_change_between_arm_loads(tmp_path, panel, monkeypatch):
    import scripts.compare_ppr_predictions as cli

    raw, base, cand = panel
    shared_path = tmp_path / "shared.csv"
    shared = base.assign(candidate_prediction=cand.prediction)
    shared.to_csv(shared_path, index=False)
    base_meta = metadata(shared_path, shared, "base")
    cand_meta = dict(base_meta, label="candidate", prediction_column="candidate_prediction")
    bm, cm = tmp_path / "base.json", tmp_path / "candidate.json"
    bm.write_text(json.dumps(base_meta))
    cm.write_text(json.dumps(cand_meta))
    raw_path = tmp_path / "truth.csv"
    raw.to_csv(raw_path, index=False)

    def load_then_mutate(path):
        result = load_predictions(path)
        if path == bm:
            shared.prediction += 1
            shared.candidate_prediction += 2
            shared.to_csv(shared_path, index=False)
            cand_meta["rows_sha256"] = file_sha256(shared_path)
            cm.write_text(json.dumps(cand_meta))
        return result

    monkeypatch.setattr(cli, "load_predictions", load_then_mutate)
    with pytest.raises(ValueError, match="input changed"):
        cli.run_comparison(bm, cm, raw_path, tmp_path / "result", n_bootstrap=100)
    assert not (tmp_path / "result").exists()


def test_prepare_reads_database_without_changing_source(tmp_path, panel):
    from scripts.prepare_plan_a_comparison import prepare
    import hashlib

    raw, base, cand = panel
    selector_dir = tmp_path / "selector"
    selector_dir.mkdir()
    rows = base[KEY].assign(fold=0, actual_ppr=base.recorded_actual,
                            predicted_ppr_baseline=base.prediction, predicted_ppr_selected=cand.prediction)
    rows.to_csv(selector_dir / "ppr_oof_rows.csv", index=False)
    source_hashes = {}
    for i in range(16):
        path = tmp_path / f"source_{i}.csv"
        path.write_text("immutable saved source")
        source_hashes[str(path)] = file_sha256(path)
    ordered = raw.sort_values(["season", "week", "team", "player_id"])
    truth_hash = hashlib.sha256(pd.util.hash_pandas_object(ordered[TARGETS + KEY], index=False).to_numpy().tobytes()).hexdigest()
    payload = {"scoring_definition": SCORING_DEFINITION, "input_sha256": source_hashes,
               "raw_truth_sha256": truth_hash,
               "comparison": {"pooled": {"n": 5}, "folds": {"0": {"selected_mae": .6, "baseline_mae": 2.0}}},
               "input_audit": {"folds": {"0": {"season": 2024, "rows": 5}}}}
    (selector_dir / "joint_selector.json").write_text(json.dumps(payload))
    database = tmp_path / "test.db"
    with sqlite3.connect(database) as connection:
        raw.to_sql("team_week_player_shares", connection, index=False)
    original = file_sha256(database)
    result = prepare(selector_dir, database, tmp_path / "prepared")
    assert result["rows"] == 5
    assert file_sha256(database) == original
    load_predictions(tmp_path / "prepared/plan_a.manifest.json")
    # A legitimate zero becoming nonzero must invalidate the old truth lineage.
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE team_week_player_shares SET rushing_yards = 1 WHERE player_id = '003'")
    with pytest.raises(ValueError, match="labels differ"):
        prepare(selector_dir, database, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
