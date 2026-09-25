"""The walk-forward log parser must not mis-attribute folds.

`train.py --walk-forward` prints only `mean +/- std` and discards the
per-fold metrics, so comparing two runs means recovering those numbers from
the logs. A parser that quietly assigns a fold's metrics to the wrong season
would produce a confident, wrong A/B verdict -- worse than no comparison. The
cases pinned here are the ones that would do that: the pre-walk-forward data
load emitting a fold-like line, folds appearing in non-ascending order, and
a fold that failed partway and produced only some positions.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_walkforward_runs import parse_fold_metrics_json, parse_log  # noqa: E402


def _fold_block(season, metrics):
    """Reproduce the real log shape for one fold."""
    out = [f"  Test season: {season}", "Loading data for QB..."]
    for position, (rmse, mae, r2) in metrics.items():
        out += [
            f"  Saved {position} model to /tmp/x/model_{position.lower()}_1w.joblib",
            "",
            f"{position} Training Metrics:",
            f"1w: RMSE={rmse}, MAE={mae}, R²={r2}",
            "=" * 40,
        ]
    return out


FULL = {"QB": ("7.80", "6.36", "0.246"), "RB": ("6.76", "4.89", "0.263"),
        "WR": ("6.36", "4.45", "0.311"), "TE": ("4.43", "2.93", "0.350")}


def _write(tmp_path, lines, name="run.log"):
    path = tmp_path / name
    path.write_text("\n".join(lines))
    return path


def test_parses_each_fold_with_its_own_test_season(tmp_path):
    lines = ["[1/5] Loading training data...", "  Test season: 2026",
             "Loading data for QB...", "  Loaded 13042 records for QB"]
    for season in (2023, 2024, 2025):
        lines += _fold_block(season, FULL)
    folds, _ = parse_log(_write(tmp_path, lines))

    assert [season for season, _ in folds] == [2023, 2024, 2025]
    assert folds[0][1]["QB"]["mae"] == 6.36


def test_the_initial_data_load_is_not_counted_as_a_fold(tmp_path):
    """The pre-walk-forward load emits "Loading data for QB" too. Counting it
    shifts every fold's season label by one -- the mis-attribution that would
    silently invert an A/B verdict.
    """
    lines = ["  Test season: 2026", "Loading data for QB...",
             "  Loaded 13042 records for QB"]
    lines += _fold_block(2023, FULL)
    folds, _ = parse_log(_write(tmp_path, lines))

    assert len(folds) == 1
    assert folds[0][0] == 2023


def test_a_fold_with_only_some_positions_is_kept_partially(tmp_path):
    """A fold that failed after WR must contribute its completed positions and
    simply be absent for the rest, not drop out entirely or report zeros.
    """
    partial = {k: FULL[k] for k in ("QB", "RB")}
    lines = _fold_block(2023, FULL) + _fold_block(2024, partial)
    folds, _ = parse_log(_write(tmp_path, lines))

    assert len(folds) == 2
    assert set(folds[1][1]) == {"QB", "RB"}
    assert "TE" not in folds[1][1]


def test_summary_line_is_recovered_for_contrast(tmp_path):
    lines = _fold_block(2023, FULL) + [
        "Walk-Forward Validation Summary (mean +/- std)",
        "  QB: RMSE 7.93 +/- 0.30  MAE 6.37 +/- 0.27",
        "  TE: RMSE 4.44 +/- 0.08  MAE 3.06 +/- 0.01",
    ]
    _, summary = parse_log(_write(tmp_path, lines))
    assert summary["QB"]["mae"] == 6.37 and summary["QB"]["mae_std"] == 0.27
    assert summary["TE"]["rmse"] == 4.44


def test_optuna_noise_is_not_mistaken_for_metrics(tmp_path):
    """Tuning output contains RMSE-like values on nearly every line."""
    lines = _fold_block(2023, FULL)
    lines[3:3] = [
        "[I 2026-09-24] Trial 6 finished with value: 0.4286690356726823",
        "  Tuning Random Forest (100 trials)...",
        "1w: RMSE=99.99, MAE=99.99, R²=0.999  <- not preceded by a header",
    ]
    folds, _ = parse_log(_write(tmp_path, lines))
    assert folds[0][1]["QB"]["mae"] == 6.36, "matched an unheaded metric line"


def test_no_folds_when_the_log_has_no_metric_blocks(tmp_path):
    lines = ["  Test season: 2026", "Loading data for QB...", "crashed early"]
    folds, _ = parse_log(_write(tmp_path, lines))
    assert folds == []


def test_json_path_is_full_precision(tmp_path):
    """The JSON artifact exists because the log rounds to 2dp."""
    path = tmp_path / "walk_forward_fold_metrics.json"
    path.write_text(json.dumps({
        "folds": [
            {"test_season": 2023, "by_position": {"QB": {"mae": 6.3612345, "rmse": 7.8}}},
            {"test_season": 2024, "by_position": {"QB": {"mae": 6.3634567, "rmse": 7.9}}},
        ]
    }))
    folds, summary = parse_fold_metrics_json(path)

    assert [s for s, _ in folds] == [2023, 2024]
    delta = folds[1][1]["QB"]["mae"] - folds[0][1]["QB"]["mae"]
    assert 0 < delta < 0.005, "a delta invisible at 2dp must survive via JSON"
    assert summary == {}
