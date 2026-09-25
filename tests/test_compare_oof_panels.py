"""The row-level two-panel comparator -- the "missing half" the 2026-09-25
methodology review flagged: a panel with no way to compare two runs is not
yet an A/B instrument. These tests cover the join, the refusal guardrail,
and the cluster-CI reporting end to end.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_oof_panels as cmp  # noqa: E402

from src.models.oof_capture import add_experience_segments  # noqa: E402


def _panel(seasons_weeks, residuals, players=None, positions=None):
    n = len(seasons_weeks)
    players = players or [f"p{i}" for i in range(n)]
    positions = positions or ["QB"] * n
    df = pd.DataFrame({
        "player_id": players,
        "season": [s for s, w in seasons_weeks],
        "week": [w for s, w in seasons_weeks],
        "position": positions,
        "predicted_points": [10.0 + r for r in residuals],
        "actual_points": [10.0] * n,
        "residual": residuals,
    })
    return add_experience_segments(df)


def test_paired_frame_keeps_only_shared_player_weeks():
    baseline = _panel([(2024, 1), (2024, 2)], [1.0, 2.0], players=["a", "b"])
    variant = _panel([(2024, 1), (2024, 3)], [0.5, 9.0], players=["a", "c"])
    paired = cmp.paired_frame(baseline, variant)
    assert len(paired) == 1
    assert paired["player_id"].iloc[0] == "a"


def test_paired_delta_sign_is_variant_minus_baseline():
    baseline = _panel([(2024, 1)], [4.0], players=["a"])   # |error| = 4
    variant = _panel([(2024, 1)], [1.0], players=["a"])    # |error| = 1, more accurate
    paired = cmp.paired_frame(baseline, variant)
    assert paired["paired_delta"].iloc[0] == pytest.approx(1.0 - 4.0)


def test_refuses_to_report_when_overlap_is_too_small(tmp_path, capsys):
    baseline = _panel([(2024, w) for w in range(1, 41)], [1.0] * 40)
    variant = _panel([(2024, w) for w in range(1, 4)], [1.0] * 3,
                     players=[f"p{i}" for i in range(3)])
    base_path, var_path = tmp_path / "base.parquet", tmp_path / "var.parquet"
    baseline.to_parquet(base_path)
    variant.to_parquet(var_path)

    # Invoke via argv rather than calling internals directly, to exercise the
    # actual CLI refusal path end to end.
    sys.argv = ["compare_oof_panels.py", "--baseline", str(base_path),
               "--variant", str(var_path)]
    exit_code = cmp.main()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "REFUSING" in out


def test_reports_a_segment_table_when_overlap_is_sufficient(tmp_path, capsys):
    n = 40
    baseline = _panel([(2024, w) for w in range(1, n + 1)], [2.0] * n,
                      players=[f"p{i}" for i in range(n)])
    variant = _panel([(2024, w) for w in range(1, n + 1)], [1.0] * n,
                     players=[f"p{i}" for i in range(n)])
    base_path, var_path = tmp_path / "base.parquet", tmp_path / "var.parquet"
    baseline.to_parquet(base_path)
    variant.to_parquet(var_path)

    sys.argv = ["compare_oof_panels.py", "--baseline", str(base_path),
               "--variant", str(var_path), "--by", "position", "--n-boot", "200"]
    exit_code = cmp.main()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "mean_paired_delta" in out
    assert "straddles_zero" in out


def test_warns_when_pointed_at_the_flat_latest_pointer(tmp_path, capsys):
    panel = _panel([(2024, 1)], [1.0], players=["a"])
    pointer = tmp_path / "walk_forward_oof_predictions.parquet"
    panel.to_parquet(pointer)
    cmp._load(pointer)
    err = capsys.readouterr().err
    assert "overwritten by whichever run wrote last" in err


def test_segment_comparison_flags_a_ci_that_straddles_zero():
    n = 20
    paired = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "position": ["QB"] * n,
        # tiny, noisy deltas straddling zero -- no real effect.
        "paired_delta": [0.1 if i % 2 == 0 else -0.1 for i in range(n)],
    })
    report = cmp.segment_comparison(paired, by=("position",), n_boot=500)
    assert report["straddles_zero"].iloc[0] == True or report["straddles_zero"].iloc[0] is None


def test_segment_comparison_detects_a_clear_effect():
    n = 20
    paired = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "position": ["QB"] * n,
        "paired_delta": [-5.0] * n,  # variant consistently better, no noise
    })
    report = cmp.segment_comparison(paired, by=("position",), n_boot=500)
    assert report["straddles_zero"].iloc[0] == False
    assert report["ci_hi"].iloc[0] < 0
