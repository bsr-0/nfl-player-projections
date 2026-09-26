"""Multiple-comparison correction integrated into compare_oof_panels.py.

The whole point: a table with many segment cells run through independent
per-cell tests will show some cells as "significant" purely from testing
many of them, even with no real effect anywhere. `significant_corrected`
is what should survive that; `straddles_zero`/`p_value` alone should not be
read as a verdict once there is more than one cell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_oof_panels as cmp  # noqa: E402

from src.models.oof_capture import add_experience_segments  # noqa: E402


def _cell(position, n, delta_mean, delta_sd, seed):
    """One segment's worth of paired rows: n distinct player clusters (one
    row each), so the cluster bootstrap behaves like a per-row bootstrap and
    a real effect is resolvable at this n.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player_id": [f"{position}_{i}" for i in range(n)],
        "position": position,
        "paired_delta": rng.normal(delta_mean, delta_sd, size=n),
    })


def test_pure_noise_family_yields_fewer_significant_cells_after_correction():
    """20 cells, EVERY one built from noise centred on zero (no real effect
    anywhere). An uncorrected per-cell view should still show roughly one in
    twenty looking "significant" by chance; Holm correction should erase
    essentially all of them.
    """
    cells = pd.concat([
        _cell(f"pos{i}", n=30, delta_mean=0.0, delta_sd=1.0, seed=i)
        for i in range(20)
    ], ignore_index=True)

    report = cmp.segment_comparison(cells, by=("position",), n_boot=500,
                                    correction="holm", seed=0)
    n_uncorrected = int((report["straddles_zero"] == False).sum())
    n_corrected = int(report["significant_corrected"].sum())
    assert n_corrected <= n_uncorrected
    # Holm at alpha=0.05 across 20 pure-noise cells should reject very few,
    # typically zero -- allow a small margin rather than asserting exactly 0
    # to avoid a flaky test on an unlucky bootstrap draw.
    assert n_corrected <= 1


def test_a_clear_real_effect_survives_correction_even_with_many_cells():
    """A strong, unambiguous effect in one cell should not be erased by
    correcting for many surrounding noise cells -- correction should cost
    power at the margin, not destroy it entirely for an obvious signal.
    """
    noise_cells = pd.concat([
        _cell(f"noise{i}", n=30, delta_mean=0.0, delta_sd=1.0, seed=100 + i)
        for i in range(15)
    ], ignore_index=True)
    real_effect = _cell("real", n=40, delta_mean=-5.0, delta_sd=1.0, seed=999)
    cells = pd.concat([noise_cells, real_effect], ignore_index=True)

    report = cmp.segment_comparison(cells, by=("position",), n_boot=1000,
                                    correction="holm", seed=0)
    real_row = report[report["position"] == "real"].iloc[0]
    assert real_row["significant_corrected"] == True
    assert real_row["mean_paired_delta"] < 0


def test_correction_none_reproduces_uncorrected_per_cell_view():
    cells = pd.concat([
        _cell(f"pos{i}", n=30, delta_mean=0.0, delta_sd=1.0, seed=i)
        for i in range(5)
    ], ignore_index=True)
    report = cmp.segment_comparison(cells, by=("position",), n_boot=300, correction="none")
    assert report["significant_corrected"].isna().all()
    # straddles_zero / p_value are still computed regardless of correction choice.
    assert report["p_value"].notna().all()


def test_bh_never_rejects_fewer_cells_than_holm_on_the_same_table():
    cells = pd.concat([
        _cell(f"pos{i}", n=25, delta_mean=0.3 if i % 3 == 0 else 0.0, delta_sd=1.0, seed=i)
        for i in range(12)
    ], ignore_index=True)
    holm_report = cmp.segment_comparison(cells, by=("position",), n_boot=500,
                                         correction="holm", seed=0)
    bh_report = cmp.segment_comparison(cells, by=("position",), n_boot=500,
                                       correction="bh", seed=0)
    assert int(bh_report["significant_corrected"].sum()) >= int(holm_report["significant_corrected"].sum())


def test_untestable_single_cluster_cells_do_not_shrink_the_family():
    """A cell with only one player cluster can't be bootstrapped (NaN
    p-value) and must not eat into the multiplicity budget of the cells that
    CAN be tested.
    """
    testable = pd.concat([
        _cell(f"pos{i}", n=30, delta_mean=0.0, delta_sd=1.0, seed=i)
        for i in range(10)
    ], ignore_index=True)
    untestable = pd.DataFrame({
        "player_id": ["only_one"] * 5,
        "position": "single_cluster",
        "paired_delta": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    cells = pd.concat([testable, untestable], ignore_index=True)

    report = cmp.segment_comparison(cells, by=("position",), n_boot=500, correction="holm", seed=0)
    single_row = report[report["position"] == "single_cluster"].iloc[0]
    assert np.isnan(single_row["p_value"])
    assert single_row["significant_corrected"] == False

    only_testable = report[report["position"] != "single_cluster"]
    assert only_testable["p_value"].notna().all()


def test_cli_prints_correction_summary_and_significant_corrected_column(tmp_path, capsys):
    n = 40
    baseline_rows, variant_rows = [], []
    for pos, delta in (("QB", 0.0), ("RB", 0.0), ("WR", 0.0), ("TE", 0.0)):
        baseline_rows.append(pd.DataFrame({
            "player_id": [f"{pos}_{i}" for i in range(n)],
            "season": 2024, "week": [1 + i % 4 for i in range(n)],
            "position": pos, "predicted_points": 10.0, "actual_points": 10.0,
            "residual": 0.0,
        }))
        variant_rows.append(pd.DataFrame({
            "player_id": [f"{pos}_{i}" for i in range(n)],
            "season": 2024, "week": [1 + i % 4 for i in range(n)],
            "position": pos, "predicted_points": 10.0 + delta,
            "actual_points": 10.0, "residual": delta,
        }))
    baseline = add_experience_segments(pd.concat(baseline_rows, ignore_index=True))
    variant = add_experience_segments(pd.concat(variant_rows, ignore_index=True))

    base_path, var_path = tmp_path / "base.parquet", tmp_path / "var.parquet"
    baseline.to_parquet(base_path)
    variant.to_parquet(var_path)

    sys.argv = ["compare_oof_panels.py", "--baseline", str(base_path),
               "--variant", str(var_path), "--by", "position", "--n-boot", "300"]
    exit_code = cmp.main()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "significant_corrected" in out
    assert "after Holm" in out or "Holm (family-wise" in out


def test_cli_correction_none_warns_about_per_cell_rate(tmp_path, capsys):
    n = 40
    baseline = add_experience_segments(pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)], "season": 2024,
        "week": [1 + i % 4 for i in range(n)], "position": "QB",
        "predicted_points": 10.0, "actual_points": 10.0, "residual": 0.0,
    }))
    variant = baseline.copy()
    base_path, var_path = tmp_path / "base.parquet", tmp_path / "var.parquet"
    baseline.to_parquet(base_path)
    variant.to_parquet(var_path)

    sys.argv = ["compare_oof_panels.py", "--baseline", str(base_path),
               "--variant", str(var_path), "--correction", "none", "--n-boot", "200"]
    cmp.main()
    out = capsys.readouterr().out
    assert "applies to EACH cell individually" in out
