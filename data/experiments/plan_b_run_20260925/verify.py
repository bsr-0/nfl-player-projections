"""Independent saved-row and input-lineage verification for this research run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREFLIGHT = ROOT.parent / "plan_b_preflight_20260925"
TARGETS = ("targets", "rushing_attempts", "receiving_yards", "rushing_yards")
KEYS = ["player_id", "season", "week", "team", "position"]
ARMS = ("rolling3", "fixed_ridge", "mixed_effects")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict:
    summary = {"status": "verified", "metric": "held_out_player_week_share_mae",
               "test_seasons": [2023, 2024, 2025], "targets": {},
               "comparison": "mixed_effects_minus_control; positive means mixed effects is worse"}
    code_hashes = None
    slot_hash = None
    heldout_keys = {}
    for target in TARGETS:
        here = ROOT / target
        report = json.loads((here / "report.json").read_text())
        run_manifest = json.loads((here / "manifest.json").read_text())
        preflight_path = PREFLIGHT / target / "manifest.json"
        preflight = json.loads(preflight_path.read_text())
        assert report["status"] == "complete" and report["saved_row_mae_verified"]
        assert run_manifest["mode"] == "run" and run_manifest["target"] == target
        assert run_manifest["preflight_manifest"]["sha256"] == sha256(preflight_path)
        assert run_manifest["slots_csv"]["sha256"] == preflight["slots_csv"]["sha256"]
        assert run_manifest["joined_input"]["sha256"] == preflight["joined_input"]["sha256"]
        assert run_manifest["joined_input"]["rows"] == preflight["joined_input"]["rows"]
        assert sha256(here / "input_panel.csv") == preflight["joined_input"]["sha256"]
        assert sha256(here / "predictions.csv") == report["predictions_sha256"]
        assert code_hashes is None or code_hashes == run_manifest["code_sha256"]
        code_hashes = run_manifest["code_sha256"]
        assert slot_hash is None or slot_hash == run_manifest["slots_csv"]["sha256"]
        slot_hash = run_manifest["slots_csv"]["sha256"]

        rows = pd.read_csv(here / "predictions.csv", dtype={"player_id": str, "team": str, "position": str},
                           float_precision="round_trip")
        input_rows = pd.read_csv(here / "input_panel.csv",
                                 usecols=KEYS + [f"share_of_team_{target}"],
                                 dtype={"player_id": str, "team": str, "position": str},
                                 float_precision="round_trip")
        expected = input_rows.loc[input_rows.season.isin(summary["test_seasons"])].copy()
        expected = expected.sort_values(KEYS).reset_index(drop=True)
        got = rows.sort_values(KEYS).reset_index(drop=True)
        assert not rows.duplicated(KEYS).any()
        pd.testing.assert_frame_equal(got[KEYS], expected[KEYS])
        assert np.allclose(got.actual_share.to_numpy(),
                           expected[f"share_of_team_{target}"].to_numpy(), rtol=0, atol=1e-15)
        assert np.isfinite(rows[list(ARMS) + ["actual_share"]].to_numpy(float)).all()
        assert ((rows[list(ARMS)] >= 0) & (rows[list(ARMS)] <= 1)).all().all()
        assert (rows.train_end_season < rows.season).all()
        assert all(f["fit"]["converged"] and not f["fit"]["fallback"]
                   and f["prediction"]["mean_fallback_rows"] == 0 for f in report["folds"])
        assert sum(f["n_test"] for f in report["folds"]) == len(rows)
        for fold in report["folds"]:
            assert int((rows.season == fold["test_season"]).sum()) == fold["n_test"]

        mae = {arm: float(np.abs(rows[arm] - rows.actual_share).mean()) for arm in ARMS}
        for arm in ARMS:
            assert report["pooled"][arm]["n"] == len(rows)
            assert abs(mae[arm] - report["pooled"][arm]["mae"]) < 1e-12
        intervals = {}
        for control in ("rolling3", "fixed_ridge"):
            key = f"mixed_effects_minus_{control}"
            result = report["comparisons"][key]
            delta = float((np.abs(rows.mixed_effects - rows.actual_share)
                           - np.abs(rows[control] - rows.actual_share)).mean())
            assert abs(delta - result["point_estimate"]) < 1e-12
            assert abs(delta - (mae["mixed_effects"] - mae[control])) < 1e-12
            intervals[control] = {"delta": delta, "ci_low": result["ci_low"],
                                  "ci_high": result["ci_high"], "bootstrap_method": result["method"],
                                  "interval_excludes_zero": bool(result["ci_high"] < 0 or result["ci_low"] > 0)}
        season_counts = {str(int(s)): int(n) for s, n in rows.groupby("season").size().items()}
        warnings = {str(f["test_season"]): f["fit"]["attempts"][-1]["warnings"] for f in report["folds"]}
        summary["targets"][target] = {"heldout_rows": len(rows), "heldout_by_season": season_counts,
                                      "mae": mae, "paired_intervals": intervals,
                                      "converged_folds": len(report["folds"]),
                                      "fit_warnings_by_season": warnings,
                                      "model_clipped_rows": sum(f["clipped_rows"]["mixed_effects"] for f in report["folds"]),
                                      "rolling3_null_to_zero_rows": sum(f["rolling3_null_to_zero_rows"] for f in report["folds"]),
                                      "represented_rows": report["coverage"]["represented_rows"],
                                      "excluded_rows": report["coverage"]["excluded_rows"],
                                      "predictions_sha256": report["predictions_sha256"],
                                      "joined_input_sha256": preflight["joined_input"]["sha256"]}
        heldout_keys[target] = got[KEYS]
    pd.testing.assert_frame_equal(heldout_keys["targets"], heldout_keys["receiving_yards"])
    pd.testing.assert_frame_equal(heldout_keys["rushing_attempts"], heldout_keys["rushing_yards"])
    summary["slot_csv_sha256"] = slot_hash
    summary["code_sha256"] = code_hashes
    summary["limitations"] = [
        "Four share targets only; no PPR or full joint-team result.",
        "Roster caps exclude many canonical player-weeks; metrics are on represented rows.",
        "All mixed-effects folds warned that the random-effects covariance is singular or at the boundary.",
        "Intervals use paired calendar-week blocks within season and do not cover serial dependence across weeks.",
    ]
    return summary


if __name__ == "__main__":
    result = verify()
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for target, item in result["targets"].items():
        print(target, item["heldout_rows"], item["mae"],
              item["paired_intervals"]["rolling3"])
