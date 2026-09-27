#!/usr/bin/env python3
"""Merge Plan B's confirmed joint-MAE allocation into Plan A's existing
per-target full-PPR allocation CSVs, as one more candidate arm.

Writes a full set of 8 target CSVs (the 4 Plan B now covers, with its arm
attached, plus the 4 sparse targets copied through unchanged) into a new
directory consumable by scripts/evaluate_joint_ppr_selector.py's
--allocation-dir exactly as-is -- no changes to that script or to
src/evaluation/joint_ppr_selector.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS
from src.evaluation.plan_b_arm_adapter import (
    attach_plan_b_arm, reshape_plan_b_predictions, season_to_fold_map,
)
from src.evaluation.ppr_truth import KEY, checked_truth
from src.models.team_allocation.features import load_share_rows

PLAN_B_TARGETS = ["rushing_yards", "receiving_yards", "receptions", "passing_yards"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-allocation-dir", required=True, type=Path,
                    help="Existing per-target {target}_predictions.csv directory (Plan A's arms)")
    ap.add_argument("--plan-b-confirm-dir", required=True, type=Path,
                    help="data/experiments/plan_b_uncapped_joint_mae_confirm_.../ -- one subdir per target")
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument("--gap-seasons", type=int, default=None,
                    help="Defaults to config.settings.TEAM_ALLOCATION_MODEL_CONFIG['cv_gap_seasons']")
    ap.add_argument("--arm-name", default="joint_other_mae")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.output_dir.exists():
        ap.error(f"output already exists: {args.output_dir}; choose a new directory")

    from config.settings import TEAM_ALLOCATION_MODEL_CONFIG
    gap_seasons = (args.gap_seasons if args.gap_seasons is not None
                  else TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"])
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    fold_map = season_to_fold_map(seasons, args.n_test_seasons, gap_seasons)
    all_folds = set(fold_map.values())

    args.output_dir.mkdir(parents=True)
    manifest = {"source_allocation_dir": str(args.source_allocation_dir.resolve()),
                "plan_b_confirm_dir": str(args.plan_b_confirm_dir.resolve()),
                "seasons": seasons, "n_test_seasons": args.n_test_seasons,
                "gap_seasons": gap_seasons, "fold_map": fold_map, "arm_name": args.arm_name,
                "plan_b_targets": PLAN_B_TARGETS, "sha256": {}}

    raw = load_share_rows(seasons=seasons)
    canonical_alloc = pd.read_csv(args.source_allocation_dir / "rushing_yards_predictions.csv",
                                  usecols=KEY + ["arm"], float_precision="round_trip")
    canonical = canonical_alloc[canonical_alloc.arm.eq("rolling3")]
    truth = checked_truth(raw, canonical[KEY])

    for target in PPR_RECONSTRUCTION_TARGETS:
        source_path = args.source_allocation_dir / f"{target}_predictions.csv"
        dest_path = args.output_dir / f"{target}_predictions.csv"
        if target not in PLAN_B_TARGETS:
            shutil.copyfile(source_path, dest_path)
            manifest["sha256"][target] = {"source": sha256(source_path), "output": sha256(dest_path),
                                          "plan_b_arm_attached": False}
            continue
        existing = pd.read_csv(source_path, float_precision="round_trip")
        plan_b_wide = pd.read_csv(args.plan_b_confirm_dir / target / "predictions.csv",
                                  dtype={"player_id": str, "team": str, "position": str, "slot": str},
                                  float_precision="round_trip")
        reshaped = reshape_plan_b_predictions(plan_b_wide, args.arm_name, fold_map)
        updated = attach_plan_b_arm({"predictions": existing}, reshaped, target, truth, folds=all_folds)
        updated["predictions"].to_csv(dest_path, index=False)
        manifest["sha256"][target] = {"source": sha256(source_path), "output": sha256(dest_path),
                                      "plan_b_arm_attached": True,
                                      "plan_b_input_sha256": sha256(args.plan_b_confirm_dir / target / "predictions.csv"),
                                      "plan_b_rows_attached": int((updated["predictions"].arm == args.arm_name).sum())}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Wrote merged allocation CSVs (8 targets, {len(PLAN_B_TARGETS)} with Plan B arm attached): {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
