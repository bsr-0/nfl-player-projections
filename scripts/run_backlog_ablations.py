#!/usr/bin/env python3
"""Run isolated, leakage-safe ablations for the ranked feature backlog.

Each arm removes one feature family from the existing causal feature list and
reuses the normal Phase 4 walk-forward evaluator. No production artifact is
written; outputs go under data/experiments/backlog_ablations/.
"""
from __future__ import annotations
import argparse, json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config.settings as settings
from src.models.single_week_ppr.evaluate import run_final_validation

ARMS = {
    "dvoa_opponent_fpa": ("opp_fpts_allowed", "opp_fpts_allowed_dvoa_adjusted_lag1"),
    "weather": ("wind_speed_mph", "precipitation_flag", "temperature_bucket", "is_dome"),
    "redzone_goal_line": ("redzone_target_share_pct_roll3_mean", "goal_line_carry_share_pct_roll3_mean"),
    "coordinator_scheme": ("coaching_change", "coaching_adaptation_score", "coaching_stability", "coaching_change_impact"),
    "wopr": ("wopr_roll3",),
    "personnel": ("team_pct_11_personnel_roll3_mean", "team_pct_12_personnel_roll3_mean", "team_pct_13_personnel_roll3_mean", "team_pct_21_personnel_roll3_mean", "team_pct_22_personnel_roll3_mean"),
}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--positions", nargs="+", default=["QB", "RB", "WR", "TE"])
    ap.add_argument("--arms", nargs="*", choices=["all", *ARMS], default=["all"])
    ap.add_argument("--output-dir", type=Path, default=Path("data/experiments/backlog_ablations"))
    args = ap.parse_args()
    arms = list(ARMS) if "all" in args.arms else args.arms
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {p: list(v) for p, v in settings.CAUSAL_FEATURES.items()}
    manifest = {"arms": arms, "seasons": args.seasons, "positions": args.positions, "feature_groups": ARMS}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for arm in arms:
            settings.CAUSAL_FEATURES = {
                p: [f for f in fs if f not in ARMS[arm]] for p, fs in baseline.items()
            }
            out = args.output_dir / f"{arm}.csv"
            run_final_validation(positions=args.positions, seasons=args.seasons, output_path=out)
    finally:
        settings.CAUSAL_FEATURES = baseline
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
