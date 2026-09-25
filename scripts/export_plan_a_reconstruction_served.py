#!/usr/bin/env python3
"""Export and score the frozen reconstruction-aware Plan A serving path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.team_allocation.features import load_share_rows, filter_population
from src.models.team_allocation.reconstruct import reconstruct_partial_fantasy_points
from src.evaluation.team_share_backtester import bootstrap_mae_delta
from src.models.team_allocation.reconstruction_serving import ReconstructionArtifact


KEY = ["player_id", "season", "week", "team", "position"]


def _target_predictions(all_rows, target, seasons, artifact_dir):
    outputs = []
    for season in seasons:
        test = filter_population(all_rows[all_rows.season == season].copy(), target)
        artifact_path = artifact_dir / target / f"pre{season}.joblib"
        artifact = ReconstructionArtifact.fit(all_rows, target, season)
        artifact.save(artifact_path)
        loaded = ReconstructionArtifact.load(artifact_path)
        share, total = loaded.predict(test)
        out = test[KEY + [target, f"share_of_team_{target}_roll3", f"team_{target}_roll3"]].copy()
        out["predicted_share"] = share
        out["predicted_team_total"] = total
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def _score(rush, receive):
    merged = rush[KEY + ["rushing_yards", "predicted_share", "predicted_team_total", "share_of_team_rushing_yards_roll3", "team_rushing_yards_roll3"]].merge(
        receive[KEY + ["receiving_yards", "predicted_share", "predicted_team_total", "share_of_team_receiving_yards_roll3", "team_receiving_yards_roll3"]],
        on=KEY, how="outer", suffixes=("__rush", "__receive"),
    )
    group_keys = ["season", "week", "team"]
    # Broadcast a teammate's team total to QB rows, while retaining NaN for a
    # genuinely cold-start team-week. Those rows must be excluded, not scored
    # as a fabricated zero prediction.
    for c in ("predicted_team_total__rush", "predicted_team_total__receive", "team_rushing_yards_roll3", "team_receiving_yards_roll3"):
        merged[c] = merged.groupby(group_keys)[c].transform(lambda s: s.ffill().bfill())
    for c in ("receiving_yards", "predicted_share__receive"):
        merged[c] = merged[c].fillna(0.0)
    for c in ("rushing_yards", "predicted_share__rush"):
        merged[c] = merged[c].fillna(0.0)
    actual = reconstruct_partial_fantasy_points({
        "rushing_yards": merged["rushing_yards"].to_numpy(float),
        "receiving_yards": merged["receiving_yards"].to_numpy(float),
    })
    candidate = reconstruct_partial_fantasy_points({
        "rushing_yards": merged["predicted_share__rush"].to_numpy(float) * merged["predicted_team_total__rush"].to_numpy(float),
        "receiving_yards": merged["predicted_share__receive"].to_numpy(float) * merged["predicted_team_total__receive"].to_numpy(float),
    })
    baseline = reconstruct_partial_fantasy_points({
        "rushing_yards": merged["share_of_team_rushing_yards_roll3"].fillna(0).to_numpy(float) * merged["team_rushing_yards_roll3"].fillna(0).to_numpy(float),
        "receiving_yards": merged["share_of_team_receiving_yards_roll3"].fillna(0).to_numpy(float) * merged["team_receiving_yards_roll3"].fillna(0).to_numpy(float),
    })
    valid = np.isfinite(actual) & np.isfinite(candidate) & np.isfinite(baseline)
    return {
        "n_rows": int(valid.sum()),
        "candidate_mae": float(mean_absolute_error(actual[valid], candidate[valid])),
        "rolling3_mae": float(mean_absolute_error(actual[valid], baseline[valid])),
        "candidate_minus_rolling3": float(np.mean(np.abs(actual[valid] - candidate[valid]) - np.abs(actual[valid] - baseline[valid]))),
        "vs_rolling3_bootstrap": bootstrap_mae_delta(
            actual[valid].to_numpy(float), candidate[valid].to_numpy(float), baseline[valid].to_numpy(float)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--artifact-dir", type=Path, default=Path("data/experiments/plan_a_reconstruction_served_artifacts"))
    ap.add_argument("--output", type=Path, default=Path("data/experiments/plan_a_reconstruction_served_report.json"))
    args = ap.parse_args()
    all_rows = load_share_rows()
    rush = _target_predictions(all_rows, "rushing_yards", args.seasons, args.artifact_dir)
    receive = _target_predictions(all_rows, "receiving_yards", args.seasons, args.artifact_dir)
    report = {str(s): _score(rush[rush.season == s], receive[receive.season == s]) for s in args.seasons}
    payload = {"schema_version": "plan-a-reconstruction-served-report-v1", "seasons": args.seasons, "report": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
