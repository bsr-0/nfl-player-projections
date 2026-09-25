#!/usr/bin/env python3
"""Train/load serialized Plan A artifacts and export exact served rows.

Each requested test season is predicted by an artifact trained strictly on
prior seasons. The artifact is loaded again before prediction, exercising the
same serialization boundary used by serving. This generic artifact path is a
serving bridge; target-specific blend/two-stage promotion artifacts can adopt
the same contract once their final model configuration is frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.team_allocation.features import VOLUME_COLS, filter_population, load_share_rows
from src.models.team_allocation.serving import (
    TARGET_ARMS,
    PlanAShareArtifact,
    fit_validated_artifact,
    predict_validated_artifact,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--target", choices=VOLUME_COLS, action="append")
    ap.add_argument("--artifact-dir", type=Path, default=Path("data/experiments/plan_a_served_artifacts"))
    ap.add_argument("--output", type=Path, default=Path("data/experiments/plan_a_served_rows.csv"))
    args = ap.parse_args()
    targets = args.target or list(VOLUME_COLS)
    rows = []
    for target in targets:
        all_rows = load_share_rows()
        for season in sorted(args.seasons):
            train = all_rows[all_rows.season < season]
            test = filter_population(all_rows[all_rows.season == season].copy(), target)
            if train.empty or test.empty:
                raise SystemExit(f"No train/test rows for {target}/{season}")
            artifact_path = args.artifact_dir / target / f"xgb_pre{season}.joblib"
            artifact = fit_validated_artifact(train, target, season)
            artifact.save(artifact_path)
            loaded = PlanAShareArtifact.load(artifact_path)
            prediction = predict_validated_artifact(loaded, test)
            out = test[["player_id", "season", "week", "team"]].copy()
            out["target"] = target
            out["predicted_share"] = prediction
            out["arm"] = loaded.payload["arm"]
            out["model_version"] = f"{loaded.payload['schema_version']}:{target}:{loaded.payload['arm']}:pre{season}"
            rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata = {
        "schema_version": "plan-a-served-export-v1",
        "targets": targets,
        "test_seasons": sorted(args.seasons),
        "rows": len(result),
        "artifact_dir": str(args.artifact_dir),
    }
    result_meta = args.output.with_suffix(".json")
    result_meta.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
