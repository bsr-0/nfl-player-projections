#!/usr/bin/env python3
"""Evaluate a chronologically out-of-sample simulation run.

Inputs must be produced before the actual games occurred. This script cannot
make an in-sample run honest; it records the input paths in the result so the
calling backtest can audit provenance.

Usage:
  python scripts/evaluate_simulation.py \
    --player-draws data/simulations/2025/simulation_player_draws.parquet \
    --game-draws data/simulations/2025/simulation_game_draws.parquet \
    --player-actuals data/eval/player_actuals_2025.csv \
    --game-actuals data/eval/game_actuals_2025.csv \
    --output data/eval/simulation_2025_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simulation_evaluation import (
    evaluate_joint_player_draws,
    game_draw_calibration,
    marginal_calibration,
)
from src.models.simulation_schema import summarize_player_draws

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        raise ValueError("JSON table input must be a record list, not a full simulation payload")
    raise ValueError(f"unsupported input type: {path.suffix}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player-draws", required=True, type=Path)
    parser.add_argument("--game-draws", required=True, type=Path)
    parser.add_argument("--player-actuals", required=True, type=Path)
    parser.add_argument("--game-actuals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    player_draws = _read_table(args.player_draws)
    game_draws = _read_table(args.game_draws)
    player_actuals = _read_table(args.player_actuals)
    game_actuals = _read_table(args.game_actuals)
    player_summary = pd.DataFrame(summarize_player_draws(
        player_draws.to_dict(orient="records")))

    result = {
        "evaluation_type": "simulation_holdout",
        "input_provenance": {
            "player_draws": str(args.player_draws),
            "game_draws": str(args.game_draws),
            "player_actuals": str(args.player_actuals),
            "game_actuals": str(args.game_actuals),
        },
        "marginal": marginal_calibration(player_summary, player_actuals),
        "joint_player": evaluate_joint_player_draws(player_draws, player_actuals),
        "game": game_draw_calibration(game_draws, game_actuals),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
