#!/usr/bin/env python3
"""Score a published calibrated-simulation week against realized outcomes.

Reads the Parquet tables generate_simulation_data.py --write-parquet wrote for
a week, before its games were played, and scores them once the actuals exist.
This cannot make a run honest after the fact; it records the input paths so the
caller can audit when they were produced.

Usage:
  python scripts/evaluate_simulation.py \
    --player-draws data/simulations/2026/simulation_2026_wk5_player_draws.parquet \
    --players data/simulations/2026/simulation_2026_wk5_players.parquet \
    --player-actuals data/eval/player_actuals_2026_wk5.csv \
    --output data/eval/simulation_2026_wk5_metrics.json

--player-actuals needs game_id, player_id and fantasy_points for every
simulated player (players who did not play must be present, or the run fails).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simulation_evaluation import evaluate_joint_player_draws, marginal_calibration  # noqa: E402
from src.models.simulation_schema import summarize_values  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input type: {path.suffix}")


def evaluate(player_draws: pd.DataFrame, players: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    draws = player_draws.merge(players[["game_id", "player_id", "served_prediction"]],
                               on=["game_id", "player_id"], how="left", validate="many_to_one")
    if draws["served_prediction"].isna().any():
        raise ValueError("draws reference players absent from the players table")
    summary = pd.DataFrame([
        {"game_id": game_id, "player_id": player_id, **summarize_values(group["fantasy_points"].to_numpy())}
        for (game_id, player_id), group in draws.groupby(["game_id", "player_id"], sort=True)])
    return {
        "marginal": marginal_calibration(summary, actuals),
        "joint_player": evaluate_joint_player_draws(draws, actuals, scale_column="served_prediction"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player-draws", required=True, type=Path)
    parser.add_argument("--players", required=True, type=Path)
    parser.add_argument("--player-actuals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "evaluation_type": "simulation_holdout",
        "input_provenance": {name: str(getattr(args, name)) for name in ("player_draws", "players", "player_actuals")},
        **evaluate(_read_table(args.player_draws), _read_table(args.players), _read_table(args.player_actuals)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
