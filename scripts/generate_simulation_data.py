#!/usr/bin/env python3
"""Generate game-script simulation artifacts from existing serving JSON files.

This is intentionally downstream of generate_weekly_data.py and
generate_game_predictions_data.py. It does not access SQLite or retrain
models.

Usage:
    python scripts/generate_simulation_data.py --season 2026
    python scripts/generate_simulation_data.py --season 2026 --week 2 --draws 2000
    python scripts/generate_simulation_data.py --season 2026 --write-parquet
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.game_simulation import simulate_game_scripts, simulate_players
from src.models.simulation_adapter import simulation_inputs_from_predictions
from src.models.simulation_io import build_and_write_site_simulation_payload
from src.models.simulation_readiness import require_production_ready

DOCS_DATA = PROJECT_ROOT / "docs" / "data"
PARQUET_DATA = PROJECT_ROOT / "data" / "simulations"

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.DataFrame(json.loads(path.read_text()))

def _weeks(season: int, requested: int | None) -> list[int]:
    if requested is not None:
        return [requested]
    values = []
    for path in DOCS_DATA.glob(f"game_predictions_{season}_wk*.json"):
        try:
            values.append(int(path.stem.rsplit("wk", 1)[1]))
        except ValueError:
            continue
    return sorted(set(values))

def generate_week(season: int, week: int, draws: int, seed: int,
                  write_parquet: bool, production: bool,
                  role_correlation_artifact: str | None,
                  calibration_artifact: str | None) -> Path | None:
    game_path = DOCS_DATA / f"game_predictions_{season}_wk{week}.json"
    player_path = DOCS_DATA / f"weekly_{season}_wk{week}.json"
    if not game_path.exists() or not player_path.exists():
        print(f"  wk{week}: missing game/player input, skipped")
        return None
    games = _read(game_path)
    players = _read(player_path)
    games["season"], games["week"] = season, week
    players["season"], players["week"] = season, week
    if production:
        require_production_ready(
            players, role_correlation_artifact=role_correlation_artifact,
            calibration_artifact=calibration_artifact)
        raise RuntimeError(
            "production simulation is disabled: artifact loading and calibration "
            "application must be implemented and evaluated before use")
    inputs = simulation_inputs_from_predictions(games, players)
    if not inputs:
        print(f"  wk{week}: no matched player/game rows, skipped")
        return None

    game_draws, player_draws = [], []
    for game_id, (game, game_players) in inputs.items():
        scripts = simulate_game_scripts(game, n_draws=draws, seed=seed)
        game_draws.extend(asdict(row) | {"game_id": game_id} for row in scripts)
        player_draws.extend(simulate_players(
            game, game_players, n_draws=draws, seed=seed))
    output = DOCS_DATA / f"simulation_{season}_wk{week}.json"
    parquet_dir = PARQUET_DATA / str(season) if write_parquet else None
    build_and_write_site_simulation_payload(
        game_draws, player_draws, seed=seed, site_json_path=output,
        model_config={
            "outcome_model": "logistic", "margin_total_model": "ridge",
            "draws": draws, "season": season, "week": week,
            "simulation_status": "exploratory_volume_only",
            "correlation_mode": "independent_residuals",
            "usage_mode": "team_volume_only",
        },
        parquet_dir=parquet_dir, parquet_stem=output.stem)
    print(f"  wk{week}: {len(inputs)} games, {len(player_draws)} player draws -> {output.name}")
    return output

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--production", action="store_true",
                        help="Fail closed after checking production prerequisites; production artifact application is not enabled yet.")
    parser.add_argument("--role-correlation-artifact", default=None)
    parser.add_argument("--calibration-artifact", default=None)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    weeks = _weeks(args.season, args.week)
    if not weeks:
        print(f"no game prediction files found for {args.season}")
        return 1
    written = [generate_week(
        args.season, week, args.draws, args.seed, args.write_parquet,
        args.production, args.role_correlation_artifact, args.calibration_artifact)
               for week in weeks]
    return 0 if any(written) else 1

if __name__ == "__main__":
    raise SystemExit(main())
