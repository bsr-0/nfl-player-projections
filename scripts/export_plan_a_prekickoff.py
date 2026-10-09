#!/usr/bin/env python3
"""Forecast a frozen pre-kickoff list with a frozen Plan A artifact.

Plan A predicts each player's share of a team total and renormalizes the
shares within each team-week, so its forecast depends on who else is in the
team-week. It was validated on the full weekly roster (every status: active,
inactive, reserve, practice squad). Before kickoff that roster is not known,
so the renormalization population for (season, week) is:

* each team's most recent roster earlier in the same season
  (`canonical_player_weeks`, all statuses), re-keyed to the target week, plus
* every player on the frozen list (scripts/build_prekickoff_population.py),
  whose team and position take precedence.

Week 1 has no earlier roster in the season, so it uses the list alone.
Forecasts are written for listed players only. Rows are built from
pre-kickoff data by scripts/build_prekickoff_share_rows.py (leakage-audited).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH, POSITIONS
from scripts.build_prekickoff_share_rows import prekickoff_rows
from src.evaluation.paired_ppr_comparison import file_sha256
from src.models.team_allocation.joint_serving import JointPlanAArtifact


def renormalization_population(con: sqlite3.Connection, listed: pd.DataFrame,
                               season: int, week: int) -> tuple[pd.DataFrame, dict]:
    prior = pd.read_sql(
        "SELECT player_id, team, position, week FROM canonical_player_weeks "
        f"WHERE season = ? AND week < ? AND position IN ({','.join('?' * len(POSITIONS))})",
        con, params=[season, week, *POSITIONS])
    if not prior.empty:
        last = prior.groupby("team").week.transform("max")
        prior = prior[prior.week == last].drop(columns="week")
    roster = prior[~prior.player_id.isin(listed.player_id)]
    pop = pd.concat([listed[["player_id", "team", "position"]], roster], ignore_index=True)
    if pop.player_id.duplicated().any():
        raise ValueError("player appears twice in the renormalization population")
    return pop, {"listed": int(len(listed)), "added_from_prior_roster": int(len(roster))}


def export(artifact_path: Path, population_dir: Path, season: int, weeks: list[int], output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    art = JointPlanAArtifact.load(artifact_path)
    parts, weeks_meta = [], {}
    with sqlite3.connect(str(DB_PATH)) as con:
        for week in weeks:
            path = population_dir / f"population_{season}_w{week:02d}.csv"
            meta = json.loads(path.with_suffix(".json").read_text())
            if file_sha256(path) != meta["sha256"]:
                raise ValueError(f"frozen list changed: {path}")
            listed = pd.read_csv(path, dtype={"player_id": str})
            pop, info = renormalization_population(con, listed, season, week)
            rows = prekickoff_rows(season, week, pop, con)
            pred = art.predict(rows, include_actual=False)
            pred = pred[pred.player_id.isin(listed.player_id)]
            if len(pred) != len(listed) or not np.isfinite(pred.predicted_ppr).all():
                raise ValueError(f"week {week}: expected {len(listed)} finite forecasts, got {len(pred)}")
            parts.append(pred)
            weeks_meta[week] = info | {"list_sha256": meta["sha256"]}
            print(json.dumps({"week": week, **info}), flush=True)
    out = pd.concat(parts, ignore_index=True)
    output_dir.mkdir(parents=True)
    out_path = output_dir / "predictions.csv"
    out.to_csv(out_path, index=False)
    manifest = {
        "label": "plan_a_prekickoff", "season": season, "weeks": weeks,
        "artifact": str(artifact_path), "artifact_sha256": file_sha256(artifact_path),
        "model_version": art.payload["model_version"],
        "renormalization_population": "most recent same-season roster per team (all statuses) + frozen list",
        "weeks_detail": weeks_meta, "rows": int(len(out)),
        "predictions_sha256": file_sha256(out_path), "script_sha256": file_sha256(Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--population-dir", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    m = export(a.artifact, a.population_dir, a.season, a.weeks, a.output_dir)
    print(json.dumps({k: v for k, v in m.items() if k != "weeks_detail"}, indent=2))


if __name__ == "__main__":
    main()
