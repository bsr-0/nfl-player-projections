#!/usr/bin/env python3
"""The forward test (G10) of docs/PRODUCTION_SELECTION_RULE.md, one week at a time.

    freeze   write data/experiments/selection_forward_2026/lineage.json once:
             the rule commits, the served model's files and kappa, the Plan A
             and rolling-3 artifacts, every script this pipeline runs, and the
             start week (the first week whose first game date follows the
             freeze). Refuses to overwrite.
    run      the real upcoming week, before its first game date (US/Eastern
             midnight -- conservative, as the schedule has dates, not times):
               1. every lineage hash must still match;
               2. load the season's weekly rosters and rebuild
                  canonical_player_weeks, refusing if any completed week
                  before the previous one changes key, team or position;
               3. require the previous week's stats and pass the leakage
                  audit on it;
               4. one production predict() call gives the frozen list, the
                  incumbent (predicted_points, the live blend) and the
                  unblended served model (predicted_points_model);
               5. Plan A and rolling-3 forecast exactly the list;
               6. write the list and forecasts with sha256s; refuse to
                  overwrite a week.
    dry-run  the same steps into dry_run/, deadline recorded, not enforced. Runs
             before the freeze (the rule starts the window only after a
             passing dry run): it takes the artifacts as arguments and checks
             the lineage only if one exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH, PACE_BLEND_KAPPA
from scripts.build_canonical_player_weeks import build_panel
from scripts.build_prekickoff_population import population
from scripts.build_prekickoff_share_rows import audit
from scripts.export_plan_a_prekickoff import forecast_week
from src.evaluation.paired_ppr_comparison import file_sha256
from src.models.team_allocation.joint_serving import JointPlanAArtifact
from src.predict import NFLPredictor, get_prediction_target_week

OUT = ROOT / "data/experiments/selection_forward_2026"
LINEAGE = OUT / "lineage.json"
RULE = "docs/PRODUCTION_SELECTION_RULE.md"
SERVED_FILES = [  # every file a live predict() reads that defines the served model
    "data/models/feature_scaler_bounded.joblib", "data/models/injury_risk_model.joblib",
    "data/models/multiweek_qb.joblib", "data/models/multiweek_rb.joblib",
    "data/models/multiweek_te.joblib", "data/models/multiweek_wr.joblib",
    "data/models/qb_target_choice.json", "data/models/utilization_percentile_bounds.json",
    "data/models/utilization_weights.json", "data/step8_pace_by_season.csv",
    "data/draft_picks.parquet",
]
SCRIPTS = [
    "scripts/run_forward_week.py", "scripts/build_prekickoff_population.py",
    "scripts/build_prekickoff_share_rows.py", "scripts/export_plan_a_prekickoff.py",
    "scripts/build_team_week_player_shares.py", "scripts/build_canonical_player_weeks.py",
    "scripts/backfill_weekly_rosters.py", "src/predict.py",
    "src/models/team_allocation/joint_serving.py", "src/evaluation/team_reconstruction_candidates.py",
    "src/evaluation/selection_gates.py",
]
EASTERN = ZoneInfo("America/New_York")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def first_game_deadline(season: int, week: int) -> datetime:
    with sqlite3.connect(str(DB_PATH)) as con:
        first = con.execute("SELECT MIN(game_time) FROM schedule WHERE season = ? AND week = ?",
                            (season, week)).fetchone()[0]
    if not first:
        raise ValueError(f"no schedule for {season} week {week}")
    day = datetime.fromisoformat(str(first)[:10])
    return day.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def hashes(paths: list[str]) -> dict[str, str]:
    return {p: file_sha256(ROOT / p) for p in paths}


def freeze(plan_a: Path, rolling3: Path) -> dict:
    if LINEAGE.exists():
        raise SystemExit(f"lineage already frozen: {LINEAGE}")
    if _git("status", "--porcelain", "--", RULE, *SCRIPTS):
        raise SystemExit("rule or pipeline scripts have uncommitted changes; commit them first")
    now = datetime.now(timezone.utc)
    with sqlite3.connect(str(DB_PATH)) as con:
        weeks = con.execute("SELECT week, MIN(game_time) FROM schedule WHERE season = 2026 "
                            "GROUP BY week ORDER BY week").fetchall()
    start = next(w for w, first in weeks
                 if datetime.fromisoformat(str(first)[:10]).replace(tzinfo=EASTERN) > now)
    lineage = {
        "frozen_at": now.isoformat(),
        "rule": RULE, "rule_sha256": file_sha256(ROOT / RULE),
        "rule_commits": {"original": "147bc1b80b6a52aafdcc3d33fa9ccc9da60db8ad",
                         "amendment_1": "66df69530a17e2471a22e7a3e52ff77f820a22e3"},
        "head_commit": _git("rev-parse", "HEAD"),
        "season": 2026, "start_week": int(start), "window_weeks": 8,
        "served": {"pace_blend_kappa": PACE_BLEND_KAPPA, "files_sha256": hashes(SERVED_FILES)},
        "plan_a": {"artifact": str(plan_a.relative_to(ROOT)), "sha256": file_sha256(plan_a),
                   "model_version": JointPlanAArtifact.load(plan_a).payload["model_version"]},
        "rolling3": {"artifact": str(rolling3.relative_to(ROOT)), "sha256": file_sha256(rolling3)},
        "scripts_sha256": hashes(SCRIPTS),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LINEAGE.write_text(json.dumps(lineage, indent=2) + "\n")
    return lineage


def check_lineage() -> dict:
    lin = json.loads(LINEAGE.read_text())
    problems = []
    for group, current in (("served files", hashes(SERVED_FILES)), ("scripts", hashes(SCRIPTS))):
        frozen = lin["served"]["files_sha256"] if group == "served files" else lin["scripts_sha256"]
        problems += [f"{group}: {p}" for p in frozen if current.get(p) != frozen[p]]
    for key in ("plan_a", "rolling3"):
        if file_sha256(ROOT / lin[key]["artifact"]) != lin[key]["sha256"]:
            problems.append(f"{key} artifact")
    if file_sha256(ROOT / lin["rule"]) != lin["rule_sha256"]:
        problems.append("rule text")
    if PACE_BLEND_KAPPA != lin["served"]["pace_blend_kappa"]:
        problems.append("PACE_BLEND_KAPPA")
    if problems:
        raise SystemExit("lineage broken -- the window is invalid: " + "; ".join(problems))
    return lin


def refresh_data(season: int, week: int) -> dict:
    subprocess.run([sys.executable, "scripts/backfill_weekly_rosters.py", "-s", str(season), str(season)],
                   cwd=ROOT, check=True, capture_output=True)
    with sqlite3.connect(str(DB_PATH)) as con:
        old = pd.read_sql("SELECT player_id, season, week, team, position FROM canonical_player_weeks", con)
        new = build_panel(con, 2013, season)[["player_id", "season", "week", "team", "position"]]
    settled = lambda f: f[(f.season < season) | (f.week < week - 1)]
    o, n = settled(old), settled(new)
    joint = o.merge(n, on=["player_id", "season", "week"], how="outer", suffixes=("_o", "_n"), indicator=True)
    changed = joint[(joint._merge != "both") | (joint.team_o != joint.team_n) | (joint.position_o != joint.position_n)]
    if len(changed):
        raise SystemExit(f"canonical rebuild would change {len(changed)} settled rows; not writing")
    audit_csv = OUT / "canonical_audit" / f"{season}_w{week:02d}.csv"
    subprocess.run([sys.executable, "scripts/build_canonical_player_weeks.py", "--write",
                    "--seasons", "2013", str(season), "--audit-csv", str(audit_csv)],
                   cwd=ROOT, check=True, capture_output=True)
    with sqlite3.connect(str(DB_PATH)) as con:
        stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT team) FROM player_weekly_stats "
                            "WHERE season = ? AND week = ?", (season, week - 1)).fetchone()
        teams = con.execute("SELECT COUNT(*) * 2 FROM schedule WHERE season = ? AND week = ?",
                            (season, week - 1)).fetchone()[0]
    if week > 1 and (stats[1] < teams or stats[0] < 250):
        raise SystemExit(f"week {week - 1} stats incomplete: {stats[0]} rows, {stats[1]} of {teams} teams")
    return {"settled_rows_checked": int(len(o)), "prev_week_stat_rows": int(stats[0]),
            "prev_week_teams": int(stats[1])}


def run_week(dry_run: bool, artifacts: dict[str, Path] | None = None) -> dict:
    if dry_run and not LINEAGE.exists():
        lin = {key: {"artifact": str(path.relative_to(ROOT))} for key, path in artifacts.items()}
    else:
        lin = check_lineage()
    season, week = (int(x) for x in get_prediction_target_week())
    deadline = first_game_deadline(season, week)
    started = datetime.now(timezone.utc)
    if not dry_run:
        if season != lin["season"] or not lin["start_week"] <= week < lin["start_week"] + lin["window_weeks"]:
            raise SystemExit(f"{season} week {week} is outside the frozen window")
        if started >= deadline:
            raise SystemExit(f"too late: week {week}'s first game date began {deadline.isoformat()}")
    out_dir = (OUT / "dry_run" if dry_run else OUT) / f"week_{week:02d}"
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    data = refresh_data(season, week)
    if week > 1:
        audit([(season, week - 1)])
    predictor = NFLPredictor()
    predictor.initialize()
    listed, list_meta, prod = population(predictor, season, week, replay=False)
    prod = prod.drop_duplicates("player_id").set_index("player_id")
    fc = listed.copy()
    fc["incumbent_live_blend"] = fc.player_id.map(prod.predicted_points).to_numpy(float)
    fc["served_unblended"] = fc.player_id.map(prod.predicted_points_model).to_numpy(float)
    with sqlite3.connect(str(DB_PATH)) as con:
        for col, key in (("plan_a", "plan_a"), ("rolling3", "rolling3")):
            art = JointPlanAArtifact.load(ROOT / lin[key]["artifact"])
            pred, info = forecast_week(art, con, listed, season, week)
            fc[col] = fc.player_id.map(pred.set_index("player_id").predicted_ppr).to_numpy(float)
            list_meta[f"{col}_renormalization"] = info
    models = ["incumbent_live_blend", "served_unblended", "plan_a", "rolling3"]
    if not np.isfinite(fc[models].to_numpy(float)).all():
        raise SystemExit("a model failed to forecast every listed player")
    out_dir.mkdir(parents=True)
    listed.to_csv(out_dir / "list.csv", index=False)
    fc.to_csv(out_dir / "forecasts.csv", index=False)
    written = datetime.now(timezone.utc)
    manifest = {
        "season": season, "week": week, "dry_run": dry_run,
        "deadline_utc": deadline.isoformat(), "started_utc": started.isoformat(),
        "written_utc": written.isoformat(), "before_deadline": written < deadline,
        "lineage_sha256": file_sha256(LINEAGE) if LINEAGE.exists() else None,
        "artifacts_sha256": {k: file_sha256(ROOT / v["artifact"]) for k, v in lin.items()
                             if k in ("plan_a", "rolling3")},
        "data_refresh": data, "list": list_meta,
        "list_sha256": file_sha256(out_dir / "list.csv"),
        "forecasts_sha256": file_sha256(out_dir / "forecasts.csv"), "models": models,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--plan-a", type=Path, required=True)
    f.add_argument("--rolling3", type=Path, required=True)
    sub.add_parser("run")
    d = sub.add_parser("dry-run")
    d.add_argument("--plan-a", type=Path)
    d.add_argument("--rolling3", type=Path)
    a = ap.parse_args()
    if a.cmd == "freeze":
        result = freeze(a.plan_a.resolve(), a.rolling3.resolve())
    elif a.cmd == "dry-run":
        if not LINEAGE.exists() and not (a.plan_a and a.rolling3):
            ap.error("before the freeze, dry-run needs --plan-a and --rolling3")
        arts = {"plan_a": a.plan_a.resolve(), "rolling3": a.rolling3.resolve()} if a.plan_a else None
        result = run_week(dry_run=True, artifacts=arts)
    else:
        result = run_week(dry_run=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
