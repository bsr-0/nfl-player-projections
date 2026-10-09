#!/usr/bin/env python3
"""Same-week leakage audit of the live serving path (NFLPredictor.predict).

Two perturbation tests; neither needs a reference table, so neither can share
a blind spot with the code under test.

  poison      For each as-of week (S, W): copy the database, overwrite every
              post-game outcome in rows at or after (S, W) with a different
              number (x * 3.7 + 11) -- in the copy, and in the schedules the
              path downloads (scores, measured weather) -- and run
              predict(as_of=(S, W)) on the copy.
              Predictions and every column of the model-input frame must equal
              the run on the real database. A difference means the path read
              data it could not have had before kickoff.
              A positive control poisons from (S, W-1) instead -- the last
              legitimately visible game -- and must change the predictions;
              otherwise the poison does not reach the model and the pass above
              means nothing.

  truncation  Prepare features on the played history only, and again with the
              target week and everything after it present. Every feature of a
              played row must be identical. This is the lookahead test for
              the frame-based features (rolling windows, team-relative
              shares, season-long aggregates).

The copy runs in a scratch tree of the code (source reads its database by
path), with the real data directory symlinked except for the database.

    python scripts/audit_serving_leakage.py poison --weeks 2025:6 2025:14
    python scripts/audit_serving_leakage.py truncation --weeks 2025:6
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Post-game outcome tables: every numeric column except identifiers is an
# outcome. Rows are poisoned from (season, week) onward, and season-level
# summary rows (week 0) for that season too.
WEEKLY_TABLES = [
    "player_weekly_stats", "team_stats", "team_defense_stats", "team_offense_stats",
    "team_personnel_stats", "team_scheme_tendencies", "snap_counts", "ngs_passing",
    "ngs_receiving", "ngs_rushing", "weekly_pfr", "pbp_pass_participation",
    "utilization_scores", "team_week_player_shares",
]
SEASON_TABLES = ["seasonal_pfr", "qbr"]  # no week: poisoned for season >= S
EXPLICIT = {  # table -> outcome columns (the rest of the table is known pre-game)
    "schedule": ["home_score", "away_score"],
    "canonical_player_weeks": [
        "offense_snaps", "offense_pct", "fantasy_points", "has_stats_row", "has_snap_row",
        "missing_stats_row", "played_without_stats_row", "zero_snaps_without_stats_row",
        "evidence_stats", "evidence_snaps"],
}
IDENTIFIERS = {"id", "season", "week", "player_id", "game_id", "gsis_id"}


def poison_database(path: Path, season: int, week: int) -> dict:
    """Overwrite outcomes at or after (season, week) in the database at `path`."""
    touched = {}
    with sqlite3.connect(str(path)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def numeric(table):
            return [r[1] for r in con.execute(f"PRAGMA table_info({table})")
                    if r[2].upper() in ("INTEGER", "REAL") and r[1].lower() not in IDENTIFIERS]

        def update(table, cols, where):
            if not cols:
                return
            sets = ", ".join(f'"{c}" = "{c}" * 3.7 + 11' for c in cols)
            cur = con.execute(f'UPDATE "{table}" SET {sets} WHERE {where}')
            touched[table] = {"columns": len(cols), "rows": cur.rowcount}

        after = f"(season > {season} OR (season = {season} AND (week >= {week} OR week = 0)))"
        for t in WEEKLY_TABLES:
            if t not in tables:
                raise SystemExit(f"poison list names a table that is not there: {t}")
            update(t, numeric(t), after)
        for t in SEASON_TABLES:
            if t not in tables:
                raise SystemExit(f"poison list names a table that is not there: {t}")
            update(t, numeric(t), f"season >= {season}")
        for t, cols in EXPLICIT.items():
            have = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
            if set(cols) - have:
                raise SystemExit(f"{t}: no columns {sorted(set(cols) - have)}")
            update(t, cols, after)
        con.commit()
    empty = sorted(t for t, v in touched.items() if v["rows"] == 0)
    required = {"player_weekly_stats", "team_stats", "schedule"}
    if required & set(empty):
        raise SystemExit(f"poison touched no rows in {sorted(required & set(empty))}: wrong season/week")
    return touched | {"_tables_with_no_rows_to_poison": empty}


SCHEDULE_OUTCOMES = ["home_score", "away_score", "result", "total", "overtime", "temp", "wind"]


def poison_schedule_fetch(season: int, week: int) -> None:
    """Poison the nfl_data_py schedules the serving path downloads.

    Weather, Vegas lines and head coaches are read from `nfl.import_schedules`,
    not from the database, so database poisoning never reaches them. Scores and
    measured weather of games at or after (season, week) are overwritten here;
    lines, rest days and venue are known before kickoff and stay.
    """
    import nfl_data_py as nfl
    original = nfl.import_schedules

    def poisoned(*args, **kwargs):
        df = original(*args, **kwargs).copy()
        late = (df["season"] > season) | ((df["season"] == season) & (df["week"] >= week))
        for c in SCHEDULE_OUTCOMES:
            if c in df.columns:
                df[c] = pd_to_numeric(df[c])
                df.loc[late, c] = df.loc[late, c] * 3.7 + 11
        if "weather" in df.columns:
            df["weather"] = df["weather"].astype(object)
            df.loc[late & df["weather"].notna(), "weather"] = "Temp: 5\u00b0 F, Humidity: 90%, Wind: NW 30 mph"
        return df

    nfl.import_schedules = poisoned


def pd_to_numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")


def build_scratch_tree(scratch: Path) -> None:
    """Copy of the code with the real data symlinked in, minus the database."""
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for pkg in ("src", "config", "scripts"):
        shutil.copytree(ROOT / pkg, scratch / pkg, ignore=ignore)
    (scratch / "data").mkdir()
    skip = {"nfl_data.db", "nfl_data.db-wal", "nfl_data.db-shm"}
    for item in (ROOT / "data").iterdir():
        if item.name not in skip:
            (scratch / "data" / item.name).symlink_to(item)
    for item in ROOT.iterdir():
        if item.name not in {"src", "config", "scripts", "data", ".git", ".venv", "__pycache__"} \
                and not (scratch / item.name).exists():
            (scratch / item.name).symlink_to(item)


def copy_database(dst: Path) -> None:
    src = sqlite3.connect(f"file:{ROOT / 'data' / 'nfl_data.db'}?mode=ro", uri=True)
    out = sqlite3.connect(str(dst))
    src.backup(out)
    out.close()
    src.close()


def _model_features(predictor) -> set:
    feats = set()
    for mw in predictor.predictor.position_models.values():
        feats |= set(getattr(mw.models.get(1), "feature_names", []))
    return feats


def capture(out: Path, season: int, week: int, expect_root: Path | None) -> None:
    """Run predict(as_of) and keep the model-input frame and the predictions."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path.cwd()))
    import config.settings as settings
    if expect_root is not None and not str(Path(settings.__file__).resolve()).startswith(str(expect_root.resolve())):
        raise SystemExit(f"imported {settings.__file__}, expected a tree under {expect_root}")
    if os.environ.get("AUDIT_POISON_SCHEDULES"):
        poison_schedule_fetch(*[int(x) for x in os.environ["AUDIT_POISON_SCHEDULES"].split(":")])
    from src.predict import NFLPredictor

    p = NFLPredictor()
    if not p.initialize():
        raise SystemExit("predictor failed to initialize")
    seen = {}
    original = p.predictor.predict

    def spy(frame, *a, **k):
        seen["latest"] = frame.copy()
        return original(frame, *a, **k)

    p.predictor.predict = spy
    results = p.predict(n_weeks=1, top_n=100_000, as_of=(season, week))
    if results.empty or "latest" not in seen:
        raise SystemExit("predict returned nothing")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"latest": seen["latest"], "results": results,
                     "model_features": sorted(_model_features(p)),
                     "db": str(settings.DB_PATH)}, fh)


def _same(a, b):
    import numpy as np
    import pandas as pd
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.to_numpy(float), b.to_numpy(float)
        return np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)
    return (a.astype(str).to_numpy() == b.astype(str).to_numpy())


def compare(base_path: Path, other_path: Path) -> dict:
    base, other = (pickle.load(open(p, "rb")) for p in (base_path, other_path))
    a = base["latest"].set_index("player_id").sort_index()
    b = other["latest"].set_index("player_id").sort_index()
    if not a.index.equals(b.index):
        return {"same_players": False, "n_base": len(a), "n_other": len(b)}
    model_feats = set(base["model_features"])
    differing = {}
    for c in sorted(set(a.columns) & set(b.columns)):
        ok = _same(a[c], b[c])
        if not ok.all():
            differing[c] = {"rows": int((~ok).sum()), "model_feature": c in model_feats}
    ra = base["results"].set_index("player_id").sort_index()
    rb = other["results"].set_index("player_id").sort_index()
    if not ra.index.equals(rb.index):
        return {"same_players": True, "same_predictions_index": False}
    pred = {}
    for c in ("predicted_points", "predicted_ppg", "expected_points"):
        if c in ra.columns:
            d = (ra[c] - rb[c]).abs()
            pred[c] = {"max_abs_diff": float(d.max()), "rows_differing": int((d > 1e-9).sum())}
    return {"same_players": True, "players": len(a), "columns_compared": len(set(a.columns) & set(b.columns)),
            "differing_columns": differing,
            "model_features_differing": sorted(c for c, v in differing.items() if v["model_feature"]),
            "predictions": pred}


def run_capture(tree: Path, out: Path, season: int, week: int, expect: Path | None,
                poison_from: tuple[int, int] | None = None) -> None:
    env = dict(os.environ, PYTHONPATH=str(tree))
    env.pop("AUDIT_POISON_SCHEDULES", None)
    if poison_from:
        env["AUDIT_POISON_SCHEDULES"] = f"{poison_from[0]}:{poison_from[1]}"
    cmd = [sys.executable, str(Path(__file__).resolve()), "_capture", "--out", str(out),
           "--season", str(season), "--week", str(week)]
    if expect:
        cmd += ["--expect-root", str(expect)]
    subprocess.run(cmd, cwd=str(tree), env=env, check=True)


def poison_audit(weeks: list[tuple[int, int]], work: Path, output: Path) -> dict:
    scratch = work / "tree"
    build_scratch_tree(scratch)
    report = {}
    failed = False
    for season, week in weeks:
        key = f"{season}-w{week:02d}"
        base = work / f"{key}_base.pkl"
        run_capture(ROOT, base, season, week, ROOT)
        entry = {}
        for label, (ps, pw) in {"after_target": (season, week), "control_from_previous_week": (season, week - 1)}.items():
            db = scratch / "data" / "nfl_data.db"
            db.unlink(missing_ok=True)
            copy_database(db)
            entry[f"{label}_poisoned"] = poison_database(db, ps, pw)
            other = work / f"{key}_{label}.pkl"
            run_capture(scratch, other, season, week, scratch, poison_from=(ps, pw))
            entry[label] = compare(base, other)
            db.unlink(missing_ok=True)
        main, control = entry["after_target"], entry["control_from_previous_week"]
        leaks = main.get("model_features_differing", []) or [
            c for c, p in main.get("predictions", {}).items() if p["rows_differing"]]
        sensitive = any(p["rows_differing"] for p in control.get("predictions", {}).values())
        entry["verdict"] = {
            "no_leak_detected": not leaks and main.get("same_players", False)
                                and not main.get("differing_columns"),
            "control_changed_predictions": sensitive,
        }
        failed |= not entry["verdict"]["no_leak_detected"] or not sensitive
        report[key] = entry
        print(json.dumps({key: {"verdict": entry["verdict"],
                                "differing_columns": main.get("differing_columns"),
                                "predictions": main.get("predictions"),
                                "control_predictions": control.get("predictions")}}, indent=1), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    report["_failed"] = failed
    return report


def truncation_audit(weeks: list[tuple[int, int]], output: Path) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from src.predict import NFLPredictor

    p = NFLPredictor()
    if not p.initialize():
        raise SystemExit("predictor failed to initialize")
    model_feats = _model_features(p)
    base = p._load_player_data(None, min_games=1)
    s, w = pd.to_numeric(base.season), pd.to_numeric(base.week)
    report, failed = {}, False
    for season, week in weeks:
        key = f"{season}-w{week:02d}"
        played = (s < season) | ((s == season) & (w < week))
        trunc = p._prepare_features(base[played].copy())
        full = p._prepare_features(base.copy())
        k = ["player_id", "season", "week"]
        left = trunc.drop_duplicates(k).set_index(k).sort_index()
        right = full.drop_duplicates(k).set_index(k).sort_index().reindex(left.index)
        if right.isna().all(axis=1).any():
            raise SystemExit(f"{key}: played rows missing from the full-frame preparation")
        differing = {}
        for c in sorted(set(left.columns) & set(right.columns)):
            ok = _same(left[c], right[c])
            if not ok.all():
                differing[c] = {"rows": int((~ok).sum()), "model_feature": c in model_feats}
        bad = sorted(c for c, v in differing.items() if v["model_feature"])
        failed |= bool(bad)
        report[key] = {"rows_compared": len(left), "columns_compared": len(set(left.columns) & set(right.columns)),
                       "differing_columns": differing, "model_features_differing": bad}
        print(json.dumps({key: {"rows": len(left), "model_features_differing": bad,
                                "other_columns_differing": sorted(set(differing) - set(bad))}}, indent=1), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    report["_failed"] = failed
    return report


def parse_weeks(items):
    return [tuple(int(x) for x in w.split(":")) for w in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("poison", "truncation"):
        s = sub.add_parser(name)
        s.add_argument("--weeks", nargs="+", required=True, help="season:week, e.g. 2025:6")
        s.add_argument("--output", type=Path, default=ROOT / f"data/experiments/serving_leakage_audit/{name}.json")
        s.add_argument("--work", type=Path, default=Path("/tmp/serving_leakage_audit"))
    c = sub.add_parser("_capture")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--season", type=int, required=True)
    c.add_argument("--week", type=int, required=True)
    c.add_argument("--expect-root", type=Path)
    a = ap.parse_args()
    if a.cmd == "_capture":
        capture(a.out, a.season, a.week, a.expect_root)
        return
    weeks = parse_weeks(a.weeks)
    report = poison_audit(weeks, a.work, a.output) if a.cmd == "poison" else truncation_audit(weeks, a.output)
    print("AUDIT FAILED" if report["_failed"] else "audit passed")
    sys.exit(1 if report["_failed"] else 0)


if __name__ == "__main__":
    main()
