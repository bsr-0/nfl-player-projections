#!/usr/bin/env python3
"""Shadow forecasts for docs/PREREG_WR_VOLUME_2026.md (WR season-to-date volume, weeks 7-13).

    build    refit C0' (the live WR model's 70 features) and C1' (+ three season-to-date volume columns)
             on seasons 2006-2025 with the live WR model's tuned parameters held fixed, and save both
             as frozen artifacts (never under data/models)
    freeze   write lineage.json once: artifact, script, live WR model and rule hashes, window
    run      the real upcoming week, before its first game date (US/Eastern midnight, the forward
             runner's rule): one production predict() call with a read-only hook that copies the exact
             frame the live models receive; both arms forecast the WR rows; written and hashed.
             Refuses to overwrite a week. `--dry-run` writes under dry_run/ and enforces nothing.
    check    verify the weeks written so far against their manifests (reads no outcome)

The hook only copies its argument. `run` also proves the copy is the live input: the live WR model
applied to it must reproduce production's own `predicted_points_model` for every WR to 1e-9.
The calendar's week is the forward runner's: set NFL_TARGET_WEEK=2026:N as for that runner.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

OUT = ROOT / "data/experiments/wr_volume_2026"
ART = OUT / "artifacts"
LINEAGE = OUT / "lineage.json"
LIVE_WR = ROOT / "data/models/multiweek_wr.joblib"
PREFLIGHT = ROOT / "data/experiments/full_ppr_head_to_head_20260930/served_2025_preflight_v5"
RULE = "docs/PREREG_WR_VOLUME_2026.md"
SEASON, WINDOW = 2026, range(7, 14)
KEY = ["player_id", "season", "week"]
NEW = ["targets_s2d", "receptions_s2d", "receiving_yards_s2d"]
SCRIPTS = ["scripts/wr_volume_shadow.py", "scripts/wr_feature_candidates_2025.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_model():
    import joblib
    return next(iter(joblib.load(LIVE_WR)["unique_models"].values()))["model"]


# --- features --------------------------------------------------------------------------------------

def latest_volume(stats: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Per player, the season-to-date means through the player's latest game strictly before (season, week).

    Same definition as training: the expanding mean over the origin game's own season up to and
    including that game, so a player with no game yet this season carries last season's final mean.
    """
    from scripts.wr_feature_candidates_2025 import season_to_date
    s2d = season_to_date(stats)
    before = s2d[(s2d.season < season) | ((s2d.season == season) & (s2d.week < week))]
    return before.sort_values(KEY).groupby("player_id").tail(1)[["player_id", *NEW]]


def arm_forecast(artifact: dict, wr: pd.DataFrame) -> np.ndarray:
    """One arm's WR forecast from the captured live frame plus the volume columns."""
    X = wr.reindex(columns=artifact["features"]).copy()
    for col, med in artifact["medians"].items():
        if col in NEW:
            X[col] = X[col].fillna(med)
    return np.asarray(artifact["model"].predict(X, n_weeks=1), float)


# --- capture ---------------------------------------------------------------------------------------

@contextlib.contextmanager
def capture_model_input():
    """Copy the frame handed to EnsemblePredictor.predict, changing nothing."""
    from src.models.ensemble import EnsemblePredictor
    original = EnsemblePredictor.predict
    box: dict = {}

    def wrapped(self, player_data, n_weeks=1):
        box["frame"] = player_data.copy()
        box["n_weeks"] = n_weeks
        return original(self, player_data, n_weeks)

    EnsemblePredictor.predict = wrapped
    try:
        yield box
    finally:
        EnsemblePredictor.predict = original


# --- build -----------------------------------------------------------------------------------------

def build() -> None:
    import joblib
    from config.settings import POSITIONS
    from scripts import wr_feature_candidates_2025 as h
    from src.models.feature_preparation import _prepare_training_data
    from src.models.position_models import MultiWeekModel, PositionModel
    from src.utils.models_dir import redirect_models_dir
    ART.mkdir(parents=True, exist_ok=True)
    before = h.live_hashes()
    raw_train = pd.read_parquet(PREFLIGHT / "raw_train.parquet")
    raw_2025 = pd.read_parquet(PREFLIGHT / "raw_test.parquet")
    if raw_train.season.max() != 2024 or set(raw_2025.season.unique()) != {2025}:
        raise SystemExit("unexpected preflight seasons")
    train = pd.concat([raw_train, raw_2025], ignore_index=True)
    work = OUT / "models_scratch"
    work.mkdir(exist_ok=True)
    with redirect_models_dir(work):
        ptrain, _, _ = _prepare_training_data(train, raw_2025, list(POSITIONS), tune_hyperparameters=False,
                                              n_trials=1, fit_models=False)
    wr = ptrain[ptrain.position == "WR"].reset_index(drop=True)
    stats = h.season_to_date(h.load_stats())
    wr = wr.merge(stats[KEY + h.C1], on=KEY, how="left")
    live = live_model()
    tuned = copy.deepcopy(live.best_params)
    base = list(live.feature_names)
    missing = [c for c in base if c not in wr.columns]
    if missing:
        raise SystemExit(f"training frame lacks live model inputs: {missing[:5]}")
    original = PositionModel._get_default_params
    PositionModel._get_default_params = lambda self: copy.deepcopy(tuned)
    try:
        with redirect_models_dir(work):
            for name, cols in (("c0", base), ("c1", base + h.C1)):
                X, y, seasons, sw, medians = h.training_matrix(wr, cols)
                model = MultiWeekModel("WR").fit(X, {1: y}, tune_hyperparameters=False, sample_weight=sw, seasons=seasons)
                joblib.dump({"model": model, "features": cols, "medians": medians,
                             "train_rows": int(len(X)), "train_seasons": [int(s) for s in sorted(set(seasons))],
                             "tuned_params_from": "data/models/multiweek_wr.joblib"}, ART / f"{name}.joblib")
                print(name, len(cols), "features,", len(X), "rows", flush=True)
    finally:
        PositionModel._get_default_params = original
    if h.live_hashes() != before:
        raise SystemExit("live model files changed during build")
    # The live scaler is applied to serving inputs; the bounded scaler fit in this build must match it.
    a = joblib.load(work / "feature_scaler_bounded.joblib")
    b = joblib.load(ROOT / "data/models/feature_scaler_bounded.joblib")
    print("bounded scaler artifact keys equal:", sorted(a) == sorted(b) if isinstance(a, dict) else type(a))
    (OUT / "build_info.json").write_text(json.dumps({"live_wr_sha256": sha(LIVE_WR),
                                                      "live_artifacts_unchanged": True}, indent=2) + "\n")


# --- lineage ---------------------------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def freeze() -> None:
    if LINEAGE.exists():
        raise SystemExit(f"lineage already frozen: {LINEAGE}")
    if git("status", "--porcelain", "--", RULE, *SCRIPTS):
        raise SystemExit("rule or scripts have uncommitted changes")
    lineage = {"frozen_at": datetime.now(timezone.utc).isoformat(), "rule": RULE, "rule_sha256": sha(ROOT / RULE),
               "head_commit": git("rev-parse", "HEAD"), "season": SEASON, "window": [min(WINDOW), max(WINDOW)],
               "artifacts_sha256": {p.name: sha(p) for p in sorted(ART.glob("*.joblib"))},
               "live_wr_sha256": sha(LIVE_WR), "scripts_sha256": {p: sha(ROOT / p) for p in SCRIPTS}}
    LINEAGE.write_text(json.dumps(lineage, indent=2) + "\n")
    print(json.dumps(lineage, indent=2))


def check_lineage() -> dict:
    lin = json.loads(LINEAGE.read_text())
    bad = [f"artifact {k}" for k, v in lin["artifacts_sha256"].items() if sha(ART / k) != v]
    bad += [f"script {k}" for k, v in lin["scripts_sha256"].items() if sha(ROOT / k) != v]
    if sha(ROOT / lin["rule"]) != lin["rule_sha256"]:
        bad.append("rule text")
    if sha(LIVE_WR) != lin["live_wr_sha256"]:
        bad.append("live WR model changed (tuned parameters and inputs are defined by it)")
    if bad:
        raise SystemExit("lineage broken -- the shadow window is invalid: " + "; ".join(bad))
    return lin


# --- weekly run ------------------------------------------------------------------------------------

def run_week(dry_run: bool) -> dict:
    import joblib
    from scripts import wr_feature_candidates_2025 as h
    from scripts.run_forward_week import OUT as FORWARD_DIR, first_game_deadline
    from src.predict import NFLPredictor, get_prediction_target_week
    lin = None if dry_run and not LINEAGE.exists() else check_lineage()
    season, week = (int(x) for x in get_prediction_target_week())
    deadline = first_game_deadline(season, week)
    started = datetime.now(timezone.utc)
    if not dry_run:
        if season != lin["season"] or not lin["window"][0] <= week <= lin["window"][1]:
            raise SystemExit(f"{season} week {week} is outside the frozen window")
        if started >= deadline:
            raise SystemExit(f"too late: week {week}'s first game date began {deadline.isoformat()}")
    out_dir = (OUT / "dry_run" if dry_run else OUT) / f"week_{week:02d}"
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    arms = {k: joblib.load(ART / f"{k}.joblib") for k in ("c0", "c1")}
    predictor = NFLPredictor()
    predictor.initialize()
    with capture_model_input() as box:
        prod = predictor.predict(n_weeks=1, top_n=1_000_000)
    frame = box["frame"]
    wr = frame[frame.position == "WR"].drop_duplicates("player_id").reset_index(drop=True)
    # The copy must be what the live WR model received: reproduce production's own unblended output.
    live = predictor.predictor.position_models["WR"]
    ref = np.asarray(live.predict(wr.reindex(columns=live_model().feature_names), n_weeks=1), float)
    served = prod.drop_duplicates("player_id").set_index("player_id").predicted_points_model.reindex(wr.player_id).to_numpy(float)
    gap = float(np.nanmax(np.abs(ref - served)))
    if not gap < 1e-9:
        raise SystemExit(f"captured frame does not reproduce the live WR forecast (max diff {gap:.3g})")
    vol = latest_volume(h.load_stats(), season, week)
    wr = wr.drop(columns=[c for c in NEW if c in wr.columns]).merge(vol, on="player_id", how="left")
    fc = wr[["player_id", "team", "position"]].assign(season=season, week=week)
    fc["live_unblended"] = ref
    fc["c0"], fc["c1"] = (arm_forecast(arms[k], wr) for k in ("c0", "c1"))
    for col in NEW:
        fc[col] = wr[col].to_numpy(float)
    fc = fc.sort_values("player_id").reset_index(drop=True)
    if not np.isfinite(fc[["c0", "c1", "live_unblended"]].to_numpy(float)).all():
        raise SystemExit("an arm failed to forecast every WR")
    listed = FORWARD_DIR / ("dry_run" if dry_run else "") / f"week_{week:02d}" / "list.csv"
    list_check = None
    if listed.exists():
        wr_list = set(pd.read_csv(listed, dtype={"player_id": str}).query("position == 'WR'").player_id)
        list_check = {"runner_list_wr": len(wr_list), "shadow_wr": int(fc.player_id.nunique()),
                      "in_list_not_shadow": len(wr_list - set(fc.player_id))}
        if list_check["in_list_not_shadow"]:
            raise SystemExit(f"the shadow run lacks {list_check['in_list_not_shadow']} WRs on the frozen list")
    out_dir.mkdir(parents=True)
    fc.to_csv(out_dir / "forecasts.csv", index=False)
    written = datetime.now(timezone.utc)
    manifest = {"season": season, "week": week, "dry_run": dry_run, "deadline_utc": deadline.isoformat(),
                "started_utc": started.isoformat(), "written_utc": written.isoformat(),
                "before_deadline": written < deadline, "wr_rows": int(len(fc)),
                "lineage_sha256": sha(LINEAGE) if LINEAGE.exists() else None,
                "runner_list_check": list_check, "capture_reproduces_live_max_diff": gap,
                "forecasts_sha256": sha(out_dir / "forecasts.csv"), "models": ["live_unblended", "c0", "c1"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def check() -> dict:
    valid, excluded = {}, {}
    for week in WINDOW:
        d = OUT / f"week_{week:02d}"
        if not (d / "manifest.json").exists():
            excluded[week] = "not written"
            continue
        m = json.loads((d / "manifest.json").read_text())
        if m.get("dry_run") or m.get("week") != week:
            excluded[week] = "dry run or wrong week"
        elif not m.get("before_deadline"):
            excluded[week] = "written after the deadline"
        elif sha(d / "forecasts.csv") != m["forecasts_sha256"]:
            excluded[week] = "forecasts differ from the manifest"
        else:
            valid[week] = str(d / "forecasts.csv")
    return {"valid_weeks": sorted(valid), "excluded": excluded}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "freeze", "check"):
        sub.add_parser(name)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "freeze":
        freeze()
    elif a.cmd == "check":
        print(json.dumps(check(), indent=2))
    else:
        print(json.dumps(run_week(a.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
