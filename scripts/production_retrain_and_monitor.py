"""
Production: weekly retrain, drift monitoring, versioning and rollback.

- Weekly retrain: run this script (e.g. via cron) to refresh data and retrain.
- Drift: compare latest backtest RMSE to previous; flag if degradation > 20%.
- Versioning: backup current models before overwrite; rollback restores previous.
"""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, MODELS_DIR, RETRAINING_CONFIG

BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
DRIFT_THRESHOLD_PCT = float(RETRAINING_CONFIG.get("degradation_threshold_pct", 20.0))
DRIFT_POSITION_THRESHOLD_PCT = float(RETRAINING_CONFIG.get("drift_position_threshold_pct", 25.0))
MODEL_BACKUP_SUFFIX = ".bak"
DRIFT_STATUS_FILE = MODELS_DIR / "drift_status.json"
RETRAIN_STATUS_FILE = MODELS_DIR / "retrain_status.json"
RETRAIN_SLA_SECONDS = int(RETRAINING_CONFIG.get("retrain_sla_seconds", 24 * 3600))
MAX_DATA_STALENESS_HOURS = int(RETRAINING_CONFIG.get("max_data_staleness_hours", 168))


def _is_retrain_day() -> bool:
    """Return True when today matches configured retrain day or cadence."""
    configured = str(RETRAINING_CONFIG.get("retrain_day", "Tuesday")).strip().lower()
    today = datetime.now().strftime("%A").lower()
    if today == configured:
        return True
    # Cadence-based: check if enough days have passed since last retrain
    if RETRAIN_STATUS_FILE.exists():
        try:
            with open(RETRAIN_STATUS_FILE) as f:
                last = json.load(f)
            last_dt = datetime.fromisoformat(last.get("finished_at", last.get("started_at", "")))
            days_since = (datetime.now() - last_dt).total_seconds() / 86400
            # Use in-season vs off-season cadence
            in_season_cadence = int(RETRAINING_CONFIG.get("in_season_cadence_days", 7))
            off_season_cadence = int(RETRAINING_CONFIG.get("off_season_cadence_days", 30))
            try:
                from src.utils.nfl_calendar import current_season_has_weeks_played
                cadence = in_season_cadence if current_season_has_weeks_played() else off_season_cadence
            except Exception:
                cadence = in_season_cadence
            if days_since >= cadence:
                return True
        except Exception:
            pass
    return False


def _check_data_freshness() -> dict:
    """Check if data is fresh enough for retraining."""
    result = {"fresh": True, "staleness_hours": None, "has_current_season": None}
    try:
        db_path = DATA_DIR / "nfl_data.db"
        if db_path.exists():
            import os
            mtime = os.path.getmtime(db_path)
            staleness_h = (time.time() - mtime) / 3600
            result["staleness_hours"] = round(staleness_h, 1)
            if staleness_h > MAX_DATA_STALENESS_HOURS:
                result["fresh"] = False
                result["reason"] = f"Data is {staleness_h:.0f}h old (max {MAX_DATA_STALENESS_HOURS}h)"
    except Exception:
        pass
    if RETRAINING_CONFIG.get("require_current_season_data", True):
        try:
            from src.utils.nfl_calendar import get_current_nfl_season, current_season_has_weeks_played
            if current_season_has_weeks_played():
                from src.utils.database import DatabaseManager
                db = DatabaseManager()
                current = get_current_nfl_season()
                # Quick check: does DB have any rows for current season?
                import sqlite3
                conn = sqlite3.connect(str(DATA_DIR / "nfl_data.db"))
                count = conn.execute(
                    "SELECT COUNT(*) FROM player_stats WHERE season = ?", (current,)
                ).fetchone()[0]
                conn.close()
                result["has_current_season"] = count > 0
                if count == 0:
                    result["fresh"] = False
                    result["reason"] = f"No data for current season {current}"
        except Exception:
            pass
    return result


def run_weekly_retrain(force: bool = False) -> bool:
    """Run data refresh and full model training. Returns True on success."""
    start = time.time()
    status = {
        "started_at": datetime.now().isoformat(),
        "force": bool(force),
        "retrain_day_config": RETRAINING_CONFIG.get("retrain_day"),
        "in_season_cadence_days": RETRAINING_CONFIG.get("in_season_cadence_days"),
        "off_season_cadence_days": RETRAINING_CONFIG.get("off_season_cadence_days"),
    }
    if not RETRAINING_CONFIG.get("auto_retrain", True) and not force:
        print("Auto retrain is disabled in config; skipping.")
        status.update({"completed": False, "skipped": True, "reason": "auto_retrain_disabled"})
        _write_retrain_status(status, start)
        return False
    if not force and not _is_retrain_day():
        print(f"Today is not configured retrain day ({RETRAINING_CONFIG.get('retrain_day')}); skipping.")
        status.update({"completed": False, "skipped": True, "reason": "not_retrain_day"})
        _write_retrain_status(status, start)
        return False

    # Data freshness gate
    freshness = _check_data_freshness()
    status["data_freshness"] = freshness
    if not freshness["fresh"] and not force:
        reason = freshness.get("reason", "data_not_fresh")
        print(f"Data freshness check failed: {reason}")
        status.update({"completed": False, "skipped": True, "reason": reason})
        _write_retrain_status(status, start)
        return False

    try:
        from src.data.auto_refresh import auto_refresh_data
        auto_refresh_data()
    except Exception as e:
        print(f"Data refresh failed: {e}")
        status.update({"completed": False, "error": f"data_refresh_failed: {e}"})
        _write_retrain_status(status, start)
        return False
    try:
        from src.models.train import train_models
        train_models(tune_hyperparameters=False)
        status.update({"completed": True, "error": None})
        _write_retrain_status(status, start)

        # Post-retrain drift check and auto-rollback
        if RETRAINING_CONFIG.get("drift_check_after_retrain", True):
            drift = check_drift()
            if RETRAINING_CONFIG.get("enable_drift_status_file", True):
                write_drift_status(drift)
            if drift.get("drift_detected") and RETRAINING_CONFIG.get("drift_auto_rollback", True):
                print("WARNING: Post-retrain drift detected; auto-rolling back.")
                rollback_models()
                status["auto_rollback_triggered"] = True
                _write_retrain_status(status, start)

        # SLA breach check
        elapsed = time.time() - start
        if elapsed > RETRAIN_SLA_SECONDS and RETRAINING_CONFIG.get("alert_on_sla_breach", True):
            print(f"WARNING: Retrain SLA breached ({elapsed:.0f}s > {RETRAIN_SLA_SECONDS}s)")
            status["sla_breached"] = True

        return True
    except Exception as e:
        print(f"Training failed: {e}")
        status.update({"completed": False, "error": f"training_failed: {e}"})
        _write_retrain_status(status, start)
        return False


def backup_models() -> None:
    """Copy current model files to .bak so we can rollback."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODELS_DIR.glob("*.joblib"):
        bak = f.with_suffix(f.suffix + MODEL_BACKUP_SUFFIX)
        shutil.copy2(f, bak)
    # Backup metadata/artifact files used by serving and monitoring.
    for f in MODELS_DIR.glob("*.json"):
        bak = f.with_suffix(f.suffix + MODEL_BACKUP_SUFFIX)
        shutil.copy2(f, bak)
    for f in MODELS_DIR.glob("*.txt"):
        bak = f.with_suffix(f.suffix + MODEL_BACKUP_SUFFIX)
        shutil.copy2(f, bak)
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and (d / "model.keras").exists():
            bak_dir = Path(str(d) + MODEL_BACKUP_SUFFIX)
            if bak_dir.exists():
                shutil.rmtree(bak_dir)
            shutil.copytree(d, bak_dir)


def rollback_models() -> bool:
    """Restore models from .bak. Returns True if rollback was performed."""
    restored = False
    for f in MODELS_DIR.glob(f"*{MODEL_BACKUP_SUFFIX}"):
        if f.suffix == MODEL_BACKUP_SUFFIX:
            orig = Path(str(f).replace(MODEL_BACKUP_SUFFIX, ""))
            shutil.copy2(f, orig)
            restored = True
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and str(d).endswith(MODEL_BACKUP_SUFFIX):
            orig = Path(str(d).replace(MODEL_BACKUP_SUFFIX, ""))
            if orig.exists():
                shutil.rmtree(orig)
            shutil.copytree(d, orig)
            restored = True
    return restored


def check_drift() -> dict:
    """
    Compare latest backtest RMSE to previous (overall and per-position).
    Returns dict with drift_detected, current_rmse, previous_rmse, pct_change,
    and per-position drift details.
    """
    if not BACKTEST_RESULTS_DIR.exists():
        return {"drift_detected": False, "error": "No backtest results dir"}
    # Tie-break on filename to avoid same-second mtime collisions in tests/fast filesystems.
    files = sorted(
        BACKTEST_RESULTS_DIR.glob("*.json"),
        key=lambda p: (-p.stat().st_mtime, p.name),
    )
    if len(files) < 2:
        return {"drift_detected": False, "current_rmse": None, "previous_rmse": None}
    try:
        with open(files[0]) as f:
            latest = json.load(f)
        with open(files[1]) as f:
            previous = json.load(f)
    except Exception as e:
        return {"drift_detected": False, "error": str(e)}
    latest_rmse = latest.get("metrics", {}).get("rmse")
    prev_rmse = previous.get("metrics", {}).get("rmse")
    if latest_rmse is None or prev_rmse is None or prev_rmse == 0:
        return {"drift_detected": False, "current_rmse": latest_rmse, "previous_rmse": prev_rmse}
    pct_change = (latest_rmse - prev_rmse) / prev_rmse * 100
    overall_drift = pct_change > DRIFT_THRESHOLD_PCT

    # Per-position drift detection
    position_drift = {}
    latest_by_pos = latest.get("by_position", {})
    prev_by_pos = previous.get("by_position", {})
    any_pos_drift = False
    for pos in ["QB", "RB", "WR", "TE"]:
        l_rmse = latest_by_pos.get(pos, {}).get("rmse")
        p_rmse = prev_by_pos.get(pos, {}).get("rmse")
        if l_rmse is not None and p_rmse is not None and p_rmse > 0:
            pos_pct = (l_rmse - p_rmse) / p_rmse * 100
            pos_drifted = pos_pct > DRIFT_POSITION_THRESHOLD_PCT
            if pos_drifted:
                any_pos_drift = True
            position_drift[pos] = {
                "current_rmse": l_rmse,
                "previous_rmse": p_rmse,
                "pct_change": round(pos_pct, 1),
                "drift_detected": pos_drifted,
            }

    return {
        "drift_detected": overall_drift or any_pos_drift,
        "overall_drift": overall_drift,
        "current_rmse": latest_rmse,
        "previous_rmse": prev_rmse,
        "pct_change": round(pct_change, 1),
        "position_drift": position_drift,
        "any_position_drift": any_pos_drift,
    }


def write_drift_status(drift: dict) -> None:
    """Persist drift status for monitoring dashboards/alerts."""
    payload = {
        "checked_at": datetime.now().isoformat(),
        "drift_threshold_pct": DRIFT_THRESHOLD_PCT,
        **drift,
    }
    DRIFT_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_retrain_status(status: dict, start_ts: float) -> None:
    """Persist retrain execution status and SLA check."""
    elapsed = max(0.0, time.time() - start_ts)
    payload = {
        **status,
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "sla_seconds": RETRAIN_SLA_SECONDS,
        "sla_met": elapsed <= RETRAIN_SLA_SECONDS,
    }
    RETRAIN_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Production: retrain, drift check, rollback")
    parser.add_argument("--retrain", action="store_true", help="Run weekly retrain (backup first)")
    parser.add_argument("--check-drift", action="store_true", help="Check for model drift")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous model backup")
    parser.add_argument("--backup", action="store_true", help="Backup current models only")
    parser.add_argument("--force", action="store_true", help="Force retrain even if schedule says skip")
    args = parser.parse_args()

    if args.backup:
        backup_models()
        print("Models backed up.")
        return
    if args.rollback:
        if rollback_models():
            print("Rollback complete.")
        else:
            print("No backup found to rollback.")
        return
    if args.check_drift:
        drift = check_drift()
        drift["drift_threshold_pct"] = DRIFT_THRESHOLD_PCT
        write_drift_status(drift)
        if drift.get("drift_detected"):
            print("WARNING: Model drift detected above configured threshold.")
        print(json.dumps(drift, indent=2))
        return
    if args.retrain:
        backup_models()
        print("Backed up existing models.")
        if run_weekly_retrain(force=args.force):
            print("Weekly retrain completed.")
        else:
            print("Retrain failed; consider rollback.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
