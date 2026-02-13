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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
DRIFT_THRESHOLD_PCT = 20.0
MODEL_BACKUP_SUFFIX = ".bak"


def run_weekly_retrain() -> bool:
    """Run data refresh and full model training. Returns True on success."""
    try:
        from src.data.auto_refresh import auto_refresh_data
        auto_refresh_data()
    except Exception as e:
        print(f"Data refresh failed: {e}")
        return False
    try:
        from src.models.train import train_models
        train_models(tune_hyperparameters=False)
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def backup_models() -> None:
    """Copy current model files to .bak so we can rollback."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODELS_DIR.glob("*.joblib"):
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
    Compare latest backtest RMSE to previous. Returns dict with
    drift_detected, current_rmse, previous_rmse, pct_change.
    """
    if not BACKTEST_RESULTS_DIR.exists():
        return {"drift_detected": False, "error": "No backtest results dir"}
    files = sorted(BACKTEST_RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
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
    return {
        "drift_detected": pct_change > DRIFT_THRESHOLD_PCT,
        "current_rmse": latest_rmse,
        "previous_rmse": prev_rmse,
        "pct_change": round(pct_change, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Production: retrain, drift check, rollback")
    parser.add_argument("--retrain", action="store_true", help="Run weekly retrain (backup first)")
    parser.add_argument("--check-drift", action="store_true", help="Check for model drift")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous model backup")
    parser.add_argument("--backup", action="store_true", help="Backup current models only")
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
        print(json.dumps(drift, indent=2))
        return
    if args.retrain:
        backup_models()
        print("Backed up existing models.")
        if run_weekly_retrain():
            print("Weekly retrain completed.")
        else:
            print("Retrain failed; consider rollback.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
