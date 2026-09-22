#!/usr/bin/env python3
"""Train + walk-forward-evaluate Plan A's team-allocation share models.

Standalone from src/models/train.py and scripts/train_game_outcome_model.py
-- see docs/TEAM_LEVEL_ALLOCATION_MODELS.md and src/models/team_allocation/.

Usage:
    python scripts/train_team_share_model.py                    # all 4 targets, full history
    python scripts/train_team_share_model.py --target targets
    python scripts/train_team_share_model.py --seasons 2013 2025
    python scripts/train_team_share_model.py --no-save          # backtest only

Writes, per target per model arm, a joblib artifact + JSON metadata sidecar
to data/models/ (MODELS_DIR), and prints the walk-forward metrics table.
Requires scripts/build_team_week_player_shares.py --write to have been run
first.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODELS_DIR
from src.evaluation.team_share_backtester import run_walk_forward_backtest
from src.models.team_allocation.features import VOLUME_COLS, feature_columns, filter_population, load_share_rows
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel, save_model


def _print_report(report: dict) -> None:
    target = report["target"]
    print(f"\n[{target}] {report['n_rows_total']:,} total player-weeks, {report['n_folds']} walk-forward fold(s)\n")
    for fold in report["folds"]:
        print(
            f"Fold {fold['fold']}: train seasons {fold['train_seasons'][0]}-{fold['train_seasons'][-1]} "
            f"({fold['n_train']} rows) -> test season(s) {fold['test_seasons']} ({fold['n_test']} rows)"
        )
        for arm, m in fold["arms"].items():
            print(f"    {arm:10s} mae={m['mae']:.4f}  rmse={m['rmse']:.4f}  r2={m['r2']:.3f}")
    print("Pooled across all folds:")
    for arm, m in report["pooled"].items():
        print(f"    {arm:10s} n={m['n']:5d}  mae={m['mae']:.4f}  rmse={m['rmse']:.4f}  r2={m['r2']:.3f}")

    print("\nBy position (pooled):")
    for arm, m in report["pooled"].items():
        for pos, seg in m.get("by_position", {}).items():
            print(f"    {arm:10s} {pos:4s} n={seg['n']:5d}  mae={seg['mae']:.4f}")

    print("\nBy cold-start (pooled; 0=established, 1=cold start):")
    for arm, m in report["pooled"].items():
        for flag, seg in m.get("by_cold_start", {}).items():
            print(f"    {arm:10s} cold_start={flag}  n={seg['n']:5d}  mae={seg['mae']:.4f}")

    baseline_mae = report["pooled"].get("rolling3", {}).get("mae")
    print("\nvs. rolling-3 baseline (paired bootstrap CI on the MAE delta; "
          "negative = candidate has lower error):")
    for arm in ("ridge", "xgboost"):
        m = report["pooled"].get(arm, {})
        mae = m.get("mae")
        boot = m.get("vs_rolling3_bootstrap", {})
        if mae is None or baseline_mae is None or not boot:
            continue
        sig = "PASSES criterion 1 (95% CI excludes 0)" if boot.get("significant_improvement") else \
              "does NOT reliably beat baseline (CI includes 0 or is positive)"
        print(
            f"    {arm}: mae={mae:.4f} vs rolling3={baseline_mae:.4f}  "
            f"delta CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]  n_bootstrap={boot['n_bootstrap']}  "
            f"-> {sig}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=VOLUME_COLS, default=None, help="Default: run all 4.")
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"), default=None)
    ap.add_argument("--n-test-seasons", type=int, default=None)
    ap.add_argument("--no-save", action="store_true", help="Run the backtest only; skip persisting final models.")
    args = ap.parse_args()

    seasons = list(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else None
    targets = [args.target] if args.target else list(VOLUME_COLS)

    for target in targets:
        print(f"\nRunning walk-forward backtest for target={target!r}...")
        report = run_walk_forward_backtest(target, seasons=seasons, n_test_seasons=args.n_test_seasons)
        _print_report(report)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "seasons_requested": seasons,
            "no_save": args.no_save,
            "backtest": report,
        }

        # Population/feature-schema info is cheap (a read + column-name
        # audit, no model fitting) and belongs in the metadata sidecar even
        # on a bare --no-save evaluation run -- otherwise a pure backtest
        # run leaves nothing diffable across runs, and "did criterion 1
        # pass" becomes something only readable from stdout at the moment
        # the command was run. Model fitting/saving is the only part that
        # actually gets skipped for --no-save.
        df = load_share_rows(seasons=seasons)
        df = filter_population(df, target)
        feat_cols = feature_columns(df)
        label_col = f"share_of_team_{target}"
        metadata["feature_columns"] = feat_cols
        metadata["n_training_rows"] = int(len(df))

        if not args.no_save:
            print(f"\nFitting final {target} models on the full available history...")
            X, y = df[feat_cols], df[label_col].to_numpy()

            ridge = ShareRidgeModel().fit(X, y)
            xgb_model = ShareXGBModel().fit(X, y)

            save_model(ridge, MODELS_DIR / f"team_share_{target}_ridge.joblib")
            save_model(xgb_model, MODELS_DIR / f"team_share_{target}_xgb.joblib")

        metadata_path = MODELS_DIR / f"team_share_{target}_model_metadata.json"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if args.no_save:
            print(f"Wrote backtest-only metadata (no models saved) to {metadata_path}")
        else:
            print(f"Saved {target} models + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
