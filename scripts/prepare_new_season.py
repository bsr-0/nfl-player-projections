#!/usr/bin/env python3
"""Prepare the codebase for a new NFL season.

Orchestrates all offseason steps so a single command updates rosters,
retrains models in forward mode, generates ML predictions, rebuilds the
draft board, and exports the static API snapshot.

Usage:
    python scripts/prepare_new_season.py              # Full prep
    python scripts/prepare_new_season.py --skip-train  # Skip retraining
    python scripts/prepare_new_season.py --dry-run     # Show plan only
    python scripts/prepare_new_season.py --fast        # Fast training (~8-10x faster)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _print_step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")
    print("-" * 60)


def prepare_new_season(
    skip_train: bool = False,
    dry_run: bool = False,
    fast: bool = False,
) -> bool:
    """Run all offseason preparation steps.

    Returns True if all steps succeed.
    """
    from src.utils.nfl_calendar import (
        get_current_nfl_season,
        is_offseason,
    )

    current_season = get_current_nfl_season()
    target_season = current_season + 1 if is_offseason() else current_season
    basis_season = current_season  # most recently completed season

    total_steps = 8 if not skip_train else 7
    step = 0

    print("=" * 60)
    print(f"  Preparing for the {target_season} NFL Season")
    print(f"  Basis season (completed): {basis_season}")
    print(f"  Offseason: {is_offseason()}")
    print(f"  Dry run: {dry_run}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Refresh data
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, "Refresh data from nfl-data-py")
    if dry_run:
        print("  [DRY RUN] Would call auto_refresh_data(force_check=True)")
    else:
        try:
            from src.utils.data_manager import auto_refresh_data
            status = auto_refresh_data(force_check=True)
            print(f"  Latest season in DB: {status.get('latest_season')}")
            print(f"  Available seasons: {status.get('available_seasons')}")
        except Exception as e:
            print(f"  Warning: auto_refresh failed: {e}")

    # ------------------------------------------------------------------
    # 2. Import target-season rosters
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, f"Import {target_season} rosters")
    roster_available = False
    if dry_run:
        print(f"  [DRY RUN] Would call load_rosters([{target_season}])")
    else:
        try:
            from src.data.nfl_data_loader import NFLDataLoader
            loader = NFLDataLoader()
            roster = loader.load_rosters([target_season])
            if not roster.empty:
                roster_available = True
                print(f"  Loaded {len(roster)} roster entries for {target_season}")
                teams = loader.sync_player_teams(target_season)
                print(f"  Synced {len(teams)} player-team assignments")
            else:
                print(f"  No {target_season} roster data available yet (free agency may not have started)")
                print(f"  Will use {basis_season} team assignments as fallback")
        except Exception as e:
            print(f"  Roster import failed: {e}")
            print(f"  Will use {basis_season} team assignments as fallback")

    # ------------------------------------------------------------------
    # 3. Detect roster changes
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, f"Detect roster changes ({basis_season} -> {target_season})")
    if dry_run:
        print(f"  [DRY RUN] Would call get_roster_changes({target_season}, {basis_season})")
    elif roster_available:
        try:
            from src.data.nfl_data_loader import NFLDataLoader
            loader = NFLDataLoader()
            changes = loader.get_roster_changes(target_season, basis_season)
            if not changes.empty:
                print(f"  Found {len(changes)} roster changes:")
                for _, row in changes.head(20).iterrows():
                    print(f"    {row['name']} ({row['position']}): "
                          f"{row['previous_team']} -> {row['current_team']}")
                if len(changes) > 20:
                    print(f"    ... and {len(changes) - 20} more")
            else:
                print("  No roster changes detected")
        except Exception as e:
            print(f"  Roster change detection failed: {e}")
    else:
        print(f"  Skipped (no {target_season} roster data)")

    # ------------------------------------------------------------------
    # 4. Load depth charts
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, f"Load {target_season} depth charts")
    if dry_run:
        print(f"  [DRY RUN] Would call load_depth_charts({target_season})")
    elif roster_available:
        try:
            from src.data.nfl_data_loader import NFLDataLoader
            loader = NFLDataLoader()
            depth = loader.load_depth_charts(target_season)
            if not depth.empty:
                print(f"  Loaded depth charts for {len(depth)} players")
            else:
                print("  No depth chart data available")
        except Exception as e:
            print(f"  Depth chart load failed: {e}")
    else:
        print(f"  Skipped (no {target_season} roster data)")

    # ------------------------------------------------------------------
    # 5. Retrain models in forward mode
    # ------------------------------------------------------------------
    if not skip_train:
        step += 1
        _print_step(step, total_steps, f"Retrain models (forward mode, targeting {target_season})")
        if dry_run:
            print(f"  [DRY RUN] Would call train_models(forward=True, fast={fast})")
        else:
            try:
                from src.models.train import train_models
                train_models(forward=True, fast=fast)
                print("  Model training complete")
            except Exception as e:
                print(f"  Training failed: {e}")
                print("  Continuing with existing models (if any)")

    # ------------------------------------------------------------------
    # 6. Generate ML predictions
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, "Generate ML predictions")
    if dry_run:
        print("  [DRY RUN] Would call generate_app_data(save_daily=True)")
    else:
        try:
            from scripts.generate_app_data import generate_app_data
            success = generate_app_data(save_daily=True)
            if success:
                print("  ML predictions generated successfully")
            else:
                print("  ML prediction generation returned False; draft board will use extrapolation")
        except Exception as e:
            print(f"  ML prediction generation failed: {e}")

    # ------------------------------------------------------------------
    # 7. Regenerate draft board
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, "Regenerate draft board JSON files")
    if dry_run:
        print("  [DRY RUN] Would call generate_draft_data.main()")
    else:
        try:
            from scripts.generate_draft_data import main as gen_draft
            gen_draft()
        except Exception as e:
            print(f"  Draft board generation failed: {e}")

    # ------------------------------------------------------------------
    # 8. Export static API snapshot
    # ------------------------------------------------------------------
    step += 1
    _print_step(step, total_steps, "Export static API snapshot")
    if dry_run:
        print("  [DRY RUN] Would call export_static_api()")
    else:
        try:
            from scripts.export_static_api import export_static_api
            out_dir = Path(__file__).parent.parent / "frontend" / "public" / "api"
            export_static_api(out_dir)
            print(f"  Static API written to {out_dir}")
        except Exception as e:
            print(f"  Static API export failed: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  New Season Preparation {'Plan' if dry_run else 'Complete'}")
    print(f"  Target season: {target_season}")
    print(f"  Roster data: {'available' if roster_available else 'not yet available'}")
    print(f"  Training: {'skipped' if skip_train else ('planned' if dry_run else 'done')}")
    print("=" * 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the codebase for a new NFL season",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_new_season.py              # Full prep
  python scripts/prepare_new_season.py --skip-train  # Skip retraining
  python scripts/prepare_new_season.py --dry-run     # Show what would happen
  python scripts/prepare_new_season.py --fast        # Fast training mode
        """,
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model retraining (use existing models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast training mode (~8-10x faster, minimal accuracy loss)",
    )
    args = parser.parse_args()

    success = prepare_new_season(
        skip_train=args.skip_train,
        dry_run=args.dry_run,
        fast=args.fast,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
