"""
Update data with current NFL season stats (dynamic year).

This script:
1. Loads existing data (historical seasons)
2. Aggregates current season stats from play-by-play data
3. Merges and saves updated predictions file

Usage:
  python scripts/update_2025_data.py              # Use current NFL season
  python scripts/update_2025_data.py --season 2026  # Specific season
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.pbp_stats_aggregator import PBPStatsAggregator
from src.features.utilization import engineer_all_features


def update_with_current_season_data(season: int = None):
    """Update the predictions file with current (or specified) season data."""
    if season is None:
        from src.utils.nfl_calendar import get_current_nfl_season
        season = get_current_nfl_season()

    data_dir = Path(__file__).parent.parent / "data"
    predictions_file = data_dir / "daily_predictions.parquet"

    # Load existing data
    print("Loading existing data...")
    existing_df = pd.read_parquet(predictions_file)
    print(f"  Existing data: {len(existing_df)} rows, seasons {existing_df['season'].unique()}")

    # Remove any existing data for target season (in case of re-run)
    existing_df = existing_df[existing_df['season'] != season]
    print(f"  After removing season {season}: {len(existing_df)} rows")

    # Load season data from PBP
    print(f"\nLoading season {season} data from play-by-play...")
    aggregator = PBPStatsAggregator()
    stats_season = aggregator.aggregate_all_stats(season)
    print(f"  Season {season} data: {len(stats_season)} rows")

    # Create a clean copy with all columns from existing data
    print("\nAligning columns...")
    stats_clean = pd.DataFrame()

    for col in existing_df.columns:
        if col in stats_season.columns:
            stats_clean[col] = stats_season[col].values
        else:
            stats_clean[col] = np.nan

    print(f"  Aligned season {season} data: {len(stats_clean)} rows, {len(stats_clean.columns)} columns")

    # Combine
    print("\nCombining datasets...")
    combined = pd.concat([existing_df, stats_clean], ignore_index=True)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    print(f"  Combined: {len(combined)} rows")
    print(f"  Seasons: {sorted(combined['season'].unique())}")

    # Add basic rolling features for target season only
    print(f"\nAdding basic features for season {season} data...")
    mask = combined['season'] == season
    combined.loc[mask, 'fp_rolling_3'] = combined.loc[mask].groupby('name')['fantasy_points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Save
    print(f"\nSaving to {predictions_file}...")
    combined.to_parquet(predictions_file, index=False)

    # Verify
    verify_df = pd.read_parquet(predictions_file)
    print(f"\nVerification:")
    print(f"  Total rows: {len(verify_df)}")
    print(f"  Seasons: {sorted(verify_df['season'].unique())}")

    df_season = verify_df[verify_df['season'] == season]
    print(f"\nSeason {season} Summary:")
    print(f"  Rows: {len(df_season)}")
    print(f"  Weeks: {sorted(df_season['week'].unique())}")
    print(f"  Max week: {df_season['week'].max()}")

    week_21 = df_season[df_season['week'] == 21]
    if len(week_21) > 0:
        print(f"\n  Teams in Week 21 (Conference Championship):")
        print(f"    {sorted(week_21['team'].unique())}")

    print("\nData update complete!")
    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update predictions with current (or specified) NFL season data")
    parser.add_argument("--season", type=int, default=None, help="NFL season year (default: current NFL season)")
    args = parser.parse_args()
    update_with_current_season_data(season=args.season)
