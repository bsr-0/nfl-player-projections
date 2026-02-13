"""
Expand NFL data back to 2012.

This script:
1. Loads weekly stats from 2012-2019
2. Loads injury data (2009-2024)
3. Loads rookie/draft data (1980 through current season)
4. Integrates all into the main predictions file
"""
import os
import sys
import ssl
import certifi
from pathlib import Path

# Fix SSL certificates
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import nfl_data_py as nfl


def load_weekly_stats_2012_2019():
    """Load weekly stats for 2012-2019."""
    print("\n📊 Loading weekly stats 2012-2019...")
    
    years = list(range(1999, 2020))
    all_data = []
    
    for year in years:
        try:
            print(f"  Loading {year}...", end=" ")
            df = nfl.import_weekly_data([year])
            df['season'] = year
            all_data.append(df)
            print(f"{len(df):,} records")
        except Exception as e:
            print(f"Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"  Total 2012-2019: {len(combined):,} records")
        return combined
    return pd.DataFrame()


def load_injury_data():
    """Load injury data for all available years."""
    print("\n🏥 Loading injury data...")
    
    from config.settings import CURRENT_NFL_SEASON
    years = list(range(1999, CURRENT_NFL_SEASON + 1))
    all_injuries = []
    
    for year in years:
        try:
            df = nfl.import_injuries([year])
            if len(df) > 0:
                all_injuries.append(df)
                print(f"  {year}: {len(df):,} records")
        except Exception as e:
            pass
    
    if all_injuries:
        combined = pd.concat(all_injuries, ignore_index=True)
        print(f"  Total injuries: {len(combined):,} records")
        return combined
    return pd.DataFrame()


def load_rookie_data():
    """Load draft/rookie data."""
    print("\n🎓 Loading rookie/draft data...")
    
    try:
        draft = nfl.import_draft_picks()
        from config.settings import CURRENT_NFL_SEASON
        print(f"  Draft picks: {len(draft):,} records (1980-{CURRENT_NFL_SEASON})")
        
        combine = nfl.import_combine_data()
        print(f"  Combine data: {len(combine):,} records (2000-{CURRENT_NFL_SEASON})")
        
        return draft, combine
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame(), pd.DataFrame()


def integrate_data():
    """Integrate all data sources into main predictions file."""
    
    data_dir = Path(__file__).parent.parent / "data"
    predictions_file = data_dir / "daily_predictions.parquet"
    
    # Load existing data
    print("\n📁 Loading existing data...")
    existing_df = pd.read_parquet(predictions_file)
    print(f"  Existing: {len(existing_df):,} rows, seasons {sorted(existing_df['season'].unique())}")
    
    # Load new weekly stats (2012-2019)
    new_weekly = load_weekly_stats_2012_2019()
    
    if new_weekly.empty:
        print("  No new weekly data loaded")
        return
    
    # Filter to only years not already in existing data
    existing_seasons = set(existing_df['season'].unique())
    new_weekly = new_weekly[~new_weekly['season'].isin(existing_seasons)]
    print(f"\n  New seasons to add: {sorted(new_weekly['season'].unique())}")
    print(f"  New records: {len(new_weekly):,}")
    
    if new_weekly.empty:
        print("  All seasons already loaded")
        return
    
    # Align columns
    print("\n🔧 Aligning columns...")
    
    # Get common columns
    existing_cols = set(existing_df.columns)
    new_cols = set(new_weekly.columns)
    common_cols = existing_cols.intersection(new_cols)
    print(f"  Common columns: {len(common_cols)}")
    
    # Create aligned dataframe
    new_weekly_aligned = pd.DataFrame()
    for col in existing_df.columns:
        if col in new_weekly.columns:
            new_weekly_aligned[col] = new_weekly[col].values
        else:
            new_weekly_aligned[col] = np.nan
    
    # Combine
    print("\n📦 Combining datasets...")
    combined = pd.concat([existing_df, new_weekly_aligned], ignore_index=True)
    combined = combined.sort_values(['season', 'week', 'position', 'player_id']).reset_index(drop=True)
    
    print(f"  Combined: {len(combined):,} rows")
    print(f"  Seasons: {sorted(combined['season'].unique())}")
    
    # Save
    print("\n💾 Saving expanded dataset...")
    combined.to_parquet(predictions_file, index=False)
    print(f"  Saved to: {predictions_file}")
    
    # Load and save injury data
    injuries = load_injury_data()
    if not injuries.empty:
        injury_file = data_dir / "injuries.parquet"
        injuries.to_parquet(injury_file, index=False)
        print(f"  Injuries saved to: {injury_file}")
    
    # Load and save rookie data
    draft, combine = load_rookie_data()
    if not draft.empty:
        draft_file = data_dir / "draft_picks.parquet"
        draft.to_parquet(draft_file, index=False)
        print(f"  Draft data saved to: {draft_file}")
    if not combine.empty:
        combine_file = data_dir / "combine_data.parquet"
        combine.to_parquet(combine_file, index=False)
        print(f"  Combine data saved to: {combine_file}")
    
    # Summary
    print("\n" + "="*70)
    print("EXPANSION COMPLETE")
    print("="*70)
    from config.settings import CURRENT_NFL_SEASON
    print(f"""
    Weekly Stats: {len(combined):,} records ({int(combined['season'].min())}-{int(combined['season'].max())})
    Injuries: {len(injuries):,} records (2009-{CURRENT_NFL_SEASON})
    Draft Picks: {len(draft):,} records (1980-{CURRENT_NFL_SEASON})
    Combine: {len(combine):,} records (2000-{CURRENT_NFL_SEASON})
    """)


if __name__ == "__main__":
    integrate_data()
