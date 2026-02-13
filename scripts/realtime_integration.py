"""
Real-Time Data Integration for NFL Predictor
Integrates nflverse play-by-play data for current NFL season.

Key Features:
- Pulls play-by-play data through current week
- Aggregates to player-level weekly stats
- Calculates utilization scores in real-time
- Works for any point during the season
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nfl_data_py as nfl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# REAL-TIME DATA FETCHER
# ============================================================================

class RealTimeDataIntegrator:
    """Integrates current season data with historical training data."""
    
    def __init__(self, current_season=None):
        if current_season is None:
            from src.utils.nfl_calendar import get_current_nfl_season
            current_season = get_current_nfl_season()
        self.current_season = current_season
        self.current_week = self._determine_current_week()
    
    def _determine_current_week(self):
        """Current NFL week from shared nfl_calendar."""
        from src.utils.nfl_calendar import get_current_nfl_week
        info = get_current_nfl_week()
        return info.get("week_num", 0)
    
    def fetch_current_season_pbp(self):
        """Fetch play-by-play data for current season."""
        print(f"Fetching {self.current_season} season play-by-play through Week {self.current_week}...")
        
        try:
            pbp = nfl.import_pbp_data([self.current_season], downcast=True)
            
            # Filter to completed weeks only
            pbp = pbp[pbp['week'] <= self.current_week]
            
            print(f"✅ Loaded {len(pbp):,} plays from {self.current_season} season")
            return pbp
            
        except Exception as e:
            print(f"❌ Error fetching play-by-play: {e}")
            return pd.DataFrame()
    
    def aggregate_to_weekly_stats(self, pbp):
        """Convert play-by-play to weekly player stats."""
        if pbp.empty:
            return pd.DataFrame()
        
        print("Aggregating play-by-play to weekly stats...")
        
        weekly_stats = []
        
        # Process each week
        for week in range(1, self.current_week + 1):
            week_pbp = pbp[pbp['week'] == week]
            
            # Passing stats (QBs)
            passing = self._aggregate_passing(week_pbp, week)
            
            # Rushing stats (RBs, some QBs)
            rushing = self._aggregate_rushing(week_pbp, week)
            
            # Receiving stats (WRs, TEs, RBs)
            receiving = self._aggregate_receiving(week_pbp, week)
            
            weekly_stats.extend([passing, rushing, receiving])
        
        # Combine all stats
        all_stats = pd.concat([df for df in weekly_stats if not df.empty], ignore_index=True)
        
        print(f"✅ Aggregated {len(all_stats):,} player-week records")
        return all_stats
    
    def _aggregate_passing(self, pbp, week):
        """Aggregate passing stats."""
        passing = pbp[pbp['pass_attempt'] == 1].groupby(['passer_player_id', 'passer_player_name', 'posteam']).agg({
            'pass_attempt': 'sum',
            'complete_pass': 'sum',
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum',
            'sack': 'sum',
        }).reset_index()
        
        passing.columns = ['player_id', 'player_name', 'team', 'attempts', 'completions', 
                          'yards', 'tds', 'ints', 'sacks']
        passing['position'] = 'QB'
        passing['week'] = week
        passing['season'] = self.current_season
        
        return passing
    
    def _aggregate_rushing(self, pbp, week):
        """Aggregate rushing stats."""
        rushing = pbp[pbp['rush_attempt'] == 1].groupby(['rusher_player_id', 'rusher_player_name', 'posteam']).agg({
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
        }).reset_index()
        
        rushing.columns = ['player_id', 'player_name', 'team', 'attempts', 'yards', 'tds']
        rushing['position'] = 'RB'  # Simplified - would need roster data for accurate positions
        rushing['week'] = week
        rushing['season'] = self.current_season
        
        return rushing
    
    def _aggregate_receiving(self, pbp, week):
        """Aggregate receiving stats."""
        # First aggregate completions and yards
        receiving = pbp[pbp['pass_attempt'] == 1].groupby(['receiver_player_id', 'receiver_player_name', 'posteam']).agg({
            'complete_pass': 'sum',
            'receiving_yards': 'sum',
            'pass_touchdown': 'sum',
        }).reset_index()
        
        # Calculate targets separately and merge
        targets = pbp[pbp['pass_attempt'] == 1].groupby('receiver_player_id').size().reset_index(name='targets')
        receiving = receiving.merge(targets, left_on='receiver_player_id', right_on='receiver_player_id', how='left')
        
        receiving.columns = ['player_id', 'player_name', 'team', 'receptions', 'yards', 'tds', 'targets']
        receiving['position'] = 'WR'  # Simplified
        receiving['week'] = week
        receiving['season'] = self.current_season
        
        return receiving
    
    def _normalize_stats_for_utilization(self, stats):
        """Normalize aggregated stats to columns expected by single-source utilization_score."""
        if stats.empty:
            return stats
        df = stats.copy()
        if 'season' not in df.columns:
            df['season'] = self.current_season
        if 'rushing_attempts' not in df.columns:
            df['rushing_attempts'] = 0
            if 'attempts' in df.columns and 'position' in df.columns:
                df.loc[df['position'].isin(['RB', 'WR', 'TE']), 'rushing_attempts'] = df.loc[df['position'].isin(['RB', 'WR', 'TE']), 'attempts'].fillna(0)
        if 'targets' not in df.columns:
            df['targets'] = 0
        df['targets'] = df['targets'].fillna(0)
        if 'receptions' not in df.columns:
            df['receptions'] = 0
        df['receptions'] = df['receptions'].fillna(0)
        if 'receiving_yards' not in df.columns and 'yards' in df.columns:
            df['receiving_yards'] = df['yards'].fillna(0)
        elif 'receiving_yards' not in df.columns:
            df['receiving_yards'] = 0
        if 'receiving_tds' not in df.columns:
            df['receiving_tds'] = df.get('tds', pd.Series(0, index=df.index)).fillna(0)
            if 'position' in df.columns:
                df.loc[df['position'] == 'QB', 'receiving_tds'] = 0
        if 'passing_attempts' not in df.columns:
            df['passing_attempts'] = 0
            if 'attempts' in df.columns and 'position' in df.columns:
                df.loc[df['position'] == 'QB', 'passing_attempts'] = df.loc[df['position'] == 'QB', 'attempts'].fillna(0)
        if 'passing_tds' not in df.columns:
            df['passing_tds'] = 0
            if 'tds' in df.columns and 'position' in df.columns:
                df.loc[df['position'] == 'QB', 'passing_tds'] = df.loc[df['position'] == 'QB', 'tds'].fillna(0)
        if 'rushing_tds' not in df.columns:
            df['rushing_tds'] = 0
        if 'name' not in df.columns and 'player_name' in df.columns:
            df['name'] = df['player_name']
        elif 'name' not in df.columns:
            df['name'] = df.get('player_id', '')
        df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        return df

    def calculate_utilization_scores(self, stats):
        """Calculate utilization scores using single-source utilization_score (config weights, optional bounds)."""
        from src.features.utilization_score import calculate_utilization_scores, load_percentile_bounds
        from config.settings import MODELS_DIR
        print("Calculating utilization scores...")
        df = self._normalize_stats_for_utilization(stats)
        if df.empty:
            stats['utilization_score'] = 50.0
            return stats
        bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
        percentile_bounds = load_percentile_bounds(bounds_path) if bounds_path.exists() else None
        result = calculate_utilization_scores(df, team_df=pd.DataFrame(), percentile_bounds=percentile_bounds)
        key = [c for c in ['player_id', 'season', 'week'] if c in stats.columns]
        if key and len(result) > 0 and 'utilization_score' in result.columns:
            stats = stats.merge(
                result[key + ['utilization_score']].drop_duplicates(key),
                on=key, how='left'
            )
            stats['utilization_score'] = stats['utilization_score'].fillna(50).clip(0, 100)
        else:
            stats = result if len(result) > 0 else stats
        print(f"✅ Calculated utilization scores (mean: {stats['utilization_score'].mean():.1f})")
        return stats
    
    def merge_with_historical(self, current_stats, historical_path='data/nfl_stats.db'):
        """Merge current season with historical training data."""
        print("Merging with historical data...")
        
        try:
            # Load historical data
            import sqlite3
            conn = sqlite3.connect(historical_path)
            historical = pd.read_sql("SELECT * FROM player_weekly_stats WHERE season < ?", conn, params=[self.current_season])
            conn.close()
            
            # Combine
            combined = pd.concat([historical, current_stats], ignore_index=True)
            
            print(f"✅ Combined dataset: {len(combined):,} records ({len(historical):,} historical + {len(current_stats):,} current)")
            return combined
            
        except Exception as e:
            print(f"⚠️  Historical data not found: {e}")
            return current_stats
    
    def save_for_dashboard(self, stats, output_path='data/current_season_stats.parquet'):
        """Save processed stats for dashboard consumption."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        stats.to_parquet(output, index=False)
        print(f"✅ Saved to {output}")
    
    def run_full_pipeline(self):
        """Execute complete real-time integration pipeline."""
        print("="*70)
        print("REAL-TIME DATA INTEGRATION PIPELINE")
        print("="*70)
        print(f"Season: {self.current_season}")
        print(f"Current Week: {self.current_week}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
        # Step 1: Fetch play-by-play
        pbp = self.fetch_current_season_pbp()
        if pbp.empty:
            print("❌ No data fetched. Exiting.")
            return None
        
        # Step 2: Aggregate to weekly
        weekly_stats = self.aggregate_to_weekly_stats(pbp)
        
        # Step 3: Calculate utilization
        weekly_stats = self.calculate_utilization_scores(weekly_stats)
        
        # Step 4: Merge with historical
        full_dataset = self.merge_with_historical(weekly_stats)
        
        # Step 5: Save
        self.save_for_dashboard(full_dataset)
        
        print("="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)
        
        return full_dataset


# ============================================================================
# SUPER BOWL SPECIFIC EXTRACTOR
# ============================================================================

class SuperBowlPredictor:
    """Extract stats for Super Bowl teams and predict utilization."""
    
    def __init__(self, integrator):
        self.integrator = integrator
        self.super_bowl_teams = ['NE', 'SEA']  # Patriots vs Seahawks
    
    def get_super_bowl_rosters(self):
        """Get rosters for Super Bowl teams."""
        try:
            rosters = nfl.import_rosters([self.current_season])
            sb_rosters = rosters[rosters['team'].isin(self.super_bowl_teams)]
            return sb_rosters
        except:
            return pd.DataFrame()
    
    def extract_recent_performance(self, stats):
        """Get last 3 weeks of stats for SB players."""
        recent_weeks = list(range(self.integrator.current_week - 2, self.integrator.current_week + 1))
        
        sb_stats = stats[
            (stats['team'].isin(self.super_bowl_teams)) &
            (stats['week'].isin(recent_weeks))
        ]
        
        # Average last 3 weeks
        recent_avg = sb_stats.groupby(['player_id', 'player_name', 'team', 'position']).agg({
            'utilization_score': 'mean',
            'rush_share': 'mean',
            'target_share': 'mean',
            'snap_share': 'mean',
        }).reset_index()
        
        return recent_avg
    
    def predict_super_bowl(self, stats):
        """Generate Super Bowl utilization predictions."""
        recent = self.extract_recent_performance(stats)
        
        # Add confidence intervals (±10%)
        recent['util_low'] = recent['utilization_score'] * 0.9
        recent['util_high'] = recent['utilization_score'] * 1.1
        
        # Classify tier
        def classify(score):
            if score >= 85: return 'elite'
            elif score >= 70: return 'high'
            elif score >= 50: return 'moderate'
            else: return 'low'
        
        recent['tier'] = recent['utilization_score'].apply(classify)
        
        return recent.sort_values('utilization_score', ascending=False)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize integrator
    integrator = RealTimeDataIntegrator()
    
    # Run pipeline
    full_data = integrator.run_full_pipeline()
    
    if full_data is not None:
        # Generate Super Bowl predictions
        print("\n" + "="*70)
        print("SUPER BOWL LX PREDICTIONS")
        print("="*70)
        
        sb_predictor = SuperBowlPredictor(integrator)
        sb_predictions = sb_predictor.predict_super_bowl(full_data)
        
        print("\n📊 Top Players by Utilization:")
        print(sb_predictions[['player_name', 'team', 'position', 'utilization_score', 'tier']].head(15).to_string(index=False))
        
        # Save predictions
        sb_predictions.to_parquet('data/super_bowl_predictions.parquet', index=False)
        print("\n✅ Super Bowl predictions saved to data/super_bowl_predictions.parquet")
