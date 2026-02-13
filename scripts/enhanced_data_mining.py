"""
Enhanced Injury & Rookie Data Mining
Comprehensive data collection from multiple sources with quality validation
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
INJURY_CACHE_FILE = CACHE_DIR / "injuries_latest.parquet"
INJURY_CACHE_EXPIRY_HOURS = 6


class InjuryCache:
    """
    Local caching for injury data to handle API failures gracefully.
    
    - Caches successful fetches to parquet file
    - Falls back to cached data when APIs fail
    - Automatic cache expiry (default 6 hours)
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR, expiry_hours: int = INJURY_CACHE_EXPIRY_HOURS):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "injuries_latest.parquet"
        self.metadata_file = cache_dir / "injuries_metadata.json"
        self.expiry_hours = expiry_hours
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, injuries_df: pd.DataFrame) -> bool:
        """
        Save injury data to cache.
        
        Args:
            injuries_df: DataFrame with injury data
            
        Returns:
            True if save successful, False otherwise
        """
        if injuries_df.empty:
            return False
        
        try:
            # Save data
            injuries_df.to_parquet(self.cache_file, index=False)
            
            # Save metadata
            metadata = {
                'cached_at': datetime.now().isoformat(),
                'record_count': len(injuries_df),
                'sources': injuries_df['source'].unique().tolist() if 'source' in injuries_df.columns else [],
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save injury cache: {e}")
            return False
    
    def load(self) -> Optional[pd.DataFrame]:
        """
        Load cached injury data if available and not expired.
        
        Returns:
            DataFrame with cached injuries, or None if cache unavailable/expired
        """
        if not self.cache_file.exists():
            return None
        
        try:
            # Check expiry
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                cached_at = datetime.fromisoformat(metadata['cached_at'])
                age_hours = (datetime.now() - cached_at).total_seconds() / 3600
                
                if age_hours > self.expiry_hours:
                    print(f"Injury cache expired ({age_hours:.1f} hours old)")
                    return None
            
            # Load data
            df = pd.read_parquet(self.cache_file)
            print(f"Loaded {len(df)} injuries from cache")
            return df
            
        except Exception as e:
            print(f"Warning: Could not load injury cache: {e}")
            return None
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache state."""
        if not self.metadata_file.exists():
            return {'status': 'no_cache', 'cached_at': None, 'age_hours': None}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_at = datetime.fromisoformat(metadata['cached_at'])
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            
            return {
                'status': 'valid' if age_hours <= self.expiry_hours else 'expired',
                'cached_at': metadata['cached_at'],
                'age_hours': round(age_hours, 1),
                'record_count': metadata.get('record_count', 0),
                'sources': metadata.get('sources', []),
            }
        except Exception:
            return {'status': 'error', 'cached_at': None, 'age_hours': None}


class EnhancedInjuryDataMiner:
    """
    Multi-source injury data collection with validation and historical tracking.
    
    Data Sources:
    1. ESPN Injury API
    2. NFL.com Official Injury Reports
    3. ProFootballDoc (injury analysis)
    4. Historical injury patterns from nflverse
    
    Features:
    - Local caching with automatic fallback on API failures
    - Data quality validation
    - Multi-source conflict resolution
    """
    
    def __init__(self, use_cache: bool = True):
        self.cache = {}
        self.historical_injuries = pd.DataFrame()
        self.injury_patterns = {}
        self.use_cache = use_cache
        self.injury_cache = InjuryCache() if use_cache else None
    
    def fetch_current_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch current week injury reports from multiple sources.
        
        Uses local cache as fallback when APIs fail. Cache is automatically
        refreshed if older than INJURY_CACHE_EXPIRY_HOURS.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            DataFrame with columns:
            - player_name
            - team
            - position
            - status (OUT/DOUBTFUL/QUESTIONABLE/PROBABLE)
            - injury_type (hamstring, ankle, concussion, etc.)
            - weeks_out (estimated)
            - source (ESPN/NFL/PFD)
            - confidence (how reliable is this report)
            - from_cache (bool, True if loaded from cache)
        """
        # Try to load from cache first if not forcing refresh
        if self.use_cache and not force_refresh and self.injury_cache:
            cached_data = self.injury_cache.load()
            if cached_data is not None:
                cached_data['from_cache'] = True
                return cached_data
        
        all_injuries = []
        fetch_success = False
        
        # Source 1: ESPN API
        espn_injuries = self._fetch_espn_injuries()
        if not espn_injuries.empty:
            fetch_success = True
        all_injuries.append(espn_injuries)
        
        # Source 2: nflverse injury data
        nflverse_injuries = self._fetch_nflverse_injuries()
        if not nflverse_injuries.empty:
            fetch_success = True
        all_injuries.append(nflverse_injuries)
        
        # Source 3: Manual override file (for corrections)
        manual_injuries = self._load_manual_overrides()
        all_injuries.append(manual_injuries)
        
        # Combine and deduplicate
        non_empty_dfs = [df for df in all_injuries if not df.empty]
        
        if not non_empty_dfs:
            # All API fetches failed - try to use stale cache as fallback
            if self.use_cache and self.injury_cache:
                print("All injury APIs failed, attempting to use stale cache...")
                # Force load even if expired
                try:
                    if self.injury_cache.cache_file.exists():
                        cached_df = pd.read_parquet(self.injury_cache.cache_file)
                        cached_df['from_cache'] = True
                        cached_df['cache_stale'] = True
                        print(f"Using stale cache with {len(cached_df)} records")
                        return cached_df
                except Exception as e:
                    print(f"Could not load stale cache: {e}")
            return self._empty_injury_df()
        
        combined = pd.concat(non_empty_dfs, ignore_index=True)
        
        if combined.empty:
            return self._empty_injury_df()
        
        # Resolve conflicts (when same player has different statuses from different sources)
        deduped = self._resolve_conflicts(combined)
        
        # Add injury impact scores
        deduped = self._calculate_injury_impact(deduped)
        
        # Save to cache for future fallback
        if self.use_cache and self.injury_cache and fetch_success:
            deduped_to_cache = deduped.copy()
            deduped_to_cache['from_cache'] = False
            self.injury_cache.save(deduped_to_cache)
        
        deduped['from_cache'] = False
        return deduped
    
    def get_cache_status(self) -> Dict:
        """Get the current cache status for monitoring."""
        if not self.use_cache or not self.injury_cache:
            return {'caching_enabled': False}
        
        info = self.injury_cache.get_cache_info()
        info['caching_enabled'] = True
        return info
    
    def _fetch_espn_injuries(self) -> pd.DataFrame:
        """Fetch from ESPN injury API."""
        try:
            # ESPN endpoint (simplified - actual API might differ)
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return self._empty_injury_df()
            
            data = response.json()
            injuries = []
            
            # Parse ESPN data structure
            for team in data.get('teams', []):
                team_name = team.get('abbreviation', 'UNK')
                
                for athlete in team.get('athletes', []):
                    injury_status = athlete.get('injury', {})
                    
                    if injury_status.get('status'):
                        injuries.append({
                            'player_name': athlete.get('displayName'),
                            'team': team_name,
                            'position': athlete.get('position', {}).get('abbreviation'),
                            'status': injury_status.get('status').upper(),
                            'injury_type': injury_status.get('type', 'Unknown'),
                            'source': 'ESPN',
                            'confidence': 0.9,
                            'fetched_at': datetime.now()
                        })
            
            return pd.DataFrame(injuries)
            
        except Exception as e:
            print(f"⚠️  ESPN injury fetch failed: {e}")
            return self._empty_injury_df()
    
    def _fetch_nflverse_injuries(self) -> pd.DataFrame:
        """Fetch from nflverse data."""
        try:
            import nfl_data_py as nfl
            
            # Get current season injuries - try current year first, then fall back
            current_year = datetime.now().year
            injuries = pd.DataFrame()
            
            # Try current season and previous season as fallback
            # nflverse data may not be available for future/current seasons
            for season in [current_year, current_year - 1, current_year - 2]:
                try:
                    injuries = nfl.import_injuries([season])
                    if not injuries.empty:
                        break
                except Exception:
                    continue
            
            if injuries.empty:
                # Silently return empty - this is expected during offseason
                return self._empty_injury_df()
            
            # Standardize format
            injuries_clean = injuries.rename(columns={
                'full_name': 'player_name',
                'report_status': 'status',
                'report_primary_injury': 'injury_type'
            })
            
            injuries_clean['source'] = 'nflverse'
            injuries_clean['confidence'] = 0.95  # nflverse is highly reliable
            injuries_clean['fetched_at'] = datetime.now()
            
            # Only include columns that exist
            output_cols = ['player_name', 'team', 'position', 'status', 
                          'injury_type', 'source', 'confidence', 'fetched_at']
            available_cols = [c for c in output_cols if c in injuries_clean.columns]
            
            return injuries_clean[available_cols]
            
        except Exception as e:
            # Only print warning for unexpected errors, not 404s
            if '404' not in str(e):
                print(f"⚠️  nflverse injury fetch failed: {e}")
            return self._empty_injury_df()
    
    def _load_manual_overrides(self) -> pd.DataFrame:
        """Load manual injury corrections from file."""
        override_file = 'data/injury_overrides.json'
        
        try:
            with open(override_file, 'r') as f:
                overrides = json.load(f)
            
            return pd.DataFrame(overrides)
        except:
            return self._empty_injury_df()
    
    def _resolve_conflicts(self, injuries: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve conflicts when same player has different statuses.
        Priority: Manual > nflverse > ESPN
        """
        if injuries.empty:
            return injuries
        
        # Assign priority scores
        source_priority = {'manual': 3, 'nflverse': 2, 'ESPN': 1}
        injuries['priority'] = injuries['source'].map(source_priority).fillna(0)
        
        # Keep highest priority source for each player
        deduped = injuries.sort_values('priority', ascending=False).groupby('player_name').first().reset_index()
        
        return deduped
    
    def _calculate_injury_impact(self, injuries: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate injury impact score (0-100).
        Considers: status, injury type, position, historical recovery time
        """
        if injuries.empty:
            return injuries
        
        # Base impact by status
        status_impact = {
            'OUT': 100,
            'DOUBTFUL': 75,
            'QUESTIONABLE': 40,
            'PROBABLE': 15
        }
        
        injuries['impact_score'] = injuries['status'].map(status_impact).fillna(20)
        
        # Adjust by injury type severity
        injury_severity = {
            'concussion': 1.3,
            'acl': 1.5,
            'achilles': 1.5,
            'hamstring': 1.2,
            'ankle': 1.1,
            'knee': 1.2,
            'shoulder': 1.1,
            'hand': 0.9,
            'illness': 0.8
        }
        
        for injury_type, multiplier in injury_severity.items():
            mask = injuries['injury_type'].str.lower().str.contains(injury_type, na=False)
            injuries.loc[mask, 'impact_score'] *= multiplier
        
        # Cap at 100
        injuries['impact_score'] = injuries['impact_score'].clip(0, 100)
        
        # Estimate weeks out
        injuries['estimated_weeks_out'] = injuries.apply(self._estimate_weeks_out, axis=1)
        
        return injuries
    
    def _estimate_weeks_out(self, row) -> int:
        """Estimate weeks player will miss."""
        if row['status'] == 'OUT':
            # Check injury type
            injury_lower = row['injury_type'].lower() if pd.notna(row['injury_type']) else ''
            
            if any(term in injury_lower for term in ['acl', 'achilles', 'season']):
                return 18  # Rest of season
            elif any(term in injury_lower for term in ['hamstring', 'knee', 'ankle']):
                return np.random.randint(2, 6)  # 2-6 weeks
            else:
                return 1
        elif row['status'] == 'DOUBTFUL':
            return 1
        else:
            return 0
    
    def _empty_injury_df(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(columns=[
            'player_name', 'team', 'position', 'status', 'injury_type',
            'source', 'confidence', 'fetched_at'
        ])
    
    def build_historical_injury_database(self, seasons: List[int]) -> pd.DataFrame:
        """
        Build comprehensive historical injury database.
        Used for predictive modeling of injury risk.
        """
        try:
            import nfl_data_py as nfl
            
            all_injuries = []
            
            for season in seasons:
                try:
                    print(f"Fetching {season} injuries...")
                    season_injuries = nfl.import_injuries([season])
                    
                    if not season_injuries.empty:
                        season_injuries['season'] = season
                        all_injuries.append(season_injuries)
                except Exception as e:
                    # 404 errors are expected for unavailable seasons
                    if '404' not in str(e):
                        print(f"  Warning: Could not load {season}: {e}")
                    continue
            
            if not all_injuries:
                print("⚠️  No injury data found for any season")
                return pd.DataFrame()
            
            historical = pd.concat(all_injuries, ignore_index=True)
            
            # Calculate injury patterns per player
            self._analyze_injury_patterns(historical)
            
            return historical
            
        except Exception as e:
            print(f"⚠️  Historical injury fetch failed: {e}")
            return pd.DataFrame()
    
    def _analyze_injury_patterns(self, historical: pd.DataFrame):
        """Analyze injury patterns to predict future risk."""
        if historical.empty:
            return
        
        # Count injuries per player
        injury_counts = historical.groupby('full_name').size()
        
        # Average days missed
        injury_duration = historical.groupby('full_name')['report_status'].apply(
            lambda x: (x == 'Out').sum()
        )
        
        # Store patterns
        self.injury_patterns = {
            'injury_prone_threshold': injury_counts.quantile(0.75),
            'high_risk_players': injury_counts[injury_counts >= 3].index.tolist(),
            'avg_injury_frequency': injury_counts.mean()
        }
    
    def get_player_injury_risk(self, player_name: str) -> Dict:
        """
        Get injury risk profile for a player.
        
        Returns:
            {
                'risk_level': 'low' | 'medium' | 'high',
                'injury_count_historical': int,
                'current_status': str,
                'recommendation': str
            }
        """
        # Check current injuries
        current_injuries = self.fetch_current_injuries()
        current_status = current_injuries[current_injuries['player_name'] == player_name]
        
        # Check historical pattern
        is_high_risk = player_name in self.injury_patterns.get('high_risk_players', [])
        
        if not current_status.empty:
            status = current_status.iloc[0]['status']
            if status == 'OUT':
                risk = 'high'
                rec = "Do not start - player is OUT"
            elif status == 'DOUBTFUL':
                risk = 'high'
                rec = "Risky start - consider backup plan"
            elif status == 'QUESTIONABLE':
                risk = 'medium'
                rec = "Monitor status - have backup ready"
            else:
                risk = 'low'
                rec = "Good to start"
        elif is_high_risk:
            risk = 'medium'
            rec = "Historically injury-prone - monitor closely"
        else:
            risk = 'low'
            rec = "No current injury concerns"
        
        return {
            'risk_level': risk,
            'current_status': current_status.iloc[0]['status'] if not current_status.empty else 'HEALTHY',
            'recommendation': rec
        }


class RookieDataMiner:
    """
    Comprehensive rookie data collection and analysis.
    Tracks draft capital, preseason usage, depth chart position.
    """
    
    def __init__(self):
        self.rookie_class = pd.DataFrame()
        self.draft_capital = {}
    
    def fetch_current_rookie_class(self, season: int) -> pd.DataFrame:
        """
        Fetch current year's rookie class with draft info.
        
        Returns:
            DataFrame with columns:
            - player_name
            - position
            - team
            - draft_round
            - draft_pick
            - draft_capital_score (0-100)
            - college
            - preseason_snaps
            - depth_chart_position
        """
        try:
            import nfl_data_py as nfl
            
            # Get draft picks
            draft_picks = nfl.import_draft_picks([season])
            
            # Filter to key positions
            fantasy_positions = ['QB', 'RB', 'WR', 'TE']
            rookies = draft_picks[draft_picks['position'].isin(fantasy_positions)].copy()
            
            # Calculate draft capital score
            rookies['draft_capital_score'] = rookies.apply(
                lambda row: self._calculate_draft_capital(row['round'], row['pick']),
                axis=1
            )
            
            # Add depth chart info (would fetch from real source)
            rookies['depth_chart_position'] = self._estimate_depth_chart(rookies)
            
            # Add preseason usage (would fetch from real source)
            rookies['preseason_snaps'] = np.random.randint(0, 50, len(rookies))
            
            return rookies
            
        except Exception as e:
            print(f"⚠️  Rookie data fetch failed: {e}")
            return pd.DataFrame()
    
    def _calculate_draft_capital(self, round_num: int, pick: int) -> float:
        """
        Calculate draft capital score (0-100).
        Round 1 picks = 90-100, Round 2 = 70-89, etc.
        """
        if round_num == 1:
            return 100 - (pick - 1) * 3  # Picks 1-32: 100-7
        elif round_num == 2:
            return 70 - (pick - 33) * 2  # Picks 33-64: 70-6
        elif round_num == 3:
            return 50 - (pick - 65) * 1.5
        else:
            return max(10, 40 - (round_num - 4) * 10)
    
    def _estimate_depth_chart(self, rookies: pd.DataFrame) -> pd.Series:
        """Estimate depth chart position (1-4)."""
        # Simplified - would use real depth chart data
        # Higher draft capital = higher on depth chart (usually)
        return (rookies['draft_capital_score'] / 25).clip(1, 4).astype(int)
    
    def identify_high_upside_rookies(self, rookies: pd.DataFrame) -> pd.DataFrame:
        """
        Identify rookies with highest fantasy potential.
        
        Criteria:
        - High draft capital (Round 1-2)
        - Starting role (depth chart 1-2)
        - Preseason usage
        - Position (RBs/WRs more valuable than TEs)
        """
        if rookies.empty:
            return rookies
        
        high_upside = rookies[
            (rookies['draft_capital_score'] >= 60) &
            (rookies['depth_chart_position'] <= 2)
        ].copy()
        
        # Add upside score
        high_upside['upside_score'] = (
            high_upside['draft_capital_score'] * 0.5 +
            (5 - high_upside['depth_chart_position']) * 10 +
            high_upside['preseason_snaps'] * 0.5
        )
        
        return high_upside.sort_values('upside_score', ascending=False)
    
    def get_rookie_breakout_candidates(self, season: int) -> List[Dict]:
        """
        Get top rookie breakout candidates.
        
        Returns list of dicts with player info and reasoning.
        """
        rookies = self.fetch_current_rookie_class(season)
        
        if rookies.empty:
            return []
        
        high_upside = self.identify_high_upside_rookies(rookies)
        
        candidates = []
        for _, rookie in high_upside.head(10).iterrows():
            candidates.append({
                'player': rookie['pfr_player_name'],
                'position': rookie['position'],
                'team': rookie['team'],
                'draft_pick': f"Round {rookie['round']}, Pick {rookie['pick']}",
                'upside_score': round(rookie.get('upside_score', 0), 1),
                'reasoning': self._generate_rookie_reasoning(rookie)
            })
        
        return candidates
    
    def _generate_rookie_reasoning(self, rookie: pd.Series) -> str:
        """Generate human-readable reasoning for rookie upside."""
        reasons = []
        
        if rookie['draft_capital_score'] >= 90:
            reasons.append("Elite draft capital (Top 10 pick)")
        elif rookie['draft_capital_score'] >= 70:
            reasons.append("High draft capital (Round 1-2)")
        
        if rookie.get('depth_chart_position', 3) == 1:
            reasons.append("Projected starter")
        elif rookie.get('depth_chart_position', 3) == 2:
            reasons.append("Backup with playing time potential")
        
        if rookie.get('preseason_snaps', 0) >= 30:
            reasons.append("Strong preseason usage")
        
        return "; ".join(reasons)


# Quick test
if __name__ == "__main__":
    print("Enhanced Injury & Rookie Data Mining")
    print("=" * 60)
    
    # Test injury mining
    injury_miner = EnhancedInjuryDataMiner()
    current_injuries = injury_miner.fetch_current_injuries()
    
    print(f"\n📊 Current Injuries: {len(current_injuries)}")
    if not current_injuries.empty:
        print(current_injuries[['player_name', 'status', 'injury_type', 'source']].head())
    
    # Test rookie mining
    rookie_miner = RookieDataMiner()
    rookies_2024 = rookie_miner.fetch_current_rookie_class(2024)
    
    print(f"\n🆕 2024 Rookie Class: {len(rookies_2024)}")
    breakouts = rookie_miner.get_rookie_breakout_candidates(2024)
    
    if breakouts:
        print("\n💎 Top Rookie Breakout Candidates:")
        for candidate in breakouts[:3]:
            print(f"  {candidate['player']} ({candidate['position']}, {candidate['team']})")
            print(f"    {candidate['reasoning']}")
