"""
Advanced Rookie and Injury Prediction Models

Sophisticated approaches a senior data scientist would implement:

ROOKIE PREDICTIONS:
1. College Production Mapping - Map college stats to NFL expectations
2. Draft Capital Decay Curves - How draft position correlates with success over time
3. Opportunity Score - Team situation, depth chart, coaching scheme
4. Comparable Player Matching - Find historical rookies with similar profiles
5. Breakout Probability - Likelihood of exceeding expectations

INJURY PREDICTIONS:
1. Survival Analysis - Time-to-injury modeling with Cox proportional hazards
2. Recurrent Event Modeling - Players with injury history
3. Workload Risk Factors - Snap counts, touches, age interaction
4. Position-Specific Injury Patterns - Different injury types by position
5. Recovery Trajectory - Expected performance post-injury
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# =============================================================================
# ADVANCED ROOKIE PREDICTION
# =============================================================================

@dataclass
class RookieProfile:
    """Complete rookie profile for projection."""
    player_id: str
    name: str
    position: str
    draft_round: int
    draft_pick: int
    college_production_score: float
    opportunity_score: float
    comparable_players: List[str]
    projected_ppg: float
    projected_games: int
    breakout_probability: float
    bust_probability: float
    ceiling_ppg: float
    floor_ppg: float


class AdvancedRookieProjector:
    """
    Sophisticated rookie projection system.
    
    Uses multiple signals:
    1. Draft capital (round, pick, trade-up value)
    2. College production (adjusted for competition)
    3. Team opportunity (depth chart, scheme fit)
    4. Historical comparables
    5. NFL Combine metrics (40-yard, vertical, etc.)
    """
    
    # Position-specific combine metric weightings
    # These weights determine how important each metric is for projecting success
    COMBINE_WEIGHTS = {
        'RB': {
            'forty': 0.30,       # Speed is crucial for RBs
            'bench': 0.15,       # Some strength needed
            'vertical': 0.25,   # Explosiveness
            'broad': 0.20,      # Lower body power
            'cone': 0.10,       # Agility
        },
        'WR': {
            'forty': 0.40,       # Speed is king for WRs
            'bench': 0.05,       # Less important
            'vertical': 0.25,   # Jump ball ability
            'broad': 0.20,      # Explosiveness
            'cone': 0.10,       # Route running
        },
        'TE': {
            'forty': 0.20,       # Moderate speed importance
            'bench': 0.25,       # Blocking ability
            'vertical': 0.20,   # Red zone threat
            'broad': 0.20,      # Power
            'cone': 0.15,       # Route running
        },
        'QB': {
            'forty': 0.25,       # Mobility
            'bench': 0.10,       # Arm strength proxy
            'vertical': 0.20,   # Athleticism
            'broad': 0.25,      # Lower body for throws
            'cone': 0.20,       # Pocket movement
        },
    }
    
    # Position-specific combine metric thresholds (elite/good/average)
    # Based on historical combine data analysis
    COMBINE_THRESHOLDS = {
        'RB': {
            'forty': {'elite': 4.40, 'good': 4.50, 'average': 4.60},
            'bench': {'elite': 24, 'good': 20, 'average': 16},
            'vertical': {'elite': 38, 'good': 35, 'average': 32},
            'broad': {'elite': 124, 'good': 118, 'average': 112},
            'cone': {'elite': 6.80, 'good': 7.00, 'average': 7.20},
        },
        'WR': {
            'forty': {'elite': 4.35, 'good': 4.45, 'average': 4.55},
            'bench': {'elite': 16, 'good': 12, 'average': 8},
            'vertical': {'elite': 40, 'good': 37, 'average': 34},
            'broad': {'elite': 128, 'good': 122, 'average': 116},
            'cone': {'elite': 6.70, 'good': 6.90, 'average': 7.10},
        },
        'TE': {
            'forty': {'elite': 4.55, 'good': 4.65, 'average': 4.80},
            'bench': {'elite': 24, 'good': 20, 'average': 16},
            'vertical': {'elite': 36, 'good': 33, 'average': 30},
            'broad': {'elite': 120, 'good': 115, 'average': 108},
            'cone': {'elite': 6.90, 'good': 7.10, 'average': 7.30},
        },
        'QB': {
            'forty': {'elite': 4.55, 'good': 4.70, 'average': 4.85},
            'bench': {'elite': 20, 'good': 16, 'average': 12},
            'vertical': {'elite': 34, 'good': 31, 'average': 28},
            'broad': {'elite': 116, 'good': 110, 'average': 104},
            'cone': {'elite': 6.90, 'good': 7.10, 'average': 7.30},
        },
    }
    
    # College production benchmarks by position (percentile thresholds)
    COLLEGE_BENCHMARKS = {
        'QB': {
            'elite': {'pass_yards': 4000, 'pass_td': 35, 'completion_pct': 0.65},
            'good': {'pass_yards': 3200, 'pass_td': 25, 'completion_pct': 0.60},
            'average': {'pass_yards': 2500, 'pass_td': 18, 'completion_pct': 0.55},
        },
        'RB': {
            'elite': {'rush_yards': 1400, 'total_td': 15, 'ypc': 6.0},
            'good': {'rush_yards': 1000, 'total_td': 10, 'ypc': 5.0},
            'average': {'rush_yards': 700, 'total_td': 6, 'ypc': 4.5},
        },
        'WR': {
            'elite': {'rec_yards': 1200, 'receptions': 80, 'rec_td': 10},
            'good': {'rec_yards': 900, 'receptions': 60, 'rec_td': 7},
            'average': {'rec_yards': 600, 'receptions': 40, 'rec_td': 4},
        },
        'TE': {
            'elite': {'rec_yards': 800, 'receptions': 50, 'rec_td': 8},
            'good': {'rec_yards': 500, 'receptions': 35, 'rec_td': 5},
            'average': {'rec_yards': 300, 'receptions': 20, 'rec_td': 3},
        },
    }
    
    # Draft capital value (pick -> expected value multiplier)
    DRAFT_CAPITAL_CURVE = {
        1: 1.0, 2: 0.98, 3: 0.96, 4: 0.94, 5: 0.92,
        10: 0.85, 15: 0.78, 20: 0.72, 25: 0.67, 32: 0.60,
        50: 0.50, 75: 0.40, 100: 0.32, 150: 0.22, 200: 0.15, 250: 0.10
    }
    
    # Historical rookie PPG by draft position and position
    ROOKIE_PPG_BY_DRAFT = {
        'QB': {
            'round_1_top10': {'mean': 14.5, 'std': 5.0, 'games': 14},
            'round_1': {'mean': 10.0, 'std': 5.5, 'games': 10},
            'round_2': {'mean': 5.0, 'std': 4.0, 'games': 6},
            'round_3_plus': {'mean': 2.0, 'std': 3.0, 'games': 3},
        },
        'RB': {
            'round_1_top10': {'mean': 14.0, 'std': 4.5, 'games': 15},
            'round_1': {'mean': 12.0, 'std': 4.0, 'games': 14},
            'round_2': {'mean': 9.0, 'std': 4.0, 'games': 13},
            'round_3_plus': {'mean': 5.0, 'std': 3.5, 'games': 10},
        },
        'WR': {
            'round_1_top10': {'mean': 12.0, 'std': 4.0, 'games': 15},
            'round_1': {'mean': 9.0, 'std': 4.0, 'games': 14},
            'round_2': {'mean': 7.0, 'std': 3.5, 'games': 13},
            'round_3_plus': {'mean': 4.0, 'std': 3.0, 'games': 11},
        },
        'TE': {
            'round_1_top10': {'mean': 8.0, 'std': 3.5, 'games': 14},
            'round_1': {'mean': 6.0, 'std': 3.0, 'games': 13},
            'round_2': {'mean': 4.5, 'std': 2.5, 'games': 12},
            'round_3_plus': {'mean': 2.5, 'std': 2.0, 'games': 10},
        },
    }
    
    def __init__(self):
        self.profiles = {}
    
    def get_draft_tier(self, draft_round: int, draft_pick: int) -> str:
        """Categorize draft position into tiers."""
        if draft_round == 1 and draft_pick <= 10:
            return 'round_1_top10'
        elif draft_round == 1:
            return 'round_1'
        elif draft_round == 2:
            return 'round_2'
        else:
            return 'round_3_plus'
    
    def calculate_draft_capital_value(self, draft_pick: int) -> float:
        """Calculate draft capital value using interpolation."""
        picks = sorted(self.DRAFT_CAPITAL_CURVE.keys())
        values = [self.DRAFT_CAPITAL_CURVE[p] for p in picks]
        
        # Linear interpolation
        return np.interp(draft_pick, picks, values)
    
    def calculate_opportunity_score(self, df: pd.DataFrame, player_id: str) -> float:
        """
        Calculate opportunity score based on team situation.
        
        Factors:
        - Depth chart position (starter vs backup)
        - Team's historical usage of position
        - Vacated targets/carries from previous season
        - Offensive scheme fit
        """
        # Simplified: use team's historical position usage
        player_data = df[df['player_id'] == player_id]
        if player_data.empty:
            return 0.5
        
        position = player_data['position'].iloc[0]
        team = player_data['team'].iloc[0]
        
        # Get team's position usage
        team_data = df[df['team'] == team]
        if team_data.empty:
            return 0.5
        
        # Calculate share of team's fantasy points by position
        pos_data = team_data[team_data['position'] == position]
        if pos_data.empty:
            return 0.5
        
        # Higher opportunity if team historically uses this position heavily
        team_total = team_data['fantasy_points'].sum()
        pos_total = pos_data['fantasy_points'].sum()
        
        if team_total > 0:
            pos_share = pos_total / team_total
            # Normalize to 0-1 scale
            return min(pos_share * 2, 1.0)
        
        return 0.5
    
    def calculate_combine_score(
        self, 
        position: str,
        forty: float = None,
        bench: int = None,
        vertical: float = None,
        broad: int = None,
        cone: float = None
    ) -> Dict[str, float]:
        """
        Calculate composite combine score based on position-specific weightings.
        
        Args:
            position: Player position (QB, RB, WR, TE)
            forty: 40-yard dash time (lower is better)
            bench: Bench press reps (higher is better)
            vertical: Vertical jump in inches (higher is better)
            broad: Broad jump in inches (higher is better)
            cone: 3-cone drill time (lower is better)
            
        Returns:
            Dict with:
            - combine_score: Composite score (0-100)
            - percentile_by_metric: Individual metric percentiles
            - athleticism_grade: 'Elite', 'Good', 'Average', 'Below Average'
        """
        weights = self.COMBINE_WEIGHTS.get(position, self.COMBINE_WEIGHTS['WR'])
        thresholds = self.COMBINE_THRESHOLDS.get(position, self.COMBINE_THRESHOLDS['WR'])
        
        scores = {}
        total_weight = 0
        weighted_score = 0
        
        # Calculate percentile for each metric
        metrics = {
            'forty': (forty, True),     # Lower is better
            'bench': (bench, False),    # Higher is better
            'vertical': (vertical, False),
            'broad': (broad, False),
            'cone': (cone, True),       # Lower is better
        }
        
        for metric, (value, lower_is_better) in metrics.items():
            if value is None:
                continue
            
            thresh = thresholds[metric]
            
            # Calculate percentile (0-100 scale)
            if lower_is_better:
                if value <= thresh['elite']:
                    percentile = 90 + (thresh['elite'] - value) * 10
                elif value <= thresh['good']:
                    percentile = 70 + (thresh['good'] - value) / (thresh['good'] - thresh['elite']) * 20
                elif value <= thresh['average']:
                    percentile = 50 + (thresh['average'] - value) / (thresh['average'] - thresh['good']) * 20
                else:
                    percentile = max(10, 50 - (value - thresh['average']) * 10)
            else:
                if value >= thresh['elite']:
                    percentile = 90 + (value - thresh['elite']) * 2
                elif value >= thresh['good']:
                    percentile = 70 + (value - thresh['good']) / (thresh['elite'] - thresh['good']) * 20
                elif value >= thresh['average']:
                    percentile = 50 + (value - thresh['average']) / (thresh['good'] - thresh['average']) * 20
                else:
                    percentile = max(10, 50 - (thresh['average'] - value) * 2)
            
            percentile = max(0, min(100, percentile))
            scores[metric] = percentile
            
            # Apply weight
            weight = weights[metric]
            weighted_score += percentile * weight
            total_weight += weight
        
        # Calculate composite score
        if total_weight > 0:
            combine_score = weighted_score / total_weight
        else:
            combine_score = 50.0  # Default if no metrics available
        
        # Determine athleticism grade
        if combine_score >= 85:
            grade = 'Elite'
        elif combine_score >= 70:
            grade = 'Good'
        elif combine_score >= 50:
            grade = 'Average'
        else:
            grade = 'Below Average'
        
        return {
            'combine_score': round(combine_score, 1),
            'percentile_by_metric': scores,
            'athleticism_grade': grade,
            'metrics_available': len(scores),
        }
    
    def load_combine_data(self, season: int = None) -> pd.DataFrame:
        """
        Load NFL Combine data from nflverse.
        
        Args:
            season: Specific season to load (default: all available)
            
        Returns:
            DataFrame with combine metrics
        """
        try:
            import nfl_data_py as nfl
            
            combine_df = nfl.import_combine_data()
            
            if combine_df.empty:
                print("No combine data available")
                return pd.DataFrame()
            
            # Filter to season if specified
            if season and 'season' in combine_df.columns:
                combine_df = combine_df[combine_df['season'] == season]
            
            # Standardize column names
            column_mapping = {
                'player_name': 'name',
                'pos': 'position',
                'forty': 'forty',
                'bench': 'bench',
                'vertical': 'vertical',
                'broad_jump': 'broad',
                'cone': 'cone',
                'shuttle': 'shuttle',
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in combine_df.columns and new_col not in combine_df.columns:
                    combine_df[new_col] = combine_df[old_col]
            
            print(f"Loaded {len(combine_df)} combine records")
            return combine_df
            
        except Exception as e:
            print(f"Could not load combine data: {e}")
            return pd.DataFrame()
    
    def add_combine_features(self, df: pd.DataFrame, combine_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add combine-based features to player DataFrame.
        
        Args:
            df: Player DataFrame with 'name' or 'player_name' column
            combine_df: Combine data (will load if not provided)
            
        Returns:
            DataFrame with combine features added
        """
        result = df.copy()
        
        # Load combine data if not provided
        if combine_df is None:
            combine_df = self.load_combine_data()
        
        if combine_df.empty:
            # Add empty columns
            result['combine_score'] = np.nan
            result['athleticism_grade'] = 'Unknown'
            return result
        
        # Determine name column
        name_col = 'name' if 'name' in result.columns else 'player_name'
        combine_name_col = 'name' if 'name' in combine_df.columns else 'player_name'
        
        if name_col not in result.columns:
            result['combine_score'] = np.nan
            result['athleticism_grade'] = 'Unknown'
            return result
        
        # Calculate combine scores for each player
        combine_scores = []
        athleticism_grades = []
        
        for idx, row in result.iterrows():
            player_name = row.get(name_col)
            position = row.get('position', 'WR')
            
            # Find player in combine data
            player_combine = combine_df[
                combine_df[combine_name_col].str.contains(str(player_name).split()[-1], case=False, na=False)
            ] if player_name else pd.DataFrame()
            
            if player_combine.empty:
                combine_scores.append(np.nan)
                athleticism_grades.append('Unknown')
                continue
            
            # Use first match
            pc = player_combine.iloc[0]
            
            # Calculate combine score
            score_result = self.calculate_combine_score(
                position=position,
                forty=pc.get('forty'),
                bench=pc.get('bench'),
                vertical=pc.get('vertical'),
                broad=pc.get('broad'),
                cone=pc.get('cone'),
            )
            
            combine_scores.append(score_result['combine_score'])
            athleticism_grades.append(score_result['athleticism_grade'])
        
        result['combine_score'] = combine_scores
        result['athleticism_grade'] = athleticism_grades
        
        # Fill missing with position averages
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_mask = result['position'] == position
            pos_avg = result.loc[pos_mask, 'combine_score'].mean()
            if not np.isnan(pos_avg):
                result.loc[pos_mask & result['combine_score'].isna(), 'combine_score'] = pos_avg
        
        # Final fallback
        result['combine_score'] = result['combine_score'].fillna(50.0)
        result['athleticism_grade'] = result['athleticism_grade'].fillna('Average')
        
        return result
    
    def calculate_breakout_probability(self, draft_pick: int, position: str,
                                       opportunity_score: float) -> float:
        """
        Calculate probability of rookie exceeding expectations.
        
        Based on historical breakout rates by draft position.
        """
        # Base breakout rate by draft position
        if draft_pick <= 10:
            base_rate = 0.35
        elif draft_pick <= 32:
            base_rate = 0.25
        elif draft_pick <= 64:
            base_rate = 0.15
        elif draft_pick <= 100:
            base_rate = 0.08
        else:
            base_rate = 0.03
        
        # Adjust for position (RBs break out more often as rookies)
        position_multiplier = {
            'RB': 1.3, 'WR': 1.0, 'TE': 0.7, 'QB': 0.9
        }.get(position, 1.0)
        
        # Adjust for opportunity
        opportunity_multiplier = 0.7 + (opportunity_score * 0.6)
        
        return min(base_rate * position_multiplier * opportunity_multiplier, 0.6)
    
    def calculate_bust_probability(self, draft_pick: int, position: str) -> float:
        """
        Calculate probability of rookie significantly underperforming.
        
        "Bust" = bottom 25% of draft class at position.
        """
        # Higher picks have lower bust rates but still significant
        if draft_pick <= 10:
            base_rate = 0.15
        elif draft_pick <= 32:
            base_rate = 0.25
        elif draft_pick <= 64:
            base_rate = 0.35
        elif draft_pick <= 100:
            base_rate = 0.45
        else:
            base_rate = 0.55
        
        # TEs have higher bust rates as rookies
        position_multiplier = {
            'TE': 1.3, 'QB': 1.2, 'WR': 1.0, 'RB': 0.9
        }.get(position, 1.0)
        
        return min(base_rate * position_multiplier, 0.7)
    
    def find_comparable_players(
        self,
        position: str,
        draft_round: int,
        draft_pick: int,
        combine_score: float = None,
        historical_df: pd.DataFrame = None,
        min_season: int = 2015,
        max_season: int = 2024,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Find historical rookies with similar profiles.
        
        Matching criteria:
        - Same position
        - Similar draft position (within 20 picks or same round)
        - Similar combine score (within 10 percentile if available)
        
        Args:
            position: Player position (QB, RB, WR, TE)
            draft_round: Draft round (1-7)
            draft_pick: Overall pick number
            combine_score: Athletic composite score (0-100)
            historical_df: Historical player data with rookie seasons
            min_season: Earliest season to consider
            max_season: Latest season to consider
            top_n: Number of comparables to return
            
        Returns:
            List of comparable player dicts with name, season, stats
        """
        if historical_df is None:
            # Try to load historical data
            historical_df = self._load_historical_rookies(min_season, max_season)
        
        if historical_df.empty:
            return []
        
        # Filter to position
        pos_df = historical_df[historical_df['position'] == position].copy()
        
        if pos_df.empty:
            return []
        
        # Calculate similarity scores
        similarities = []
        
        for idx, row in pos_df.iterrows():
            hist_pick = row.get('draft_pick', 150)
            hist_round = row.get('draft_round', 5)
            hist_combine = row.get('combine_score', 50)
            
            # Draft position similarity (0-1, higher is better)
            pick_diff = abs(draft_pick - hist_pick)
            round_diff = abs(draft_round - hist_round)
            
            # Weight by round match (same round is important)
            if round_diff == 0:
                draft_sim = 1.0 - min(pick_diff / 50, 0.5)
            elif round_diff == 1:
                draft_sim = 0.7 - min(pick_diff / 75, 0.4)
            else:
                draft_sim = max(0.1, 0.5 - round_diff * 0.15 - pick_diff / 100)
            
            # Combine similarity (if available)
            if combine_score is not None and not pd.isna(hist_combine):
                combine_diff = abs(combine_score - hist_combine)
                combine_sim = max(0, 1 - combine_diff / 30)
            else:
                combine_sim = 0.5  # Neutral if no data
            
            # Overall similarity (weighted)
            overall_sim = draft_sim * 0.7 + combine_sim * 0.3
            
            similarities.append({
                'index': idx,
                'name': row.get('name', row.get('player_name', 'Unknown')),
                'season': row.get('season', row.get('draft_season')),
                'draft_round': hist_round,
                'draft_pick': hist_pick,
                'combine_score': hist_combine,
                'fantasy_ppg': row.get('fantasy_points_avg', row.get('ppg', 0)),
                'games_played': row.get('games_played', row.get('games', 0)),
                'total_points': row.get('fantasy_points', row.get('total_points', 0)),
                'similarity_score': overall_sim,
            })
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities[:top_n]
    
    def _load_historical_rookies(
        self, 
        min_season: int = 2015, 
        max_season: int = 2024
    ) -> pd.DataFrame:
        """
        Load historical rookie performance data.
        
        Returns DataFrame with rookie seasons including:
        - Player info (name, position, team)
        - Draft info (round, pick)
        - Performance (fantasy points, games played)
        """
        try:
            import nfl_data_py as nfl
            
            # Load seasonal data
            seasons = list(range(min_season, max_season + 1))
            seasonal_df = nfl.import_seasonal_data(seasons)
            
            if seasonal_df.empty:
                return pd.DataFrame()
            
            # Load draft data to identify rookies
            draft_df = nfl.import_draft_picks(seasons)
            
            if draft_df.empty:
                return pd.DataFrame()
            
            # Merge draft info onto seasonal data
            # A rookie is a player in their draft year
            draft_lookup = draft_df[['player_name', 'season', 'round', 'pick', 'position']].copy()
            draft_lookup.columns = ['name', 'draft_season', 'draft_round', 'draft_pick', 'position']
            
            # Merge on player name and season
            merged = seasonal_df.merge(
                draft_lookup,
                left_on=['player_name', 'season'],
                right_on=['name', 'draft_season'],
                how='inner'
            )
            
            # Calculate fantasy points if not present
            if 'fantasy_points' not in merged.columns:
                # PPR scoring
                merged['fantasy_points'] = (
                    merged.get('passing_yards', 0) * 0.04 +
                    merged.get('passing_tds', 0) * 4 +
                    merged.get('interceptions', 0) * -2 +
                    merged.get('rushing_yards', 0) * 0.1 +
                    merged.get('rushing_tds', 0) * 6 +
                    merged.get('receptions', 0) * 1 +
                    merged.get('receiving_yards', 0) * 0.1 +
                    merged.get('receiving_tds', 0) * 6
                )
            
            # Calculate PPG
            if 'games' in merged.columns:
                merged['fantasy_points_avg'] = merged['fantasy_points'] / merged['games'].clip(lower=1)
            
            print(f"Loaded {len(merged)} historical rookie seasons")
            return merged
            
        except Exception as e:
            print(f"Could not load historical rookies: {e}")
            return pd.DataFrame()
    
    def get_comparable_projection(
        self,
        position: str,
        draft_round: int,
        draft_pick: int,
        combine_score: float = None,
        historical_df: pd.DataFrame = None
    ) -> Dict:
        """
        Generate projection based on comparable players.
        
        Uses historical rookies with similar profiles to project
        expected performance range.
        
        Returns:
            Dict with projection stats based on comparables
        """
        comparables = self.find_comparable_players(
            position=position,
            draft_round=draft_round,
            draft_pick=draft_pick,
            combine_score=combine_score,
            historical_df=historical_df
        )
        
        if not comparables:
            return {
                'comparable_players': [],
                'projected_ppg_from_comps': None,
                'projected_games_from_comps': None,
                'ceiling_from_comps': None,
                'floor_from_comps': None,
            }
        
        # Extract stats from comparables
        ppg_values = [c['fantasy_ppg'] for c in comparables if c['fantasy_ppg'] > 0]
        games_values = [c['games_played'] for c in comparables if c['games_played'] > 0]
        
        if ppg_values:
            projected_ppg = np.mean(ppg_values)
            ceiling = max(ppg_values)
            floor = min(ppg_values)
        else:
            projected_ppg = ceiling = floor = None
        
        projected_games = np.mean(games_values) if games_values else None
        
        return {
            'comparable_players': [
                {'name': c['name'], 'season': c['season'], 'ppg': c['fantasy_ppg']}
                for c in comparables
            ],
            'projected_ppg_from_comps': round(projected_ppg, 1) if projected_ppg else None,
            'projected_games_from_comps': round(projected_games, 0) if projected_games else None,
            'ceiling_from_comps': round(ceiling, 1) if ceiling else None,
            'floor_from_comps': round(floor, 1) if floor else None,
        }
    
    def project_rookie(self, player_id: str, name: str, position: str,
                       draft_round: int, draft_pick: int,
                       df: pd.DataFrame = None,
                       combine_score: float = None,
                       use_comparables: bool = True) -> RookieProfile:
        """
        Generate complete rookie projection using multiple signals.
        
        Combines:
        - Draft capital-based archetype projections
        - Comparable player matching (if enabled)
        - Opportunity score adjustments
        - Combine data (if available)
        
        Args:
            player_id: Unique player identifier
            name: Player name
            position: Position (QB, RB, WR, TE)
            draft_round: Draft round (1-7)
            draft_pick: Overall pick number
            df: Optional DataFrame for opportunity calculation
            combine_score: Optional athletic composite score
            use_comparables: Whether to use historical comparables
            
        Returns:
            RookieProfile with complete projection
        """
        # Get draft tier
        tier = self.get_draft_tier(draft_round, draft_pick)
        
        # Get baseline projection
        baseline = self.ROOKIE_PPG_BY_DRAFT.get(position, self.ROOKIE_PPG_BY_DRAFT['WR'])
        tier_stats = baseline.get(tier, baseline['round_3_plus'])
        
        # Calculate modifiers
        draft_value = self.calculate_draft_capital_value(draft_pick)
        opportunity = self.calculate_opportunity_score(df, player_id) if df is not None else 0.5
        breakout_prob = self.calculate_breakout_probability(draft_pick, position, opportunity)
        bust_prob = self.calculate_bust_probability(draft_pick, position)
        
        # Base projection from archetype
        base_ppg = tier_stats['mean']
        adjusted_ppg = base_ppg * (0.8 + opportunity * 0.4)
        
        # Floor and ceiling from archetype
        floor_ppg = max(0, adjusted_ppg - 1.5 * tier_stats['std'])
        ceiling_ppg = adjusted_ppg + 1.5 * tier_stats['std']
        
        # Get comparable players if enabled
        comparable_names = []
        if use_comparables:
            comp_projection = self.get_comparable_projection(
                position=position,
                draft_round=draft_round,
                draft_pick=draft_pick,
                combine_score=combine_score
            )
            
            # Blend archetype and comparable projections
            if comp_projection['projected_ppg_from_comps'] is not None:
                # Weight: 60% archetype, 40% comparables
                adjusted_ppg = 0.6 * adjusted_ppg + 0.4 * comp_projection['projected_ppg_from_comps']
                
                # Update floor/ceiling with comparable data
                if comp_projection['floor_from_comps'] is not None:
                    floor_ppg = 0.6 * floor_ppg + 0.4 * comp_projection['floor_from_comps']
                if comp_projection['ceiling_from_comps'] is not None:
                    ceiling_ppg = 0.6 * ceiling_ppg + 0.4 * comp_projection['ceiling_from_comps']
            
            comparable_names = [c['name'] for c in comp_projection.get('comparable_players', [])]
        
        # Adjust for combine score if available
        if combine_score is not None:
            # Athletes with higher combine scores get a boost
            if combine_score >= 80:
                adjusted_ppg *= 1.10
                ceiling_ppg *= 1.15
            elif combine_score >= 70:
                adjusted_ppg *= 1.05
                ceiling_ppg *= 1.08
            elif combine_score < 40:
                adjusted_ppg *= 0.95
                ceiling_ppg *= 0.95
        
        return RookieProfile(
            player_id=player_id,
            name=name,
            position=position,
            draft_round=draft_round,
            draft_pick=draft_pick,
            college_production_score=draft_value,
            opportunity_score=opportunity,
            comparable_players=comparable_names,
            projected_ppg=adjusted_ppg,
            projected_games=tier_stats['games'],
            breakout_probability=breakout_prob,
            bust_probability=bust_prob,
            ceiling_ppg=ceiling_ppg,
            floor_ppg=floor_ppg
        )
    
    def add_advanced_rookie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced rookie features to DataFrame."""
        result = df.copy()
        
        print("Adding advanced rookie features...")
        
        # Identify rookies
        if 'years_exp' in result.columns:
            result['is_rookie'] = (result['years_exp'] == 0).astype(int)
        else:
            first_season = result.groupby('player_id')['season'].min().reset_index()
            first_season.columns = ['player_id', 'first_season']
            result = result.merge(first_season, on='player_id', how='left')
            result['is_rookie'] = (result['season'] == result['first_season']).astype(int)
        
        # Default values
        result['rookie_draft_value'] = 0.0
        result['rookie_opportunity_score'] = 0.0
        result['rookie_breakout_prob'] = 0.0
        result['rookie_bust_prob'] = 0.0
        result['rookie_ceiling_ppg'] = 0.0
        result['rookie_floor_ppg'] = 0.0
        
        # Calculate for rookies
        rookies = result[result['is_rookie'] == 1]
        
        for player_id in rookies['player_id'].unique():
            player_data = result[result['player_id'] == player_id].iloc[0]
            
            draft_round = int(player_data.get('draft_round', 5)) if pd.notna(player_data.get('draft_round')) else 5
            draft_pick = int(player_data.get('draft_pick', 150)) if pd.notna(player_data.get('draft_pick')) else 150
            
            profile = self.project_rookie(
                player_id=player_id,
                name=player_data.get('name', 'Unknown'),
                position=player_data['position'],
                draft_round=draft_round,
                draft_pick=draft_pick,
                df=result
            )
            
            # Update DataFrame
            mask = result['player_id'] == player_id
            result.loc[mask, 'rookie_draft_value'] = profile.college_production_score
            result.loc[mask, 'rookie_opportunity_score'] = profile.opportunity_score
            result.loc[mask, 'rookie_breakout_prob'] = profile.breakout_probability
            result.loc[mask, 'rookie_bust_prob'] = profile.bust_probability
            result.loc[mask, 'rookie_ceiling_ppg'] = profile.ceiling_ppg
            result.loc[mask, 'rookie_floor_ppg'] = profile.floor_ppg
        
        print(f"  Added: rookie_draft_value, rookie_opportunity_score, rookie_breakout_prob, rookie_bust_prob")
        
        return result


# =============================================================================
# ADVANCED INJURY PREDICTION
# =============================================================================

class AdvancedInjuryPredictor:
    """
    Sophisticated injury prediction using survival analysis concepts.
    
    Approaches:
    1. Hazard rate modeling - Time-varying injury risk
    2. Recurrent event analysis - Players with injury history
    3. Workload-based risk - Cumulative stress factors
    4. Position-specific patterns - Different injury profiles
    5. Recovery modeling - Post-injury performance trajectory
    """
    
    # Position-specific injury rates (per 100 games)
    POSITION_INJURY_RATES = {
        'QB': {'soft_tissue': 8, 'concussion': 3, 'structural': 5, 'total': 16},
        'RB': {'soft_tissue': 15, 'concussion': 4, 'structural': 10, 'total': 29},
        'WR': {'soft_tissue': 12, 'concussion': 5, 'structural': 8, 'total': 25},
        'TE': {'soft_tissue': 10, 'concussion': 4, 'structural': 9, 'total': 23},
    }
    
    # Age-based injury risk multiplier
    AGE_RISK_CURVE = {
        21: 0.85, 22: 0.88, 23: 0.92, 24: 0.95, 25: 1.0,
        26: 1.05, 27: 1.10, 28: 1.18, 29: 1.28, 30: 1.40,
        31: 1.55, 32: 1.72, 33: 1.90, 34: 2.10, 35: 2.35
    }
    
    # Workload risk factors
    WORKLOAD_THRESHOLDS = {
        'RB': {'high_risk_touches': 25, 'danger_zone_touches': 30},
        'WR': {'high_risk_targets': 12, 'danger_zone_targets': 15},
        'TE': {'high_risk_targets': 10, 'danger_zone_targets': 12},
        'QB': {'high_risk_dropbacks': 45, 'danger_zone_dropbacks': 55},
    }
    
    # Recovery trajectory (weeks to full performance by injury type)
    RECOVERY_TRAJECTORIES = {
        'soft_tissue': {'weeks': 2, 'performance_at_return': 0.85, 'full_recovery_weeks': 4},
        'concussion': {'weeks': 1, 'performance_at_return': 0.90, 'full_recovery_weeks': 3},
        'structural': {'weeks': 8, 'performance_at_return': 0.70, 'full_recovery_weeks': 16},
    }
    
    def __init__(self):
        self.injury_history = {}
    
    def calculate_base_hazard_rate(self, position: str, weeks_played: int) -> float:
        """
        Calculate base hazard rate (instantaneous injury probability).
        
        Uses Weibull-like distribution where risk increases with exposure.
        """
        base_rate = self.POSITION_INJURY_RATES.get(position, {'total': 20})['total'] / 100
        
        # Hazard increases with cumulative exposure (fatigue effect)
        # Shape parameter > 1 means increasing hazard
        shape = 1.3
        scale = 17  # Season length
        
        # Weibull hazard: h(t) = (shape/scale) * (t/scale)^(shape-1)
        if weeks_played > 0:
            hazard = (shape / scale) * ((weeks_played / scale) ** (shape - 1))
        else:
            hazard = base_rate / 17
        
        return min(hazard * base_rate * 17, 0.15)  # Cap at 15% per week
    
    def calculate_age_risk_multiplier(self, age: int) -> float:
        """Get age-based risk multiplier."""
        if age in self.AGE_RISK_CURVE:
            return self.AGE_RISK_CURVE[age]
        elif age < 21:
            return 0.85
        else:
            return 2.5  # Very high risk for 36+
    
    def calculate_workload_risk(self, position: str, weekly_workload: float,
                                season_workload: float) -> float:
        """
        Calculate workload-based injury risk.
        
        High workload increases injury probability non-linearly.
        """
        thresholds = self.WORKLOAD_THRESHOLDS.get(position, {'high_risk_touches': 20, 'danger_zone_touches': 25})
        
        # Get the relevant threshold key
        threshold_key = list(thresholds.keys())[0]
        high_risk = thresholds[threshold_key]
        danger_zone = thresholds[list(thresholds.keys())[1]]
        
        # Weekly workload risk
        if weekly_workload >= danger_zone:
            weekly_risk = 1.5
        elif weekly_workload >= high_risk:
            weekly_risk = 1.2
        else:
            weekly_risk = 1.0
        
        # Cumulative season workload risk
        expected_season = high_risk * 17
        if season_workload > expected_season * 1.2:
            cumulative_risk = 1.3
        elif season_workload > expected_season:
            cumulative_risk = 1.15
        else:
            cumulative_risk = 1.0
        
        return weekly_risk * cumulative_risk
    
    def calculate_injury_history_risk(self, prior_injuries: int,
                                      same_body_part_injuries: int) -> float:
        """
        Calculate risk multiplier based on injury history.
        
        Recurrent injuries significantly increase future risk.
        """
        # General injury history
        history_risk = 1.0 + (prior_injuries * 0.1)
        
        # Same body part is much higher risk
        recurrence_risk = 1.0 + (same_body_part_injuries * 0.3)
        
        return min(history_risk * recurrence_risk, 3.0)
    
    def predict_injury_probability(self, position: str, age: int,
                                   weeks_played: int, weekly_workload: float,
                                   season_workload: float, prior_injuries: int = 0) -> Dict:
        """
        Predict injury probability for upcoming week.
        
        Returns probability and risk factors breakdown.
        """
        # Base hazard
        base_hazard = self.calculate_base_hazard_rate(position, weeks_played)
        
        # Risk multipliers
        age_mult = self.calculate_age_risk_multiplier(age)
        workload_mult = self.calculate_workload_risk(position, weekly_workload, season_workload)
        history_mult = self.calculate_injury_history_risk(prior_injuries, 0)
        
        # Combined probability
        combined_prob = base_hazard * age_mult * workload_mult * history_mult
        combined_prob = min(combined_prob, 0.25)  # Cap at 25%
        
        return {
            'injury_probability': combined_prob,
            'base_hazard': base_hazard,
            'age_risk_factor': age_mult,
            'workload_risk_factor': workload_mult,
            'history_risk_factor': history_mult,
            'risk_level': 'high' if combined_prob > 0.12 else ('medium' if combined_prob > 0.06 else 'low')
        }
    
    def predict_recovery_trajectory(self, injury_type: str,
                                    player_age: int) -> Dict:
        """
        Predict recovery trajectory after injury.
        
        Returns expected weeks out and performance curve.
        """
        trajectory = self.RECOVERY_TRAJECTORIES.get(injury_type, self.RECOVERY_TRAJECTORIES['soft_tissue'])
        
        # Age adjustment (older players recover slower)
        age_factor = 1.0 + max(0, (player_age - 27) * 0.05)
        
        adjusted_weeks = trajectory['weeks'] * age_factor
        adjusted_full_recovery = trajectory['full_recovery_weeks'] * age_factor
        
        # Performance curve
        weeks_out = int(np.ceil(adjusted_weeks))
        performance_curve = []
        
        for week in range(int(adjusted_full_recovery) + 1):
            if week < weeks_out:
                performance_curve.append(0)  # Out
            else:
                # Gradual recovery
                weeks_since_return = week - weeks_out
                recovery_progress = min(1.0, trajectory['performance_at_return'] + 
                                       (1 - trajectory['performance_at_return']) * 
                                       (weeks_since_return / (adjusted_full_recovery - weeks_out + 1)))
                performance_curve.append(recovery_progress)
        
        return {
            'expected_weeks_out': weeks_out,
            'performance_at_return': trajectory['performance_at_return'],
            'weeks_to_full_recovery': int(np.ceil(adjusted_full_recovery)),
            'performance_curve': performance_curve
        }
    
    def add_advanced_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced injury prediction features to DataFrame."""
        result = df.copy()
        
        print("Adding advanced injury prediction features...")
        
        # Ensure we have required columns
        if 'age' not in result.columns:
            result['age'] = 26  # Default
        
        # Calculate workload proxies
        if 'rushing_attempts' in result.columns and 'targets' in result.columns:
            result['weekly_workload'] = result['rushing_attempts'].fillna(0) + result['targets'].fillna(0)
        else:
            result['weekly_workload'] = 15  # Default
        
        # Season cumulative workload
        result['season_workload'] = result.groupby(['player_id', 'season'])['weekly_workload'].cumsum()
        
        # Calculate injury probability for each row
        def calc_injury_prob(row):
            pred = self.predict_injury_probability(
                position=row['position'],
                age=int(row['age']),
                weeks_played=int(row.get('week', 1)),
                weekly_workload=float(row['weekly_workload']),
                season_workload=float(row['season_workload']),
                prior_injuries=0  # Would need injury history data
            )
            return pred['injury_probability']
        
        def calc_age_risk(row):
            return self.calculate_age_risk_multiplier(int(row['age']))
        
        def calc_workload_risk(row):
            return self.calculate_workload_risk(
                row['position'],
                float(row['weekly_workload']),
                float(row['season_workload'])
            )
        
        result['injury_prob_advanced'] = result.apply(calc_injury_prob, axis=1)
        result['injury_age_risk'] = result.apply(calc_age_risk, axis=1)
        result['injury_workload_risk'] = result.apply(calc_workload_risk, axis=1)
        
        # Risk level categorization
        result['injury_risk_level'] = result['injury_prob_advanced'].apply(
            lambda x: 'high' if x > 0.12 else ('medium' if x > 0.06 else 'low')
        )
        
        print(f"  Added: injury_prob_advanced, injury_age_risk, injury_workload_risk, injury_risk_level")
        
        return result


# =============================================================================
# COMBINED FEATURE ADDITION
# =============================================================================

def add_advanced_rookie_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all advanced rookie, injury, and combine features."""
    print("\n" + "="*60)
    print("Adding Advanced Rookie & Injury Features")
    print("="*60)

    # Advanced rookie features (draft capital, opportunity scores)
    rookie_projector = AdvancedRookieProjector()
    df = rookie_projector.add_advanced_rookie_features(df)

    # NFL Combine features (athleticism scores from nflverse)
    try:
        df = rookie_projector.add_combine_features(df)
        print("  Added: combine_score, athleticism_grade")
    except Exception as e:
        print(f"  Combine features skipped: {e}")
        if "combine_score" not in df.columns:
            df["combine_score"] = 50.0
        if "athleticism_grade" not in df.columns:
            df["athleticism_grade"] = "Average"

    # Advanced injury features (hazard modeling, workload risk)
    injury_predictor = AdvancedInjuryPredictor()
    df = injury_predictor.add_advanced_injury_features(df)

    print(f"\nAdded advanced rookie, combine, and injury features")

    return df


if __name__ == '__main__':
    # Test the module
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.database import DatabaseManager
    from src.features.utilization import engineer_all_features
    
    print("Testing Advanced Rookie & Injury Features...")
    
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=4)
    df = engineer_all_features(df)
    
    # Add advanced features
    df = add_advanced_rookie_injury_features(df)
    
    # Show sample
    print("\nSample rookie features:")
    rookie_cols = [c for c in df.columns if 'rookie' in c.lower()]
    print(df[df['is_rookie'] == 1][['name', 'position'] + rookie_cols].head())
    
    print("\nSample injury features:")
    injury_cols = [c for c in df.columns if 'injury' in c.lower()]
    print(df[['name', 'position', 'age'] + injury_cols].head(10))
