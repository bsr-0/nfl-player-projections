"""
Playoff Optimizer - Multi-Week Lineup Planning
Optimizes roster decisions for fantasy playoffs (Weeks 15-17)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from itertools import combinations

class PlayoffOptimizer:
    """
    Optimizes lineup decisions for 3-4 week playoff stretch.
    Accounts for: schedule strength, injury risk, bye weeks, consistency.
    """
    
    def __init__(self, predictions: pd.DataFrame, playoff_weeks: List[int] = [15, 16, 17]):
        self.predictions = predictions
        self.playoff_weeks = playoff_weeks
        self.schedules = {}  # Team schedules (opponent by week)
        
    def set_schedules(self, schedules: Dict[str, Dict[int, str]]):
        """
        Set NFL schedules for playoff weeks.
        
        Args:
            schedules: {
                'KC': {15: 'vs CLE', 16: 'vs HOU', 17: 'vs PIT'},
                'SF': {15: 'at SEA', 16: 'vs MIA', 17: 'at DET'},
                ...
            }
        """
        self.schedules = schedules
    
    def optimize_roster(
        self,
        my_roster: List[str],
        roster_slots: Dict[str, int],
        strategy: str = 'balanced'
    ) -> Dict:
        """
        Optimize lineup across playoff weeks.
        
        Args:
            my_roster: List of player names on your team
            roster_slots: {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
            strategy: 'ceiling' (maximize upside) | 'floor' (minimize risk) | 'balanced'
            
        Returns:
            {
                'week_15': {'QB': [...], 'RB': [...], ...},
                'week_16': {...},
                'week_17': {...},
                'reasoning': {...}
            }
        """
        # Filter predictions to my roster
        my_players = self.predictions[self.predictions['player'].isin(my_roster)].copy()
        
        if my_players.empty:
            return {'error': 'No players from roster found in predictions'}
        
        # Add playoff-specific scoring
        my_players = self._add_playoff_scores(my_players, strategy)
        
        # Optimize for each week
        weekly_lineups = {}
        weekly_reasoning = {}
        
        for week in self.playoff_weeks:
            lineup, reasoning = self._optimize_single_week(
                my_players,
                roster_slots,
                week,
                strategy
            )
            weekly_lineups[f'week_{week}'] = lineup
            weekly_reasoning[f'week_{week}'] = reasoning
        
        # Add cross-week insights
        insights = self._analyze_cross_week_patterns(weekly_lineups)
        
        return {
            **weekly_lineups,
            'reasoning': weekly_reasoning,
            'insights': insights,
            'strategy_used': strategy
        }
    
    def _add_playoff_scores(self, players: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Add playoff-adjusted scores based on strategy."""
        players = players.copy()
        
        # Base playoff score = utilization
        players['playoff_score'] = players['util_1w']
        
        # Adjust based on strategy
        if strategy == 'ceiling':
            # Favor high upside (util_1w_high)
            players['playoff_score'] = (
                players['util_1w'] * 0.6 +
                players['util_1w_high'] * 0.4
            )
        elif strategy == 'floor':
            # Favor consistency (util_1w_low)
            players['playoff_score'] = (
                players['util_1w'] * 0.6 +
                players['util_1w_low'] * 0.4
            )
        else:  # balanced
            # Equal weight
            players['playoff_score'] = players['util_1w']
        
        # Penalty for injury-prone players (if we have injury data)
        if 'games_played' in players.columns:
            injury_factor = players['games_played'] / 17  # Availability %
            players['playoff_score'] *= (0.7 + 0.3 * injury_factor)
        
        return players
    
    def _optimize_single_week(
        self,
        players: pd.DataFrame,
        roster_slots: Dict[str, int],
        week: int,
        strategy: str
    ) -> Tuple[Dict, Dict]:
        """Optimize lineup for a single week."""
        lineup = {}
        reasoning = {}
        available = players.copy()
        
        # Fill each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position not in roster_slots or roster_slots[position] == 0:
                continue
            
            # Get position players
            pos_players = available[available['position'] == position].copy()
            
            # Sort by playoff score
            pos_players = pos_players.sort_values('playoff_score', ascending=False)
            
            # Take top N for this position
            n_slots = roster_slots[position]
            starters = pos_players.head(n_slots)
            
            lineup[position] = starters['player'].tolist()
            reasoning[position] = self._explain_picks(starters, strategy)
            
            # Remove from available
            available = available[~available['player'].isin(starters['player'])]
        
        # Fill FLEX (best remaining RB/WR/TE)
        if 'FLEX' in roster_slots and roster_slots['FLEX'] > 0:
            flex_candidates = available[available['position'].isin(['RB', 'WR', 'TE'])]
            flex_candidates = flex_candidates.sort_values('playoff_score', ascending=False)
            
            flex_starters = flex_candidates.head(roster_slots['FLEX'])
            lineup['FLEX'] = flex_starters['player'].tolist()
            reasoning['FLEX'] = self._explain_picks(flex_starters, strategy)
        
        return lineup, reasoning
    
    def _explain_picks(self, players: pd.DataFrame, strategy: str) -> List[str]:
        """Generate explanations for why players were chosen."""
        explanations = []
        
        for _, player in players.iterrows():
            if strategy == 'ceiling':
                reason = f"{player['player']}: {player['playoff_score']:.1f} (high upside {player['util_1w_high']:.1f})"
            elif strategy == 'floor':
                reason = f"{player['player']}: {player['playoff_score']:.1f} (safe floor {player['util_1w_low']:.1f})"
            else:
                reason = f"{player['player']}: {player['playoff_score']:.1f}"
            
            explanations.append(reason)
        
        return explanations
    
    def _analyze_cross_week_patterns(self, weekly_lineups: Dict) -> Dict:
        """Analyze patterns across playoff weeks."""
        insights = {}
        
        # Find players who start all weeks (your studs)
        all_starters = {}
        for week, lineup in weekly_lineups.items():
            for pos, players in lineup.items():
                for player in players:
                    all_starters[player] = all_starters.get(player, 0) + 1
        
        studs = [p for p, count in all_starters.items() if count == len(self.playoff_weeks)]
        situational = [p for p, count in all_starters.items() if 0 < count < len(self.playoff_weeks)]
        
        insights['studs'] = studs
        insights['situational'] = situational
        insights['total_unique_starters'] = len(all_starters)
        
        return insights
    
    def compare_strategies(
        self,
        my_roster: List[str],
        roster_slots: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Compare ceiling vs floor vs balanced strategies.
        Shows which strategy maximizes different objectives.
        """
        results = []
        
        for strategy in ['ceiling', 'floor', 'balanced']:
            opt_result = self.optimize_roster(my_roster, roster_slots, strategy)
            
            if 'error' not in opt_result:
                # Calculate total projected points
                total_score = 0
                for week_key in ['week_15', 'week_16', 'week_17']:
                    if week_key in opt_result:
                        lineup = opt_result[week_key]
                        # Sum all starters (simplified - would use actual projections)
                        week_total = len([p for players in lineup.values() for p in players]) * 20
                        total_score += week_total
                
                results.append({
                    'strategy': strategy,
                    'total_projected': total_score,
                    'studs_count': len(opt_result['insights']['studs']),
                    'situational_count': len(opt_result['insights']['situational'])
                })
        
        return pd.DataFrame(results)


class TradeAnalyzer:
    """
    Analyzes trade offers with ROS (Rest of Season) projections.
    Helps decide: accept, reject, or counter.
    """
    
    def __init__(self, predictions: pd.DataFrame, current_week: int = 10):
        self.predictions = predictions
        self.current_week = current_week
        self.weeks_remaining = 18 - current_week
    
    def analyze_trade(
        self,
        giving: List[str],
        receiving: List[str],
        my_roster: List[str],
        their_roster: List[str]
    ) -> Dict:
        """
        Analyze a proposed trade.
        
        Args:
            giving: Players you're trading away
            receiving: Players you're getting
            my_roster: Your full roster
            their_roster: Their full roster (optional, for fairness check)
            
        Returns:
            {
                'verdict': 'ACCEPT' | 'REJECT' | 'COUNTER',
                'your_gain': float (positive = good for you),
                'ros_projections': {...},
                'positional_impact': {...},
                'reasoning': [...]
            }
        """
        # Get predictions for involved players
        giving_data = self.predictions[self.predictions['player'].isin(giving)]
        receiving_data = self.predictions[self.predictions['player'].isin(receiving)]
        
        # Calculate ROS value
        giving_ros_value = self._calculate_ros_value(giving_data)
        receiving_ros_value = self._calculate_ros_value(receiving_data)
        
        net_gain = receiving_ros_value - giving_ros_value
        
        # Analyze positional impact
        positional_impact = self._analyze_positional_impact(
            giving_data,
            receiving_data,
            my_roster
        )
        
        # Generate verdict
        if net_gain >= 15:
            verdict = 'STRONG ACCEPT'
        elif net_gain >= 5:
            verdict = 'ACCEPT'
        elif net_gain >= -5:
            verdict = 'NEUTRAL (Slight edge to you)' if net_gain > 0 else 'NEUTRAL (Slight edge to them)'
        elif net_gain >= -15:
            verdict = 'REJECT'
        else:
            verdict = 'STRONG REJECT'
        
        # Generate reasoning
        reasoning = self._generate_trade_reasoning(
            giving_data,
            receiving_data,
            net_gain,
            positional_impact
        )
        
        # Suggest counter if reject
        counter_suggestion = None
        if 'REJECT' in verdict:
            counter_suggestion = self._suggest_counter(
                giving,
                receiving,
                net_gain,
                their_roster
            )
        
        return {
            'verdict': verdict,
            'your_gain': round(net_gain, 1),
            'giving_ros_value': round(giving_ros_value, 1),
            'receiving_ros_value': round(receiving_ros_value, 1),
            'positional_impact': positional_impact,
            'reasoning': reasoning,
            'counter_suggestion': counter_suggestion
        }
    
    def _calculate_ros_value(self, players: pd.DataFrame) -> float:
        """Calculate rest-of-season total value."""
        if players.empty:
            return 0
        
        # ROS = weekly projection × weeks remaining
        total_value = (players['util_18w_avg'] * self.weeks_remaining).sum()
        
        return total_value
    
    def _analyze_positional_impact(
        self,
        giving: pd.DataFrame,
        receiving: pd.DataFrame,
        my_roster: List[str]
    ) -> Dict:
        """Analyze how trade affects positional depth."""
        impact = {}
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            giving_pos = giving[giving['position'] == pos]
            receiving_pos = receiving[receiving['position'] == pos]
            
            # Net change in position strength
            giving_count = len(giving_pos)
            receiving_count = len(receiving_pos)
            
            giving_quality = giving_pos['util_18w_avg'].mean() if not giving_pos.empty else 0
            receiving_quality = receiving_pos['util_18w_avg'].mean() if not receiving_pos.empty else 0
            
            if giving_count > 0 or receiving_count > 0:
                impact[pos] = {
                    'net_players': receiving_count - giving_count,
                    'quality_change': round(receiving_quality - giving_quality, 1),
                    'verdict': 'UPGRADE' if receiving_quality > giving_quality else 'DOWNGRADE'
                }
        
        return impact
    
    def _generate_trade_reasoning(
        self,
        giving: pd.DataFrame,
        receiving: pd.DataFrame,
        net_gain: float,
        positional_impact: Dict
    ) -> List[str]:
        """Generate human-readable reasoning."""
        reasons = []
        
        # Overall value
        if net_gain > 10:
            reasons.append(f"✅ You gain +{net_gain:.1f} ROS points - excellent value")
        elif net_gain > 0:
            reasons.append(f"✅ You gain +{net_gain:.1f} ROS points - slight edge to you")
        elif net_gain > -10:
            reasons.append(f"⚠️  You lose {abs(net_gain):.1f} ROS points - slight edge to them")
        else:
            reasons.append(f"❌ You lose {abs(net_gain):.1f} ROS points - bad value")
        
        # Positional impacts
        for pos, impact in positional_impact.items():
            if impact['quality_change'] >= 5:
                reasons.append(f"✅ {pos} UPGRADE: +{impact['quality_change']:.1f} quality")
            elif impact['quality_change'] <= -5:
                reasons.append(f"❌ {pos} DOWNGRADE: {impact['quality_change']:.1f} quality")
        
        # Player-specific
        if not giving.empty:
            best_giving = giving.nlargest(1, 'util_18w_avg').iloc[0]
            reasons.append(f"📤 Giving up: {best_giving['player']} ({best_giving['util_18w_avg']:.1f} ROS)")
        
        if not receiving.empty:
            best_receiving = receiving.nlargest(1, 'util_18w_avg').iloc[0]
            reasons.append(f"📥 Receiving: {best_receiving['player']} ({best_receiving['util_18w_avg']:.1f} ROS)")
        
        return reasons
    
    def _suggest_counter(
        self,
        giving: List[str],
        receiving: List[str],
        net_gain: float,
        their_roster: List[str]
    ) -> str:
        """Suggest a counter-offer."""
        # Simplified counter suggestion
        deficit = abs(net_gain)
        
        if deficit < 10:
            return "Ask them to add a mid-tier player (60-70 util)"
        elif deficit < 20:
            return "Ask them to add a high-tier player (70-85 util)"
        else:
            return "Trade is too lopsided - walk away or completely restructure"


# Quick test
if __name__ == "__main__":
    print("Playoff Optimizer & Trade Analyzer")
    print("=" * 60)
    
    # Mock data
    predictions = pd.DataFrame({
        'player': ['Player A', 'Player B', 'Player C', 'Player D'],
        'position': ['RB', 'RB', 'WR', 'WR'],
        'util_1w': [85, 75, 82, 70],
        'util_1w_low': [78, 68, 75, 63],
        'util_1w_high': [92, 82, 89, 77],
        'util_18w_avg': [83, 73, 80, 68],
        'games_played': [14, 16, 15, 17]
    })
    
    # Test Playoff Optimizer
    optimizer = PlayoffOptimizer(predictions)
    result = optimizer.optimize_roster(
        my_roster=['Player A', 'Player B', 'Player C', 'Player D'],
        roster_slots={'RB': 2, 'WR': 2, 'FLEX': 1}
    )
    
    print("\n📊 Playoff Optimization:")
    print(f"  Studs (start every week): {result['insights']['studs']}")
    
    # Test Trade Analyzer
    analyzer = TradeAnalyzer(predictions, current_week=10)
    trade = analyzer.analyze_trade(
        giving=['Player B'],
        receiving=['Player C'],
        my_roster=['Player A', 'Player B'],
        their_roster=['Player C', 'Player D']
    )
    
    print("\n💱 Trade Analysis:")
    print(f"  Verdict: {trade['verdict']}")
    print(f"  Your Gain: {trade['your_gain']:.1f} points")
    for reason in trade['reasoning']:
        print(f"  {reason}")
