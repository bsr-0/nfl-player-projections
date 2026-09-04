"""
ESPN Fantasy Football Integration

Connects to ESPN Fantasy API to import user's team data for personalized recommendations.

PRIVACY & SECURITY NOTICE:
- All user credentials (espn_s2, SWID) are stored ONLY in memory (session state)
- NO user data is written to disk, database, or any persistent storage
- All data is cleared when the browser session ends or user disconnects
- Credentials are never logged, printed, or transmitted to any third party
- This module only communicates with ESPN's official API
"""

from espn_api.football import League
from typing import Optional, Dict, List, Any
import pandas as pd


class ESPNFantasyConnector:
    """
    Connect to ESPN Fantasy Football league and extract team data.
    
    PRIVACY GUARANTEE:
    - No credentials or user data are persisted to disk
    - All data exists only in memory during the active session
    - Call clear_sensitive_data() to explicitly clear credentials
    """
    
    def __init__(self, league_id: int, year: int, espn_s2: Optional[str] = None, swid: Optional[str] = None):
        """
        Initialize ESPN Fantasy connection.
        
        Args:
            league_id: ESPN league ID (found in league URL)
            year: Season year (e.g., 2024)
            espn_s2: ESPN authentication cookie (required for private leagues) - NOT PERSISTED
            swid: ESPN SWID cookie (required for private leagues) - NOT PERSISTED
            
        Note: Credentials are stored only in memory and never written to disk.
        """
        self.league_id = league_id
        self.year = year
        # Credentials stored in memory only - never persisted
        self._espn_s2 = espn_s2
        self._swid = swid
        self.league = None
        self.connected = False
    
    def clear_sensitive_data(self):
        """Explicitly clear all sensitive credentials from memory."""
        self._espn_s2 = None
        self._swid = None
        self.league = None
        self.connected = False
    
    @property
    def espn_s2(self):
        """Access espn_s2 (read-only, not exposed in repr/str)."""
        return self._espn_s2
    
    @property
    def swid(self):
        """Access swid (read-only, not exposed in repr/str)."""
        return self._swid
    
    def __repr__(self):
        """Safe repr that doesn't expose credentials."""
        return f"ESPNFantasyConnector(league_id={self.league_id}, year={self.year}, connected={self.connected})"
    
    def __str__(self):
        """Safe str that doesn't expose credentials."""
        return self.__repr__()
        
    def connect(self) -> bool:
        """
        Establish connection to ESPN Fantasy league.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.espn_s2 and self.swid:
                # Private league
                self.league = League(
                    league_id=self.league_id,
                    year=self.year,
                    espn_s2=self.espn_s2,
                    swid=self.swid
                )
            else:
                # Public league
                self.league = League(
                    league_id=self.league_id,
                    year=self.year
                )
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to ESPN Fantasy: {e}")
            self.connected = False
            return False
    
    def get_league_info(self) -> Dict[str, Any]:
        """Get basic league information."""
        if not self.connected:
            return {}
        
        return {
            'name': self.league.settings.name,
            'year': self.year,
            'num_teams': len(self.league.teams),
            'current_week': self.league.current_week,
            'scoring_type': self.league.settings.scoring_type if hasattr(self.league.settings, 'scoring_type') else 'Unknown'
        }
    
    def get_my_team(self, team_name: Optional[str] = None, team_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get roster for a specific team.
        
        Args:
            team_name: Team name to find (partial match)
            team_id: Team ID (1-indexed)
            
        Returns:
            Dictionary with team info and roster
        """
        if not self.connected:
            return {}
        
        team = None
        
        if team_id:
            for t in self.league.teams:
                if t.team_id == team_id:
                    team = t
                    break
        elif team_name:
            team_name_lower = team_name.lower()
            for t in self.league.teams:
                if team_name_lower in t.team_name.lower():
                    team = t
                    break
        
        if not team:
            return {'error': 'Team not found'}
        
        roster = []
        for player in team.roster:
            roster.append({
                'name': player.name,
                'position': player.position,
                'team': player.proTeam,
                'projected_points': player.projected_points if hasattr(player, 'projected_points') else 0,
                'points': player.points if hasattr(player, 'points') else 0,
                'slot': player.slot_position if hasattr(player, 'slot_position') else 'Unknown',
                'injury_status': player.injuryStatus if hasattr(player, 'injuryStatus') else 'Active'
            })
        
        return {
            'team_name': team.team_name,
            'team_id': team.team_id,
            'wins': team.wins,
            'losses': team.losses,
            'points_for': team.points_for,
            'points_against': team.points_against,
            'roster': roster
        }
    
    def get_all_teams(self) -> List[Dict[str, Any]]:
        """Get summary of all teams in the league."""
        if not self.connected:
            return []
        
        teams = []
        for team in self.league.teams:
            teams.append({
                'team_id': team.team_id,
                'team_name': team.team_name,
                'owner': team.owner if hasattr(team, 'owner') else 'Unknown',
                'wins': team.wins,
                'losses': team.losses,
                'points_for': team.points_for
            })
        
        return sorted(teams, key=lambda x: x['points_for'], reverse=True)
    
    def get_roster_as_dataframe(self, team_name: Optional[str] = None, team_id: Optional[int] = None) -> pd.DataFrame:
        """Get team roster as a pandas DataFrame."""
        team_data = self.get_my_team(team_name=team_name, team_id=team_id)
        
        if 'error' in team_data or not team_data.get('roster'):
            return pd.DataFrame()
        
        return pd.DataFrame(team_data['roster'])
    
    def analyze_team_needs(self, team_name: Optional[str] = None, team_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze team roster to identify needs and strengths.
        
        Returns:
            Dictionary with position breakdown, depth analysis, and recommendations
        """
        roster_df = self.get_roster_as_dataframe(team_name=team_name, team_id=team_id)
        
        if roster_df.empty:
            return {'error': 'Could not load roster'}
        
        # Position counts
        position_counts = roster_df['position'].value_counts().to_dict()
        
        # Ideal roster composition (standard league)
        ideal_counts = {'QB': 2, 'RB': 4, 'WR': 4, 'TE': 2}
        
        # Identify needs
        needs = []
        strengths = []
        
        for pos, ideal in ideal_counts.items():
            actual = position_counts.get(pos, 0)
            if actual < ideal:
                needs.append({
                    'position': pos,
                    'current': actual,
                    'ideal': ideal,
                    'priority': 'High' if actual == 0 else 'Medium'
                })
            elif actual > ideal:
                strengths.append({
                    'position': pos,
                    'current': actual,
                    'surplus': actual - ideal
                })
        
        # Calculate average projected points by position
        avg_by_position = roster_df.groupby('position')['projected_points'].mean().to_dict()
        
        return {
            'position_counts': position_counts,
            'needs': needs,
            'strengths': strengths,
            'avg_projected_by_position': avg_by_position,
            'total_projected': roster_df['projected_points'].sum(),
            'recommendations': self._generate_recommendations(needs, strengths)
        }
    
    def _generate_recommendations(self, needs: List[Dict], strengths: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on team analysis."""
        recommendations = []
        
        # High priority needs
        high_needs = [n for n in needs if n['priority'] == 'High']
        if high_needs:
            positions = ', '.join([n['position'] for n in high_needs])
            recommendations.append(f"🚨 Critical: You need to add {positions} to your roster")
        
        # Medium priority needs
        med_needs = [n for n in needs if n['priority'] == 'Medium']
        if med_needs:
            for need in med_needs:
                recommendations.append(f"📋 Consider adding another {need['position']} (have {need['current']}, ideal is {need['ideal']})")
        
        # Trade opportunities from strengths
        if strengths:
            surplus_positions = [s['position'] for s in strengths if s['surplus'] >= 2]
            if surplus_positions:
                recommendations.append(f"💱 Trade opportunity: You have surplus at {', '.join(surplus_positions)}")
        
        # FLEX optimization
        if not recommendations:
            recommendations.append("✅ Your roster looks balanced! Focus on matchup-based decisions.")
        
        return recommendations
    
    def get_free_agents(self, position: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Get available free agents.
        
        Args:
            position: Filter by position (QB, RB, WR, TE)
            limit: Maximum number of players to return
            
        Returns:
            List of free agent player dictionaries
        """
        if not self.connected:
            return []
        
        try:
            if position:
                free_agents = self.league.free_agents(position=position, size=limit)
            else:
                free_agents = self.league.free_agents(size=limit)
            
            return [{
                'name': player.name,
                'position': player.position,
                'team': player.proTeam,
                'projected_points': player.projected_points if hasattr(player, 'projected_points') else 0,
                'percent_owned': player.percent_owned if hasattr(player, 'percent_owned') else 0
            } for player in free_agents]
        except Exception as e:
            print(f"Error fetching free agents: {e}")
            return []


    def get_league_settings(self) -> Dict[str, Any]:
        """Roster slots and scoring rules.

        These are what make league points comparable to the model's raw
        fantasy points: a projection is only meaningful once you know the
        league's PPR value and how many of each slot it starts.
        """
        if not self.connected:
            return {}

        st = self.league.settings
        return {
            'name': st.name,
            'team_count': st.team_count,
            'scoring_type': st.scoring_type,
            'reg_season_count': st.reg_season_count,
            'playoff_team_count': st.playoff_team_count,
            'playoff_matchup_period_length': st.playoff_matchup_period_length,
            'trade_deadline': st.trade_deadline,
            'keeper_count': st.keeper_count,
            'faab': st.faab,
            'acquisition_budget': st.acquisition_budget,
            'position_slot_counts': st.position_slot_counts,
            # Only scoring rules with a non-zero value; the full table is
            # ~150 rows of mostly-zero stat ids and obscures the real format.
            'scoring_format': [
                {'abbr': i.get('abbr'), 'label': i.get('label'),
                 'points': i.get('points')}
                for i in st.scoring_format if i.get('points')
            ],
        }

    def get_matchups(self) -> List[Dict[str, Any]]:
        """Head-to-head schedule, one row per team per week.

        Built from team.schedule (opponent id by week) rather than
        box_scores(), because box scores require played games and this is
        useful before kickoff -- the matchup schedule is fixed when the
        league is created.

        `score` and `outcome` stay None until a week is played.
        """
        if not self.connected:
            return []

        names = {t.team_id: t.team_name for t in self.league.teams}
        rows = []
        for team in self.league.teams:
            for i, opponent in enumerate(team.schedule):
                # Older espn_api returns opponent ids, newer returns Team
                # objects. Accept either rather than pinning a version.
                opp_id = getattr(opponent, 'team_id', opponent)
                scores = getattr(team, 'scores', [])
                outcomes = getattr(team, 'outcomes', [])
                score = scores[i] if i < len(scores) else None
                outcome = outcomes[i] if i < len(outcomes) else None
                rows.append({
                    'week': i + 1,
                    'team_id': team.team_id,
                    'team_name': team.team_name,
                    'opponent_id': opp_id,
                    'opponent_name': names.get(opp_id, 'Unknown'),
                    # 0.0 is ESPN's placeholder for an unplayed week; keep it
                    # NULL so an unplayed week cannot be read as a shutout.
                    'score': score if score else None,
                    'outcome': outcome if outcome and outcome != 'U' else None,
                })
        return rows

def get_espn_auth_instructions() -> str:
    """Return instructions for getting ESPN authentication cookies."""
    return """
## How to Get ESPN Authentication Cookies

For **private leagues**, you need two cookies: `espn_s2` and `SWID`

### Steps:
1. **Log in** to ESPN Fantasy at https://fantasy.espn.com
2. **Open Developer Tools** (F12 or right-click → Inspect)
3. Go to **Application** tab (Chrome) or **Storage** tab (Firefox)
4. Click on **Cookies** → `https://fantasy.espn.com`
5. Find and copy these values:
   - `espn_s2` - A long string of characters
   - `SWID` - A GUID in format `{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`

### Finding Your League ID:
- Go to your league page on ESPN
- Look at the URL: `https://fantasy.espn.com/football/league?leagueId=XXXXXXX`
- The number after `leagueId=` is your League ID

### Note:
- Public leagues don't require authentication cookies
- Cookies expire periodically, so you may need to refresh them
"""
