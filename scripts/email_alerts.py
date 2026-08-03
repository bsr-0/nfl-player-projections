"""
Email Alert System - Weekly Fantasy Intel
Sends automated emails with personalized insights
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pandas as pd
from typing import List, Dict
from datetime import datetime
import os

class WeeklyEmailAlerts:
    """
    Sends weekly fantasy football insights via email.
    Personalized based on user's roster (optional).
    """
    
    def __init__(self, smtp_config: Dict = None):
        """
        Args:
            smtp_config: Optional dict with server, port, username, password.
                Prefer env vars SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
                (see .env.example). Do not put real passwords in code.
        """
        self.smtp_config = smtp_config or self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get SMTP config from environment variables."""
        return {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', 587)),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', '')
        }
    
    def generate_weekly_insights(
        self,
        predictions: pd.DataFrame,
        performance_data: Dict = None,
        user_roster: List[str] = None
    ) -> str:
        """
        Generate HTML email content with weekly insights.
        
        Args:
            predictions: Current week predictions
            performance_data: Model performance metrics
            user_roster: Optional list of player names on user's team
            
        Returns:
            HTML email content
        """
        current_week = self._get_current_week()
        
        # Identify hot/cold players
        hot_players = self._identify_hot_players(predictions)
        cold_players = self._identify_cold_players(predictions)
        sleepers = self._identify_sleepers(predictions)
        
        # Personalized roster insights
        roster_insights = ""
        if user_roster:
            roster_insights = self._generate_roster_insights(predictions, user_roster)
        
        # Build HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; text-align: center; }}
                .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; 
                          border-left: 4px solid #667eea; }}
                .hot {{ color: #10b981; font-weight: bold; }}
                .cold {{ color: #ef4444; font-weight: bold; }}
                .sleeper {{ color: #3b82f6; font-weight: bold; }}
                .player-card {{ background: white; padding: 15px; margin: 10px 0; 
                              border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin-right: 20px; }}
                .footer {{ text-align: center; color: #6b7280; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚡ Week {current_week} Fantasy Intel</h1>
                <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
            </div>
            
            {roster_insights}
            
            <div class="section">
                <h2>🔥 Hot Players - Start With Confidence</h2>
                <p>Trending up based on recent utilization and matchups:</p>
                {self._format_player_list(hot_players, 'hot')}
            </div>
            
            <div class="section">
                <h2>❄️ Cold Players - Consider Benching</h2>
                <p>Trending down - proceed with caution:</p>
                {self._format_player_list(cold_players, 'cold')}
            </div>
            
            <div class="section">
                <h2>💎 Sleepers - Waiver Wire Gems</h2>
                <p>Under-the-radar players with upside:</p>
                {self._format_player_list(sleepers, 'sleeper')}
            </div>
            
            {self._format_performance_section(performance_data)}
            
            <div class="footer">
                <p>Powered by NFL Utilization Analytics Dashboard</p>
                <p><small>To unsubscribe or update preferences, reply to this email</small></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_current_week(self) -> int:
        """Calculate current NFL week."""
        from src.utils.nfl_calendar import get_current_nfl_season, _season_start
        season = get_current_nfl_season()
        season_start = _season_start(season)
        today = datetime.now()
        
        if today < season_start:
            return 1
        
        days_since_start = (today - season_start).days
        current_week = min(18, days_since_start // 7 + 1)
        
        return current_week
    
    def _identify_hot_players(self, predictions: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Identify trending up players."""
        # In real implementation, would compare to previous week
        # For now, use high utilization + confidence
        if 'confidence' in predictions.columns:
            hot = predictions[
                (predictions['util_1w'] >= 75) &
                (predictions['confidence'] >= 0.7)
            ].nlargest(n, 'util_1w')
        else:
            hot = predictions[predictions['util_1w'] >= 75].nlargest(n, 'util_1w')
        
        return hot
    
    def _identify_cold_players(self, predictions: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Identify trending down players."""
        # Would normally compare week-over-week decline
        cold = predictions[
            (predictions['util_1w'] < 60) &
            (predictions['tier'].isin(['moderate', 'low']))
        ].head(n)
        
        return cold
    
    def _identify_sleepers(self, predictions: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """Identify high-upside sleepers."""
        # Players with good upside but not elite
        if 'util_1w_high' in predictions.columns:
            sleepers = predictions[
                (predictions['util_1w'].between(60, 75)) &
                (predictions['util_1w_high'] >= 80)
            ].head(n)
        else:
            sleepers = predictions[
                predictions['util_1w'].between(60, 75)
            ].head(n)
        
        return sleepers
    
    def _generate_roster_insights(
        self,
        predictions: pd.DataFrame,
        user_roster: List[str]
    ) -> str:
        """Generate personalized insights for user's roster."""
        my_players = predictions[predictions['player'].isin(user_roster)]
        
        if my_players.empty:
            return ""
        
        # Start/Sit recommendations
        must_starts = my_players[my_players['util_1w'] >= 85]
        sit_candidates = my_players[my_players['util_1w'] < 50]
        
        html = f"""
        <div class="section" style="border-left-color: #10b981;">
            <h2>👤 Your Roster - Personalized Insights</h2>
            
            <h3>Must Starts ({len(must_starts)} players):</h3>
            {self._format_player_list(must_starts, 'hot')}
            
            <h3>Sit Candidates ({len(sit_candidates)} players):</h3>
            {self._format_player_list(sit_candidates, 'cold')}
        </div>
        """
        
        return html
    
    def _format_player_list(self, players: pd.DataFrame, category: str) -> str:
        """Format player list as HTML cards."""
        if players.empty:
            return "<p><em>None this week</em></p>"
        
        html = ""
        for _, player in players.head(5).iterrows():
            util_range = ""
            if 'util_1w_low' in player and 'util_1w_high' in player:
                util_range = f"(Range: {player['util_1w_low']:.1f}-{player['util_1w_high']:.1f})"
            
            html += f"""
            <div class="player-card">
                <span class="{category}">{player['player']}</span> - 
                {player['position']}, {player['team']}
                <br>
                <span class="metric">Projected Util: {player['util_1w']:.1f}</span>
                <span class="metric">{util_range}</span>
            </div>
            """
        
        return html
    
    def _format_performance_section(self, performance_data: Dict = None) -> str:
        """Format model performance metrics."""
        if not performance_data:
            return ""
        
        html = f"""
        <div class="section">
            <h2>📊 Model Performance</h2>
            <p>How accurate were last week's predictions?</p>
            <div class="player-card">
                <span class="metric">Accuracy: {performance_data.get('accuracy', 0):.1f}%</span>
                <span class="metric">Avg Error: ±{performance_data.get('mae', 0):.1f} pts</span>
                <span class="metric">Predictions: {performance_data.get('total', 0)}</span>
            </div>
        </div>
        """
        
        return html
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str
    ) -> bool:
        """
        Send email via SMTP.
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_config['username']
            msg['To'] = to_email
            
            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            print(f"✅ Email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
    
    def send_weekly_blast(
        self,
        subscribers: List[Dict],
        predictions: pd.DataFrame,
        performance_data: Dict = None
    ):
        """
        Send weekly insights to all subscribers.
        
        Args:
            subscribers: [
                {'email': 'user@example.com', 'roster': ['Player A', 'Player B']},
                ...
            ]
        """
        current_week = self._get_current_week()
        
        for subscriber in subscribers:
            # Generate personalized content
            html_content = self.generate_weekly_insights(
                predictions,
                performance_data,
                subscriber.get('roster')
            )
            
            # Send
            subject = f"⚡ Week {current_week} Fantasy Intel - Your Personalized Insights"
            self.send_email(subscriber['email'], subject, html_content)


# Quick test
if __name__ == "__main__":
    print("Email Alert System Test")
    print("=" * 60)
    
    # Mock predictions
    predictions = pd.DataFrame({
        'player': ['Player A', 'Player B', 'Player C'],
        'position': ['RB', 'WR', 'QB'],
        'team': ['KC', 'SF', 'BUF'],
        'util_1w': [88, 75, 65],
        'util_1w_low': [82, 68, 58],
        'util_1w_high': [94, 82, 72],
        'tier': ['elite', 'high', 'moderate']
    })
    
    # Create email system
    email_system = WeeklyEmailAlerts()
    
    # Generate content
    html = email_system.generate_weekly_insights(
        predictions,
        performance_data={'accuracy': 78.5, 'mae': 4.2, 'total': 120},
        user_roster=['Player A', 'Player B']
    )
    
    print("✅ Generated email content")
    print(f"   Length: {len(html)} characters")
    print("\n📧 To send emails, configure SMTP:")
    print("   export SMTP_SERVER=smtp.gmail.com")
    print("   export SMTP_USERNAME=your_email@gmail.com")
    print("   export SMTP_PASSWORD=<your-app-password>")
