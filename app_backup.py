"""
NFL Fantasy Predictor - Modern Web Application

A sleek, actionable fantasy football prediction tool featuring:
- Utilization Score analysis (opportunity-based methodology)
- AI-powered predictions with uncertainty quantification
- Draft rankings with Value Over Replacement
- Start/Sit recommendations with confidence levels
- Backtesting stats proving model accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import POSITIONS, DATA_DIR, MODELS_DIR
from src.utils.database import DatabaseManager
from src.features.utilization import engineer_all_features, UtilizationCalculator
from src.features.qb_features import add_qb_features
from src.data.auto_refresh import get_data_status, auto_refresh

# Page config
st.set_page_config(
    page_title="Fantasy Predictor Pro",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODERN CSS DESIGN
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Hero section */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    .metric-sublabel {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Player cards */
    .player-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s;
    }
    
    .player-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .player-info {
        display: flex;
        flex-direction: column;
    }
    
    .player-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .player-meta {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    .player-stats {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    
    .stat-box {
        text-align: center;
        min-width: 60px;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .stat-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Confidence badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-high {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-medium {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-low {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Action buttons */
    .action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .action-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Utilization bar */
    .util-bar-container {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .util-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .recommendation-title {
        font-size: 1rem;
        font-weight: 600;
        color: #166534;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Data tables */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: #f8fafc !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=3600)
def load_player_data():
    """Load player data from database."""
    db = DatabaseManager()
    return db.get_player_stats()


@st.cache_data(ttl=3600)
def load_player_data_with_features():
    """Load player data with utilization, external, multi-week, and season-long features."""
    from src.data.external_data import add_external_features
    from src.features.multiweek_features import add_multiweek_features
    from src.features.season_long_features import add_season_long_features
    
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=1)
    if not df.empty:
        df = engineer_all_features(df)
        df = add_qb_features(df)
        # Add external features (injury, defense, weather, Vegas)
        df = add_external_features(df)
        # Add multi-week features (schedule strength, aggregation, injury probability)
        df = add_multiweek_features(df, horizons=[1, 5, 18])
        # Add season-long features (age curves, games projection, rookie projections, ADP)
        df = add_season_long_features(df)
    return df


@st.cache_data(ttl=3600)
def load_model_results():
    """Load model training results."""
    results_path = DATA_DIR / "advanced_model_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def get_confidence_badge(r2: float) -> tuple:
    """Get confidence badge based on R² score."""
    if r2 >= 0.85:
        return "High Confidence", "badge-high", "✓"
    elif r2 >= 0.70:
        return "Good Confidence", "badge-medium", "●"
    else:
        return "Moderate", "badge-low", "○"


def get_current_nfl_week():
    """
    Determine the current NFL week based on today's date.
    
    NFL 2025-26 Season Schedule:
    - Regular Season: Week 1 starts Sept 4, 2025
    - Week 18 ends Jan 4, 2026
    - Wild Card: Jan 10-11, 2026
    - Divisional: Jan 17-18, 2026
    - Conference Championships: Jan 25, 2026
    - Super Bowl LX: Feb 8, 2026
    """
    from datetime import datetime, timedelta
    
    today = datetime.now()
    
    # NFL 2025-26 key dates
    season_start = datetime(2025, 9, 4)  # Week 1 Thursday
    super_bowl_date = datetime(2026, 2, 8)  # Super Bowl LX
    
    # Check if we're in Super Bowl week (Feb 1-8, 2026)
    super_bowl_week_start = datetime(2026, 2, 1)
    if super_bowl_week_start <= today <= super_bowl_date + timedelta(days=1):
        return {
            'week': 'Super Bowl',
            'week_num': 22,
            'season': 2025,
            'is_playoffs': True,
            'is_super_bowl': True,
            'game_date': super_bowl_date,
            'description': 'Super Bowl LX'
        }
    
    # Conference Championships (Jan 25-26, 2026)
    conf_champ_start = datetime(2026, 1, 25)
    conf_champ_end = datetime(2026, 1, 26)
    if conf_champ_start <= today <= conf_champ_end + timedelta(days=1):
        return {
            'week': 'Conference Championships',
            'week_num': 21,
            'season': 2025,
            'is_playoffs': True,
            'is_super_bowl': False,
            'game_date': conf_champ_start,
            'description': 'AFC & NFC Championship Games'
        }
    
    # Divisional Round (Jan 17-18, 2026)
    div_start = datetime(2026, 1, 17)
    div_end = datetime(2026, 1, 18)
    if div_start <= today <= div_end + timedelta(days=1):
        return {
            'week': 'Divisional Round',
            'week_num': 20,
            'season': 2025,
            'is_playoffs': True,
            'is_super_bowl': False,
            'game_date': div_start,
            'description': 'Divisional Playoff Games'
        }
    
    # Wild Card (Jan 10-12, 2026)
    wc_start = datetime(2026, 1, 10)
    wc_end = datetime(2026, 1, 12)
    if wc_start <= today <= wc_end + timedelta(days=1):
        return {
            'week': 'Wild Card',
            'week_num': 19,
            'season': 2025,
            'is_playoffs': True,
            'is_super_bowl': False,
            'game_date': wc_start,
            'description': 'Wild Card Playoff Games'
        }
    
    # Regular season calculation
    if today < season_start:
        return {
            'week': 'Preseason',
            'week_num': 0,
            'season': 2025,
            'is_playoffs': False,
            'is_super_bowl': False,
            'game_date': season_start,
            'description': 'Season starts Sept 4, 2025'
        }
    
    # Calculate regular season week
    days_since_start = (today - season_start).days
    week_num = min((days_since_start // 7) + 1, 18)
    
    # Calculate next game date (next Thursday/Sunday)
    week_start = season_start + timedelta(weeks=week_num - 1)
    
    return {
        'week': f'Week {week_num}',
        'week_num': week_num,
        'season': 2025,
        'is_playoffs': False,
        'is_super_bowl': False,
        'game_date': week_start,
        'description': f'Regular Season Week {week_num}'
    }


def render_upcoming_games_section(df):
    """
    Render upcoming games predictions based on current date.
    
    Shows player-level and team-level predictions for the upcoming week.
    """
    from datetime import datetime
    
    current_week = get_current_nfl_week()
    
    # Header with current week info
    if current_week['is_super_bowl']:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); 
                    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏆</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">
                {current_week['description']}
            </div>
            <div style="font-size: 1rem; color: #78350f; margin-top: 0.5rem;">
                {current_week['game_date'].strftime('%B %d, %Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif current_week['is_playoffs']:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; text-align: center;">
            <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">🏈</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: white;">
                {current_week['description']}
            </div>
            <div style="font-size: 0.9rem; color: #e2e8f0; margin-top: 0.5rem;">
                Playoff Predictions
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-title">🏈 Upcoming: {current_week["description"]}</div>', unsafe_allow_html=True)
    
    # Get predictions for upcoming week
    df_features = load_player_data_with_features()
    
    if df_features.empty:
        st.info("Loading predictions...")
        return
    
    # Get latest available data as proxy for upcoming week
    latest_season = df_features['season'].max()
    latest_week = df_features[df_features['season'] == latest_season]['week'].max()
    
    upcoming_df = df_features[
        (df_features['season'] == latest_season) & 
        (df_features['week'] == latest_week)
    ].copy()
    
    if upcoming_df.empty:
        st.info("No upcoming game data available.")
        return
    
    # Player-Level Predictions
    st.markdown("### 🎯 Top Player Predictions")
    
    # Use projection columns if available, otherwise use rolling average
    if 'projection_1w' in upcoming_df.columns:
        proj_col = 'projection_1w'
    elif 'fp_rolling_3' in upcoming_df.columns:
        proj_col = 'fp_rolling_3'
    else:
        proj_col = 'fantasy_points'
    
    # Get top players by position
    tabs = st.tabs(["QB", "RB", "WR", "TE"])
    
    for i, pos in enumerate(["QB", "RB", "WR", "TE"]):
        with tabs[i]:
            pos_df = upcoming_df[upcoming_df['position'] == pos].copy()
            
            if pos_df.empty:
                st.info(f"No {pos} predictions available.")
                continue
            
            # Sort by projection
            pos_df = pos_df.sort_values(proj_col, ascending=False).head(10)
            
            for _, row in pos_df.iterrows():
                proj = row[proj_col] if pd.notna(row[proj_col]) else row['fantasy_points']
                
                # Get floor/ceiling if available
                if 'floor_1w' in row and pd.notna(row['floor_1w']):
                    floor_val = row['floor_1w']
                    ceiling_val = row['ceiling_1w']
                else:
                    floor_val = proj * 0.7
                    ceiling_val = proj * 1.3
                
                # Get matchup info
                opp = row.get('opponent', 'TBD')
                matchup_score = row.get('opp_matchup_score', 0.5)
                
                # Matchup color
                if matchup_score > 0.6:
                    matchup_color = "#22c55e"
                    matchup_label = "Easy"
                elif matchup_score < 0.4:
                    matchup_color = "#ef4444"
                    matchup_label = "Tough"
                else:
                    matchup_color = "#f59e0b"
                    matchup_label = "Avg"
                
                col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1.5])
                
                with col1:
                    st.markdown(f"""
                    <div style="padding: 0.25rem 0;">
                        <span style="font-weight: 600;">{row['name']}</span>
                        <span style="color: #64748b; font-size: 0.85rem;"> • {row['team']} vs {opp}</span>
                        <span style="background: {matchup_color}; color: white; padding: 0.1rem 0.4rem; 
                                     border-radius: 4px; font-size: 0.7rem; margin-left: 0.5rem;">{matchup_label}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Proj", f"{proj:.1f}")
                
                with col3:
                    st.metric("Floor", f"{floor_val:.1f}")
                
                with col4:
                    st.metric("Ceiling", f"{ceiling_val:.1f}")
    
    # Team Matchup Summary
    st.markdown("### 🏟️ Team Matchup Summary")
    
    if 'opp_matchup_score' in upcoming_df.columns:
        # Get unique team matchups
        team_matchups = upcoming_df.groupby(['team', 'opponent']).agg({
            'opp_matchup_score': 'first',
            'opp_defense_rank': 'first',
            'implied_team_total': 'first' if 'implied_team_total' in upcoming_df.columns else 'count'
        }).reset_index()
        
        team_matchups = team_matchups.sort_values('opp_matchup_score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Offensive Matchups**")
            for _, row in team_matchups.head(5).iterrows():
                score = row['opp_matchup_score'] * 100
                implied = row.get('implied_team_total', 0)
                implied_str = f" | Impl: {implied:.1f}" if implied > 0 else ""
                st.markdown(f"**{row['team']}** vs {row['opponent']}: {score:.0f}/100{implied_str}")
        
        with col2:
            st.markdown("**Toughest Matchups**")
            for _, row in team_matchups.tail(5).iterrows():
                score = row['opp_matchup_score'] * 100
                st.markdown(f"**{row['team']}** vs {row['opponent']}: {score:.0f}/100")
    else:
        st.info("Team matchup data not available.")


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render modern sidebar."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem;">🏈</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: white; margin-top: 0.5rem;">
                Fantasy Predictor
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Pro Edition</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            options=[
                "🏠 Dashboard",
                "📊 Utilization Analysis",
                "🎯 Weekly Predictions",
                "📅 Multi-Week Projections",
                "🏟️ Matchup Analysis",
                "📋 Draft Rankings",
                "💎 Draft Value Analysis",
                "🔄 Start/Sit Tool",
                "🧪 Model Lab",
                "� Methodology",
                "� Model Accuracy",
                "🔍 Player Search",
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Data status
        try:
            status = get_data_status()
            current_season = status.get('current_season', 2025)
            local_seasons = status.get('local_seasons', [])
            
            st.markdown(f"""
            <div style="color: #94a3b8; font-size: 0.8rem;">
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: #667eea;">●</span> NFL Season: {current_season}
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: #667eea;">●</span> Data: {min(local_seasons) if local_seasons else 'N/A'}-{max(local_seasons) if local_seasons else 'N/A'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show if 2025 data is pending
            if current_season not in local_seasons:
                st.markdown(f"""
                <div style="color: #fbbf24; font-size: 0.75rem; margin-top: 0.5rem;">
                    ⏳ {current_season} data pending
                </div>
                """, unsafe_allow_html=True)
        except:
            pass
        
        # Refresh button
        if st.button("🔄 Check for Updates", use_container_width=True):
            with st.spinner("Checking for new data..."):
                try:
                    result = auto_refresh()
                    if result.get('seasons_loaded') or result.get('weeks_updated'):
                        st.success("New data loaded!")
                        st.rerun()
                    else:
                        st.info("Already up-to-date")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        return page


# =============================================================================
# DASHBOARD PAGE
# =============================================================================
def render_dashboard():
    """Render main dashboard."""
    st.markdown('<h1 class="hero-title">Fantasy Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-powered predictions with utilization analysis and uncertainty quantification</p>', unsafe_allow_html=True)
    
    results = load_model_results()
    df = load_player_data()
    
    # Model accuracy cards
    st.markdown('<div class="section-title">🎯 Model Accuracy by Position</div>', unsafe_allow_html=True)
    
    if results and 'backtest_results' in results:
        cols = st.columns(4)
        
        for i, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
            if pos in results['backtest_results']:
                m = results['backtest_results'][pos]
                r2 = m['r2']
                rmse = m['rmse']
                label, badge_class, icon = get_confidence_badge(r2)
                
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;">{pos}</div>
                        <div class="metric-value">{r2:.0%}</div>
                        <div class="metric-label">Accuracy (R²)</div>
                        <div style="margin-top: 0.75rem;">
                            <span class="badge {badge_class}">{icon} {label}</span>
                        </div>
                        <div class="metric-sublabel">±{rmse:.1f} pts avg error</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Why trust section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-title">✅ Why Trust These Predictions?</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #f8fafc; border-radius: 12px; padding: 1.25rem; border: 1px solid #e2e8f0;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div>
                    <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">📊 Rigorous Backtesting</div>
                    <div style="font-size: 0.85rem; color: #64748b;">Trained on 2020-2023, tested on 2024 data</div>
                </div>
                <div>
                    <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">🔒 No Data Leakage</div>
                    <div style="font-size: 0.85rem; color: #64748b;">Only historical data used for predictions</div>
                </div>
                <div>
                    <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">📈 Utilization-Driven</div>
                    <div style="font-size: 0.85rem; color: #64748b;">RB/WR/TE predictions built on opportunity scores (targets, touches, snap share)</div>
                </div>
                <div>
                    <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">🎯 Uncertainty Quantified</div>
                    <div style="font-size: 0.85rem; color: #64748b;">Floor/ceiling ranges, not just point estimates</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title">📅 Data Coverage</div>', unsafe_allow_html=True)
        
        if not df.empty:
            seasons = sorted(df['season'].unique())
            latest_week = df[df['season'] == max(seasons)]['week'].max()
            
            st.metric("Seasons", f"{min(seasons)}-{max(seasons)}")
            st.metric("Latest Data", f"Week {latest_week}, {max(seasons)}")
            st.metric("Total Games", f"{len(df):,}")
    
    st.markdown("---")
    
    # Upcoming Games Section
    render_upcoming_games_section(df)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown('<div class="section-title">🚀 Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: left;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">Utilization Analysis</div>
            <div style="font-size: 0.85rem; color: #64748b;">See who's getting the opportunities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: left;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">Weekly Predictions</div>
            <div style="font-size: 0.85rem; color: #64748b;">Get projections with floor/ceiling</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: left;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔄</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">Start/Sit Tool</div>
            <div style="font-size: 0.85rem; color: #64748b;">Compare players head-to-head</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# UTILIZATION ANALYSIS PAGE
# =============================================================================
def render_utilization():
    """Render utilization analysis page."""
    st.markdown('<h1 class="hero-title">📊 Utilization Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Opportunity-based player analysis using advanced metrics</p>', unsafe_allow_html=True)

    # Explainer banner: how utilization drives the model
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; color: #e2e8f0;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">🔬</span>
            <span style="font-size: 1.1rem; font-weight: 600; color: #00f5ff;">Why Utilization Score Matters</span>
        </div>
        <p style="font-size: 0.9rem; margin: 0; line-height: 1.6;">
            For <strong>RB, WR, and TE</strong>, our model predicts future utilization first, then converts
            it to fantasy points — making this score the <strong>primary prediction target</strong> for 3 of 4 positions.
            For <strong>QB</strong>, the system evaluates a utilization model against a direct fantasy-points model
            and automatically selects whichever performs better.
            See the <em>Methodology</em> page for the full pipeline breakdown.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_player_data_with_features()

    if df.empty:
        st.warning("No data available. Load data first.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position = st.selectbox("Position", POSITIONS)
    
    with col2:
        seasons = sorted(df['season'].unique(), reverse=True)
        season = st.selectbox("Season", seasons)
    
    with col3:
        min_games = st.slider("Min Games", 1, 17, 4)
    
    # Filter data
    pos_df = df[(df['position'] == position) & (df['season'] == season)]
    
    # Aggregate by player
    player_stats = pos_df.groupby(['player_id', 'name', 'team']).agg({
        'fantasy_points': ['mean', 'std', 'sum', 'count'],
        'target_share': 'mean',
        'rush_share': 'mean',
        'snap_share_calc': 'mean',
        'wopr': 'mean',
        'utilization_score': 'mean',
        'targets': 'mean',
        'rushing_attempts': 'mean',
    }).reset_index()
    
    player_stats.columns = ['player_id', 'name', 'team', 'ppg', 'volatility', 'total_pts', 'games',
                           'target_share', 'rush_share', 'snap_share', 'wopr', 'util_score',
                           'targets_pg', 'carries_pg']
    
    player_stats = player_stats[player_stats['games'] >= min_games]
    player_stats = player_stats.sort_values('util_score', ascending=False)
    
    st.markdown("---")
    
    # Utilization leaderboard
    st.markdown(f'<div class="section-title">🏆 {position} Utilization Leaders - {season}</div>', unsafe_allow_html=True)
    
    for _, row in player_stats.head(15).iterrows():
        util_pct = min(row['util_score'], 100)
        
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
        
        with col1:
            st.markdown(f"""
            <div class="player-info">
                <div class="player-name">{row['name']}</div>
                <div class="player-meta">{row['team']} • {int(row['games'])} games</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{row['ppg']:.1f}</div>
                <div class="stat-label">PPG</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{util_pct:.0f}</div>
                <div class="stat-label">Util Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if position in ['WR', 'TE']:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{row['target_share']*100:.1f}%</div>
                    <div class="stat-label">Tgt Share</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{row['rush_share']*100:.1f}%</div>
                    <div class="stat-label">Rush Share</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div style="width: 100%; margin-top: 0.5rem;">
                <div class="util-bar-container">
                    <div class="util-bar" style="width: {util_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    # Utilization explanation
    st.markdown("---")
    st.markdown('<div class="section-title">📖 Understanding Utilization Score</div>', unsafe_allow_html=True)

    from src.app_data import get_utilization_weights_merged
    util_weights_all = get_utilization_weights_merged()
    weights = util_weights_all.get(position, {})

    # Component descriptions lookup
    _comp_desc = {
        "snap_share": "Percentage of the team's offensive snaps played",
        "rush_share": "Share of team rushing attempts",
        "target_share": "Share of team passing targets",
        "redzone_share": "Share of team red-zone opportunities",
        "touch_share": "Combined carries + receptions as share of team touches",
        "high_value_touch": "Rushes inside the 10-yard line and targets with 15+ air yards",
        "air_yards_share": "Share of team total air yards (route depth)",
        "redzone_targets": "Red-zone target involvement",
        "route_participation": "Routes run as share of team pass plays",
        "inline_rate": "Usage as inline blocker vs. pass-catching role",
        "dropback_rate": "Pass attempts as share of total team plays",
        "rush_attempt_share": "Designed runs and scrambles",
        "redzone_opportunity": "Red-zone scoring opportunity (TD proxy)",
        "play_volume": "Total plays (pass + rush) normalized across the league",
    }

    is_primary = position != "QB"
    role_note = (
        "This score is the <strong>primary prediction target</strong> for our model — "
        "the ensemble predicts future utilization, then converts it to fantasy points."
        if is_primary else
        "For QBs, the model evaluates a utilization-based approach against a direct fantasy-points "
        "approach and automatically selects whichever performs better on held-out data."
    )

    st.markdown(f"""
    <div style="background: #f8fafc; border-radius: 12px; padding: 1.25rem; border: 1px solid #e2e8f0;">
        <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">{position} Utilization Formula</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 1rem;">{role_note}</div>
    """, unsafe_allow_html=True)

    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for metric, weight in sorted_weights:
        pct = weight * 100
        desc = _comp_desc.get(metric, metric.replace("_", " ").title())
        st.markdown(f"""
        <div style="margin-bottom: 0.6rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 600; color: #1e293b;">{metric.replace('_', ' ').title()}</span>
                    <span style="color: #667eea; font-weight: 700; margin-left: 0.5rem;">{pct:.0f}%</span>
                </div>
            </div>
            <div style="background: #e2e8f0; border-radius: 4px; height: 6px; overflow: hidden; margin: 0.2rem 0;">
                <div style="background: #667eea; width: {pct * 3.33}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.75rem; border-top: 1px solid #e2e8f0; padding-top: 0.5rem;">
            Weights shown are defaults. During model training, weights are optimized per position
            using non-negative least squares to maximize correlation with future production.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# WEEKLY PREDICTIONS PAGE
# =============================================================================
def render_predictions():
    """Render weekly predictions page."""
    st.markdown('<h1 class="hero-title">🎯 Weekly Predictions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI projections with floor, ceiling, and confidence intervals</p>', unsafe_allow_html=True)
    
    df = load_player_data_with_features()
    results = load_model_results()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position = st.selectbox("Position", ["All"] + POSITIONS)
    
    with col2:
        seasons = sorted(df['season'].unique(), reverse=True)
        season = st.selectbox("Season", seasons)
    
    with col3:
        weeks = sorted(df[df['season'] == season]['week'].unique(), reverse=True)
        week = st.selectbox("Week", weeks)
    
    # Get model RMSE for uncertainty
    rmse = 4.0
    if results and 'backtest_results' in results:
        if position != "All" and position in results['backtest_results']:
            rmse = results['backtest_results'][position]['rmse']
    
    # Filter data
    week_df = df[(df['season'] == season) & (df['week'] == week)]
    if position != "All":
        week_df = week_df[week_df['position'] == position]
    
    # Calculate predictions (using rolling averages as proxy)
    if 'fp_rolling_3' in week_df.columns:
        week_df['prediction'] = week_df['fp_rolling_3'].fillna(week_df['fantasy_points'])
    else:
        week_df['prediction'] = week_df['fantasy_points']
    
    week_df['floor'] = (week_df['prediction'] - 1.28 * rmse).clip(0)
    week_df['ceiling'] = week_df['prediction'] + 1.28 * rmse
    week_df['actual'] = week_df['fantasy_points']
    
    week_df = week_df.sort_values('prediction', ascending=False)
    
    st.markdown("---")
    st.markdown(f'<div class="section-title">📋 Week {week} Projections</div>', unsafe_allow_html=True)
    
    # Display predictions
    for _, row in week_df.head(25).iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1])
        
        with col1:
            st.markdown(f"""
            <div class="player-info">
                <div class="player-name">{row['name']}</div>
                <div class="player-meta">{row['position']} • {row['team']} vs {row.get('opponent', 'TBD')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Projection", f"{row['prediction']:.1f}")
        
        with col3:
            st.metric("Floor", f"{row['floor']:.1f}")
        
        with col4:
            st.metric("Ceiling", f"{row['ceiling']:.1f}")
        
        with col5:
            st.metric("Actual", f"{row['actual']:.1f}")
        
        with col6:
            diff = row['actual'] - row['prediction']
            if abs(diff) <= rmse:
                st.markdown('<span class="badge badge-high">✓ Hit</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge badge-low">{diff:+.1f}</span>', unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.25rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)


# =============================================================================
# DRAFT RANKINGS PAGE
# =============================================================================
def render_draft_rankings():
    """Render draft rankings page."""
    st.markdown('<h1 class="hero-title">📋 Draft Rankings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Season-long projections with Value Over Replacement analysis</p>', unsafe_allow_html=True)
    
    df = load_player_data()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    latest_season = df['season'].max()
    
    # Calculate season totals
    season_df = df[df['season'] == latest_season].groupby(['player_id', 'name', 'position', 'team']).agg({
        'fantasy_points': ['sum', 'mean', 'std', 'count']
    }).reset_index()
    season_df.columns = ['player_id', 'name', 'position', 'team', 'total', 'ppg', 'volatility', 'games']
    season_df = season_df[season_df['games'] >= 8]
    
    # Calculate VOR
    replacement = {'QB': 12, 'RB': 24, 'WR': 30, 'TE': 12}
    
    for pos in POSITIONS:
        pos_data = season_df[season_df['position'] == pos].sort_values('ppg', ascending=False)
        if len(pos_data) >= replacement.get(pos, 12):
            repl_ppg = pos_data.iloc[replacement[pos] - 1]['ppg']
        else:
            repl_ppg = pos_data['ppg'].min() if len(pos_data) > 0 else 0
        
        season_df.loc[season_df['position'] == pos, 'vor'] = (
            season_df.loc[season_df['position'] == pos, 'ppg'] - repl_ppg
        )
    
    # Tabs
    tabs = st.tabs(["🏆 Overall"] + [f"{pos}" for pos in POSITIONS])
    
    with tabs[0]:
        st.markdown('<div class="section-title">Overall Rankings by VOR</div>', unsafe_allow_html=True)
        
        display = season_df.sort_values('vor', ascending=False).head(50).copy()
        display['Rank'] = range(1, len(display) + 1)
        display['PPG'] = display['ppg'].round(1)
        display['Total'] = display['total'].round(0).astype(int)
        display['VOR'] = display['vor'].round(1)
        display['Games'] = display['games'].astype(int)
        
        st.dataframe(
            display[['Rank', 'name', 'position', 'team', 'PPG', 'Total', 'VOR', 'Games']].rename(columns={
                'name': 'Player', 'position': 'Pos', 'team': 'Team'
            }),
            use_container_width=True,
            hide_index=True,
            height=600
        )
    
    for i, pos in enumerate(POSITIONS):
        with tabs[i + 1]:
            st.markdown(f'<div class="section-title">{pos} Rankings</div>', unsafe_allow_html=True)
            
            pos_data = season_df[season_df['position'] == pos].sort_values('ppg', ascending=False).copy()
            pos_data['Rank'] = range(1, len(pos_data) + 1)
            pos_data['PPG'] = pos_data['ppg'].round(1)
            pos_data['Total'] = pos_data['total'].round(0).astype(int)
            pos_data['Volatility'] = pos_data['volatility'].round(1)
            pos_data['Games'] = pos_data['games'].astype(int)
            
            st.dataframe(
                pos_data[['Rank', 'name', 'team', 'PPG', 'Total', 'Volatility', 'Games']].rename(columns={
                    'name': 'Player', 'team': 'Team'
                }),
                use_container_width=True,
                hide_index=True,
                height=500
            )


# =============================================================================
# START/SIT TOOL PAGE
# =============================================================================
def render_start_sit():
    """Render start/sit comparison tool."""
    st.markdown('<h1 class="hero-title">🔄 Start/Sit Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Compare players head-to-head with statistical analysis</p>', unsafe_allow_html=True)
    
    df = load_player_data()
    results = load_model_results()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Position filter
    position = st.selectbox("Select Position", POSITIONS)
    
    pos_df = df[df['position'] == position]
    players = sorted(pos_df['name'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Player A")
        player1 = st.selectbox("Select player", players, key="p1")
    
    with col2:
        st.markdown("### Player B")
        other_players = [p for p in players if p != player1]
        player2 = st.selectbox("Select player", other_players, key="p2")
    
    if player1 and player2:
        st.markdown("---")
        
        # Get stats
        p1_df = pos_df[pos_df['name'] == player1]
        p2_df = pos_df[pos_df['name'] == player2]
        
        p1_avg, p1_std = p1_df['fantasy_points'].mean(), p1_df['fantasy_points'].std()
        p2_avg, p2_std = p2_df['fantasy_points'].mean(), p2_df['fantasy_points'].std()
        
        # Recommendation
        if p1_avg > p2_avg:
            winner, loser = player1, player2
            diff = p1_avg - p2_avg
            winner_avg, loser_avg = p1_avg, p2_avg
        else:
            winner, loser = player2, player1
            diff = p2_avg - p1_avg
            winner_avg, loser_avg = p2_avg, p1_avg
        
        confidence = min(diff / (p1_std + p2_std + 0.1) * 50 + 50, 95)
        
        st.markdown(f"""
        <div class="recommendation-box">
            <div class="recommendation-title">🎯 Recommendation: START {winner}</div>
            <div style="color: #166534;">
                {winner} projects {diff:.1f} points higher ({winner_avg:.1f} vs {loser_avg:.1f} PPG)
                <br>
                <span style="font-weight: 600;">Confidence: {confidence:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {player1}")
            st.metric("Avg PPG", f"{p1_avg:.1f}")
            st.metric("Floor (10th)", f"{p1_avg - 1.28*p1_std:.1f}")
            st.metric("Ceiling (90th)", f"{p1_avg + 1.28*p1_std:.1f}")
            st.metric("Volatility", f"±{p1_std:.1f}")
            st.metric("Games", len(p1_df))
        
        with col2:
            st.markdown(f"### {player2}")
            st.metric("Avg PPG", f"{p2_avg:.1f}")
            st.metric("Floor (10th)", f"{p2_avg - 1.28*p2_std:.1f}")
            st.metric("Ceiling (90th)", f"{p2_avg + 1.28*p2_std:.1f}")
            st.metric("Volatility", f"±{p2_std:.1f}")
            st.metric("Games", len(p2_df))
        
        # Distribution chart
        st.markdown("### Performance Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=p1_df['fantasy_points'], name=player1, opacity=0.7, 
                                   marker_color='#667eea'))
        fig.add_trace(go.Histogram(x=p2_df['fantasy_points'], name=player2, opacity=0.7,
                                   marker_color='#764ba2'))
        fig.update_layout(
            barmode='overlay',
            template='plotly_white',
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MODEL ACCURACY PAGE
# =============================================================================
def render_model_accuracy():
    """Render model accuracy page."""
    st.markdown('<h1 class="hero-title">📈 Model Accuracy</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Backtesting results and validation methodology</p>', unsafe_allow_html=True)
    
    results = load_model_results()
    
    if not results:
        st.warning("No model results. Run: `python3 src/models/train_advanced.py`")
        return
    
    # Summary cards
    st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for i, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
        if pos in results.get('backtest_results', {}):
            m = results['backtest_results'][pos]
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-weight: 600; color: #475569;">{pos}</div>
                    <div class="metric-value">{m['r2']:.0%}</div>
                    <div class="metric-label">R² Accuracy</div>
                    <div class="metric-sublabel">
                        RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f}<br>
                        Model: {m['best_model']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology
    st.markdown('<div class="section-title">Validation Methodology</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Time-Series Cross-Validation**
        - Train on seasons 2020-2023
        - Test on season 2024
        - No future data used in training
        
        **Proper Scaling**
        - Scaler fit on training data only
        - Applied to test data without refitting
        """)
    
    with col2:
        st.markdown("""
        **Feature Engineering**
        - Only historical/lagged features
        - Utilization scores (FantasyLife method)
        - Rolling averages and trends
        
        **Model Selection**
        - Compared Ridge, GBM, XGBoost, LightGBM
        - Best model selected per position
        """)
    
    # Interpretation guide
    st.markdown("---")
    st.markdown('<div class="section-title">How to Interpret</div>', unsafe_allow_html=True)
    
    st.markdown("""
    | Metric | Meaning | Good Value |
    |--------|---------|------------|
    | **R²** | % of variance explained | > 80% |
    | **RMSE** | Average error in points | < 4.0 |
    | **MAE** | Mean absolute error | < 3.0 |
    
    **Confidence Levels:**
    - ✓ **High** (R² > 85%): Very reliable
    - ● **Good** (R² 70-85%): Reliable with some uncertainty  
    - ○ **Moderate** (R² < 70%): Use with caution
    """)


# =============================================================================
# PLAYER SEARCH PAGE
# =============================================================================
def render_player_search():
    """Render player search page."""
    st.markdown('<h1 class="hero-title">🔍 Player Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Search and analyze individual players</p>', unsafe_allow_html=True)
    
    df = load_player_data_with_features()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Search
    players = sorted(df['name'].unique())
    player = st.selectbox("Search for a player", players)
    
    if player:
        player_df = df[df['name'] == player].sort_values(['season', 'week'])
        latest = player_df.iloc[-1]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 16px; padding: 1.5rem; color: white; margin: 1rem 0;">
            <div style="font-size: 1.75rem; font-weight: 700;">{player}</div>
            <div style="opacity: 0.9;">{latest['position']} • {latest['team']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg PPG", f"{player_df['fantasy_points'].mean():.1f}")
        with col2:
            st.metric("Total Points", f"{player_df['fantasy_points'].sum():.0f}")
        with col3:
            st.metric("Games", len(player_df))
        with col4:
            st.metric("Volatility", f"±{player_df['fantasy_points'].std():.1f}")
        
        # Chart
        st.markdown('<div class="section-title">Performance Over Time</div>', unsafe_allow_html=True)
        
        fig = px.line(player_df, x='week', y='fantasy_points', color='season',
                     markers=True, template='plotly_white')
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Utilization metrics if available
        if 'utilization_score' in player_df.columns:
            st.markdown('<div class="section-title">Utilization Metrics</div>', unsafe_allow_html=True)
            
            latest_util = player_df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                util = latest_util.get('utilization_score', 0)
                st.metric("Util Score", f"{util:.0f}")
            with col2:
                tgt = latest_util.get('target_share', 0) * 100
                st.metric("Target Share", f"{tgt:.1f}%")
            with col3:
                rush = latest_util.get('rush_share', 0) * 100
                st.metric("Rush Share", f"{rush:.1f}%")
            with col4:
                wopr = latest_util.get('wopr', 0)
                st.metric("WOPR", f"{wopr:.2f}")


# =============================================================================
# MATCHUP ANALYSIS PAGE
# =============================================================================
def render_matchup_analysis():
    """Render matchup analysis page with external data features."""
    st.markdown('<h1 class="hero-title">🏟️ Matchup Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Defense rankings, injury status, weather, and Vegas lines</p>', unsafe_allow_html=True)
    
    df = load_player_data_with_features()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position = st.selectbox("Position", POSITIONS, key="matchup_pos")
    
    with col2:
        seasons = sorted(df['season'].unique(), reverse=True)
        season = st.selectbox("Season", seasons, key="matchup_season")
    
    with col3:
        weeks = sorted(df[df['season'] == season]['week'].unique(), reverse=True)
        week = st.selectbox("Week", weeks, key="matchup_week")
    
    # Filter data
    week_df = df[(df['season'] == season) & (df['week'] == week) & (df['position'] == position)]
    
    if week_df.empty:
        st.info("No data for selected filters.")
        return
    
    st.markdown("---")
    
    # Defense Rankings Section
    st.markdown('<div class="section-title">🛡️ Opponent Defense Rankings</div>', unsafe_allow_html=True)
    
    if 'opp_defense_rank' in week_df.columns:
        # Get unique matchups
        matchups = week_df.groupby(['team', 'opponent']).agg({
            'opp_defense_rank': 'first',
            'opp_matchup_score': 'first',
            'opp_pts_allowed': 'first'
        }).reset_index().sort_values('opp_matchup_score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Matchups (Easiest Defenses)**")
            for _, row in matchups.head(5).iterrows():
                score = row['opp_matchup_score'] * 100
                st.markdown(f"""
                <div style="background: #dcfce7; padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0;">
                    <strong>{row['team']}</strong> vs {row['opponent']} - 
                    Rank #{int(row['opp_defense_rank'])}, Score: {score:.0f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Worst Matchups (Toughest Defenses)**")
            for _, row in matchups.tail(5).iterrows():
                score = row['opp_matchup_score'] * 100
                st.markdown(f"""
                <div style="background: #fee2e2; padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0;">
                    <strong>{row['team']}</strong> vs {row['opponent']} - 
                    Rank #{int(row['opp_defense_rank'])}, Score: {score:.0f}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Defense rankings not available. Run training to generate.")
    
    st.markdown("---")
    
    # Weather & Venue Section
    st.markdown('<div class="section-title">🌤️ Weather & Venue</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    if 'is_dome' in week_df.columns:
        dome_games = week_df[week_df['is_dome'] == 1]['team'].nunique()
        outdoor_games = week_df[week_df['is_outdoor'] == 1]['team'].nunique()
        avg_weather = week_df['weather_score'].mean() if 'weather_score' in week_df.columns else 0.9
        
        with col1:
            st.metric("Dome Games", dome_games)
        with col2:
            st.metric("Outdoor Games", outdoor_games)
        with col3:
            st.metric("Avg Weather Score", f"{avg_weather:.0%}")
    
    st.markdown("---")
    
    # Vegas Lines Section
    st.markdown('<div class="section-title">💰 Vegas Implied Totals</div>', unsafe_allow_html=True)
    
    if 'implied_team_total' in week_df.columns:
        team_totals = week_df.groupby('team')['implied_team_total'].first().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Highest Implied Totals**")
            for team, total in team_totals.head(5).items():
                st.markdown(f"**{team}**: {total:.1f} pts")
        
        with col2:
            st.markdown("**Lowest Implied Totals**")
            for team, total in team_totals.tail(5).items():
                st.markdown(f"**{team}**: {total:.1f} pts")
    
    st.markdown("---")
    
    # Player Matchup Table
    st.markdown('<div class="section-title">📋 Player Matchup Factors</div>', unsafe_allow_html=True)
    
    display_cols = ['name', 'team', 'opponent']
    if 'opp_defense_rank' in week_df.columns:
        display_cols.append('opp_defense_rank')
    if 'opp_matchup_score' in week_df.columns:
        display_cols.append('opp_matchup_score')
    if 'injury_score' in week_df.columns:
        display_cols.append('injury_score')
    if 'weather_score' in week_df.columns:
        display_cols.append('weather_score')
    if 'implied_team_total' in week_df.columns:
        display_cols.append('implied_team_total')
    
    display_cols.append('fantasy_points')
    
    available_cols = [c for c in display_cols if c in week_df.columns]
    
    display_df = week_df[available_cols].sort_values('fantasy_points', ascending=False).head(30)
    
    # Rename for display
    rename_map = {
        'name': 'Player',
        'team': 'Team',
        'opponent': 'Opp',
        'opp_defense_rank': 'Def Rank',
        'opp_matchup_score': 'Matchup',
        'injury_score': 'Health',
        'weather_score': 'Weather',
        'implied_team_total': 'Impl Total',
        'fantasy_points': 'FP'
    }
    
    display_df = display_df.rename(columns=rename_map)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =============================================================================
# MULTI-WEEK PROJECTIONS PAGE
# =============================================================================
def render_multiweek_projections():
    """Render multi-week projections page."""
    st.markdown('<h1 class="hero-title">📅 Multi-Week Projections</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">5-week and season-long projections with schedule strength and injury risk</p>', unsafe_allow_html=True)
    
    df = load_player_data_with_features()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position = st.selectbox("Position", POSITIONS, key="mw_pos")
    
    with col2:
        horizon = st.selectbox("Projection Horizon", [5, 18, 1], 
                               format_func=lambda x: f"{x} Week{'s' if x > 1 else ''}")
    
    with col3:
        seasons = sorted(df['season'].unique(), reverse=True)
        season = st.selectbox("Season", seasons, key="mw_season")
    
    # Filter to latest week of selected season
    season_df = df[df['season'] == season]
    latest_week = season_df['week'].max()
    week_df = season_df[(season_df['week'] == latest_week) & (season_df['position'] == position)]
    
    if week_df.empty:
        st.info("No data for selected filters.")
        return
    
    st.markdown("---")
    
    # Key metrics
    st.markdown(f'<div class="section-title">📊 {horizon}-Week Projection Summary</div>', unsafe_allow_html=True)
    
    proj_col = f'projection_{horizon}w'
    floor_col = f'floor_{horizon}w'
    ceiling_col = f'ceiling_{horizon}w'
    sos_col = f'sos_next_{horizon}'
    injury_col = f'injury_prob_next_{horizon}'
    games_col = f'expected_games_next_{horizon}'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if proj_col in week_df.columns:
            avg_proj = week_df[proj_col].mean()
            st.metric(f"Avg {horizon}W Projection", f"{avg_proj:.1f}")
    
    with col2:
        if games_col in week_df.columns:
            avg_games = week_df[games_col].mean()
            st.metric("Avg Expected Games", f"{avg_games:.1f}")
    
    with col3:
        if injury_col in week_df.columns:
            avg_injury = week_df[injury_col].mean() * 100
            st.metric("Avg Injury Risk", f"{avg_injury:.1f}%")
    
    with col4:
        if sos_col in week_df.columns:
            avg_sos = week_df[sos_col].mean()
            st.metric("Avg Schedule Strength", f"{avg_sos:.2f}")
    
    st.markdown("---")
    
    # Top projections table
    st.markdown(f'<div class="section-title">🏆 Top {position} Projections ({horizon} Weeks)</div>', unsafe_allow_html=True)
    
    display_cols = ['name', 'team']
    
    if proj_col in week_df.columns:
        display_cols.append(proj_col)
    if floor_col in week_df.columns:
        display_cols.append(floor_col)
    if ceiling_col in week_df.columns:
        display_cols.append(ceiling_col)
    if games_col in week_df.columns:
        display_cols.append(games_col)
    if sos_col in week_df.columns:
        display_cols.append(sos_col)
    if injury_col in week_df.columns:
        display_cols.append(injury_col)
    
    available_cols = [c for c in display_cols if c in week_df.columns]
    
    if proj_col in week_df.columns:
        display_df = week_df[available_cols].sort_values(proj_col, ascending=False).head(30)
    else:
        display_df = week_df[available_cols].head(30)
    
    # Rename for display
    rename_map = {
        'name': 'Player',
        'team': 'Team',
        proj_col: f'{horizon}W Proj',
        floor_col: 'Floor',
        ceiling_col: 'Ceiling',
        games_col: 'Exp Games',
        sos_col: 'SOS',
        injury_col: 'Inj Risk'
    }
    
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    
    # Format injury risk as percentage
    if 'Inj Risk' in display_df.columns:
        display_df['Inj Risk'] = (display_df['Inj Risk'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Schedule strength breakdown
    st.markdown('<div class="section-title">📈 Schedule Strength Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    fav_col = f'favorable_matchups_next_{horizon}'
    
    with col1:
        st.markdown("**Easiest Schedules (High SOS)**")
        if sos_col in week_df.columns:
            easy = week_df.nlargest(5, sos_col)[['name', 'team', sos_col]]
            for _, row in easy.iterrows():
                st.markdown(f"**{row['name']}** ({row['team']}): SOS {row[sos_col]:.2f}")
    
    with col2:
        st.markdown("**Hardest Schedules (Low SOS)**")
        if sos_col in week_df.columns:
            hard = week_df.nsmallest(5, sos_col)[['name', 'team', sos_col]]
            for _, row in hard.iterrows():
                st.markdown(f"**{row['name']}** ({row['team']}): SOS {row[sos_col]:.2f}")
    
    st.markdown("---")
    
    # Injury risk breakdown
    st.markdown('<div class="section-title">🏥 Injury Risk Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Lowest Injury Risk**")
        if injury_col in week_df.columns:
            low_risk = week_df.nsmallest(5, injury_col)[['name', 'team', injury_col]]
            for _, row in low_risk.iterrows():
                risk_pct = row[injury_col] * 100
                st.markdown(f"**{row['name']}** ({row['team']}): {risk_pct:.1f}%")
    
    with col2:
        st.markdown("**Highest Injury Risk**")
        if injury_col in week_df.columns:
            high_risk = week_df.nlargest(5, injury_col)[['name', 'team', injury_col]]
            for _, row in high_risk.iterrows():
                risk_pct = row[injury_col] * 100
                st.markdown(f"**{row['name']}** ({row['team']}): {risk_pct:.1f}%")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main application."""
    page = render_sidebar()
    
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "📊 Utilization Analysis":
        render_utilization()
    elif page == "🎯 Weekly Predictions":
        render_predictions()
    elif page == "📅 Multi-Week Projections":
        render_multiweek_projections()
    elif page == "🏟️ Matchup Analysis":
        render_matchup_analysis()
    elif page == "📋 Draft Rankings":
        render_draft_rankings()
    elif page == "💎 Draft Value Analysis":
        render_draft_value_analysis()
    elif page == "🔄 Start/Sit Tool":
        render_start_sit()
    elif page == "🧪 Model Lab":
        render_model_lab()
    elif page == "📖 Methodology":
        render_methodology()
    elif page == "📈 Model Accuracy":
        render_model_accuracy()
    elif page == "🔍 Player Search":
        render_player_search()
    else:
        render_dashboard()


# =============================================================================
# DRAFT VALUE ANALYSIS PAGE
# =============================================================================
def render_draft_value_analysis():
    """Render draft value analysis page with ADP, age curves, and rookie projections."""
    st.markdown('<h1 class="hero-title">💎 Draft Value Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">ADP value picks, age curves, games projection, and rookie analysis</p>', unsafe_allow_html=True)
    
    df = load_player_data_with_features()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        position = st.selectbox("Position", POSITIONS, key="dv_pos")
    
    with col2:
        seasons = sorted(df['season'].unique(), reverse=True)
        season = st.selectbox("Season", seasons, key="dv_season")
    
    # Filter to latest week of selected season
    season_df = df[df['season'] == season]
    latest_week = season_df['week'].max()
    week_df = season_df[(season_df['week'] == latest_week) & (season_df['position'] == position)]
    
    if week_df.empty:
        st.info("No data for selected filters.")
        return
    
    st.markdown("---")
    
    # === ADP VALUE PICKS ===
    st.markdown('<div class="section-title">💰 ADP Value Picks</div>', unsafe_allow_html=True)
    st.markdown("*Players projected to outperform their ADP (positive value = undervalued)*")
    
    if 'adp_value_score' in week_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Value Picks (Undervalued)**")
            value_picks = week_df.nlargest(8, 'adp_value_score')[['name', 'team', 'adp_value_score', 'estimated_adp_round', 'projected_adp_round']]
            for _, row in value_picks.iterrows():
                value = row['adp_value_score']
                color = "#22c55e" if value > 0 else "#ef4444"
                st.markdown(f"""
                <div style="background: #f0fdf4; padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0; border-left: 3px solid {color};">
                    <strong>{row['name']}</strong> ({row['team']})<br>
                    <span style="font-size: 0.85rem;">ADP Rd {int(row['estimated_adp_round'])} → Proj Rd {int(row['projected_adp_round'])} | Value: <strong style="color: {color};">{value:+.1f}</strong></span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Overvalued Players (Avoid)**")
            overvalued = week_df.nsmallest(8, 'adp_value_score')[['name', 'team', 'adp_value_score', 'estimated_adp_round', 'projected_adp_round']]
            for _, row in overvalued.iterrows():
                value = row['adp_value_score']
                color = "#22c55e" if value > 0 else "#ef4444"
                st.markdown(f"""
                <div style="background: #fef2f2; padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0; border-left: 3px solid {color};">
                    <strong>{row['name']}</strong> ({row['team']})<br>
                    <span style="font-size: 0.85rem;">ADP Rd {int(row['estimated_adp_round'])} → Proj Rd {int(row['projected_adp_round'])} | Value: <strong style="color: {color};">{value:+.1f}</strong></span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === AGE CURVES ===
    st.markdown('<div class="section-title">📉 Age & Decline Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'age_factor' in week_df.columns:
        with col1:
            avg_age = week_df['age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f}")
        
        with col2:
            in_prime = week_df['is_in_prime'].mean() * 100
            st.metric("% In Prime", f"{in_prime:.0f}%")
        
        with col3:
            avg_factor = week_df['age_factor'].mean()
            st.metric("Avg Age Factor", f"{avg_factor:.2f}")
        
        with col4:
            avg_decline = week_df['decline_rate'].mean() * 100
            st.metric("Avg Decline Rate", f"{avg_decline:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Players in Prime Window**")
        if 'is_in_prime' in week_df.columns:
            prime_players = week_df[week_df['is_in_prime'] == 1].nlargest(5, 'fantasy_points')[['name', 'team', 'age', 'age_factor']]
            for _, row in prime_players.iterrows():
                st.markdown(f"**{row['name']}** ({row['team']}): Age {int(row['age'])}, Factor {row['age_factor']:.2f}")
    
    with col2:
        st.markdown("**Players Past Prime (Decline Risk)**")
        if 'years_from_peak' in week_df.columns:
            declining = week_df[week_df['years_from_peak'] > 2].nlargest(5, 'decline_rate')[['name', 'team', 'age', 'decline_rate']]
            for _, row in declining.iterrows():
                decline_pct = row['decline_rate'] * 100
                st.markdown(f"**{row['name']}** ({row['team']}): Age {int(row['age'])}, Decline {decline_pct:.1f}%/yr")
    
    st.markdown("---")
    
    # === GAMES PLAYED PROJECTION ===
    st.markdown('<div class="section-title">🏥 Games Played Projection</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    if 'projected_games_season' in week_df.columns:
        with col1:
            avg_games = week_df['projected_games_season'].mean()
            st.metric("Avg Projected Games", f"{avg_games:.1f}")
        
        with col2:
            full_season = (week_df['projected_games_season'] >= 15).mean() * 100
            st.metric("% Full Season (15+)", f"{full_season:.0f}%")
        
        with col3:
            injury_risk = (week_df['projected_games_season'] < 12).mean() * 100
            st.metric("% Injury Risk (<12)", f"{injury_risk:.0f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Durable (Highest Games)**")
        if 'projected_games_season' in week_df.columns:
            durable = week_df.nlargest(5, 'projected_games_season')[['name', 'team', 'projected_games_season', 'projected_games_floor']]
            for _, row in durable.iterrows():
                st.markdown(f"**{row['name']}** ({row['team']}): {row['projected_games_season']:.1f} games (floor: {row['projected_games_floor']:.1f})")
    
    with col2:
        st.markdown("**Highest Injury Risk (Lowest Games)**")
        if 'projected_games_season' in week_df.columns:
            risky = week_df.nsmallest(5, 'projected_games_season')[['name', 'team', 'projected_games_season', 'projected_games_ceiling']]
            for _, row in risky.iterrows():
                st.markdown(f"**{row['name']}** ({row['team']}): {row['projected_games_season']:.1f} games (ceiling: {row['projected_games_ceiling']:.1f})")
    
    st.markdown("---")
    
    # === ROOKIE ANALYSIS ===
    st.markdown('<div class="section-title">🌟 Rookie Analysis</div>', unsafe_allow_html=True)
    
    rookies = week_df[week_df['is_rookie'] == 1] if 'is_rookie' in week_df.columns else pd.DataFrame()
    
    if not rookies.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_rookies = len(rookies)
            st.metric("Rookies", num_rookies)
        
        with col2:
            if 'rookie_projected_ppg' in rookies.columns:
                avg_ppg = rookies['rookie_projected_ppg'].mean()
                st.metric("Avg Rookie Proj PPG", f"{avg_ppg:.1f}")
        
        with col3:
            if 'rookie_projected_total' in rookies.columns:
                avg_total = rookies['rookie_projected_total'].mean()
                st.metric("Avg Rookie Proj Total", f"{avg_total:.1f}")
        
        st.markdown("**Top Rookie Projections**")
        if 'rookie_projected_total' in rookies.columns:
            top_rookies = rookies.nlargest(5, 'rookie_projected_total')[['name', 'team', 'rookie_projected_ppg', 'rookie_projected_games', 'rookie_projected_total']]
            for _, row in top_rookies.iterrows():
                st.markdown(f"**{row['name']}** ({row['team']}): {row['rookie_projected_ppg']:.1f} PPG × {row['rookie_projected_games']:.0f} games = **{row['rookie_projected_total']:.1f} total**")
    else:
        st.info("No rookies found in selected data.")
    
    st.markdown("---")
    
    # === FULL TABLE ===
    st.markdown('<div class="section-title">📋 Complete Draft Value Table</div>', unsafe_allow_html=True)
    
    display_cols = ['name', 'team']
    
    if 'adp_value_score' in week_df.columns:
        display_cols.append('adp_value_score')
    if 'age' in week_df.columns:
        display_cols.append('age')
    if 'age_factor' in week_df.columns:
        display_cols.append('age_factor')
    if 'projected_games_season' in week_df.columns:
        display_cols.append('projected_games_season')
    if 'is_rookie' in week_df.columns:
        display_cols.append('is_rookie')
    if 'fantasy_points' in week_df.columns:
        display_cols.append('fantasy_points')
    
    available_cols = [c for c in display_cols if c in week_df.columns]
    
    if 'adp_value_score' in week_df.columns:
        display_df = week_df[available_cols].sort_values('adp_value_score', ascending=False).head(30)
    else:
        display_df = week_df[available_cols].head(30)
    
    rename_map = {
        'name': 'Player',
        'team': 'Team',
        'adp_value_score': 'ADP Value',
        'age': 'Age',
        'age_factor': 'Age Factor',
        'projected_games_season': 'Proj Games',
        'is_rookie': 'Rookie',
        'fantasy_points': 'FP'
    }
    
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =============================================================================
# MODEL LAB PAGE
# =============================================================================
def render_model_lab():
    """
    Interactive Model Lab for comparing different modeling approaches.
    
    Features:
    - Compare ML, Simulation, Optimization models
    - View performance metrics
    - Customize based on user preferences
    - Monte Carlo simulation
    """
    st.markdown('<h1 class="hero-title">🧪 Model Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Compare models, run simulations, and customize predictions</p>', unsafe_allow_html=True)
    
    # Load model comparison results
    results_path = Path("data/model_comparison_results.json")
    
    if results_path.exists():
        with open(results_path) as f:
            model_results = json.load(f)
    else:
        model_results = {}
        st.warning("Model comparison results not found. Run model comparison first.")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", 
        "🎲 Monte Carlo Simulator",
        "⚙️ User Preferences",
        "🏆 Lineup Optimizer"
    ])
    
    # === TAB 1: MODEL COMPARISON ===
    with tab1:
        st.markdown("### Compare Model Performance")
        st.markdown("*Select different models to view their performance on unseen 2024 data*")
        
        if model_results:
            # Model category filter
            categories = list(set(r['category'] for r in model_results.values()))
            selected_categories = st.multiselect(
                "Filter by Category",
                categories,
                default=categories
            )
            
            # Filter results
            filtered_results = {
                k: v for k, v in model_results.items() 
                if v['category'] in selected_categories
            }
            
            if filtered_results:
                # Create comparison DataFrame
                comparison_data = []
                for name, result in filtered_results.items():
                    comparison_data.append({
                        'Model': result['name'],
                        'Category': result['category'],
                        'Test RMSE': result['rmse'],
                        'Test R²': result['r2'],
                        'MAE': result['mae'],
                        'Overfit Ratio': result['overfitting_ratio'],
                        'Train Time (s)': result['training_time']
                    })
                
                comparison_df = pd.DataFrame(comparison_data).sort_values('Test RMSE')
                
                # Highlight best model
                best_model = comparison_df.iloc[0]['Model']
                
                st.markdown(f"**🏆 Best Model: {best_model}**")
                
                # Display comparison table
                st.dataframe(
                    comparison_df.style.highlight_min(subset=['Test RMSE', 'MAE'], color='lightgreen')
                                       .highlight_max(subset=['Test R²'], color='lightgreen'),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # Model details
                st.markdown("### Model Details")
                
                selected_model = st.selectbox(
                    "Select a model to view details",
                    list(filtered_results.keys()),
                    format_func=lambda x: filtered_results[x]['name']
                )
                
                if selected_model:
                    model_info = filtered_results[selected_model]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{model_info['name']}**")
                        st.markdown(f"*{model_info['description']}*")
                        st.markdown(f"**Category:** {model_info['category']}")
                        
                        st.markdown("**Pros:**")
                        for pro in model_info.get('pros', []):
                            st.markdown(f"- ✅ {pro}")
                    
                    with col2:
                        st.metric("Test RMSE", f"{model_info['rmse']:.3f}")
                        st.metric("Test R²", f"{model_info['r2']:.3f}")
                        st.metric("Training Time", f"{model_info['training_time']:.2f}s")
                        
                        st.markdown("**Cons:**")
                        for con in model_info.get('cons', []):
                            st.markdown(f"- ⚠️ {con}")
                
                # Visual comparison
                st.markdown("---")
                st.markdown("### Visual Comparison")
                
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Bar chart of RMSE
                fig = px.bar(
                    comparison_df, 
                    x='Model', 
                    y='Test RMSE',
                    color='Category',
                    title='Model Performance (Lower RMSE = Better)'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results available. Run the model comparison pipeline first.")
            if st.button("Run Model Comparison"):
                with st.spinner("Running model comparison... This may take a few minutes."):
                    try:
                        from src.models.advanced_modeling import run_model_comparison
                        run_model_comparison()
                        st.success("Model comparison complete! Refresh the page to see results.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error running comparison: {e}")
    
    # === TAB 2: MONTE CARLO SIMULATOR ===
    with tab2:
        st.markdown("### Monte Carlo Simulation")
        st.markdown("*Simulate thousands of possible outcomes for player projections*")
        
        df = load_player_data_with_features()
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                position = st.selectbox("Position", POSITIONS, key="mc_pos")
            
            with col2:
                n_sims = st.slider("Number of Simulations", 1000, 50000, 10000, step=1000)
            
            # Get players
            latest_season = df['season'].max()
            latest_week = df[df['season'] == latest_season]['week'].max()
            
            pos_df = df[
                (df['position'] == position) & 
                (df['season'] == latest_season) & 
                (df['week'] == latest_week)
            ].copy()
            
            if not pos_df.empty:
                player_options = pos_df.sort_values('fantasy_points', ascending=False)['name'].unique()[:20]
                selected_player = st.selectbox("Select Player", player_options)
                
                if st.button("Run Simulation", type="primary"):
                    player_row = pos_df[pos_df['name'] == selected_player].iloc[0]
                    
                    # Prepare player data for simulation
                    player_data = {
                        'projection': player_row.get('fp_rolling_3', player_row['fantasy_points']),
                        'volatility': player_row.get('weekly_volatility', 5.0),
                        'matchup_score': player_row.get('opp_matchup_score', 1.0),
                        'injury_prob': player_row.get('injury_prob_next_1', 0.05),
                        'boom_factor': player_row.get('boom_bust_range', 10) / 20
                    }
                    
                    # Run simulation
                    from src.models.advanced_modeling import MonteCarloSimulator
                    simulator = MonteCarloSimulator(n_simulations=n_sims)
                    results = simulator.simulate_player(player_data, n_sims)
                    
                    # Display results
                    st.markdown(f"### Simulation Results for {selected_player}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Expected", f"{results['mean']:.1f}")
                    with col2:
                        st.metric("Floor (10%)", f"{results['floor']:.1f}")
                    with col3:
                        st.metric("Ceiling (90%)", f"{results['ceiling']:.1f}")
                    with col4:
                        st.metric("Boom Prob", f"{results['boom_prob']*100:.1f}%")
                    
                    # Distribution plot
                    import plotly.figure_factory as ff
                    
                    fig = ff.create_distplot(
                        [results['simulations'][results['simulations'] > 0]],
                        [selected_player],
                        bin_size=1,
                        show_rug=False
                    )
                    fig.update_layout(
                        title=f"Simulated Outcome Distribution ({n_sims:,} simulations)",
                        xaxis_title="Fantasy Points",
                        yaxis_title="Probability Density"
                    )
                    
                    # Add vertical lines for key percentiles
                    fig.add_vline(x=results['floor'], line_dash="dash", line_color="red", 
                                  annotation_text="Floor")
                    fig.add_vline(x=results['mean'], line_dash="solid", line_color="green",
                                  annotation_text="Expected")
                    fig.add_vline(x=results['ceiling'], line_dash="dash", line_color="blue",
                                  annotation_text="Ceiling")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional stats
                    st.markdown("#### Detailed Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Median", f"{results['median']:.1f}")
                        st.metric("Std Dev", f"{results['std']:.1f}")
                    
                    with stats_col2:
                        st.metric("25th Percentile", f"{results['p25']:.1f}")
                        st.metric("75th Percentile", f"{results['p75']:.1f}")
                    
                    with stats_col3:
                        st.metric("Bust Prob (<50%)", f"{results['bust_prob']*100:.1f}%")
                        st.metric("Zero/DNP Prob", f"{results['zero_prob']*100:.1f}%")
    
    # === TAB 3: USER PREFERENCES ===
    with tab3:
        st.markdown("### Customize Your Predictions")
        st.markdown("*Adjust predictions based on your fantasy play style*")
        
        # Profile selection
        profile_type = st.radio(
            "Select Your Play Style",
            ["Conservative", "Balanced", "Aggressive", "Boom or Bust", "Custom"],
            horizontal=True
        )
        
        if profile_type == "Custom":
            col1, col2 = st.columns(2)
            
            with col1:
                risk_tolerance = st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1,
                                          help="0 = Very Conservative, 1 = Very Aggressive")
                prefer_ceiling = st.checkbox("Prefer Ceiling Over Floor",
                                            help="Weight ceiling projections more heavily")
            
            with col2:
                prefer_consistency = st.checkbox("Prefer Consistent Players",
                                                help="Bonus for low-variance players")
                boom_bust_pref = st.slider("Boom/Bust Preference", 0.0, 1.0, 0.5, 0.1,
                                          help="0 = Safe plays, 1 = High variance plays")
        else:
            profiles = {
                "Conservative": (0.2, False, True, 0.2),
                "Balanced": (0.5, False, False, 0.5),
                "Aggressive": (0.8, True, False, 0.8),
                "Boom or Bust": (1.0, True, False, 1.0)
            }
            risk_tolerance, prefer_ceiling, prefer_consistency, boom_bust_pref = profiles[profile_type]
        
        # Display profile summary
        st.markdown("---")
        st.markdown("### Your Profile")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Risk Level", f"{risk_tolerance*100:.0f}%")
        with col2:
            st.metric("Focus", "Ceiling" if prefer_ceiling else "Floor")
        with col3:
            st.metric("Consistency", "Yes" if prefer_consistency else "No")
        with col4:
            st.metric("Boom/Bust", f"{boom_bust_pref*100:.0f}%")
        
        # Show how this affects predictions
        st.markdown("---")
        st.markdown("### How This Affects Your Predictions")
        
        if risk_tolerance < 0.3:
            st.info("🛡️ **Conservative**: Predictions weighted toward floor projections. Ideal for protecting leads or in favorable matchups.")
        elif risk_tolerance > 0.7:
            st.warning("🚀 **Aggressive**: Predictions weighted toward ceiling projections. Ideal for chasing points when behind.")
        else:
            st.success("⚖️ **Balanced**: Equal weight to floor and ceiling. Good for most situations.")
    
    # === TAB 4: LINEUP OPTIMIZER ===
    with tab4:
        st.markdown("### Lineup Optimizer")
        st.markdown("*Build optimal lineups based on your preferences*")
        
        df = load_player_data_with_features()
        
        if not df.empty:
            # Optimization objective
            objective = st.radio(
                "Optimization Objective",
                ["Maximize Expected Points", "Maximize Floor (Safe)", "Maximize Ceiling (Upside)"],
                horizontal=True
            )
            
            obj_map = {
                "Maximize Expected Points": "expected",
                "Maximize Floor (Safe)": "floor",
                "Maximize Ceiling (Upside)": "ceiling"
            }
            
            if st.button("Generate Optimal Lineup", type="primary"):
                # Get latest data
                latest_season = df['season'].max()
                latest_week = df[df['season'] == latest_season]['week'].max()
                
                current_df = df[
                    (df['season'] == latest_season) & 
                    (df['week'] == latest_week)
                ].copy()
                
                if not current_df.empty:
                    from src.models.advanced_modeling import LineupOptimizer
                    
                    optimizer = LineupOptimizer()
                    lineup = optimizer.optimize_lineup(
                        current_df, 
                        objective=obj_map[objective]
                    )
                    
                    if not lineup.empty:
                        st.markdown("### Optimal Lineup")
                        
                        # Display lineup
                        display_cols = ['position', 'name', 'team']
                        
                        if 'projection_1w' in lineup.columns:
                            display_cols.append('projection_1w')
                        elif 'fp_rolling_3' in lineup.columns:
                            display_cols.append('fp_rolling_3')
                        
                        if 'floor_1w' in lineup.columns:
                            display_cols.extend(['floor_1w', 'ceiling_1w'])
                        
                        available_cols = [c for c in display_cols if c in lineup.columns]
                        
                        lineup_display = lineup[available_cols].copy()
                        lineup_display.columns = [c.replace('_1w', '').replace('_', ' ').title() for c in available_cols]
                        
                        st.dataframe(lineup_display, use_container_width=True, hide_index=True)
                        
                        # Total projection
                        if 'projection_1w' in lineup.columns:
                            total = lineup['projection_1w'].sum()
                        elif 'fp_rolling_3' in lineup.columns:
                            total = lineup['fp_rolling_3'].sum()
                        else:
                            total = lineup['fantasy_points'].sum()
                        
                        st.metric("Total Projected Points", f"{total:.1f}")
                    else:
                        st.warning("Could not generate lineup.")
                else:
                    st.warning("No current week data available.")
        else:
            st.warning("No data available.")


# =============================================================================
# METHODOLOGY PAGE
# =============================================================================
def render_methodology():
    """
    Render methodology page explaining the prediction approach with empirical evidence.
    """
    st.markdown('<h1 class="hero-title">📖 Methodology</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Understanding our prediction approach with empirical evidence</p>', unsafe_allow_html=True)
    
    # Load approach comparison results
    results_path = Path("data/approach_comparison_results.json")
    
    if results_path.exists():
        with open(results_path) as f:
            approach_results = json.load(f)
    else:
        approach_results = {}
    
    # Executive Summary
    st.markdown("## 🎯 Executive Summary")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; color: white;">
        <h3 style="color: white; margin-bottom: 1rem;">Key Finding: Multi-Factor Approach Wins</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            Our empirical analysis shows that <strong>combining multiple feature types</strong> yields the best predictions,
            not relying on any single approach like utilization alone.
        </p>
        <p style="font-size: 0.95rem; opacity: 0.9;">
            Best Model: R² = 0.885 | RMSE = 2.80 fantasy points
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Approach Comparison
    st.markdown("## 📊 Approach Comparison Study")
    st.markdown("*We tested 6 different feature approaches on held-out 2024 data*")
    
    if approach_results and 'approach_comparison' in approach_results:
        comparison = approach_results['approach_comparison']
        
        # Create comparison table
        comparison_data = []
        for name, metrics in comparison.items():
            comparison_data.append({
                'Approach': name,
                'Test RMSE': metrics['rmse'],
                'Test R²': metrics['r2'],
                'Features': metrics['n_features'],
                'Rank': metrics['rank']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Rank')
        
        # Display with highlighting
        st.dataframe(
            comparison_df.style.highlight_min(subset=['Test RMSE'], color='lightgreen')
                               .highlight_max(subset=['Test R²'], color='lightgreen'),
            use_container_width=True,
            hide_index=True
        )
        
        # Visual comparison
        import plotly.express as px
        
        fig = px.bar(
            comparison_df.sort_values('Test RMSE'),
            x='Approach',
            y='Test RMSE',
            color='Test R²',
            color_continuous_scale='RdYlGn_r',
            title='Prediction Error by Approach (Lower = Better)'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Utilization Analysis
    st.markdown("## 🔍 Is Utilization the Best Approach?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### The Honest Answer: No, But It Helps")
        
        st.markdown("""
        Our empirical analysis reveals:
        
        - **Utilization alone**: R² = 0.279 (5th out of 6 approaches)
        - **Utilization contribution**: Only 2.9% of total feature importance
        - **Best single category**: Season-long features (R² = 0.883)
        
        **However**, utilization metrics are still valuable because they:
        1. Capture opportunity share (targets, carries, snaps)
        2. Predict future role changes before stats reflect them
        3. Identify breakout candidates early
        """)
    
    with col2:
        st.markdown("### Feature Importance Breakdown")
        
        if approach_results and 'feature_importance' in approach_results:
            importance = approach_results['feature_importance']
            
            # Create pie chart
            import plotly.graph_objects as go
            
            labels = list(importance.keys())[:8]
            values = [importance[k] for k in labels]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo='label+percent'
            )])
            fig.update_layout(
                title='Top Feature Importance',
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # What Actually Works
    st.markdown("## ✅ What Actually Works Best")
    
    st.markdown("""
    Based on our analysis of 27,000+ player-games across 5 NFL seasons, the most predictive factors are:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #f0fdf4; border-radius: 12px; padding: 1rem; border-left: 4px solid #22c55e;">
            <h4 style="color: #166534;">🏆 #1: Position Context</h4>
            <p style="font-size: 0.9rem; color: #166534;">
                <strong>81.6% importance</strong><br>
                Position rank within the league is the strongest predictor.
                Elite players consistently produce.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; border-radius: 12px; padding: 1rem; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e;">🏆 #2: Age & Durability</h4>
            <p style="font-size: 0.9rem; color: #92400e;">
                <strong>8.7% importance</strong><br>
                Age curves and games played projections
                significantly impact season-long value.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #eff6ff; border-radius: 12px; padding: 1rem; border-left: 4px solid #3b82f6;">
            <h4 style="color: #1e40af;">🏆 #3: Historical Performance</h4>
            <p style="font-size: 0.9rem; color: #1e40af;">
                <strong>5.9% importance</strong><br>
                Rolling averages and recent trends
                capture current form and consistency.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Our Recommended Approach
    st.markdown("## 🎯 Our Recommended Approach")
    
    st.markdown("""
    <div style="background: #f8fafc; border-radius: 16px; padding: 1.5rem; border: 2px solid #e2e8f0;">
        <h3 style="color: #1e293b;">Multi-Factor Ensemble Model</h3>
        <p style="color: #64748b; margin-bottom: 1rem;">
            We combine the strengths of multiple approaches into a unified prediction system:
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div>
                <strong>📊 Position & Context (81.6%)</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #64748b;">
                    <li>Position rank within league</li>
                    <li>Team offensive context</li>
                    <li>Role stability</li>
                </ul>
            </div>
            <div>
                <strong>📈 Season-Long Factors (8.7%)</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #64748b;">
                    <li>Age/decline curves</li>
                    <li>Games played projection</li>
                    <li>Injury probability</li>
                </ul>
            </div>
            <div>
                <strong>📉 Historical Performance (5.9%)</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #64748b;">
                    <li>Rolling 3/5 week averages</li>
                    <li>Consistency score</li>
                    <li>Boom/bust range</li>
                </ul>
            </div>
            <div>
                <strong>🎯 Utilization & Matchups (3.8%)</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #64748b;">
                    <li>Target/rush share</li>
                    <li>Opponent defense rank</li>
                    <li>Vegas implied totals</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Utilization Score: Role in the Prediction Pipeline
    st.markdown("## 🔬 How Utilization Score Powers Predictions")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; color: #e2e8f0;">
        <h3 style="color: #00f5ff; margin-bottom: 0.75rem;">Utilization Score: The Prediction Engine for RB, WR & TE</h3>
        <p style="font-size: 1rem; margin-bottom: 0.75rem;">
            For <strong>Running Backs</strong>, <strong>Wide Receivers</strong>, and <strong>Tight Ends</strong>,
            our model doesn't predict fantasy points directly.  Instead, it first predicts each player's
            <strong>future Utilization Score</strong> (a 0-100 measure of opportunity), then converts
            that predicted utilization into fantasy points using a separate position-specific model.
        </p>
        <p style="font-size: 0.9rem; opacity: 0.85;">
            This two-step approach captures the insight that <em>opportunity drives production</em> —
            a player who commands a large share of their team's touches, targets, and red-zone work
            is likely to score fantasy points regardless of week-to-week efficiency variance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("### Prediction Pipeline by Position")

    col_pipe1, col_pipe2 = st.columns(2)

    with col_pipe1:
        st.markdown("""
        <div style="background: #f0fdf4; border-radius: 12px; padding: 1.25rem; border: 2px solid #22c55e;">
            <h4 style="color: #166534; margin-bottom: 0.75rem;">RB / WR / TE — Utilization-First Pipeline</h4>
            <div style="font-family: monospace; font-size: 0.85rem; color: #1e293b; line-height: 1.8;">
                <div style="background: #dcfce7; padding: 0.4rem 0.75rem; border-radius: 6px; margin-bottom: 0.4rem;">1. Calculate player's historical utilization components</div>
                <div style="text-align: center; color: #22c55e; font-weight: bold;">&#8595;</div>
                <div style="background: #dcfce7; padding: 0.4rem 0.75rem; border-radius: 6px; margin-bottom: 0.4rem;">2. Build lagged & rolling utilization features (lag1-4, roll3/5/8)</div>
                <div style="text-align: center; color: #22c55e; font-weight: bold;">&#8595;</div>
                <div style="background: #bbf7d0; padding: 0.4rem 0.75rem; border-radius: 6px; margin-bottom: 0.4rem; font-weight: 600;">3. Ensemble model predicts <em>future utilization</em> (1w, 4w, 18w)</div>
                <div style="text-align: center; color: #22c55e; font-weight: bold;">&#8595;</div>
                <div style="background: #bbf7d0; padding: 0.4rem 0.75rem; border-radius: 6px; font-weight: 600;">4. Conversion model maps predicted utilization &#8594; fantasy points</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_pipe2:
        st.markdown("""
        <div style="background: #eff6ff; border-radius: 12px; padding: 1.25rem; border: 2px solid #3b82f6;">
            <h4 style="color: #1e40af; margin-bottom: 0.75rem;">QB — Dual-Track Selection</h4>
            <div style="font-family: monospace; font-size: 0.85rem; color: #1e293b; line-height: 1.8;">
                <div style="background: #dbeafe; padding: 0.4rem 0.75rem; border-radius: 6px; margin-bottom: 0.4rem;">1. Train <strong>both</strong> a utilization model and a direct fantasy-points model</div>
                <div style="text-align: center; color: #3b82f6; font-weight: bold;">&#8595;</div>
                <div style="background: #dbeafe; padding: 0.4rem 0.75rem; border-radius: 6px; margin-bottom: 0.4rem;">2. Evaluate each on held-out validation data</div>
                <div style="text-align: center; color: #3b82f6; font-weight: bold;">&#8595;</div>
                <div style="background: #bfdbfe; padding: 0.4rem 0.75rem; border-radius: 6px; font-weight: 600;">3. Automatically select whichever model performs better</div>
            </div>
            <p style="font-size: 0.8rem; color: #64748b; margin-top: 0.75rem;">
                QB play volume and rushing involvement make utilization less reliable as the sole
                predictor, so the system picks the best approach per training run.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Position-specific component breakdowns
    st.markdown("### What Goes Into Each Position's Utilization Score")
    st.markdown("*Each component is normalized to a 0-100 scale. Weights are optimized from training data.*")

    from src.app_data import get_utilization_weights_merged
    util_weights = get_utilization_weights_merged()

    # Human-readable descriptions for each component
    component_descriptions = {
        "snap_share": "Percentage of the team's offensive snaps played",
        "rush_share": "Share of team rushing attempts",
        "target_share": "Share of team passing targets",
        "redzone_share": "Share of team red-zone opportunities",
        "touch_share": "Combined carries + receptions as share of team touches",
        "high_value_touch": "Rushes inside the 10-yard line and targets 15+ air yards",
        "air_yards_share": "Share of team total air yards (route depth)",
        "redzone_targets": "Red-zone target involvement",
        "route_participation": "Routes run as share of team pass plays",
        "inline_rate": "Usage as inline blocker vs. pass-catching role",
        "dropback_rate": "Pass attempts as share of total team plays",
        "rush_attempt_share": "Designed runs and scrambles",
        "redzone_opportunity": "Red-zone scoring opportunity (TD proxy)",
        "play_volume": "Total plays (pass + rush) normalized across the league",
    }

    pos_colors = {
        "RB": ("#22c55e", "#f0fdf4", "#166534", "#dcfce7"),
        "WR": ("#8b5cf6", "#f5f3ff", "#5b21b6", "#ede9fe"),
        "TE": ("#f59e0b", "#fffbeb", "#92400e", "#fef3c7"),
        "QB": ("#3b82f6", "#eff6ff", "#1e40af", "#dbeafe"),
    }

    tab_rb, tab_wr, tab_te, tab_qb = st.tabs(["RB", "WR", "TE", "QB"])

    for tab, pos in zip([tab_rb, tab_wr, tab_te, tab_qb], ["RB", "WR", "TE", "QB"]):
        with tab:
            accent, bg_light, text_dark, bg_bar = pos_colors[pos]
            weights = util_weights.get(pos, {})
            is_primary = pos != "QB"

            role_label = "Primary prediction target" if is_primary else "Dual-track (selected automatically)"
            role_icon = "&#9679;" if is_primary else "&#9675;"

            st.markdown(f"""
            <div style="background: {bg_light}; border-radius: 12px; padding: 1.25rem; border-left: 4px solid {accent}; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <h4 style="color: {text_dark}; margin: 0;">{pos} Utilization Score</h4>
                    <span style="background: {bg_bar}; color: {text_dark}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                        {role_icon} {role_label}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: {text_dark}; margin-bottom: 1rem;">
                    {"The ensemble predicts future utilization, then a dedicated conversion model translates that into fantasy points." if is_primary else "The system trains both a utilization model and a direct fantasy-points model, then picks whichever validates better."}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show each component as a weighted bar
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for comp, w in sorted_weights:
                pct = w * 100
                desc = component_descriptions.get(comp, comp.replace("_", " ").title())
                st.markdown(f"""
                <div style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.2rem;">
                        <span style="font-weight: 600; color: #1e293b;">{comp.replace('_', ' ').title()}</span>
                        <span style="color: {accent}; font-weight: 700;">{pct:.0f}%</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: {accent}; width: {pct * 3.33}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.15rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Takeaways
    st.markdown("## 💡 Key Takeaways")

    if approach_results and 'key_findings' in approach_results:
        for finding in approach_results['key_findings']:
            st.markdown(f"- {finding}")

    st.markdown("""
    ### Bottom Line

    **Don't rely on any single approach.** The best fantasy predictions come from combining:

    1. **Who the player is** (position rank, historical production)
    2. **How they're being used** (utilization, opportunity share)
    3. **What's the context** (matchup, weather, Vegas lines)
    4. **What's the outlook** (age, injury risk, games projection)

    Our system integrates all of these factors, weighted by their empirical predictive power.
    For RB, WR, and TE, the utilization score is the **primary prediction target** — the model
    first forecasts how a player will be used, then translates that usage into fantasy points.
    """)

    st.markdown("---")

    # Technical Details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Model Architecture
        
        - **Algorithm**: Gradient Boosting (best single model) or Ensemble Stack
        - **Training Data**: 2020-2023 NFL seasons (21,817 player-games)
        - **Test Data**: 2024 NFL season (5,658 player-games)
        - **Features**: 15-50 carefully selected (89% reduction from 140 raw features)
        - **Validation**: Time-series cross-validation with purging
        
        ### Feature Engineering Pipeline
        
        1. **Correlation Filtering**: Remove features with >0.95 correlation
        2. **Stability Selection**: 30 bootstrap samples, 40% threshold
        3. **Permutation Importance**: Model-agnostic feature ranking
        4. **Optimal Feature Count**: CV with 1-SE rule
        
        ### Overfitting Prevention
        
        - Train/test gap monitoring (target: <30%)
        - Regularization tuning (Ridge α=100, tree depth=6)
        - Feature reduction (140 → 15 features)
        - Adversarial validation for distribution shift detection
        """)


if __name__ == "__main__":
    main()
