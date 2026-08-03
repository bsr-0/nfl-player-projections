"""Minimal test for Draft Rankings page"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.database import DatabaseManager

st.title("Draft Rankings Test")

st.write("Step 1: Loading database...")
try:
    db = DatabaseManager()
    df = db.get_player_stats()
    st.write(f"Step 2: Got {len(df)} records")
    st.write(f"Step 3: Columns = {list(df.columns)[:5]}...")
    
    # Show sample data
    st.write("Step 4: Sample data:")
    st.dataframe(df[['name', 'position', 'fantasy_points']].head(10))
    
    # Try groupby
    st.write("Step 5: Grouping data...")
    player_stats = df.groupby(['player_id', 'name', 'position']).agg(
        avg_ppg=('fantasy_points', 'mean'),
        games=('fantasy_points', 'count'),
    ).reset_index()
    st.write(f"Step 6: Got {len(player_stats)} players")
    
    # Filter
    player_stats = player_stats[player_stats['games'] >= 6]
    st.write(f"Step 7: {len(player_stats)} players with 6+ games")
    
    # Show rankings
    player_stats['projection'] = player_stats['avg_ppg'] * 17
    player_stats = player_stats.sort_values('projection', ascending=False)
    
    st.write("Step 8: Top 20 players:")
    st.dataframe(player_stats[['name', 'position', 'avg_ppg', 'projection']].head(20))
    
    st.success("All steps completed successfully!")
    
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
