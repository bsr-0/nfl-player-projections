"""
Database Migration - SQLite to PostgreSQL
Migrates from file-based storage to proper database
"""

import pandas as pd
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class PlayerWeeklyStats(Base):
    """Player weekly statistics table."""
    __tablename__ = 'player_weekly_stats'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), index=True)
    player_name = Column(String(100), index=True)
    season = Column(Integer, index=True)
    week = Column(Integer, index=True)
    team = Column(String(3))
    position = Column(String(5), index=True)
    
    # Stats
    utilization_score = Column(Float)
    target_share = Column(Float)
    rush_share = Column(Float)
    snap_share = Column(Float)
    targets = Column(Integer)
    carries = Column(Integer)
    receptions = Column(Integer)
    
    # Predictions
    predicted_util = Column(Float)
    prediction_confidence = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_player_season', 'player_id', 'season'),
        Index('idx_season_week', 'season', 'week'),
        Index('idx_position_week', 'position', 'season', 'week'),
    )


class DatabaseMigration:
    """Migrate from SQLite/Parquet to PostgreSQL."""
    
    def __init__(self, postgres_url: str = None):
        """
        Args:
            postgres_url: Set via DATABASE_URL env var (e.g. postgresql://user:xxx@host:5432/dbname)
                         If None, uses SQLite for demo
        """
        if postgres_url:
            self.engine = create_engine(postgres_url)
        else:
            # Fallback to SQLite
            db_path = 'data/nfl_predictor.db'
            self.engine = create_engine(f'sqlite:///{db_path}')
        
        self.Session = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
        print("✅ Created database tables")
    
    def migrate_from_parquet(self, parquet_file: str):
        """Migrate data from Parquet file to database."""
        print(f"📥 Reading {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        print(f"💾 Migrating {len(df)} records...")
        df.to_sql('player_weekly_stats', self.engine, if_exists='append', index=False)
        
        print("✅ Migration complete")
    
    def migrate_from_csv(self, csv_file: str):
        """Migrate data from CSV file to database."""
        print(f"📥 Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(f"💾 Migrating {len(df)} records...")
        df.to_sql('player_weekly_stats', self.engine, if_exists='append', index=False)
        
        print("✅ Migration complete")
    
    def query_recent_players(self, n_seasons: int = 2):
        """Query recent player data."""
        current_year = datetime.now().year
        seasons = [current_year - i for i in range(n_seasons)]
        
        query = f"""
        SELECT * FROM player_weekly_stats
        WHERE season IN ({','.join(map(str, seasons))})
        ORDER BY season DESC, week DESC
        """
        
        return pd.read_sql(query, self.engine)
    
    def get_player_history(self, player_name: str):
        """Get all historical data for a player."""
        query = f"""
        SELECT * FROM player_weekly_stats
        WHERE player_name = '{player_name}'
        ORDER BY season, week
        """
        
        return pd.read_sql(query, self.engine)


if __name__ == "__main__":
    print("Database Migration Tool")
    print("=" * 60)
    
    # Create migrator (using SQLite for demo)
    migrator = DatabaseMigration()
    
    # Create tables
    migrator.create_tables()
    
    print("\n📊 To migrate from PostgreSQL:")
    print("  export DATABASE_URL='postgresql://user:PASSWORD@host:5432/nfl_predictor'")
    print("  python database_migration.py")
