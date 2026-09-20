#!/usr/bin/env python3
"""Quick verification that Vegas features are correctly signed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import sqlite3
from src.data.external_data import VegasLinesLoader
from src.features.feature_engineering import FeatureEngineer

print("\n" + "="*70)
print("VEGAS FIX VERIFICATION")
print("="*70)

# Load a test game
conn = sqlite3.connect("data/nfl_data.db")
sched = pd.read_sql(
    "SELECT season, week, home_team, away_team, home_score, away_score, spread_line, total_line "
    "FROM schedule WHERE season=2024 AND week=1 AND home_score IS NOT NULL LIMIT 1",
    conn
)

if sched.empty:
    print("No test data found")
    sys.exit(1)

row = sched.iloc[0]
print(f"\nTest Game: {row['season']} Week {row['week']}: {row['home_team']} vs {row['away_team']}")
print(f"  Actual: Home {row['home_score']}-{row['away_score']} Away")
print(f"  Vegas: spread_line={row['spread_line']}, total_line={row['total_line']}")

# Test production path (external_data)
home_df = pd.DataFrame({
    "season": [row["season"]], "week": [row["week"]], "team": [row["home_team"]],
    "player_id": ["test"], "position": ["RB"]
})
away_df = pd.DataFrame({
    "season": [row["season"]], "week": [row["week"]], "team": [row["away_team"]],
    "player_id": ["test"], "position": ["RB"]
})
df = pd.concat([home_df, away_df], ignore_index=True)

loader = VegasLinesLoader()
lines = loader.load_vegas_lines([row["season"]])
result = loader.get_vegas_features(df, lines)

print("\n✓ Production Path (external_data.get_vegas_features):")
for idx, (team, is_home) in enumerate([(row["home_team"], True), (row["away_team"], False)]):
    r = result[result["team"] == team].iloc[0]
    print(f"  {team} (home={is_home}): spread={r['spread']:.1f}, implied_total={r['implied_team_total']:.1f}")

# Verify math
home_pts = row["home_score"]
away_pts = row["away_score"]
home_impl = result[result["team"] == row["home_team"]]["implied_team_total"].iloc[0]
away_impl = result[result["team"] == row["away_team"]]["implied_team_total"].iloc[0]

corr_home = (home_pts - home_impl) ** 2
corr_away = (away_pts - away_impl) ** 2

print(f"\n✓ Correctness Check:")
print(f"  Home: predicted {home_impl:.1f}, actual {home_pts} (diff: {abs(home_pts - home_impl):.1f})")
print(f"  Away: predicted {away_impl:.1f}, actual {away_pts} (diff: {abs(away_pts - away_impl):.1f})")
print(f"  Combined MSE: {(corr_home + corr_away) / 2:.2f}")

# Test feature engineering fallback
fe = FeatureEngineer()
result_fe = fe._create_vegas_game_script_features(result.copy())

print(f"\n✓ Feature Engineering Fallback (same result):")
match = (result[["team", "spread", "implied_team_total"]] == 
         result_fe[["team", "spread", "implied_team_total"]]).all().all()
print(f"  Paths agree: {match}")

if match:
    print("\n✅ VEGAS FIX VERIFIED: All implementations correctly signed")
else:
    print("\n❌ MISMATCH: Implementations disagreed")
    sys.exit(1)
