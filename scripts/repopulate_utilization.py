"""Repopulate utilization_scores table from existing player_weekly_stats.

Run after fixing the column name mismatch in nfl_data_loader.py
(snap_share_pct / target_share_pct / rush_share_pct / redzone_share_pct).

Usage:
    python scripts/repopulate_utilization.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.database import DatabaseManager
from src.features.utilization_score import UtilizationScoreCalculator


def main():
    db = DatabaseManager()
    calc = UtilizationScoreCalculator()

    print("Loading player data from DB...")
    player_df = db.get_all_players_for_training(min_games=1)
    print(f"  {len(player_df)} rows loaded")

    team_df = pd.DataFrame()
    try:
        team_df = db.get_team_stats()
    except Exception:
        pass

    print("Calculating utilization scores...")
    scored = calc.calculate_all_scores(player_df, team_df)
    active = scored[scored["utilization_score"] > 0]
    print(f"  {len(active)} rows with utilization_score > 0")

    # Check that key columns are populated before inserting
    for col in ("snap_share_pct", "target_share_pct", "rush_share_pct", "redzone_share_pct"):
        non_null = active[col].notna().sum() if col in active.columns else 0
        pct = 100 * non_null / len(active) if len(active) else 0
        print(f"  {col}: {pct:.1f}% non-null ({non_null}/{len(active)})")

    print("\nInserting utilization scores...")
    count = 0
    errors = 0
    for _, row in active.iterrows():
        try:
            db.insert_utilization_score({
                "player_id": str(row["player_id"]),
                "season": int(row["season"]),
                "week": int(row["week"]),
                "utilization_score": float(row["utilization_score"]),
                "snap_share": float(row.get("snap_share_pct", 0) or 0),
                "target_share": float(row.get("target_share_pct", 0) or 0),
                "rush_share": float(row.get("rush_share_pct", 0) or 0),
                "redzone_share": float(row.get("redzone_share_pct", 0) or 0),
                "air_yards_share": float(row.get("air_yards_share_pct", 0) or 0),
            })
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Insert error: {e}")

    print(f"\nDone: {count} rows inserted, {errors} errors")

    # Spot-check: query back a few rows to verify non-zero values
    print("\nSpot-check — sample of stored utilization rows:")
    with db._get_connection() as conn:
        sample = pd.read_sql_query(
            "SELECT player_id, season, week, snap_share, rush_share, target_share, redzone_share "
            "FROM utilization_scores WHERE rush_share > 0 LIMIT 5",
            conn,
        )
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
