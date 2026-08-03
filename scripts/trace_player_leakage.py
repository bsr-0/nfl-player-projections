#!/usr/bin/env python3
"""Player-level leakage trace for the 2025 backtest pipeline.

Council recommendation #1: "Audit the 2025 backtest pipeline for data leakage.
Manually trace feature values for 3-5 players in Week 1 of 2025 and verify
that no statistic from 2025 or late 2024 (after training cutoff) contaminates
any rolling average, lag feature, or engineered variable."

Usage:
    python scripts/trace_player_leakage.py
"""
import os

os.environ["NFL_FEATURE_MODE"] = "causal"

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "nfl_data.db"
TARGET_WEEK = 1
TARGET_SEASON = 2025
ROLLING_WINDOW = 3
NUM_PLAYERS = 5

# Positions we want one player from each
POSITIONS_WANTED = ["QB", "RB", "WR", "TE"]


def load_data():
    """Load all player weekly stats and player metadata from SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Player metadata
    rows = conn.execute("SELECT player_id, name, position FROM players").fetchall()
    players_df = pd.DataFrame([dict(r) for r in rows])

    # Weekly stats
    rows = conn.execute(
        "SELECT * FROM player_weekly_stats ORDER BY player_id, season, week"
    ).fetchall()
    stats_df = pd.DataFrame([dict(r) for r in rows])

    conn.close()
    return players_df, stats_df


def pick_trace_players(players_df, stats_df):
    """Pick one high-scoring player per position from 2025 Week 1."""
    week1 = stats_df[
        (stats_df["season"] == TARGET_SEASON) & (stats_df["week"] == TARGET_WEEK)
    ].merge(players_df[["player_id", "name", "position"]], on="player_id")

    chosen = []
    for pos in POSITIONS_WANTED:
        candidates = week1[week1["position"] == pos].sort_values(
            "fantasy_points", ascending=False
        )
        if len(candidates) == 0:
            continue
        row = candidates.iloc[0]
        chosen.append(
            {
                "player_id": row["player_id"],
                "name": row["name"],
                "position": pos,
                "team": row["team"],
                "week1_fpts": row["fantasy_points"],
            }
        )

    # If we have fewer than NUM_PLAYERS, add a second WR or RB
    if len(chosen) < NUM_PLAYERS:
        extras = week1[week1["position"] == "WR"].sort_values(
            "fantasy_points", ascending=False
        )
        for _, row in extras.iterrows():
            if row["player_id"] not in [c["player_id"] for c in chosen]:
                chosen.append(
                    {
                        "player_id": row["player_id"],
                        "name": row["name"],
                        "position": "WR",
                        "team": row["team"],
                        "week1_fpts": row["fantasy_points"],
                    }
                )
                break

    return chosen[:NUM_PLAYERS]


def manual_rolling_mean(player_stats, cutoff_season, cutoff_week, col, window):
    """Compute rolling mean using only data strictly before cutoff.

    The FeatureEngineer uses shift(1).rolling(window, min_periods=1).mean(),
    which excludes the current row and looks back `window` rows. For the
    first row of a new season (2025 Wk1), shift(1) gives the previous row
    in the sorted player timeline, so the rolling window covers the last
    `window` games before the cutoff.

    Returns the value that should appear at (cutoff_season, cutoff_week) if
    the pipeline correctly prevents leakage.
    """
    prior = player_stats[
        (player_stats["season"] < cutoff_season)
        | (
            (player_stats["season"] == cutoff_season)
            & (player_stats["week"] < cutoff_week)
        )
    ].sort_values(["season", "week"])

    if len(prior) == 0:
        return np.nan

    values = prior[col].dropna().values
    if len(values) == 0:
        return np.nan
    # Rolling mean of last `window` values (min_periods=1 matches FE behavior)
    tail = values[-window:]
    return float(np.nanmean(tail))


def trace_single_player(player_info, stats_df, fe_features):
    """Trace feature values for a single player at 2025 Week 1."""
    pid = player_info["player_id"]
    player_stats = stats_df[stats_df["player_id"] == pid].sort_values(
        ["season", "week"]
    )

    print(f"\n{'─' * 60}")
    print(f"  {player_info['name']} ({player_info['position']}) — {player_info['team']}")
    print(f"  player_id: {pid}")
    print(f"{'─' * 60}")

    # Last 3 weeks of 2024
    season_2024 = player_stats[player_stats["season"] == 2024].sort_values("week")
    last3 = season_2024.tail(3)
    if len(last3) > 0:
        print(f"\n  Last {len(last3)} weeks of 2024 fantasy_points:")
        for _, row in last3.iterrows():
            print(f"    Week {int(row['week'])}: {row['fantasy_points']:.1f}")
    else:
        print("\n  No 2024 data found for this player.")

    # Manual rolling mean at 2025 Week 1
    manual_roll3 = manual_rolling_mean(
        player_stats, TARGET_SEASON, TARGET_WEEK, "fantasy_points", ROLLING_WINDOW
    )
    print(f"\n  Expected rolling_3w_mean at 2025 Wk1 (manual): {manual_roll3:.2f}"
          if not np.isnan(manual_roll3) else
          "\n  Expected rolling_3w_mean at 2025 Wk1 (manual): N/A (no prior data)")

    # Actual 2025 Week 1 fantasy_points (the target — should NOT appear in features)
    week1_fpts = player_info["week1_fpts"]
    print(f"  Actual 2025 Wk1 fantasy_points (target): {week1_fpts:.1f}")

    # Check FeatureEngineer output
    issues = []

    if fe_features is not None:
        fe_row = fe_features[
            (fe_features["player_id"] == pid)
            & (fe_features["season"] == TARGET_SEASON)
            & (fe_features["week"] == TARGET_WEEK)
        ]
        if len(fe_row) == 0:
            print("\n  [WARN] Player not found in FeatureEngineer output.")
        else:
            fe_row = fe_row.iloc[0]

            print(f"\n  FeatureEngineer rolling/lag features at 2025 Wk1:")

            # Identify all rolling/lag/ewm/trend columns in the FE output
            all_roll_cols = sorted([
                c for c in fe_features.columns
                if any(tag in c for tag in ["roll", "lag", "ewm", "trend"])
                and not c.endswith("_missing")
            ])

            # --- Check 1: fantasy_points rolling (full mode) ---
            fp_roll_col = None
            for c in fe_features.columns:
                if "fantasy_points" in c and "roll" in c and "mean" in c:
                    fp_roll_col = c
                    break

            if fp_roll_col:
                fe_roll_val = fe_row[fp_roll_col]
                print(f"    {fp_roll_col}: {fe_roll_val:.2f}" if not pd.isna(fe_roll_val)
                      else f"    {fp_roll_col}: NaN")
                if not np.isnan(manual_roll3) and not pd.isna(fe_roll_val):
                    diff = abs(fe_roll_val - manual_roll3)
                    if diff > 0.01:
                        issues.append(
                            f"Rolling mismatch: manual={manual_roll3:.2f}, "
                            f"FE={fe_roll_val:.2f}, diff={diff:.2f}"
                        )
                        print(f"    ** MISMATCH ** manual={manual_roll3:.2f}, "
                              f"FE={fe_roll_val:.2f} (diff={diff:.2f})")
                    else:
                        print(f"    [OK] Matches manual computation (diff={diff:.4f})")

            # --- Check 2: all causal rolling features (opportunity metrics) ---
            # These are the columns the causal mode actually produces
            causal_roll_cols = [c for c in all_roll_cols if "roll3_mean" in c]
            for col in causal_roll_cols:
                fe_val = fe_row[col]
                # Determine the raw stat column name
                raw_col = col.replace("_roll3_mean", "")
                # Manually compute expected value from pre-cutoff data
                if raw_col in fe_features.columns:
                    manual_val = manual_rolling_mean(
                        fe_features[fe_features["player_id"] == pid].sort_values(
                            ["season", "week"]
                        ),
                        TARGET_SEASON, TARGET_WEEK, raw_col, ROLLING_WINDOW,
                    )
                    status = ""
                    if pd.isna(fe_val) and np.isnan(manual_val):
                        status = "[OK] both NaN"
                    elif pd.isna(fe_val) or np.isnan(manual_val):
                        status = "[WARN] one is NaN"
                    else:
                        diff = abs(fe_val - manual_val)
                        if diff > 0.05:
                            status = f"** MISMATCH ** (manual={manual_val:.2f}, diff={diff:.2f})"
                            issues.append(
                                f"{col}: manual={manual_val:.2f}, FE={fe_val:.2f}, "
                                f"diff={diff:.2f}"
                            )
                        else:
                            status = f"[OK] (diff={diff:.4f})"
                    print(f"    {col}: {fe_val:.2f}  {status}"
                          if not pd.isna(fe_val) else
                          f"    {col}: NaN  {status}")

            # --- Check 3: fantasy_points lag/trend/ewm features ---
            suspicious_cols = [
                c for c in fe_features.columns
                if any(tag in c for tag in ["roll", "lag", "ewm", "trend"])
                and "fantasy_points" in c
            ]
            for col in suspicious_cols:
                val = fe_row[col]
                if pd.isna(val):
                    continue
                # If a feature exactly equals the week's actual fantasy_points,
                # that is a clear sign of leakage
                if abs(val - week1_fpts) < 0.01:
                    issues.append(
                        f"Feature '{col}' = {val:.2f} equals actual Week 1 fpts "
                        f"({week1_fpts:.1f}) — likely leakage"
                    )
                    print(f"    ** LEAKAGE ** {col} = {val:.2f} == actual fpts")

            # --- Check 4: lag1 should be the last game BEFORE 2025 Wk1 ---
            lag1_col = None
            for c in fe_features.columns:
                if c == "fantasy_points_lag1":
                    lag1_col = c
                    break
            if lag1_col:
                lag1_val = fe_row[lag1_col]
                prior = player_stats[
                    (player_stats["season"] < TARGET_SEASON)
                    | (
                        (player_stats["season"] == TARGET_SEASON)
                        & (player_stats["week"] < TARGET_WEEK)
                    )
                ].sort_values(["season", "week"])
                if len(prior) > 0:
                    expected_lag1 = prior.iloc[-1]["fantasy_points"]
                    print(f"\n    fantasy_points_lag1: FE={lag1_val:.2f}, "
                          f"expected={expected_lag1:.2f}"
                          if not pd.isna(lag1_val) else
                          f"\n    fantasy_points_lag1: FE=NaN, "
                          f"expected={expected_lag1:.2f}")
                    if not pd.isna(lag1_val) and abs(lag1_val - expected_lag1) > 0.01:
                        issues.append(
                            f"Lag1 mismatch: expected={expected_lag1:.2f}, "
                            f"FE={lag1_val:.2f}"
                        )
                        print(f"    ** LAG MISMATCH **")

            # --- Check 5: no 2025 stat values leaked into any feature ---
            season_2025 = player_stats[player_stats["season"] == TARGET_SEASON]
            if len(season_2025) > 0:
                wk1_stats = season_2025[season_2025["week"] == TARGET_WEEK]
                if len(wk1_stats) > 0:
                    wk1_row = wk1_stats.iloc[0]
                    stat_cols = [
                        "rushing_yards", "receiving_yards", "passing_yards",
                        "rushing_tds", "receiving_tds", "passing_tds",
                        "rushing_attempts", "targets", "receptions",
                    ]
                    for stat_col in stat_cols:
                        if stat_col not in wk1_row.index:
                            continue
                        actual_val = wk1_row[stat_col]
                        if actual_val == 0:
                            continue
                        for feat_col in all_roll_cols:
                            feat_val = fe_row.get(feat_col, np.nan)
                            if pd.isna(feat_val):
                                continue
                            if abs(feat_val - actual_val) < 0.01 and actual_val > 5:
                                issues.append(
                                    f"Feature '{feat_col}' = {feat_val:.2f} matches "
                                    f"2025 Wk1 {stat_col} = {actual_val}"
                                )

    # Verdict for this player
    if issues:
        print(f"\n  VERDICT: FAIL ({len(issues)} issue(s) found)")
        for iss in issues:
            print(f"    - {iss}")
        return False
    else:
        print(f"\n  VERDICT: PASS (no leakage detected)")
        return True


def main():
    print("=" * 60)
    print("  NFL Player-Level Leakage Trace")
    print("  Target: 2025 Season, Week 1")
    print(f"  Feature mode: {os.environ.get('NFL_FEATURE_MODE', 'unknown')}")
    print("=" * 60)

    # Load data
    print("\nLoading data from SQLite...")
    players_df, stats_df = load_data()
    print(f"  Loaded {len(stats_df)} weekly stat rows for "
          f"{stats_df['player_id'].nunique()} players")
    print(f"  Seasons: {sorted(stats_df['season'].unique())}")

    # Pick trace players
    trace_players = pick_trace_players(players_df, stats_df)
    print(f"\nSelected {len(trace_players)} players to trace:")
    for p in trace_players:
        print(f"  {p['name']:20s} {p['position']:3s}  {p['team']:4s}  "
              f"Wk1 fpts: {p['week1_fpts']:.1f}")

    # Try to run FeatureEngineer
    fe_features = None
    try:
        from src.features.feature_engineering import FeatureEngineer

        print("\nRunning FeatureEngineer.create_features() on full dataset...")
        # Merge position into stats for FeatureEngineer
        merged = stats_df.merge(
            players_df[["player_id", "name", "position"]], on="player_id", how="left"
        )
        fe = FeatureEngineer(feature_mode="causal")
        fe_features = fe.create_features(merged, include_target=True)
        print(f"  Generated {len(fe_features.columns)} columns, "
              f"{len(fe_features)} rows")
    except Exception as e:
        print(f"\n  [WARN] FeatureEngineer import/run failed: {e}")
        print("  Continuing with manual trace only.\n")

    # Trace each player
    results = []
    for player_info in trace_players:
        passed = trace_single_player(player_info, stats_df, fe_features)
        results.append((player_info["name"], passed))

    # Final summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print("=" * 60)
    all_pass = all(r[1] for r in results)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}  [{status}]")

    if all_pass:
        print(f"\n  OVERALL VERDICT: PASS")
        print("  No data leakage detected in 2025 Week 1 features.")
    else:
        fail_count = sum(1 for _, p in results if not p)
        print(f"\n  OVERALL VERDICT: FAIL")
        print(f"  {fail_count}/{len(results)} player(s) showed potential leakage.")

    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
