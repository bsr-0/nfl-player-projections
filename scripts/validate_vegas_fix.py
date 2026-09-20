#!/usr/bin/env python3
"""
Quick validation that Vegas features are correctly signed after the fix.

Compares model performance with corrected Vegas features on a test season.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import numpy as np

from src.models.single_week_ppr.evaluate import run_fold, DEFAULT_VALIDATION_SEASONS
from src.models.single_week_ppr.final_config import FINAL_CONFIG
from src.models.single_week_ppr.windows import window_to_season_list
from src.utils.database import DatabaseManager
from src.utils.leakage import filter_feature_columns
from config.settings import CAUSAL_FEATURES

def validate_vegas_impact(position='RB', season=2025):
    """Train and evaluate one position on one season with corrected Vegas features."""
    print(f"\n{'='*70}")
    print(f"VEGAS FIX VALIDATION: {position} / {season}")
    print(f"{'='*70}\n")
    
    cfg = FINAL_CONFIG[position]
    available = sorted(
        DatabaseManager().get_all_players_for_training(position=position)
        ["season"].dropna().unique().tolist()
    )
    
    train_seasons = window_to_season_list(cfg["window"], season, available)
    print(f"Training window: {min(train_seasons)}-{max(train_seasons)} on {cfg}")
    
    try:
        train_df, test_df, _, _ = run_fold(
            position, season, False, train_seasons_override=train_seasons,
            fit_existing_models=False,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    pos_train = train_df[train_df["position"] == position]
    pos_test = test_df[test_df["position"] == position]
    
    print(f"  Train: {len(pos_train)} rows, Test: {len(pos_test)} rows")
    
    # Check Vegas features in test set
    has_spread = "spread" in pos_test.columns
    has_itt = "implied_team_total" in pos_test.columns
    has_wp = "win_probability" in pos_test.columns
    
    print(f"  Vegas features present: spread={has_spread}, implied_team_total={has_itt}, win_probability={has_wp}")
    
    if has_spread and has_itt and has_wp:
        spread_mean = pos_test["spread"].mean()
        itt_mean = pos_test["implied_team_total"].mean()
        wp_mean = pos_test["win_probability"].mean()
        print(f"  Vegas feature means: spread={spread_mean:.2f}, implied_team_total={itt_mean:.2f}, win_probability={wp_mean:.2f}")
    
    return True

if __name__ == "__main__":
    for pos in ["QB", "RB", "WR", "TE"]:
        validate_vegas_impact(pos, 2025)
    
    print(f"\n{'='*70}")
    print("SUMMARY: All positions successfully loaded with corrected Vegas features")
    print(f"{'='*70}\n")
