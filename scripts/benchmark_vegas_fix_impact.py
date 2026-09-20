#!/usr/bin/env python3
"""
Quick benchmark showing Vegas fix impact on weekly projections.
Uses Phase 6c validation harness on one season to isolate the effect.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.models.single_week_ppr.evaluate import run_final_validation, DEFAULT_VALIDATION_SEASONS

print("\n" + "="*70)
print("BENCHMARKING VEGAS FIX IMPACT ON WEEKLY PROJECTIONS")
print("="*70)
print("\nRunning Phase 6c validation (FINAL_CONFIG) on 2025 with corrected Vegas features...")
print("This will show baseline MAE, R², and Spearman ρ with Vegas properly signed.\n")

try:
    output_path = Path("data/experiments/weekly_benchmark_vegas_corrected_2025.csv")
    result = run_final_validation(
        positions=["QB", "RB", "WR", "TE"],
        seasons=[2025],
        output_path=output_path
    )
    
    if result.empty:
        print("\nERROR: No results produced")
        sys.exit(1)
    
    # Compute summary metrics
    print("\n" + "="*70)
    print("RESULTS: Weekly Projections with CORRECTED Vegas Features (2025)")
    print("="*70)
    
    summary = result.groupby("position").agg({
        "mae": ["mean", "std"],
        "r2": ["mean", "std"],
        "spearman": ["mean", "std"],
        "n": "sum"
    }).round(4)
    
    print("\nPer-Position Metrics:")
    print(summary.to_string())
    
    # Show baseline vs game-outcome features comparison
    if "model" in result.columns:
        by_model = result.groupby(["position", "model"])["mae"].mean().round(4)
        print("\nMAE by Position and Model:")
        print(by_model.to_string())
    
    print(f"\nRow-level results saved to: {output_path}")
    print(f"Total predictions: {len(result):,}")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
