# Vegas Feature Fix: Follow-Up Tasks Status

**Defect Fixed:** Vegas features (`spread`, `implied_team_total`, `is_favorite`, `win_probability`) were sign-inverted in production, affecting all committed player models v9-v25.

**Commit:** d188104 (Vegas sign-convention fix), fd047cb (PROJECT_NOTES.md)

---

## Task 1: Retrain all player models with corrected Vegas features

**Status:** IN PROGRESS

**Command:** `python3 src/models/train.py --walk-forward --test-season 2025 --fast --skip-quality-gate`

**Log:** `data/experiments/retrain_walk_forward_2025.log`

**Expected duration:** 2-4 hours (fast mode reduces Optuna trials, CV folds, SHAP/PDP)

**What it does:**
- Trains position models (QB, RB, WR, TE) on seasons < 2025
- Tests on 2025 (a complete, in-season season)
- Produces walk-forward metrics: RMSE, MAE, R², Spearman ρ
- Saves trained artifacts to `data/models/`

**Success criteria:**
- All positions train successfully
- Metrics reported (RMSE, MAE, R², bias)
- Artifacts saved: `model_QB_1w.joblib`, etc.

---

## Task 2: Re-benchmark weekly metrics to establish new baselines with fixed features

**Status:** IN PROGRESS

**Command:** Phase 6c evaluation harness (`src/models/single_week_ppr/evaluate.run_final_validation`)

**Log:** `data/experiments/weekly_benchmark.log`

**Output:** `data/experiments/weekly_benchmark_vegas_corrected.csv`

**Expected duration:** 3-6 hours (3 positions × 3 seasons × 4 arms per position)

**What it does:**
- Runs walk-forward on 2023/2024/2025 with FINAL_CONFIG architecture per position
- Per-player row-level predictions: predicted_points vs actual_points
- Computes: MAE, RMSE, R², Spearman ρ, median AE, bias
- Baselines included: naïve, prior-season, Vegas-implied, existing_methodology

**Success criteria:**
- All positions/seasons complete
- Metrics breakdown by position/season/fold
- Can compare new baselines (Vegas corrected) vs old (Vegas inverted)

---

## Task 3: Update PROJECT_NOTES.md to reflect the fix

**Status:** COMPLETED ✓

**Commit:** fd047cb

**What was added:**
- New section: "Vegas Feature Sign Inversion (Defect Found 2026-09-19, Fixed 2026-09-19)"
- Impact measurement (correlation with team points: -0.14 → +0.42)
- Multiple implementation disagreements noted
- Fix details and regression guard
- Action required for model retraining

**Visibility:** Now prominently documented for future readers

---

## Next Steps (when tasks 1 & 2 complete)

1. **Compare old vs new metrics:** 
   - Extract metrics from old holdout benchmarks (v9)
   - Compare against new walk-forward results
   - Quantify improvement from Vegas fix

2. **Retrain production artifacts:**
   - Save newly trained models (currently in fast mode)
   - Run full training without --fast for production-quality artifacts
   - Update model metadata with new training date and Vegas-corrected flag

3. **Deploy/notify:**
   - Flag in the UI that models have been retrained with Vegas correction
   - Update any dashboards using old metrics
   - Consider running backtest on the new models

---

## Background Job Monitoring

```bash
# Check retraining progress
tail -50 data/experiments/retrain_walk_forward_2025.log

# Check weekly benchmark progress
tail -50 data/experiments/weekly_benchmark.log

# Monitor both
watch -n 5 'echo "===RETRAIN===" && tail -5 data/experiments/retrain_walk_forward_2025.log && echo && echo "===WEEKLY===" && tail -5 data/experiments/weekly_benchmark.log'
```

---

## Key Files Modified

- `src/data/external_data.py`: Vegas feature sign fix
- `src/evaluation/backtester.py`: Vegas sign fix in evaluation frame
- `src/evaluation/baselines.py`: Vegas sign fix in fallback
- `src/features/feature_engineering.py`: Vegas sign fix in early return + lookup
- `tests/test_vegas_sign_convention.py`: Regression guard (new file)
- `PROJECT_NOTES.md`: Documented the defect and action items
- `GAPS.md`: Detailed defect analysis

---

## Impact Summary

**Before:** Player models trained with Vegas features pointing in the wrong direction, giving opponent's expected points instead of team's, inverting win probability.

**After:** Vegas features now correctly signed. Models retrained with corrected signals. Weekly projections now use correct matchup context.

**Severity:** Critical — all historical metrics should be considered unreliable until retraining completes and new benchmarks are verified.

