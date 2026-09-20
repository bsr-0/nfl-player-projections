# Follow-Up Process Monitoring

**Started:** 2026-09-19 ~11:12 PM  
**User Action:** Run follow-up processes

## Processes Running

### 1. Retraining Models (PID: 42820)
**Command:** `python3 src/models/train.py --walk-forward --test-season 2025 --skip-quality-gate`  
**Log:** `data/experiments/retrain_production.log`  
**Status:** 🟢 Running (5+ seconds, consuming 680MB RAM)

**What it does:**
- Trains QB/RB/WR/TE position models on seasons < 2025
- Tests on 2025 with walk-forward validation
- Saves retrained artifacts with corrected Vegas features
- Expected duration: 2-4 hours

**Current activity:**
- Fetching weekly data (with network retries)
- Building training/test datasets
- Will soon start model training

### 2. Benchmarking Weekly Metrics (PID: 42821)
**Command:** `run_final_validation(positions=['QB','RB','WR','TE'], seasons=[2023,2024,2025])`  
**Log:** `data/experiments/benchmark_production.log`  
**Status:** 🟢 Running (5+ seconds, consuming 430MB RAM)

**What it does:**
- Validates predictions for 2023/2024/2025 with FINAL_CONFIG
- Generates row-level predictions vs actual for each player
- Computes: MAE, RMSE, R², Spearman ρ, bias
- Includes baselines: naive, prior-season, Vegas-implied
- Expected duration: 1-3 hours

**Current activity:**
- Starting up, preparing data
- Will soon begin fold processing

---

## Monitoring Commands

```bash
# Real-time monitoring
tail -f data/experiments/retrain_production.log
tail -f data/experiments/benchmark_production.log

# Check process status
ps aux | grep -E "train.py|run_final_validation" | grep -v grep

# Check memory usage
top -p 42820,42821

# Check completion
tail data/experiments/retrain_production.log | grep -E "saved|complete"
tail data/experiments/benchmark_production.log | grep -E "complete|Output"
```

---

## Expected Outputs

### When Retraining Completes
- New model artifacts: `data/models/model_QB_*.joblib`, etc.
- Walk-forward metrics: RMSE, MAE, R² per fold
- Training report: `data/models/retrain_results.json`

### When Benchmarking Completes
- CSV file: `data/experiments/weekly_benchmark_vegas_corrected.csv`
- Row-level predictions: 10,000+ rows (players × seasons × baselines)
- Summary metrics per position/season

---

## Key Metrics to Compare

When complete, we can compare:

**Per-Position MAE (vs old inverted Vegas):**
- QB: ? → (improved with correct Vegas)
- RB: ? → (improved with correct Vegas)
- WR: ? → (improved with correct Vegas)
- TE: ? → (improved with correct Vegas)

**R² Improvements:**
- Overall: ? → (should improve)
- By position: ? → (should improve)
- By season: ? → (should improve)

---

## Next Steps After Completion

1. **Extract metrics** from output files
2. **Compare** new baselines vs old v9 metrics
3. **Quantify** improvement from Vegas fix
4. **Deploy** new models if improvement validated
5. **Update** dashboards with new baselines

---

## Process Status Updates

Run these to check progress:

```bash
# Show last 50 lines of each log
echo "=== RETRAIN ===" && tail -50 data/experiments/retrain_production.log
echo "=== BENCHMARK ===" && tail -50 data/experiments/benchmark_production.log
```

---

**Monitoring started:** 2026-09-19 23:12 UTC  
**Status:** Both processes active and running
