#   
#   
# Always keep this updated based on measurable progress:   
  
```
1. SERVED WEEKLY PPR
   Status ........ LIVE (unblended model; served through the pace blend, row 2)
   Metric ........ MAE 4.356 on 2025 selection set (5,824 rows); 4.44 OOF 2024–26
   Open .......... fails the selection rule vs the live blend (10-08: G1, RB/WR guardrails)

2. SERVED + BLEND  (pace blend, 09-17)
   Status ........ LIVE since 09-17 (src/predict.py, kappa=3)
   Metric ........ MAE 4.266 on 2025 selection set (5,824 rows, raw outcomes)
   Open .......... incumbent; stays in production under the selection rule (10-08)

3. ROLLING-3 BASELINE
   Status ........ BASELINE
   Metric ........ MAE 2.491 (Plan A scale); 4.475 on 2025 selection set
   Open .......... none

4. PLAN A
   Status ........ RESEARCH; not selected (10-08)
   Metric ........ corrected: MAE 2.426 (Plan A scale, -0.065 vs rolling-3); 4.268 on 2025 selection set
   Open .......... ties the live blend (+0.002); fails G1 and RMSE/bias guardrails

5. PLAN A + PLAN B ARM
   Status ........ DROPPED from selection (10-08); results invalid (same leak)
   Metric ........ none valid; 2.346 used same-game team totals
   Open .......... no serving path for the Plan B arm

6. PLAN B  (shares only)
   Status ........ RESEARCH
   Metric ........ beats rolling-3 on share MAE (4 targets)
   Open .......... no points-level standalone result

7. SIM MARGINALS
   Status ........ LIVE
   Metric ........ CRPS 3.09; coverage 0.49 / 0.79 (50% / 80%)
   Open .......... none

8. CORRELATION LAYER
   Status ........ LIVE, UNPROVEN
   Metric ........ stack CRPS 9.39 vs 9.48 (not significant)
   Open .......... needs more 2026 weeks

9. GAME OUTCOME MODELS
   Status ........ LIVE
   Metric ........ not re-evaluated
   Open .......... accuracy review

```
