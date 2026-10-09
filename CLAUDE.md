#   
#   
# Always keep this updated based on measurable progress:   
  
```
1. SERVED WEEKLY PPR
   Status ........ LIVE (unblended model; served through the pace blend, row 2)
   Metric ........ MAE 4.44 (OOF 2024–26, raw outcomes); 4.36 on 5,826-game 2025 matched set
   Open .......... lost to Plan A at every position on matched set (10-08)

2. SERVED + BLEND  (pace blend, 09-17)
   Status ........ LIVE since 09-17 (src/predict.py, kappa=3)
   Metric ........ MAE 4.35 -> 4.24 (2025, n=6,356)
   Open .......... not re-run on raw targets; not on matched set

3. ROLLING-3 BASELINE
   Status ........ BASELINE
   Metric ........ MAE 2.491 (Plan A scale)
   Open .......... none

4. PLAN A
   Status ........ RESEARCH, LEADS SERVED
   Metric ........ MAE 2.370 (Plan A scale); 4.08 on 5,826-game 2025 matched set
   Open .......... beat served by 0.27 MAE; QB 0.81, RB 0.30, WR 0.19, TE 0.15 (all CIs exclude 0)

5. PLAN A + PLAN B ARM
   Status ........ RESEARCH
   Metric ........ MAE 2.346 (Plan A scale); -0.145 vs rolling-3, CI [-0.160, -0.129]
   Open .......... never paired vs served; best Plan A-scale MAE, run on matched set next

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
