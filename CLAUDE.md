#   
#   
# Always keep this updated based on measurable progress:   
  
```
1. SERVED WEEKLY PPR
   Status ........ LIVE (unblended model; served through the pace blend, row 2)
   Metric ........ MAE 4.44 (OOF 2024–26, raw outcomes); 4.36 on 5,826-game 2025 matched set
   Open .......... the 10-08 loss to Plan A is void (Plan A leak, row 4)

2. SERVED + BLEND  (pace blend, 09-17)
   Status ........ LIVE since 09-17 (src/predict.py, kappa=3)
   Metric ........ MAE 4.35 -> 4.24 (2025, n=6,356)
   Open .......... not re-run on raw targets; not on matched set

3. ROLLING-3 BASELINE
   Status ........ BASELINE
   Metric ........ MAE 2.491 (Plan A scale)
   Open .......... none

4. PLAN A
   Status ........ RESEARCH, RESULTS INVALID (10-08: same-week team leak)
   Metric ........ none valid; 2.370 and 4.08 used same-game team totals
   Open .......... selector re-run without same-week team columns; pre-kickoff export

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
