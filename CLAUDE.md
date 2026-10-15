#   
#   
# Always keep this updated based on measurable progress:   
  
```
1. SERVED WEEKLY PPR
   Status ........ LIVE (unblended model; served through the pace blend, row 2)
   Metric ........ MAE 4.356 on 2025 selection set (5,824 rows); 4.44 OOF 2024–26
   Open .......... fails the selection rule vs the live blend (10-08); forward-test candidate (2026 wks 6-13)

2. SERVED + BLEND  (pace blend, 09-17)
   Status ........ LIVE since 09-17 (src/predict.py, kappa=3)
   Metric ........ MAE 4.266 on 2025 selection set (5,824 rows, raw outcomes)
   Open .......... incumbent; stays in production under the selection rule (10-08)

3. ROLLING-3 BASELINE
   Status ........ BASELINE
   Metric ........ MAE 2.491 (Plan A scale); 4.475 on 2025 selection set
   Open .......... none

4. PLAN A
   Status ........ RESEARCH; not selected (10-08); forward-test candidate (2026 wks 6-13)
   Metric ........ corrected: MAE 2.426 (Plan A scale, -0.065 vs rolling-3); 4.268 on 2025 selection set
   Open .......... ties the live blend (+0.002); fails G1 and RMSE/bias guardrails

5. PLAN A + PLAN B ARM
   Status ........ CLOSED (10-10); not pursued, no serving path
   Metric ........ leak-free: MAE 2.403 (Plan A scale, -0.023 vs Plan A); gain is QB passing yards
   Open .......... none (QB passing-yards-only path dropped; revisit only after the forward window)

6. PLAN B  (shares only)
   Status ........ CLOSED (10-10); not pursued
   Metric ........ beats rolling-3 on share MAE (4 targets); points result = row 5 (allocator only)
   Open .......... none

7. SIM MARGINALS
   Status ........ LIVE
   Metric ........ CRPS 3.09; cov 0.49 / 0.79 (2024-25 backtest); 2026 wks 1-4 (n=1,431): 3.20, 0.49 / 0.77
   Open .......... centred on the unblended model since 10-11 (was the blend: team/game 80% cov 0.56 / 0.52); on the site only if
                   docs/SIMULATION_FORWARD_CHECK.md Decision 2 passes (weeks 5-8, checkpoint Tue 11-03); not shown on any page yet

8. CORRELATION LAYER
   Status ........ LIVE, UNPROVEN
   Metric ........ stack CRPS 9.39 vs 9.48 (not significant); 2026 wks 1-4: 9.77 vs 9.90 (-0.13, ~2.8 se)
   Open .......... pre-registered (docs/SIMULATION_FORWARD_CHECK.md Decision 1): stack CRPS wks 5-13, KEEP iff upper 95% < 0, else REVERT (Tue 12-08)

9. GAME OUTCOME MODELS
   Status ........ CLOSED (10-11): LIVE as a market-tracking model; matches the Vegas line, no edge; no further model work
   Metric ........ walk-forward 2022-25 + 2026 wks 1-5, tuned params (1,201 games): win acc 0.673 logistic = Vegas favorite 0.673
                   (Brier 0.213 vs 0.211); margin MAE 9.645 vs line 9.558; total 10.198 vs 10.211; ridge ATS 46.9%, O/U 52.9%
   Site .......... spread/total shown as "leans vs market", neutral, with the 46.9% record; 2026 wks 1-4 backtest panel
                   (logistic 41-23 = market favorite 41-23); weeks with nothing published show labelled backtest rows
   Saved ......... `python scripts/refit_game_models.py` (7 s; joblibs gitignored; manifest data/models/game_models_manifest.json
                   with hyperparameter hash); generate_game_predictions_data.py refits when stale; training scripts now reuse
                   the recorded hyperparameters unless --tune/--tune-optuna/--default-params
   Open .......... none

10. NO-PRIOR SCALED  (unblended x 0.9218; hybrid keeps the blend for QBs)
   Status ........ PRE-REGISTERED 10-14 (docs/PREREG_NO_PRIOR_2026.md, scripts/score_no_prior_prereg.py), before wk 6
   Metric ........ exploratory, 2025 selection set: MAE 4.269 (+0.003 vs blend), hybrid 4.257 (-0.010); 0 of 17 guardrails fail;
                   under-projection of players 3+ PPG ahead of prior cut 0.43 / 0.31 pts. Not evidence (chosen after viewing 2025)
   Open .......... non-inferiority (margin 0.10, Holm k=2) + 17 guardrails + S1 on 2026 wks 6-13; `score` once after wk 13

```
