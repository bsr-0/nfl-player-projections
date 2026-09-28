# LLM Council — 2026-09-28

**Question:** What should the owner do next with the weekly PPR-points-per-player model, given a verified audit of how it is built, served and measured?

**Point Person triage:** Bucket 5 (genuinely new). No `council-transcript-*.md` exists in the repo or its full git history; the three transcripts the code cites (20260414-034617, 20260422-032550, 20260423-051434) were never committed, so no earlier open items could be aggregated. Defects fixed before convening: commit `4f227d6`.

## Chairman's Verdict

## Bottom Line
**Do:** Change what you serve this week: per-game pace, if a re-score today split by position, games played and cold-start status supports it, plus honest page labels. Then test the alignment fix against a control retrain of the current recipe in one overnight batch, and serve it only if it wins.
**Don't:** Don't ship per-game pace untested and don't judge it on MAE. It already over-projected cold-start players by 1.83 points in week 1, and MAE rewards numbers that run low.

## Critical Next Steps
1. **DO TODAY:** Re-score per-game pace inside the blend. No retrain is needed. Use the stored 2025 walk-forward rows and 2026 weeks 1–3, and recompute the served number with pace = Step 8 season total ÷ Step 8's own E[games] (the same definition GAPS.md scored). Keep w = g/(g+3) unchanged. Report bias, RMSE and within-position pairwise accuracy, split by position × games played (0, 1–3, 4+) × veteran/cold-start. Show MAE, but don't let it decide. Time: 3–4 hours. Success signal: three things hold. For veterans in weeks 2+, per-game pace/actual lands at 0.95–1.05 at every position (÷17 sits at 0.73–0.84). The blend's RMSE and |bias| fall at every position. Pairwise accuracy holds in the low-g and cold-start slices.
2. Ship the page fixes before Thursday's week-4 kickoff, plus per-game pace if step 1 passed. Label every projection "if he plays". Replace the identical future-week numbers with one rest-of-season figure marked "not matchup-adjusted". Label the accuracy panel with the model it actually measured: the recipe before the Vegas fix, trained through 2024. Save a timestamped snapshot of exactly what you publish. Time: 2–3 hours. Success signal: the week-4 page is live with those labels, and the snapshot matches it number for number.
3. Fix the test-DB regression, add a Tuesday scorer, and commit this action list to the repo. A test run in a clean checkout must never create data/nfl_data.db:
   - Unit tests get a temporary database.
   - Integration tests skip without creating the file.
   - The 3 tests with no guard get the same guard.

   The tests write no data. The problem is the schema-only file they leave behind, which flips 8 skip guards. The scorer runs every Tuesday and grades each frozen snapshot by position on bias, RMSE and pairwise accuracy against the season-to-date average. Time: half a day. Success signal: two back-to-back clean-checkout runs give identical results and leave no database file, and the week-4 report prints next Tuesday.
4. Build the aligned features in both the training and the serving paths, then run two retrains through 2024 in one overnight batch.
   - **Aligned recipe:** label = game t, history through game t−1, and game-t context exactly as it was known when the page is published. Record each context feature's as-of time first. Re-source or drop any feature whose stored history comes from after publication: closing lines, final injury designations, observed weather, and opponent aggregates that include game t.
   - **Control:** the current recipe with the Vegas-sign fix, unchanged.

   Score both on 2025 with isotonic on and off, using the pace you shipped. Commit step 5's ship rule to the repo before the run starts. Time: 2–3 days of code plus one overnight run. Success signal: a four-cell table (recipe × isotonic on/off) with week-clustered CIs, and the control's 2025 numbers replace the retired model's on the accuracy panel.
5. Ship or freeze. Ship the aligned recipe only if all of these hold against the control:
   - It beats the control on within-position pairwise accuracy (pooled week-clustered 95% CI excluding zero).
   - It beats the control on RMSE.
   - No position is worse.
   - |bias| ≤ 0.5 at every position.

   If it passes, retrain it through 2025 with the winning isotonic setting and serve it from week 6. Otherwise keep the current recipe for the rest of 2026. Commit the batch results either way. Time: one overnight retrain plus an hour. Success signal: the number you serve comes from a recipe that has a serving-path walk-forward on file.

## Council Convergence
- The ÷17 pace is in the wrong units for a number scored only when the player plays. The weekly model was within a quarter point of unbiased in live weeks, so pace is what pulls the served number 1.3–2.0 points low.
- MAE can't referee these changes, because on right-skewed PPR it rewards numbers that run low. Decide on bias, RMSE and within-position pairwise accuracy against the season-to-date average, which is the bar the model has to clear.
- The feature/label misalignment is the one model fix worth a retrain. Fold the calibrator fix into the same run and score isotonic on and off; don't spend a retrain on calibration alone.
- The page claims more than the model knows: one number printed beside a dozen opponents, and accuracy figures from a model you no longer serve. Both can be fixed this week without a retrain.

## Council Disagreement
**The real tradeoff:** one lean retrain vs. a controlled comparison.
**Side A:** Run a single retrain with the alignment fix, score isotonic on and off, and compare it with the stored 2025 walk-forward. Every retrain ties up your machine for hours mid-season, so one run is all the budget allows.
**Side B:** Also retrain the current recipe through 2024 as a control, and write the win rule down before the run. The recipe you serve has never been measured, so without a control you can't tell an alignment gain apart from the Vegas fix, or know whether you are replacing a better model.
**Chairman's call:** Side B. The only stored baseline predates the Vegas fix, so a one-arm test cannot show that the aligned model beats the model you actually serve.

## Blind Spots Caught in Review
- Per-game pace has already been tested once. GAPS.md scored it on week 1 of 2021–2025, with these results:
  - Veterans: RMSE improved (6.09 vs 6.23) and bias improved (+0.56 vs −1.13), but MAE got worse (4.67 vs 4.53).
  - Cold-start players: clearly worse, with MAE 4.47 vs 3.70 and bias +1.83 vs −0.18. That is a level error, which the MAE-skew argument cannot explain away.

  So ÷17 was partly cancelling a Step 8 per-game rate that runs high, at least in week 1. All five advisors reached the same "units error" diagnosis from the same brief, so their agreement is one inference, not five confirmations. A pooled re-score can pass while the low-g and cold-start rows, which lean hardest on pace, get worse. That is why step 1 splits them out.
- The leakage rule for the aligned features is "known when the page is published," not "known at kickoff." Two of the plans had no leakage check at all. Their "beats the season average" gate would pass a model trained on closing lines or final injury status.

## Kill Criteria
- **Pace.** Keep or go back to ÷17 if either of these happens:
  - In step 1, per-game pace raises RMSE or lowers pairwise accuracy at any position, or in the low-g or cold-start slices.
  - After it ships, pooled served bias lands beyond ±1.0 on two straight Tuesday scores.

  Then trace Step 8's per-game rate, including how it defines games played and its rookie prior, before trying again. If only the cold-start rows run high (bias above +1.0) while ordering holds, keep per-game pace and fix Step 8's cold-start rate at its source before step 4. Never fix it with a veterans-only branch or per-position multipliers.
- **Model.** The best recipe from step 4 must beat the season-to-date average on pairwise accuracy pooled across QB, RB and TE, with a week-clustered CI excluding zero. If it doesn't, stop model work until the offseason. State on the page that at those positions the projection ranks players no better than their season average.

## Original Question

> Council status: audit report of model for weekly predictions for PPR points per player.

## Framed Question

DECISION: What should the owner do next with the weekly PPR-points-per-player model (the number published on docs/weekly.html for each player each week) — keep serving it as is, change what is served, or fix the model first, and in what order?

CONTEXT (solo owner, personal fantasy-football projection site + private league reports; season 2026 is at week 4; the full DB and trained models live only on the owner's machine, so every change below needs a local retrain + walk-forward to validate):

How the served number is built today:
- served = w × weekly_model + (1 − w) × pace, with w = g/(g+3), g = games the player has played this season. Before kickoff (g=0) it is 100% pace.
- pace = the "Step 8" season-total projection / 17. That season total is E[PPR per game | plays] × E[games played], so pace already bakes in expected missed games. Evaluation (and the served number's documented meaning) is "points given the player plays": only players who actually played are scored.
- weekly_model = per-position stacked ensemble (RF + XGBoost + LightGBM + Ridge → RidgeCV meta) → isotonic calibration → inverse log1p with smearing.

Verified measurements (all reproduced from committed artifacts during this audit):
1. 2025 serving-path walk-forward (6,356 player-weeks, models trained through 2024): served blend MAE 4.24, bias −0.26; raw weekly model MAE 4.35, bias +0.33; pace alone MAE ~4.45, bias −1.50. Pace/actual ratio by position: QB 0.73, RB 0.74, TE 0.82, WR 0.84 (the pace is 16–28% low, position-dependent).
2. Live 2026: week 1 MAE 4.76, bias −2.02 (n=356; served ≈ pace; raw model alone bias −0.14, MAE 4.86; QB pace bias −4.8). Week 2 MAE 4.29, bias −1.33 (raw model alone MAE 4.61, bias −0.23).
3. Against simple baselines on 2025 rows with ≥3 prior games (n=4,634, week-clustered bootstrap 95% CIs): beats trailing-3-game average by 0.22 MAE [0.15, 0.29]; TIES the player's season-to-date average on MAE (+0.006 [−0.03, +0.05]); beats it on squared error.
4. Start/sit value: pairwise ordering accuracy within position-week among start/sit-relevant players, served model vs season-to-date average — QB 0.547 vs 0.554, RB 0.656 vs 0.658, TE 0.562 vs 0.576 (no edge), WR 0.612 vs 0.583 (+0.029, CI [0.007, 0.052]). Among each week's top-12 QBs by projection, rank correlation with outcome was −0.05.
5. Resolution: across all 6,356 2025 predictions the raw model produced only 41/62/70/71 distinct values (QB/RB/TE/WR). In 2026 week 4, the top 20% of WRs (55 players) share 14 distinct values; three elite WRs are tied at exactly 22.32. This is consistent with the isotonic calibrator (a step function). Its enable-gate was in-sample (could essentially never reject) — fixed in code this session, takes effect at next retrain.
6. Future weeks (2026 weeks 4–18): the raw model's number is identical across all 15 weeks for 666 of 685 players (97%), despite a median of 12 different opponents each — zero opponent sensitivity for unplayed weeks; the page still lists the different opponents.
7. Feature/target alignment (verified by running the production feature functions on a toy frame): the label is the NEXT game's PPR (row t → game t+1), but every history feature is lagged one game (row t uses games ≤ t−1). So at serving the player's most recent game never enters the model. Matchup/context features (Vegas implied total, spread, opponent defense, weather, injury status) in training describe game t, while the label is game t+1; at serving they describe the target game — a train/serve semantic mismatch that should attenuate any matchup signal.
8. Published-accuracy integrity: the weekly page was showing the PRE-blend run's accuracy (QB/RB/WR bias +0.9/+0.6/+0.5 instead of −0.6/−0.2/−0.0; "loses to a blended heuristic" instead of beating it by 0.9%) because the artifact was chosen by file name — fixed this session. Also fixed: the serving-path backtester would score production models (trained through 2025) on 2025 in-sample while labeling it "unseen"; numpy booleans were serialized as truthy strings "False".
9. Staleness: the served models were retrained 2026-09-25 (after a Vegas-sign correction, now trained through 2025). The newest serving-path walk-forward is from 2026-09-17 on models trained through 2024, before that retrain — the currently served recipe has no serving-path measurement.
10. Process: code cites three prior council transcripts (2026-04-14, 04-22, 04-23) that are not in the repo, so earlier action items cannot be tracked. A clean test run creates data/nfl_data.db (regression of a prior audit finding), after which 8 DB-dependent tests stop skipping and fail.

WHAT'S AT STAKE: every week the owner (and league-mates) read these numbers for start/sit and lineup decisions. The measured evidence says the served number is roughly as good as a season-to-date average for decisions at QB/RB/TE, slightly better at WR, and biased low by ~1.3–2 points per player in live 2026 weeks. Fixing the alignment (#7) or calibration (#5) requires retraining and a fresh walk-forward on the owner's machine; each retrain + walk-forward costs hours. The season is in progress, so changes land week by week.

**Orchestrator addendum** (given to the peer reviewers and the chairman, not the advisors: a prior measurement the framing omitted):

GAPS.md, 2026-09-05, scored per-game pace (season total ÷ Step 8's own E[games]) against the shipped ÷17 pace on 2021–2025 WEEK 1 ONLY. Veterans (n=1,428): per-game MAE 4.67 vs 4.53, bias +0.56 vs −1.13, RMSE 6.09 vs 6.23, R² 0.402 vs 0.375. Cold-start players (n=208): per-game MAE 4.47 vs 3.70, bias +1.83 vs −0.18. The explanation recorded there: week 1 is a below-average usage week, so ÷17's games discount accidentally cancels a season rate that over-predicts week 1. Per-game pace has never been tested for weeks 2+ or inside the g/(g+3) blend.

**Orchestrator addendum 2** (given to the chairman only; measured after the advisors responded, and it corrects one advisor claim): a full test run with read-only instrumentation of sqlite3.connect shows 16 test files open data/nfl_data.db, and every statement they execute on it is CREATE TABLE/INDEX IF NOT EXISTS (1,419 statements; zero INSERT/UPDATE/DELETE). Tests do not write data into the real database. The harm is narrower: in a checkout without the database they leave a schema-only file behind, which flips the `skipif(not DB_PATH.exists())` guards so 8 integration tests fail on the next run (3 of those 8 have no DB guard at all).

## Advisor Responses

### The Contrarian

Don't keep it as is, and don't retrain first.

**The pace isn't "biased low". It's a units error.** It equals E[PPR|plays] × E[games]/17, but it's only scored when the player plays. The 0.73–0.84 ratios match about 12–14 expected games ÷ 17. The damage is worst where w leans on pace: week 1 (QB −4.8), backups and players coming back from injury. Those are the hardest start/sit calls. Serve the per-game component instead. Don't patch it with per-position multipliers.

**The 2025 blend "win" is two errors cancelling out.** The raw model before the Vegas fix was +0.33, pace was −1.50, and the blend came out at −0.26. In live 2026 the raw model is close to unbiased, so the blend just imports pace's shortfall. The "3" in g/(g+3) has only ever been tested against the broken pace. Re-fit it on the stored walk-forward components. That's arithmetic, not a retrain.

**Your yardstick rewards the bug.** PPR is right-skewed, and MAE is minimized by the median. So a projection shaded 1–2 points low can "beat" an unbiased one, which is exactly the live pattern. Smearing and Step 8 both target a mean. Judge on RMSE and pairwise ordering, or the pace fix can look like a regression and get reverted.

**The alignment fix (#7) is the real fix and also the most dangerous.** Once training features describe the target game, anything not known at publish time becomes leakage that the walk-forward will reward. Examples: final injury status, observed weather, closing lines. Distrust any big jump until the as-of timestamps are audited.

**Order:**
1. Fix the test-DB regression. A test suite that breaks itself can't gate anything.
2. Fix the pace. Collapse the future weeks that show one number against different opponents. Label the accuracy panel with the retired model it actually measured.
3. Do one retrain with the alignment fix, saving both calibrated and uncalibrated outputs (a free ablation). Commit in advance that it must beat the season-to-date average on QB/RB/TE pairwise ordering (top-12 QB ρ is −0.05). If it doesn't, say so on the page.
4. Commit this council's action list to the repo.

### The First Principles Thinker

Wrong question. "Keep, change, or fix" assumes you have a weekly model. You don't.

A weekly number can only beat a season average by using what's new this week: the matchup and the most recent game. Your pipeline can use neither. The label is game t+1, but history stops at t−1 and context describes game t. The latest game never enters, and matchup and injury features are attached to the wrong game. The symptoms fit: 97% of players get one number against a dozen different opponents, it ties the season-to-date average, and top-12 QBs show a −0.05 rank correlation with outcomes. In practice it's a season average with extra steps.

Second error: the blend averages two different quantities. Pace is points per *scheduled* week, with absences baked in. The model is points *given he plays*. That mismatch, not the model, causes your live −1.3 to −2.0 bias. Pace carries 100% of the weight in week 1, 50% at g=3 and still 23% at g=10.

Order:

1. **This week, no retrain.** Set pace to the E[PPR/game | plays] factor itself, not season total ÷ 17, and confirm the 0.73–0.84 pace/actual ratios move to about 1.0. Stop printing opponents next to numbers that ignore them, or label the numbers "not matchup-adjusted." Score each week's numbers as frozen before kickoff. For a model trained through 2025, live 2026 is the only out-of-sample test there is.

2. **One batched retrain.** One row = one game. Features = everything knowable before that game: history through the previous game, context of *this* game. Label = this game. That removes both misalignments. Fold in the calibrator fix, but never spend a retrain on calibration alone: adding resolution to a number that ignores matchups is polish.

3. **Change the pass/fail bar.** Judge on within-position pairwise ordering instead of MAE, against a challenger that needs no training: w·season-to-date average + (1−w)·corrected pace. At any position where the model can't beat it with a CI that excludes zero, serve the challenger.

### The Statistician

The pace bias is the finding I'd bet on, because it's a units error rather than a noisy estimate. Season total / 17 is expected points per scheduled week, with zeros for missed games included, but you score it only on games the player plays. A ratio of 0.73–0.84 is what 12–14 expected games out of 17 produces, and the same shortfall shows up in both live weeks. It also does more than shift every number down. It penalizes injury-prone players for games they will play, it skews flex calls across positions, and it still carries 50% weight at week 4.

**Fix it first; no model retrain is needed.** Divide by E[games] instead of 17, re-score the 2025 walk-forward, and check that the pace/actual ratio is about 1.0 at each position. If it isn't, Step 8 has a second bug. Refit the blend's "3" on that 2025 data, never on 2026. The 2026 data is 2–3 weeks whose players share the same league-wide scoring swings, not ~700 independent rows.

**Choose what the number means before choosing the metric.** PPR is right-skewed, so MAE rewards forecasting the median, which sits below the mean. If the pace fix nudges MAE up, don't revert it. Lead with bias by position, RMSE, and pairwise start/sit accuracy against the season-to-date average.

On that metric the ensemble hasn't earned its complexity. It shows no edge at QB, RB or TE, and WR's +0.029 (z≈2.5) only just survives a correction for testing four positions. The top-12 QB −0.05 has a standard error of about 0.07 and no baseline to compare against, so ignore it.

The test is partly stacked against the model, though. Isotonic calibration squeezes it to 41–71 distinct values, and because it is monotone it can only merge ranks, never improve them. Score the pre-calibration outputs (free if they are stored). That measures how much ranking edge the calibration erased. It also shows whether the 97% of players whose future-week number never changes with the opponent reflects calibration flattening or matchup features that carry no signal (#7).

**Then do two retrains through 2024:**
1. The current recipe, because the model being served has no out-of-sample measurement.
2. The recipe with the aligned features.

Turn isotonic and the pace fix on and off at scoring time, which gives eight cells from those two retrains. Write down the win criterion before looking; 2025 has been reused for so many decisions that it is now validation data, not a clean test. Ship the alignment fix only if it beats the season-to-date average.

### The Outsider

The explanation for your bias is already in your own context, but nobody has connected it. Pace = season total ÷ 17, and that total already multiplies by expected games played. Then you score it only on players who played. So pace answers "points per scheduled week," while the page promises "points if he plays." The pace/actual ratio should therefore sit near E[games]/17, and your measured 0.73–0.84 looks like exactly that discount. Week 1 was 100% pace, with bias −2.02; the raw model's was −0.14. This is a units error in the blend, not a modeling problem, and it needs no retrain: serve E[PPR per game | plays] as pace. First confirm that E[games]/17 by position matches those ratios, then ship it before week 5.

Now read the page the way a league-mate does. Three elite WRs sit at exactly 22.32. Numbers are shown to two decimals when they typically miss by 4+ points. Each player gets one identical projection for 15 future weeks, printed next to a dozen different opponents. To someone who doesn't know the history, that looks broken, or like the model claims a precision and a sense of matchups it doesn't have. Also do these this week:
- round to whole points;
- print "if he plays" on the page;
- hide future weeks, or label them "season pace, not matchup-adjusted."

Next is the thing any reader would find absurd once told: the forecast never sees the player's most recent game. That is the retrain worth your hours. Bundle it with the calibrator fix so you do one retrain and one walk-forward on the exact recipe you serve. The recipe you serve now has never been measured.

Until that walk-forward beats "his average so far" at QB/RB/TE, show the season average next to the projection. People making start/sit calls should know that the complex model's number currently does no better than the simple average.

### The Executor

Order: fix what's served before Thursday, run one gated retrain after that, and freeze the model if the retrain fails.

**Do the pace fix first.** Its value shrinks each week. Pace is half of the week-4 number (w = 3/6) and about a quarter of it by week 10. Pace is the season total ÷ 17, which bakes in missed games, but accuracy is scored only on games the player plays. That is the likeliest cause of the −1.5 to −2 point bias.

**Judge every change on squared error and on per-position pairwise start/sit accuracy, not on MAE.** On right-skewed PPR, MAE barely punishes numbers that run low. In week 1, pace was 2 points low and still beat the nearly unbiased model on MAE.

**Before Thursday's kickoff (no retrain):**
1. Serve pace = E[PPR/game | plays]. Step 8 already computes this rate before multiplying by E[games]; if it isn't saved, rerun only Step 8. The fix adds no fitted parameters, so re-scoring the committed 2025 walk-forward rows plus 2026 weeks 1–3 is valid evidence. Ship it if it beats today's blend on those measures. Then rebuild the page's accuracy section from those same rows.
2. For future weeks, show one rest-of-season figure. Right now the page shows 15 identical numbers next to 15 different opponents.
3. Write a script that runs every Tuesday and scores last week's served numbers by position. That is the only out-of-sample evidence the models retrained on 09-25 will ever get.

**This week (one overnight run):**
4. First fix the test database leak. A test suite that creates data/nfl_data.db may be writing to your real database.
5. Do one retrain for #7:
   - Set label = game t, the same game the context features describe.
   - Keep history features at games ≤ t−1.
   - At serving, build the row for the upcoming game.

   Before you change the label, confirm every context feature is known before kickoff. For example, opponent-defense aggregates must leave out game t. Otherwise the fix turns into leakage, and the walk-forward will make it look better than it is.

   Isotonic calibration is a separate final layer. Score with and without it on the same base models, which answers #5 at no extra cost. Train through 2024 and walk-forward 2025 against the fixed-pace blend.
   - If it passes, retrain through 2025 and serve from week 5 or 6.
   - If it fails, freeze the model until the offseason.

**Don't** tune w or chase QB rankings.

## Peer Reviews

Anonymization mapping (reviewers saw letters only): **A** = The First Principles Thinker, **B** = The Contrarian, **C** = The Outsider, **D** = The Statistician, **E** = The Executor. Reviewers also received the orchestrator addendum (prior per-game pace measurement).

### Reviewer 1 (Contrarian lens)

1. E is strongest. It is the only response that puts a check on both risky moves. The pace fix has to beat today's blend on squared error and pairwise accuracy over 2025 plus 2026 weeks 1–3, which is exactly the case nobody has tested. The alignment retrain gets a leakage audit, an isotonic on/off comparison and a freeze-if-it-fails exit. E also scores the live numbers every Tuesday. Flaws: it never compares against the season average, and its leakage rule should be "known at publish time," not "known at kickoff."

2. D has the biggest blind spot. It is the most rigorous plan (two retrains, eight cells) but never mentions leakage. Its "beat the season average" test would pass a model fed closing lines or final injury status. C has the same gap.

3. All five prescribe per-game pace, but it was already tested (GAPS.md, week 1). For veterans it was mixed: worse MAE, better RMSE. For cold-start players it was worse: MAE 4.47 vs 3.70, bias +1.83. Dividing by 17 was hiding a Step 8 rate that over-predicts. Players with g=0 get 100% pace, so test weeks 2+ inside the blend, split by g and cold-start, before shipping.

### Reviewer 2 (Statistician lens)

**1. D.** It is the only response with a control arm: it retrains the current recipe through 2024, which separates the alignment fix from the Vegas fix. It also has a factorial ablation and a win criterion written down in advance, and it flags 2025 as spent test data and QB ρ=−0.05 as noise. Gap: it never checks the aligned features for leakage (B and E do).

**2. A.** It turns an MAE tie into "you don't have a weekly model." That ignores the win over the trailing-3 average (0.22 [0.15, 0.29]), the squared-error win and the WR edge. It also reads the noisy QB ρ as a symptom.

**3. All five missed the addendum.** Per-game pace was already tested on week 1. Veterans got worse MAE but better RMSE; cold-start players got worse (MAE 4.47 vs 3.70, bias +1.83). So "units error" is half the story: the rate itself runs high, at least in week 1. A season-wide re-score dilutes this and can pass while cold-start and low-g rows worsen. Before shipping, test weeks 2+ inside the blend, split by g and cold-start.

### Reviewer 3 (Outsider lens)

1. E. You can follow it step by step as written: the steps are in order, have deadlines, each change must pass a test before it ships, and it says when to stop. Its pace test (2025 plus 2026 weeks 1–3, inside the blend) is exactly the test the addendum says was never run. It also checks the realignment for leakage. Weakness: it doesn't test cold-start players separately (see 3).

2. B. It commits to per-game pace up front and picks its metric so the fix won't "look like a regression and get reverted." The addendum shows that swap already failed for cold-start players on bias and MAE (+1.83 vs −0.18; 4.47 vs 3.70). Its MAE critique can't explain that away.

3. All five agreeing on the "units error" is one conclusion drawn from the brief, not five separate confirmations. Dividing by 17 was cancelling a second error: the per-game rate over-predicts week 1, worst for cold-start players. Test veterans and cold-start separately, in weeks 2+, inside the blend, before shipping.
