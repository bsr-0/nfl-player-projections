# Pre-registration: WR season-to-date volume features, 2026 forward test (weeks 7-13)

Status: **pre-registered 2026-10-14/15, binding from the commit that adds this file.** It confirms, on
2026, the development result of `docs/PREREG_WR_FEATURES_2025.md` (reports in
`data/experiments/wr_features_2025/`). It does not amend `docs/PRODUCTION_SELECTION_RULE.md` or
`docs/PREREG_NO_PRIOR_2026.md`, edits no pinned file, and changes nothing that is served. No 2026 week
has been scored for this candidate, and none will be before the scoring step below.

## Question

Does adding season-to-date targets, receptions and receiving yards to the WR 1-week model reduce its
error on 2026 weeks 7-13, relative to the same model without them?

## Evidence so far (2025, not blind)

C1 (C0 + `targets_s2d`, `receptions_s2d`, `receiving_yards_s2d`) vs the identical refit baseline C0, on
2025 WRs: weeks 1-5 paired MAE difference -0.137; weeks 6-18 -0.061 (95% -0.129 to +0.007), RMSE lower,
rank correlation higher, form-lag slope +0.27 -> -0.10; better in 12 of 13 confirmation weeks. Pooled
all-weeks MAE was seen before the confirmation window was scored (Amendment 2 of that document), and 2025
chose the features, so it is development evidence only.

## Window and why it is 7 weeks, not 8

Weeks 7-13 of 2026. Week 6's deadline (00:00 US/Eastern on its first game date, the forward runner's
rule) passed before this was committed, so week 6 is not in this test. Each week's shadow forecasts are
written and sha256-hashed before that week's first game date at 00:00 ET, by a script outside the pinned
set. A week counts only if both arms were hashed before the deadline. If fewer than 5 weeks are valid the
test is void.

## Arms

All fit by the same recipe (`scripts/wr_feature_candidates_2025.py` `fit`, run on seasons 2006-2025), on the
WR rows only, with the live WR model's tuned hyperparameters (`data/models/multiweek_wr.joblib`, sha256
recorded at freeze) held fixed and no retuning:

| Arm | Inputs |
|---|---|
| **C0'** reference | the live WR model's 70 features |
| **C1'** candidate | C0' + `targets_s2d`, `receptions_s2d`, `receiving_yards_s2d` |

C1' differs from C0' only by the three added columns, so the paired comparison isolates them. C0' is the
primary reference. The runner's `served_unblended` (the live WR model, trained 2026-09-25) and
`incumbent_live_blend` WR forecasts are reported as context, not as the primary comparison, because their
training recipe differs.

Features, from `player_weekly_stats`: the mean of the stat over the player's games this season up to the
origin game; NaN (a first game of a season does not occur at serving; players with no 2026 game) is filled
with the training median. Both arms are frozen as artifacts, hashed in a lineage file, before the first
window week's forecasts; no refit or retuning during the window.

## Evaluation set and score

Each valid week's frozen list (`week_XX/list.csv`, written by the forward runner), WRs only, scored on the
raw ten-component full PPR of that week's game (`player_weekly_stats` row; stored `fantasy_points` must agree
within 0.05, a player-week with no stat row is not scored). Both arms must forecast every listed WR.

## Inference

Two-way (week x player) pigeonhole bootstrap, B = 10,000, seed 42 (`selection_gates._weight_batches`). d =
|err_C1'| - |err_C0'| per player-week.

## Decision rule

C1' passes if all hold:
1. mean d < 0 and the one-sided 95% upper bound of d is below +0.03 (non-inferiority with a point gain);
2. RMSE not significantly worse, and mean within-week Spearman not significantly worse (one-sided 95% bound
   on worsening not above 0);
3. the form-lag slope (actual - projected on season-to-date PPG minus projection, rows with at least 3
   prior games) is closer to 0 than C0''s.

Reported, not decisions: MAE by week and position-free tiers (1-12, 13-36, 37+ by C0' projection), lean,
pair-ordering accuracy, each arm against `served_unblended` and the live blend, and superiority (one-sided
95% bound below 0).

A pass supports proposing C1 for the WR model after the weeks 6-13 selection window closes, through a
lineage re-freeze or the next preseason retrain; it never changes anything during the window. A fail leaves
the WR model unchanged.

## Power (from 2025 confirmation weeks, stand-in sd)

The 2025 confirmation bootstrap sd of the mean difference was 0.034 over 13 weeks and 1,812 WR rows; for 7
weeks and about 1,000 rows it is about 0.046. With margin 0.03 and one-sided alpha 0.05: if the true
difference equals 2025's -0.061, rule 1 passes with probability about 63%; if it is 0, about 15%; if it is
+0.03, about 5%. A failure is therefore weak evidence of no effect.

## What this cannot show

Gains confined to a subgroup, interactions with the pace blend (the arms are unblended WR forecasts; a
blended or scaled version is a separate question), or anything about QB/RB/TE. 2026 weeks 1-5 are not used.

## Procedure

1. Commit this file before week 7's deadline (Wed 2026-10-21 23:00 CDT = Thu 00:00 ET) and before the
   artifacts are used.
2. Build and commit `scripts/run_wr_shadow_week.py` (captures the exact WR feature frame of one production
   `predict()` call through a read-only hook, adds the three columns, forecasts both arms, writes and
   hashes `data/experiments/wr_volume_2026/week_XX/`), `scripts/score_wr_volume_prereg.py`, tests with
   synthetic rows, and a `lineage.json` with the artifact and script sha256s. Record the script hashes here
   before the first window week.
3. Run the shadow script each week after the forward runner, before the deadline, with the same week
   override as the runner (`NFL_TARGET_WEEK`).
4. After week 13's stats are complete, run the scorer once; commit its `report.json`.

## Amendment 1 (2026-10-15), before the lineage freeze and before any window week

**Dry run (week 7 mechanics, under `NFL_TARGET_WEEK=2026:7`, `dry_run/week_07`).**
- The shadow runner's copy of the model input reproduces production's `predicted_points_model` for all 285
  WRs exactly (max difference 0.0) once it mirrors production's rule that a model input absent from the frame
  is filled with the model's training median. The first dry run, without that fill, failed the 1e-9 check
  (max difference 1.21); the fill is now applied to the reference and to both arms.
- The one affected input is `is_power5`: it is absent from the serving frame, so production (and therefore
  both arms) serve it as its training median although training had the real value. This is an existing
  train/serve skew of the live WR model, shared equally by C0' and C1'; it is reported, not fixed.
- Structure: 285 WRs, no missing forecasts, no missing volume features; C0' tracks the live WR forecast
  (correlation 0.997, mean difference -0.04); C1' moves forecasts by sd 1.06 relative to C0'. No outcome
  exists for week 7, so no accuracy was computed.

**Frozen inputs (sha256).**
- `scripts/wr_volume_shadow.py`: `498a1025257d1a393fcc2fa07e3d29b8f84845226591404d89ab38746b6e90cd`
- `scripts/wr_feature_candidates_2025.py`: `d3b85d772cd0f684051f0ad34c0d47ac6527f8ee3fa28f6427c1230dc62ff343`
- `scripts/score_wr_volume_prereg.py`: `1e5d3940a3831ef08c7a68ab4507947528c2ef7bacddbd5f67aca83ddc5083ce`
  (imports `scripts/score_no_prior_prereg.py`, sha256 `5dfa449eaa649f886c76ae1f1756ca43d7c1626f39c127ef08f853048d7917bf`)
- C0' artifact `artifacts/c0.joblib`: `06edee29ee56c5fafccc9e3c475ba91518f3e91c64c9a9e560b934a57da58077`
- C1' artifact `artifacts/c1.joblib`: `47688e59999f86663e498c2372cb0f53a86517bfd28b94b151a73ec691ec574d`
- Both fit on seasons 2006-2025 (41,809 WR rows after dropping rows without a 1-week target), the live WR
  model's tuned parameters held fixed (live `multiweek_wr.joblib` sha256
  `a7421314758ea6a1a76b3ac3df26fe2a3c6c875e072f43164271c6879cd8dc4a`); the bounded scaler fit in the build
  matches the live one in every transform value.

**Weekly order.** For week N: (1) `NFL_TARGET_WEEK=2026:N python scripts/run_forward_week.py run`, (2)
`NFL_TARGET_WEEK=2026:N python scripts/wr_volume_shadow.py run`, both before that week's deadline (00:00 ET on
its first game date). The shadow run refuses without the runner's frozen list for the week and requires every
WR on it to be forecast. Changing any file above, or the live WR model, breaks `lineage.json` and voids the
remaining window.
