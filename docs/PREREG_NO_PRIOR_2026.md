# Pre-registration: the weekly model without the pace prior (2026 forward window)

Status: **pre-registered 2026-10-14, binding from the commit that adds this
file.** That commit lands before week 6's first game date (2026-10-15), and
before the week 6 forecasts are written. No window week existed, and no window
result had been computed or viewed, when it was written. Changes after this
commit go in an "Amendments" section with date and reason, before any window
week is scored.

This is a separate pre-registration. It does not amend
`docs/PRODUCTION_SELECTION_RULE.md` (the rule): the rule's text is hashed in the
forward-test lineage, and editing it would void the window. Nothing here changes
the rule's candidates, gates, G10 or its decision. This document adds one
question, answered from forecasts the rule's weekly runner already writes and
hashes before kickoff, and evaluated only when the window is complete.

## Question

Can the served weekly model, **without the Step 8 pace prior**, match the live
blend's accuracy while removing the blend's lag toward a stale pre-season number?

## Why (exploratory, not evidence under this document)

On the 2025 matched set (5,824 player-weeks; both arms already viewed under the
rule on 2026-10-08), analysed on 2026-10-14:

- The prior (Step 8 season total / 17) counts expected missed games as zeros and
  runs 1.6 points low for players who play. Its weight (3/(g+3)) pulls the blend
  down 0.57 points on average.
- That pull explains the blend's whole MAE edge over the unblended model (4.266 vs
  4.356). Scores are right-skewed (61% of games land below the unblended
  projection), so MAE rewards a lower number. Moving the unblended model down by
  the same average amount, with no prior, matches the blend.
- The prior adds little ranking information (within position-week Spearman
  +0.006, not significant). QBs are the exception.
- The blend under-projects players running ahead of their prior. With 3 or more
  games played and season-to-date PPG at least 3 above the prior, it is 1.43
  points low on average. Per point of form gap, it misses by +0.19 (unblended:
  near 0).

The candidates and the constant below were chosen after viewing 2025. 2025 is
therefore **not** evidence for them; only the forward window is.

## Candidates

`served_unblended` and `incumbent_live_blend` are the columns of each
window week's `data/experiments/selection_forward_2026/week_XX/forecasts.csv`.
That file is written and sha256-hashed before kickoff by `scripts/run_forward_week.py run`.

| Name | Forecast | Prior used |
|---|---|---|
| `no_prior_scaled` | `served_unblended × 0.9218` | none |
| `no_prior_scaled_qb_blend` | QB: `incumbent_live_blend`; RB/WR/TE: `served_unblended × 0.9218` | QB only |

- **K = 0.9218** is sum(blend) / sum(unblended) on the 2025 set. It puts the
  candidate at the blend's average level, so the test isolates the prior's
  player-level information rather than a change in level.
- **Why scale, not shift.** A flat −0.57 shift failed three 2025 guardrail cells
  (RMSE; |bias| in tier 37+ under both tierings), because it over-penalises low
  projections. The scaled form failed none.
- **No fitting during the window.** K is fixed here and nothing is fitted during
  the window.

Incumbent: `incumbent_live_blend`, the live blend as served (kappa = 3).

## Evaluation set

- **Rows.** The same scored rows as the rule's G10:
  - each window week's frozen list;
  - scored on the raw ten-component full PPR of that week's game;
  - players with a `player_weekly_stats` row only.
  Players on no list are unscored, and the same key-set assertion applies.
- **Actuals and positions.**
  - The actual is the raw ten-component full PPR of the player's 2026 stat row
    for that week, scored from its components.
  - Scoring stops with an error if the stored `fantasy_points` differs by more
    than 0.05, or if a player-week has two stat rows.
  - Position and team come from the frozen list (pre-kickoff); QB, RB, WR and
    TE only.
- **Window.** Weeks 6-13 of 2026, the rule's frozen window. No week is added,
  dropped or reused.
  - **Why only these weeks.** They are the only 2026 weeks with no-prior
    forecasts written and hashed before kickoff by the frozen runner, which
    refuses any other week.
  - **Weeks 1-5** are played. They also motivated these candidates (the
    2026-10-14 under-projection analysis), so they are not blind.
  - **Weeks 14-18** are outside this test. Extending to them would need an
    amendment, committed before week 13 is scored, with its own pre-kickoff
    hashing outside the pinned runner.
- **A missing or unhashed week.** A week is excluded for every arm if its
  manifest is a dry run, it was written after the deadline, or its forecasts
  differ from the manifest sha256. If more than one week is excluded, this
  test is void.

## Inference

- **Bootstrap.** Two-way (week × player) pigeonhole bootstrap, B = 10,000,
  seed 42, the rule's own weights (`src/evaluation/selection_gates.py`).
  The same draws serve every cell and both candidates.
- **Primary, P1: non-inferiority on pooled MAE vs the incumbent.**
  - Paired difference: d = |err_candidate| − |err_incumbent|.
  - Margin: M = 0.10 PPR, the rule's own practical-difference threshold (G1 and
    the selection tiebreak).
  - p = (#{bootstrap d̄ ≥ M} + 1) / (B + 1).
  - Holm across the two candidates, one-sided α = 0.05.
  - Passes when Holm rejects "d̄ ≥ M".
- **Guardrails: the rule's G2-G6 cells, unchanged** (17 cells, both tierings,
  Spearman t-interval). A cell fails when its one-sided 95% lower bound of
  worsening is above 0. Any one failure blocks; no multiplicity correction.
- **Secondary, S1: the reason for the change.**
  - Population: rows with games played this season before the week ≥ 3, and
    form gap ≥ +3.
  - Form gap = mean `player_weekly_stats.fantasy_points` over the player's 2026
    games before the week, minus his 2026 `step8_pace`
    (`data/step8_pace_by_season.csv`, pinned in the lineage).
  - Statistic: |mean(pred − actual)| for the incumbent minus the same for the
    candidate. Positive means the candidate leans less.
  - Required: a point estimate > 0. Its one-sided 95% interval is reported.
- **Report only (no decision):**
  - RMSE;
  - MAE by position;
  - MAE for weeks 6-9 and 10-13;
  - form-lag slope (OLS of actual − pred on form gap, games ≥ 3) per arm;
  - pair-ordering accuracy within position-week;
  - G7 is not evaluated: the forward files carry no interval for the
    unblended model.

## Decision

1. A candidate is **eligible** if it passes P1 and all 17 guardrails, and S1's
   point estimate is > 0.
2. If both are eligible, adopt `no_prior_scaled` (simpler: no prior anywhere).
3. If one is eligible, adopt it. If neither is eligible, the live blend stays.
4. Adoption happens after week 13 is scored, never during the window (every
   pinned file stays unchanged until then). It means serving
   `predicted_points_model × 0.9218` (QBs keep the blend for the hybrid). The
   80% range becomes the weekly model's own interval × 0.9218. Its coverage is
   reported over the first 8 weeks after adoption; that report triggers no
   decision.
5. This decision is independent of the rule's G10 result. If the rule selects a
   different model for production, this question is re-asked against that model
   in a new window.

## Power (from 2025 weeks 6-13 as a stand-in window, 2,641 rows)

- **Bootstrap sd of d̄:** 0.028 (`no_prior_scaled`), 0.020 (hybrid).
- **2025 point differences in those weeks:** +0.012 and +0.008.

Probability that P1 passes at M = 0.10 with one-sided α = 0.025 (the Holm worst case):

| True difference | no_prior_scaled | hybrid |
|---|---|---|
| 0 | 95% | ~100% |
| 2025 value | 88% | ~100% |
| +0.05 | 43% | 71% |

- **Guardrails.** An equal model is blocked 8-18% of the time over 8 weeks
  (rule, A1.7). A QB-only regression the size seen on 2025 (+0.12 MAE for
  `no_prior_scaled`) is caught about 25% of the time. This is the main reason
  the QB hybrid is registered.
- **S1.** On 2025 weeks 6-13, the |lean| cut was +0.49 (`no_prior_scaled`) and
  +0.36 (hybrid). Both were positive in 97-98% of bootstrap draws.

## What this cannot show

- Small regressions confined to one position or tier (see the rule's "Power").
- Interval quality (no forward intervals for the unblended model).
- Weeks 1-5, where the prior's weight is largest and the window has no weeks.
  Adoption would change those weeks too, untested.
- Whether K holds in 2026. It is fixed from 2025 and is not re-estimated.

## Procedure

1. **This file and the scorer are committed together** before week 6's first
   game date. The commit time is the pre-registration time.
2. **Scorer:** `scripts/score_no_prior_prereg.py`, sha256
   `5dfa449eaa649f886c76ae1f1756ca43d7c1626f39c127ef08f853048d7917bf`.
   - Tests: `tests/test_score_no_prior_prereg.py`, synthetic rows only.
   - It reads only:
     - the window's `forecasts.csv` files, each checked against its manifest
       sha256;
     - `player_weekly_stats`;
     - the pace table, checked against the lineage sha256.
   - `check` verifies the weeks written so far and reads no outcome.
   - `score` refuses to run before week 13's stats are complete, and refuses to
     overwrite a report.
   - Any edit to the scorer changes its sha256 and needs an amendment first.
3. **After week 13's stats are complete,** run
   `python scripts/score_no_prior_prereg.py score` once. Commit
   `data/experiments/no_prior_prereg_2026/report.json`. No candidate result is
   computed or viewed before then.
