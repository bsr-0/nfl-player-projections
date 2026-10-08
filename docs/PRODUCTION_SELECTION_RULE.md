# Production model selection rule (pre-registered)

Status: **frozen at the commit that adds this file** (pre-registered
2026-10-08). That commit's hash is recorded in the forward-test lineage file
at freeze (see G10). Nothing below may change after any candidate's results on
an evaluation set are viewed. Any later change goes under "Amendments" with its
date and reason, and the affected comparisons are re-run.

## Candidates and reference

| Role | Model |
|---|---|
| Incumbent | Served weekly PPR (held-out fold of the served architecture) |
| Candidate | Plan A |
| Candidate | Plan A + Plan B arm |
| Candidate | Served + pace blend (re-run on raw outcomes) |
| Naive reference | Rolling-3 average (reported, not selectable) |

## Evaluation sets

### Primary: 2025 matched set

Frozen as the population manifest with sha256
`287775de00883b09111b82214e1b80cdcbe28c91f14b840f2e5f82f332e7682d`
(5,826 player-weeks). Its definition:

- Regular season 2025 only; the 251 postseason targets are excluded.
- One row per forecast target: the player's **next observed game**, meaning a
  row in `player_weekly_stats`. Players who were inactive or did not play have
  no row and are not targets. No minimum snap, route, or games-played filter.
- Players with a stat row who scored zero (or negative) points are included.
- The 16 rows whose team or position disagree between source panels are
  excluded.
- Every model must supply exactly the population's keys, or the comparison
  stops with an error (`assert_identical_key_sets` in
  `src/evaluation/paired_ppr_comparison.py`, called by the comparator).
  Coverage (eligible, matched, and dropped with reasons) is reported for each
  model.

### Confirmation: prospective 2026 shadow (G10, below)

### Label

Raw ten-component full PPR. A model that does not forecast a component
declares it as zero.

## Primary metric

Pooled MAE on the matched set, paired per player-week:
`d = |err_candidate| − |err_served|`. A negative `d` favors the candidate.
RMSE is a guardrail.

## Inference

- **Resampling: two-way (week × player) pigeonhole bootstrap**, B = **10,000**,
  seed **42**, for every interval except Spearman's. This was the pre-declared
  fallback, and it is the only method that met the 93% coverage trigger (see
  "Pre-registration checks"). It is conservative: its 95% intervals covered
  98.7–100% in synthetic tests, so it under-rejects.
- **Considered and rejected:**
  - **Week-cluster bootstrap:** 86.9–90.6% coverage, because 17 clusters are
    too few.
  - **Player-cluster bootstrap:** 92.1–95.1%, and below 93% in 3 of 9 settings.
    Players in the same game share shocks, which it misses.
- **Spearman** is computed within each position-week. Its interval is a
  one-sided t-interval over the week means (16 degrees of freedom). In checks,
  a week bootstrap of it rejected a true zero 9–17% of the time where 5% was
  expected.
- **Superiority (G1):** one-sided, α = 0.05, Holm-corrected across the 3
  candidates. **The Holm family is G1 only.** The candidates share the served
  baseline and overlap, so Holm is conservative but valid.
- **Guardrails (G2–G6) are a no-significant-regression test:** a cell fails if
  the one-sided 95% **lower** bound of its worsening (candidate minus served,
  signed so that positive is worse) is above 0. **Guardrails are not
  multiplicity-corrected**, because correction would let a regression pass.
- **No point-estimate backstop, and no thresholds from null spreads.** A
  backstop at 2× the margins fired on 94–98% of truly equal models. A backstop
  scaled to each cell's null spread did not carry over between windows (47%
  false blocks in weeks 9–16), because the spread estimates are unstable. No
  gate may use an estimated null spread.

## Projection tiers

Within each (season, week, position), rank players by projection, highest
first; ties are broken by `player_id` ascending. The tiers are ranks
**1–12**, **13–36**, and **37+**.

Every tier gate is evaluated twice: once tiered by the **rolling-3**
projection (neutral) and once by the **served** projection. **A candidate must
pass under both tierings.** If any matched key has no rolling-3 projection, the
comparison stops.

## Gates

| # | Gate | Test |
|---|---|---|
| G1 | Pooled paired MAE improvement vs served | Holm one-sided superiority **and** point improvement ≥ **0.10** PPR |
| G2 | Pooled RMSE | no significant regression (1 cell) |
| G3 | MAE by position: QB, RB, WR, TE | no significant regression (4 cells) |
| G4 | MAE by tier (1–12, 13–36), both tierings | no significant regression (4 cells) |
| G5 | \|bias\| (mean signed error): overall, plus per tier (1–12, 13–36, 37+) under both tierings | no significant regression on the \|bias\| difference (7 cells) |
| G6 | Spearman within position-week, averaged across groups | no significant regression, t-interval (1 cell) |
| G7 | 80% interval coverage (only if the model outputs intervals) | point estimate inside [0.75, 0.85] |
| G8 | Weeks 1–3 MAE | **report-only** |
| G9 | Beats rolling-3 on pooled MAE | point improvement > 0 |
| G10 | Prospective 2026 confirmation | see below |

The top-12 hit rate within position-week is reported but is not a gate.

**Family rule: any single guardrail failure among the 17 cells (G2–G6) blocks**
that stage. This choice has a cost. In checks, a truly equal model was blocked
17–18% of the time on 2025 and 9–10% of the time in an 8-week window, so it
**passes both stages only about 75% of the time.** Requiring 2 or more
failures would block only 1–3% of equal models, but it roughly halves
detection of mid-sized regressions (see "Power").

## What a guardrail failure means

A guardrail failure produces a **written report**, naming the failing cells
with their point estimates and bounds. It is **not a permanent rejection.**

- A candidate that passes G1 and G9 on 2025 but fails a guardrail is not
  selected on 2025 evidence. It may be **re-tested only in the forward window**
  (G10), where every guardrail must pass. It is never re-tested by re-running
  2025.
- A guardrail failure in the forward window is also reported. A further
  re-test needs a **new, later forward window** of the same length that starts
  after the failing window ends; weeks are never reused.
- A failure of G1 or G9 on 2025 is final for this rule.

## What the guardrails can and cannot catch

They protect against **broad** regressions, not narrow ones. Detection rates
for injected regressions are in "Power".

- **Broad regressions are caught:**
  - Noise spread across a quarter of players: 99% on 2025, 88% forward.
  - A bias shift of 0.3: 100% on 2025, 94% forward.
- **A large, single-position regression is caught only sometimes:** QB MAE
  +0.70 is caught 68% on 2025 and 44% forward.
- **Small regressions confined to one position or tier are mostly missed:**
  QB MAE +0.2 or top-12 MAE +0.17 is caught 6–26% of the time.

This rule makes no stronger claim than that.

## Confirmation season (G10): prospective 2026 shadow

2024 is **not** out-of-sample for any candidate, so it is not used:

- **Plan A:** its selector picks the 2025 configuration using 2023 and 2024
  fold results (`GAPS.md`, 2026-09-24), and its design was developed after
  viewing all three folds.
- **Plan A + Plan B arm:** the arm was added after its 2023–2025 confirmation,
  which includes 2024, had been viewed (`docs/PLAN_B_FUTURE_RUN.md`).
- **Served:** its feature promotions were judged on predictions covering
  2024–2026.

Instead, G10 is a prospective test on 2026 regular-season weeks:

- **Lineage freeze.** `data/experiments/selection_forward_2026/lineage.json` is
  written once, at freeze. It records:
  - the commit hash of this rule;
  - each candidate's weights and the served weights in use, with sha256;
  - the sha256 of the prediction-export and comparison scripts;
  - the **start week**, set at freeze and never changed afterward.

  There is no retuning, refitting, or feature change during the window.
- **Window:** the first **8** consecutive regular-season weeks whose first
  game kicks off after the **later** of this rule's commit and the lineage
  freeze. Weeks played before that point are excluded.
- **Pre-kickoff storage:** each week, every model's predictions and their
  sha256 are written before that week's first kickoff. A week counts only if
  every model's predictions were hashed before kickoff. A missing week, or a
  hash that does not match, fails G10.
- **Pass condition:** the pooled paired MAE difference vs served has the same
  sign as on 2025, **and** all 17 guardrail cells pass under the family rule.

## Validity checks (precondition — a failure voids the comparison)

- Strict walk-forward: each fold trains on seasons strictly before its test
  season.
- Features available as of Thursday kickoff only; no post-kickoff injury or
  inactive information.
- Identical player-week key sets across all models, asserted with a loud
  failure; input hashes recorded.
- Saved actuals equal raw truth exactly.

## Selection

1. A candidate is eligible if it passes G1, G7 and G9 on 2025, all guardrails
   (on 2025, or on a forward re-test as above), and G10.
2. Of the eligible candidates, pick the lowest pooled MAE on 2025.
3. If two are within 0.10 MAE of each other, prefer the lower week-to-week MAE
   standard deviation, then the simpler model.
4. If none is eligible, served stays in production.

## Known limitation of this pre-registration

Plan A vs served on the 2025 matched set was viewed on 2026-10-08, before this
rule was written (pooled Δ −0.27 favoring Plan A). For that pair the 2025
result is not blind; G10 is the clean test. The other candidates have not been
run on the matched set.

## Pre-registration checks (run 2026-10-08)

- **Script:** `scripts/selection_rule_checks.py`, sha256
  `8e85a25f90e39b5276b75a019d5ef09afe67e46c3304a4115febf966a499d052`. That
  hash is the one in `report.json` and the one committed with it; any later
  edit invalidates these results.
- **Output:** `data/experiments/selection_rule_checks_20261008/`
  (`report.json`, `run.log`).
- **Inputs:** the 2025 matched-set paired rows (served vs Plan A) and the
  rolling-3 projections in `full_ppr_raw_truth_20260924/ppr_oof_rows.csv`.
  Their hashes are in `report.json`. All 5,826 keys have a rolling-3
  projection, with identical team and position.

**Synthetic coverage.** Synthetic deltas were built from a true difference plus
week, game (week × team), player and resampled residual effects. The variances
were estimated from the 2025 deltas: week sd 0.006, game 0.338, player 0.499,
residual 2.82. These estimates are crude and run high, which makes the test a
stress test. Each setting below ran 900 replicates over 3 seeds. The weekly
shock was set to 1×, 2× and 4× its estimate, and the true difference to 0,
0.10 and 0.20.

| Bootstrap | Two-sided 95% coverage, min–max over 9 settings |
|---|---|
| Week cluster | 0.869–0.906 |
| Player cluster | 0.921–0.951 (below 0.93 in 3 settings) |
| Two-way week × player | 0.987–1.000 |

**Null (truly equal model).** The equal model was built by swapping served and
Plan A predictions for a random half of players. Its expected difference is
zero, and it keeps realistic player-level disagreement. Guardrails used the
two-way bootstrap with B = 1,000; Spearman used the t-interval.

| Stage | Per-cell false-fail rate | Blocked: any 1 cell fails | Blocked: 2+ cells fail |
|---|---|---|---|
| 2025 matched set (R = 200) | 0–10% | 17–18% | 1–3% |
| 8 weeks, weeks 6–13 (R = 100) | 0–6% | 9–10% | 1–3% |
| 8 weeks, weeks 9–16 (R = 100) | 0–3% | 10% | 2% |

**Power.** Regressions were injected into the candidate of a null pair. Noise
was sized to the sd of the Plan A − served prediction gap; "full", "half" and
"quarter" are multiples of it.

| Injected regression | Induced worsening | Blocked (any 1), 2025 | Blocked (any 1), forward | Blocked (2+), 2025 |
|---|---|---|---|---|
| 25% of players, full noise | RMSE +0.24, QB MAE +0.16 | 0.99 | 0.88 | 0.97 |
| All QB rows, full noise | QB MAE +0.70 | 0.68 | 0.44 | 0.40 |
| All QB rows, half noise | QB MAE +0.21 | 0.26 | 0.06 | 0.09 |
| Top-12 rows, half noise | top-12 MAE +0.17 | 0.24 | 0.08 | 0.10 |
| All rows, quarter noise | RMSE +0.06 | 0.64 | 0.28 | 0.34 |
| \|bias\| pushed 0.3 further from zero | \|bias\| +0.30 | 1.00 | 0.94 | 0.97 |

**G1 feasibility.** Under the two-way bootstrap, the one-sided 95% half-width
of the pooled paired MAE difference is 0.117. G1 therefore needs a true gain
of roughly 0.12 or more to pass.

## Before the rule is applied

1. **Extend the comparator.** `scripts/compare_full_ppr_head_to_head.py`
   currently handles two models and computes MAE with a paired week-block
   bootstrap. It needs:
   - more than two models;
   - the two-way bootstrap with B = 10,000 and seed 42;
   - RMSE, signed bias, both tierings, Spearman with the t-interval, and the
     top-12 hit rate;
   - the weeks 1–3 slice;
   - Holm-corrected one-sided superiority and the lower-bound guardrails;
   - per-gate verdicts in `report.json`.
2. **Test the extended comparator on synthetic data** with a known true
   difference. Assert the recovered difference and the interval coverage.
3. **Write the forward-test lineage file and start week at freeze**, as above.

## Amendments

None.
