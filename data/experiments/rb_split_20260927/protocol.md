# RB rushing/receiving split — protocol v1

Registered before implementation or candidate scoring on 2026-09-27 UTC.
Baseline source/input hashes are in baseline_manifest.json. The baseline includes
preexisting working-tree edits. No production deployment is authorized.

## Hypothesis and implementation
For RB i, split the supplied PPR center mu into mu*(1-f_i) rushing and mu*f_i
receiving. Apply each component's own carry/pass-target opportunity multiplier,
then add the SAME residual and apply the SAME activity mask and clipping as the
pinned simulator. Components sum to the original center. Legacy calls and other positions without newly supplied RB target shares must
remain bit-for-bit unchanged. If an RB target share is explicitly supplied, WR/TE
shares on that team must be allocated in the same joint target-pool draw; their
individual draws then change to preserve conservation. The archived replay has
no supplied shares, so all its non-RB draws must remain identical. The split also
supports separate RB target shares in the receiver pool, with unmodeled remainder;
existing RB usage_share continues to mean carry share. Shares are optional and
unavailable in the archived served files, so that replay uses team-volume-only
multipliers for BOTH arms. No additional model, availability, noise, scoring,
pace, or game-score changes are included.

## Split estimator, folds, budget
One prespecified candidate: last 8 observed prior RB games, shrink their rushing
and receiving PPR component sums toward 4 games of league RB component means.
Receiving component = receptions + 0.1*receiving_yards + 6*receiving_tds;
rushing component = 0.1*rushing_yards + 6*rushing_tds. Clamp each historical
component at zero only when estimating the fraction; do NOT clamp actual PPR
labels. The fraction is rec_sum/(rec_sum+rush_sum), with the shrunk prior ensuring
cold-start coverage. Fumbles and rare passing/2-point events are not separately
forecast: this estimates a split of the EXISTING total, not a new scoring system.
Validate prior fit through 2022 on 2023; refit through 2023, validate 2024; report
2025 separately as already-inspected development. Player rolling histories are
strictly prior to the forecast week (including earlier weeks in each validation
season). Only prior seasons enter fitted league means. Report fraction MAE on
positive-component rows (fraction undefined otherwise); this is an auxiliary
model diagnostic, NOT PPR accuracy evidence. Final prior fit through 2025.
No hyperparameter search or choice from confirmation: candidate count 1, no
adaptive expansion. Budget: one implementation, causal/correctness tests, the
three temporal diagnostics, one archived replay with two Monte Carlo seeds.
A failed/inconclusive replay does not trigger another candidate.

## Data and archived replay
Use actual archived served weekly 2026 week-2 projections and game predictions
at git f0d99b3847df824346abd0600416a06c1fb0cf96. They were available by its
commit timestamp. The weekly player file was generated 2026-09-19 00:25:27 UTC.
Exclude games on or before the archive's UTC date: schedule has date-only times,
so this conservative rule excludes Thursday and avoids guessing kickoff hours.
All remaining archived QB/RB/WR/TE player forecasts are the eligible cohort,
including backups, low projections, and inactive players; no actual-dependent
eligibility. A null/nonfinite/missing forecast, ambiguous schedule, or opponent
mismatch is a reported data failure, never silently dropped. Archive actual_points
is NEVER an input or label. Use separately frozen raw stats as truth.
Missing stat rows are UNKNOWN, not zero. Retain them in the prediction artifact;
report missing outcomes and descriptive observed-row metrics separately. The
primary cohort metrics/gates are UNAVAILABLE unless every eligible outcome is
resolved (including verified zeros for inactive/no-production players).
Archived player forecast is the baseline itself, NOT a reconstructed model.
The current Game Sim and split arm are explicitly retrospective reconstructions
from those archived inputs. History through 2025 plus 2026 week 1 only supplies
fractions. No same-week result, full-season usage, or future roster selects f.
Historical data were downloaded retrospectively: exact vintage of later stat
corrections is not recoverable and must be disclosed. No full-population or
untouched-confirmation claim follows from this development replay.

## Simulators and Monte Carlo
Use pinned defaults (64 plays, 0.58 pass rate, 10-point margin/total SD), archived
80% intervals for player SD (fallback 8), archived participation/share inputs when
present (otherwise participation 1, no shares), independent normal residuals,
zero clipping, identical player order and scripts. Run 10,000 draws each at
seeds 42 and 20260927, concatenate (20,000 draws) for point means. Use common
random numbers in old/new arms. Report between-seed mean and MAE-delta changes.
Do not change draw count/seeds based on whether gates pass. No correlation fit.

## Metrics, weighting, uncertainty and slices
Equal weight per eligible player-week; overall includes QB/RB/WR/TE. MAE uses
mean simulated PPR; served baseline uses archived predicted_points. RB RMSE and
position/slice metrics are diagnostics. Receiving slices fixed by pregame f:
low <0.25, mixed >=0.25 and <0.50, high >=0.50. Cold starts separately counted.
For each baseline, delta = updated MAE - baseline MAE; improvement % =
100*(baseline MAE-updated MAE)/baseline MAE (undefined at baseline MAE zero).
Resample complete season-week schedule blocks (Thursday through Monday treated
as one NFL calendar week), preserving all three paired forecasts and all players.
5,000 bootstrap replicates, seed 20260927, percentile two-sided 95% intervals.
Recompute sum of absolute-error differences / number of sampled player-weeks,
NOT an unweighted mean of week MAEs. All intervals unavailable with fewer than
two blocks; fewer than 8 complete weeks makes confirmation inconclusive even
if other numerical gates happen to pass. This minimum is a sufficiency rule,
not a relaxed gate. Report overall guardrail uncertainty too.

## Confirmation and freeze
2025 is NOT untouched. Available 2026 weeks 1–3 have already appeared in reports
and are NOT untouched. Proposed confirmation is regular-season 2026 weeks 4–18
(2026-10-01 onward; exact dates in frozen schedule.csv). At protocol registration,
those dates are future and local stats end at week 3. Thus their outcomes cannot
currently support a decision. Before first confirmation prediction: freeze code,
model, config, protocol and their hashes. For each week archive the existing served
forecast and game inputs strictly before that week's first kickoff; record UTC
publication/capture times and immutable hashes. Only information from completed
prior weeks supplies RB histories; no intraweek updates. Archive eligible cohort
at forecast time; resolve actuals for ALL eligible players after the week. Keep
missing/unresolved labels visible; they block primary scoring. Do not score
confirmation until the window ends, then evaluate once, without candidate tuning.
If those prospective archives/complete labels are unavailable, confirmation is
BLOCKED and the current decision is INCONCLUSIVE. Do not manufacture a historical
untouched period or treat a one-week replay as confirmation.

## Acceptance (unchanged user criteria)
PASS requires confirmation RB MAE <=0.95*EACH baseline RB MAE; EACH paired 95%
interval for RB delta wholly below zero; overall MAE <= EACH baseline overall
MAE. Overall intervals, RB RMSE and receiving-role slices with sample sizes and
deltas are reported but are not additional performance gates. Complete, valid,
sufficient confirmation violating any gate is FAIL; unavailable/invalid/insufficient
confirmation is INCONCLUSIVE. No inference of improvement from unit tests.
