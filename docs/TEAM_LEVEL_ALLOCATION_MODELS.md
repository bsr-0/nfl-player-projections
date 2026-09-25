# Team/Game-Level Player Allocation Models: Plan A and Plan B

## Motivation

Every production model in this repo (`src/models/single_week_ppr/`) predicts a
player's fantasy points independently of teammates. This is architecturally
simple and already leakage-audited, but it cannot represent within-team
competition for touches (e.g. RB1/RB2 timeshare, WR target competition) — each
player is a separate regression problem that never sees what the rest of the
roster is doing that week.

`src/models/game_outcome/` already predicts team-level outcomes (win
probability, margin, total) from team+week features, using the same
`SeasonAwareTimeSeriesSplit` walk-forward harness the position models use. This
doc plans two ways to connect that team-level signal down to individual
players, instead of treating each player as fully independent:

- **Plan A — compositional shares**: predict each player's *share* of team
  volume (targets, carries, yards), reconstruct a point estimate as
  `team_total × player_share`.
- **Plan B — joint/hierarchical team model**: model the whole roster jointly
  per team-week so the model can represent trade-offs between teammates
  directly, rather than reconstructing them after the fact from independent
  share predictions.

Both were investigated together because both need the same team+week training
data; see "Shared data prerequisite" below. **Plan A is scoped for a first
implementation; Plan B is documented for later, pending Plan A's result** (see
"Decision gate" below).

Prior, directly relevant result: GAPS.md (2026-09-19, "Game-outcome
predictions as weekly-player features") fed the `game_outcome` model's
win-prob/margin/total predictions into the existing independent per-player
models as extra *features* and found no effect. That is not evidence against
Plan A or B — it only shows that handing team-level numbers to an
architecture that still treats each player independently doesn't help. Plan A
and B both change the architecture itself, not just the feature set.

## Shared data prerequisite

Neither plan can start until a `team_week_player_shares` table exists:

- For each `(team, season, week)`, sum `player_weekly_stats` volume columns
  (`targets`, `rushing_attempts`, `receiving_yards`, `rushing_yards`,
  `air_yards`) to get the team total for that column that week. Use
  `player_weekly_stats` itself as the denominator source (not `team_stats`),
  since it is guaranteed to sum exactly across teammates — `team_stats`'
  `pass_attempts`/`rush_attempts` are a separate, independently-loaded source
  and are not guaranteed to reconcile 1:1 with the sum of individual
  attempts/targets in the same week.
- Join back per player: `share_of_team_X = player_X / team_X` (0 when the
  team total is 0, not NaN — a team that didn't record a stat that week has a
  legitimate 0 share for every player, not a missing value).
- Lag exactly like `_lagged_form` in `src/models/game_outcome/features.py` and
  `CAUSAL_FEATURES` in the position-model pipeline: a player's share history
  (season-to-date, rolling-3) must be `shift(1)`'d before aggregating. The
  same-week share itself is the *label*, never a feature, for the same reason
  `home_win` is never a feature in `build_game_outcome_rows`.
- Source rows: `canonical_player_weeks` (Phase 1's audited panel, see
  `docs/CANONICAL_PLAYER_WEEKS.md`) restricted to QB/RB/WR/TE, joined to
  `player_weekly_stats` for the volume columns and to `team_stats`/`schedule`
  for team-week context (Vegas implied total, pace, `neutral_pass_rate_oe`).

This table is the only new data-engineering surface both plans share. Once it
exists, Plan A and Plan B diverge.

## Plan A — compositional shares model

### Targets

One share target per volume type, each defined only over the "in the same
target pool" population (e.g. `share_of_team_targets` is 0 for a run-only RB,
not undefined):

| Target | Population | Definition |
|---|---|---|
| `share_of_team_targets` | QB excluded | `player_targets / team_targets` |
| `share_of_team_rush_attempts` | all | `player_rushing_attempts / team_rush_attempts` |
| `share_of_team_receiving_yards` | QB excluded | `player_receiving_yards / team_receiving_yards` |
| `share_of_team_rushing_yards` | all | `player_rushing_yards / team_rushing_yards` |

Each team-week's shares should sum to ≤ 1 across the roster by construction
(they're literally `part / whole`); this is a cheap, free-standing sanity
check to run on every training frame before fitting.

### Features

Reuse the existing lagged, leakage-audited features as-is:

- rolling-3 / season-to-date share history (new, built alongside the targets
  above, same lag discipline);
- existing `CAUSAL_FEATURES`: snap-share trend, depth-chart rank, injury
  status, prior-season role;
- team-level context: `implied_team_total`, `neutral_pass_rate_oe`,
  `spread`/`is_favorite` (already sign-corrected per GAPS.md 2026-09-19),
  pace/plays-per-game.

### Model

One regressor per share target (XGBoost, same shallow-depth conservative
hyperparameter philosophy as `src/models/game_outcome/models.py`), or a
single multi-output model if sharing statistical strength across targets
turns out to matter. Predicted shares within a team-week are renormalized
(divide by their own sum) so they sum to exactly 1 across the roster at
serving time — raw independent regressor outputs won't do this on their own.

### Reconstruction

```
predicted_volume(player) = predicted_team_total(stat) × renormalized_share(player)
predicted_fantasy_points = scoring_formula(predicted_volume components)
```

`predicted_team_total` starts as the simple lagged season-to-date/rolling-3
team average (cheapest, no new model) — `scripts/build_team_week_player_shares.py`'s
`_lagged_team_totals()` now builds this alongside the player-level share lags
(`team_{stat}_s2d`/`team_{stat}_roll3`, same shift(1)-before-aggregating
discipline, no cold-start fallback in this first cut — NaN for a team's first
tracked weeks is correct, not something to zero-fill). Swapping in an actual
team-total forecasting model (a sibling to `game_outcome`'s margin/total
regressors) is a later step once the share model itself is validated —
decoupling these two sources of error makes it possible to tell which one is
responsible for any accuracy change.

`renormalize_shares()`/`reconstruct_volume()`/`reconstruct_partial_fantasy_points()`
in `src/models/team_allocation/reconstruct.py` implement the formula above.

**Scope limitation found while implementing this** (see that module's
docstring): the four share targets cover targets/rush-attempts/receiving-
yards/rushing-yards only — no touchdowns, receptions, or passing production.
`calculate_fantasy_points_df` (`src/utils/helpers.py`, this repo's single
scoring-formula source of truth) contributes 0 for any stat not supplied, so
reconstructing points from only `rushing_yards`/`receiving_yards` yields a
**yardage-only partial fantasy-points number**, not the real target.
`targets`/`rushing_attempts` have no direct entry in `config.settings.SCORING`
at all and are excluded from the points reconstruction (they remain useful
model inputs — volume is a leading indicator of yardage/reception
opportunity — just not something to convert into points directly). This
changes acceptance criterion 2 below; it does not block continuing Plan A,
but the comparison it was originally meant to buy (this vs. the production
model's full-PPR accuracy) isn't available yet.

#### Full-PPR schema extension (data foundation)

The `team_week_player_shares` builder now also emits the component columns
needed for a full-PPR reconstruction: `receptions`, `receiving_tds`,
`rushing_tds`, `passing_yards`, `passing_tds`, and `interceptions`, together
with same-week shares, lagged season-to-date/rolling-3 shares, and lagged team
totals. Receiving targets exclude QB rows; passing targets are QB-only; rush
attempts/rushing yards/rushing TDs remain all-position populations. The
original four Plan A targets remain the default model set, so existing
experiments are not silently expanded. Run
`scripts/build_team_week_player_shares.py --write` to rebuild the table.

This is only the schema and causal-history layer. Receptions/TDs/passing
components still require their own allocation models and are not yet included
in `POINTS_ELIGIBLE_VOLUME_COLS` or production artifacts.

### Evaluation

Reuse `SeasonAwareTimeSeriesSplit` exactly as the `game_outcome` backtesters
do — same held-out seasons the position models use, `strict=True` (see
`src/models/position_models.py`) since this is a real held-out accuracy claim
in the same class as `game_outcome_backtester.py`, not inner-CV tuning.

Primary comparison: reconstructed `predicted_fantasy_points` MAE/RMSE vs. the
existing production per-position model's MAE/RMSE, on the identical
held-out player-weeks. Secondary: share-prediction calibration (do shares sum
near 1 pre-renormalization; does share MAE beat a naive rolling-3-share
baseline, the same baseline class Phase 2 already uses).

### Acceptance criteria (mirrors Phase 2's style)

1. Share models must beat a rolling-3-share baseline on share MAE — if they
   don't, the reconstruction has no chance of beating the current models
   either, and there's no point looking at reconstructed accuracy at all.
2. ~~Reconstructed fantasy-point MAE must beat the existing per-position
   production model on the same held-out seasons, by position.~~ **Revised**
   (see the Reconstruction scope-limitation note above): this reconstruction
   only ever produces a yardage-only PARTIAL fantasy-points number, so
   comparing it against the production model's full-PPR MAE is not
   apples-to-apples. Revised criterion: reconstructed partial-points MAE
   must beat a naive partial-points baseline computed the same way from the
   production model's own predicted `rushing_yards`/`receiving_yards`
   components (if exposed) or from a rolling-3 yardage baseline if not —
   i.e. compare like-for-like slices, not this model's partial output
   against the production model's full total. Full parity with criterion 2
   as originally stated requires adding touchdown and reception share
   targets first (not yet scoped).
3. No unexplained population/coverage discontinuity by season or position
   (same discipline as Phase 2's acceptance criteria).
4. All causal-contract and walk-forward tests pass (new
   `tests/test_team_week_player_shares.py`, mirroring
   `tests/test_game_outcome_leakage.py`'s sentinel-value-lag pattern).

If criterion 1 fails, stop — don't proceed to reconstruction accuracy, and
don't proceed to Plan B (see decision gate below).

## Plan B — joint/hierarchical team model (documented, not yet scoped for build)

### Why it's different from Plan A

Plan A predicts each player's share independently and only enforces
"sums to 1" as a post-hoc renormalization. Plan B instead models the whole
roster jointly per team-week, so the model itself — not a renormalization
step — represents "more targets to X implies fewer to teammate Y."

### Data shape — built (2026-09)

`scripts/build_team_week_roster_slots.py` produces `team_week_roster_slots`:
one row per `(team, season, week, slot)` — long, not wide, so a set-model or
mixed-effects regression can key off `slot` as an ordinary categorical
rather than needing a fixed number of columns — where `slot` is
`f"{position}{rank}"` (e.g. `RB1`, `WR3`) and `rank` comes from a
deterministic sort: `(depth_chart_rank asc, lagged season-to-date
snap_share desc, player_id asc)`.

**Correction to this doc's original claim** ("no existing infra in this repo
to audit [depth-chart leakage]; it doesn't inherit safety from anything
already built here") — that was wrong, found while actually building this:
`src/features/feature_engineering.py` already has a well-audited, pregame-safe
as-of depth-chart lookup (`_load_depth_chart_asof_table` /
`_add_depth_chart_rank`, with a 1-season staleness bound and a documented
`week <= target week` convention — that week's OWN snapshot is legitimate
pre-game info, same category as Vegas lines), already used by the production
per-position models via `depth_chart_rank`. This script reuses that lookup
rather than re-deriving one.

What's genuinely new (not solved by the reused lookup): `depth_chart_rank`
is not unique — multiple players routinely share a rank, or all fall back to
the same "3" (missing/stale) default — which is fine for using it as a
*feature* but breaks a fixed-slot assignment, which needs uniqueness. This
script's actual contribution is the deterministic tie-break above
(`snap_share` lag, then `player_id`) plus a hard-enforced invariant
(`validate_roster_slots`: at most one player per slot, at most one slot per
player, per team-week) and a per-slot coverage report
(`audit_slot_coverage`: how often each slot is filled, how often the
missing/stale default drove the assignment) — read that report before
trusting anything built on this table, same discipline as Plan A's coverage
report.

**Two real bugs found and fixed while building this** (worth recording,
per this repo's standing rule against assuming reused code is safe without
verifying): (1) the reused `_load_depth_chart_asof_table` always reads
`config.settings.DB_PATH` directly and caches its result at process scope,
ignoring whatever connection/`--db` path a caller passes — this script's own
`--db` argument would have been silently ignored for the depth-chart lookup
only (while still respected for `canonical_player_weeks`/
`player_weekly_stats`), a real cross-database inconsistency risk if not
handled; fixed by pointing `config.settings.DB_PATH` at the same file and
clearing the cache before calling it (see `load_depth_chart_rank_asof`'s
docstring), with a regression test proving two different DB files produce
different results in the same process rather than a stale cached one. (2)
The snap-share tie-break was originally computed only over weeks that had a
`player_weekly_stats` row, silently skipping a player-week with none (Phase
1's "unknown" participation state can produce exactly this) — the same
failure mode `_append_placeholder_rows` in `src/models/game_outcome/features.py`
already exists to prevent; fixed by lagging against the full
`canonical_player_weeks` population instead of the raw stats table, caught by
a failing test rather than assumed correct.

Caps (`MAX_SLOTS_PER_POSITION = {"QB": 2, "RB": 4, "WR": 6, "TE": 3}`) are a
starting assumption, not a measured constant — a team-week with more
rostered players at a position than its cap has the excess simply dropped
(flagged in the coverage report, not silently lost). Revisit once the
coverage report is run against real data.

15 tests (`tests/test_team_week_roster_slots.py`), including a sentinel-lag
test (a snap-share spike in week W must affect week W+1's tie-break, never
week W's own) and the DB-path/cache regression test above. Not yet run
against real data — this script is a Plan A-style building block, same
sandbox blocker as the rest of this doc.

#### The three specific risk vectors named for this step, addressed individually

("depth chart changes, byes, injuries reshuffling mid-week" — the original
risk list's own wording. Each is a different kind of risk; lumping them
together as "leakage-prone" would have hidden that only one of them is
actually about leakage.)

- **Depth chart changes (mid-season, real role changes)**: not a leakage
  risk given the reused lookup's `week <= target week` as-of convention —
  a promotion becomes visible starting the week it's actually listed, never
  earlier. Verified, not just assumed: `test_depth_chart_promotion_mid_season_flips_slots_at_the_right_week`
  builds a player promoted from RB2 to RB1 starting week 3 and checks the
  slot flip lands exactly there, not bleeding backward into weeks 1-2 or
  lagging past week 3.
- **Byes**: not a new risk at all — inherited for free.
  `canonical_player_weeks` only has rows for games a team actually had
  scheduled (`build_canonical_player_weeks.py`'s inner join to `schedule`),
  so a bye week is simply ABSENT from the population this script reads, not
  a row needing special handling. Verified, not just assumed:
  `test_bye_week_produces_no_slot_rows_at_all` seeds a 3-week gap and checks
  no row exists for it, rather than trusting the inheritance blindly.
- **Injuries reshuffling mid-week**: real, but an ACCURACY limitation, not
  a leakage one — worth being precise about the difference. `slot`
  reflects the LISTED depth-chart role, not confirmed game-day
  availability; a starter ruled out after that week's depth-chart snapshot
  still occupies their listed slot in this table even though they didn't
  play. This cannot leak anything forward (the snapshot is still
  exclusively pregame-known), but it can occasionally mis-rank who the
  "actual" starter was for that specific week. This has no separate
  fix needed: the downstream LABEL (`share_of_team_X`, from
  `team_week_player_shares`) is built from real recorded production, so a
  listed-but-inactive player's row correctly shows 0 (or near-0) actual
  share regardless of their assigned slot — "RB1 sometimes gets 0" is a
  real pattern for a model to learn from a genuinely pregame-unpredictable
  event, not a bug to engineer around. Cross-referencing real injury data
  (`InjuryDataLoader`'s kickoff-filtered pregame status, already used
  elsewhere in this repo) to adjust the tie-break itself would be a
  legitimate future refinement, not a correctness requirement of this
  first cut.

**Verified against this repo's own incident history, not re-derived from
scratch** (GAPS.md 2026-08-19, "Historical coverage backfill: depth charts,
injuries, NGS" and "Unbounded depth-chart staleness + dirty `depth_charts`
table"): the reused lookup already has a 1-season staleness bound
(`DEPTH_CHART_MAX_STALENESS_SEASONS`) and deterministic same-week-conflict
resolution (`MIN` across duplicate listings) built in; `depth_charts`
coverage runs 2013-2025 with 83-86% of skill-position rows carrying a real
rank across that whole range (2013-2019 were 100% default before a since-
applied backfill fixed this — an EARLIER GAPS.md entry describing that
still-open gap would have been stale, and is superseded by the later
entry; checked the dates before citing either). Coverage still isn't
perfectly uniform even post-backfill (2020-2023 are skill-position-only,
2024-2025 cover every position; granularity itself differs by season) —
which is exactly why `audit_slot_coverage_by_season` (new, this pass)
exists: a per-season, per-position breakdown of how often the missing/stale
default drove the assignment, so a real coverage regime shift shows up in
the report directly rather than only being discoverable after a confusing
accuracy result.

Still genuinely true from the original text below: the architecture and
evaluation-remapping work haven't started, and are real, separate cost —
this data-shape work does not by itself reduce Plan B's overall cost/benefit
case relative to Plan A, it only replaces one assumed-risky item with a
built-and-tested one.

### Architecture options

- **Mixed-effects regression — built (2026-09), see below.**
- **Permutation-invariant set model**: Deep Sets or a small transformer over
  the roster, so the model can learn intra-team trade-offs directly. Highest
  potential ceiling, highest engineering cost. `torch` is in
  `requirements.txt` but genuinely unused anywhere in this codebase (checked
  while starting this section) -- so "no precedent in this codebase" still
  holds even though the raw dependency happens to already be available; a
  training loop, data loader, and serialization convention would all be new.
  Not started.

### Mixed-effects regression — built

`src/models/team_hierarchical/` (`features.py`, `models.py`) implements a
random-intercept-per-player mixed model via `statsmodels` `MixedLM`, reusing
Plan A's `team_week_player_shares` labels/features (joined with
`team_week_roster_slots` for the `slot` fixed effect) and, unmodified, Plan
A's reconstruction utilities.

**Design correction from this doc's original sketch above** ("team-week
random effect + player random effect"), found while actually building this,
not assumed away: a random effect's value is a deviation estimated FROM
THAT GROUP'S OWN DATA. A walk-forward test row is always a genuinely NEW
(team, season, week) by construction, so a random effect grouped by
team-week has zero observations to estimate from at prediction time and
contributes nothing to a forecast -- confirmed empirically (see
`src/models/team_hierarchical/models.py`'s module docstring): even
statsmodels' own `MixedLMResults.predict()` ignores every group's random
effect by default, known or not. Grouping by `player_id` instead avoids
this: a player with prior training history keeps the same group across
future weeks, so their persistent deviation (talent/role tendency beyond
what slot+features predict) genuinely carries forward; a true cold-start
player correctly falls back to the fixed-effects-only prediction, the same
"don't fabricate, default sensibly" behavior used throughout this repo.
Team-environment effects (what the team-week grouping was meant to capture)
are instead carried by Plan A's existing lagged team-total features already
in the fixed-effects design (`team_{stat}_s2d/roll3`) -- an observed,
pre-game-known proxy, unlike an unobservable-until-the-fact team-week
intercept. `predict()` manually adds the known-player random effect back on
top of statsmodels' fixed-effects-only output, since the library itself
won't.

Other things handled, found while implementing rather than assumed safe:
an unseen `slot` category at prediction time (no estimated coefficient)
falls back to that position's rank-1 slot, and to the fold's constant mean
if even that is unseen; a numeric fixed-effect column with zero variance in
a fold is dropped before fitting (would otherwise make the design matrix
singular); non-convergence does NOT raise an exception in statsmodels --
verified empirically -- so a non-converged fit is explicitly discarded and
the model falls back to the same trivial-constant treatment
`VegasFavoriteBaseline` already gives a degenerate fold elsewhere in this
repo, rather than silently keeping untrustworthy parameters.

13 tests (5 features, 8 models) covering all of the above, including a
synthetic scenario specifically designed so `slot` does NOT fully determine
player identity (several players per team share each slot label) -- the
actual condition the random effect is meant to explain, which a naive
1-player-per-slot test would trivially and misleadingly pass regardless of
whether the random-effect logic works. Verified end-to-end against a real
synthetic DB through the full `build_team_week_player_shares.py` ->
`build_team_week_roster_slots.py` -> `load_slot_share_rows` ->
`MixedEffectsShareModel` pipeline.

Not yet built: a walk-forward backtester for this model (mirroring
`src/evaluation/team_share_backtester.py`'s `SeasonAwareTimeSeriesSplit`
harness), and the slot-to-player_id evaluation re-mapping the "why this is
deferred" section below still correctly flags as outstanding.

### Why this is deferred, not abandoned

**Updated twice now**: the slot-assignment scheme (Data shape, above) and a
first working mixed-effects model (Mixed-effects regression, above) are both
built and tested, reusing this repo's existing depth-chart infra and Plan
A's reconstruction utilities rather than inventing new leakage surface or a
parallel scoring formula. What's still genuinely missing, relative to Plan
A: (1) a walk-forward backtester for this model, so its accuracy can
actually be measured the same honest way every other model in this repo is
-- right now it's built and unit-tested, not evaluated; (2) the
set/graph-net architecture, which is still real new ground (no precedent in
this codebase, see above); (3) the evaluation re-mapping (predictions must
be re-mapped from slot back to `player_id` before scoring -- once a roster
changes week to week, a slot's occupant isn't fixed, which is itself a
place to introduce a silent bug). That's real, separate cost still to
justify before knowing whether the underlying idea (team-level structure
improves on independent per-player prediction) pays off at all -- the gap
keeps shrinking, but it isn't closed.

## Decision gate

Plan A is the cheap, fast test of the shared underlying hypothesis: does
explicit team-structure modeling beat independent per-player prediction?
Plan B is only worth its much larger cost if Plan A shows a real,
held-out-season, walk-forward-validated improvement over the current
production models. If Plan A fails acceptance criterion 1 or 2 above, treat
that as evidence against investing in Plan B too, not just against Plan A's
specific (simpler) architecture — re-open this doc and reconsider before
building anything from the Plan B section.

## Status

- [x] `team_week_player_shares` builder + leakage tests
      (`scripts/build_team_week_player_shares.py`,
      `tests/test_team_week_player_shares.py`). Same-week shares are
      correct-by-construction (built from the same source rows as their own
      team-week denominator); `_s2d`/`_roll3` lag columns verified with a
      sentinel-value test mirroring `test_game_outcome_leakage.py`. **Run
      against the real database (2026-09-22)** -- `--write` against
      `data/nfl_data.db` initially failed `validate_shares`: 671 team-weeks
      had a share outside `[0, 1]`, because a player's own
      receiving_yards/rushing_yards can be genuinely negative for a single
      game (670 negative `receiving_yards` rows, 1,927 negative
      `rushing_yards` rows in the real data -- e.g. a screen pass stuffed
      for a loss), which pulls a teammate's share above 1 or that player's
      own share below 0 when summed naively. Fixed by flooring volume at 0
      for the share/team-total computation only (raw `panel[c]` and
      fantasy-point scoring elsewhere are untouched) -- a net-negative week
      genuinely captured none of the team's positive offensive output, so 0
      is the correct share, not a fabricated value. New regression test:
      `test_negative_volume_is_floored_for_share_computation_only`. After
      the fix: 164,298 player-weeks, seasons 2013-2026, cold-start rate a
      consistent ~19-25% every season (2026 shows 100% only because it's 1
      week into that season), share sums correct. Coverage report read and
      clean before proceeding to the regressors below.
- [x] Plan A share regressors (one per volume target) --
      `src/models/team_allocation/` (`features.py`, `models.py`,
      `baseline.py`), `src/evaluation/team_share_backtester.py`,
      `scripts/train_team_share_model.py`. Ridge + XGBoost arms vs. a
      `RollingShareBaseline` (the "market line" equivalent for this
      problem -- predicts the player's own trailing-3-game share
      verbatim), same `SeasonAwareTimeSeriesSplit(strict=True)` walk-forward
      harness as `game_margin_backtester.py`. `TEAM_ALLOCATION_MODEL_CONFIG`
      added to `config/settings.py`. No inner-CV hyperparameter tuning yet
      (fixed conservative defaults, same phase-1 philosophy as
      `game_outcome`) -- a fast-follow, not blocking.
      Verified end-to-end (build shares -> train -> backtest -> print
      report) against a synthetic populated DB. **Run against real data
      (2026-09-22) -- ACCEPTANCE CRITERION 1 FAILED, all four targets.**
      `scripts/check_team_share_acceptance.py`'s paired-bootstrap gate:

      | Target | ridge vs rolling3 | xgboost vs rolling3 |
      |---|---|---|
      | targets | 0.0325 vs 0.0300 -- FAIL | 0.0315 vs 0.0300 -- FAIL |
      | rushing_attempts | 0.0273 vs 0.0240 -- FAIL | 0.0265 vs 0.0240 -- FAIL |
      | receiving_yards | 0.0392 vs 0.0377 -- FAIL | 0.0387 vs 0.0377 -- FAIL |
      | rushing_yards | 0.0317 vs 0.0287 -- FAIL | 0.0310 vs 0.0287 -- FAIL |

      `rolling3` has the lowest MAE on every target, and the 95% bootstrap CI
      on the MAE delta is entirely positive in all 8 (arm x target) cases --
      ridge/xgboost are reliably WORSE than the naive trailing-3-game share,
      not just statistically tied to it. Holds up in the position and
      cold-start breakdowns too (`rolling3` wins on every position slice on
      every target checked). R^2 is 0.4-0.7 for all three arms (real signal
      is being learned), so this is not a broken pipeline -- the tunable
      models just don't add anything beyond a player's own trailing-3-game
      share. Most likely explanation on record: no inner-CV hyperparameter
      tuning yet (see above) -- untested whether tuning would close the gap;
      decision was made to stop here rather than chase that (see Decision
      gate below). Per this doc's own criterion 1, reconstruction-level
      accuracy (criterion 2) was NOT evaluated -- there is no point scoring
      a reconstruction built on a share model that already lost to the
      baseline it needed to beat first.
- [x] Coverage report + always-written metadata (robustness follow-ups):
      `build_team_week_player_shares.py` now prints/writes a season/position
      coverage summary (`audit_coverage`, `--audit-csv`, mirroring
      `build_canonical_player_weeks.py`'s `audit_panel()`) and a team-week
      share-sum diagnostic table (`audit_team_week_sums` -- a report to
      read, not a gate; `validate_shares` still hard-fails any sum > 1).
      `train_team_share_model.py` now writes the metadata JSON sidecar on
      every run, including bare `--no-save` evaluation runs (only the
      joblib model artifacts are skipped) -- so a pure backtest run is
      diffable across commits instead of living only in stdout. **Read the
      coverage report before trusting any MAE number** on the first real
      run -- a season/position discontinuity there invalidates the
      accuracy comparison regardless of what it says.
- [x] Statistical significance + segment breakdown + acceptance gate:
      `src/evaluation/team_share_backtester.py`'s `bootstrap_mae_delta()`
      adds a paired bootstrap CI (2000 resamples, config-driven) on
      (candidate MAE - rolling3 MAE) to every tunable arm's pooled result --
      "beats the baseline" now means the 95% CI excludes 0
      (`significant_improvement`), not just a lower point estimate. Pooled
      and per-fold results also break down by position and by
      `is_cold_start` (`_segment_metrics`) so a lift can't hide behind a
      pooled-only average. `scripts/check_team_share_acceptance.py` reads
      the metadata JSON and exits non-zero unless at least one tunable arm
      reliably beats rolling3 for a target -- a repeatable gate instead of
      an eyeball read of stdout. All verified end-to-end (including a case
      where ridge passes and xgboost doesn't) against a synthetic DB.
- [x] Plan A reconstruction building blocks: `src/models/team_allocation/reconstruct.py`
      (`renormalize_shares` -- sums each team-week's predicted shares to 1,
      with a documented equal-split fallback for the degenerate all-zero
      case; `reconstruct_volume` -- share x team-total, NaN-propagating;
      `reconstruct_partial_fantasy_points` -- reuses
      `src/utils/helpers.py:calculate_fantasy_points_df`, this repo's single
      scoring-formula source of truth) and the lagged team-total columns
      (`_lagged_team_totals` in `build_team_week_player_shares.py`,
      `team_{stat}_s2d`/`team_{stat}_roll3`, same shift(1) discipline as
      everything else in this pipeline). 10 new tests, all passing.
      **Scope discovery**: this can only ever reconstruct a yardage-only
      PARTIAL fantasy-points number (no TDs/receptions/passing) -- see the
      Reconstruction section's scope-limitation note and the revised
      acceptance criterion 2 above.
- [x] Wired reconstruction into a walk-forward backtester/CLI:
      `src/evaluation/team_share_backtester.py:walk_forward_oof_predictions()`
      (per-row, per-fold, per-arm out-of-fold share predictions -- a
      parallel sibling to `run_walk_forward_backtest`, not a refactor of
      it, same rationale the game_outcome/game_margin backtesters already
      use for staying separate) feeds
      `src/evaluation/team_reconstruction_backtester.py:run_reconstruction_backtest()`,
      which renormalizes + reconstructs + scores PARTIAL fantasy points per
      arm (ridge/xgboost/rolling3), with the same position breakdown and
      paired-bootstrap-CI-vs-rolling3 treatment as the share backtest.
      `scripts/evaluate_team_share_reconstruction.py` is the CLI entry
      point, writing `team_share_reconstruction_metadata.json`. The
      `rolling3` arm here means "reconstruction fed each player's own
      rolling-3 share" -- the correct baseline for the revised criterion 2
      (reconstructed-vs-reconstructed, not against the production model's
      full-PPR MAE). A QB row has no `receiving_yards` population (see
      `reconstruct.py`'s docstring) and correctly reconstructs to 0
      receiving yards rather than a gap. 16 new tests (5 on the
      reconstruction backtester using real `build_shares()` output through
      a 3-position synthetic DB, not hand-synthesized columns; 6 more on
      `walk_forward_oof_predictions` itself). Verified end-to-end against a
      synthetic DB -- QB partial-points MAE came out lowest (small
      rushing-only production) and RB highest (both rushing and receiving
      volume), the expected qualitative shape. **Deliberately not run
      against real data**: criterion 1 failed first (see above), and this
      doc's own decision gate says not to proceed to reconstruction accuracy
      once that happens.
- [x] Full-PPR guarded validation (2026-09-24): the eight-component
      allocation/team-total reconstruction now has a three-fold 2006--2025
      result on 40,559 identical player-weeks. A fold-local selector begins
      with rolling-3, admits learned allocations and convex blends only after
      a paired share-MAE non-inferiority gate on completed OOF folds, and
      selects against reconstructed PPR. After correcting actual labels to
      the raw eight player components, the selected configuration improved
      MAE from 2.49130 to 2.36984 (40,559 rows; paired delta CI -0.13532
      to -0.10674). Corrected artifacts are under
      `data/experiments/full_ppr_raw_truth_20260924/`; the earlier 2.36521
      estimate is superseded. This
      supersedes the earlier blanket "Plan A failed" conclusion only for the
      now-complete full-PPR guarded harness; it does **not** promote Plan A
      into the serving/UI path. Sparse TD allocation remains rolling-3.
      The shadow serving path now uses zero fallback for learned sparse
      allocations and supports rolling-3 team totals; its frozen fold-zero
      2023 export exactly matches all 13,404 OOF PPR predictions when the
      saved prediction CSVs use round-trip float parsing.
- [ ] Touchdown/reception share targets (needed for criterion 2 as
      originally stated -- moot for now; criterion 1 failed first, see
      Decision-gate review below).
- [x] Decision-gate review (2026-09-22): **Plan A failed acceptance
      criterion 1** on real data, across all four volume targets (see
      above) -- the rolling-3-game share baseline reliably beats both ridge
      and XGBoost, not just on a point estimate. Per this doc's own gate,
      this is treated as evidence against investing in Plan B's larger
      cost too, not just against Plan A's specific architecture. Decision:
      stop here rather than chase hyperparameter tuning immediately: don't
      build Plan B's walk-forward backtester, don't scope touchdown/
      reception share targets, don't reopen this doc, unless/until someone
      revisits whether tuning (the one acknowledged gap in Plan A's model
      config) changes this result. Plan B's data-shape and mixed-effects
      groundwork below remains as documented, tested prep -- explicitly not
      a decision to proceed with Plan B.
- [x] Plan B data shape: `scripts/build_team_week_roster_slots.py` +
      `tests/test_team_week_roster_slots.py` (15 tests). Built ahead of the
      decision gate above, deliberately, as groundwork prep while waiting
      on real data for Plan A -- not a decision to proceed with Plan B's
      actual model. Corrected this doc's original claim that slot
      assignment would need new, unaudited leakage infra: it reuses
      `feature_engineering.py`'s existing pregame-safe depth-chart as-of
      lookup. Found and fixed two real bugs in the process (see Data shape
      section): a DB_PATH/cache gotcha in that reused lookup, and a
      placeholder-row gap in the tie-break's own lag computation.
- [x] The three named risk vectors (depth chart changes, byes, injuries
      reshuffling mid-week) addressed individually, not just declared
      closed by the data-shape work above -- see "The three specific risk
      vectors named for this step, addressed individually" under Data
      shape. Two verified with new regression tests (mid-season promotion
      timing, bye-week absence); the injury vector correctly identified as
      an accuracy limitation, not a leakage one, with no separate fix
      needed. Also added `audit_slot_coverage_by_season` -- coverage isn't
      uniform across seasons even after GAPS.md's 2026-08-19 depth-chart
      backfill, so a real regime shift is now visible in the report
      directly rather than discoverable only after a confusing result.
- [x] Plan B mixed-effects model, first cut: `src/models/team_hierarchical/`
      (`features.py`, `models.py`). Random intercept per `player_id` (not
      team-week -- see the Architecture section's design correction, found
      empirically while building this: statsmodels' own `.predict()`
      ignores every group's random effect by default, and a team-week
      grouping has no observations to estimate from at walk-forward
      prediction time regardless). Reuses Plan A's labels/features and
      reconstruction utilities unmodified. 13 tests, including a
      synthetic scenario specifically designed so `slot` doesn't fully
      determine player identity (the actual condition the random effect is
      meant to explain). Verified end-to-end against a real synthetic DB.
      Built ahead of the decision gate, deliberately, as groundwork prep --
      not a decision to proceed with Plan B as the chosen path.
- [ ] Walk-forward backtester for the mixed-effects model (not started --
      it's fit/predict-tested, not yet accuracy-evaluated the honest way
      every other model in this repo is)
- [ ] Slot-to-player_id evaluation re-mapping, and the set/graph-net
      architecture (not started; still correctly gated on Plan A's
      real-data result per the decision gate)
