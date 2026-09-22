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

### Data shape

One row per `(team, season, week)` holding a variable-length roster. This
needs a slot-assignment scheme (e.g. rank each position group by depth-chart
rank within the team-week) since most joint architectures want fixed
structure. This scheme is itself a new leakage surface — depth charts change
week to week (injury, bye, benching) — with no existing infra in this repo to
audit it; it doesn't inherit safety from anything already built here.

### Architecture options

- **Mixed-effects regression**: team-week random effect + player random
  effect + position fixed effects (`statsmodels` `MixedLM` or a Bayesian
  hierarchical model). Cheapest joint option; the team-week random effect
  captures shared game-environment variance without any new architecture.
- **Permutation-invariant set model**: Deep Sets or a small transformer over
  the roster, so the model can learn intra-team trade-offs directly. Highest
  potential ceiling, highest engineering cost, no precedent in this codebase
  (everything here is `sklearn`/`xgboost` pipelines).

### Why this is deferred, not abandoned

Relative to Plan A this needs: a new slot-assignment scheme (new leakage
surface, no existing audit pattern to lean on), a new modeling stack (mixed
effects or a set/graph net — neither exists anywhere else in this repo), and
a new evaluation step (predictions must be re-mapped from slot back to
`player_id` before scoring, which is itself a place to introduce a silent
bug). None of this is reusable from what already exists; every part would be
new, audited from scratch. That's a lot of new surface area to justify before
knowing whether the underlying idea (team-level structure improves on
independent per-player prediction) pays off at all.

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
      sentinel-value test mirroring `test_game_outcome_leakage.py`. Not yet
      run against the real database (only synthetic-DB tests so far) --
      running `--write` against `data/nfl_data.db` and inspecting the
      `is_cold_start`/coverage counts is the next step before building the
      regressors below.
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
      report) against a synthetic populated DB; **not yet run against real
      data** -- this sandbox's `data/nfl_data.db` has empty tables (no
      network access to refresh it here). Running
      `scripts/build_team_week_player_shares.py --write` then
      `scripts/train_team_share_model.py` against the real database, and
      checking whether ridge/xgboost actually beat `rolling3` on real
      held-out seasons, is the next step -- that result is Plan A's
      acceptance criterion 1 and 2 (see above).
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
      acceptance criterion 2 above. Not yet wired into the backtester/CLI
      (still module-level building blocks); not yet run against real data
      for the same sandbox reason as the rest of Plan A.
- [ ] Wire reconstruction into the walk-forward backtester/CLI (produce a
      per-fold/pooled partial-points MAE the way share MAE already is),
      then compare against the revised criterion 2 baseline once real data
      is available.
- [ ] Touchdown/reception share targets (needed for criterion 2 as
      originally stated -- not yet scoped, deferred until the yardage-only
      slice above shows the underlying approach has legs).
- [ ] Decision-gate review
- [ ] Plan B (not started; revisit only after the gate above)
