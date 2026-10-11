# Simulation forward check, 2026 weeks 5-13 (pre-registered 2026-10-11)

Closes CLAUDE.md rows 7 (simulation marginals, and whether the simulation is
shown on the site) and 8 (the role-factor correlation layer). Written and
committed before any 2026 week 5 Sunday game kicked off and before any week
5-13 simulation was scored. The rules below are fixed; a change is an amendment
committed, with its reason, before the checkpoint it affects.

## What is scored

- **The simulation as published.** For every final game of weeks 5-13, the
  version of `docs/data/simulation_2026_wk{N}.json` committed before that game's
  kickoff (git committer time, nflverse kickoff time; the same rule as
  `src/evaluation/published_record.py`). Only files centred on the unblended
  weekly model (`model_config.centre` = `predicted_points_model`, from commit
  4573788d on) count. A game with no such file before kickoff is listed and not
  scored; week 5's TB@DAL is one (its pre-kickoff file was centred on the blend).
- **Draws.** The site file holds summaries, so the draws are rebuilt from the
  weekly and game files of the same commit with the same artifacts, seed and
  draw count, and checked against the published per-player means (tolerance
  1e-6). A game that does not rebuild is excluded and listed. Artifacts: run
  `20261008T003959900405Z` (fitted through 2026 week 3), calibration.json
  sha256 `2289c043b4ad83f8ef784dc61df704a607f3ef4b5a2b63664a5b8e0d97ee3678`,
  dependence.json `931b0e4db794833cd78e0b69c4f7c837448fae0c5ce1f80233320c4b2e5f763c`.
- **Arms.** `role_factor` (published) and `independent` (the same marginal
  draws without the role-factor reordering).
- **Outcomes.** `player_weekly_stats.fantasy_points` (raw PPR) for QB/RB/WR/TE.
- **Two populations** (`src/evaluation/simulation_forward.py`):
  - `played`: players with a stat line; stacks and totals recomputed over them
    (the 2024-25 backtest's rule).
  - `published`: the sums as published, every simulated player, a player who
    did not play scoring 0; team and game fantasy totals against the real
    totals (all of the team's QB/RB/WR/TE stat lines, including players the
    simulation did not list). Player ranges: players with a stat line.
- **Quantities.** Player 80% range; QB+2 stack (top QB and two top WR/TE by
  the centre); team fantasy total; game fantasy total. These are fantasy points
  of QB/RB/WR/TE, not the scoreboard.
- **Code.** `scripts/score_simulation_forward.py`, committed with this file.

## Decision 1: the correlation layer (row 8)

- **When:** once every week 13 game is final (on or after Tue 2026-12-08),
  weeks 5-13.
- **Metric:** stack CRPS, `role_factor` minus `independent`, population
  `played`; averaged over the stacks of each game, then over games; standard
  error over games.
- **Rule:** KEEP if mean + 1.96 SE < 0. Otherwise REVERT: the generator stops
  applying the dependence model (independent draws) from the next generation.
- Team and game totals, both populations, are reported with no decision weight.
- Row 8 is closed either way: "LIVE, PROVEN" or "REMOVED".

## Decision 2: whether the simulation is shown on the site (row 7)

- **When:** once every week 8 game is final (on or after Tue 2026-11-03),
  weeks 5-8, arm `role_factor` (as published), population `published`.
- **Rule, per quantity:** it may be shown on the site if its 80% interval
  coverage is within [0.72, 0.88] and its 50% coverage within [0.40, 0.60].
  A quantity that fails is not shown.
- **Re-check:** at the Decision 1 checkpoint on weeks 5-13, with whichever arm
  Decision 1 keeps. A shown quantity that fails then is removed.
- **Noise, accepted in advance:** by week 8 there are about 56 games, so a
  perfectly calibrated game total fails this band about one time in four.
  Team totals and stacks (about 110) fail it about one time in fifteen.
- Row 7 is closed when Decision 2 is applied.

## Known risk, and the one change allowed in advance

The draws are "points if he plays" for every player the weekly page lists
(about 22 per team), and the weekly rows carried no injury status for 2026
week 5's Sunday games when this was written. Population `published` counts a
listed player who sits out as 0, so availability errors count against the
team and game totals. If Decision 2 fails a sum for that reason, an
availability layer may be added, but only as a new pre-registration committed
before it is used, scored on weeks after it ships. It never re-scores weeks 5-8.

Otherwise nothing that shapes the draws changes during weeks 5-13: no change to
the generator, the artifacts or the weekly model's unblended path. Two things
are allowed: the input refreshes the weekly job already runs (stats, rosters,
injury reports), and the REVERT action.

## Out of scope

Plan A and the forward test of the weekly model (docs/PRODUCTION_SELECTION_RULE.md)
are not touched; this check reads the weekly files, it does not change them.
