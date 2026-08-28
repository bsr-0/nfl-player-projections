# Phase 7 history-NaN arm — results, 2026-08-28

Pre-registration: `../phase7_histnan_20260826/PRE_REGISTRATION.md` (v2),
committed before this run. Analysis executed in the order specified there.

**Outcome: rule met, flag flipped to default ON.** The effect is entirely
rookies; pooled numbers understate it ~5x and overstate what it means for a
typical player.

## Falsification — PASS

Placebo pair (OFF vs OFF, 2015): **538/538 identical predictions**, ΔMAE
exactly 0. The noise floor is zero, so every ON−OFF difference below is
attributable to the flag and not to seeds, threading or ordering.

This replaced v1's zero-dose stratum, which could not exist: the flag changes
the TRAINING matrix, so each arm fits a different model and no row is
untreated. See the v1 post-mortem below.

## Primary — paired ΔMAE by fold (negative favours ON)

| season | n | ΔMAE |
|--------|-----|---------|
| 2015 | 538 | −0.9018 |
| 2016 | 536 | −0.1995 |
| 2017 | 540 | **+0.6626** |
| 2018 | 561 | −1.5036 |
| 2019 | 563 | −0.8522 |
| 2020 | 601 | −0.1445 |
| 2021 | 613 | −1.1521 |
| 2022 | 599 | −1.2131 |
| 2023 | 580 | −1.3952 |
| 2024 | 579 | **+0.3436** |
| 2025 | 585 | −0.8820 |

folds 11 · favouring ON 9 · mean **−0.6580** · SE 0.2170 · **t = −3.03**

Decision rule, fixed in advance:

1. ≥8 of 11 folds favour ON → **9, MET**
2. mean ΔMAE ≤ −0.25 → **−0.658, MET**

## Effect size — the part that is easy to overstate

    MAE OFF 43.022  ->  ON 42.355     -1.55% of baseline
    ON closer on 50.0% of 6,295 player-seasons

Exactly 50.0%. Per individual player this is a coin flip. The aggregate gain is
real and fold-consistent, but no single projection is meaningfully better.

## Secondary — where the effect actually is

| group | n | ΔMAE | folds favouring ON | MAE off → on |
|-------|------|---------|-----|--------------|
| **rookie** | 1,188 | **−3.2596** | **10/11** | 44.4 → 41.1 (−7.3%) |
| veteran | 5,107 | −0.0649 | 8/11 | 42.7 → 42.6 (nil) |

**The entire effect is rookies.** Veterans move by 0.06 points. The pooled
−0.658 is a rookie-only effect diluted ~5x by a 19%-rookie population.

Per position (exploratory, underpowered):

| pos | n | ΔMAE | folds |
|-----|------|---------|-------|
| QB | 829 | −0.0276 | 5/11 |
| RB | 1,564 | −1.7321 | 9/11 |
| WR | 2,466 | −0.3573 | 8/11 |
| TE | 1,436 | −0.4114 | 7/11 |

RB carries most of it; QB shows nothing.

## Mechanism note

96.5% of scored feature rows are byte-identical between arms (95.1–97.8% per
fold), yet essentially no predictions match. The gain is therefore
**model-level** — the flag changes the training matrix — not the debut rows
being scored differently. The design cannot separate those two channels and
does not claim to.

The honest summary: training on honest debut-week NaN produces a model that is
better at rookies. It is NOT established that preserving NaN on a given
rookie's row improves that row's prediction.

## How to describe this

> ~7% better rookie season projections; no measurable veteran effect.

NOT "1.55% more accurate", which is true of the pooled number but invites the
reader to think every projection improved.

## What went wrong first (kept deliberately)

Two prior attempts failed their own falsification checks, both on the strata:

- **v1** stratified by dose computed from a full-history matrix of REAL weekly
  rows, while `--cold-start --preseason-mode` scores SYNTHETIC rows built per
  week. Different populations, so "zero dose" never meant zero dose. 12 hours,
  detected only at the end.
- **v2** fixed the population (added `--feature-rows-output` to capture the
  rows actually scored) and failed again, because no null stratum can exist at
  all: the flag changes the training matrix, so every prediction moves. Caught
  after ONE season by the mid-run gate, not eleven.

`scripts/check_phase7_arms.py` now runs after each season pair and aborts the
run. That is what turned a 12-hour failure into a 1-hour one.

## Reproduce

    # placebo gate, then 11-fold interleaved, with per-pair validation
    # (both arms per season so the gate is meaningful after fold 1)
    NFL_PRESERVE_HISTORY_MISSINGNESS=0|1 python scripts/run_phase7_season_projection.py \
      --cold-start --preseason-mode --seasons <S> \
      --output main/arm_<off|on>/phase7_<S>.csv \
      --feature-rows-output main/arm_<off|on>/features_<S>.parquet

    python scripts/check_phase7_arms.py --dir main --season <S>
    python scripts/check_phase7_arms.py --dir placebo --season 2015 --placebo

Commit: `6640495` (pre-registration v2) → run at `6640495`.
