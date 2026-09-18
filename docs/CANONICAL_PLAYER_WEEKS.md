# Canonical Player-Week Panel (Phase 1)

Phase 1 creates a standalone `canonical_player_weeks` table. It is deliberately
separate from `player_weekly_stats`.

## Why

`player_weekly_stats` is box-score-triggered, so an absent row can mean multiple
things: no participation, zero offensive snaps, played but produced no box-score
line, or missing evidence. A PPR model should not be asked to learn those states
implicitly.

## Contract

One row per `(player_id, season, week)` for QB/RB/WR/TE when the player's team
actually played a regular-season game and at least one source provides evidence.

Participation uses the repository's existing authority:

- `snap_counts offense_snaps > 0` -> `confirmed_played`
- mapped `snap_counts offense_snaps == 0` -> `confirmed_zero_snaps`
- no mapped snap measurement -> `unknown`

A missing `player_weekly_stats` row **never becomes 0 PPR in Phase 1**.
`fantasy_points` remains NULL unless a real stats row exists.

Roster/status context comes from `weekly_rosters_v2` when available, then
`weekly_rosters`. Roster status is contextual evidence, not proof that the
player took an offensive snap.

## Build

```bash
python scripts/build_canonical_player_weeks.py --dry-run
python scripts/build_canonical_player_weeks.py --write --seasons 2013 2026
```

The builder reports:
- participation-state counts;
- missing stats rows;
- confirmed played weeks with no stats row;
- confirmed zero-snap weeks with no stats row;
- a season/position audit table written by default to
  `data/experiments/canonical_player_weeks_audit.csv`.

## Phase 1 acceptance gates

The build fails rather than silently continuing when:
- `(player_id, season, week)` is duplicated;
- team/opponent or QB/RB/WR/TE identity is unresolved;
- a confirmed-played row has non-positive snaps;
- a confirmed-zero row is not exactly zero snaps;
- a missing stats row receives fabricated fantasy points;
- a mapped snap row is classified as unknown;
- an existing `player_weekly_stats.fantasy_points` value changes.

FB/HB source positions are normalized to RB consistently with the rest of the
ingest pipeline.

Phase 1 is complete when these gates pass on the repository database and the
season/position audit has been inspected for unexpected discontinuities. The
audit intentionally does not impose arbitrary maximum rates on `unknown` or
missing-stat rows; those are measurements to carry into Phase 2, not defects to
hide.

## Phase 1 non-goals

- No participation model.
- No WR1/RB1 labels.
- No injury inference.
- No changes to production PPR training.
- No synthetic fantasy-point targets.

Those come only after this population and its missingness are audited.
