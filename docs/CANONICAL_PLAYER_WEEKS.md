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
- confirmed zero-snap weeks with no stats row.

## Phase 1 non-goals

- No participation model.
- No WR1/RB1 labels.
- No injury inference.
- No changes to production PPR training.
- No synthetic fantasy-point targets.

Those come only after this population and its missingness are audited.
