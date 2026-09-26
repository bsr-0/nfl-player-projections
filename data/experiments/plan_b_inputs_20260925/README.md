# Plan B roster snapshot — 2026-09-25

The existing roster-slot builder exported `slots.csv` for 2013–2025 with
`--dry-run`. It did not create or replace a SQLite table. `manifest.json`
records the file hashes, builder hash, source table row counts, and independent
identity and position-cap checks.

The database file changed when the separate A/B jobs finished. A read-only
rebuild from the current database produced the same slot CSV hash, and a fresh
join produced the same saved rushing-yards input-panel hash. The results and
current database metadata are in `reverification.json`.

| Measure | Rows |
| --- | ---: |
| Canonical/share player-weeks | 163,358 |
| Saved roster slots | 101,138 |
| Excluded by QB2/RB4/WR6/TE3 caps | 62,220 |

`slot_coverage.csv` and `slot_coverage_by_season.csv` show the fraction of
rankings driven by default depth-chart values. In 2025, this is 29.3% of saved
WR rows and 27.9% of saved TE rows. The exclusions and default rates must be
considered when interpreting any future score on this roster subset.

All four target preflights using this exact snapshot passed; their reports are
under `../plan_b_preflight_20260925/`. The subsequent first-cut model run is
under `../plan_b_run_20260925/`.
