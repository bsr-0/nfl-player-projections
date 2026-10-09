# Dry run after Amendment 2 (2026-10-09)

`python scripts/run_forward_week.py dry-run` on the re-frozen lineage
(`lineage.json`, head `c1867609`, start week 6) exited 0: lineage intact,
canonical rebuild left all 165,857 settled rows unchanged, week 4 stats
complete (358 rows, 32 teams), leakage audit on week 4 passed, every model
forecast every listed player. Output in `dry_run/week_05/`; the superseded
run is in `dry_run_superseded_20261009/`.

## Comparison with the superseded dry run (A2.3)

The two runs are not like for like: rosters and injury reports were refreshed
in between (the list grew from 657 to 661 players), so differences mix data
and code. Cross-run, 74 players changed club, and rows changed for
incumbent 6 (max 0.19), unblended 31 (max 0.32), Plan A 275 (max 1.22),
rolling-3 14 (max 0.93).

To isolate the code change, one live `predict()` was run twice on identical
data in one process, with the roster-club step off and on (706 players, 86
moved club):

| Output | Rows changed | Max change |
| --- | ---: | ---: |
| `predicted_points` (incumbent, live blend) | 0 | 0 |
| `predicted_points_model` (unblended served) | 27 | 0.285 |

A2.3 expected both to be identical. The incumbent is. The unblended model is
not for 27 players: they are movers whose game-line and opponent features
changed, and they have no 2026 game yet, so the pace blend
(w = g / (g + kappa), g = 0) ignores the model for them and the incumbent does
not move. Plan A's change is the intended one (its team grouping follows the
roster club).
