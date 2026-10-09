# Dry run after Amendment 3 (2026-10-09)

`python scripts/run_forward_week.py dry-run` on the second re-frozen lineage
(`lineage.json`, head `d173718e`, start week 6) exited 0: lineage intact,
canonical rebuild left all 165,857 settled rows unchanged, week 4 stats
complete (358 rows, 32 teams), leakage audit on week 4 passed, every model
forecast every listed player (661). Output in `dry_run/week_05/`; the
Amendment 2 run is in `dry_run_superseded_a2_20261009/` (same 661 players).

The live frame no longer logs the "defaulted on 100.0% of rows" warning for
`opp_fpts_allowed_s2d_lag1` / `opp_fpts_allowed_dvoa_adjusted_lag1`.

## Change against the Amendment 2 dry run (same data, new code)

| Model | Rows changed (of 661) | Mean abs change | Max abs change |
| --- | ---: | ---: | ---: |
| Incumbent (live blend) | 229 | 0.050 | 1.63 |
| Unblended served | 344 | 0.135 | 2.86 |
| Plan A | 0 | 0 | 0 |
| Rolling-3 | 0 | 0 | 0 |

The served model moves because its two opponent-strength inputs are now real
instead of 0.0 (A3.4 expected this; no pass criterion). Plan A and rolling-3
do not read those features and are identical.
