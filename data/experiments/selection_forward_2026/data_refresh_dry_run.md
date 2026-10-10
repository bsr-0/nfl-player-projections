# Dry run after the three live-data refreshes (2026-10-09)

No pinned file changed, so there was no re-freeze: the lineage (`lineage.json`,
Amendment 3) is intact. Three data refreshes landed after the Amendment 3 dry
run (`amendment3_dry_run.md`) and none had a dry run of its own:

1. PBP-derived 2026 columns repaired (`scripts/repair_pbp_advanced_columns.py`)
2. 2026 weekly PFR, NGS, snap counts and 2025 seasonal PFR loaded
   (`scripts/refresh_live_inputs.py`)
3. `draft_picks_v2` 2026 ids replaced by the official GSIS ids
   (`scripts/refresh_draft_identity.py`; backup
   `data/backups/nfl_data_pre_draft_identity_20261009_211045.db`)

`python scripts/run_forward_week.py dry-run` on the final data exited 0: lineage
intact, canonical rebuild left all 165,857 settled rows unchanged, week 4 stats
complete (358 rows, 32 teams), the same 661 listed players (`list.csv` identical),
every model forecast every player. Output in `dry_run/week_05/`; the Amendment 3
run is in `dry_run_superseded_data_20261009/`.

## Change against the Amendment 3 dry run (all three refreshes together)

| Model | Rows changed (of 661) | Mean abs change | Max abs change |
| --- | ---: | ---: | ---: |
| Incumbent (live blend) | 239 | 0.109 | 3.13 |
| Unblended served | 239 | 0.204 | 5.48 |
| Plan A | 503 | 0.391 | 4.23 |
| Rolling-3 | 0 | 0 | 0 |

## Which refresh moved which model

Plan A re-run on the database as it stood before each step (same listed players,
same artifacts; `pre_pbp_repair` reproduces the Amendment 3 dry run to 0.0000):

| Step | Plan A rows changed | Rolling-3 rows changed |
| --- | ---: | ---: |
| 1. PBP column repair | 503 (mean 0.391, max 4.23) | 0 |
| 2. PFR / NGS / snap / seasonal loads | 0 | 0 |
| 3. Draft identity | 0 | 0 |

Plan A reads the 2026 `redzone_targets`, `neutral_*` and similar columns when it
builds a player's history, which step 1 filled from zero. The move is toward the
2025 training inputs. This was noted at the time but not measured; it is the
largest effect of the three steps and it is on a candidate, not the incumbent.

The incumbent's per-step effects were measured on scratch copies of identical
data: 159 rows (step 1), 206 (step 2) and 39 (step 3, below).

## Step 3 alone, on the live frame (2026 week 5, 706 players)

Same database copy with and without the 254 id updates; Plan A and rolling-3 do
not read draft data. Before the update every debuted 2026 skill rookie had
`draft_round` 8 and `is_undrafted` 1 (the undrafted sentinel), because the draft
row could not be joined: a round 1 pick got the undrafted prior of 5.20 instead
of 13.13. 50 rows now carry real draft round, pick, value and capital.

- Live blend: 39 of 706 players move (WR 20, TE 11, RB 8), all but a few upward
  (mean +0.48 among the movers, 17 by more than 0.25, 6 by more than 1).
- Unblended served: the same 39 players, mean +0.92, max 5.48.
- Largest: J. Love (pick 3) 6.65 -> 9.78, K. Concepcion (24) 3.95 -> 5.96,
  J. Price (32) 3.94 -> 5.80, M. Lemon (20) 2.97 -> 4.64, K. Sadiq (16) 2.31 -> 3.83.
- Rookies drafted in rounds 6-7 mostly do not move (10 of 12 by less than 0.02;
  B. Sharp +0.18, M. Benson -0.16): the undrafted prior sits next to round 7.
- No same-week information is involved: draft round and pick are facts known
  before the player's first game.

## Rehearsal of the week 6 steps (2026-10-09, late)

The weekly job's steps were run by hand against the real database, except
`auto_refresh`: on a Friday it would store week 5 from Thursday's game alone
(see GAPS.md, "Week 6 rehearsal"). Found and fixed on the way: a PBP cache
holding a partial week 5 that the loader would have reused through Wednesday,
a team-level PBP cache stuck at week 1 (18 `team_stats` columns NULL for weeks
2-4; repaired, backup `nfl_data_pre_team_pbp_20261009_234609.db`), and schedule
scores missing after week 1 (weeks 1-4 loaded, backup
`nfl_data_pre_schedule_scores_20261009_234228.db`).

`refresh_live_inputs`, `refresh_draft_identity` and `refresh_participation` had
nothing to add, `backfill_injuries` wrote 1,398 rows, `check_live_inputs` passed
(warnings: no 2026 participation yet). Dry run exited 0, lineage intact, 165,857
settled canonical rows unchanged, same 661 listed players. Previous run in
`dry_run_superseded_team_pbp_20261009/`.

| Model | Rows changed (of 661) | Mean abs change | Max abs change |
| --- | ---: | ---: | ---: |
| Incumbent (live blend) | 91 | 0.013 | 1.29 |
| Unblended served | 91 | 0.024 | 2.26 |
| Plan A | 0 | 0 | 0 |
| Rolling-3 | 0 | 0 | 0 |

90 of the 91 changes are the `team_stats` repair (`team_neutral_pass_rate_oe_roll3_mean`,
`team_pace_sec_per_play_roll3_mean`); each matches the repair's isolated effect on
scratch copies of identical data, where it moves 101 of 706 live players (mean abs
0.09 among them, mean signed +0.001). The 91st is B. Mayfield (+0.64): the injury
backfill added 327 2026 rows, among them his final week 5 status ("Out", thumb;
before: none). That an "Out" status raises the incumbent's forecast is odd and
not investigated here.
