# Same-week leakage audit of the live serving path — 2026-10-09

Method and tests: `scripts/audit_serving_leakage.py` (see its docstring).
Weeks audited: 2025 weeks 6 and 14, on the code as of this commit.

## Poison test — passed (`poison.json`)

Every post-game outcome at or after the target week was overwritten
(x * 3.7 + 11) in a copy of the database (player stats, team stats, snap
counts, NGS, PFR, shares, canonical panel, schedule scores) and in the
schedules the path downloads (scores, measured weather), and
`predict(as_of=(2025, W))` was re-run on the copy.

| Week | Players | Columns compared | Columns changed | Predictions changed |
| --- | ---: | ---: | ---: | ---: |
| 2025-w06 | 642 | 504 | 0 | 0 |
| 2025-w14 | 646 | 504 | 0 | 0 |

Positive control (poison from the last legitimately visible week): 160
columns and 231 predictions (w06), 161 columns and 265 predictions (w14)
changed, by up to 1.3 points. The
poison reaches the model, so the pass is not vacuous.

Not poisoned, because they are pre-game by design: Vegas lines, rest days,
venue, depth charts, rosters, injury reports (kickoff-guarded, see
`tests/test_leakage_guards.py`), draft and combine data.
**Correction (2026-10-09): `game_weather` is not pre-game.** It holds observed
weather from a historical archive, was not poisoned, and feeds three model
features. Replacing them with their no-weather defaults moves raw predictions by
0.02 points on average (GAPS.md, 2026-10-09), so the pass above excludes a
small, real dependence on measured weather. `qbr` and
`seasonal_pfr` have no rows for 2025 onward.

## Truncation test — five model features depend on later rows (`truncation.json`)

Features of already-played rows were compared between a frame holding only the
played history and a frame that also holds the target week and later. 5 model
features differ (13 columns in all at w06, 29 at w14):
`combine_score`, `qb_pressure_pct_roll3_mean`, `rb_yac_avg_roll3_mean`,
`rb_ybc_avg_roll3_mean`, `team_neutral_pass_rate_oe_roll3_mean`.

In every case the differing rows hold one to four distinct constants: these are
imputed fills (a position, or position-season, median/mean taken over the
whole frame), not a player's own later data. Example: 2025 QB pressure rate is
filled 0.0 when only weeks 1-5 exist and 0.5 once the whole season is in the
frame. This is not a leak into replays or live serving (both truncate before
feature engineering; the poison test shows it), but it means the training
frame's fills see the future and differ from serving's. Logged in GAPS.md,
not fixed: changing it changes model inputs.

## Re-run on the final code (Amendment 3) — `poison_after_amendment3.json`

The poison test was repeated for 2025 week 6 after the roster-club, unplayed-week
and opponent-feature changes: no feature and no prediction changed; the control
still changed predictions. (The opponent-feature change is a no-op on every
historical key, so replays are unaffected by construction; this confirms it.)
