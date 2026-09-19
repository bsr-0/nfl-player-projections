# Undrafted rookie debut-stub measurement (2026-09-18)

Same question as the drafted-rookie measurement
(rookie_debut_stub_2022_2025_README.md), scoped to the undrafted half added
the same day: for a UDFA rookie with zero games this season, is the weekly
ensemble's own prediction better than the Step 8 pace it would otherwise be
shrunk to 100%?

Method: identical to the drafted measurement, but the candidate population
is `_drafted_rookie_stub_rows`' undrafted half specifically (roster
`years_exp <= 1`, never in draft_picks_v2, no prior history) -- NOT "any
undrafted player with zero games," which would also catch established
undrafted veterans (Ekeler, Meyers, Bourne) having a bye-adjacent week and
wrongly count their real career signal as if it came from a blank-slate
rookie stub. That mistake was caught and fixed before this result was
recorded.

Result, 2022-2025 real week-1 debuts (32 rows with a real outcome, out of
49-per-season true UDFA-rookie candidates -- most of a UDFA class never
plays week 1 at all):

| | MAE | RMSE | bias |
|---|---|---|---|
| model (`predicted_points_model`) | 1.223 | 1.745 | -0.042 |
| pace (`pace_prior`) | 1.410 | 1.714 | +0.315 |

Paired bootstrap over players, model - pace MAE: -0.186, CI95
[-0.418, +0.056] -- **no clear winner**. Unlike the drafted case (165 rows,
decisive pace win), there is not enough data here to move off the existing
default, and the point estimate does not even point the same direction pace
did for drafted rookies -- an UDFA debut being a bench/committee/injury-fill
role by definition (undrafted and unproven), both numbers are small and
close together, so the drafted case's assumption that the model would do
*at least as badly* here does not obviously hold.

Decision: unchanged. `_blend_toward_season_pace` still serves pure pace
(w=0) at g=0 for undrafted rookies, same as every other zero-game player --
not because pace is shown to win here, but because there isn't enough
evidence to justify anything else, and w=0 is already the default for a
player with no pace-blend evidence either way. Revisit if a future season
gives this population enough real debuts to narrow the interval.
