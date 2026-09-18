# Rookie debut-stub measurement (2026-09-18)

Question: for a drafted rookie with zero games played this season, is the
weekly ensemble's own prediction (built from `_drafted_rookie_stub_rows` --
draft capital, combine, college, team context; every own-history feature
NaN) better than the Step 8 season pace it would otherwise be shrunk to
100%?

Method: `NFLPredictor.predict(as_of=(season, 1))` for seasons 2022-2025,
restricted to players in that season's `draft_picks_v2` (QB/RB/WR/TE) who
had zero games before week 1 -- i.e., every rookie who could conceivably
debut in week 1 -- scored against their REAL week-1 outcome where they
actually played (165 of 291 candidates suited up).

Result: pace wins clearly.

| | MAE | RMSE | bias (pred - actual) |
|---|---|---|---|
| model (`predicted_points_model`) | 4.20 | 6.51 | -3.17 |
| pace (`pace_prior`) | 3.67 | 5.11 | -0.20 |

By position (n, MAE model / pace): QB 9, 7.90/5.91 -- RB 42, 4.35/3.45 --
TE 36, 2.49/2.74 (the one position where the model is close, even
slightly ahead) -- WR 78, 4.48/3.97. See
`rookie_debut_stub_2022_2025.csv` for the full row-level data.

Why: the model's volume-driving features (targets/carries/snap-share
rolling means) are all structurally missing for a debut row and get
median-imputed to a league-wide typical, which is a LOW-usage player --
but the population that actually suits up in week 1 as a rookie is
selected for being fantasy-relevant, so the model systematically
undershoots (mean predicted 1.63 vs. actual mean 4.80).

Decision: `_blend_toward_season_pace` keeps w=0 (100% pace) at g=0 for
these rows, same as any other zero-game player. The stub still makes the
rookie visible (previously he had no row at all) and computes a real,
inspectable `predicted_points_model` for transparency/future work, but
that number is not served.

Revisit if the weekly model gains features that describe an incoming
player's PROJECTED role (a real usage forecast) rather than only his draft
pedigree -- at that point this comparison should be rerun.
