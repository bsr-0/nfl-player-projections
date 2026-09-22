"""Plan B: joint/hierarchical team model.

See docs/TEAM_LEVEL_ALLOCATION_MODELS.md's Plan B section. Where Plan A
(src/models/team_allocation/) predicts each player's share of team volume
independently and only enforces "sums to 1" as a post-hoc renormalization,
this package models the whole roster jointly per team-week via a mixed-
effects regression: a team-week random intercept captures shared game-
environment variance (game script, pace, weather) that Plan A's independent
per-player models have no way to represent at all.

Reuses Plan A's data (team_week_player_shares) and reconstruction utilities
(src/models/team_allocation/reconstruct.py) unmodified -- this predicts the
identical share targets, just with a different model structure, so the two
are a fair, apples-to-apples comparison.
"""
