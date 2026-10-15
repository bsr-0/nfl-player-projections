# Week 6 target-week override (2026-10-14)

**Problem.** `scripts/run_forward_week.py` takes the week from
`get_prediction_target_week()` -> `nfl_calendar.get_next_n_nfl_weeks`, which
names week N current from that week's first Thursday 00:00 local. The runner's
deadline for week N is the same Thursday 00:00 US/Eastern. A real run could
never start before the deadline (the week-5 dry run worked only because it does
not enforce one).

**Change.** `src/utils/nfl_calendar.py` (not in the lineage hash):
`get_next_n_nfl_weeks` honours env `NFL_TARGET_WEEK=SEASON:WEEK` when called
without a date. Unset, behaviour is unchanged. No pinned file was edited, so
the lineage check passed. `get_current_nfl_week` still reports the real date.

**Runs.** `NFL_TARGET_WEEK=2026:6 python scripts/run_forward_week.py dry-run`
(started 2026-10-15T01:40Z) and `... run` (written 2026-10-15T01:50:02Z,
`before_deadline: true`, deadline 04:00Z). Both give forecasts sha256
`744f5c67f78bb67164f57db85bcd35a21679d818b44734305df6e1973c40b271`
(631 players). Weeks 7-13 need the same variable, set to that week, because
the runner's deadline is again earlier than the calendar's change of week.

**Not examined.** No forecast accuracy or per-player value was viewed. Checks
were structural only: list size, bye teams absent, finite values.
