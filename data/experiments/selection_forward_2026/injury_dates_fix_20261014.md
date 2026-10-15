# Injury-date loader fix before week 6 (2026-10-13/14)

**What broke.** nflverse now publishes 2025 and 2026 injury reports without a
`date_modified` column; 2024 and earlier keep it (tz-aware, microsecond).
`InjuryDataLoader.load_injuries` concatenates seasons, so the column mixed
Timestamps and NaN, and `pd.to_datetime(..., utc=True)` in
`get_player_injury_status` raised a bare `AssertionError`. Both callers catch
it and skip the whole external block (Vegas lines, weather, injury status):
`_prepare_training_data` ("External features ... skipped") and
`src/predict.py::_prepare_features` ("WARNING: external ... skipped at serving").
Any load spanning a pre-2025 season hit it; serving loads every season.
When it started is not known (no run logs); the 2025 served fold of
2026-10-07 was not affected.

**Fix.** `src/data/external_data.py`, `InjuryDataLoader.load_injuries`: after
the concat, `date_modified` is parsed row by row (ISO strings, UTC). Seasons
without the column get NaT, so their reports are kept (no timestamp to check),
exactly as before for a single 2025/2026 load. The post-kickoff leak guard is
unchanged. Test: `tests/test_external_injury_dates.py` (fails on the old code
with the same AssertionError, passes now).

**Forward test.** `src/data/external_data.py` is not in the lineage hash
(`scripts/run_forward_week.py` SERVED_FILES / SCRIPTS), so the lineage is
intact; no pinned file changed. The fix lands before the first window week, so
weeks 6-13 all run with it. As with the other data/loader fixes, this moves
inputs without a re-freeze. No forward dry run was made (it would compute the
blinded candidate forecasts).
