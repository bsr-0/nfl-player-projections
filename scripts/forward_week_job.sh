#!/bin/zsh
# Weekly forward-test job (G10 of docs/PRODUCTION_SELECTION_RULE.md), run by
# launchd (~/Library/LaunchAgents/com.benrosen.nfl-forward-test.plist) on
# Wednesdays, ahead of each Thursday deadline. Steps:
#   1. load last week's stats, rosters and schedule (the site refresh's step 1)
#   1b. load weekly PFR, NGS, snap counts and prior-season PFR for the weeks just
#       loaded (scripts/refresh_live_inputs.py; the model reads them)
#   1c. replace draft-feed placeholder ids in draft_picks_v2 with the official GSIS
#       ids nflverse has published (scripts/refresh_draft_identity.py; rookies who
#       debuted otherwise have no draft capital)
#   2. load this week's injury reports (step 2; the live forecast reads them)
#   2b. scripts/check_live_inputs.py -- fails if the loaded weeks' team-level PBP
#       columns read zero (Plan A would silently forecast a run-heavy league);
#       on failure no forecast is written and the notification says so
#   3. scripts/run_forward_week.py run -- refuses if the lineage broke, the
#      week is outside 2026 weeks 6-13, the deadline passed, or the week is
#      already written (so the second Wednesday attempt is a no-op).
# Output: data/experiments/selection_forward_2026/logs/, plus a notification.
# After week 13's deadline (2026-12-03) the job unloads itself.
set -u
REPO="/Users/benrosen/Documents/prediction_models/nfl-player-projections"
PY="$REPO/.venv/bin/python"
LOGDIR="$REPO/data/experiments/selection_forward_2026/logs"
LABEL="com.benrosen.nfl-forward-test"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(date +%Y%m%d-%H%M%S).log"
cd "$REPO" || exit 1

notify() { /usr/bin/osascript -e "display notification \"$1\" with title \"NFL forward test\"" >/dev/null 2>&1 || true; }

if [[ "$(date +%Y%m%d)" > "20261203" ]]; then
  notify "Window over (weeks 6-13); unloading the weekly job."
  /bin/launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1
  exit 0
fi

{
  echo "== $(date) auto_refresh"
  "$PY" -m src.data.auto_refresh || echo "auto_refresh exited $?"
  echo "== $(date) refresh_live_inputs"
  "$PY" scripts/refresh_live_inputs.py --seasons 2026 --write || echo "refresh_live_inputs exited $?"
  echo "== $(date) refresh_draft_identity"
  "$PY" scripts/refresh_draft_identity.py --seasons 2026 --write || echo "refresh_draft_identity exited $?"
  echo "== $(date) backfill_injuries"
  "$PY" scripts/backfill_injuries.py --seasons 2026 2026 || echo "backfill_injuries exited $?"
  echo "== $(date) check_live_inputs"
  if "$PY" scripts/check_live_inputs.py; then
    echo "== $(date) run_forward_week run"
    "$PY" scripts/run_forward_week.py run
  else
    echo "LIVE INPUT CHECK FAILED: no forecast written; fix the data, then rerun the steps by hand before the deadline"
    false
  fi
} >"$LOG" 2>&1
rc=$?  # not `status`: read-only in zsh

if [[ $rc -eq 0 ]]; then
  week=$(grep -m1 '"week":' "$LOG" | tr -dc '0-9')
  notify "Week $week forecasts frozen before the deadline."
elif grep -q "refusing to overwrite" "$LOG"; then
  : # this week was already written by the earlier attempt
elif grep -q "LIVE INPUT CHECK FAILED" "$LOG"; then
  notify "Live inputs failed the check -- NO forecast written; see $LOG"
else
  notify "Weekly run FAILED -- see $LOG"
fi
exit $rc
