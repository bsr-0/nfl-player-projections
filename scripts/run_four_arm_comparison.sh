#!/usr/bin/env bash
# Four-arm preseason comparison on corrected data.
#
# Run AFTER the Phase 7 history-NaN experiment completes: this consumes its
# winning arm's output, so the arm choice must be settled first.
#
# Produces two outputs, deliberately distinct:
#
#   1. INTERSECTED four-arm comparison -- the only defensible head-to-head.
#      Every arm scores the same players.
#
#   2. COVERAGE report -- phase7's performance on the players the other three
#      structurally exclude. Descriptive, NO comparator, because none exists:
#      candidate/production/step8 all require a prior season with MIN_GAMES=6
#      games and phase7 in cold-start does not. Reporting a head-to-head on
#      that population would be a rigged win (an unintersected phase7-vs-step8
#      run shows QB R2 0.92 vs 0.517 purely from ~22 extra near-zero players).
#
# Usage:
#   scripts/run_four_arm_comparison.sh <phase7_arm_dir> <out_dir>
# e.g.
#   scripts/run_four_arm_comparison.sh \
#     data/experiments/phase7_histnan_v3_20260827/main/arm_on \
#     data/experiments/four_arm_20260827
set -euo pipefail

ARM_DIR="${1:?usage: $0 <phase7_arm_dir> <out_dir>}"
OUT="${2:?usage: $0 <phase7_arm_dir> <out_dir>}"
SEASONS="${SEASONS:-2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025}"

mkdir -p "$OUT"
COMBINED="$OUT/phase7_combined.csv"

echo "git_commit: $(git rev-parse --short HEAD)"
echo "phase7 arm: $ARM_DIR"
echo "start: $(date -Iseconds)"

# --- assemble the phase7 input, loudly ---------------------------------------
python - "$ARM_DIR" "$COMBINED" "$SEASONS" <<'PY'
import sys, glob, os
import pandas as pd
arm_dir, out, seasons = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split()]
files = sorted(glob.glob(os.path.join(arm_dir, "phase7_*.csv")))
if not files:
    raise SystemExit(f"FATAL: no phase7_*.csv in {arm_dir}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
have = sorted(int(s) for s in df.season.unique())
missing = sorted(set(seasons) - set(have))
if missing:
    raise SystemExit(
        f"FATAL: phase7 arm covers {have}; fold set needs {seasons}; missing {missing}. "
        f"Refusing to run -- those folds would score phase7 on an empty frame "
        f"and record it as a loss.")
dupes = df.duplicated(subset=["player", "position", "season"]).sum()
if dupes:
    raise SystemExit(f"FATAL: {dupes} duplicate (player, position, season) rows")
df.to_csv(out, index=False)
print(f"phase7 input: {len(df)} rows, seasons {have} -> {out}")
PY

# --- 1. intersected four-arm --------------------------------------------------
echo
echo "##### 1/2  INTERSECTED FOUR-ARM (same players, all arms) #####"
python scripts/walk_forward_preseason.py \
  --test-seasons $SEASONS \
  --phase7-csv "$COMBINED" \
  --require-all-arms \
  --intersect-populations \
  2>&1 | tee "$OUT/four_arm_intersected.log"

# --- 2. coverage report -------------------------------------------------------
echo
echo "##### 2/2  COVERAGE: phase7 on the players the others exclude #####"
python - "$COMBINED" "$OUT" "$SEASONS" <<'PY'
import sys, sqlite3
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from config.settings import DB_PATH
from src.models.preseason_features import MIN_GAMES

combined, out, seasons = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split()]
p7 = pd.read_csv(combined)
p7 = p7[p7.season.isin(seasons)]

c = sqlite3.connect(str(DB_PATH))
g = pd.read_sql(
    "SELECT player_id, season, COUNT(*) g FROM player_weekly_stats "
    "WHERE week <= 18 GROUP BY player_id, season", c)
first = pd.read_sql(
    "SELECT player_id, MIN(season) fs FROM player_weekly_stats GROUP BY player_id",
    c).set_index("player_id").fs
c.close()

# Eligible for the other arms = had a PRIOR season with >= MIN_GAMES games.
elig = g[g.g >= MIN_GAMES].copy()
elig["season"] = elig.season + 1
elig = set(map(tuple, elig[["player_id", "season"]].values))

p7["eligible_elsewhere"] = [(r.player, r.season) in elig for r in p7.itertuples()]
p7["is_rookie"] = [first.get(r.player, 9999) == r.season for r in p7.itertuples()]
p7["abs_err"] = (p7.predicted_season_total - p7.actual_season_total).abs()

rows = []
for pos in ["QB", "RB", "WR", "TE"]:
    d = p7[p7.position == pos]
    for label, sub in (("comparable (other arms can score)", d[d.eligible_elsewhere]),
                       ("EXCLUDED by other arms", d[~d.eligible_elsewhere])):
        if len(sub):
            rows.append({"position": pos, "group": label, "n": len(sub),
                         "rookie_pct": round(100 * sub.is_rookie.mean(), 1),
                         "phase7_MAE": round(sub.abs_err.mean(), 2),
                         "mean_actual": round(sub.actual_season_total.mean(), 1)})
r = pd.DataFrame(rows)
r.to_csv(f"{out}/phase7_coverage.csv", index=False)
print(r.to_string(index=False))
print("\nNo comparator column by design: no other arm can score the excluded")
print("group at all, so any head-to-head there would be uncontested.")
PY

echo
echo "DONE $(date -Iseconds)"
echo "outputs in $OUT"
