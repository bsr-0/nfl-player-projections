"""Replay 2026 game predictions as of each week: fixed data vs the stale state that was served."""
import shutil, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
SP = Path(sys.argv[1]); sys.path.insert(0, str(SP))
from compare_states import predict, ROOT
import pandas as pd
cur = sqlite3.connect(f"file:{ROOT / 'data/nfl_data.db'}?mode=ro", uri=True)
actual = pd.read_sql("SELECT season, week, home_team, away_team, home_score, away_score FROM schedule "
                     "WHERE season = 2026 AND home_score IS NOT NULL", cur)
# fixed: blank scores from week w on, predict week w (lagged features look strictly before w)
fixed_db = SP / "replay_fixed.db"; shutil.copyfile(SP / "both_fixed.db", fixed_db)
con = sqlite3.connect(str(fixed_db)); parts = []
for w in (5, 4, 3, 2):
    con.execute("UPDATE schedule SET home_score = NULL, away_score = NULL WHERE season = 2026 AND week >= ?", (w,)); con.commit()
    out, _ = predict(con, 2026, w); parts.append(out.assign(state="fixed"))
con.close()
# stale: what the served path saw from week 2 on (only week 1 scored, team_stats NULL weeks 2-4)
con = sqlite3.connect(str(SP / "before_fixes.db"))
for w in (2, 3, 4, 5):
    out, _ = predict(con, 2026, w); parts.append(out.assign(state="stale"))
con.close()
p = pd.concat(parts).merge(actual, on=["season", "week", "home_team", "away_team"])
p["home_win"] = (p.home_score > p.away_score).astype(float); p["margin"] = p.home_score - p.away_score
p["total"] = p.home_score + p.away_score
p.to_pickle(SP / "replay.pkl"); print(p.groupby(["state", "week"]).size().to_string())
