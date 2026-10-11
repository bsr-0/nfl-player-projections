"""Score an as-of replay of the served weekly model against what happened.

docs/data/weekly_{season}_wk{N}.json holds what was published before kickoff.
Those rows were built from whatever the database held at the time: weeks 2-4 of
2026 were forecast while the PBP-derived team columns were still empty and the
schedule had scores for week 1 only (GAPS.md, "Week 6 rehearsal"). The replay
runs today's serving path -- the same weekly ensemble and pace blend -- for each
completed week as of that week, on the repaired data, and scores it on the same
terms as the published record. It answers "how does the pipeline as it now
stands do out of sample", which the published rows cannot.

It is a replay, not a publication, and the page says so. What keeps it out of
sample: the weekly models were fitted on seasons before 2026
(data/models/model_metadata.json); the forecast universe, rookie teams and
positions are recomputed as of the week (scripts/build_prekickoff_population.py,
the same helper the selection evidence used); every form feature reads only
earlier games. What it cannot remove: injury and weather inputs are the stored
values, and the 2026 participation/personnel features are empty in the replay
exactly as they are live.

Scored like the published record: players with a projection and a stat line.
Players projected but with no stat line are counted, not scored; players with a
stat line and no projection are counted and the biggest are listed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.published_record import POSITIONS, attach_actuals, player_metrics

ROW_COLUMNS = ["player_id", "name", "team", "position", "opponent", "predicted_points", "predicted_points_model",
               "pace_prior", "pace_weight", "games_played_season", "prediction_ci80_lower",
               "prediction_ci80_upper", "actual_points"]


def score_replay(rows: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(scored rows, stat lines with no projection, summary) for one replayed week.

    `rows` carry player_id and predicted_points; `actuals` are the week's stat
    lines of the teams that played (published_record.player_actuals).
    """
    rows = rows[rows["position"].isin(POSITIONS)].drop_duplicates("player_id").reset_index(drop=True)
    acts = actuals[actuals["position"].isin(POSITIONS)]
    scored, unprojected = attach_actuals(rows, acts)
    have = scored["actual_points"].notna()
    summary = {"n_projected": int(len(scored)), "n_no_stat_line": int((~have).sum()),
               "n_unprojected": int(len(unprojected)),
               "n_unprojected_5plus": int((unprojected["actual_points"] >= 5).sum()) if len(unprojected) else 0}
    m = player_metrics(scored)
    if m:
        summary.update(m)
        summary["by_position"] = {p: player_metrics(scored[scored["position"] == p])
                                  for p in POSITIONS if (scored["position"] == p).any()}
    summary["unprojected_top"] = (unprojected.sort_values("actual_points", ascending=False).head(10)
                                  [["name", "team", "position", "actual_points"]].round(1).to_dict("records"))
    return scored[have].reset_index(drop=True), unprojected, summary


def compare_with_published(replay: pd.DataFrame, published: pd.DataFrame) -> dict | None:
    """Replay vs published error on the players both cover, so the two are compared like for like."""
    if replay.empty or published.empty or "player_id" not in published.columns:
        return None
    pub = published[published["actual_points"].notna() & published["predicted_points"].notna()]
    pub = pub.drop_duplicates("player_id").set_index("player_id")
    both = replay[replay["player_id"].isin(pub.index)]
    if both.empty:
        return None
    a = both["actual_points"].to_numpy(float)
    r = both["predicted_points"].to_numpy(float)
    p = pub.loc[both["player_id"], "predicted_points"].to_numpy(float)
    return {"n_common": int(len(both)),
            "mae_published": round(float(np.abs(p - a).mean()), 2), "mae_replay": round(float(np.abs(r - a).mean()), 2),
            "bias_published": round(float((p - a).mean()), 2), "bias_replay": round(float((r - a).mean()), 2),
            "mean_abs_change": round(float(np.abs(r - p).mean()), 2)}
