import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.evaluation import published_record as R


def _commit(repo, files: dict, when: str):
    for rel, payload in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", when],
                   cwd=repo, check=True, env=env)


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    R._json_at.cache_clear()
    return tmp_path


def _games(rows):
    df = pd.DataFrame(rows, columns=["week", "home_team", "away_team", "kickoff", "home_score", "away_score"])
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df["final"] = df["home_score"].notna() & df["away_score"].notna()
    return df


def test_snapshot_before_is_the_newest_strictly_before_kickoff():
    versions = [("c3", 300), ("c2", 200), ("c1", 100)]
    assert R.snapshot_before(versions, 250) == ("c2", 200)
    assert R.snapshot_before(versions, 200) == ("c1", 100)
    assert R.snapshot_before(versions, 100) is None


def test_each_game_uses_the_last_version_committed_before_its_own_kickoff(repo):
    wk = "docs/data/weekly_2026_wk2.json"
    meta = "docs/data/weekly_meta.json"
    rows_v1 = [{"player_id": "a", "name": "A.A", "team": "KC", "position": "QB", "predicted_points": 10.0},
               {"player_id": "b", "name": "B.B", "team": "BUF", "position": "WR", "predicted_points": 8.0}]
    rows_v2 = [{**r, "predicted_points": r["predicted_points"] + 5} for r in rows_v1]
    rows_v3 = [{**r, "predicted_points": 99.0} for r in rows_v1]
    _commit(repo, {wk: rows_v1, meta: {"mode": "weekly_model"}}, "2026-09-16T12:00:00+00:00")
    _commit(repo, {wk: rows_v2, meta: {"mode": "weekly_blend"}}, "2026-09-19T12:00:00+00:00")
    _commit(repo, {wk: rows_v3, meta: {"mode": "weekly_blend"}}, "2026-09-25T12:00:00+00:00")  # after every kickoff
    games = _games([(2, "KC", "DEN", "2026-09-18T00:15:00Z", 24.0, 20.0),     # Thursday: before v2
                    (2, "BUF", "MIA", "2026-09-20T17:00:00Z", 30.0, 10.0),    # Sunday: after v2
                    (2, "NYJ", "NE", "2026-09-15T17:00:00Z", None, None)])    # before any version
    now = pd.Timestamp("2026-10-01", tz="UTC")
    rec = R.published_rows("weekly", 2026, 2, games, now, repo)
    got = dict(zip(rec.rows["team"], rec.rows["predicted_points"]))
    assert got == {"KC": 10.0, "BUF": 13.0}
    assert dict(zip(rec.rows["team"], rec.rows["published_mode"])) == {"KC": "weekly_model", "BUF": "weekly_blend"}
    assert rec.not_published == ["NE@NYJ"]
    assert [s["mode"] for s in rec.snapshots] == ["weekly_model", "weekly_blend"]


def test_games_that_have_not_kicked_off_are_not_taken_from_history(repo):
    _commit(repo, {"docs/data/game_predictions_2026_wk5.json":
                   [{"home_team": "KC", "away_team": "DEN", "home_win_prob_logistic": .6}]}, "2026-10-01T00:00:00+00:00")
    games = _games([(5, "KC", "DEN", "2026-10-11T17:00:00Z", None, None)])
    rec = R.published_rows("game", 2026, 5, games, pd.Timestamp("2026-10-10", tz="UTC"), repo)
    assert rec.rows.empty and rec.not_published == []


def test_actuals_join_by_id_then_name_then_initial_and_surname():
    rows = pd.DataFrame([
        {"player_id": "id1", "name": "X.One", "team": "KC", "position": "QB", "predicted_points": 20.0},
        {"player_id": None, "name": "B.Robinson", "team": "ATL", "position": "RB", "predicted_points": 15.0},
        {"player_id": None, "name": "M.Wilson", "team": "ARI", "position": "WR", "predicted_points": 9.0},
        {"player_id": None, "name": "J.Dup", "team": "NYJ", "position": "WR", "predicted_points": 5.0},
        {"player_id": None, "name": "J.Dup", "team": "NYJ", "position": "WR", "predicted_points": 2.0},
    ])
    acts = pd.DataFrame([
        {"player_id": "id1", "name": "X.One", "team": "KC", "position": "QB", "actual_points": 22.0},
        {"player_id": "id2", "name": "Bi.Robinson", "team": "ATL", "position": "RB", "actual_points": 31.3},
        {"player_id": "id3", "name": "M.Wilson", "team": "ARI", "position": "WR", "actual_points": 10.6},
        {"player_id": "id4", "name": "J.Dup", "team": "NYJ", "position": "WR", "actual_points": 7.0},
        {"player_id": "id5", "name": "C.Wentz", "team": "MIN", "position": "QB", "actual_points": 19.2},
    ])
    out, unmatched = R.attach_actuals(rows, acts)
    assert out["actual_points"].tolist()[:3] == [22.0, 31.3, 10.6]
    assert out["actual_points"].iloc[3:].isna().all()            # two different J.Dup rows: ambiguous
    assert sorted(unmatched["player_id"]) == ["id4", "id5"]


def test_repeated_identical_listings_count_once():
    rows = pd.DataFrame([{"name": "A.A", "team": "KC", "position": "WR", "predicted_points": 5.0}] * 2
                        + [{"name": "B.B", "team": "KC", "position": "WR", "predicted_points": 5.0}])
    out, dropped = R.collapse_duplicate_rows(rows)
    assert dropped == 1 and len(out) == 2


def test_player_metrics_score_only_rows_with_an_actual_and_report_range_coverage():
    rows = pd.DataFrame({"predicted_points": [10.0, 5.0, 8.0], "actual_points": [12.0, 5.0, np.nan],
                         "prediction_ci80_lower": [4.0, 6.0, 1.0], "prediction_ci80_upper": [11.0, 9.0, 15.0]})
    m = R.player_metrics(rows)
    assert m["n"] == 2 and m["mae"] == 1.0 and m["bias"] == -1.0
    assert m["coverage80"] == 0.0 and m["n_with_range"] == 2


def test_game_metrics_compare_with_the_market_and_count_pushes():
    rows = pd.DataFrame([
        # home won by 7; model favoured home; line home -3 (spread_line +3): favorite right
        {"home_score": 27, "away_score": 20, "home_win_prob_logistic": .7, "spread_line": 3.0, "total_line": 44.0,
         "predicted_margin_ridge": 5.0, "predicted_total_ridge": 50.0, "final": True},
        # away won by 3; model favoured home (miss); line home +3 (spread_line -3): favorite right; margin -3 = push
        {"home_score": 17, "away_score": 20, "home_win_prob_logistic": .55, "spread_line": -3.0, "total_line": 40.0,
         "predicted_margin_ridge": 1.0, "predicted_total_ridge": 45.0, "final": True},   # total 37 < 40: over call loses
        {"home_score": None, "away_score": None, "home_win_prob_logistic": .5, "spread_line": 1.0, "total_line": 40.0,
         "predicted_margin_ridge": 0.0, "predicted_total_ridge": 41.0, "final": False},
    ])
    m = R.game_metrics(rows)
    assert m["n"] == 2
    assert m["win_loss"]["logistic"]["correct"] == 1 and m["win_loss"]["market_favorite"]["correct"] == 2
    assert m["margin"]["ridge"]["ats_won"] == 1 and m["margin"]["ridge"]["ats_push"] == 1
    assert m["total"]["ridge"]["ou_won"] == 1 and m["total"]["ridge"]["ou_lost"] == 1
    assert m["margin"]["market_line"]["mae"] == pytest.approx(((7 - 3) + abs(-3 + 3)) / 2)
