import numpy as np
import pandas as pd

from src.evaluation import simulation_forward as F


def _game():
    rows = []
    for team in ("H", "A"):
        for pos, pts in (("QB", 18.0), ("WR", 12.0), ("WR", 9.0), ("TE", 6.0), ("RB", 10.0)):
            rows.append({"game_id": "2026_05_A_H", "player_id": f"{team}{pos}{pts:.0f}", "team": team,
                         "position": pos, "predicted_points": pts, "home_team": "H", "away_team": "A"})
    return pd.DataFrame(rows)


def _draws(g, rng):
    pred = g["predicted_points"].to_numpy()
    m = pred + rng.normal(0, 5, size=(2000, len(g)))
    return {"role_factor": m, "independent": m[:, rng.permutation(len(g))] * 0 + m}


def test_interval_hits():
    s = np.arange(1, 101, dtype=float)
    assert F.interval_hits(s, 50.0) == (1.0, 1.0)
    assert F.interval_hits(s, 85.0) == (0.0, 1.0)
    assert F.interval_hits(s, 95.0) == (0.0, 0.0)


def test_populations_differ_on_who_counts():
    rng = np.random.default_rng(0)
    g = _game()
    actual = pd.Series({pid: 10.0 for pid in g["player_id"]})
    actual = actual.drop("HWR12")                               # did not play
    team_totals = {"H": 70.0, "A": 60.0}                        # include players the simulation did not list
    rows = pd.DataFrame(F.score_game(g, _draws(g, rng), actual, team_totals))
    pub = rows[(rows.population == "published") & (rows.arm == "role_factor")]
    played = rows[(rows.population == "played") & (rows.arm == "role_factor")]
    # published team total is scored against the real total; played sums only the players who played
    assert pub[(pub.kind == "team_total") & (pub.team == "H")]["actual"].iloc[0] == 70.0
    assert played[(played.kind == "team_total") & (played.team == "H")]["actual"].iloc[0] == 40.0
    # the published H stack keeps the WR who did not play, at 0
    assert pub[(pub.kind == "stack") & (pub.team == "H")]["actual"].iloc[0] == 10.0 + 0.0 + 10.0
    assert len(pub[pub.kind == "player"]) == 9 and pub[pub.kind == "game_total"]["actual"].iloc[0] == 130.0


def test_correlation_verdict_keeps_only_when_the_upper_bound_is_below_zero():
    def scores(diffs):
        out = []
        for i, d in enumerate(diffs):
            for team in ("H", "A"):
                out.append({"game_id": f"g{i}", "team": team, "population": "played", "kind": "stack", "arm": "independent", "crps": 10.0})
                out.append({"game_id": f"g{i}", "team": team, "population": "played", "kind": "stack", "arm": "role_factor", "crps": 10.0 + d})
        return pd.DataFrame(out)
    keep = F.correlation_verdict(scores([-0.2, -0.1, -0.15, -0.12, -0.18]))
    assert keep["verdict"] == "KEEP" and keep["n_games"] == 5 and keep["n_stacks"] == 10
    assert F.correlation_verdict(scores([-0.3, 0.3, -0.2, 0.25, -0.1]))["verdict"] == "REVERT"
    assert F.correlation_verdict(scores([-0.1]))["verdict"] == "REVERT"      # one game: no SE


def test_display_gate_bands():
    rows = []
    for kind, c80, c50 in (("player", 0.8, 0.5), ("stack", 0.70, 0.5), ("team_total", 0.8, 0.35), ("game_total", 0.88, 0.6)):
        n = 100
        for i in range(n):
            rows.append({"population": "published", "arm": "role_factor", "kind": kind, "crps": 1.0,
                         "cov80": float(i < c80 * n), "cov50": float(i < c50 * n)})
    gate = F.display_gate(pd.DataFrame(rows))
    assert gate["player"]["show"] and gate["game_total"]["show"]
    assert not gate["stack"]["show"] and not gate["team_total"]["show"]
