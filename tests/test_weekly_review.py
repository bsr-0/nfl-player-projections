"""The live-test review: its numbers, its blinding, and where its page goes.

The calculations most likely to go wrong each have a test here: the
game-resampled intervals, the paired baseline comparison, the sign of the
signed error, the accounting of projections with no stat line, the handling of
incomplete weeks, the joins, and the rules that turn intervals into sentences.
"""
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.evaluation import weekly_review as wr

MODULE = PROJECT_ROOT / "src" / "evaluation" / "weekly_review.py"
SCRIPT = PROJECT_ROOT / "scripts" / "render_weekly_review.py"
IMPORTANCE = PROJECT_ROOT / "scripts" / "feature_group_importance.py"


def _players(n_games=4, per_game=5, week=3, mode="weekly_blend"):
    rows = []
    for g in range(n_games):
        for i in range(per_game):
            pred = 10.0 + i
            rows.append({"week": week, "game": f"A{g}@H{g}", "player_id": f"p{week}{g}{i}",
                         "name": f"P{g}{i}", "team": f"H{g}", "position": "WR",
                         "predicted_points": pred, "prediction_ci80_lower": pred - 5,
                         "prediction_ci80_upper": pred + 5,
                         "actual_points": pred + (2.0 if i % 2 else -2.0) + (8 if i == 0 else 0),
                         "rolling3": pred + 3.0, "final": True, "published_mode": mode,
                         "published_commit": "abc", "published_at": "2026-09-20T17:00+00:00",
                         "kickoff": "2026-09-21T17:00+00:00"})
    df = pd.DataFrame(rows)
    df["gkey"] = "wk" + df["week"].astype(str) + " " + df["game"]
    return df


def _module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- intervals --------------------------------------------------------------------

def test_mean_ci_equals_a_brute_force_game_bootstrap():
    """The per-game-sum shortcut must give exactly what concatenating the
    resampled games gives, draw for draw."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=60)
    games = np.repeat([f"g{i}" for i in range(12)], 5)
    ci = wr.mean_ci(v, games, b=400, seed=7)
    codes, uniq = pd.factorize(games)
    k = len(uniq)
    w = np.random.default_rng(7).multinomial(k, np.full(k, 1 / k), size=400)
    sims = [np.concatenate([np.tile(v[codes == j], c) for j, c in enumerate(row)]).mean() for row in w]
    lo, hi = np.percentile(sims, [2.5, 97.5])
    assert ci["lo"] == pytest.approx(lo) and ci["hi"] == pytest.approx(hi)
    assert ci["est"] == pytest.approx(v.mean()) and ci["n"] == 60 and ci["games"] == 12


def test_mean_ci_is_reproducible_and_needs_two_games():
    v, g = np.arange(10.0), ["a"] * 5 + ["b"] * 5
    assert wr.mean_ci(v, g) == wr.mean_ci(v, g)
    one = wr.mean_ci(v, ["a"] * 10)
    assert one["lo"] is None and one["hi"] is None and one["est"] == pytest.approx(4.5)


def test_coverage_interval_resamples_games_not_players():
    """Players in one game share shocks; when whole games land in or out of
    range together the interval must be wider than a player-level binomial."""
    rows = _players(n_games=10)
    miss = rows["game"].isin([f"A{g}@H{g}" for g in range(3)])
    rows.loc[miss, "actual_points"] = rows.loc[miss, "prediction_ci80_upper"] + 1
    rows.loc[~miss, "actual_points"] = rows.loc[~miss, "predicted_points"]
    cov = wr.stats_for(rows)["cov80"]
    assert cov["est"] == pytest.approx(0.7)
    from scipy import stats
    binom = stats.binomtest(35, 50).proportion_ci(0.95, method="exact")
    assert (cov["hi"] - cov["lo"]) > (binom.high - binom.low)


# --- the three headline measures ---------------------------------------------------

def test_signed_error_is_projected_minus_actual():
    rows = _players(n_games=6)
    rows["actual_points"] = rows["predicted_points"] - 2.0           # projected 2 points high
    assert wr.stats_for(rows)["bias"]["est"] == pytest.approx(2.0)
    rows["actual_points"] = rows["predicted_points"] + 3.0           # projected low
    assert wr.stats_for(rows)["bias"]["est"] == pytest.approx(-3.0)


def test_baseline_comparison_uses_identical_player_weeks():
    rows = _players(n_games=6)
    rows.loc[rows.index[:7], "rolling3"] = np.nan                    # no three-game history
    h = wr.stats_for(rows)
    p = rows.dropna(subset=["rolling3"])
    model = (p["predicted_points"] - p["actual_points"]).abs().mean()
    base = (p["rolling3"] - p["actual_points"]).abs().mean()
    assert h["n_paired"] == h["model_mae_paired"]["n"] == h["base_mae"]["n"] == h["vs_rolling3"]["n"] == 23
    assert h["model_mae_paired"]["est"] == pytest.approx(model)
    assert h["base_mae"]["est"] == pytest.approx(base)
    assert h["vs_rolling3"]["est"] == pytest.approx(model - base)
    assert h["mae"]["n"] == 30                                        # the overall MAE keeps every row


def test_above_and_below_range_are_split_by_side():
    rows = _players(n_games=6)
    rows.loc[rows.index[:3], "actual_points"] = rows.loc[rows.index[:3], "prediction_ci80_lower"] - 1
    h = wr.stats_for(rows)
    below = (rows["actual_points"] < rows["prediction_ci80_lower"]).mean()
    above = (rows["actual_points"] > rows["prediction_ci80_upper"]).mean()
    assert h["below"]["est"] == pytest.approx(below) and h["above"]["est"] == pytest.approx(above)
    assert h["cov80"]["est"] == pytest.approx(1 - below - above)
    assert h["width"] == pytest.approx(10.0)


def test_rows_without_a_range_leave_coverage_but_stay_in_mae():
    rows = _players(n_games=6)
    rows.loc[rows.index[:10], ["prediction_ci80_lower", "prediction_ci80_upper"]] = np.nan
    h = wr.stats_for(rows)
    assert h["n"] == 30 and h["n_range"] == 20 and h["cov80"]["n"] == 20


def test_unfinished_games_are_not_scored():
    rows = _players()
    rows.loc[rows["game"] == "A0@H0", "final"] = False
    assert wr.player_section(rows, 3, {})["n"] == 15


# --- every published projection ------------------------------------------------------

def _unscored():
    return pd.DataFrame([
        {"player_id": "a", "participation_state": "confirmed_played", "roster_status": "ACT", "predicted_points": 6},
        {"player_id": "b", "participation_state": "confirmed_zero_snaps", "roster_status": "ACT", "predicted_points": 2},
        {"player_id": "c", "participation_state": "unknown", "roster_status": "INA", "predicted_points": 9},
        {"player_id": "d", "participation_state": "unknown", "roster_status": "ACT", "predicted_points": 7},
        {"player_id": "e", "participation_state": None, "roster_status": None, "predicted_points": 1},
        {"player_id": None, "participation_state": None, "roster_status": None, "predicted_points": 3},
        {"player_id": "g", "participation_state": "unknown", "roster_status": "DEV", "predicted_points": 0.5},
    ])


def test_no_stat_line_classes_need_evidence():
    cls = wr.classify_no_stat(_unscored())
    assert cls["class"].tolist() == [wr.PLAYED_NO_STAT, wr.VERIFIED_DNP, wr.VERIFIED_DNP, wr.UNRESOLVED,
                                     wr.UNRESOLVED, wr.UNRESOLVED, wr.VERIFIED_DNP]
    assert cls["reason"].tolist()[3:6] == ["active, no snap row", "no roster or snap record",
                                           "no player id (week 1 names)"]


def test_accounting_reconciles_and_never_scores_the_unscored():
    rows = _players(n_games=6)
    u = _unscored().assign(week=3, game="A0@H0", gkey="wk3 A0@H0", final=True, actual_points=np.nan,
                           position="WR", team="H0", rolling3=np.nan, prediction_ci80_lower=np.nan,
                           prediction_ci80_upper=np.nan)
    both = pd.concat([rows, u], ignore_index=True)
    ps = wr.player_section(both, 3, {3: {"n_unprojected": 4, "n_unprojected_5plus": 1}})
    a = ps["accounting"]
    assert a["published"] == 37 and a["scored"] == 30 and a["no_stat_line"] == 7
    assert sum(c["n"] for c in a["classes"]) == 7
    assert {c["label"]: c["projected5"] for c in a["classes"]} == {wr.VERIFIED_DNP: 1, wr.PLAYED_NO_STAT: 1,
                                                                    wr.UNRESOLVED: 1}
    assert a["unprojected"] == 4 and a["unprojected_5plus"] == 1
    # The unscored rows change nothing in the accuracy measures.
    alone = wr.player_section(rows, 3, {})
    assert ps["headline"]["mae"] == alone["headline"]["mae"] and ps["n"] == 30


def test_accounting_works_without_roster_columns():
    rows = _players(n_games=3)
    rows.loc[0, "actual_points"] = np.nan
    a = wr.player_section(rows, 3, {})["accounting"]
    assert a["no_stat_line"] == 1 and [c["n"] for c in a["classes"]] == [0, 0, 1]


# --- weeks and model versions ----------------------------------------------------------

def test_incomplete_weeks_are_flagged_and_kept_out():
    rows = pd.concat([_players(n_games=4, week=3), _players(n_games=1, week=4)], ignore_index=True)
    games = pd.DataFrame({"week": [3] * 4 + [4] * 4})
    ts = wr.test_set(rows, games, [3, 4])
    assert [w["complete"] for w in ts] == [True, False]
    complete = [w["week"] for w in ts if w["complete"]]
    ps = wr.player_section(rows, 4, {}, weeks=complete)
    assert ps["weeks"] == [3] and ps["n"] == 20
    assert {c["check"]: c["ok"] for c in wr.audit(rows, ps, ts)}["Headline uses complete weeks only"]
    bad = wr.player_section(rows, 4, {}, weeks=[3, 4])
    assert not {c["check"]: c["ok"] for c in wr.audit(rows, bad, ts)}["Headline uses complete weeks only"]


def test_version_table_splits_models_within_a_week_and_pools_complete_weeks():
    a = _players(n_games=3, week=2, mode="weekly_blend")
    b = _players(n_games=1, week=2, mode="weekly_model").assign(game="X@Y", team="Y", gkey="wk2 X@Y",
                                                                 player_id=lambda d: "m" + d["player_id"])
    c = _players(n_games=4, week=3, mode="weekly_blend")
    rows = pd.concat([a, b, c], ignore_index=True)
    ts = wr.test_set(rows, pd.DataFrame({"week": [2] * 4 + [3] * 4}), [2, 3])
    vt = wr.version_table(rows, ts)
    assert [(r["week"], r["mode"], r["n"]) for r in vt["rows"]] == [(2, "weekly_model", 5), (2, "weekly_blend", 15),
                                                                     (3, "weekly_blend", 20)]
    pooled = {r["mode"]: r for r in vt["pooled"]}
    assert pooled["weekly_blend"]["n"] == 35 and pooled["weekly_blend"]["weeks"] == [2, 3]
    assert vt["versions_in_headline"] == ["weekly_blend", "weekly_model"]


# --- intervals agree across sections; the audit -------------------------------------------

def test_same_rows_show_the_same_interval_everywhere():
    rows = _players(n_games=8)
    rows.loc[rows.index[::2], "position"] = "TE"
    ps = wr.player_section(rows, 3, {})
    table = {r["label"]: r["cov80"] for r in ps["ranges"]["rows"]}
    forest = {r["label"]: r["cov80"] for g in ps["segments"] if g["group"] == "Position" for r in g["rows"]}
    panel = {r["label"]: r["cov80"] for r in ps["failure_panel"]}
    assert table["All"] == ps["headline"]["cov80"]
    assert table["TE"] == forest["TE"] == panel["TE"] and table["WR"] == forest["WR"]


def test_audit_passes_clean_rows_and_catches_planted_faults():
    rows = _players(n_games=6)
    ts = wr.test_set(rows, pd.DataFrame({"week": [3] * 6}), [3])
    ps = wr.player_section(rows, 3, {})
    checks = wr.audit(rows, ps, ts)
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    late = rows.copy()
    late.loc[0, "published_at"] = "2026-09-21T18:00+00:00"           # after its kickoff
    dup = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)         # one player scored twice
    wrong = rows.copy()
    wrong.loc[0, "team"] = "ZZZ"                                       # player not in his game
    for bad, name in [(late, "Every projection was committed before its game's kickoff"),
                      (dup, "One projection per player per week"),
                      (wrong, "Each scored player's team played in that game")]:
        res = {c["check"]: c["ok"] for c in wr.audit(bad, wr.player_section(bad, 3, {}), ts)}
        assert not res[name]


def test_joins_never_match_missing_ids_or_add_rows():
    rows = pd.DataFrame({"player_id": ["a", None], "week": [1, 1]})
    right = pd.DataFrame({"player_id": ["a", "a", None], "week": [1, 1, 1], "x": [1.0, 2.0, 9.0]})
    out = wr._attach(rows, right, ["player_id", "week"])
    assert len(out) == 2 and out["x"].tolist()[0] == 2.0 and pd.isna(out["x"].tolist()[1])


# --- sentences follow the intervals -----------------------------------------------------------

def _ps(d, b, c, qb_bias=None, te_cov=None):
    ci = lambda e, lo, hi: {"est": e, "lo": lo, "hi": hi, "n": 100, "games": 40}
    head = {"n": 100, "games": 40, "mae": ci(4.5, 4.2, 4.8), "vs_rolling3": ci(*d), "bias": ci(*b),
            "cov80": ci(*c)}
    panel = []
    if qb_bias:
        panel.append({"label": "QB", "bias": ci(*qb_bias), "cov80": ci(0.8, 0.7, 0.9), "above": ci(.1, .05, .15)})
    if te_cov:
        panel.append({"label": "TE", "bias": ci(-1, -2, 0.5), "cov80": ci(*te_cov), "above": ci(.22, .17, .28)})
    return {"headline": head, "weeks": [1, 2, 3, 4], "failure_panel": panel}


def test_conclusion_claims_only_what_the_interval_supports():
    text = " ".join(wr.conclusion(_ps((-0.27, -0.48, -0.06), (-1.6, -2.0, -1.3), (0.79, 0.76, 0.81))))
    assert "0.27 points less" in text and "0.06 to 0.48 less" in text
    assert "ran low by 1.6" in text and "in line with the 80% target" in text
    tie = " ".join(wr.conclusion(_ps((-0.1, -0.5, 0.3), (-0.2, -0.6, 0.2), (0.70, 0.65, 0.75))))
    assert "could not be told apart" in tie and "no clear lean" in tie and "too narrow" in tie
    assert " less" not in tie.replace("less than", "")


def test_weak_spots_appear_only_when_their_interval_says_so():
    yes = " ".join(wr.conclusion(_ps((-0.3, -0.5, -0.1), (-1, -2, -0.5), (0.8, 0.77, 0.83),
                                     qb_bias=(-3.3, -4.8, -2.0), te_cov=(0.66, 0.60, 0.72))))
    assert "QB projections ran 3.3 points low" in yes and "TE ranges held only 66%" in yes
    no = " ".join(wr.conclusion(_ps((-0.3, -0.5, -0.1), (-1, -2, -0.5), (0.8, 0.77, 0.83),
                                    qb_bias=(-0.5, -1.5, 0.4), te_cov=(0.75, 0.68, 0.82))))
    assert "Weak spots" not in no


def test_one_game_makes_no_claim():
    ps = wr.player_section(_players(n_games=1), 3, {})
    assert "too few games" in " ".join(wr.conclusion(ps))


# --- rows ---------------------------------------------------------------------------------------

def test_short_names_match_the_week1_form():
    assert wr.short_name("Amon-Ra St. Brown") == "A.St. Brown"
    assert wr.short_name("Kenneth Walker III") == "K.Walker"


def test_week1_ids_recovered_only_when_unambiguous():
    con = sqlite3.connect(":memory:")
    pd.DataFrame([
        {"player_id": "a", "player_name": "Caleb Williams", "position": "QB", "team": "CHI", "season": 2026},
        {"player_id": "b", "player_name": "Jameson Williams", "position": "WR", "team": "DET", "season": 2026},
        {"player_id": "c", "player_name": "Josh Williams", "position": "WR", "team": "DET", "season": 2026},
    ]).to_sql("rosters", con)
    rows = pd.DataFrame([{"name": "C.Williams", "position": "QB", "team": "CHI", "player_id": None},
                         {"name": "J.Williams", "position": "WR", "team": "DET", "player_id": None},
                         {"name": "X.Known", "position": "RB", "team": "KC", "player_id": "z"}])
    out = wr.fill_ids(rows, con, 2026)
    assert out["player_id"].tolist()[0] == "a"
    assert pd.isna(out["player_id"].tolist()[1])          # two J.Williams on DET: left alone
    assert out["player_id"].tolist()[2] == "z"


def test_prior_snap_share_uses_only_earlier_games():
    con = sqlite3.connect(":memory:")
    pd.DataFrame([{"player_id": "a", "season": s, "week": w, "participation_state": "confirmed_played",
                   "offense_pct": v, "status": "ACT", "has_snap_row": 1, "has_stats_row": 1}
                  for s, w, v in [(2025, 17, 0.9), (2026, 1, 0.3), (2026, 2, 0.6), (2026, 3, 0.0)]]) \
        .to_sql("canonical_player_weeks", con)
    pd.DataFrame([{"player_id": "a", "season": 2026, "week": 3, "report_status": "Questionable"}]) \
        .to_sql("player_injuries", con)
    rows = pd.DataFrame({"player_id": ["a", "a", "a", None], "week": [1, 2, 3, 3]})
    out = wr.add_context(rows, con, 2026)
    assert len(out) == 4
    a = out.dropna(subset=["player_id"]).set_index("week")
    assert a.loc[1, "snap_prior3"] == pytest.approx(0.9)            # last season only
    assert a.loc[3, "snap_prior3"] == pytest.approx((0.9 + 0.3 + 0.6) / 3)
    assert a.loc[3, "snap_pct"] == 0.0 and a.loc[3, "injury_status"] == "Questionable"
    assert a.loc[3, "roster_status"] == "ACT" and a.loc[3, "has_stats_row"] == 1
    assert pd.isna(a.loc[2, "injury_status"]) and pd.isna(out.iloc[3]["snap_prior3"])


# --- importance timing audit ------------------------------------------------------------------

def test_importance_families_group_correlated_features():
    mod = _module(IMPORTANCE)
    assert mod.family("team_wr_target_share_roll3_mean") == "Usage"
    assert mod.family("opp_fpts_allowed_dvoa_adjusted_lag1") == "Opponent"
    assert mod.family("implied_team_total") == mod.family("spread") == "Betting line"
    assert mod.family("prev_season_ppg") == "Player history & profile"
    assert mod.family("yards_per_carry_roll3_mean") == "Recent production & efficiency"


def test_truncation_check_finds_a_feature_that_saw_later_weeks():
    mod = _module(IMPORTANCE)
    full = pd.DataFrame({"player_id": ["a"] * 4, "season": 2025, "week": [1, 2, 3, 4],
                         "clean": [1.0, 2.0, 3.0, 4.0], "leaky": [5.0, 5.0, 5.0, 5.0]})
    cut = full[full["week"] <= 2].assign(leaky=[3.0, 3.0])            # season mean over fewer weeks
    res = mod.truncation_check(full, cut, 2, ["clean", "leaky"])
    assert res["rows_compared"] == 2 and list(res["features_changed"]) == ["leaky"]


# --- blinding -------------------------------------------------------------------------------------

BLINDED = re.compile(r"predicted_points_model|plan_a|served_unblended|selection_forward|"
                     r"simulation_|forecasts\.csv", re.I)


@pytest.mark.parametrize("path", [MODULE, SCRIPT, IMPORTANCE])
def test_blinded_forecasts_are_never_referenced(path):
    """Candidates under docs/PRODUCTION_SELECTION_RULE.md are scored only by
    the pre-registered comparison; this report must not compute them."""
    assert not BLINDED.search(path.read_text())


def test_whitelists_drop_everything_else():
    extra = _players().assign(predicted_points_model=1.0, plan_a=2.0)
    kept = wr._keep(extra, wr.PLAYER_COLS)
    assert "predicted_points_model" not in kept and "plan_a" not in kept
    assert not any(BLINDED.search(c) for c in wr.PLAYER_COLS)


# --- the page -------------------------------------------------------------------------------------

def test_page_is_self_contained_private_and_leads_with_three_measures():
    mod = _module(SCRIPT)
    rows = _players(n_games=6)
    ts = wr.test_set(rows, pd.DataFrame({"week": [3] * 6}), [3])
    ps = wr.player_section(rows, 3, {})
    page = mod.render(2026, 3, ps, None, "", ["abc"], ts=ts, vt=wr.version_table(rows, ts),
                      checks=wr.audit(rows, ps, ts))
    tiles = re.findall(r'<div class="figures three">(.*?)</div>\s*<p class="scope">', page, re.S)[0]
    assert [t for t in ("MAE vs last-3 average", "Mean signed error", "80% range coverage") if t in tiles] == \
        ["MAE vs last-3 average", "Mean signed error", "80% range coverage"]
    assert "only projected players who recorded a stat line" in page
    assert "<svg" in page and "spread" not in page.lower()
    assert not re.search(r"https?://|<script|\ssrc=", page)
    assert "@page" in page and "@media print" in page
    assert mod.minus_nums("bias -1.6 (-2.1 to -1.2), 3-1 record") == \
        "bias −1.6 (−2.1 to −1.2), 3-1 record"
    docs = (PROJECT_ROOT / "docs").resolve()
    out = mod.REPORT_DIR.resolve()
    assert docs not in out.parents and out != docs
    assert ESPN_PRIVATE_DIR.resolve() in out.parents


def test_page_prose_comes_from_the_data():
    """The version note names what was actually served, and the byline only
    claims pre-kickoff freezing when that check passed."""
    mod = _module(SCRIPT)
    rows = pd.concat([_players(n_games=2, week=3, mode="season_prorated"),
                      _players(n_games=2, week=4)], ignore_index=True)
    rows.loc[rows["week"] == 3, ["prediction_ci80_lower", "prediction_ci80_upper"]] = np.nan
    ts = wr.test_set(rows, pd.DataFrame({"week": [3, 3, 4, 4]}), [3, 4])
    ps = wr.player_section(rows, 4, {}, weeks=[3, 4])
    checks = wr.audit(rows, ps, ts)
    page = mod.render(2026, 4, ps, None, "", ["abc"], True, ts=ts, vt=wr.version_table(rows, ts), checks=checks)
    assert "season model ÷ games in week 3 (2 games, no ranges published)" in page
    assert "weekly + pace blend in week 4 (2 games)" in page
    assert "every projection committed to git before its game's kickoff" in page
    late = [dict(c, ok=False) if c["check"].startswith("Every projection was committed") else c for c in checks]
    page = mod.render(2026, 4, ps, None, "", ["abc"], True, ts=ts, vt=wr.version_table(rows, ts), checks=late)
    assert "every projection committed" not in page and "pre-kickoff check failed" in page


def test_numbers_that_round_to_zero_carry_no_sign():
    mod = _module(SCRIPT)
    assert mod.f(-0.04, 1) == "0.0" and mod.f(-0.4, 1) == "-0.4" and mod.f(-10.0, 1) == "-10.0"
    assert mod.pct(-0.001) == "0%" and mod.pct(0.788) == "79%"


def test_a_player_moved_mid_week_counts_once_on_his_roster_team():
    con = sqlite3.connect(":memory:")
    pd.DataFrame([{"player_id": "x", "season": 2026, "week": 5, "team": "LV"}]).to_sql("weekly_rosters", con)
    rows = pd.DataFrame([
        {"player_id": "x", "week": 5, "team": "TB", "published_at": "2026-10-08T03:28+00:00"},
        {"player_id": "x", "week": 5, "team": "LV", "published_at": "2026-10-10T17:11+00:00"},
        {"player_id": "y", "week": 5, "team": "KC", "published_at": "2026-10-08T03:28+00:00"},
        {"player_id": "y", "week": 5, "team": "BUF", "published_at": "2026-10-10T17:11+00:00"},
        {"player_id": None, "week": 5, "team": "KC", "published_at": "2026-10-10T17:11+00:00"},
        {"player_id": None, "week": 5, "team": "KC", "published_at": "2026-10-10T17:11+00:00"}])
    out = wr.one_row_per_player_week(rows, con, 2026)
    assert out[out.player_id == "x"]["team"].tolist() == ["LV"]       # matches the roster
    assert out[out.player_id == "y"]["team"].tolist() == ["BUF"]      # no roster row: later snapshot
    assert out["player_id"].isna().sum() == 2 and out.attrs["moved_players_dropped"] == 2
