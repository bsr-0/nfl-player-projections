"""This week's predictions against what actually happened, as one HTML page.

Two sections, both limited to games that have been played:

  1. Fantasy -- your lineup and bench, and your opponent's, with the model's
     projection, ESPN's projection and the actual score side by side.
  2. Games   -- the model's pick, margin and total for each NFL game against
     the final score.

Actuals come live from ESPN (league box scores and the public scoreboard), so
the page is a snapshot of right now; re-run it after the Monday game. Players
whose game has not finished are shown as pending and left out of every error
figure. Model projections for the fantasy section come from the week's report
JSON (scripts/league_report.py) -- the opponent's bench has no model number.

Output stays under data/espn_private/reports/ (league data, never docs/).

Usage:
    python scripts/render_week_results_report.py [--team 1] [--week 5]
"""
import argparse
import html
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT

REPORT_DIR = ESPN_PRIVATE_DIR / "reports"
GAMES_FILE = PROJECT_ROOT / "docs" / "data" / "game_predictions_{season}_wk{week}.json"
SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
              "scoreboard?week={week}&seasontype=2&dates={season}")
# The games file uses nflverse abbreviations; ESPN uses its own.
ESPN_TO_MODEL = {"LAR": "LA", "WSH": "WAS"}
OFF_BENCH = {"BE", "IR"}
SLOT_RANK = {s: i for i, s in enumerate(
    ["QB", "RB", "WR", "TE", "RB/WR/TE", "D/ST", "K", "BE", "IR"])}


def esc(v):
    return html.escape("" if v is None else str(v))


def num(v, dash="&ndash;", sign=False):
    if v is None:
        return dash
    return f"{float(v):+.1f}" if sign else f"{float(v):.1f}"


# --------------------------------------------------------------- fantasy

def load_league_week(week, team_id):
    from espn_api.football import League
    league = League(league_id=int(os.environ["ESPN_LEAGUE_ID"]),
                    year=int(os.environ.get("ESPN_YEAR", 2026)))
    for box in league.box_scores(week):
        if team_id in (box.home_team.team_id, box.away_team.team_id):
            mine_home = box.home_team.team_id == team_id
            side = lambda home: dict(
                name=(box.home_team if home else box.away_team).team_name,
                score=box.home_score if home else box.away_score,
                lineup=box.home_lineup if home else box.away_lineup)
            return side(mine_home), side(not mine_home)
    raise SystemExit(f"team {team_id} has no matchup in week {week}")


def model_lookup(report):
    out = {}
    for p in report["starters"] + report["bench"] + report["opponent_starters"]:
        if p.get("points_source") == "model" and p.get("points") is not None:
            out[p["name"]] = p["points"]
    return out


def team_rows(side, model):
    rows = []
    for p in side["lineup"]:
        played = (p.game_played or 0) >= 100
        bye = bool(p.on_bye_week)
        rows.append(dict(
            slot=p.slot_position, name=p.name, pos=p.position,
            team=p.proTeam, played=played, bye=bye,
            actual=p.points if played else None,
            espn=None if bye else p.projected_points,
            model=None if bye else model.get(p.name)))
    rows.sort(key=lambda r: (SLOT_RANK.get(r["slot"], 9), -(r["espn"] or 0)))
    return rows


def mae(rows, key):
    errs = [abs(r[key] - r["actual"]) for r in rows
            if r["actual"] is not None and r[key] is not None]
    return (sum(errs) / len(errs), len(errs)) if errs else (None, 0)


def status_cell(r):
    if r["bye"]:
        return '<span class="tag mute">BYE</span>'
    if not r["played"]:
        return '<span class="tag pend">pending</span>'
    return ""


def err_cell(proj, r):
    if r["actual"] is None or proj is None:
        return '<td class="n mute">&ndash;</td>'
    e = r["actual"] - proj
    cls = "up" if e > 0.05 else "down" if e < -0.05 else "mute"
    return f'<td class="n {cls}">{e:+.1f}</td>'


def team_table(rows):
    body, sep = [], False
    for r in rows:
        bench = r["slot"] in OFF_BENCH
        if bench and not sep:
            body.append('<tr class="sep"><td colspan="8">Bench &amp; IR</td></tr>')
            sep = True
        body.append(
            f'<tr class="{"bench" if bench else ""}">'
            f'<td class="slot">{esc(r["slot"])}</td>'
            f'<td>{esc(r["name"])} <span class="mute">{esc(r["pos"])} '
            f'{esc(r["team"])}</span> {status_cell(r)}</td>'
            f'<td class="n">{num(r["model"])}</td>'
            f'<td class="n">{num(r["espn"])}</td>'
            f'<td class="n act">{num(r["actual"])}</td>'
            f'{err_cell(r["model"], r)}{err_cell(r["espn"], r)}</tr>')
    return ('<table><thead><tr><th>Slot</th><th>Player</th>'
            '<th class="n">Model</th><th class="n">ESPN</th>'
            '<th class="n">Actual</th><th class="n">Model err</th>'
            '<th class="n">ESPN err</th></tr></thead><tbody>'
            + "".join(body) + '</tbody></table>')


def starters_total(rows, key):
    done = [r for r in rows if r["slot"] not in OFF_BENCH and r["played"]]
    return sum((r[key] or 0) for r in done)


def fantasy_section(week, mine, opp, model):
    mrows = team_rows(mine, model)
    cards = []
    for label, rows in (("Model", "model"), ("ESPN", "espn")):
        m, n = mae(mrows, rows)
        cards.append(
            f'<div class="stat"><span>{label} MAE</span>'
            f'<b>{num(m)}</b><small>{n} played players</small></div>')
    # What the model's lineup would have scored, among players who have played.
    model_lineup = sum(r["actual"] or 0 for r in mrows
                       if r["actual"] is not None and r["model"] is not None
                       and r["slot"] not in OFF_BENCH)
    cards.insert(0,
        f'<div class="stat"><span>{esc(mine["name"])}</span>'
        f'<b>{num(mine["score"])}</b><small>live, ESPN proj '
        f'{num(starters_total(mrows, "espn"))} for played slots</small></div>'
        f'')
    return f"""
<section>
  <h2>Fantasy <small>week {week} &mdash; projections vs actual</small></h2>
  <div class="stats">{"".join(cards)}</div>
  <div class="cols one">
    <div class="card"><h3>{esc(mine["name"])} <small>you</small></h3>
      {team_table(mrows)}</div>
  </div>
  <p class="note">Error = actual &minus; projection, so positive means the
  player beat the projection. Pending players are excluded from MAE. Model
  numbers are missing for kickers and defences (not modelled). Actuals are ESPN's scoring for this league.</p>
</section>"""


# ----------------------------------------------------------------- games

def fetch_finals(season, week):
    with urllib.request.urlopen(SCOREBOARD.format(season=season, week=week),
                                timeout=20) as r:
        data = json.load(r)
    out = {}
    for e in data["events"]:
        c = e["competitions"][0]
        team = {t["homeAway"]: t for t in c["competitors"]}
        ab = lambda s: ESPN_TO_MODEL.get(team[s]["team"]["abbreviation"],
                                         team[s]["team"]["abbreviation"])
        out[(ab("home"), ab("away"))] = dict(
            state=e["status"]["type"]["name"],
            final=e["status"]["type"]["completed"],
            home=float(team["home"]["score"]),
            away=float(team["away"]["score"]),
            detail=e["status"]["type"]["shortDetail"],
            kickoff=e["date"])
    return out


def avg(g, keys):
    vals = [g[k] for k in keys if g.get(k) is not None]
    return sum(vals) / len(vals) if vals else None


def games_section(season, week):
    games = json.loads(Path(str(GAMES_FILE).format(
        season=season, week=week)).read_text())
    live = fetch_finals(season, week)
    rows, done, hits, margin_err, total_err, line_hits = [], 0, 0, [], [], 0
    order = lambda g: g.get("kickoff") or live.get(
        (g["home_team"], g["away_team"]), {}).get("kickoff", "")
    for g in sorted(games, key=order):
        h, a = g["home_team"], g["away_team"]
        prob = avg(g, ["home_win_prob_logistic", "home_win_prob_xgb",
                       "home_win_prob_rf"])
        margin = avg(g, ["predicted_margin_ridge", "predicted_margin_xgb",
                         "predicted_margin_rf"])
        total = avg(g, ["predicted_total_ridge", "predicted_total_xgb",
                        "predicted_total_rf"])
        pick = h if prob >= 0.5 else a
        conf = prob if prob >= 0.5 else 1 - prob
        res = live.get((h, a), {})
        final = res.get("final")
        if final:
            hs, as_ = res["home"], res["away"]
            winner = h if hs > as_ else a if as_ > hs else "TIE"
            ok = pick == winner
            done += 1
            hits += ok
            me = (hs - as_) - margin
            te = (hs + as_) - total
            margin_err.append(abs(me))
            total_err.append(abs(te))
            # Market comparison: did the model's side of the spread cover?
            line_home = g["spread_line"]
            if line_home is not None and margin != line_home:
                line_hits += ((hs - as_) > line_home) == (margin > line_home)
            result = (f'<td class="n act">{as_:.0f}&ndash;{hs:.0f}</td>'
                      f'<td>{esc(winner)} '
                      f'<span class="tag {"ok" if ok else "bad"}">'
                      f'{"hit" if ok else "miss"}</span></td>'
                      f'<td class="n {"up" if me > 0 else "down"}">{me:+.1f}</td>'
                      f'<td class="n {"up" if te > 0 else "down"}">{te:+.1f}</td>')
        else:
            state = res.get("detail") or "scheduled"
            live_score = (f' {res["away"]:.0f}&ndash;{res["home"]:.0f}'
                          if res.get("state") == "STATUS_IN_PROGRESS" else "")
            result = (f'<td class="n mute">{esc(state)}{live_score}</td>'
                      '<td><span class="tag pend">pending</span></td>'
                      '<td class="n mute">&ndash;</td><td class="n mute">&ndash;</td>')
        rows.append(
            f'<tr><td>{esc(a)} @ {esc(h)}</td>'
            f'<td><span class="tag {"ok" if g.get("source") == "published" else "mute"}">'
            f'{"pre-kick" if g.get("source") == "published" else "live"}</span></td>'
            f'<td>{esc(pick)} <span class="mute">{conf:.0%}</span></td>'
            f'<td class="n">{h} {margin:+.1f}</td>'
            f'<td class="n mute">{num(g.get("spread_line"), sign=True)}</td>'
            f'<td class="n">{total:.1f}</td>{result}</tr>')
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    cards = (
        f'<div class="stat"><span>Winner picks</span><b>{hits}/{done}</b>'
        f'<small>{hits / done:.0%} on finished games</small></div>' if done else "")
    cards += (
        f'<div class="stat"><span>Margin MAE</span><b>{num(mean(margin_err))}</b>'
        f'<small>points</small></div>'
        f'<div class="stat"><span>Total MAE</span><b>{num(mean(total_err))}</b>'
        f'<small>points</small></div>'
        f'<div class="stat"><span>Finished</span><b>{done}/{len(games)}</b>'
        f'<small>games this week</small></div>')
    return f"""
<section>
  <h2>Games <small>week {week} &mdash; model picks vs final scores</small></h2>
  <div class="stats">{cards}</div>
  <div class="card">
  <table><thead><tr><th>Game</th><th>Forecast</th><th>Model pick</th>
    <th class="n">Model margin</th><th class="n">Line</th>
    <th class="n">Model total</th><th class="n">Final (A&ndash;H)</th>
    <th>Winner</th><th class="n">Margin err</th><th class="n">Total err</th>
  </tr></thead><tbody>{"".join(rows)}</tbody></table></div>
  <p class="note"><b>Forecast column:</b> <i>pre-kick</i> rows were published
  before kickoff and are a fair test. <i>live</i> rows were regenerated by the
  site after the schedule moved on, so for a finished game they may have seen
  more than a true pre-game forecast would &mdash; read their hit rate as
  optimistic. Pick, margin and total average the three published models
  (logistic/ridge, XGBoost, random forest). Margin is home minus away; the
  line column is the sportsbook home spread (positive = home favoured). Error
  = actual &minus; model, so positive means the game finished higher (more
  home margin / more points) than the model said.</p>
</section>"""


# ------------------------------------------------------------------ page

CSS = """
:root{--bg:#0e1116;--card:#161b22;--line:#252c36;--ink:#e6edf3;--mute:#7d8896;
--accent:#6ea8fe;--up:#3fb68b;--down:#f0716b;--pend:#d8a93c}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f9;--card:#fff;--line:#e3e7ee;
--ink:#16202b;--mute:#6b7787;--accent:#2563c9;--up:#13875e;--down:#c8423c;--pend:#9a6b00}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:28px 24px 48px}
main{max-width:1200px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:baseline;
border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:6px}
h1{font-size:22px;margin:0;letter-spacing:-.01em}
h1 span{color:var(--accent)}
header p{margin:0;color:var(--mute);font-size:12px}
h2{font-size:16px;margin:30px 0 12px}
h2 small,h3 small{color:var(--mute);font-weight:400;font-size:12px;margin-left:6px}
h3{font-size:13px;margin:0 0 8px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;min-width:150px;flex:1}
.stat span{display:block;color:var(--mute);font-size:11px;text-transform:uppercase;
letter-spacing:.06em}
.stat b{display:block;font-size:24px;font-weight:650;margin:2px 0;
font-variant-numeric:tabular-nums}
.stat small{color:var(--mute);font-size:11px}
.cols.one{grid-template-columns:1fr}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 14px;overflow-x:auto}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:10.5px;color:var(--mute);text-transform:uppercase;
letter-spacing:.05em;font-weight:600;padding:4px 6px;border-bottom:1px solid var(--line)}
td{padding:4px 6px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
.n{text-align:right;font-variant-numeric:tabular-nums}
.slot{color:var(--mute);font-size:11px;width:64px}
.mute{color:var(--mute)}
.act{font-weight:650}
.up{color:var(--up)}.down{color:var(--down)}
tr.bench td{color:var(--mute)}
tr.bench .act{color:var(--ink)}
tr.sep td{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
color:var(--mute);background:transparent;padding-top:10px}
.tag{font-size:10px;font-weight:650;border-radius:4px;padding:0 5px;
border:1px solid currentColor;letter-spacing:.03em}
.tag.pend{color:var(--pend)}.tag.mute{color:var(--mute)}
.tag.ok{color:var(--up)}.tag.bad{color:var(--down)}
.note{color:var(--mute);font-size:11.5px;max-width:880px;margin:10px 0 0}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=int(os.environ.get("ESPN_TEAM_ID", 1)))
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    latest = json.loads((REPORT_DIR / "latest.json").read_text())
    week = args.week or latest["week"]
    season = latest["season"]
    report_file = REPORT_DIR / f"report_{season}_wk{week}.json"
    model = model_lookup(json.loads(report_file.read_text()))

    mine, opp = load_league_week(week, args.team)
    from datetime import datetime
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week {week} results</title><style>{CSS}</style></head>
<body><main>
<header><h1>Week {week} <span>results</span></h1>
<p>Snapshot {datetime.now():%a %b %d, %H:%M} &middot; completed games only</p></header>
{fantasy_section(week, mine, opp, model)}
{games_section(season, week)}
</main></body></html>"""
    out = REPORT_DIR / f"results_{season}_wk{week}.html"
    out.write_text(page)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
