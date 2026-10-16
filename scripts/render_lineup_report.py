"""One-page lineup report: your lineup and bench.

A single table per team -- the lineup you have set in ESPN on top, the bench
and IR beneath it -- with this project's projection for each player. Your side
adds the model's call (start / sit) so the disagreement with ESPN is visible
in place. Nothing else: no waivers, trades or tabs.

Built straight from the latest ESPN snapshot and the week's published
projections (the same join league_report.py uses). Writes report_<season>_wk<week>.html and latest.html under
data/espn_private/reports/ (replacing the old layout there) -- league data,
never docs/.

Usage:
    python scripts/render_lineup_report.py [--team 1] [--week 5]
"""
import argparse
import html
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.integrations.league_insights import (
    _players, find_team, lineup_total, optimal_lineup, starting_slots,
    win_probability,
)
from src.integrations.league_join import (
    join_players, load_projections, load_snapshot,
)

REPORT_DIR = ESPN_PRIVATE_DIR / "reports"
OFF = {"BE", "IR"}
SLOT_RANK = {s: i for i, s in enumerate(
    ["QB", "RB", "WR", "TE", "RB/WR/TE", "D/ST", "K", "BE", "IR"])}
STATUS = {"OUT": "OUT", "INJURY_RESERVE": "IR", "SUSPENSION": "SUS",
          "DOUBTFUL": "D", "QUESTIONABLE": "Q", "DAY_TO_DAY": "DTD"}


def esc(v):
    return html.escape("" if v is None else str(v))


def num(v, dash="&ndash;"):
    return dash if v is None else f"{float(v):.1f}"


def live(p):
    """What a player contributes this week: nothing on a bye or ruled out."""
    return float(p["points"]) if p.get("available") and p.get("points") \
        is not None else 0.0


def ordered(players):
    return sorted(players, key=lambda p: (
        SLOT_RANK.get(p.get("lineup_slot"), 9), -(p.get("points") or -1)))


def name_cell(p):
    tags = ""
    if p.get("unavailable_reason") == "BYE":
        tags += ' <span class="tag mute">BYE</span>'
    elif STATUS.get(p.get("injury_status")):
        sev = "bad" if p["injury_status"] in (
            "OUT", "INJURY_RESERVE", "SUSPENSION", "DOUBTFUL") else "warn"
        tags += f' <span class="tag {sev}">{STATUS[p["injury_status"]]}</span>'
    if p.get("points_source") == "espn":
        tags += ' <span class="tag mute" title="ESPN number, not modelled">ESPN</span>'
    elif p.get("points") is None:
        tags += ' <span class="tag mute">no proj</span>'
    return (f'{esc(p["name"])} <span class="mute">{esc(p["position"])} '
            f'{esc(p.get("nfl_team"))}</span>{tags}')


def range_cell(p):
    lo, hi = (p.get("week_ci") or [None, None])[:2]
    if lo is None or hi is None:
        return '<td class="n mute">&ndash;</td>'
    return f'<td class="n dim">{lo:.0f}&ndash;{hi:.0f}</td>'


def table(players, calls=None):
    """calls: name -> 'START' | 'SIT' for the model's disagreements."""
    rows, sep = [], False
    for p in ordered(players):
        bench = p.get("lineup_slot") in OFF
        if bench and not sep:
            cols = 7 if calls is not None else 6
            rows.append(f'<tr class="sep"><td colspan="{cols}">Bench &amp; IR</td></tr>')
            sep = True
        opp = p.get("opponent")
        call = ""
        if calls is not None:
            c = calls.get(p["name"])
            call = ('<td></td>' if not c else
                    f'<td><span class="call {c.lower()}">{c}</span></td>')
        dead = "" if p.get("available") else " out"
        rows.append(
            f'<tr class="{"bench" if bench else ""}{dead}">'
            f'<td class="slot">{esc(p.get("lineup_slot"))}</td>'
            f'<td>{name_cell(p)}</td>'
            f'<td class="dim">{("vs " + esc(opp)) if opp else ""}</td>'
            f'<td class="n pt">{num(live(p) if p.get("points") is not None else None)}</td>'
            f'{range_cell(p)}{call}</tr>')
    head = ('<th>Slot</th><th>Player</th><th>Opp</th>'
            '<th class="n">Proj</th><th class="n">80% range</th>'
            + ('<th>Model</th>' if calls is not None else ''))
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def set_total(players):
    return sum(live(p) for p in players if p.get("lineup_slot") not in OFF)


CSS = """
:root{--bg:#0e1116;--card:#161b22;--line:#252c36;--ink:#e6edf3;--mute:#7d8896;
--dim:#a8b2bf;--accent:#6ea8fe;--up:#3fb68b;--down:#f0716b;--warn:#d8a93c}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f9;--card:#fff;--line:#e3e7ee;
--ink:#16202b;--mute:#6b7787;--dim:#4a5666;--accent:#2563c9;--up:#13875e;
--down:#c8423c;--warn:#9a6b00}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:28px 24px 44px}
main{max-width:1200px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:12px}
h1{font-size:22px;margin:0;letter-spacing:-.01em}
h1 span{color:var(--accent)}
header p{margin:0;color:var(--mute);font-size:12px}
.stats{display:flex;gap:10px;margin:16px 0}
.stat{flex:1;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px}
.stat span{display:block;color:var(--mute);font-size:11px;text-transform:uppercase;
letter-spacing:.06em}
.stat b{display:block;font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.stat small{color:var(--mute);font-size:11px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
@media(max-width:960px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 14px;overflow-x:auto}
h2{font-size:14px;margin:0 0 8px;display:flex;justify-content:space-between}
h2 small{color:var(--mute);font-weight:400;font-size:12px}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:10.5px;color:var(--mute);text-transform:uppercase;
letter-spacing:.05em;font-weight:600;padding:4px 6px;border-bottom:1px solid var(--line)}
td{padding:4px 6px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
.n{text-align:right;font-variant-numeric:tabular-nums}
.pt{font-weight:650}
.slot{color:var(--mute);font-size:11px;width:62px}
.mute{color:var(--mute)}.dim{color:var(--dim)}
tr.bench td{color:var(--mute)}
tr.bench .pt{color:var(--ink)}
tr.out td:nth-child(2),tr.out .pt{opacity:.55}
tr.sep td{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
color:var(--mute);padding-top:12px}
tr.total td{border-top:1px solid var(--line);font-weight:650}
.tag{font-size:10px;font-weight:650;border-radius:4px;padding:0 5px;
border:1px solid currentColor;letter-spacing:.03em}
.tag.mute{color:var(--mute)}.tag.warn{color:var(--warn)}.tag.bad{color:var(--down)}
.call{font-size:10.5px;font-weight:700;letter-spacing:.05em;padding:1px 7px;
border-radius:999px}
.call.start{background:rgba(63,182,139,.16);color:var(--up)}
.call.sit{background:rgba(240,113,107,.16);color:var(--down)}
.note{color:var(--mute);font-size:11.5px;margin:12px 0 0;max-width:900px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=os.environ.get("ESPN_TEAM_ID", "1"))
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    snap = load_snapshot()
    week = snap.week if args.week is None else args.week
    proj = load_projections(snap.season, week)
    mine = find_team(snap, args.team)
    slots = starting_slots(snap.settings)

    league = _players(join_players(snap.rostered(), proj).to_dict("records"))
    roster = [p for p in league if p["fantasy_team"] == mine["team_name"]]
    match = snap.opponent_for(mine["team_id"], week)
    opp_name = match.get("opponent_name")
    opp = [p for p in league if p["fantasy_team"] == opp_name]

    best, _ = optimal_lineup(roster, slots)
    best_idx = {p["index"] for p in best}
    calls = {}
    for p in roster:
        set_ = p.get("lineup_slot") not in OFF
        if p["index"] in best_idx and not set_:
            calls[p["name"]] = "START"
        elif set_ and p["index"] not in best_idx:
            calls[p["name"]] = "SIT"
    opp_best, _ = optimal_lineup(opp, slots)

    your_set, model_total = set_total(roster), lineup_total(best)
    win = win_probability(best, opp_best)
    moves = len(calls)
    gain = model_total - your_set

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(mine["team_name"])} week {week}</title><style>{CSS}</style></head>
<body><main>
<header><h1>{esc(mine["team_name"])} <span>vs {esc(opp_name)}</span></h1>
<p>Week {week} &middot; {esc(snap.season)} &middot; projections {esc(proj["projection_mode"].dropna().iloc[0] if proj["projection_mode"].notna().any() else "n/a")}
&middot; snapshot {esc(snap.manifest.get("pulled_at", "")[:16].replace("T", " "))}</p></header>
<div class="stats">
  <div class="stat"><span>Your lineup, as set</span><b>{num(your_set)}</b>
    <small>model lineup {num(model_total)} ({gain:+.1f})</small></div>
  <div class="stat"><span>Win probability</span>
    <b>{"&ndash;" if win is None else f"{win:.0%}"}</b>
    <small>model lineup vs their best</small></div>
  <div class="stat"><span>Model changes</span><b>{moves}</b>
    <small>{"lineup already matches" if not moves else "start/sit calls below"}</small></div>
</div>
<div class="card"><h2>{esc(mine["team_name"])} <small>you &middot; {mine.get("wins")}-{mine.get("losses")}</small></h2>
    {table(roster, calls)}</div>
<p class="note">Tables show the lineup as set in ESPN, then bench and IR. Proj is
this project's weekly projection; <b>ESPN</b>-tagged rows are ESPN's own number
(kickers and defences are not modelled). Players on a bye or ruled out count as
0 in totals. The Model column marks where the model would change your lineup:
START = bench player it would play, SIT = starter it would bench.</p>
</main></body></html>"""
    # Same names the old report used, so this replaces it in place.
    out = REPORT_DIR / f"report_{snap.season}_wk{week}.html"
    out.write_text(page)
    (REPORT_DIR / "latest.html").write_text(page)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({len(page):,} bytes)")
    print(f"      {(REPORT_DIR / 'latest.html').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
