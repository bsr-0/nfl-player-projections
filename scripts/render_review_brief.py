"""The projections brief: two pages for a non-technical reader.

Page 1, how the projections did over the complete weeks of the season; page
2, our lineup for the next week with the model's ranges and a short guide to
reading them. Same rows and intervals as the detailed review
(scripts/render_weekly_review.py), fewer words: a handful of numbers and
charts, plain labels.

Output: data/espn_private/reports/review_{season}_brief.html (private).

Usage:
    python scripts/render_review_brief.py               # next week = first incomplete week
    python scripts/render_review_brief.py --next-week 6
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DB_PATH, ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.evaluation import review_brief as rb
from src.evaluation import weekly_review as wr
from src.evaluation.published_record import kickoff_times
from src.evaluation.review_charts import (CSS, Scale, axis_titles, esc, f, fig_calibration, fig_team, legend, minus,
                                          pct, svg, ticks, yaxis)
from src.evaluation.team_history import fantasy_season, week_lineup

REPORT_DIR = ESPN_PRIVATE_DIR / "reports"
WEEKLY_DIR = PROJECT_ROOT / "docs" / "data"
SLOT_ORDER = ["QB", "RB", "WR", "TE", "RB/WR/TE", "OP", "D/ST", "K", "BE", "IR"]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def fig_weekly_miss(ps):
    """Typical miss per week: ours and the last-3 average on the same players,
    each with a light 95% band."""
    rows = [w for w in ps["by_week"] if w["week"] in ps["weeks"] and w["model_mae_paired"]["est"] is not None]
    if not rows:
        return ""
    w, h, l, r, t, b = 600, 290, 58, 24, 40, 56
    vals = [v for x in rows for k in ("model_mae_paired", "base_mae") for v in (x[k]["lo"], x[k]["hi"], x[k]["est"])
            if v is not None]
    lo, hi = max(0, min(vals) - 0.3), max(vals) + 0.3
    sx = Scale(rows[0]["week"] - 0.3, rows[-1]["week"] + 0.3, l, w - r)
    sy = Scale(lo, hi, h - b, t)
    body = [yaxis(sy, l, w - r, ticks(lo, hi, 4), lambda v: f"{v:g}")]
    for key, cls in (("base_mae", "base"), ("model_mae_paired", "model")):
        band = [x for x in rows if x[key]["lo"] is not None]
        if len(band) > 1:
            pts = [f"{sx(x['week']):.1f},{sy(x[key]['hi']):.1f}" for x in band] + \
                  [f"{sx(x['week']):.1f},{sy(x[key]['lo']):.1f}" for x in reversed(band)]
            body.append(f'<polygon class="{cls}-band" points="{" ".join(pts)}"/>')
        body.append(f'<polyline class="{cls}-stroke" fill="none" stroke-width="2.5" points="'
                    + " ".join(f"{sx(x['week']):.1f},{sy(x[key]['est']):.1f}" for x in rows) + '"/>')
        for x in rows:
            body.append(f'<circle class="{cls}-fill ring" cx="{sx(x["week"]):.1f}" cy="{sy(x[key]["est"]):.1f}" r="5">'
                        f'<title>Week {x["week"]}: {x[key]["est"]:.2f} pts</title></circle>')
    last = rows[-1]
    for key, dy in (("model_mae_paired", 18), ("base_mae", -10)):
        body.append(f'<text class="dlabel" x="{sx(last["week"]) - 4:.1f}" y="{sy(last[key]["est"]) + dy:.1f}" '
                    f'text-anchor="end">{last[key]["est"]:.1f}</text>')
    for x in rows:
        body.append(f'<text class="tick" x="{sx(x["week"]):.1f}" y="{h - b + 20}" text-anchor="middle">Wk {x["week"]}</text>')
    body.append(legend([("model", "Model", "dot"), ("base", "Last-3-games average", "dot")], l, 16))
    body.append(axis_titles(w, h, l, r, t, b, None, "Typical miss (points)"))
    return svg(w, h, "".join(body), "Typical miss by week, ours and the last-3-games average")


def fig_closeness(dist):
    """Share of players the model got within 3, 5 and 10 points, against the last-3 average."""
    if not dist:
        return ""
    w, h, l, r = 600, 250, 118, 60
    rows = [(k, dist[f"model_within{k}"], dist[f"rolling3_within{k}"]) for k in (3, 5, 10)]
    sx = Scale(0, 1, l, w - r)
    body = []
    for v in (0, .25, .5, .75, 1):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="38" y2="{h - 26}"/>'
                    f'<text class="tick" x="{sx(v):.1f}" y="{h - 8}" text-anchor="middle">{v:.0%}</text>')
    for i, (k, m, base) in enumerate(rows):
        y0 = 48 + i * 64
        body.append(f'<text class="panel" x="{l - 12}" y="{y0 + 24}" text-anchor="end" style="font-size:16px">'
                    f'Within {k} pts</text>')
        for j, (v, cls) in enumerate(((m, "model"), (base, "base"))):
            y = y0 + j * 22
            body.append(f'<rect class="{cls}-fill" x="{l}" y="{y}" width="{sx(v) - l:.1f}" height="18" rx="3">'
                        f'<title>{"Ours" if cls == "model" else "Last-3 average"}: {v:.0%}</title></rect>'
                        f'<text class="dlabel" x="{sx(v) + 6:.1f}" y="{y + 14}">{v:.0%}</text>')
    body.append(legend([("model", "Model", "block"), ("base", "Last-3-games average", "block")], l, 16))
    return svg(w, h, "".join(body), "Share of players projected within 3, 5 and 10 points")


def fig_positions(ps):
    """Two small panels by position: which way the projections lean, and how
    often the score landed inside the model's range (target 80%)."""
    pos = ps["by_position"]
    if not pos:
        return ""
    w, h, top, rh = 600, 64 + 46 * len(pos), 58, 46
    lab = 52
    left = (lab, 300)
    right = (348, 566)
    lean = [v for x in pos for v in (x["bias"]["lo"], x["bias"]["hi"], x["bias"]["est"]) if v is not None] + [0, 1]
    sl = Scale(min(lean) - 0.4, max(lean) + 0.4, *left)
    sr = Scale(0.4, 1.0, *right)
    body = [f'<text class="panel" x="{(left[0] + left[1]) / 2:.0f}" y="18" text-anchor="middle" style="font-size:15px">'
            'Lean (points)</text>'
            f'<text class="axislabel" x="{(left[0] + left[1]) / 2:.0f}" y="36" text-anchor="middle">'
            '← projected too low · too high →</text>'
            f'<text class="panel" x="{(right[0] + right[1]) / 2:.0f}" y="18" text-anchor="middle" style="font-size:15px">'
            'Landed in the model\'s range</text>'
            f'<text class="axislabel" x="{(right[0] + right[1]) / 2:.0f}" y="36" text-anchor="middle">'
            'built for 80%</text>']
    base_y = top + rh * len(pos)
    for v in ticks(sl.d0, sl.d1, 4):
        body.append(f'<line class="grid" x1="{sl(v):.1f}" x2="{sl(v):.1f}" y1="{top - 6}" y2="{base_y}"/>'
                    f'<text class="tick" x="{sl(v):.1f}" y="{base_y + 18}" text-anchor="middle">{minus(f"{v:g}")}</text>')
    for v in (0.4, 0.6, 0.8, 1.0):
        body.append(f'<line class="grid" x1="{sr(v):.1f}" x2="{sr(v):.1f}" y1="{top - 6}" y2="{base_y}"/>'
                    f'<text class="tick" x="{sr(v):.1f}" y="{base_y + 18}" text-anchor="middle">{v:.0%}</text>')
    body.append(f'<line class="target" x1="{sl(0):.1f}" x2="{sl(0):.1f}" y1="{top - 6}" y2="{base_y}"/>'
                f'<line class="target" stroke-dasharray="4 3" x1="{sr(0.8):.1f}" x2="{sr(0.8):.1f}" y1="{top - 6}" '
                f'y2="{base_y}"/>')
    for i, x in enumerate(pos):
        y = top + rh * i + rh / 2
        b, c = x["bias"], x["cov80"]
        body.append(f'<text class="panel" x="0" y="{y + 5:.1f}" style="font-size:15px">{esc(x["label"])}</text>')
        x0, x1 = sorted((sl(0), sl(b["est"])))
        body.append(f'<rect class="model-fill" x="{x0:.1f}" y="{y - 9:.1f}" width="{x1 - x0:.1f}" height="18" rx="3">'
                    f'<title>{x["label"]}: {b["est"]:+.1f} pts (95% {f(b["lo"], 1)} to {f(b["hi"], 1)}), '
                    f'n={x["n"]}</title></rect>')
        if b["lo"] is not None:
            body.append(f'<line class="ink-stroke" x1="{sl(b["lo"]):.1f}" x2="{sl(b["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        lean_txt = minus(f"{b['est']:+.1f}")
        # The label sits just beyond the far end of the whisker, away from zero.
        far = (b["lo"] if b["est"] < 0 else b["hi"]) if b["lo"] is not None else b["est"]
        tx = (sl(far) - 6, "end") if b["est"] < 0 else (sl(far) + 6, "start")
        body.append(f'<text class="dlabel" x="{tx[0]:.1f}" y="{y + 5:.1f}" text-anchor="{tx[1]}">'
                    f'{lean_txt}</text>')
        if c["est"] is not None:
            body.append(f'<rect class="base-fill" x="{sr(0.4):.1f}" y="{y - 9:.1f}" width="{sr(c["est"]) - sr(0.4):.1f}" '
                        f'height="18" rx="3"><title>{x["label"]}: {c["est"]:.0%} in range (95% {pct(c["lo"])} to '
                        f'{pct(c["hi"])}), n={x["n_range"]}</title></rect>')
            if c["lo"] is not None:
                body.append(f'<line class="ink-stroke" x1="{sr(max(c["lo"], 0.4)):.1f}" x2="{sr(c["hi"]):.1f}" '
                            f'y1="{y:.1f}" y2="{y:.1f}"/>')
            body.append(f'<text class="dlabel" x="{sr(0.4) + 6:.1f}" y="{y + 5:.1f}" style="fill:var(--paper)">'
                        f'{c["est"]:.0%}</text>')
    return svg(w, h + 24, "".join(body), "Lean and range hit rate by position")


def fig_head_to_head(h2h):
    """How often the higher projection won a same-week, same-position
    head-to-head, by how far apart the two projections were."""
    bins = [x for x in h2h if x["share"] is not None]
    if not bins:
        return ""
    w, h, l, r, t, b = 600, 280, 58, 16, 30, 62
    sx = Scale(-0.5, len(bins) - 0.5, l, w - r)
    sy = Scale(0.4, 1.0, h - b, t)
    body = [yaxis(sy, l, w - r, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda v: f"{v:.0%}"),
            f'<line class="target" stroke-dasharray="4 3" x1="{l}" x2="{w - r}" y1="{sy(0.5):.1f}" y2="{sy(0.5):.1f}"/>']
    for i, x in enumerate(bins):
        cx = sx(i)
        body.append(f'<rect class="model-fill" x="{cx - 34:.1f}" y="{sy(x["share"]):.1f}" width="68" '
                    f'height="{sy(0.4) - sy(x["share"]):.1f}" rx="3"><title>{x["pairs"]:,} pairs</title></rect>'
                    f'<text class="dlabel" x="{cx:.1f}" y="{sy(x["share"]) - 7:.1f}" text-anchor="middle" '
                    f'style="font-weight:600">{x["share"]:.0%}</text>')
        gap = f'{x["lo"]:g}+ pts' if x["hi"] is None else f'{x["lo"]:g}–{x["hi"]:g} pts'
        body.append(f'<text class="tick" x="{cx:.1f}" y="{h - b + 20}" text-anchor="middle">{gap}</text>'
                    f'<text class="tick" x="{cx:.1f}" y="{h - b + 37}" text-anchor="middle">{x["pairs"]:,} pairs</text>')
    body.append(axis_titles(w, h, l, r, t, b, "Gap between the two players' projections", "Higher projection won"))
    return svg(w, h, "".join(body), "How often the higher projection won, by projection gap")


def fig_lineup(lp):
    """Our upcoming lineup: each player's 80% range as a bar, the model's projection
    as a dot, ESPN's as a hollow diamond."""
    order = {s: i for i, s in enumerate(SLOT_ORDER)}
    starters = sorted([x for x in lp["players"] if x["slot"] not in ("BE", "IR")],
                      key=lambda x: order.get(x["slot"], 9))
    bench = sorted([x for x in lp["players"] if x["slot"] == "BE" and not x["bye"]
                    and (x["model"] or x["espn"] or 0) >= 4],
                   key=lambda x: -(x["model"] if x["model"] is not None else (x["espn"] or 0)))[:3]
    rows = [("head", "Starters")] + [("p", x) for x in starters]
    rows += ([("head", "Best bench options")] + [("p", x) for x in bench]) if bench else []
    top_v = max([v for _, x in rows if _ == "p" for v in (x["hi"], x["model"], x["espn"]) if v is not None] + [10])
    w, rh, lab, t = 900, 26, 300, 40
    h = t + rh * len(rows) + 30
    sx = Scale(0, top_v * 1.05, lab, w - 46)
    body = []
    for v in ticks(0, top_v * 1.05, 5):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t - 4}" y2="{h - 26}"/>'
                    f'<text class="tick" x="{sx(v):.1f}" y="{h - 8}" text-anchor="middle">{v:g}</text>')
    for i, (kind, x) in enumerate(rows):
        y = t + rh * i + rh / 2
        if kind == "head":
            body.append(f'<text class="panel" x="0" y="{y + 5:.1f}" style="font-size:13px;letter-spacing:.06em">'
                        f'{esc(x.upper())}</text>')
            continue
        tag = {"QUESTIONABLE": " (Q)", "DOUBTFUL": " (D)", "OUT": " (OUT)"}.get(x["status"] or "", "")
        opp = f' <tspan class="tick">{esc(x["opponent"])}</tspan>' if x["opponent"] and not x["bye"] else \
            (' <tspan class="tick">bye</tspan>' if x["bye"] else "")
        slot = {"BE": x["position"], "RB/WR/TE": "FLEX"}.get(x["slot"], x["slot"])
        body.append(f'<text class="tick" x="0" y="{y + 5:.1f}">{esc(slot)}</text>'
                    f'<text class="dlabel" x="56" y="{y + 5:.1f}">{esc(x["name"])}{esc(tag)}{opp}</text>')
        if x["lo"] is not None and x["hi"] is not None:
            body.append(f'<rect class="range" x="{sx(x["lo"]):.1f}" y="{y - 6:.1f}" width="{sx(x["hi"]) - sx(x["lo"]):.1f}" '
                        f'height="12" rx="6"><title>80% range {x["lo"]:.1f}–{x["hi"]:.1f}</title></rect>')
        if x["espn"] is not None:
            e = sx(x["espn"])
            body.append(f'<path class="espn-mark" d="M{e:.1f},{y - 7:.1f} l7,7 l-7,7 l-7,-7 z">'
                        f'<title>ESPN {x["espn"]:.1f}</title></path>')
        if x["model"] is not None:
            body.append(f'<circle class="model-fill ring" cx="{sx(x["model"]):.1f}" cy="{y:.1f}" r="6">'
                        f'<title>Ours {x["model"]:.1f}</title></circle>'
                        f'<text class="dlabel" x="{w - 4}" y="{y + 5:.1f}" text-anchor="end" style="font-weight:600">'
                        f'{x["model"]:.1f}</text>')
        elif x["espn"] is not None:
            body.append(f'<text class="tick" x="{w - 4}" y="{y + 5:.1f}" text-anchor="end">{x["espn"]:.1f}*</text>')
    leg = legend([("model", "Model", "dot"), ("espn", "ESPN", "diamond2"), ("range", "Model's 80% range", "range")],
                 lab, 16)
    return svg(w, h, leg + "".join(body), "Our lineup with the model's projections and ranges")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

BRIEF_CSS = """
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid var(--ink);
border-bottom:1px solid var(--rule);margin:10px 0 14px}
.cards div{padding:10px 14px;border-right:1px solid var(--rule)}.cards div:last-child{border-right:none}
.cards b{display:block;font-size:30px;font-weight:500;line-height:1.1;font-variant-numeric:lining-nums}
.cards span{display:block;font-size:13.5px;font-weight:600;color:var(--ink);margin-top:2px}
.cards small{display:block;font-size:12px;color:var(--ink-2);margin-top:2px}
.lead{font-size:18px;line-height:1.45;margin:4px 0 6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px 22px}
.grid2 figure{margin:0}
figure h3{font-size:15.5px;margin:4px 0 2px;font-weight:600}
.guide{background:var(--paper-2);border-left:3px solid var(--model);padding:8px 16px;margin:8px 0}
.guide ul{margin:4px 0;padding-left:18px}.guide li{margin:4px 0}
.model-band{fill:var(--model);opacity:.13}.base-band{fill:var(--base);opacity:.13}
.ink-stroke{stroke:var(--ink);stroke-width:1.6}
.range{fill:var(--model);opacity:.22}
.espn-mark{fill:var(--paper);stroke:var(--ink-2);stroke-width:1.8}
.note-sm{font-size:12.5px;color:var(--muted);margin:2px 0 0}
section.week{margin-top:22px}
@media screen and (max-width:760px){.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}
.cards div:nth-child(2){border-right:none}}
@media print{section.week{break-before:page;margin-top:0}.cards b{font-size:20pt}.lead{font-size:11.5pt}
.grid2{gap:4px 16px}figure h3{font-size:10.5pt}}
"""


def card(value, label, sub):
    return f"<div><b>{value}</b><span>{esc(label)}</span><small>{sub}</small></div>"


def page(season, ps, wb, h2h, guide, lp, fs, proj_note, checks_ok):
    h = ps["headline"]
    span = f"weeks {ps['weeks'][0]}–{ps['weeks'][-1]}"
    d, b, c = h["vs_rolling3"], h["bias"], h["cov80"]
    cards = (card(f'{h["mae"]["est"]:.1f} pts', "Typical miss per player",
                  f'vs {h["base_mae"]["est"]:.1f} for a last-3-games average')
             + card(f'{wb["beaten"]} of {wb["weeks"]}', "Weeks the model beat that average",
                    f'{wb["clearly"]} by a clear margin; overall edge {abs(d["est"]):.1f} pts '
                    f'(likely {abs(d["hi"]):.1f} to {abs(d["lo"]):.1f})' if d["lo"] is not None and d["hi"] < 0
                    else "no clear edge overall")
             + card(f'{abs(b["est"]):.1f} pts',
                    "Projections run " + ("low" if b["est"] < 0 else "high") + " by",
                    "players scored more than the model projected, on average" if b["est"] < 0 else
                    "players scored less than the model projected, on average")
             + card(pct(c["est"]), "Landed inside the model's range", "the range is built to catch 8 in 10"))
    dist = ps.get("distribution") or {}
    page1 = f"""<section>
<h2>How the model's projections did, {span}</h2>
<p class="lead">{esc(rb.summary(ps, wb))}</p>
<div class="cards">{cards}</div>
<div class="grid2">
<figure><h3>Typical miss, week by week</h3>{fig_weekly_miss(ps)}
<figcaption>Lower is better. Shading shows how much each week's number could move by chance.</figcaption></figure>
<figure><h3>How close did the model get?</h3>{fig_closeness(dist)}
<figcaption>Share of players whose score landed within 3, 5 or 10 points of the projection
({dist.get("n", 0):,} player-games).</figcaption></figure>
<figure><h3>When the model projected X, players scored&hellip;</h3>{fig_calibration(dict(ps, calibration=[c for c in ps["calibration"] if c["n"] >= 30]))}
<figcaption>Each dot is a group of players by projection (groups of 30+ shown). Dots above the line: they
scored more than projected.</figcaption></figure>
<figure><h3>By position</h3>{fig_positions(ps)}
<figcaption>Bars show the average; thin lines show the likely range of that average.</figcaption></figure>
</div></section>"""
    if lp:
        names = lambda xs: ", ".join(esc(x["name"]) for x in xs)
        empty = [x for x in lp["players"] if x["slot"] not in ("BE", "IR") and not x["available"]]
        benched = [x for x in lp["sit"] if x["available"]]
        moves = (f'Start {names(lp["start"])}' + (f'; bench {names(benched)}' if benched else "")
                 if lp["start"] else "lineup already matches")
        cards2 = (card(f'{lp["model_total"]:.1f}', "Our starters, the model's projected total",
                       f'ESPN: {lp["espn_total"]:.1f}'
                       + (f"; {lp['fallback']} spot{'s' if lp['fallback'] != 1 else ''} use ESPN's number" if lp["fallback"] else ""))
                  + card(str(len(lp["start"])), "Changes the model would make", moves)
                  + card(str(len(empty)), "Starters on bye or out",
                         (names(empty) + ": no one on our roster can fill the spot" if empty and not lp["start"] else
                          names(empty)) if empty else "none"))
        lineup = f"""<div class="cards" style="grid-template-columns:repeat(3,1fr)">{cards2}</div>
<figure>{fig_lineup(lp)}<figcaption>{proj_note} Bar = where the model expects 8 in 10 outcomes to land. * = ESPN's number
(no model projection for that spot). Lineup as set in ESPN when this was made.</figcaption></figure>"""
    else:
        lineup = '<p class="note">ESPN lineup could not be loaded.</p>'
    guide_html = "".join(f"<li>{esc(x)}</li>" for x in guide)
    team = (f"""<figure><h3>Our season so far</h3>{fig_team(fs)}
<figcaption>Points as set, what the model's lineup would have scored, and the best possible lineup. Only
{fs["done"]} weeks: an illustration, not proof.</figcaption></figure>""" if fs else "<div></div>")
    page2 = f"""<section class="week">
<h2>Week {lp["week"] if lp else ""}: our lineup{f" vs {esc(lp['opponent'])}" if lp else ""}</h2>
{lineup}
<div class="guide"><b>How to read the model's numbers this week</b><ul>{guide_html}</ul></div>
<div class="grid2">
<figure><h3>Close calls: how often the higher projection won</h3>{fig_head_to_head(h2h)}
<figcaption>Every pair of same-position players both projected {rb.STARTABLE:g}+ in the same week. The dashed
line is a coin flip (50%).</figcaption></figure>
{team}
</div></section>"""
    foot = (f"How it's measured: every number uses the projections saved before each kickoff, scored against what "
            f"players actually did ({h['n']:,} player-games in {h['games']} games, {span}). Players who never "
            "recorded a stat are left out, never counted as zero. Likely ranges are 95% intervals. "
            + ("All data checks passed. " if checks_ok else "Some data checks failed: see the full report. ")
            + "Full detail: the season review report.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Projections Brief</title><style>{CSS}{BRIEF_CSS}</style></head>
<body><main>
<p class="kicker">Baby Back Gibbs &middot; projections brief &middot; {season}</p>
<h1>How the model's projections are doing{f", and week {lp['week']}" if lp else ""}</h1>
<div class="byline">{datetime.now():%A %B %d, %Y}</div>
{page1}
<footer>{esc(foot)}</footer>
{page2}
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--next-week", type=int, default=None)
    ap.add_argument("--team", type=int, default=int(os.environ.get("ESPN_TEAM_ID", 1)))
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    with sqlite3.connect(DB_PATH) as con:
        games = kickoff_times(args.season, con)
        played = sorted(int(w) for w in games.loc[games["kickoff"] <= now, "week"].unique())
        rows, summaries = wr.collect(con, args.season, played, games, now)
    ts = wr.test_set(rows, games, played)
    complete = [w["week"] for w in ts if w["complete"]]
    nxt = args.next_week or (max(complete) + 1 if complete else 1)
    ps = wr.player_section(rows, max(complete), summaries, weeks=complete)
    checks = wr.audit(rows, ps, ts)
    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}: {c['detail']}")
    wb = rb.weeks_beaten(ps)
    wk = wr.scored(rows)
    h2h = rb.head_to_head(wk[wk["week"].isin(complete)])

    path = WEEKLY_DIR / f"weekly_{args.season}_wk{nxt}.json"
    proj = rb.week_projections(pd.DataFrame(json.loads(path.read_text()))) if path.exists() else pd.DataFrame()
    made = datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None
    proj_note = (f"The model's projections and ranges as of {made:%b %d %H:%M}." if made else "No projections file found.")
    lp = week_lineup(args.team, nxt, proj) if not proj.empty else None
    fs = fantasy_season(rows, complete, args.team)

    html_text = page(args.season, ps, wb, h2h, rb.trust_guide(ps, h2h), lp, fs, proj_note,
                     all(c["ok"] for c in checks))
    out = REPORT_DIR / f"review_{args.season}_brief.html"
    docs = (PROJECT_ROOT / "docs").resolve()
    if docs == out.resolve().parent or docs in out.resolve().parents:
        raise SystemExit(f"refusing to write a private report inside the published site: {out}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({len(html_text):,} bytes)")
    print("  " + rb.summary(ps, wb))
    for g in rb.trust_guide(ps, h2h):
        print("  - " + g)


if __name__ == "__main__":
    main()
