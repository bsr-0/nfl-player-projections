"""The Weekly Review: an independent evaluation of the served PPR projections.

Scores the player projections the site published before each kickoff against
what happened, with intervals on every headline number and a naive baseline
beside them. Sections: the test set (model, freeze time and counts per week),
accuracy, 80% ranges by position, where the misses come from, what the model
relies on (2025 held out, from scripts/feature_group_importance.py), and your
fantasy team as an illustration. Computation lives in
src/evaluation/weekly_review.py; this file draws the page (inline SVG, no
scripts, no external requests).

In the season view the headline uses complete weeks only; a week still being
scored is shown and labelled provisional.

Player actuals come from player_weekly_stats, which the Tuesday refresh loads;
before that the page says PARTIAL.

Output: data/espn_private/reports/review_{season}_wk{N}.html, or
review_{season}_season.html with --all-weeks (private).

Usage:
    python scripts/render_weekly_review.py              # latest week with a final game
    python scripts/render_weekly_review.py --week 4
    python scripts/render_weekly_review.py --all-weeks  # season to date
"""
import argparse
import html
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DB_PATH, ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.evaluation import weekly_review as wr
from src.evaluation.published_record import kickoff_times
from src.evaluation.review_charts import (CSS, Scale, axis_titles, esc, f, fig_calibration, fig_team,
                                          legend, minus, minus_nums, pct, signed, svg, ticks, yaxis)
from src.evaluation.team_history import fantasy_season

REPORT_DIR = ESPN_PRIVATE_DIR / "reports"
IMPORTANCE = PROJECT_ROOT / "data" / "experiments" / "feature_group_importance_2025" / "importance.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def default_week(games: pd.DataFrame) -> int:
    done = games[games["final"]]
    return int(done["week"].max()) if not done.empty else 1


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def fig_mae_by_week(ps):
    """Weekly MAE, model against the last-3 average on the same player-weeks,
    with game-resampled intervals; a week still being scored is hollow."""
    rows = [r for r in ps["by_week"] if r["model_mae_paired"]["est"] is not None]
    if not rows:
        return ""
    w, h, l, r, t, b = 640, 318, 66, 44, 44, 72
    weeks = [x["week"] for x in rows]
    m_of, b_of = (lambda x: x["model_mae_paired"]), (lambda x: x["base_mae"])
    vals = [v for x in rows for ci in (m_of(x), b_of(x)) for v in (ci["lo"], ci["hi"], ci["est"]) if v is not None]
    lo, hi = math.floor(min(vals) * 2) / 2 - 0.25, math.ceil(max(vals) * 2) / 2 + 0.25
    sx = Scale(min(weeks) - 0.5, max(weeks) + 0.5, l, w - r)
    sy = Scale(lo, hi, h - b, t)
    body = [yaxis(sy, l, w - r, ticks(lo, hi, 5), lambda v: f"{v:.1f}")]
    for x in rows:
        sub = f'n={x["n_paired"]}' + (", provisional" if x.get("partial") else "")
        body.append(f'<text class="tick" x="{sx(x["week"]):.1f}" y="{h - b + 18}" text-anchor="middle">Wk {x["week"]}</text>'
                    f'<text class="tick" x="{sx(x["week"]):.1f}" y="{h - b + 34}" text-anchor="middle">{sub}</text>')
    done = [x for x in rows if not x.get("partial")]
    for get, cls, off, name in ((b_of, "base", 5, "Last-3 average"), (m_of, "model", -5, "Model")):
        line = [(sx(x["week"]) + off, sy(get(x)["est"])) for x in done]
        if len(line) > 1:
            body.append(f'<polyline class="{cls}-stroke" fill="none" stroke-width="2" points="'
                        + " ".join(f"{a:.1f},{c:.1f}" for a, c in line) + '"/>')
        for x in rows:
            ci, cx = get(x), sx(x["week"]) + off
            if ci["lo"] is not None:
                body.append(f'<line class="{cls}-stroke" stroke-width="2" x1="{cx:.1f}" x2="{cx:.1f}" '
                            f'y1="{sy(ci["lo"]):.1f}" y2="{sy(ci["hi"]):.1f}"/>')
            mark = (f'class="hollow" style="stroke:var(--{cls})"' if x.get("partial") else f'class="{cls}-fill ring"')
            body.append(f'<circle {mark} cx="{cx:.1f}" cy="{sy(ci["est"]):.1f}" r="4.5"><title>Week {x["week"]} '
                        f'{name}: MAE {ci["est"]:.2f} (95% {f(ci["lo"])} to {f(ci["hi"])}), n={x["n_paired"]}'
                        f'{", provisional" if x.get("partial") else ""}</title></circle>')
    items = [("model", "Model", "dot"), ("base", "Last-3 average", "dot")]
    items += [("model", "Provisional week", "hollow")] if len(done) < len(rows) else []
    body.append(legend(items, l, 16))
    body.append(axis_titles(w, h, l, r, t, b, "Week (n = same player-weeks for both)", "MAE (pts)"))
    return svg(w, h, "".join(body), "Mean absolute error by week, model and last-3 average on the same player-weeks")


def fig_scatter(ps):
    """F2: predicted vs actual, one panel per position; hollow = outside 80% range.
    Separate panels in a grid: four across on a page, two on a phone."""
    pts = pd.DataFrame(ps["points"])
    if pts.empty:
        return ""
    panels = [p for p in wr.POSITIONS if (pts["position"] == p).any()]
    pw, ph, pl = 300, 256, 54
    has_range = pts["prediction_ci80_lower"].notna().all()
    items = [("model", "Inside 80% range", "dot"), ("model", "Outside range", "hollow")]
    items += [] if has_range else [("bench", "No range published", "dot")]
    leg = svg(600, 26, legend(items + [("diag", "Exact", "line")], 8, 14), "Legend")
    out = [leg.replace("<svg ", '<svg style="width:auto;height:21px;max-width:100%" ', 1)]
    cells = []
    for p in panels:
        d = pts[pts["position"] == p]
        top = max(10, math.ceil(max(d["actual_points"].max(), d["predicted_points"].max()) / 10) * 10)
        sx, sy = Scale(0, top, pl, pw - 8), Scale(0, top, ph - 48, 26)
        body = [f'<text class="panel" x="{pl}" y="15">{p} <tspan class="tick">n={len(d)}</tspan></text>']
        for v in ticks(0, top, 4):
            body.append(f'<line class="grid" x1="{pl}" x2="{pw - 8}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                        f'<text class="tick" x="{pl - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
                        f'<text class="tick" x="{sx(v):.1f}" y="{ph - 28}" text-anchor="middle">{v:g}</text>')
        body.append(axis_titles(pw, ph, pl, 8, 26, 48, "Projected (pts)", "Actual (pts)"))
        body.append(f'<line class="diag" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(top):.1f}" y2="{sy(top):.1f}"/>')
        for r in d.itertuples():
            lo, hi = r.prediction_ci80_lower, r.prediction_ci80_upper
            known = lo == lo and hi == hi
            cls = ("bench-fill ring" if not known else
                   "model-fill ring" if lo <= r.actual_points <= hi else "hollow")
            rng = f", 80% range {lo:.1f}–{hi:.1f}" if known else ""
            body.append(f'<circle class="{cls}" cx="{sx(r.predicted_points):.1f}" cy="{sy(r.actual_points):.1f}" r="4">'
                        f'<title>{esc(r.name)}: projected {r.predicted_points:.1f}, scored {r.actual_points:.1f}{rng}</title></circle>')
        cells.append(svg(pw, ph, "".join(body), f"{p}: projected against actual points"))
    return out[0] + '<div class="quad">' + "".join(cells) + "</div>"


def fig_distribution(ps):
    """Signed misses for the model and the last-3 average, same players."""
    dist = ps.get("distribution") or {}
    if not dist:
        return ""
    w, h, l, r, t, b = 640, 300, 66, 16, 44, 56
    edges = dist["edges"]
    top = max(max(dist["model"]), max(dist["rolling3"])) * 1.1
    sx, sy = Scale(edges[0], edges[-1], l, w - r), Scale(0, top, h - b, t)
    body = [yaxis(sy, l, w - r, ticks(0, top, 4), lambda v: f"{v * 100:.0f}%")]
    for i, share in enumerate(dist["model"]):
        x0, x1 = sx(edges[i]) + 1, sx(edges[i + 1]) - 1
        body.append(f'<rect class="model-fill" x="{x0:.1f}" y="{sy(share):.1f}" width="{x1 - x0:.1f}" '
                    f'height="{sy(0) - sy(share):.1f}" rx="2"><title>Miss {edges[i]:+g} to {edges[i + 1]:+g}: '
                    f'model {share:.1%}, last-3 {dist["rolling3"][i]:.1%}</title></rect>')
    step = []
    for i, share in enumerate(dist["rolling3"]):
        step += [f"{sx(edges[i]):.1f},{sy(share):.1f}", f"{sx(edges[i + 1]):.1f},{sy(share):.1f}"]
    body.append(f'<polyline class="base-stroke" fill="none" stroke-width="2" points="{" ".join(step)}"/>')
    body.append(f'<line class="zero" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{t - 4}" y2="{h - b}"/>')
    for v in range(-24, 25, 6):
        lab = f"{v:+d}" if v else "0"
        body.append(f'<text class="tick" x="{sx(v):.1f}" y="{h - b + 18}" text-anchor="middle">{lab}</text>')
    body.append(f'<text class="axislabel" x="{l}" y="{h - 6}">← too low</text>'
                f'<text class="axislabel" x="{w - r}" y="{h - 6}" text-anchor="end">too high →</text>')
    body.append(axis_titles(w, h, l, r, t, b, "Projected \u2212 actual (pts)", "Share of players"))
    body.append(legend([("model", "Model", "block"), ("base", "Last-3 average", "line")], l, 16))
    return svg(w, h, "".join(body), "Distribution of misses, model and last-three average")


def fig_segments(ps):
    """Forest plot: four metrics (columns) for each position and experience
    group (rows), every one with its interval and sample size."""
    groups = ps.get("segments") or []
    if not groups:
        return ""
    rows = []
    for g in groups:
        rows.append(("head", g["group"]))
        rows += [("row", x) for x in g["rows"]]
    lab_w, gap, pw, rh, t = 168, 22, 170, 27, 76
    w = lab_w + 4 * pw + 3 * gap + 18
    h = t + rh * len(rows) + 8
    seg = [x for kind, x in rows if kind == "row"]
    overall = {k: ps["headline"][k]["est"] for k in ("mae", "bias", "vs_rolling3", "cov80")}

    def dom(key, ref):
        vals = [v for x in seg for v in (x[key]["lo"], x[key]["hi"], x[key]["est"]) if v is not None] + ref
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08 or 0.5
        return lo - pad, hi + pad

    panels = [("mae", ("Typical miss", "points"), dom("mae", [0]), None, lambda v: f"{v:.1f}"),
              ("bias", ("Bias", "points, ← projected low"), dom("bias", [0]), 0.0, lambda v: f"{v:+.1f}"),
              ("vs_rolling3", ("vs last-3 average", "points, ← model better"), dom("vs_rolling3", [0]), 0.0,
               lambda v: f"{v:+.1f}"),
              ("cov80", ("In 80% range", "share of players"), dom("cov80", [0.8]), 0.8, lambda v: f"{v:.0%}")]
    body = []
    for i, (key, title, (lo, hi), ref, fmt) in enumerate(panels):
        x0 = lab_w + i * (pw + gap)
        sx = Scale(lo, hi, x0 + 4, x0 + pw - 4)
        body.append(f'<text class="panel" x="{x0 + pw / 2:.0f}" y="18" text-anchor="middle" '
                    f'style="font-size:15px">{esc(title[0])}</text>'
                    f'<text class="axislabel" x="{x0 + pw / 2:.0f}" y="36" text-anchor="middle">{esc(title[1])}</text>')
        title = title[0]
        tfmt = (lambda v: f"{v:.0%}") if key == "cov80" else (lambda v: f"{v:g}".replace("-", "\u2212"))
        for v in ticks(lo, hi, 3):
            body.append(f'<line class="grid" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t - 14}" y2="{h - 8}"/>'
                        f'<text class="tick" x="{sx(v):.1f}" y="{t - 20}" text-anchor="middle">{tfmt(v)}</text>')
        if ref is not None:
            body.append(f'<line class="target" x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{t - 14}" y2="{h - 8}"/>')
        if overall[key] is not None:
            body.append(f'<line class="bench-stroke" stroke-dasharray="3 3" x1="{sx(overall[key]):.1f}" '
                        f'x2="{sx(overall[key]):.1f}" y1="{t - 14}" y2="{h - 8}" style="stroke-width:1.5"/>')
        for j, (kind, x) in enumerate(rows):
            if kind == "head":
                continue
            y = t + rh * j + rh / 2
            m = x[key]
            if m["est"] is None:
                continue
            if m["lo"] is not None:
                body.append(f'<line class="model-stroke" stroke-width="2" x1="{sx(m["lo"]):.1f}" '
                            f'x2="{sx(m["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            n = m.get("n", x["n"])
            body.append(f'<circle class="model-fill ring" cx="{sx(m["est"]):.1f}" cy="{y:.1f}" r="5">'
                        f'<title>{esc(x["label"])}, {esc(title)}: {fmt(m["est"])} (95% '
                        f'{fmt(m["lo"]) if m["lo"] is not None else "n/a"} to '
                        f'{fmt(m["hi"]) if m["hi"] is not None else "n/a"}), n={n}</title></circle>')
    for j, (kind, x) in enumerate(rows):
        y = t + rh * j + rh / 2
        if kind == "head":
            body.append(f'<text class="panel" x="0" y="{y + 5:.1f}" style="font-size:14px;'
                        f'letter-spacing:.06em">{esc(x.upper())}</text>')
        else:
            body.append(f'<line class="grid" x1="0" x2="{w}" y1="{y + rh / 2:.1f}" y2="{y + rh / 2:.1f}"/>'
                        f'<text class="dlabel" x="10" y="{y + 5:.1f}">{esc(x["label"])}</text>'
                        f'<text class="tick" x="{lab_w - 14}" y="{y + 5:.1f}" text-anchor="end">n={x["n"]:,}</text>')
    return svg(w, h, "".join(body), "Accuracy by position, projection and experience")


def team_html(fs):
    """Your team, kept apart from the evaluation: a handful of matchups
    illustrates, it does not test anything."""
    if not fs:
        return ""
    gap = fs["model_lineup"] - fs["actual"]
    span = f'week{"s" if fs["done"] != 1 else ""} 1\u2013{fs["weeks"][fs["done"] - 1]["week"]}' \
        if fs["done"] else "no completed week yet"
    flips = ("; it flips " + ", ".join(
        f'week {w["week"]} ({w["result"]}\u2192{w["model_result"]})' for w in fs["flips"])
        if fs["flips"] else "; no result changes")
    pend = [w["week"] for w in fs["weeks"] if w["pending"]]
    pend_txt = f' Week {", ".join(map(str, pend))} (hollow) is still being played; not in the totals.' if pend else ""
    pr = fs.get("projections") or {}
    proj_txt = minus_nums(f"""<p class="note"><b>Player projections on your roster</b> (starters and bench, {span}, players
with a stat line, n = {pr["n"]} player-games): model typical miss {pr["model_mae"]:.1f} pts, bias
{pr["model_bias"]:+.1f}; ESPN's {pr["espn_mae"]:.1f}, bias {pr["espn_bias"]:+.1f}, on the same players and your
league's scoring (standard full PPR). The model was closer on {pr["model_closer"]} of {pr["n"]}. Too few to rank
either source; the league-wide numbers above are the evidence.</p>""") if pr.get("n") else ""
    text = minus_nums(f"""<p class="note"><b>Lineup outcomes, {esc(fs["team"])}, {span}.</b> As set {fs["actual"]:.1f} pts,
{fs["record"]}. The model's lineups (picked from the same lineup and bench on pre-kickoff numbers; ESPN's for K
and D/ST) {fs["model_lineup"]:.1f} ({gap:+.1f}), {fs["model_record"]}. ESPN's projections {fs["espn_lineup"]:.1f};
hindsight best {fs["hindsight"]:.1f}. Model better in {fs["model_better"]} of {fs["done"]} weeks, worse in
{fs["model_worse"]}, {fs["changes"]} lineup change{"s" if fs["changes"] != 1 else ""}{flips}.{pend_txt}</p>
<p class="note">{fs["done"]} matchups illustrate how the projections play out for one team; they are not evidence
about the lineup picks in general.</p>""")
    return f"""<div class="pair"><figure>{fig_team(fs)}</figure><div>{text}{proj_txt}</div></div>"""


def _when(iso):
    try:
        t = datetime.fromisoformat(str(iso))
    except ValueError:
        return esc(iso)
    return f"{t:%b} {t.day} {t:%H:%M}"


def p1(v):
    return f(v, 1)


def p2(v):
    return f(v, 2)


def ci_txt(ci, fmt=p2):
    """'lo to hi' for a mean_ci result, or a dash when there is no interval."""
    return "&ndash;" if ci["lo"] is None else minus(f"{fmt(ci['lo'])} to {fmt(ci['hi'])}")


def cell(ci, fmt=p2, signed_=False, hot=False, stack=False):
    """A table cell: the estimate, then its 95% interval in muted type
    (beside it, or beneath it in narrow tables)."""
    if ci["est"] is None:
        return '<td class="n">&ndash;</td>'
    est = ("+" if signed_ and ci["est"] > 0 else "") + fmt(ci["est"])
    return (f'<td class="n{" hot" if hot else ""}">{minus(est)}{"<br>" if stack else " "}'
            f'<span class="muted{" ci" if stack else ""}">{ci_txt(ci, fmt)}</span></td>')


def accounting_html(ps):
    a = ps["accounting"]
    span = _span(ps["weeks"])

    def reasons(c):
        r = c.get("reasons") or {}
        return ", ".join(f"{k} {v:,}" for k, v in sorted(r.items(), key=lambda kv: -kv[1]))

    rows = [f'<tr class="total"><td>Published for final games, {span}</td><td class="n">{a["published"]:,}</td>'
            f'<td class="n"></td><td></td></tr>',
            f'<tr><td>Scored: stat line recorded</td><td class="n">{a["scored"]:,}</td><td class="n"></td>'
            f'<td class="muted">the population of every accuracy measure</td></tr>',
            f'<tr><td>No stat line</td><td class="n">{a["no_stat_line"]:,}</td>'
            f'<td class="n">{a["no_stat_projected5"]:,}</td><td class="muted">not scored'
            + (f'; {a["no_stat_projected5_listed_out"]} of the {a["no_stat_projected5"]} projected 5+ were listed '
               'Out or Doubtful' if a["no_stat_projected5_listed_out"] else "") + '</td></tr>']
    for c in a["classes"]:
        rows.append(f'<tr class="sub"><td>{esc(c["label"])}</td><td class="n">{c["n"]:,}</td>'
                    f'<td class="n">{c["projected5"]:,}</td><td class="muted">{esc(reasons(c))}</td></tr>')
    return f"""<div class="tw"><table class="t acct"><thead><tr><th></th><th class="n">Projections</th>
<th class="n">Projected 5+</th><th>Detail</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="note">Classes come from the weekly roster and snap tables: <b>verified did not play</b> means zero
offensive snaps on the snap sheet, or not on the game-day active roster with no snap; <b>unresolved</b> means
the tables cannot say (active with no snap row, no roster record, or a week-1 row without a player id). No
unresolved record is treated as a zero, and none of these projections enters an accuracy measure. Separately,
{a["unprojected"]:,} players recorded a stat line with no published projection ({a["unprojected_5plus"]:,} of them
scored 5+).</p>"""


def versions_html(vt, ps, ts):
    if not vt or not vt["rows"]:
        return ""
    frozen = {w["week"]: w["snapshots"] for w in ts or []}

    def tr(x, first, cls=""):
        when = "<br>".join(_when(s["published_at"]) for s in frozen.get(x.get("week"), [])
                           if s["mode"] == x.get("mode")) if "week" in x else ""
        return (f'<tr class="{cls}"><td>{first}</td><td>{esc(x["label"])}</td><td class="nw">{when}</td>'
                f'<td class="n">{x["n"]:,} <span class="muted">({x["games"]})</span></td>'
                f'{cell(x["mae"], p2)}{cell(x["vs_rolling3"], p2, True)}{cell(x["bias"], p1, True)}'
                f'{cell(x["cov80"], pct) if x["n_range"] else "<td class=n>no range</td>"}</tr>')
    body = []
    for x in vt["rows"]:
        tag = "" if x["complete"] else f' <span class="muted">(provisional, {x["games"]} of {x["games_scheduled"]} games)</span>'
        body.append(tr(x, f'{x["week"]}{tag}', "" if x["complete"] else "dim"))
    for x in vt["pooled"]:
        body.append(tr(x, f'Wk{"s" if len(x["weeks"]) > 1 else ""} {_span(x["weeks"], short=True)}', "pool"))
    h = ps["headline"]
    body.append(tr({**h, "label": f'All {len(vt["versions_in_headline"])} versions pooled'},
                   f'Headline, {_span(ps["weeks"], short=True)}', "total"))
    served = {}
    for x in vt["rows"]:
        if x["complete"]:
            served.setdefault(x["label"], []).append((x["week"], x["games"], x["n_range"] > 0))
    parts = []
    for label, wk in served.items():
        weeks = sorted({w for w, _, _ in wk})
        games = sum(g for _, g, _ in wk)
        ranges = "" if any(r for *_, r in wk) else ", no ranges published"
        parts.append(f"{esc(label.lower())} in week{'s' if len(weeks) > 1 else ''} {_span(weeks, short=True)} "
                     f"({games} game{'s' if games != 1 else ''}{ranges})")
    return f"""<div class="tw"><table class="t"><thead><tr><th>Week</th><th>Model served</th><th>Frozen (UTC)</th>
<th class="n">n <span class="muted">(games)</span></th><th class="n">MAE</th><th class="n">vs last-3</th>
<th class="n">Signed error</th><th class="n">In 80% range</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>
<p class="note">The headline pools {len(vt["versions_in_headline"])} served version{"s" if len(vt["versions_in_headline"]) != 1 else ""}: {"; ".join(parts)}.
Each projection is the last version committed before its game's kickoff.
vs last-3 = MAE minus the last-3 average's MAE on the same player-weeks (negative = model better). Muted figures
are 95% intervals with games resampled; one game gives no interval.</p>"""


def failure_html(ps):
    rows = ps.get("failure_panel") or []
    if not rows:
        return ""
    body = []
    for x in rows:
        te = x["label"] == "TE"
        ranged = x["n_range"] > 0
        body.append(f'<tr><td>{esc(x["label"])}</td><td class="n">{x["n"]:,}</td>{cell(x["mae"], p1)}'
                    f'{cell(x["bias"], p1, True, hot=x["label"] == "QB")}'
                    f'{cell(x["cov80"], pct, hot=te) if ranged else "<td class=n>&ndash;</td>"}'
                    f'{cell(x["above"], pct, hot=te) if ranged else "<td class=n>&ndash;</td>"}</tr>')
    return f"""<div class="tw"><table class="t"><thead><tr><th></th><th class="n">n</th><th class="n">MAE</th>
<th class="n">Signed error</th><th class="n">In 80% range</th><th class="n">Above range</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<p class="note">Signed error is projected minus actual (negative = projected low). <b>Above range</b> = scored more
than the top of the published 80% range; a well-set range puts about 10% there. Projected {wr.HIGH_PROJECTION:g}+
uses the published projection; likely starters averaged {wr.STARTER_SNAPS:.0%} or more of their team's offensive
snaps over their previous three games (both known before kickoff). Fuller splits are in the appendix.</p>"""


def ranges_html(ps):
    rg = ps.get("ranges") or {}
    if not rg.get("rows"):
        return ""
    body = []
    for r in rg["rows"]:
        c = r["cov80"]
        off = c["lo"] is not None and not c["lo"] <= 0.80 <= c["hi"]
        body.append(f'<tr class="{"total" if r["label"] == "All" else ""}{" flag" if off else ""}">'
                    f'<td>{esc(r["label"])}</td><td class="n">{r["n_range"]:,}</td>{cell(c, pct, stack=True)}'
                    f'<td class="n">{f(r["width"], 1)}</td>{cell(r["below"], pct, stack=True)}'
                    f'{cell(r["above"], pct, stack=True)}</tr>')
    wk = rg.get("weeks_without") or []
    gap = (f'Week {", ".join(map(str, wk))} ({rg["n_without"]:,} scored players) was published without ranges '
           "and is left out of coverage. ") if rg.get("n_without") else ""
    return f"""<div><div class="tw"><table class="t"><thead><tr><th></th><th class="n">n</th>
<th class="n">In range</th><th class="n">Width</th><th class="n">Below</th><th class="n">Above</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<p class="note"><b>80% ranges by position.</b> Aim: 80% inside, about 10% below and 10% above. Width in points.
Red rows miss 80% beyond their interval. Intervals resample games, as everywhere in this report. {gap}</p></div>"""


def splits_html(ps):
    groups = ps.get("splits") or []
    if not groups:
        return ""

    def table(gs):
        body = []
        for g in gs:
            body.append(f'<tr class="sep"><td colspan="5">{esc(g["group"])}</td></tr>')
            for r in g["rows"]:
                body.append(f'<tr><td>{esc(r["label"])}</td><td class="n">{r["n"]:,}</td>{cell(r["mae"], p1)}'
                            f'{cell(r["bias"], p1, True)}<td class="n">{pct(r["share"])}</td></tr>')
        return (f'<div class="tw"><table class="t"><thead><tr><th></th><th class="n">n</th><th class="n">MAE</th>'
                f'<th class="n">Signed error</th><th class="n">Share</th></tr></thead><tbody>{"".join(body)}'
                '</tbody></table></div>')
    return f'<div class="pair tight">{table(groups[0::2])}{table(groups[1::2])}</div>'


def checks_html(checks, imp):
    rows = [f'<tr class="{"" if c["ok"] else "flag"}"><td>{"Pass" if c["ok"] else "FAIL"}</td>'
            f'<td>{esc(c["check"])}</td><td class="muted">{esc(minus_nums(c["detail"]))}</td></tr>' for c in checks]
    ta = (imp or {}).get("timing_audit")
    if ta:
        for t in ta.get("truncation", []):
            ok = not t["features_changed"] and not t["rows_missing"]
            rows.append(f'<tr class="{"" if ok else "flag"}"><td>{"Pass" if ok else "FAIL"}</td>'
                        f'<td>Standard 2025 feature build: removing weeks after {t["cutoff_week"]} leaves earlier '
                        'rows unchanged</td>'
                        f'<td class="muted">{t["rows_compared"]:,} rows &times; {t["features_compared"]} features; '
                        f'{len(t["features_changed"])} changed'
                        + (": " + esc(", ".join(f'{k} ({v["rows"]:,} rows)' for k, v in t["features_changed"].items()))
                           if t["features_changed"] else "") + '</td></tr>')
        asof = ta.get("importance_built_as_of")
        detail = (f"{ta.get('asof_rows') or 0:,} rows; week k rebuilt from data through week k" if asof
                  else "not yet: rerun scripts/feature_group_importance.py")
        rows.append(f'<tr class="{"" if asof else "flag"}"><td>{"Pass" if asof else "FAIL"}</td>'
                    '<td>Variable importance uses features rebuilt as of each week</td>'
                    f'<td class="muted">{detail}</td></tr>')
    return f"""<div class="tw"><table class="t"><thead><tr><th></th><th>Check</th><th>Result</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def _span(weeks, short=False):
    weeks = list(weeks)
    if not weeks:
        return ""
    if len(weeks) == 1:
        return f"{weeks[0]}" if short else f"week {weeks[0]}"
    return (f"{weeks[0]}–{weeks[-1]}" if short else f"weeks {weeks[0]}–{weeks[-1]}")


def fig_importance(imp):
    """Grouped permutation importance by position: rows are feature families,
    panels positions, each dot the growth in typical miss with its interval."""
    pos = [p for p in wr.POSITIONS if p in imp.get("positions", {})]
    if not pos:
        return ""
    fams = sorted({k for p in pos for k in imp["positions"][p]["families"]},
                  key=lambda k: -np.mean([imp["positions"][p]["families"].get(k, {}).get("est") or 0 for p in pos]))
    lab_w, gap, pw, rh, t = 238, 18, 150, 23, 62
    w = lab_w + len(pos) * pw + (len(pos) - 1) * gap + 14
    h = t + rh * len(fams) + 8
    vals = [v for p in pos for x in imp["positions"][p]["families"].values()
            for v in (x.get("lo"), x.get("hi"), x.get("est")) if v is not None] + [0]
    lo, hi = min(vals), max(vals)
    lo, hi = lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06
    body = []
    for i, p in enumerate(pos):
        x0 = lab_w + i * (pw + gap)
        sx = Scale(lo, hi, x0 + 4, x0 + pw - 4)
        base = imp["positions"][p]
        body.append(f'<text class="panel" x="{x0 + pw / 2:.0f}" y="18" text-anchor="middle" style="font-size:15px">'
                    f'{p}</text><text class="axislabel" x="{x0 + pw / 2:.0f}" y="34" text-anchor="middle">'
                    f'base {base["base_mae"]:.2f}, n={base["n"]:,}</text>')
        for v in ticks(lo, hi, 3):
            body.append(f'<line class="grid" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t - 10}" y2="{h - 8}"/>'
                        f'<text class="tick" x="{sx(v):.1f}" y="{t - 14}" text-anchor="middle">'
                        f'{signed(v)}</text>')
        body.append(f'<line class="target" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{t - 10}" y2="{h - 8}"/>')
        for j, fam in enumerate(fams):
            m = base["families"].get(fam)
            if not m or m.get("est") is None:
                continue
            y = t + rh * j + rh / 2
            if m.get("lo") is not None:
                body.append(f'<line class="model-stroke" stroke-width="2" x1="{sx(m["lo"]):.1f}" x2="{sx(m["hi"]):.1f}" '
                            f'y1="{y:.1f}" y2="{y:.1f}"/>')
            body.append(f'<circle class="model-fill ring" cx="{sx(m["est"]):.1f}" cy="{y:.1f}" r="5">'
                        f'<title>{p}, {esc(fam)} ({m.get("features")} features): MAE {m["est"]:+.3f} '
                        f'(95% {f(m.get("lo"), 3)} to {f(m.get("hi"), 3)})</title></circle>')
    for j, fam in enumerate(fams):
        y = t + rh * j + rh / 2
        body.append(f'<line class="grid" x1="0" x2="{w}" y1="{y + rh / 2:.1f}" y2="{y + rh / 2:.1f}"/>'
                    f'<text class="dlabel" x="0" y="{y + 5:.1f}">{esc(fam)}</text>')
    body.append(f'<text class="axislabel" x="{lab_w + (w - lab_w) / 2:.0f}" y="{h + 14}" text-anchor="middle">'
                f'Increase in typical miss when the family is shuffled (pts)</text>')
    return svg(w, h + 20, "".join(body), "Grouped permutation importance by position, 2025 held out")


def importance_summary_html(imp):
    """Leading feature families by position on 2025 held out, with the
    timing audit that makes it a fair held-out test."""
    if not imp:
        return ('<p class="note">Variable importance not computed yet: run '
                '<code>python scripts/feature_group_importance.py</code>.</p>')
    body = []
    for p in wr.POSITIONS:
        x = imp["positions"].get(p)
        if not x:
            continue
        fams = sorted(x["families"].items(), key=lambda kv: -(kv[1]["est"] or 0))
        lead = [(k, v) for k, v in fams if v.get("lo") is not None and v["lo"] > 0][:3]
        txt = "; ".join(f'{esc(k)} <b>+{v["est"]:.2f}</b> <span class="muted">{v["lo"]:.2f} to {v["hi"]:.2f}</span>'
                        for k, v in lead) or '<span class="muted">none clear of zero</span>'
        body.append(f'<tr><td>{p}</td><td class="n">{x["n"]:,}</td><td class="n">{x["base_mae"]:.2f}</td>'
                    f'<td>{txt}</td></tr>')
    ta = imp.get("timing_audit") or {}
    al = ta.get("alignment") or {}
    names = {"implied_team_total": "betting line", "wind_speed_mph": "wind", "injury_score": "injury report"}
    origin = ", ".join(f'{names.get(k.split("(")[-1].rstrip(")"), k)} {v["matches_origin_game"]:.0%}'
                       for k, v in al.items() if v.get("rows_where_games_differ") and "is_dome" not in k)
    moved = max((x.get("asof_max_move") or 0) for x in imp["positions"].values())
    looked = sorted({fn for t in ta.get("truncation", []) for fn in t["features_changed"]})
    asof = ta.get("importance_built_as_of")
    if not ta:
        audit_txt = "<b>Timing audit not run yet</b> (<code>--audit-only</code>)."
    else:
        audit_txt = (
            f"<b>Timing audit.</b> Every row predicts a later game than the one its features describe, and the "
            f"betting-line, weather and injury inputs come from that earlier, already-played game (share of rows matching it: {origin}). "
            + (f"In the standard 2025 build, {len(looked)} of {ta['truncation'][0]['features_compared']} features "
               f"saw later weeks ({esc(', '.join(looked))}); " if looked else "No feature saw later weeks; ")
            + (f"these numbers use features rebuilt week by week, each from data through that week only, which "
               f"moved the held-out predictions by at most {moved:.2f} points. " if asof
               else "<b>these numbers still use that build; rerun the script to rebuild week by week.</b> ")
            + "Not verifiable: later stat corrections in the frozen inputs, and injury-report timing within a "
              "week (the 2025 feed has no timestamps).")
    return f"""<div class="tw"><table class="t"><thead><tr><th></th><th class="n">n</th>
<th class="n">Base MAE</th><th>Leading families: MAE increase when shuffled <span class="muted">95%</span></th></tr>
</thead><tbody>{"".join(body)}</tbody></table></div>
<p class="note"><b>Model reliance, not causation.</b> The served weekly model's 2025 fold (trained through 2024,
before the pace blend) predicts 2025 again with one family of features shuffled; the rise in MAE is how much the
model leans on that family. Correlated families can stand in for one another, so a near-zero family may carry
information another family duplicates: near zero is not, on its own, a reason to delete features.
{imp.get("repeats")} shuffles, target games resampled; 2026 is not used.</p>
<p class="note">{audit_txt}</p>"""


def importance_html(imp):
    if not imp:
        return ""
    return f"""<figure>{fig_importance(imp)}
<figcaption><b>Every feature family, 2025 held out.</b> Each dot is the rise in MAE when the family is shuffled;
lines are 95% intervals. Reliance, not cause.</figcaption></figure>"""


def rng(lo, hi, fmt=f):
    return "&ndash;" if lo is None or hi is None else f"{fmt(lo)} to {fmt(hi)}"


def tile(label, value, sub="", n=""):
    """A headline number with its sample size on its own line."""
    return f'<div><span>{label}</span><b>{value}</b><small>{sub}</small><em>{n}</em></div>'


def headline_html(ps, vt):
    """The three headline measures, and what population they cover."""
    h, a = ps["headline"], ps["accounting"]
    if not h["n"]:
        return '<p class="note">No final game has stat lines loaded yet. Re-run after the Tuesday refresh.</p>'
    d, b, c = h["vs_rolling3"], h["bias"], h["cov80"]
    diff = minus(f"{d['est']:+.2f}") if d["est"] is not None else "&ndash;"
    lean = minus(f"{b['est']:+.1f}")
    rw = ps["ranges"]["weeks_without"]
    no_range = f" (week {', '.join(map(str, rw))} had none)" if rw else ""
    tiles = (tile("MAE vs last-3 average",
                  f'{p2(h["model_mae_paired"]["est"])}<span class="vs"> vs </span>{p2(h["base_mae"]["est"])}',
                  f"difference {diff} pts · 95% {ci_txt(d)}",
                  f'n = {h["n_paired"]:,} identical player-weeks · {d["games"]} games')
             + tile("Mean signed error", f"{lean} pts", f"95% {ci_txt(b, p1)} · negative = projected low",
                    f'n = {h["n"]:,} player-weeks · {h["games"]} games')
             + (tile("80% range coverage", pct(c["est"]), f"95% {ci_txt(c, pct)} · target 80%",
                     f'n = {h["n_range"]:,} with a range{no_range}') if h["n_range"] else ""))
    k = len(vt["versions_in_headline"]) if vt else 1
    return f"""<div class="figures three">{tiles}</div>
<p class="scope">These measures include only projected players who recorded a stat line: {a["scored"]:,} of the
{a["published"]:,} projections published for final games in {_span(ps["weeks"])}. The other {a["no_stat_line"]:,}
are accounted for below and are never scored as zero. The MAE comparison also needs three earlier games for the
baseline, so it uses {h["n_paired"]:,} of the {h["n"]:,}.{f" The headline pools {k} served model versions." if k > 1 else ""}
Intervals are 95%, resampling whole games.</p>"""


def appendix_html(ps, imp, checks, fs):
    if not ps["n"]:
        return ""
    dist = ps.get("distribution") or {}
    within = lambda k: f'{pct(dist[f"model_within{k}"])} vs {pct(dist[f"rolling3_within{k}"])}'
    dist_cap = (f"Within 3 points: model {within(3)} for the last-3 average; within 5: {within(5)}; within 10: "
                f"{within(10)}. n = {dist['n']:,} identical player-weeks.") if dist else ""
    seg_n = {x["label"]: x for g in ps.get("segments", []) for x in g["rows"]}
    exp_n = sum(seg_n[k]["n"] for k, *_ in wr.EXPERIENCE if k in seg_n)
    team = (f'<h3>Your fantasy team <span class="muted">(illustration, not evidence)</span></h3>{team_html(fs)}'
            if fs else "")
    return f"""<div class="pair">
<figure>{fig_mae_by_week(ps)}
<figcaption><b>MAE by week</b>, model and last-3 average on the same player-weeks; bars are 95%
intervals. Hollow marks a week still being scored.</figcaption></figure>
<figure>{fig_distribution(ps)}
<figcaption><b>How big the misses are.</b> Projected minus actual in 3-point bins; the ends collect everything
beyond &plusmn;27. {dist_cap}</figcaption></figure></div>
<figure>{fig_segments(ps)}
<figcaption><b>Where the error sits.</b> Dot = the group's value, line = its 95% interval; dashed line = all
players, solid line = target (0 bias, 0 difference, 80% coverage). MAE grows with how many points a group scores,
so compare a group with the dashed line. Experience is NFL seasons before 2026 (n = {exp_n:,} of {ps["n"]:,}
matched).</figcaption></figure>
<div class="pair">{ranges_html(ps)}
<figure>{fig_calibration(ps)}
<figcaption><b>Do projections mean what they say?</b> Players grouped by projection; on the diagonal a group
averaged exactly its projection, above it they scored more.</figcaption></figure></div>
<h3>Fuller splits</h3>
{splits_html(ps)}
<p class="note">Share = the split's part of all absolute error. Snap shares come from the snap table, injury
status from the game-status designation; both are blank where the tables have no row.</p>
{importance_html(imp)}
<h3>Checks the generator ran on these numbers</h3>
{checks_html(checks, imp)}
{team}
<div class="screen-only"><h3>Every player <span class="muted">(on screen only)</span></h3><figure>{fig_scatter(ps)}
<figcaption>Projected (across) against actual (up). Filled dots landed inside the published range, hollow dots
missed it (about one in five should).</figcaption></figure></div>"""


def render(season, week, ps, fs, partial, commits, season_view=False, ts=None, imp=None, vt=None, checks=None):
    concl = "".join(f"<p>{esc(minus_nums(x))}</p>" for x in wr.conclusion(ps))
    span = _span(ps["weeks"])
    title = f"{season} season to date: {span}" if season_view else f"Week {week}: live test"
    froze = [c for c in (checks or []) if c["check"].startswith("Every projection was committed before")]
    frozen = ("every projection committed to git before its game's kickoff" if froze and froze[0]["ok"]
              else "pre-kickoff check failed: see the checks table" if froze else "pre-kickoff check not run")
    partial_html = (f'<div class="partial"><b>{"Still being scored." if season_view else "Provisional."}</b> '
                    f"{esc(partial)}</div>" if partial else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{"Season" if season_view else "Week " + str(week)} Review</title><style>{CSS}</style></head>
<body><main>
<p class="kicker">Live test &middot; served PPR projections &middot; {season}</p>
<h1>{title}</h1>
<div class="byline">{datetime.now():%A %B %d, %H:%M} &middot; {frozen} &middot; {ps["games_final"]} games</div>
{partial_html}
<div class="take">{concl}</div>
{headline_html(ps, vt)}
<section><h2>Every published projection</h2>{accounting_html(ps)}</section>
{f'<section><h2>By model version</h2>{versions_html(vt, ps, ts)}</section>' if vt else ""}
<section><h2>Where it falls short</h2>{failure_html(ps)}</section>
<section><h2>What the model relies on <span class="muted">(2025 held out)</span></h2>{importance_summary_html(imp)}</section>
<section class="appendix"><h2>Appendix</h2>{appendix_html(ps, imp, checks or [], fs)}</section>
<footer>Population: QB/RB/WR/TE projections with a stat line in a final game; error = projected &minus; actual on
raw full-PPR points. Baseline: each player's mean over the previous three games with a stat line (crossing
seasons). Intervals: 95% percentile, 2,000 resamples of whole games, seed 42. Private report, never published.
Snapshots {esc(", ".join(commits))}.</footer>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--team", type=int, default=int(os.environ.get("ESPN_TEAM_ID", 1)))
    ap.add_argument("--all-weeks", action="store_true",
                    help="pool every complete 2026 week instead of one week")
    ap.add_argument("--no-fantasy", action="store_true")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    with sqlite3.connect(DB_PATH) as con:
        games = kickoff_times(args.season, con)
        week = args.week or default_week(games)
        weeks = list(range(1, week + 1))
        print(f"Week {week}: collecting published player rows...")
        rows, summaries = wr.collect(con, args.season, weeks, games, now)

    ts = wr.test_set(rows, games, weeks if args.all_weeks else [week])
    complete = [w["week"] for w in ts if w["complete"]]
    # Season view: headline numbers on complete weeks only; a week still
    # being scored is shown, labelled, and kept out of them.
    scope = (complete or weeks) if args.all_weeks else None
    ps = wr.player_section(rows, week, summaries, weeks=scope)
    vt = wr.version_table(rows, ts)
    checks = wr.audit(rows, ps, ts if args.all_weeks else None)
    fs = None if args.no_fantasy else fantasy_season(rows, ps["weeks"], args.team)
    imp = json.loads(IMPORTANCE.read_text()) if IMPORTANCE.exists() else None

    pending = [w for w in ts if not w["complete"]]
    if args.all_weeks:
        partial = " ".join(
            f'Week {w["week"]} has results for {w["games_scored"]} of {w["games"]} games ({w["scored"]} players): '
            "it appears in the version table and the appendix chart, in no headline number." for w in pending)
    else:
        partial = (f'Results for {pending[0]["games_scored"]} of {pending[0]["games"]} games; every number here '
                   "will change once the week is complete." if pending else "")

    commits = sorted(set(rows.loc[rows["week"].isin(weeks if args.all_weeks else ps["weeks"]),
                                  "published_commit"].dropna()))
    page = render(args.season, week, ps, fs, partial, commits, args.all_weeks, ts=ts, imp=imp, vt=vt,
                  checks=checks)

    out = REPORT_DIR / (f"review_{args.season}_season.html" if args.all_weeks
                        else f"review_{args.season}_wk{week}.html")
    docs = (PROJECT_ROOT / "docs").resolve()
    if docs == out.resolve().parent or docs in out.resolve().parents:
        raise SystemExit(f"refusing to write a private report inside the published site: {out}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({len(page):,} bytes)")
    for line in wr.conclusion(ps):
        print("  " + line)
    for c in checks:
        print(f"  [{'pass' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")


if __name__ == "__main__":
    main()
