"""Shared pieces of the review pages: number formatting, small inline-SVG
chart primitives, the editorial theme, and the charts both the detailed
season review and the short brief draw. No scripts, no external requests:
every chart is plain SVG in the page."""
from __future__ import annotations

import html
import math
import re

def esc(v):
    return html.escape("" if v is None else str(v))


def f(v, d=2, dash="&ndash;"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return dash
    return _no_neg_zero(f"{v:.{d}f}")


def _no_neg_zero(text):
    """'-0.0' -> '0.0': a value that rounds to zero has no sign."""
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def pct(v, d=0, dash="&ndash;"):
    return dash if v is None else _no_neg_zero(f"{v * 100:.{d}f}") + "%"


def minus(text):
    """A true minus sign for negative numbers in running text and tables."""
    return str(text).replace("-", "\u2212")


def minus_nums(text):
    """Hyphens that are minus signs (before a digit, not inside a word or range)."""
    return re.sub(r"(?<![\w\u2013])-(?=\d)", "\u2212", text)


def signed(v):
    return ("+" if v > 0 else "") + minus(f"{v:g}")


class Scale:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>')


def yaxis(sy, x0, x1, vals, fmt=lambda v: f"{v:g}"):
    out = []
    for v in vals:
        y = sy(v)
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    return "".join(out)


def axis_titles(w, h, l, r, t, b, x=None, y=None, y_x=14):
    """Axis titles: x centred under the plot, y rotated beside it."""
    out = []
    if x:
        out.append(f'<text class="axislabel" x="{(l + w - r) / 2:.0f}" y="{h - 6}" '
                   f'text-anchor="middle">{esc(x)}</text>')
    if y:
        out.append(f'<text class="axislabel" transform="translate({y_x},{(t + h - b) / 2:.0f}) rotate(-90)" '
                   f'text-anchor="middle">{esc(y)}</text>')
    return "".join(out)


def legend(items, x, y):
    """Swatches drawn as the mark they stand for: dot, hollow dot, bar, diamond,
    line or band."""
    out, cx = [], x
    for cls, text, kind in items:
        if kind == "line":
            out.append(f'<line class="{cls}-stroke" x1="{cx}" x2="{cx + 18}" y1="{y}" y2="{y}" stroke-width="2"/>')
        elif kind == "band":
            out.append(f'<rect class="band" x="{cx}" y="{y - 5}" width="18" height="10" rx="2"/>')
        elif kind == "bar":
            out.append(f'<rect class="{cls}-fill" x="{cx + 7.5}" y="{y - 7}" width="3" height="14" rx="1"/>')
        elif kind == "diamond":
            out.append(f'<path class="actual" d="M{cx + 9},{y - 6} l6,6 l-6,6 l-6,-6 z"/>')
        elif kind == "diamond2":
            out.append(f'<path class="espn-mark" d="M{cx + 9},{y - 6} l6,6 l-6,6 l-6,-6 z"/>')
        elif kind == "range":
            out.append(f'<rect class="range" x="{cx}" y="{y - 5}" width="22" height="10" rx="5"/>')
        elif kind == "hollow":
            out.append(f'<circle class="hollow" cx="{cx + 9}" cy="{y}" r="4.5"/>')
        elif kind == "block":
            out.append(f'<rect class="{cls}-fill" x="{cx + 3}" y="{y - 6}" width="12" height="12" rx="2"/>')
        else:
            out.append(f'<circle class="{cls}-fill ring" cx="{cx + 9}" cy="{y}" r="4.5"/>')
        out.append(f'<text class="leg" x="{cx + 24}" y="{y + 4}">{esc(text)}</text>')
        cx += 34 + 8.4 * len(text)
    return "".join(out)


def fig_calibration(ps):
    """Average actual against average projection, by projected band."""
    cal = ps.get("calibration") or []
    if not cal:
        return ""
    w, h, l, r, t, b = 640, 300, 66, 24, 44, 50
    top = max(max(c["actual"]["hi"] or c["actual"]["est"] for c in cal), max(c["projected"] for c in cal))
    top = math.ceil(top / 5) * 5
    sx, sy = Scale(0, top, l, w - r), Scale(0, top, h - b, t)
    body = [yaxis(sy, l, w - r, ticks(0, top, 5))]
    for v in ticks(0, top, 5):
        body.append(f'<text class="tick" x="{sx(v):.1f}" y="{h - b + 18}" text-anchor="middle">{v:g}</text>')
    body.append(f'<line class="diag" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(top):.1f}" y2="{sy(top):.1f}"/>')
    pts = [(sx(c["projected"]), sy(c["actual"]["est"])) for c in cal]
    body.append('<polyline class="model-stroke" fill="none" stroke-width="2" points="'
                + " ".join(f"{a:.1f},{c:.1f}" for a, c in pts) + '"/>')
    for c, (x, y) in zip(cal, pts):
        a = c["actual"]
        if a["lo"] is not None:
            body.append(f'<line class="model-stroke" stroke-width="2" x1="{x:.1f}" x2="{x:.1f}" '
                        f'y1="{sy(a["lo"]):.1f}" y2="{sy(a["hi"]):.1f}"/>')
        band = f'{c["lo"]:g}–{c["hi"]:g}' if c["hi"] else f'{c["lo"]:g}+'
        body.append(f'<circle class="model-fill ring" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>Projected {band}: '
                    f'average projection {c["projected"]:.1f}, average actual {a["est"]:.1f} '
                    f'(95% {f(a["lo"], 1)}–{f(a["hi"], 1)}), n={c["n"]}</title></circle>'
                    f'<text class="dlabel" x="{x + 9:.1f}" y="{y + 18:.1f}">n={c["n"]}</text>')
    body.append(axis_titles(w, h, l, r, t, b, "Average projection (pts)", "Average actual (pts)"))
    body.append(legend([("model", "Average actual, by projection band", "dot"), ("diag", "Exact", "line")], l, 16))
    return svg(w, h, "".join(body), "Calibration: average actual by projection band")


def fig_team(fs):
    """Your weekly score as set, with the model's lineup, against the hindsight best."""
    wk = fs["weeks"]
    w, h, l, r, t, b = 640, 318, 66, 20, 44, 72
    vals = [x[k] for x in wk for k in ("actual", "model_lineup", "hindsight")]
    lo, hi = math.floor(min(vals) / 10) * 10 - 10, math.ceil(max(vals) / 10) * 10 + 5
    sx = Scale(wk[0]["week"] - 0.6, wk[-1]["week"] + 0.6, l, w - r)
    sy = Scale(lo, hi, h - b, t)
    body = [yaxis(sy, l, w - r, ticks(lo, hi, 5))]
    for x in wk:
        cx = sx(x["week"])
        a, m, hb = sy(x["actual"]), sy(x["model_lineup"]), sy(x["hindsight"])
        body.append(f'<line class="connector" x1="{cx:.1f}" x2="{cx:.1f}" y1="{a:.1f}" y2="{m:.1f}"/>')
        body.append(f'<line class="bench-stroke" x1="{cx - 11:.1f}" x2="{cx + 11:.1f}" y1="{hb:.1f}" y2="{hb:.1f}">'
                    f'<title>Week {x["week"]} hindsight best {x["hindsight"]:.1f}</title></line>')
        pend = x["pending"]
        as_set = 'class="hollow" style="stroke:var(--base)"' if pend else 'class="base-fill ring"'
        body.append(f'<circle {as_set} cx="{cx:.1f}" cy="{a:.1f}" r="5.5">'
                    f'<title>Week {x["week"]} as set: {x["actual"]:.1f}</title></circle>')
        body.append(f'<circle class="{"hollow" if pend else "model-fill ring"}" cx="{cx:.1f}" cy="{m:.1f}" r="5.5">'
                    f'<title>Week {x["week"]} model lineup: {x["model_lineup"]:.1f} '
                    f'({x["changes"]} change{"s" if x["changes"] != 1 else ""} to your starters)</title></circle>')
        d = x["model_lineup"] - x["actual"]
        label = minus(f"{d:+.1f}") if abs(d) >= 0.05 else ("same lineup" if not x["changes"] else "+0.0")
        body.append(f'<text class="dlabel" x="{cx + 10:.1f}" y="{(a + m) / 2 + 5:.1f}">{label}</text>')
        res = "pending" if pend else (x["result"] if x["result"] == x["model_result"]
                                      else f'{x["result"]}\u2192{x["model_result"]}')
        body.append(f'<text class="tick" x="{cx:.1f}" y="{h - b + 18}" text-anchor="middle">Wk {x["week"]}</text>'
                    f'<text class="tick" x="{cx:.1f}" y="{h - b + 34}" text-anchor="middle">{res}</text>')
    body.append(legend([("base", "As set", "dot"), ("model", "Model's lineup", "dot"),
                        ("bench", "Hindsight best", "line")], l, 16))
    body.append(axis_titles(w, h, l, r, t, b, "Week (result \u2192 with model's lineup)", "Fantasy points"))
    return svg(w, h, "".join(body), "Your weekly score: as set, model lineup and hindsight best")


CSS = """
:root{--paper:#faf7f0;--paper-2:#f2ede1;--ink:#1f1d1a;--ink-2:#55504a;--muted:#8a847a;
--rule:#ddd5c4;--grid:#e8e1d2;--model:#9a3b32;--base:#2f6db3;--band:rgba(31,29,26,.07);
--warn-bg:#f6e7c8;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#1e1c19;--paper-2:#26231f;
--ink:#ece6da;--ink-2:#bdb5a6;--muted:#8d877c;--rule:#3a3631;--grid:#2e2b27;--model:#d8665a;
--base:#5b8fd6;--band:rgba(236,230,218,.08);--warn-bg:#3b3222;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.5 Charter,"Iowan Old Style","Palatino Linotype",Georgia,serif;-webkit-font-smoothing:antialiased}
main{max-width:920px;margin:0 auto;padding:36px 22px 56px}
.take{font-size:15.5px}
.kicker{letter-spacing:.12em;text-transform:uppercase;color:var(--model);font-size:14px;margin:0}
h1{font-weight:400;font-size:34px;line-height:1.15;margin:4px 0 6px;letter-spacing:-.01em}
.byline{font-size:12.5px;color:var(--muted);border-top:1px solid var(--ink);
border-bottom:1px solid var(--rule);padding:6px 0;margin-bottom:18px;
}
h2{font-weight:600;font-size:19px;
margin:26px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:3px}
h2 .num{color:var(--model);margin-right:10px}
.take{background:var(--paper-2);border-left:3px solid var(--model);padding:10px 18px;margin:0 0 6px}
.take ul{margin:0;padding-left:18px}.take li{margin:3px 0}
.partial{background:var(--warn-bg);padding:7px 12px;font-size:13.5px;margin:0 0 14px;
border-left:3px solid var(--muted)}
.figures{display:flex;flex-wrap:wrap;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);
margin:8px 0 12px}
.figures div{flex:1 1 140px;padding:8px 12px;border-right:1px solid var(--rule)}
.figures div:last-child{border-right:none}
.figures b{display:block;font-size:24px;font-weight:500;
font-variant-numeric:lining-nums}
.figures span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.figures em{display:block;font-style:normal;font-size:12px;font-weight:600;color:var(--ink);margin-top:2px}
.figures small{display:block;font-size:11.5px;color:var(--ink-2)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
.pair>*{min-width:0}
.pair.tight{gap:22px}
.quad{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media screen and (max-width:760px){.pair{grid-template-columns:1fr}.quad{grid-template-columns:1fr 1fr}}
figure{margin:6px 0 10px;break-inside:avoid}
figure svg{width:100%;height:auto;display:block;font-family:inherit}
figcaption{font-size:13px;color:var(--ink-2);margin-top:4px;line-height:1.4}
figcaption b{color:var(--ink);font-weight:600}
.note{font-size:14px;color:var(--ink-2);margin:6px 0}
.muted{color:var(--muted)}
.grid{stroke:var(--grid);stroke-width:1}
.tick{fill:var(--muted);font-size:15px}
.leg{fill:var(--ink-2);font-size:15px}
.dlabel{fill:var(--ink-2);font-size:15px}
.panel{fill:var(--ink);font-size:17px;font-weight:600}
.axislabel{fill:var(--muted);font-size:15px}
.model-stroke{stroke:var(--model);stroke-linecap:round;stroke-linejoin:round}
.model-fill{fill:var(--model)}
.base-stroke{stroke:var(--base);stroke-linecap:round;stroke-linejoin:round}
.base-fill{fill:var(--base)}
.actual,.actual-fill{fill:var(--ink);stroke:var(--paper);stroke-width:1.5}
.ring{stroke:var(--paper);stroke-width:2}
.hollow{fill:var(--paper);stroke:var(--model);stroke-width:1.8}
.bench,.bench-stroke{stroke:var(--muted);stroke-width:2}.bench-fill{fill:var(--muted)}
.diag,.diag-stroke{stroke:var(--muted);stroke-width:1}
.zero{stroke:var(--ink-2);stroke-width:1}
.target{stroke:var(--ink-2);stroke-width:1}
.connector{stroke:var(--rule);stroke-width:2}
.band{fill:var(--band)}
h3{font-weight:600;font-size:16px;margin:16px 0 6px}
h2 .muted{font-weight:400}
.tw{overflow-x:auto;margin:4px 0 6px}
table.t{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums lining-nums}
table.t th{font-weight:600;font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;
text-align:left;padding:4px 6px;border-bottom:1px solid var(--ink);vertical-align:bottom}
table.t td{padding:3px 6px;border-bottom:1px solid var(--rule);vertical-align:top}
table.t .n,table.t .nw{text-align:right;white-space:nowrap}table.t .nw{text-align:left}
.figures span i{text-transform:none;font-style:normal;letter-spacing:0}
table.t tr.total td{border-top:1px solid var(--ink);font-weight:600}
table.t tr.sep td{font-size:12px;font-weight:600;color:var(--ink-2);padding-top:9px;border-bottom:none;
letter-spacing:.02em}
table.t tr.dim td{color:var(--muted)}
table.t tr.flag td:first-child,table.t tr.flag td b{color:var(--model);font-weight:700}
.figures.three div{flex:1 1 200px}
.figures .vs{font-size:15px;color:var(--muted)}
.scope{font-size:13.5px;color:var(--ink-2);margin:0 0 6px}
table.t tr.sub td:first-child{padding-left:20px}
table.t tr.pool td{color:var(--ink-2);font-style:italic}
table.t td.hot{color:var(--model);font-weight:700}
table.t td.hot .muted{font-weight:400}
table.acct td:last-child{font-size:12.5px}
.appendix h3{margin-top:20px}
table.t .ci{font-size:11px}
footer{margin-top:18px;font-size:12px;color:var(--muted);border-top:1px solid var(--rule);padding-top:6px}
@page{size:letter portrait;margin:11mm}
@media print{body{background:#fff;font-size:10.5pt;line-height:1.4}main{max-width:none;padding:0;zoom:.9}
h1{font-size:24pt}.figures b{font-size:18pt}table.t{font-size:8.5pt}table.t th{font-size:7pt}
.note,figcaption{font-size:8.5pt}footer{font-size:7.5pt;margin-top:8px}.take{font-size:9.5pt}.partial{font-size:8.5pt}
.figures small,.figures em{font-size:8pt}
figure,.figures,.pair.tight,table.t{break-inside:avoid}h2,h3{break-after:avoid}.screen-only{display:none}}
"""

