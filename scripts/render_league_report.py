"""Render the league report as one printable page.

Reads the JSON that `league_report.py` writes and emits a self-contained HTML
file beside it -- no fetch, no CDN, no external font, so it prints and it
survives being emailed to yourself. Output stays under
data/espn_private/reports/, which the gitignore covers by location.

No charts. The page is a KPI row and dense tables, which is what this data is:
three headline numbers and five ranked lists. A bar chart of nine starters
would be a table with worse alignment.

Colour is doing three jobs and each one is also spelled out in text, never
carried by hue alone -- polarity on the trade columns (blue buy / red sell,
validated pair), status on an injury or a bye, and nothing else.

Usage:
    python scripts/render_league_report.py
    python scripts/render_league_report.py --report path/to/report.json
"""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT

REPORT_DIR = ESPN_PRIVATE_DIR / "reports"

# Statuses worth a chip on the page. The text is the signal; the colour only
# agrees with it.
STATUS_CHIPS = {
    "OUT": ("critical", "OUT"),
    "INJURY_RESERVE": ("critical", "IR"),
    "SUSPENSION": ("critical", "SUS"),
    "DOUBTFUL": ("critical", "D"),
    "QUESTIONABLE": ("warning", "Q"),
    "DAY_TO_DAY": ("warning", "DTD"),
}

CSS = """
:root {
  --surface: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --hairline: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --buy: #2a78d6; --sell: #e34948;
  --critical: #d03b3b; --warning: #fab219;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface: #1a1a19; --plane: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --hairline: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --buy: #3987e5; --sell: #e66767;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 18px 22px;
  background: var(--plane); color: var(--ink);
  font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 17px; margin: 0; font-weight: 650; }
h2 {
  font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--ink-muted); margin: 14px 0 5px; font-weight: 600;
}
.meta { color: var(--ink-2); font-size: 11px; margin: 3px 0 0; }
.sub { color: var(--ink-2); font-weight: 400; }
.kpis { display: flex; gap: 8px; margin-top: 12px; }
.tile {
  flex: 1; background: var(--surface); border: 1px solid var(--ring);
  border-radius: 6px; padding: 8px 10px;
}
.tile .label {
  display: block; font-size: 10px; color: var(--ink-muted);
  text-transform: uppercase; letter-spacing: .06em;
}
.tile .value { display: block; font-size: 22px; font-weight: 620; margin-top: 2px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
.with { white-space: nowrap; }
.note { font-size: 9.5px; padding-top: 0; }
tr.note-row td { border-bottom: 1px solid var(--hairline); }
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left; font-size: 9.5px; font-weight: 600; color: var(--ink-muted);
  text-transform: uppercase; letter-spacing: .05em;
  border-bottom: 1px solid var(--hairline); padding: 2px 4px;
}
td { padding: 2.5px 4px; border-bottom: 1px solid var(--hairline); }
tr:last-child td { border-bottom: none; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.dim { color: var(--ink-2); }
.muted { color: var(--ink-muted); }
.chip {
  font-size: 9px; font-weight: 650; letter-spacing: .04em;
  padding: 0 3px; border-radius: 3px; border: 1px solid currentColor;
}
.chip.critical { color: var(--critical); }
/* Warning sits below 3:1 on the light surface by design, so the LABEL keeps
   readable ink and the hue rides the ring -- the colour agrees with the text
   rather than carrying it. */
.chip.warning  { color: var(--ink-2); border-color: var(--warning);
                 background: rgba(250,178,25,.16); }
.chip.espn, .chip.bye { color: var(--ink-muted); }
.chip.action { color: var(--buy); border-color: var(--buy); }
.chip.rookie { color: var(--ink-2); border-color: var(--ink-muted); }
.buy { color: var(--buy); }
.sell { color: var(--sell); }
.pole { font-weight: 650; }
footer {
  margin-top: 14px; border-top: 1px solid var(--hairline); padding-top: 7px;
  color: var(--ink-2); font-size: 10px;
}
footer li { margin-bottom: 2px; }
footer ul { margin: 0; padding-left: 14px; }
@page { size: letter portrait; margin: 11mm; }
@media print {
  body { background: #fff; padding: 0; font-size: 9.6px; }
  td { padding: 1.6px 4px; }
  h2 { margin: 9px 0 4px; }
  .tile { background: #fff; }
  h1 { font-size: 15px; }
  .tile .value { font-size: 19px; }
  section, table, tr { break-inside: avoid; }
}
"""


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def num(v, dash="--") -> str:
    return dash if v is None else f"{float(v):.1f}"


def status_chip(status) -> str:
    chip = STATUS_CHIPS.get(str(status or ""))
    return f' <span class="chip {chip[0]}">{chip[1]}</span>' if chip else ""


def source_chip(source) -> str:
    """Say out loud when a number came from ESPN rather than from here."""
    return ' <span class="chip espn">ESPN</span>' if source == "espn" else ""


def _short(name, width=20) -> str:
    """League team names are jokes, and some of them are long jokes."""
    name = "" if name is None else str(name)
    return name if len(name) <= width else name[:width - 1].rstrip() + "\u2026"


def action_chip(action) -> str:
    """START/SIT: this row disagrees with the ESPN lineup."""
    return (f' <span class="chip action">{esc(action)}</span>' if action
            else "")


def rookie_chip(p) -> str:
    """No NFL history at all -- the row the season model is measured to run
    low on."""
    return (' <span class="chip rookie">R</span>' if p.get("first_year")
            else "")


def _player_cell(p) -> str:
    return (f'{esc(p.get("name"))}{status_chip(p.get("injury_status"))}'
            f'{rookie_chip(p)}{action_chip(p.get("action"))}'
            f'{source_chip(p.get("points_source"))}')


def _rows(rows: list) -> str:
    return "\n".join(rows) if rows else (
        '<tr><td class="muted">nothing to report</td></tr>')


def opponent_table(starters, opponent) -> str:
    """Their lineup, not just their total: the total says how big the hill is,
    the rows say where it is."""
    rows = []
    for p in starters:
        rows.append(
            f'<tr><td class="muted">{esc(p["slot"])}</td>'
            f'<td>{_player_cell(p)}</td>'
            f'<td class="dim">{esc(p.get("nfl_team"))}</td>'
            f'<td class="num">{num(p.get("points"))}</td></tr>')
    return ('<table><tr><th>Slot</th><th>Player</th><th>Team</th>'
            f'<th class="num">Proj</th></tr>{_rows(rows)}</table>')


def streaming_table(streamers, limit=3) -> str:
    rows = []
    for st in streamers[:limit]:
        owned = st["available"].get("percent_owned")
        rows.append(
            f'<tr><td class="muted">{esc(st["position"])}</td>'
            f'<td>{esc(st["available"]["name"])}</td>'
            f'<td class="dim">for {esc(st["current"]["name"])}</td>'
            f'<td class="num">{num(st["available"]["points"])}</td>'
            f'<td class="num dim">+{num(st["gain"])}</td>'
            f'<td class="num muted">{num(owned)}%</td></tr>')
    return ('<table><tr><th>Slot</th><th>Available</th><th>Instead of</th>'
            '<th class="num">Proj</th><th class="num">Gain</th>'
            f'<th class="num">Own</th></tr>{_rows(rows)}</table>')


def win_text(matchup: dict) -> str:
    """No measured error on either side means no probability, not 50%."""
    p = matchup.get("win_probability")
    return "--" if p is None else f"{round(float(p) * 100):d}%"


def lineup_verdict(report: dict) -> str:
    """This lineup is built from projections, not read from ESPN. Say so, and
    say whether ESPN already agrees."""
    changes = report.get("lineup_changes") or []
    if not changes:
        return "matches the lineup you have set in ESPN"
    return (f"{len(changes)} change{'s' if len(changes) > 1 else ''} from the "
            "lineup you have set in ESPN")


def band_cell(p) -> str:
    """The season floor/ceiling as a per-week pace.

    Its WIDTH is the point: it is how the page says which numbers to distrust,
    and a first-year player's band runs three times a veteran's.
    """
    band = p.get("pace_band")
    if not band or band[0] is None:
        return '<td class="num muted">--</td>'
    return f'<td class="num dim">{num(band[0])}&ndash;{num(band[1])}</td>'


def starters_table(starters) -> str:
    rows = []
    for p in starters:
        opponent = p.get("opponent")
        rows.append(
            f'<tr><td class="muted">{esc(p["slot"])}</td>'
            f'<td>{_player_cell(p)}</td>'
            f'<td class="dim">{esc(p.get("nfl_team"))}'
            f'{" vs " + esc(opponent) if opponent else ""}</td>'
            f'<td class="num">{num(p.get("points"))}</td>'
            f'{band_cell(p)}</tr>')
    return ('<table><tr><th>Slot</th><th>Player</th><th>Matchup</th>'
            '<th class="num">Proj</th><th class="num">Pace band</th></tr>'
            f'{_rows(rows)}</table>')


def bench_table(bench, limit=6) -> str:
    rows = []
    for p in sorted(bench, key=lambda b: -(b.get("points") or -1))[:limit]:
        reason = p.get("unavailable_reason")
        note = (f' <span class="chip bye">{esc(reason)}</span>'
                if reason == "BYE" else "")
        unpriced = (' <span class="muted">no projection</span>'
                    if p.get("points") is None else "")
        rows.append(
            f'<tr><td class="muted">{esc(p.get("position"))}</td>'
            f'<td>{_player_cell(p)}{note}{unpriced}</td>'
            f'<td class="num">{num(p.get("points"))}</td></tr>')
    return ('<table><tr><th>Pos</th><th>Bench</th>'
            f'<th class="num">Proj</th></tr>{_rows(rows)}</table>')


def tough_table(calls, limit=5) -> str:
    rows = []
    for c in calls[:limit]:
        start, bench = c["starting"], c["benched"]
        rows.append(
            f'<tr><td class="muted">{esc(c["slot"])}</td>'
            f'<td>{_player_cell(start)} <span class="muted">over</span> '
            f'{_player_cell(bench)}</td>'
            f'<td class="num">{num(start.get("points"))} / '
            f'{num(bench.get("points"))}</td>'
            f'<td class="num dim">{num(c["gap"])}</td></tr>')
    return ('<table><tr><th>Slot</th><th>Closer than the projection can call</th>'
            f'<th class="num">Proj</th><th class="num">Gap</th></tr>'
            f'{_rows(rows)}</table>')


def waiver_table(waivers, limit=6) -> str:
    rows = []
    for w in waivers[:limit]:
        owned = w.get("percent_owned")
        rows.append(
            f'<tr><td class="muted">{esc(w["position"])}</td>'
            f'<td>{_player_cell(w)}</td>'
            f'<td class="dim">for {esc(w["instead_of"]["name"])}</td>'
            f'<td class="num">{num(w.get("points"))}</td>'
            f'<td class="num dim">+{num(w.get("over"))}</td>'
            f'<td class="num muted">{num(owned, dash="--")}%</td></tr>')
    return ('<table><tr><th>Pos</th><th>Free agent</th><th>Instead of</th>'
            '<th class="num">Proj</th><th class="num">Over</th>'
            f'<th class="num">Own</th></tr>{_rows(rows)}</table>')


def _lineup_note(change: dict) -> str:
    """What the swap actually does to the lineup, which is the point of it."""
    moves = ([f'{esc(c["name"])} starts at {esc(c["slot"])}'
              for c in change.get("in", [])]
             + [f'{esc(c["name"])} '
                + ("leaves the roster" if c.get("reason") == "traded"
                   else "to the bench")
                for c in change.get("out", [])]
             + [f'{esc(c["name"])} {esc(c["from"])} &rarr; {esc(c["to"])}'
                for c in change.get("moved", [])])
    return "; ".join(moves)


def proposal_table(trades, limit=4) -> str:
    """Concrete swaps: who with, who leaves, who arrives, what it is worth."""
    weeks = trades.get("horizon_weeks")
    rows = []
    for t in trades.get("proposals", [])[:limit]:
        give, get = t["give"], t["get"]
        rows.append(
            f'<tr><td class="with">{esc(_short(t["with"], 26))}</td>'
            f'<td class="sell">{_player_cell(give)} '
            f'<span class="muted">{esc(give["position"])} '
            f'{num(give.get("model_ppg"))}</span></td>'
            f'<td class="buy">{_player_cell(get)} '
            f'<span class="muted">{esc(get["position"])} '
            f'{num(get.get("model_ppg"))}</span></td>'
            f'<td class="num">+{num(t["our_gain_per_week"])}</td>'
            f'<td class="num">+{num(t["our_gain_over_horizon"])}</td>'
            f'<td class="num dim">+{num(t["their_gain_per_week_espn"])}</td></tr>'
            f'<tr><td></td><td colspan="5" class="muted note">'
            f'{_lineup_note(t["lineup_change"])}</td></tr>')
    return ('<table><tr><th>With</th><th>You give</th><th>You get</th>'
            f'<th class="num">You /wk</th><th class="num">Over {esc(weeks)}</th>'
            f'<th class="num">Them /wk</th></tr>{_rows(rows)}</table>')


def render(report: dict) -> str:
    m = report["matchup"]
    edge = m.get("edge") or 0.0
    coverage = report.get("coverage", {}).get("roster", {})
    caveats = "".join(f"<li>{esc(c)}</li>" for c in report.get("caveats", []))
    unmatched = coverage.get("by_reason", {})
    unmatched_text = ", ".join(f"{reason} ({n})"
                               for reason, n in unmatched.items())

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{esc(report['team']['name'])} -- week {esc(report['week'])}</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>{esc(report['team']['name'])}
    <span class="sub">week {esc(report['week'])} vs {esc(m.get('opponent'))}</span></h1>
  <p class="meta">{esc(report['season'])} season &middot;
    projections {esc(report.get('projection_mode'))} &middot;
    generated {esc(report.get('generated_at'))}</p>
</header>

<section class="kpis">
  <div class="tile"><span class="label">Projected</span>
    <span class="value">{num(m.get('projected_total'))}</span></div>
  <div class="tile"><span class="label">{esc(m.get('opponent'))}</span>
    <span class="value">{num(m.get('opponent_projected_total'))}</span></div>
  <div class="tile"><span class="label">Win probability</span>
    <span class="value {'buy' if (m.get('win_probability') or 0.5) >= 0.5 else 'sell'}">
      {win_text(m)}</span>
    <span class="label">{'+' if edge >= 0 else ''}{num(edge)} projected,
      &plusmn;{num(m.get('spread'))} spread</span></div>
</section>

<div class="cols">
  <div>
    <section><h2>Recommended lineup &mdash; {lineup_verdict(report)}</h2>
      {starters_table(report['starters'])}</section>
    <section><h2>Bench</h2>{bench_table(report['bench'])}</section>
  </div>
  <div>
    <section><h2>{esc(m.get('opponent'))} &mdash; their projected lineup</h2>
      {opponent_table(report.get('opponent_starters', []), m.get('opponent'))}</section>
    <section><h2>Tough calls</h2>{tough_table(report['tough_calls'], limit=4)}</section>
  </div>
</div>

<div class="cols">
  <section><h2>Waiver wire</h2>{waiver_table(report['waivers'], limit=4)}</section>
  <section><h2>Streaming &mdash; kicker and defence</h2>
    {streaming_table(report.get('streamers', []))}</section>
</div>

<section><h2>Proposed trades &mdash; both lineups improve, each on its own
  numbers</h2>{proposal_table(report['trades'], limit=5)}</section>

<footer>
  <ul>{caveats}
    <li>Roster coverage: {esc(coverage.get('matched'))} of
      {esc(coverage.get('players'))} players matched to a projection{
      f" -- unmatched: {esc(unmatched_text)}" if unmatched_text else ""}.</li>
  </ul>
</footer>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPORT_DIR / "latest.json",
                    help="report JSON to render (default: the latest)")
    args = ap.parse_args()

    if not args.report.exists():
        print(f"no report at {args.report} -- run scripts/league_report.py first")
        return 1

    report = json.loads(args.report.read_text())
    page = render(report)
    out = REPORT_DIR / f"report_{report['season']}_wk{report['week']}.html"
    out.write_text(page)
    (REPORT_DIR / "latest.html").write_text(page)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({len(page):,} bytes)")
    print(f"      {(REPORT_DIR / 'latest.html').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
