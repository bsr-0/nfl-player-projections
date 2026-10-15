"""The short brief: a handful of numbers and plain sentences for a
non-technical reader, built on exactly the rows and intervals of the detailed
review (``weekly_review.stats_for`` / ``mean_ci``), so the two never disagree.

Wording rule: every sentence is decided by an interval, as in
``weekly_review.conclusion``; where the interval cannot decide, the sentence
says so or is left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import weekly_review as wr

# Projection-gap bins for the head-to-head chart, in points.
GAP_BINS = [(0, 2), (2, 4), (4, 6), (6, np.inf)]
# Only players someone would think of starting enter the head-to-head pairs.
STARTABLE = 5.0
# Columns the brief may read from a published weekly file. Everything else,
# including the blinded forecasts the files also carry, is dropped on read.
WEEK_COLS = ["player_id", "name", "position", "team", "opponent", "home_away", "predicted_points",
             "prediction_ci80_lower", "prediction_ci80_upper"]


def weeks_beaten(ps: dict) -> dict:
    """Complete weeks in which the model's typical miss was below the last-3
    average's on the same players; 'clearly' = the week's interval is below 0."""
    done = [w for w in ps["by_week"] if not w.get("partial") and w["week"] in ps["weeks"]]
    d = [w["vs_rolling3"] for w in done if w["vs_rolling3"]["est"] is not None]
    return {"weeks": len(d), "beaten": sum(x["est"] < 0 for x in d),
            "clearly": sum(x["hi"] is not None and x["hi"] < 0 for x in d)}


def head_to_head(wk: pd.DataFrame, min_projection: float = STARTABLE) -> list[dict]:
    """Within one week and one position, every pair of scored players both
    projected at least `min_projection`: how often the one projected higher
    outscored the other, by how far apart their projections were. A tie in
    actual points counts as half; equal projections are skipped."""
    d = wk[wk["predicted_points"] >= min_projection]
    gaps, wins = [], []
    for _, g in d.groupby(["week", "position"]):
        p = g["predicted_points"].to_numpy(float)
        a = g["actual_points"].to_numpy(float)
        i, j = np.triu_indices(len(g), 1)
        gap = np.abs(p[i] - p[j])
        keep = gap > 0
        lead = np.where(p[i] > p[j], a[i] - a[j], a[j] - a[i])[keep]
        gaps.append(gap[keep])
        wins.append((lead > 0) + 0.5 * (lead == 0))
    gaps = np.concatenate(gaps) if gaps else np.array([])
    wins = np.concatenate(wins) if wins else np.array([])
    out = []
    for lo, hi in GAP_BINS:
        m = (gaps >= lo) & (gaps < hi)
        out.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "pairs": int(m.sum()),
                    "share": float(wins[m].mean()) if m.any() else None})
    return out


def summary(ps: dict, wb: dict) -> str:
    """The one sentence at the top of the brief."""
    h = ps["headline"]
    d, b, c = h["vs_rolling3"], h["bias"], h["cov80"]
    span = f"weeks {ps['weeks'][0]}–{ps['weeks'][-1]}" if len(ps["weeks"]) > 1 else f"week {ps['weeks'][0]}"
    if d["lo"] is None:
        return f"Only {h['games']} game has results so far: too early to judge."
    if wr._side(d) == "below":
        first = (f"Over {span} the model's projections beat a simple average of each player's last three games in "
                 f"{wb['beaten']} of {wb['weeks']} weeks, missing by about {abs(d['est']):.1f} points less per player")
    elif wr._side(d) == "above":
        first = (f"Over {span} a simple average of each player's last three games did better than the model's projections, "
                 f"by about {abs(d['est']):.1f} points per player")
    else:
        first = (f"Over {span} the model's projections and a simple average of each player's last three games were "
                 "about equally accurate")
    lean = {"below": f"they run a little low (players scored about {abs(b['est']):.1f} points more than projected)",
            "above": f"they run a little high (about {abs(b['est']):.1f} points)"}.get(wr._side(b), "they don't lean high or low")
    rng = {"below": "the model's ranges are a bit too tight", "above": "the model's ranges are a bit too wide"}.get(
        wr._side(c, 0.80), "the model's ranges are about right") if c["lo"] is not None else ""
    return f"{first}. {lean[0].upper() + lean[1:]}" + (f", and {rng}." if rng else ".")


def trust_guide(ps: dict, h2h: list[dict]) -> list[str]:
    """Short notes on reading next week's numbers. Each note is shown only
    when the track record's interval supports it."""
    notes = []
    panel = {x["label"]: x for x in ps["failure_panel"]}
    qb = panel.get("QB")
    if qb and wr._side(qb["bias"]) == "below":
        notes.append(f"QBs: the model's numbers have run about {abs(qb['bias']['est']):.1f} points low. "
                     "Lean toward the top of a QB's range.")
    te = panel.get("TE")
    if te and te["n_range"] and wr._side(te["cov80"], 0.80) == "below":
        notes.append(f"TEs: {1 - te['cov80']['est']:.0%} of TE scores landed outside the model's range (it should be "
                     f"about 20%), mostly above it ({te['above']['est']:.0%}). Expect bigger TE swings, "
                     "especially upside.")
    high = next((x for x in ps["failure_panel"] if x["label"].startswith("Projected")), None)
    if high and wr._side(high["bias"]) == "below":
        notes.append(f"Players the model projects for {wr.HIGH_PROJECTION:g}+ have beaten that number by about "
                     f"{abs(high['bias']['est']):.1f} points on average.")
    close = next((b for b in h2h if b["share"] is not None and b["pairs"] >= 50), None)
    sure = next((b for b in reversed(h2h) if b["share"] is not None and b["pairs"] >= 50), None)
    if close and sure and close is not sure:
        gap = f"{sure['lo']:g}+" if sure["hi"] is None else f"{sure['lo']:g}–{sure['hi']:g}"
        notes.append(f"Close calls: when two players are within {close['hi']:g} points, the higher projection "
                     f"won {close['share']:.0%} of the time, little better than a coin flip; at {gap} points "
                     f"apart it won {sure['share']:.0%}.")
    return notes


def week_projections(weekly: pd.DataFrame) -> pd.DataFrame:
    """A published week's rows, whitelisted to what the brief shows."""
    return weekly[[c for c in WEEK_COLS if c in weekly.columns]].copy()
