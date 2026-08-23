"""College -> conference resolution, era-aware.

Conference is a far better model input than raw college: college is a
200+-value categorical with a long tail (top 10 cover 23%, the bottom 88
schools contribute 1-2 picks each), while conference is ~12 values with real
competitive meaning.

**The map is keyed on (college, year), not college alone.** Realignment makes
a static map silently wrong for historical rows -- Texas A&M has been in the
SEC since 2012 but was Big 12 before it, USC/UCLA joined the Big Ten only in
2024, and the Big East's football members became the AAC in 2013. Labelling a
2010 Texas A&M pick "SEC" would inject a fact that was not true when they
played.

**The year used is `draft_season - 1`, not `draft_season`.** A player drafted
in year N played their final college season in N-1, so that is the year whose
conference membership describes their actual competition. This matters exactly
at realignment boundaries: a Texas A&M player drafted in 2012 played the 2011
season in the Big 12, and gets Big 12.

Coverage is deliberately measurable rather than assumed -- `UNMAPPED` is a
real, countable value, not a silent fallback to a plausible-looking guess. Run
this module directly to print coverage against the live draft table.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

UNMAPPED = "Other"

POWER5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}

# Canonical spellings differ between sources: draft_picks_v2 writes "Ohio St."
# while players writes "Ohio State", and Miami is ambiguous between FL and OH.
_ALIASES = {
    "ohio state": "Ohio St.", "penn state": "Penn St.", "florida state": "Florida St.",
    "michigan state": "Michigan St.", "oklahoma state": "Oklahoma St.",
    "arizona state": "Arizona St.", "oregon state": "Oregon St.",
    "washington state": "Washington St.", "kansas state": "Kansas St.",
    "iowa state": "Iowa St.", "mississippi state": "Mississippi St.",
    "boise state": "Boise St.", "san diego state": "San Diego St.",
    "fresno state": "Fresno St.", "colorado state": "Colorado St.",
    "utah state": "Utah St.", "north carolina state": "North Carolina St.",
    "nc state": "North Carolina St.", "miami": "Miami (FL)", "miami fl": "Miami (FL)",
    "miami ohio": "Miami (OH)", "miami oh": "Miami (OH)",
    "southern cal": "USC", "southern california": "USC", "ole miss": "Mississippi",
    "pitt": "Pittsburgh", "boston college": "Boston Col.", "ucf": "Central Florida",
    "byu": "BYU", "tcu": "TCU", "smu": "SMU", "utep": "Texas-El Paso",
    "uab": "Ala-Birmingham", "louisiana lafayette": "Louisiana",
    "texas christian": "TCU", "brigham young": "BYU",
}

# Conference as of ~2006, the start of the data. MOVES applies changes after.
_BASE: Dict[str, str] = {}


def _add(conf: str, *schools: str) -> None:
    for s in schools:
        _BASE[s] = conf


_add("SEC", "Alabama", "Arkansas", "Auburn", "Florida", "Georgia", "Kentucky", "LSU",
     "Mississippi", "Mississippi St.", "South Carolina", "Tennessee", "Vanderbilt")
_add("Big Ten", "Illinois", "Indiana", "Iowa", "Michigan", "Michigan St.", "Minnesota",
     "Northwestern", "Ohio St.", "Penn St.", "Purdue", "Wisconsin")
_add("Big 12", "Baylor", "Colorado", "Iowa St.", "Kansas", "Kansas St.", "Missouri",
     "Nebraska", "Oklahoma", "Oklahoma St.", "Texas", "Texas A&M", "Texas Tech")
_add("Pac-12", "Arizona", "Arizona St.", "California", "Oregon", "Oregon St.", "Stanford",
     "UCLA", "USC", "Washington", "Washington St.")
_add("ACC", "Boston Col.", "Clemson", "Duke", "Florida St.", "Georgia Tech", "Maryland",
     "Miami (FL)", "North Carolina", "North Carolina St.", "Virginia", "Virginia Tech",
     "Wake Forest")
_add("Big East", "Cincinnati", "Connecticut", "Louisville", "Pittsburgh", "Rutgers",
     "South Florida", "Syracuse", "West Virginia")
_add("Independent", "Notre Dame", "Navy", "Army")
_add("Mountain West", "Air Force", "BYU", "Colorado St.", "New Mexico", "San Diego St.",
     "TCU", "UNLV", "Utah", "Wyoming")
_add("WAC", "Boise St.", "Fresno St.", "Hawaii", "Idaho", "Louisiana Tech", "Nevada",
     "New Mexico St.", "San Jose St.", "Utah St.")
_add("C-USA", "Ala-Birmingham", "Central Florida", "East Carolina", "Houston", "Marshall",
     "Memphis", "Rice", "SMU", "Southern Miss", "Texas-El Paso", "Tulane", "Tulsa",
     "Texas-San Antonio", "Charlotte", "Florida Atlantic", "Florida International",
     "Middle Tenn. St.", "North Texas", "Old Dominion", "Western Kentucky")
_add("MAC", "Akron", "Ball St.", "Bowling Green", "Buffalo", "Central Michigan",
     "Eastern Michigan", "Kent St.", "Miami (OH)", "Northern Illinois", "Ohio", "Toledo",
     "Western Michigan", "Massachusetts")
# Only schools already playing FBS Sun Belt football in 2006. Appalachian St.,
# Coastal Carolina, Georgia Southern, Georgia St., James Madison, South Alabama
# and Texas St. were FCS until their moves below, so they are deliberately
# absent here and resolve to UNMAPPED for their FCS years -- which is correct,
# not a gap.
_add("Sun Belt", "Arkansas St.", "Louisiana", "La-Monroe", "Troy")
_add("MAC", "Temple")  # MAC football 2007-2011, then Big East/AAC (see _MOVES)
_add("Ivy", "Harvard", "Yale", "Princeton", "Pennsylvania", "Columbia", "Cornell",
     "Brown", "Dartmouth")

# (effective college season, new conference). Applied when college_year >= year.
_MOVES: Dict[str, List[Tuple[int, str]]] = {
    # Big Ten expansion
    "Nebraska": [(2011, "Big Ten")],
    "Maryland": [(2014, "Big Ten")],
    "Rutgers": [(2014, "Big Ten")],
    "USC": [(2024, "Big Ten")],
    "UCLA": [(2024, "Big Ten")],
    "Oregon": [(2024, "Big Ten")],
    "Washington": [(2024, "Big Ten")],
    # SEC expansion
    "Texas A&M": [(2012, "SEC")],
    "Missouri": [(2012, "SEC")],
    "Texas": [(2024, "SEC")],
    "Oklahoma": [(2024, "SEC")],
    # ACC expansion
    "Pittsburgh": [(2013, "ACC")],
    "Syracuse": [(2013, "ACC")],
    "Louisville": [(2014, "ACC")],
    "California": [(2024, "ACC")],
    "Stanford": [(2024, "ACC")],
    "SMU": [(2024, "ACC")],
    # Big 12 churn
    "TCU": [(2012, "Big 12")],
    "West Virginia": [(2012, "Big 12")],
    "BYU": [(2011, "Independent"), (2023, "Big 12")],
    "Cincinnati": [(2013, "AAC"), (2023, "Big 12")],
    "Houston": [(2013, "AAC"), (2023, "Big 12")],
    "Central Florida": [(2013, "AAC"), (2023, "Big 12")],
    "Utah": [(2011, "Pac-12"), (2024, "Big 12")],
    "Colorado": [(2011, "Pac-12"), (2024, "Big 12")],
    "Arizona": [(2024, "Big 12")],
    "Arizona St.": [(2024, "Big 12")],
    # Big East football -> AAC (2013)
    "Connecticut": [(2013, "AAC"), (2020, "Independent")],
    "South Florida": [(2013, "AAC")],
    "Temple": [(2012, "Big East"), (2013, "AAC")],
    "Memphis": [(2013, "AAC")],
    "Tulane": [(2014, "AAC")],
    "Tulsa": [(2014, "AAC")],
    "East Carolina": [(2014, "AAC")],
    "Navy": [(2015, "AAC")],
    "Ala-Birmingham": [(2023, "AAC")],
    "Rice": [(2023, "AAC")],
    "North Texas": [(2023, "AAC")],
    "Charlotte": [(2023, "AAC")],
    "Florida Atlantic": [(2023, "AAC")],
    # WAC collapse (2012) -> Mountain West
    "Boise St.": [(2011, "Mountain West")],
    "Fresno St.": [(2012, "Mountain West")],
    "Hawaii": [(2012, "Mountain West")],
    "Nevada": [(2012, "Mountain West")],
    "San Jose St.": [(2013, "Mountain West")],
    "Utah St.": [(2013, "Mountain West")],
    "Louisiana Tech": [(2013, "C-USA")],
    "New Mexico St.": [(2014, "Sun Belt"), (2018, "Independent"), (2023, "C-USA")],
    "Idaho": [(2014, "Sun Belt"), (2018, "FCS")],
    # Sun Belt / C-USA churn
    "Appalachian St.": [(2014, "Sun Belt")],
    "Coastal Carolina": [(2017, "Sun Belt")],
    "Liberty": [(2018, "Independent"), (2023, "C-USA")],
    "Marshall": [(2022, "Sun Belt")],
    "Southern Miss": [(2023, "Sun Belt")],
    "Old Dominion": [(2022, "Sun Belt")],
    "South Alabama": [(2012, "Sun Belt")],
    "Texas St.": [(2013, "Sun Belt")],
    "Western Kentucky": [(2014, "C-USA")],
    "James Madison": [(2022, "Sun Belt")],
    "Georgia Southern": [(2014, "Sun Belt")],
    "Georgia St.": [(2013, "Sun Belt")],
    "Texas-San Antonio": [(2013, "C-USA"), (2023, "AAC")],
    "Massachusetts": [(2012, "MAC"), (2016, "Independent"), (2025, "MAC")],
}


def normalize_college(raw: Optional[str]) -> Optional[str]:
    """Canonical school name, or None when there is nothing usable.

    Multi-school transfer strings ("Miami; Washington State; Incarnate Word",
    which `players.college` carries) resolve to the LAST school listed -- the
    one they left for the draft, and the one whose conference describes their
    final season. `draft_picks_v2.college` is single-school and preferred.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if ";" in s:
        s = s.split(";")[-1].strip()
    key = re.sub(r"[^a-z ]", "", s.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in _ALIASES:
        return _ALIASES[key]
    # "Ohio State" -> "Ohio St." so both source spellings collapse together.
    collapsed = re.sub(r"\bState\b", "St.", s)
    return collapsed


def conference_for(college: Optional[str], draft_season: Optional[int]) -> str:
    """Conference for `college` during the season BEFORE `draft_season`.

    Returns UNMAPPED ("Other") when the school is not in the map -- mostly FCS,
    Division II and Division III programmes, which is a meaningful signal in
    itself rather than a gap to paper over.
    """
    name = normalize_college(college)
    if name is None:
        return UNMAPPED
    conf = _BASE.get(name)
    moves = _MOVES.get(name)
    if moves is None and conf is None:
        return UNMAPPED
    if draft_season is None:
        # No year to resolve against: use the most recent known membership.
        return (moves[-1][1] if moves else conf) or UNMAPPED
    try:
        college_year = int(draft_season) - 1
    except (TypeError, ValueError):
        return conf or UNMAPPED
    for year, new_conf in sorted(moves or []):
        if college_year >= year:
            conf = new_conf
    return conf or UNMAPPED


def is_power5(conference: str) -> int:
    return int(conference in POWER5)


def add_conference_features(df):
    """Attach `college_conference` and `is_power5` to a training frame.

    Reads `draft_college` (from `draft_picks_v2`, single-school and 100%
    populated) in preference to `college` (from `players`, which carries
    multi-school transfer strings). Undrafted players have neither a draft
    college nor a draft season here, so they resolve to UNMAPPED / 0 -- which
    is honest: this feature describes draft-time pedigree, and an undrafted
    player has none to describe.

    Only `is_power5` is model-facing. `college_conference` stays a string for
    inspection and grouping; feeding a 15-value categorical to the models
    would need encoding that has not been justified yet.
    """
    out = df.copy()
    college = out["draft_college"] if "draft_college" in out.columns else out.get("college")
    if college is None:
        out["college_conference"] = UNMAPPED
        out["is_power5"] = 0
        return out
    season = out["draft_season"] if "draft_season" in out.columns else None
    if season is None:
        confs = [conference_for(c, None) for c in college]
    else:
        confs = [conference_for(c, s) for c, s in zip(college, season)]
    out["college_conference"] = confs
    out["is_power5"] = [is_power5(c) for c in confs]
    return out


if __name__ == "__main__":  # coverage report against the live draft table
    import sqlite3
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import pandas as pd

    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(
        "SELECT college, draft_season, COUNT(*) n FROM draft_picks_v2 "
        "WHERE position IN ('QB','RB','WR','TE') AND draft_season >= 2006 "
        "GROUP BY college, draft_season", conn)
    df["conference"] = [conference_for(c, s) for c, s in zip(df.college, df.draft_season)]
    total = int(df.n.sum())
    by_conf = df.groupby("conference").n.sum().sort_values(ascending=False)
    print(f"skill picks 2006+: {total}\n")
    for conf, n in by_conf.items():
        print(f"  {conf:<14} {n:5d}  {n / total:6.1%}")
    unmapped = df[df.conference == UNMAPPED]
    print(f"\nUNMAPPED: {int(unmapped.n.sum())} picks ({unmapped.n.sum() / total:.1%}), "
          f"{unmapped.college.nunique()} schools")
    top = unmapped.groupby("college").n.sum().sort_values(ascending=False).head(12)
    print("  largest unmapped:", ", ".join(f"{c}({n})" for c, n in top.items()))
