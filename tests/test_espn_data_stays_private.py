"""ESPN fantasy-league data must never reach the published site.

The site is NFL player projections. League data -- the user's own roster,
lineup, matchups -- is private and has no place in it.

The safeguard is LOCATION, not discipline. docs/ is the GitHub Pages root and
the pages fetch relative paths beneath it, so a browser cannot reach a file
outside docs/ whatever the markup says. Keeping league data under data/ makes
publishing it impossible rather than merely discouraged.

This is a regression test, not a description of a bug: the tree is already
clean. It exists because the predecessor script wrote
docs/data/my_roster.json -- inside the published tree -- and the next one could
just as easily default to the same place.
"""
import re
import subprocess

import pytest

from config.settings import PROJECT_ROOT, ESPN_PRIVATE_DIR

DOCS = PROJECT_ROOT / "docs"

# Names that would indicate league data rather than NFL data.
LEAGUE_SHAPED = re.compile(
    r"(my_roster|roster|lineup|league|matchup|waiver|owner|manager|espn)",
    re.I)

# Every data file the UI is allowed to request. Anything else appearing in a
# fetch() is either a new NFL payload (add it here deliberately) or league data
# that must not be there.
ALLOWED_FETCH = {
    "data/weekly_meta.json",
    "data/players_${p}.json",
    "data/weekly_${SEASON}_wk${week}.json",
    # season.html, the off-nav predecessor dashboard. NFL season totals.
    "data/projections_2026.json",
}


def _published_pages():
    return sorted(DOCS.glob("*.html"))


def test_no_league_shaped_data_file_under_docs():
    """docs/ holds NFL payloads only."""
    offenders = [
        p.relative_to(PROJECT_ROOT)
        for p in DOCS.rglob("*.json")
        if LEAGUE_SHAPED.search(p.name)
    ]
    assert not offenders, (
        f"league-shaped data inside the published directory: {offenders}. "
        f"League data belongs in {ESPN_PRIVATE_DIR.relative_to(PROJECT_ROOT)}")


def test_private_dir_is_outside_the_published_tree():
    """The whole guarantee rests on this one fact."""
    assert DOCS not in ESPN_PRIVATE_DIR.parents, (
        "ESPN_PRIVATE_DIR is inside docs/ and would be published")


def test_ui_fetches_only_known_nfl_payloads():
    """A future fetch("data/my_roster.json") fails here rather than shipping.

    Matches the fetch() argument verbatim, template literals included, so
    adding a payload is a deliberate edit to ALLOWED_FETCH.
    """
    seen = set()
    for page in _published_pages():
        seen |= set(re.findall(r'fetch\(\s*[`"\']([^`"\']+)[`"\']',
                               page.read_text()))
    unexpected = seen - ALLOWED_FETCH
    assert not unexpected, (
        f"UI fetches unrecognised payloads: {sorted(unexpected)}. "
        "If this is a new NFL payload, add it to ALLOWED_FETCH. If it is "
        "league data, it must not be fetched at all.")


def test_no_page_references_league_data():
    """Catches markup that names a league payload without fetch()."""
    offenders = []
    for page in _published_pages():
        for line in page.read_text().splitlines():
            if "<!--" in line or line.strip().startswith("*"):
                continue  # prose about league SIZE/settings is legitimate
            if re.search(r"my_roster|espn_|/roster|matchup", line, re.I):
                offenders.append(f"{page.name}: {line.strip()[:80]}")
    assert not offenders, f"league data referenced in published markup: {offenders}"


@pytest.mark.skipif(not (PROJECT_ROOT / ".git").exists(), reason="needs a git checkout")
def test_private_dir_contents_are_gitignored():
    """Ignoring is what keeps a pulled roster out of the repo."""
    probe = ESPN_PRIVATE_DIR / "probe_roster.json"
    rel = probe.relative_to(PROJECT_ROOT)
    r = subprocess.run(["git", "check-ignore", str(rel)],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{rel} is NOT gitignored; pulled league data could be committed")


@pytest.mark.skipif(not (PROJECT_ROOT / ".git").exists(), reason="needs a git checkout")
def test_the_rendered_report_is_gitignored_too():
    """The HTML report is league data wearing a different extension."""
    probe = ESPN_PRIVATE_DIR / "reports" / "probe_report.html"
    rel = probe.relative_to(PROJECT_ROOT)
    r = subprocess.run(["git", "check-ignore", str(rel)],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{rel} is NOT gitignored"


@pytest.mark.skipif(not (PROJECT_ROOT / ".git").exists(), reason="needs a git checkout")
def test_no_league_data_is_tracked_anywhere():
    r = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                       capture_output=True, text=True)
    tracked = [
        f for f in r.stdout.splitlines()
        if f.endswith(".json") and LEAGUE_SHAPED.search(f.rsplit("/", 1)[-1])
    ]
    assert not tracked, f"league-shaped data is tracked in git: {tracked}"
