"""Tests for the era-aware college -> conference map.

The whole point of the map is that it is keyed on (college, year). A static
map would be silently wrong for historical rows rather than failing, so the
realignment boundaries are asserted directly.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.college_conference import (
    UNMAPPED,
    add_conference_features,
    conference_for,
    is_power5,
    normalize_college,
)


class TestRealignmentBoundaries:
    """Each case brackets a real move. `draft_season - 1` is the college year,
    so a player drafted the same spring the school moved still played their
    final season in the OLD conference."""

    @pytest.mark.parametrize("college,draft_season,expected", [
        ("Texas A&M", 2012, "Big 12"),   # played 2011, still Big 12
        ("Texas A&M", 2013, "SEC"),      # played 2012, first SEC year
        ("Nebraska", 2011, "Big 12"),
        ("Nebraska", 2012, "Big Ten"),
        ("USC", 2024, "Pac-12"),         # played 2023, before the Big Ten move
        ("USC", 2025, "Big Ten"),
        ("Texas", 2024, "Big 12"),
        ("Texas", 2025, "SEC"),
        ("Pittsburgh", 2013, "Big East"),
        ("Pittsburgh", 2014, "ACC"),
        ("Cincinnati", 2012, "Big East"),
        ("Cincinnati", 2015, "AAC"),
        ("Cincinnati", 2025, "Big 12"),
    ])
    def test_boundary(self, college, draft_season, expected):
        assert conference_for(college, draft_season) == expected

    def test_utah_three_conferences_over_time(self):
        """Utah is the strongest test: MWC -> Pac-12 -> Big 12."""
        assert conference_for("Utah", 2010) == "Mountain West"
        assert conference_for("Utah", 2012) == "Pac-12"
        assert conference_for("Utah", 2025) == "Big 12"


class TestNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("Ohio State", "Ohio St."),
        ("Ohio St.", "Ohio St."),
        ("Southern Cal", "USC"),
        ("Ole Miss", "Mississippi"),
        ("NC State", "North Carolina St."),
    ])
    def test_aliases_and_spellings_collapse(self, raw, expected):
        assert normalize_college(raw) == expected

    def test_transfer_string_resolves_to_final_school(self):
        """players.college carries multi-school histories; the LAST school is
        the one whose conference described their final season."""
        assert normalize_college("Miami; Washington State") == "Washington St."
        assert conference_for("Miami; Washington State", 2020) == "Pac-12"

    @pytest.mark.parametrize("raw", [None, "", "   ", "nan"])
    def test_empty_inputs_are_unmapped_not_guessed(self, raw):
        assert normalize_college(raw) is None
        assert conference_for(raw, 2020) == UNMAPPED


class TestUnmappedIsHonest:
    def test_fcs_schools_are_unmapped_rather_than_guessed(self):
        assert conference_for("North Dakota St.", 2020) == UNMAPPED
        assert conference_for("Montana", 2015) == UNMAPPED

    def test_school_before_it_reached_fbs(self):
        """Appalachian St. was FCS until 2014; labelling those years Sun Belt
        would assert something untrue."""
        assert conference_for("Appalachian St.", 2012) == UNMAPPED
        assert conference_for("Appalachian St.", 2018) == "Sun Belt"

    def test_power5_flag(self):
        assert is_power5("SEC") == 1
        assert is_power5("MAC") == 0
        assert is_power5(UNMAPPED) == 0


class TestAddConferenceFeatures:
    def test_undrafted_gets_no_fabricated_pedigree(self):
        """Regression: add_advanced_rookie_injury_features mode-fills
        draft_college, which gave every undrafted player "Ohio St." and a
        Big Ten label. Conference must be resolved from the raw frame, so a
        null college stays UNMAPPED and is_power5 stays 0."""
        df = pd.DataFrame({
            "draft_college": ["USC", None, "Alabama"],
            "draft_season": [2020.0, float("nan"), 2019.0],
        })
        out = add_conference_features(df)
        assert list(out["college_conference"]) == ["Pac-12", UNMAPPED, "SEC"]
        assert list(out["is_power5"]) == [1, 0, 1]

    def test_missing_columns_do_not_raise(self):
        out = add_conference_features(pd.DataFrame({"season": [2020]}))
        assert out["college_conference"].iloc[0] == UNMAPPED
        assert out["is_power5"].iloc[0] == 0
