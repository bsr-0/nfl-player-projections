"""Holm and Benjamini-Hochberg against known worked examples.

A segment table with M cells runs M independent tests; a 5% per-cell false-
positive rate becomes ~1-0.95**M across the table (56% at M=16). These
corrections are what compare_oof_panels.py uses to avoid reporting that rate
as if it were 5%.
"""
import numpy as np
import pytest

from src.utils.multiple_comparisons import (
    benjamini_hochberg,
    bootstrap_two_sided_p_value,
    holm_bonferroni,
)


# --------------------------------------------------------------------------
# Holm-Bonferroni -- worked example from Holm (1979) / standard textbook case
# --------------------------------------------------------------------------

def test_holm_worked_example():
    """p = [0.01, 0.02, 0.03, 0.04, 0.20] at alpha=0.05.
    Thresholds (m=5): 0.05/5, 0.05/4, 0.05/3, 0.05/2, 0.05/1
                     = 0.010,  0.0125, 0.0167, 0.025,  0.05
    p(1)=0.01 <= 0.010  reject
    p(2)=0.02 >  0.0125 stop -- everything from here on is not rejected
    """
    p = [0.01, 0.02, 0.03, 0.04, 0.20]
    assert holm_bonferroni(p, alpha=0.05) == [True, False, False, False, False]


def test_holm_rejects_all_when_every_p_is_tiny():
    p = [1e-6, 1e-6, 1e-6]
    assert holm_bonferroni(p, alpha=0.05) == [True, True, True]


def test_holm_rejects_none_when_every_p_is_large():
    p = [0.9, 0.8, 0.7]
    assert holm_bonferroni(p, alpha=0.05) == [False, False, False]


def test_holm_is_never_more_liberal_than_uncorrected_alpha():
    """Sanity bound: Holm can only reject fewer or equal cells than a naive
    per-cell alpha=0.05 check would.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=20).tolist()
    holm_rejects = sum(holm_bonferroni(p, alpha=0.05))
    naive_rejects = sum(1 for x in p if x <= 0.05)
    assert holm_rejects <= naive_rejects


def test_holm_single_cell_matches_uncorrected():
    """With only one test, Holm's threshold is alpha/1 -- identical to no
    correction at all.
    """
    assert holm_bonferroni([0.03], alpha=0.05) == [True]
    assert holm_bonferroni([0.07], alpha=0.05) == [False]


# --------------------------------------------------------------------------
# Benjamini-Hochberg -- worked example from the original 1995 paper's style
# --------------------------------------------------------------------------

def test_bh_worked_example():
    """p = [0.01, 0.04, 0.03, 0.005, 0.20], m=5, alpha=0.05.
    Sorted: 0.005, 0.01, 0.03, 0.04, 0.20
    Thresholds (i/m)*alpha: 0.01, 0.02, 0.03, 0.04, 0.05
    i=1: 0.005 <= 0.01  ok
    i=2: 0.01  <= 0.02  ok
    i=3: 0.03  <= 0.03  ok (equality counts)
    i=4: 0.04  <= 0.04  ok
    i=5: 0.20  <= 0.05  fail
    Largest passing i is 4 -> reject the 4 smallest p-values.
    """
    p = [0.01, 0.04, 0.03, 0.005, 0.20]
    result = benjamini_hochberg(p, alpha=0.05)
    assert result == [True, True, True, True, False]


def test_bh_is_at_least_as_liberal_as_holm():
    """BH controls FDR, a weaker guarantee than Holm's FWER, so it should
    never reject strictly fewer cells than Holm on the same data.
    """
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.1, size=15).tolist()
    assert sum(benjamini_hochberg(p, alpha=0.05)) >= sum(holm_bonferroni(p, alpha=0.05))


def test_bh_rejects_none_when_every_p_is_large():
    assert benjamini_hochberg([0.9, 0.8, 0.7], alpha=0.05) == [False, False, False]


# --------------------------------------------------------------------------
# NaN handling -- untestable cells must not distort the family
# --------------------------------------------------------------------------

def test_nan_cells_are_excluded_from_the_family_not_treated_as_zero():
    """A NaN p-value (too few clusters to bootstrap) must not count toward m
    -- including it would make every OTHER cell's threshold stricter than it
    should be, punishing testable cells for an untestable one.
    """
    with_nan = holm_bonferroni([0.01, float("nan"), 0.02], alpha=0.05)
    without_nan_equivalent = holm_bonferroni([0.01, 0.02], alpha=0.05)
    assert with_nan[0] == without_nan_equivalent[0]
    assert with_nan[2] == without_nan_equivalent[1]
    assert with_nan[1] is False


def test_none_is_treated_the_same_as_nan():
    result = holm_bonferroni([0.01, None, 0.02], alpha=0.05)
    assert result[1] is False


def test_all_nan_rejects_nothing():
    assert holm_bonferroni([float("nan")] * 3, alpha=0.05) == [False, False, False]
    assert benjamini_hochberg([float("nan")] * 3, alpha=0.05) == [False, False, False]


def test_empty_input_returns_empty():
    assert holm_bonferroni([], alpha=0.05) == []
    assert benjamini_hochberg([], alpha=0.05) == []


# --------------------------------------------------------------------------
# bootstrap_two_sided_p_value
# --------------------------------------------------------------------------

def test_p_value_small_when_bootstrap_distribution_excludes_zero():
    draws = np.full(1000, 5.0)  # every draw firmly positive
    p = bootstrap_two_sided_p_value(draws)
    assert p < 0.01


def test_p_value_large_when_distribution_straddles_zero():
    rng = np.random.default_rng(2)
    draws = rng.normal(0, 1, size=1000)  # centred on zero
    p = bootstrap_two_sided_p_value(draws)
    assert p > 0.05


def test_p_value_is_never_exactly_zero():
    """Continuity correction: no finite number of draws should produce a
    p-value that claims more certainty than n_boot draws can resolve.
    """
    draws = np.full(500, 10.0)
    assert bootstrap_two_sided_p_value(draws) > 0.0


def test_p_value_coherent_with_a_95_percent_ci_from_the_same_draws():
    """The whole point of deriving both from one set of draws: p < 0.05
    should agree with the 95% CI excluding zero, for the same data."""
    rng = np.random.default_rng(3)
    draws = rng.normal(2.0, 1.0, size=2000)  # clearly positive, not huge effect
    p = bootstrap_two_sided_p_value(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    ci_excludes_zero = not (lo <= 0 <= hi)
    assert (p < 0.05) == ci_excludes_zero


def test_p_value_on_empty_draws_is_nan():
    assert np.isnan(bootstrap_two_sided_p_value(np.array([])))
