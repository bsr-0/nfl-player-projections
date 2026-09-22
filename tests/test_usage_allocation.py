import numpy as np
import pytest

from src.models.usage_allocation import draw_usage_shares

def test_usage_draws_conserve_modeled_opportunity_pool():
    rng = np.random.default_rng(3)
    draws = np.asarray([draw_usage_shares(np.array([.40, .25]), concentration=80, rng=rng)
                        for _ in range(1000)])
    assert np.all(draws.sum(axis=1) <= 1.0)
    assert draws.mean(axis=0) == pytest.approx([.40, .25], abs=.02)

def test_usage_rejects_invalid_team_share_sum():
    with pytest.raises(ValueError, match="exceed"):
        draw_usage_shares(np.array([.7, .5]), concentration=30,
                           rng=np.random.default_rng(1))
