import math

from src.models.game_outcome.market_picks import spread_pick, total_pick


def test_spread_pick_respects_home_margin_sign_convention():
    assert spread_pick("HOME", "AWAY", 6.0, 3.5) == {"pick": "HOME", "edge": 2.5}
    assert spread_pick("HOME", "AWAY", -5.0, -2.0) == {"pick": "AWAY", "edge": -3.0}


def test_total_pick_uses_forecast_relative_to_market_total():
    assert total_pick(48.0, 45.5) == {"pick": "OVER", "edge": 2.5}
    assert total_pick(41.0, 45.5) == {"pick": "UNDER", "edge": -4.5}


def test_exact_or_missing_market_values_make_no_pick():
    assert spread_pick("HOME", "AWAY", 3, 3) == {"pick": None, "edge": 0.0}
    assert total_pick(45, math.nan) == {"pick": None, "edge": None}
