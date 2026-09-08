"""tests/test_trading_policy — decision parts of the trading configuration.

Direction mode, session enablement, and the sub-dollar confidence bar live
in brain/policy; config.py re-exports the SAME singleton so telemetry and
existing import paths share one source of truth.
"""

from hanoon_prime.brain.policy.trading_policy import TRADING_CONFIG
from hanoon_prime.immune import PENNY_PRICE, PENNY_SCORE_BAR


def test_direction_helpers_unchanged():
    assert TRADING_CONFIG.direction_mode == "long_only"
    assert TRADING_CONFIG.is_direction_allowed("BUY")
    assert TRADING_CONFIG.is_direction_allowed("LONG")
    assert not TRADING_CONFIG.is_direction_allowed("SELL")
    assert not TRADING_CONFIG.is_direction_allowed("SHORT")


def test_session_helper_matches_today():
    assert TRADING_CONFIG.is_session_active("rth") is True
    assert TRADING_CONFIG.is_session_active("pre_market") is True
    assert TRADING_CONFIG.is_session_active("unknown_session") is True


def test_penny_bar_accepts_sub_dollar_below_bar():
    ok, reason = TRADING_CONFIG.is_penny_bar_cleared("PENN", PENNY_PRICE * 0.8, 0.5)
    assert ok is False
    assert reason == "low_penny_score"


def test_penny_bar_clears_with_high_score():
    assert (
        TRADING_CONFIG.is_penny_bar_cleared("PENN", PENNY_PRICE * 0.8, PENNY_SCORE_BAR)[
            0
        ]
        is True
    )


def test_penny_bar_ignores_non_penny_prices():
    assert TRADING_CONFIG.is_penny_bar_cleared("AAPL", 150.0, 0.5)[0] is True


def test_config_shim_re_exports_same_singleton():
    from hanoon_prime.config import TRADING_CONFIG as SHIM
    from hanoon_prime.config import TradingConfig as ShimClass

    assert SHIM is TRADING_CONFIG
    assert ShimClass is type(TRADING_CONFIG)


def test_to_dict_shape():
    d = TRADING_CONFIG.to_dict()
    assert set(d["sessions"]) == {"pre_market", "rth", "post_market", "overnight"}
    assert d["direction_mode"] == "long_only"
