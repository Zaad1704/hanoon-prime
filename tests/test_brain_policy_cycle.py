"""tests/test_brain_policy_cycle — slow cortex publishes the policy state.

The ConsolidationEngine (System 2) owns PortfolioRiskManager + SafetyProducer
and publishes `policy_state` / `policy_exits` into BrainState every pulse, so
the fast cortex can make every decision decision-visible.
"""

from hanoon_prime.brain.consolidation import ConsolidationEngine
from hanoon_prime.brain.policy.safety import SafetyProducer
from hanoon_prime.brain.shared_state import DEFAULT_POLICY_STATE, BrainState
from hanoon_prime.immune import DAILY_LOSS_LIMIT


def engine() -> tuple[ConsolidationEngine, BrainState]:
    state = BrainState()
    eng = ConsolidationEngine(state)
    eng.safety.set_enabled(True)  # halt rules active (producer defaults off)
    eng._update_policy()
    return eng, state


def test_default_policy_state_authorized_but_unsynced():
    assert DEFAULT_POLICY_STATE["authorized"] is True
    assert DEFAULT_POLICY_STATE["equity_synced"] is False


def test_update_policy_from_account_feed_publishes():
    eng, state = engine()
    state.update(
        account_feed={
            "daily_pnl": 100.0,
            "equity": 100_000.0,
            "positions": {"TSLA": {"value": 5000.0, "pnl": 10.0}},
        },
        positions_open={"TSLA": {}},
        consecutive_losses=0,
    )
    eng._update_policy()
    ps = state.get("policy_state")
    assert ps["equity_synced"] is True
    assert ps["authorized"] is True
    assert ps["daily_pnl"] == 100.0
    assert ps["holdings"]["TSLA"] == 5000.0
    assert ps["position_count"] == 1


def test_daily_loss_halt_shows_in_published_state():
    eng, state = engine()
    state.update(
        account_feed={"daily_pnl": -(DAILY_LOSS_LIMIT + 1.0)},
        positions_open={},
        consecutive_losses=0,
    )
    eng._update_policy()
    ps = state.get("policy_state")
    assert ps["authorized"] is False
    assert ps["halted"] is True
    assert "daily" in ps["pause_reason"]


def test_consecutive_losses_shows_in_published_state():
    eng, state = engine()
    state.update(
        account_feed={"daily_pnl": 0.0},
        positions_open={},
        consecutive_losses=3,
    )
    eng._update_policy()
    ps = state.get("policy_state")
    assert ps["authorized"] is False
    assert ps["consecutive_losses"] == 3


def test_giveback_published_as_policy_exit():
    eng, state = engine()
    state.update(
        account_feed={
            "daily_pnl": 0.0,
            "equity": 100_000.0,
            "positions": {"NVD": {"pnl": 40_000.0, "value": 80_000.0}},
        },
        positions_open={"NVD": {}},
        consecutive_losses=0,
    )
    eng._update_policy()  # first pulse sets the unrealized peak
    state.update(
        account_feed={
            "daily_pnl": 0.0,
            "equity": 100_000.0,
            "positions": {"NVD": {"pnl": 20_000.0, "value": 90_000.0}},
        },
        positions_open={"NVD": {}},
        consecutive_losses=0,
    )
    eng._update_policy()  # 50% fade -> giveback fires
    exits = state.get("policy_exits")
    assert exits, "giveback must publish a policy_exit"
    assert exits[0]["type"] == "portfolio_giveback"
    assert exits[0]["ticker"] == "NVD"


def test_safety_owned_by_slow_cortex():
    eng, _ = engine()
    assert isinstance(eng.portfolio_risk, object)
    assert isinstance(eng.safety, SafetyProducer)


def test_published_state_has_all_fast_gate_keys():
    eng, state = engine()
    state.update(
        account_feed={"daily_pnl": 0.0, "equity": 100_000.0, "positions": {}},
        positions_open={},
        consecutive_losses=0,
    )
    eng._update_policy()
    ps = state.get("policy_state")
    for key in (
        "equity",
        "equity_synced",
        "risk_scalar",
        "drawdown",
        "stress_mode",
        "exposure",
        "position_count",
        "max_positions",
        "holdings",
        "authorized",
        "halted",
        "pause_reason",
        "daily_pnl",
        "consecutive_losses",
    ):
        assert key in ps, f"policy_state missing {key}"
