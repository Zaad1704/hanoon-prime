"""tests/test_exit_ladder.py — verify the Aegis 3-tier exit ladder.

Covers the non-breaking contract (dormant TIER1/TIER2 leaves TIER3
mechanical unchanged) and each tier's activation. Thresholds are read
from a fresh AdaptiveThresholds so the assertions stay correct even if
the learned defaults drift.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from hanoon_prime.brain.adaptive_thresholds import AdaptiveThresholds
from hanoon_prime.brain.exit_checks import ExitSignal
from hanoon_prime.brain.exit_ladder import ExitLadder
from hanoon_prime.brain.exits import ExitPolicy


class _FakePolicy:
    """Minimal ExitPolicy double: records calls, returns a canned signal."""

    def __init__(self, signal: ExitSignal) -> None:
        self.calls: list = []
        self._sig = signal

    def evaluate(self, ticker, current_price, ib_pnl=0.0, direction=1):
        self.calls.append((ticker, current_price, ib_pnl, direction))
        return self._sig

    def telemetry(self) -> dict:
        return {}


_MECH = ExitSignal(False, "mechanical", "consolidation")


def _mk(tmp_path, policy_signal=None):
    """Build a ladder with a fresh (unpersisted) threshold set + fake policy."""
    policy = _FakePolicy(policy_signal or _MECH)
    at = AdaptiveThresholds(filepath=tmp_path / "at.json")
    return ExitLadder(policy, thresholds=at), policy, at


def test_dormant_defaults_delegate_to_tier3(tmp_path):
    """No exit_likelihood/stop/force → TIER1+TIER2 dormant, TIER3 runs (non-breaking)."""
    ladder, policy, _ = _mk(tmp_path)
    sig = ladder.evaluate("TSLA", 100.0, direction=1)
    assert sig.exit_type == "consolidation"
    assert len(policy.calls) == 1


def test_force_exit_triggers_tier1_before_policy(tmp_path):
    """force_exit fires the hard stop and never consults the policy."""
    ladder, policy, _ = _mk(tmp_path)
    out = ladder.evaluate("TSLA", 100.0, force_exit=True, direction=1)
    assert out.should_exit is True
    assert out.exit_type == "exit"
    assert "hard_stop" in out.reason
    assert policy.calls == []


def test_stop_breach_fires_on_long(tmp_path):
    """Long: price at/below stop_price trips the hard stop."""
    ladder, policy, _ = _mk(tmp_path)
    out = ladder.evaluate("TSLA", 100.0, stop_price=100.0, direction=1)
    assert out.should_exit is True
    assert policy.calls == []


def test_stop_breach_respects_short_direction(tmp_path):
    """Short trips when price >= stop; below the stop it does not (→ TIER3)."""
    ladder, policy, _ = _mk(tmp_path)
    up = ladder.evaluate("TSLA", 101.0, stop_price=100.0, direction=-1)
    assert up.should_exit is True  # short breached upward
    down = ladder.evaluate("TSLA", 99.0, stop_price=100.0, direction=-1)
    assert down.should_exit is False  # short not breached → falls through
    assert len(policy.calls) == 1  # TIER3 mechanical consulted for the non-breach


def test_exit_likelihood_above_threshold_triggers_tier2(tmp_path):
    """exit_likelihood >= adaptive threshold → TIER2 exit (policy skipped)."""
    ladder, policy, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)
    out = ladder.evaluate(
        "TSLA", 100.0, direction=1, exit_likelihood=thr + 0.01, win_rate=0.5
    )
    assert out.should_exit is True
    assert out.exit_type == "exit"
    assert policy.calls == []


def test_watch_band_does_not_exit_and_skips_policy(tmp_path):
    """likelihood in [watch, threshold) → watch; policy not consulted."""
    ladder, policy, at = _mk(tmp_path)
    watch = at.get_watch_threshold(0.5)
    thr = at.get_exit_threshold(0.5)
    assert watch < thr
    out = ladder.evaluate(
        "TSLA", 100.0, direction=1, exit_likelihood=(watch + thr) / 2, win_rate=0.5
    )
    assert out.should_exit is False
    assert out.exit_type == "watch"
    assert policy.calls == []


def test_ride_winners_promotes_watch(tmp_path):
    """In profit + low likelihood → ride-winners watch (policy skipped)."""
    ladder, policy, at = _mk(tmp_path)
    ride_pnl, ride_lik = at.get_ride_winners_params()
    out = ladder.evaluate(
        "TSLA",
        100.0,
        ib_pnl=ride_pnl + 1.0,
        direction=1,
        exit_likelihood=ride_lik - 0.01,
        win_rate=0.5,
    )
    assert out.should_exit is False
    assert out.exit_type == "watch"
    assert policy.calls == []


def test_tier2_hold_falls_through_to_tier3(tmp_path):
    """Below watch threshold → hold → TIER3 mechanical runs."""
    ladder, policy, at = _mk(tmp_path)
    watch = at.get_watch_threshold(0.5)
    out = ladder.evaluate(
        "TSLA", 100.0, direction=1, exit_likelihood=watch - 0.05, win_rate=0.5
    )
    assert out.exit_type == "consolidation"
    assert len(policy.calls) == 1


def test_tier1_precedence_over_tier2_and_tier3(tmp_path):
    """force_exit wins even when TIER2 would also exit."""
    ladder, policy, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)
    out = ladder.evaluate(
        "TSLA",
        100.0,
        direction=1,
        exit_likelihood=thr + 0.5,
        win_rate=0.5,
        force_exit=True,
    )
    assert out.exit_type == "exit"
    assert "hard_stop" in out.reason
    assert policy.calls == []


def test_ladder_wired_into_orchestrator():
    """JuliBrain.check_exit delegates to the ladder (non-breaking passthrough)."""
    from hanoon_prime.brain.orchestrator import NeuromorphicBrain

    brain = NeuromorphicBrain.__new__(NeuromorphicBrain)
    brain.exits = _FakePolicy(signal=_MECH)
    brain._exit_ladder = _FakePolicy(signal=_MECH)  # stand in for the real ladder
    sentinel = ExitSignal(False, "ok", "hold")
    brain._exit_ladder = MagicMock()
    brain._exit_ladder.evaluate.return_value = sentinel
    got = brain.check_exit("TSLA", 100.0)
    assert got is sentinel
    brain._exit_ladder.evaluate.assert_called_once()


def test_ladder_derives_exit_likelihood_from_policy(tmp_path):
    """Unset exit_likelihood is derived from the policy's pillar signal.

    This is the wiring that activates TIER2 for existing juli.py callers
    without changing their call sites.
    """
    _, _, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)

    class _LlPolicy:
        def exit_likelihood(self, ticker, current_price, direction):
            return thr + 0.01

        def evaluate(self, *args, **kwargs):
            return _MECH

    out = ExitLadder(_LlPolicy(), thresholds=at).evaluate("TSLA", 100.0)
    assert out.should_exit is True
    assert out.exit_type == "exit"
    assert "juli_verdict" in out.reason


def test_ladder_consults_win_rate_provider(tmp_path):
    """A win_rate_provider overrides the 0.5 default win rate."""

    class _RecThresholds:
        def __init__(self):
            self.received = None

        def get_exit_threshold(self, wr):
            self.received = wr
            return 0.0

        def get_watch_threshold(self, wr):
            self.received = wr
            return 0.0

        def get_ride_winners_params(self):
            return 999.0, 0.0

    class _LlPolicy:
        def exit_likelihood(self, ticker, current_price, direction):
            return 0.9

        def evaluate(self, *args, **kwargs):
            return _MECH

    rec = _RecThresholds()
    ladder = ExitLadder(
        _LlPolicy(), thresholds=rec, win_rate_provider=lambda: (0.66, 9)
    )
    out = ladder.evaluate("TSLA", 100.0)
    assert out.should_exit is True
    assert rec.received == 0.66


def test_policy_exit_likelihood_bounds(monkeypatch):
    """exit_likelihood is bounded to [0,1], 0 for unknown tickers, and
    grows as a position degrades (time + drawdown)."""
    base = 1_700_000_000.0
    monkeypatch.setattr("hanoon_prime.brain.exits.time.time", lambda: base)
    policy = ExitPolicy()
    assert policy.exit_likelihood("NOPE", 100.0) == 0.0
    policy.register("TSLA", 100.0, {"rsi": 0.1})
    ll = policy.exit_likelihood("TSLA", 100.0)
    assert 0.0 <= ll <= 1.0
    monkeypatch.setattr("hanoon_prime.brain.exits.time.time", lambda: base + 70 * 60)
    ll_decayed = policy.exit_likelihood("TSLA", 90.0)
    # Long-held loser degrades the setup far past a fresh flat position.
    assert ll_decayed > ll


# ── Hysteresis (rebuild HYSTERESIS_BARS port, off-by-default) ─────────
from hanoon_prime.immune import HYSTERESIS_BARS, HYSTERESIS_EXIT_ENABLED  # noqa: E402


def test_hysteresis_off_by_default_flag():
    """Both hysteresis constants locked off / faithful (R6 literals)."""
    assert HYSTERESIS_EXIT_ENABLED is False
    assert HYSTERESIS_BARS == 3


def test_soft_exit_fires_immediately_when_disabled(tmp_path):
    """Flag off → TIER2 exits on the 1st confirming bar (byte-identical old)."""
    ladder, policy, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)
    out = ladder.evaluate("TSLA", 100.0, exit_likelihood=thr + 0.01, win_rate=0.5)
    assert out.should_exit is True
    assert out.exit_type == "exit"


def test_soft_exit_requires_persistence_when_enabled(monkeypatch, tmp_path):
    """Flag on → TIER2 must fire HYSTERESIS_BARS consecutive bars (T3 faithful)."""
    monkeypatch.setattr("hanoon_prime.brain.exit_ladder.HYSTERESIS_EXIT_ENABLED", True)
    ladder, policy, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)
    kwargs = dict(exit_likelihood=thr + 0.01, win_rate=0.5)
    first = ladder.evaluate("TSLA", 100.0, **kwargs)
    second = ladder.evaluate("TSLA", 100.0, **kwargs)
    third = ladder.evaluate("TSLA", 100.0, **kwargs)
    # Pending for the first (HYSTERESIS_BARS - 1) confirming bars.
    assert first.should_exit is False and first.reason == "hysteresis (pending)"
    assert second.should_exit is False and second.reason == "hysteresis (pending)"
    # Confirmed on the Nth consecutive firing.
    assert third.should_exit is True


def test_hysteresis_resets_on_non_firing_bar(monkeypatch, tmp_path):
    """A single non-firing bar resets the streak; must re-accumulate."""
    monkeypatch.setattr("hanoon_prime.brain.exit_ladder.HYSTERESIS_EXIT_ENABLED", True)
    ladder, policy, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)
    fire = dict(exit_likelihood=thr + 0.01, win_rate=0.5)
    ladder.evaluate("TSLA", 100.0, **fire)  # streak 1
    ladder.evaluate("TSLA", 100.0, **fire)  # streak 2
    # A hold bar (below watch) falls through to TIER3 and resets the streak.
    reset = ladder.evaluate("TSLA", 100.0, exit_likelihood=0.0, win_rate=0.5)
    assert reset.should_exit is False
    # Two more fired bars are NOT enough — streak restarted from 0.
    a = ladder.evaluate("TSLA", 100.0, **fire)
    b = ladder.evaluate("TSLA", 100.0, **fire)
    assert a.should_exit is False and b.should_exit is False
    c = ladder.evaluate("TSLA", 100.0, **fire)
    assert c.should_exit is True  # 3rd consecutive after the reset


def test_tier1_hard_stop_bypasses_hysteresis_when_enabled(monkeypatch, tmp_path):
    """Hard stops are HYSTERESIS_HARD_OVERRIDE=True — never gated by the streak."""
    monkeypatch.setattr("hanoon_prime.brain.exit_ladder.HYSTERESIS_EXIT_ENABLED", True)
    ladder, policy, _ = _mk(tmp_path)
    out = ladder.evaluate("TSLA", 100.0, force_exit=True, direction=1)
    assert out.should_exit is True
    assert "hard_stop" in out.reason


def test_hysteresis_is_per_ticker(monkeypatch, tmp_path):
    """Streaks are keyed by ticker — one ticker's streak doesn't prime another."""
    monkeypatch.setattr("hanoon_prime.brain.exit_ladder.HYSTERESIS_EXIT_ENABLED", True)
    ladder, policy, at = _mk(tmp_path)
    thr = at.get_exit_threshold(0.5)
    fire = dict(exit_likelihood=thr + 0.01, win_rate=0.5)
    # TSLA fires 3x → exits (streak 3).
    for _ in range(3):
        result = ladder.evaluate("TSLA", 100.0, **fire)
    assert result.should_exit is True
    # AAPL is a fresh ticker: its 1st confirming bar is pending, not exited.
    fresh = ladder.evaluate("AAPL", 100.0, **fire)
    assert fresh.should_exit is False
    assert fresh.reason == "hysteresis (pending)"
