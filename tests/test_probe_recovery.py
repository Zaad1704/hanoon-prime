"""tests/test_probe_recovery — death-spiral PROBE recovery contract.

Locks the Aegis doctrine: the halt is ON by default. PROBE_RECOVERY_ENABLED
must be False unless a human explicitly opts in for production. When off,
maybe_probe is a no-op no matter how good the signal — the living
check_entry_allowed() halt is unchanged. When on, a probe only fires on a
real death-spiral (3+ consecutive losses), a quality entry (|score|>=0.55,
price>=5.0), and after the 600s cooldown.
"""

from __future__ import annotations

import time

from hanoon_prime.brain.probe_recovery import ProbeRecovery
from hanoon_prime.immune import (
    DEATH_SPIRAL_COOLDOWN_SEC,
    DEATH_SPIRAL_PROBE_MIN_LOSSES,
    PROBE_PRICE_FLOOR,
    PROBE_RECOVERY_ENABLED,
    PROBE_SCORE_FLOOR,
)


def test_probe_recovery_off_by_default():
    """The halt must stay ON until a human opts into recovery (Aegis doctrine)."""
    assert PROBE_RECOVERY_ENABLED is False, "R-safety: probe recovery must default OFF"
    pr = ProbeRecovery()
    # Even a perfect death-spiral setup is ignored while the flag is off.
    assert pr.maybe_probe(0.9, 10.0, 10.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is False


def test_maybe_probe_rejects_without_death_spiral(monkeypatch):
    """No probe unless the consecutive-loss streak reaches the threshold."""
    monkeypatch.setattr(
        "hanoon_prime.brain.probe_recovery.PROBE_RECOVERY_ENABLED", True
    )
    pr = ProbeRecovery()
    assert pr.maybe_probe(0.9, 10.0, 10.0, DEATH_SPIRAL_PROBE_MIN_LOSSES - 1) is False
    assert pr.is_death_spiral(DEATH_SPIRAL_PROBE_MIN_LOSSES - 1) is False
    assert pr.is_death_spiral(DEATH_SPIRAL_PROBE_MIN_LOSSES) is True


def test_maybe_probe_quality_gate(monkeypatch):
    """Weak signals and penny stocks never qualify as a recovery probe."""
    monkeypatch.setattr(
        "hanoon_prime.brain.probe_recovery.PROBE_RECOVERY_ENABLED", True
    )
    pr = ProbeRecovery()
    # score below the floor -> rejected
    assert pr.maybe_probe(0.3, 10.0, 10.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is False
    # price below the floor -> rejected
    assert pr.maybe_probe(0.9, 1.0, 1.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is False
    assert abs(0.3) < PROBE_SCORE_FLOOR
    assert (1.0 + 1.0) / 2 < PROBE_PRICE_FLOOR


def test_maybe_probe_allows_one_recovery_entry(monkeypatch):
    """A qualifying death-spiral + quality signal yields exactly one probe."""
    monkeypatch.setattr(
        "hanoon_prime.brain.probe_recovery.PROBE_RECOVERY_ENABLED", True
    )
    pr = ProbeRecovery()
    # First qualifying probe -> allowed.
    assert pr.maybe_probe(0.6, 50.0, 50.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is True
    # A second probe is throttled by the cooldown.
    assert pr.maybe_probe(0.6, 50.0, 50.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is False


def test_cooldown_allows_next_probe_after_expiry(monkeypatch):
    """After the cooldown elapses, a new recovery probe may fire."""
    monkeypatch.setattr(
        "hanoon_prime.brain.probe_recovery.PROBE_RECOVERY_ENABLED", True
    )
    pr = ProbeRecovery()
    assert pr.maybe_probe(0.6, 50.0, 50.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is True
    # Simulate the cooldown having elapsed.
    pr._last_probe_time = time.monotonic() - (DEATH_SPIRAL_COOLDOWN_SEC + 1.0)
    assert pr.maybe_probe(0.7, 60.0, 60.0, DEATH_SPIRAL_PROBE_MIN_LOSSES) is True
