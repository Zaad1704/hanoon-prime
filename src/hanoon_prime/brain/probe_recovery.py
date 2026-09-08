"""brain.probe_recovery — optional death-spiral PROBE recovery (off-by-default).

Faithful Prime translation of the rebuild's death-spiral admission path
(`ops/candidate.py:240-252`, `ops/admission.py:150-152`,
`hanoon/juli/memory_calibration.py:detect_death_spiral`).

Doctrine (Aegis, v2.3.0): the **halt is the default and stays on**.
`brain.hippocampus.check_safety_nets` still raises `RuntimeError` on a
consecutive-loss / daily-loss / position-cap trip exactly as before. This
module only runs *after* that halt returns False, and only when
`PROBE_RECOVERY_ENABLED` is True (it is, by default, False). When it fires,
one quality-gated entry bypasses the halt so the organism can re-establish
edge and resume learning — then the existing pause kicks back in, so the
bot never runs free: one probe, then back to the halt.

BUILD_NAME "Aegis". R1/R13-safe: no verdict strings here, only a boolean
"may I try one recovery entry?"
"""

from __future__ import annotations

import time

from ..immune import (
    DEATH_SPIRAL_COOLDOWN_SEC,
    DEATH_SPIRAL_PROBE_MIN_LOSSES,
    PROBE_PRICE_FLOOR,
    PROBE_RECOVERY_ENABLED,
    PROBE_SCORE_FLOOR,
)


class ProbeRecovery:
    """One-shot death-spiral probe: quality gate + cooldown, off-by-default."""

    def __init__(self) -> None:
        """No state — cooldown timestamp is the only mutable bit."""
        self._last_probe_time: float = 0.0

    @staticmethod
    def is_death_spiral(consecutive_losses: int) -> bool:
        """True when the consecutive-loss streak reaches the probe threshold.

        Mirrors rebuild ``detect_death_spiral``'s "3+ consecutive losses"
        trigger. (The richer "all WR bands < 30%" check needs RealizedStats,
        which the live halt path does not bind; the loss streak is the
        faithful, low-coupling proxy.)
        """
        return consecutive_losses >= DEATH_SPIRAL_PROBE_MIN_LOSSES

    def maybe_probe(
        self,
        score: float,
        bid: float,
        ask: float,
        consecutive_losses: int,
    ) -> bool:
        """May a HALT yield to a single quality-gated PROBE entry?

        False unless ``PROBE_RECOVERY_ENABLED``. When enabled, only fires on a
        death-spiral (consecutive-loss streak >= threshold), a quality entry
        (|score| >= PROBE_SCORE_FLOOR, price >= PROBE_PRICE_FLOOR), and after
        the DEATH_SPIRAL_COOLDOWN_SEC cooldown. Records the probe time on a
        True so repeats are throttled to one-per-cooldown.
        """
        if not PROBE_RECOVERY_ENABLED:
            return False
        if not self.is_death_spiral(consecutive_losses):
            return False
        price = (float(bid) + float(ask)) / 2.0
        if abs(float(score)) < PROBE_SCORE_FLOOR or price < PROBE_PRICE_FLOOR:
            return False
        now = time.monotonic()
        if now - self._last_probe_time < DEATH_SPIRAL_COOLDOWN_SEC:
            return False
        self._last_probe_time = now
        return True


# Module-level singleton (cf. ib_cycle's _PORTFOLIO_RISK) so the cooldown
# persists across the live bot's lifetime.
probe_recovery = ProbeRecovery()

__all__ = ["ProbeRecovery", "probe_recovery"]
