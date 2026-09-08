"""hanoon_prime.contrarian — MoE mean-reversion override port.

Faithful translation of the rebuild's ``moe_meanrev_extreme`` expert: when an
oscillator pins an extreme z-score (overbought / oversold), it votes a
CONTRARIAN direction (overbought → SHORT, oversold → LONG), and in the
rebuild it INHIBITS the generalist bullish layer (weight -0.3).

Prime has no spiking-neuron stack, so this is the lightweight equivalent:
a pure function of Prime's own z-scores. OFF by default
(``CONTRARIAN_MODE_ENABLED`` in immune.py) — returns 0 when disabled, which is
a no-op for Cortex (R1: cortex remains the sole verdict producer). The flag is
owned here so cortex adds only a 2-line guarded post-step.

Note: the rebuild fired meanrev_extreme on a *consensus* of 5+ overbought
oscillators; Prime exposes a single mean-reversion indicator
(``vwap_deviation``), so the faithful proxy is that indicator's extreme z.
"""

from __future__ import annotations

from .immune import CONTRARIAN_EXTREME_Z, CONTRARIAN_MODE_ENABLED


def contrarian_direction(z_scores: dict[str, float]) -> int:
    """Return a contrarian direction override, or 0 when none / disabled.

    +1 = contrarian LONG (oversold: vwap_deviation very negative),
    -1 = contrarian SHORT (overbought: vwap_deviation very positive),
     0 = no override (mode off, or not extreme).
    """
    if not CONTRARIAN_MODE_ENABLED:
        return 0
    z = z_scores.get("vwap_deviation", 0.0)
    if abs(z) < CONTRARIAN_EXTREME_Z:
        return 0
    # Overbought (z > 0) -> contrarian SHORT (-1); oversold (z < 0) -> LONG (+1).
    return -1 if z > 0 else 1


__all__ = ["contrarian_direction"]
