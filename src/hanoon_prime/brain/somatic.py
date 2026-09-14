"""brain.somatic — Damasio-style somatic markers: a gut-feel bias.

Biological grounding: Damasio (1994, 1996) — the ventromedial prefrontal
cortex generates somatic markers, fast "gut feel" bodily-state signals
that prune the decision space before conscious reasoning. Bechara et al.
(1996) showed Iowa Gambling Task subjects avoid the bad decks BEFORE they
can articulate why — the marker already biased them.

Here the marker is assembled from the brain's OWN body state — the
win/loss pillar, the dopamine RPE channels, and the allostatic alarm —
into a single bounded bias applied to the raw score before stabilization.
A negative marker (posture says "be careful") also precision-dampens the
learned modifiers, like a vmPFC-flavoured attenuation of noise when things
feel off. Advisory only: bounded ±SOMATIC_MAX, and it can never produce
or flip a verdict (R1 — cortex stays the sole verdict source).
"""

from __future__ import annotations

from typing import Any

from .learning_config import (
    SOMATIC_DYS_PENALTY,
    SOMATIC_MAX,
    SOMATIC_PHASIC_GAIN,
    SOMATIC_PRECISION_DAMP,
    SOMATIC_PRECISION_LOOR,
    SOMATIC_TILT_GAIN,
    SOMATIC_TONIC_GAIN,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value into ``[lo, hi]``."""
    return max(lo, min(hi, value))


def _num(rpe: dict[str, Any], key: str) -> float:
    """Numeric value of an RPE channel (0.0 when absent or invalid)."""
    raw = rpe.get(key)
    if not isinstance(raw, (int, float)):
        raw = 0.0
    return float(raw)


class SomaticMarkerGenerator:
    """Gut-feel bias assembled from pillar, RPE, and allostatic state."""

    def __init__(self) -> None:
        self._last_marker = 0.0
        self._last_precision = 1.0

    def generate(
        self,
        pillar: dict[str, Any] | None,
        rpe: dict[str, Any] | None,
        allostatic: dict[str, Any] | None,
    ) -> float:
        """Bias from the realized body-state signals (bounded ±SOMATIC_MAX).

        Pillar lean is a one-way retreat pressure (lean always pushes the
        marker negative). RPE mood is signed: worse-than-expected (negative
        channels) push down, better-than-expected lifts. Phasic surprise
        contributes along its sign. A dyshomeostatic alarm adds an explicit
        ``SOMATIC_DYS_PENALTY``. The result is clamped to ``±SOMATIC_MAX``.
        """
        marker = 0.0
        if isinstance(pillar, dict):
            tilt = _clamp(float(pillar.get("tilt", 0.0) or 0.0), 0.0, 1.0)
            marker += -SOMATIC_TILT_GAIN * tilt
        if isinstance(rpe, dict):
            marker += SOMATIC_TONIC_GAIN * _num(rpe, "tonic")
            marker += SOMATIC_PHASIC_GAIN * _num(rpe, "phasic")
        if isinstance(allostatic, dict) and bool(allostatic.get("dyshomeostatic")):
            marker += SOMATIC_DYS_PENALTY
        marker = _clamp(marker, -SOMATIC_MAX, SOMATIC_MAX)
        self._last_marker = marker
        return marker

    def precision_weight(self, marker: float) -> float:
        """Dampen the learned modifiers when the marker is negative.

        A non-negative marker leaves the modifiers at full volume (1.0);
        a negative marker scales them down, floored at
        ``SOMATIC_PRECISION_LOOR`` so the other voices are never silenced.
        """
        if marker >= 0.0:
            precision = 1.0
        else:
            precision = 1.0 + SOMATIC_PRECISION_DAMP * marker
        precision = max(SOMATIC_PRECISION_LOOR, precision)
        self._last_precision = precision
        return precision

    def snapshot(self) -> dict[str, float]:
        """Last computed marker + precision (telemetry, state-only)."""
        return {
            "marker": round(self._last_marker, 4),
            "precision": round(self._last_precision, 4),
        }
