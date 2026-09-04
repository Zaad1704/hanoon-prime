"""hanoon_prime.brain.cognitive.metacognition — Self-reflection (doubt).

Estimates how sure JULI is about its own confidence.
Combines margin uncertainty (how close to threshold) and prediction
error (how often JULI's past predictions were wrong).
Modifier bounded by THINK_BOUND (±0.04).
"""
from __future__ import annotations

from collections import deque

THINK_BOUND: float = 0.04
MARGIN_WEIGHT: float = 0.6
ERROR_WEIGHT: float = 0.4
MIN_ATTEN: float = 0.85
LOOKBACK: int = 50


class Metacognition:
    """Epistemic uncertainty — self-doubt calibration."""

    def __init__(self) -> None:
        self._pred_errors: deque[float] = deque(maxlen=LOOKBACK)

    def record_error(self, predicted: float, actual: float) -> None:
        """Record prediction error for future calibration."""
        self._pred_errors.append(abs(predicted - actual))

    def evaluate(self, score: float, threshold: float) -> float:
        """Compute confidence attenuation from self-reflection.

        Args:
            score: current signal score.
            threshold: entry threshold.

        Returns:
            Modifier in [-THINK_BOUND, +THINK_BOUND].
        """
        margin = abs(abs(score) - threshold)
        margin_unc = max(0.0, 1.0 - margin / 0.3)
        error_unc = 0.5
        if len(self._pred_errors) >= 10:
            avg_err = sum(self._pred_errors) / len(self._pred_errors)
            error_unc = max(0.0, min(1.0, 1.0 - avg_err))
        combined = MARGIN_WEIGHT * margin_unc + ERROR_WEIGHT * error_unc
        attenuation = max(MIN_ATTEN, min(1.0, combined))
        mod = (attenuation - 0.925) * 0.5
        return max(-THINK_BOUND, min(THINK_BOUND, mod))

    def snapshot(self) -> dict[str, float]:
        """Telemetry view."""
        avg = (
            sum(self._pred_errors) / len(self._pred_errors)
            if self._pred_errors
            else 0.0
        )
        return {"avg_pred_error": round(avg, 4), "samples": len(self._pred_errors)}
