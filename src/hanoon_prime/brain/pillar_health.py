"""brain.pillar_health — Runtime diagnostics for cognitive pillars.

Verifies that each pillar returns non-zero modifiers on synthetic
test inputs. Catches silent failures like type mismatches or broken imports.

Source: rebuild's pillar_health.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

_TEST_ALPHA = {
    "vpin": 0.72,
    "orderbook_imbalance": 0.65,
    "institutional_flow": 0.81,
    "momentum": 0.37,
    "rsi": 0.55,
    "macd_hist": 0.42,
    "adx": 0.48,
}


@dataclass
class PillarResult:
    name: str
    modifier: float
    ok: bool
    latency_ms: float = 0.0


class PillarHealthChecker:
    """Verify each pillar returns non-zero on test inputs."""

    def check_all(self, pillars: dict[str, object]) -> list[PillarResult]:
        """Run synthetic test through each pillar."""
        results = []
        for name, pillar in pillars.items():
            start = time.monotonic()
            try:
                if hasattr(pillar, "modify"):
                    mod = pillar.modify(
                        _TEST_ALPHA, 0.5, 1, "normal", _TEST_ALPHA, _TEST_ALPHA, 0.58
                    )
                elif hasattr(pillar, "compute"):
                    result = pillar.compute(_TEST_ALPHA)
                    mod = result.modifier if hasattr(result, "modifier") else 0.0
                else:
                    mod = 0.0
                elapsed = (time.monotonic() - start) * 1000
                results.append(PillarResult(name, mod, abs(mod) > 0.0001, elapsed))
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                log.warning("Pillar %s failed: %s", name, exc)
                results.append(PillarResult(name, 0.0, False, elapsed))
        return results

    def get_failures(self, results: list[PillarResult]) -> list[str]:
        """Return names of failed pillars."""
        return [r.name for r in results if not r.ok]
