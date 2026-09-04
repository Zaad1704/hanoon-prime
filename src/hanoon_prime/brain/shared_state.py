"""hanoon_prime.brain.shared_state — Thread-safe state exchange.

Atomic bridge between System 2 (slow background) and System 1 (fast tick).
System 2 writes, System 1 reads. No locks on the read path.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class BrainState:
    """Thread-safe shared state between System 1 and System 2."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "regime_multiplier": 1.0,
            "regime_label": "unknown",
            "halim_modifier": 0.0,
            "episodic_bias": 0.0,
            "affective_mod": 0.0,
            "salience_atten": 1.0,
            "threshold": 0.58,
            "refractory_until": 0.0,
            "risk_ceiling": 0.02,
            "indicator_weights": {},
            "timestamp": 0.0,
        }

    def update(self, **kwargs: Any) -> None:
        """Atomic update of multiple state keys."""
        with self._lock:
            self._state.update(kwargs)
            self._state["timestamp"] = time.time()

    def get(self, key: str, default: Any = None) -> Any:
        """Read a single key atomically."""
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        """Read full state atomically."""
        with self._lock:
            return dict(self._state)

    def is_refractory(self) -> bool:
        """Check if System 1 is in refractory period (suppress trading)."""
        with self._lock:
            until = self._state["refractory_until"]
            return bool(time.time() < float(until))

    def set_refractory(self, duration: float) -> None:
        """Set refractory period after a trade event."""
        with self._lock:
            self._state["refractory_until"] = time.time() + duration

    def set_latest_alpha(self, alpha: dict[str, float]) -> None:
        """Store latest alpha from System 1 for System 2 to read."""
        with self._lock:
            self._state["latest_alpha"] = dict(alpha)

    def get_latest_alpha(self) -> dict[str, float] | None:
        """Read latest alpha computed by System 1."""
        with self._lock:
            a = self._state.get("latest_alpha")
            return dict(a) if a else None

    def set_latest_prices(self, prices: list[float]) -> None:
        """Store latest price array from System 1 for regime classification."""
        with self._lock:
            self._state["latest_prices"] = list(prices[-50:]) if prices else []

    def get_latest_prices(self) -> list[float]:
        """Read latest prices computed by System 1."""
        with self._lock:
            return list(self._state.get("latest_prices", []))
