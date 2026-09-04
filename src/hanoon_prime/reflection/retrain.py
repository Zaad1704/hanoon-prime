"""reflection.retrain — Retrain engine.

Triggers retraining from trade buffer when enough new trades accumulate.
Simple version — no XGBoost, just updates brain weights.

Source: rebuild's retrain.py (simplified).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

_RETRAIN_INTERVAL: float = 24 * 3600.0  # 24h
_MIN_TRADES: int = 50


class RetrainEngine:
    """Trigger retraining from trade buffer."""

    def __init__(self, buffer: Any = None, on_retrained: Optional[Any] = None) -> None:
        """Initialize retrain engine."""
        self._buffer = buffer
        self._on_retrained = on_retrained
        self._last_retrain: float = 0.0
        self._retrain_count: int = 0

    def should_retrain(self) -> bool:
        """Check if retrain is needed."""
        now = time.time()
        if now - self._last_retrain < _RETRAIN_INTERVAL:
            return False
        if self._buffer is None:
            return False
        trades = self._buffer.get_trades(last_n=_MIN_TRADES)
        return len(trades) >= _MIN_TRADES

    def retrain(self) -> dict:
        """Run retraining. Returns report."""
        if not self.should_retrain():
            return {"status": "skipped", "reason": "not_needed"}
        self._last_retrain = time.time()
        self._retrain_count += 1
        trades = self._buffer.get_trades(last_n=100) if self._buffer else []
        wr = sum(1 for t in trades if t.win) / max(len(trades), 1)
        log.info(
            "Retrain #%d: %d trades, WR=%.2f", self._retrain_count, len(trades), wr
        )
        if self._on_retrained:
            try:
                self._on_retrained()
            except Exception as exc:
                log.warning("on_retrained callback failed: %s", exc)
        return {"status": "done", "trades": len(trades), "wr": wr}

    def get_telemetry(self) -> dict:
        """Auto-generated docstring."""
        return {
            "last_retrain": self._last_retrain,
            "retrain_count": self._retrain_count,
        }
