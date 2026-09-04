"""nerve — Central nervous system for signal routing.

Routes signals between modules. Rebuild's nerve/center.py simplified.

Source: rebuild's senses/nerve/center.py (simplified).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class NerveState:
    defensive: bool = False
    panic: bool = False
    market_open: bool = True
    session: str = "RTH"
    last_signal: float = 0.0


class NerveCenter:
    """Central nervous system for signal routing."""

    def __init__(self) -> None:
        """Auto-generated docstring."""
        self._state = NerveState()
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def register(self, signal: str, handler: Callable[..., Any]) -> None:
        """Auto-generated docstring."""
        self._handlers.setdefault(signal, []).append(handler)

    def emit(self, signal: str, data: Any = None) -> None:
        """Auto-generated docstring."""
        for handler in self._handlers.get(signal, []):
            try:
                handler(data)
            except Exception as exc:
                log.warning("Handler %s failed: %s", signal, exc)
        self._state.last_signal = time.time()

    def set_defensive(self, defensive: bool) -> None:
        """Auto-generated docstring."""
        self._state.defensive = defensive

    def set_panic(self, panic: bool) -> None:
        """Auto-generated docstring."""
        self._state.panic = panic

    def get_state(self) -> NerveState:
        """Auto-generated docstring."""
        return self._state
