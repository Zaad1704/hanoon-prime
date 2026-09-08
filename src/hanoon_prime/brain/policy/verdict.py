"""brain.policy.verdict — the decision contract between brain and execution.

One Verdict is produced for EVERY evaluated ticker — never a silent omission.
A veto is data: the log line carries the stage that vetoed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..risk import SizingResult

ENTER: str = "ENTER"
HOLD: str = "HOLD"
VETOED: str = "VETOED"


@dataclass
class Verdict:
    """One brain decision for one ticker in one cycle."""

    ticker: str
    action: str = HOLD
    reason: str = ""
    stage: str = ""
    sizing: SizingResult | None = None
    stop: float | None = None
    target: float | None = None
    horizon: str = "scalp"
    score: float = 0.0
    direction: int = 0
    thought: Any = field(repr=False, default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for journal/telemetry (execution context excluded)."""
        return {
            "ticker": self.ticker,
            "action": self.action,
            "reason": self.reason,
            "stage": self.stage,
            "horizon": self.horizon,
            "score": round(float(self.score), 4),
            "direction": int(self.direction),
        }

    def as_execution_context(self) -> SimpleNamespace:
        """Rebuild the thought-shaped context place_bracket consumes."""
        return SimpleNamespace(
            direction=int(self.direction),
            score=float(self.score),
            verdict=self.action,
        )


__all__ = ["ENTER", "HOLD", "VETOED", "Verdict"]
