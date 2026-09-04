"""hanoon_prime.brain.slow_cortex — System 2: Slow Conceptual Engine.

Runs asynchronously in a background thread every 10-30s.
Handles all network I/O, heavy compute, and disk persistence.
Updates shared BrainState for System 1 to read instantly.

Biological parallel: The prefrontal cortex during sleep —
consolidating memories, recalibrating weights, assessing risk.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from .affective import Affective
from .config import SIGNAL_THRESHOLD
from .episodic import EpisodicMemory
from .guardians import Guardians
from .halim_adapter import HalimAdapter
from .memory import JuliMemory
from .reflection import Reflector
from .regime import RegimeDetector
from .shared_state import BrainState

log = logging.getLogger(__name__)


class SlowCortex:
    """System 2: Background cognitive engine — runs every 10-30s."""

    def __init__(
        self,
        brain_state: BrainState,
        halim_url: str = "http://127.0.0.1:8765",
        interval: float = 30.0,
    ) -> None:
        self.state = brain_state
        self.interval = interval
        self.memory = JuliMemory()
        self.episodic = EpisodicMemory()
        self.regime = RegimeDetector()
        self.affective = Affective()
        self.guardians = Guardians()
        self.halim = HalimAdapter(base_url=halim_url)
        self.reflector = Reflector(self.memory, self.episodic)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start System 2 background loop."""
        self._running = True
        self._sync_initial_state()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="slow-cortex"
        )
        self._thread.start()
        log.info("System 2 started (interval=%.0fs)", self.interval)

    def stop(self) -> None:
        """Stop System 2 background loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._persist_state()
        log.info("System 2 stopped")

    def _sync_initial_state(self) -> None:
        """Push initial state to shared dict on startup."""
        weights = self.memory.get_weights()
        self.state.update(
            threshold=self.memory.threshold,
            indicator_weights=weights,
        )

    def _loop(self) -> None:
        """Main background loop — runs every interval seconds."""
        while self._running:
            try:
                self._cycle()
            except Exception as e:
                log.error("System 2 cycle error: %s", e)
            time.sleep(self.interval)

    def _cycle(self) -> None:
        """One full System 2 cognitive cycle."""
        self._update_regime()
        self._update_halim()
        self._update_affective()
        self._check_guardians()
        self._consolidate_episodic()
        self._persist_state()
        log.info(
            "System 2 write complete | regime=%.2f halim=%.3f threshold=%.3f",
            self.state.get("regime_multiplier", 1.0),
            self.state.get("halim_modifier", 0.0),
            self.state.get("threshold", 0.58),
        )

    def _update_regime(self) -> None:
        """Detect market regime from latest indicators."""
        alpha = self._get_latest_alpha()
        if not alpha:
            return
        prices = [alpha.get(k, 0.5) for k in ["vpin", "momentum", "vwap_deviation"]]
        regime = self.regime.detect(prices)
        self.state.update(
            regime_multiplier=regime.multiplier,
            regime_label=regime.regime,
        )

    def _update_halim(self) -> None:
        """Poll HALIM external AI advisor (network I/O — System 2 only)."""
        alpha = self._get_latest_alpha()
        if not alpha:
            return
        ticker = max(alpha, key=lambda k: abs(alpha.get(k, 0)))
        mod = self.halim.get_modifier(ticker, alpha, 0.0, "SCAN")
        self.state.update(halim_modifier=mod)

    def _update_affective(self) -> None:
        """Update market sentiment from recent trade outcomes."""
        aff = self.affective.evaluate()
        self.state.update(affective_mod=aff.modifier)

    def _check_guardians(self) -> None:
        """Run safety checks on learning stability."""
        weights = self.memory.get_weights()
        verdict = self.guardians.check_weights(weights)
        if not verdict.safe:
            log.warning("Guardian: %s", verdict.reason)

    def _consolidate_episodic(self) -> None:
        """Periodic episodic memory health check."""
        if self.episodic.size > 500:
            log.info("Episodic memory size: %d", self.episodic.size)

    def _persist_state(self) -> None:
        """Atomic write to state.json (disk I/O — System 2 only)."""
        try:
            state_dir = Path(__file__).resolve().parents[2] / "runtime"
            state_dir.mkdir(parents=True, exist_ok=True)
            path = state_dir / "state.json"
            data = {
                "weights": self.memory.get_weights(),
                "threshold": self.memory.threshold,
                "pred_error": self.memory.pred_error,
                "episodic_size": self.episodic.size,
                "brain_state": self.state.snapshot(),
                "timestamp": time.time(),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(path)
        except Exception as e:
            log.warning("State persist failed: %s", e)

    def _get_latest_alpha(self) -> dict[str, float] | None:
        """Read latest alpha from System 1 via shared state."""
        return self.state.get_latest_alpha()

    def on_trade_close(
        self, ticker: str, won: bool, pnl_pct: float, direction: int = 1
    ) -> None:
        """Route trade close to reflection (called from System 1 thread)."""
        alpha: dict[str, float] = {}
        score = 0.0
        self.reflector.on_trade_close(ticker, won, pnl_pct, direction, alpha, score)
        self.affective.update(won, pnl_pct)
        self.state.set_refractory(2.0)
