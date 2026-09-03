"""hanoon_prime.juli — Biological brain orchestrator.

JULI is the full brain: scanner → thalamus → amygdala/pfc → decision.
Replaces hardcoded tickers with dynamic IB scanner discovery.
Allocates data budget smartly across positions and candidates.
Works slowly and carefully, like a real biological brain.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .brain.amygdala import Amygdala
from .brain.pfc import PrefrontalCortex
from .brain.plasticity import Plasticity
from .brain.thalamus import Thalamus, ThalamusVerdict
from .cerebellum import compute_alpha
from .cortex import Cortex, Thought
from .data.budget import DataBudget
from .data.scanner import IBScanner, ScanResult

log = logging.getLogger(__name__)
SCANNER_INTERVAL: float = 300.0
MAX_CANDIDATES: int = 20
ENTRY_THRESHOLD: float = 0.65


class JuliBrain:
    """Biological brain orchestrator — scanner to decision."""

    def __init__(self, ib_client: Any) -> None:
        self.ib = ib_client
        self.scanner = IBScanner(ib_client)
        self.thalamus = Thalamus()
        self.amygdala = Amygdala()
        self.pfc = PrefrontalCortex()
        self.plasticity = Plasticity()
        self.budget = DataBudget()
        self.cortex = Cortex()
        self._candidates: list[ScanResult] = []
        self._eligible: list[ThalamusVerdict] = []
        self._last_scan: float = 0.0

    def tick(self, positions: set[str], get_snapshot: Any) -> list[dict[str, Any]]:
        """One brain cycle. Returns list of decisions."""
        self._maybe_scan()
        self._maybe_screen(get_snapshot)
        self._maybe_allocate(positions)
        return self._evaluate(positions, get_snapshot)

    def _maybe_scan(self) -> None:
        """Run scanner every 5 minutes."""
        now = time.time()
        if now - self._last_scan < SCANNER_INTERVAL:
            return
        try:
            self.scanner.scan("hot_volume")
            self._candidates = self.scanner.get_candidates()
            self._last_scan = now
            log.info("SCAN: %d raw candidates", len(self._candidates))
        except Exception as e:
            log.warning("Scan failed: %s", e)

    def _maybe_screen(self, get_snapshot: Any) -> None:
        """Screen candidates through thalamus."""
        if not self._candidates:
            return
        screened: list[ThalamusVerdict] = []
        for cand in self._candidates[:MAX_CANDIDATES]:
            snap = get_snapshot(cand.symbol)
            if snap is None:
                continue
            v = self.thalamus.screen(
                ticker=cand.symbol,
                bid=snap.get("bid", 0),
                ask=snap.get("ask", 0),
                last=snap.get("last", 0),
                volume=snap.get("volume", 0),
                daily_volume=snap.get("daily_volume", 0),
            )
            screened.append(v)
        self._eligible = self.thalamus.rank(screened)
        syms = [s.ticker for s in self._eligible[:10]]
        log.info(
            "THALAMUS: %d/%d — %s", len(self._eligible), len(self._candidates), syms
        )

    def _maybe_allocate(self, positions: set[str]) -> None:
        """Allocate data budget."""
        candidates = [s.ticker for s in self._eligible[:MAX_CANDIDATES]]
        to_sub, to_unsub = self.budget.allocate(positions, candidates)
        if to_sub or to_unsub:
            log.info(
                "BUDGET: +%d -%d tiers=%s",
                len(to_sub),
                len(to_unsub),
                self.budget.count_tiers(),
            )

    def _evaluate(self, positions: set[str], get_snapshot: Any) -> list[dict[str, Any]]:
        """Evaluate all tracked tickers."""
        decisions: list[dict[str, Any]] = []
        for ticker in self.budget.get_all_tracked():
            snap = get_snapshot(ticker)
            if snap is None:
                continue
            thought = self._eval_one(ticker, snap)
            if thought is not None and thought.direction != 0:
                decisions.append(
                    {
                        "ticker": ticker,
                        "direction": thought.direction,
                        "verdict": thought.verdict,
                        "score": thought.score,
                        "thought": thought,
                    }
                )
        return decisions

    def _eval_one(self, ticker: str, snap: dict[str, Any]) -> Optional[Thought]:
        """Evaluate one ticker through the brain pipeline."""
        prices = snap.get("prices", [])
        if len(prices) < 20:
            return None
        threat = self._check_threat(ticker, snap, prices)
        if threat.trigger_exit:
            log.info("AMYGDALA %s EXIT: %s", ticker, threat.reason)
            return None
        pfc = self.pfc.evaluate(
            ticker=ticker,
            prices=prices,
            volumes=snap.get("volumes", []),
            atr=snap.get("atr", 1.0),
        )
        cortex_score = self._brain_score(snap)
        combined = self.plasticity.synthesize(
            {"amygdala": threat.score, "pfc": pfc.intent, "hippocampus": cortex_score}
        )
        return self._make_thought(ticker, combined, pfc.regime, threat.fear)

    def _check_threat(
        self, ticker: str, snap: dict[str, Any], prices: list[float]
    ) -> Any:
        """Run amygdala threat check."""
        return self.amygdala.evaluate(
            ticker=ticker,
            bid=snap.get("bid", 0),
            ask=snap.get("ask", 0),
            last=snap.get("last", 0),
            volume=snap.get("volume", 0),
            atr=snap.get("atr", 1.0),
            prices=prices,
        )

    def _brain_score(self, snap: dict[str, Any]) -> float:
        """Compute cortex signal score."""
        alpha = compute_alpha(
            close=snap.get("close_arr"),
            volume=snap.get("vol_arr"),
            buy_volume=snap.get("buy_vol_arr"),
            bid_sizes=snap.get("bid_sizes"),
            ask_sizes=snap.get("ask_sizes"),
        )
        thought = self.cortex.evaluate(alpha) if alpha else None
        return thought.score if thought else 0.0

    def _make_thought(
        self, ticker: str, score: float, regime: str, fear: float
    ) -> Optional[Thought]:
        """Convert combined score to Thought if above threshold."""
        if abs(score) < ENTRY_THRESHOLD:
            return None
        direction = 1 if score > 0 else -1
        log.info(
            "JULI %s %s score=%.3f regime=%s fear=%.2f",
            ticker,
            "BUY" if direction > 0 else "SELL",
            score,
            regime,
            fear,
        )
        return Thought(
            verdict="BUY" if direction > 0 else "SELL",
            score=score,
            direction=direction,
            confidence=0.5 + abs(score) * 0.45,
        )
