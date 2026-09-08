"""hanoon_prime.brain.consolidation — System 2: Background consolidation.

Runs asynchronously in a background thread every 10-30s.
Handles network I/O (HALIM, regime), heavy compute (thinker), disk
persistence, and sleep replay. IB feeds trade data; ConsolidationEngine
provides the slow path for state updates and offline memory consolidation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..reflection.buffer import Fill, Trade, TradeBuffer
from ..reflection.supervisor import LearningSupervisor
from ..types import BarSeries, FillInfo
from .halim_adapter import HalimAdapter
from .memory import JuliMemory
from .neurons.sleep import SleepReplayEngine, SleepResult
from .news_sources import NewsFeedEngine
from .policy.portfolio_risk import PortfolioRiskManager
from .policy.safety import SafetyProducer
from .regime import RegimeDetector
from .shared_state import BrainState
from .thinker import Signal, Thinker

log = logging.getLogger(__name__)
HALIM_URL: str = "http://127.0.0.1:8765"
CYCLE_INTERVAL: float = 30.0


class ConsolidationEngine:
    """System 2: Background cognitive engine — runs every 10-30s."""

    def __init__(
        self,
        brain_state: BrainState,
        halim_url: str = HALIM_URL,
        interval: float = CYCLE_INTERVAL,
        sleep_engine: Optional[SleepReplayEngine] = None,
    ) -> None:
        self.state = brain_state
        self.interval = interval
        self.memory = JuliMemory()
        self.halim = HalimAdapter(base_url=halim_url)
        self.thinker = Thinker()

        self.buffer = TradeBuffer(on_trade_closed=self._on_trade_closed)
        self.supervisor = LearningSupervisor(self.buffer, self.memory)
        # News organ (System 2): live headlines → bounded sentiment into
        # BrainState. Evidence for the brain, never a gate.
        self.news = NewsFeedEngine(brain_state, interval=interval * 4.0)
        # Local regime fallback: classifies from shared prices when HALIM
        # is unreachable, so the label never stays "unknown" for long.
        self._detector = RegimeDetector()
        self._sleep_engine = sleep_engine
        self.portfolio_risk = PortfolioRiskManager()
        self.safety = SafetyProducer()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start System 2 background loop."""
        self._running = True
        self._sync_initial_state()
        self.supervisor.start()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="consolidation"
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
        self.state.update(threshold=self.memory.threshold, indicator_weights=weights)

    def _loop(self) -> None:
        """Main background loop — runs every interval seconds."""
        while self._running:
            try:
                self._cycle()
            except Exception as e:
                log.error("System 2 cycle error: %s", e)
            time.sleep(self.interval)

    def _update_policy(self) -> None:
        """Slow-cortex policy pulse: publish safety + portfolio risk state."""
        feed = self.state.get("account_feed") or {}
        self.safety.begin_call()
        self.safety.on_daily_pnl(float(feed.get("daily_pnl", 0.0)))
        self.safety.on_consecutive_losses(int(self.state.get("consecutive_losses", 0)))
        self.safety.on_position_count(len(self.state.get("positions_open") or {}))
        risk = self._portfolio_policy_state(feed)
        self._publish_giveback_exits()
        self.state.update(policy_state=risk)

    def _portfolio_policy_state(self, feed: dict[str, Any]) -> dict[str, Any]:
        """Fold portfolio risk + safety authorization into policy_state."""
        positions = feed.get("positions") or {}
        if positions:
            self.portfolio_risk.update_positions(positions)
        equity = feed.get("equity")
        if equity is not None:
            self.portfolio_risk.update_equity(float(equity))
        risk: dict[str, Any] = self.portfolio_risk.get_risk_state()
        auth, reason = self.safety.authorized()
        risk.update(
            holdings={
                sym: abs(float(p.get("value", 0.0) or 0.0))
                for sym, p in positions.items()
            },
            authorized=auth,
            enabled=self.safety.enabled,
            halted=self.safety.halted,
            pause_reason=reason if not auth else "",
            consecutive_losses=int(self.state.get("consecutive_losses", 0)),
            daily_pnl=float(feed.get("daily_pnl", 0.0)),
        )
        return risk

    def _publish_giveback_exits(self) -> None:
        """Publish giveback exits from the portfolio risk pulse."""
        gb = self.portfolio_risk.check_portfolio_giveback()
        exits: list[dict[str, Any]] = []
        if gb.fired:
            for t in gb.tickers:
                exits.append(
                    {
                        "type": "portfolio_giveback",
                        "ticker": t,
                        "reason": "giveback_fade",
                    }
                )
        self.state.update(policy_exits=exits)

    def _cycle(self) -> None:
        """One full System 2 cognitive cycle."""
        self._update_regime()
        self._update_halim()
        self._run_thinker()
        self._apply_halim_recommendations()
        self._update_policy()
        self.news.maybe_refresh()
        self._persist_state()
        log.info(
            "S2 | regime=%.2f halim=%.3f thinker=%.4f news=%.2f",
            self.state.get("regime_multiplier", 1.0),
            self.state.get("halim_modifier", 0.0),
            self.state.get("thinker_modifier", 0.0),
            self._news_bias(),
        )

    def _news_bias(self) -> float:
        """Bounded news sentiment bias for the latest alpha ticker (±0.03)."""
        try:
            alpha = self._get_latest_alpha()
            if not alpha:
                return 0.0
            ticker = max(alpha, key=lambda k: abs(alpha.get(k, 0)))
            sent = self.state.get("news_sentiment", {}) or {}  # array-safe: dict-typed
            pol = float(sent.get(ticker, 0.0))
            return max(-0.03, min(0.03, pol * 0.03))
        except Exception:
            return 0.0

    def _update_regime(self) -> None:
        """Get regime classification from HALIM (local fallback if stale)."""
        alpha = self._get_latest_alpha()
        if not alpha:
            return
        try:
            regime = self.halim.get_regime(alpha, self.state.get_latest_prices())
        except Exception as e:
            log.warning("Regime query failed: %s", e)
            regime = None
        if isinstance(regime, dict):
            self._apply_regime(regime)
        elif self.state.get("regime_label", "unknown") == "unknown":
            # HALIM down or unparsable: publish the local numpy detector's
            # classification so the strategy organs always see a real regime.
            self._local_regime()

    def _local_regime(self) -> None:
        """Publish a local RegimeDetector classification (fallback only)."""
        prices = self.state.get_latest_prices() or []
        if len(prices) < 20:
            return
        rs = self._detector.detect(prices)
        if rs.regime != "unknown":
            self.state.update(
                regime_label=rs.regime,
                regime_multiplier=rs.multiplier,
                regime_source="local_fallback",
            )
            log.info("S2 REGIME FALLBACK: %s (mult=%.2f)", rs.regime, rs.multiplier)

    def _apply_regime(self, regime: dict[str, Any]) -> None:
        """Apply regime data to shared state."""
        self.state.update(
            regime_multiplier=regime.get("multiplier", 1.0),
            regime_label=regime.get("regime", "normal"),
            regime_confidence=regime.get("confidence", 0.5),
            regime_risk=regime.get("risk_adjustment", "normal"),
            regime_drivers=regime.get("key_drivers", []),
            regime_description=regime.get("description", ""),
        )

    def _update_halim(self) -> None:
        """Poll HALIM external AI advisor (network I/O)."""
        alpha = self._get_latest_alpha()
        if not alpha:
            return
        ticker = max(alpha, key=lambda k: abs(alpha.get(k, 0)))
        mod = self.halim.get_modifier(ticker, alpha, 0.0, "SCAN")
        self.state.update(halim_modifier=mod)

    def _apply_halim_recommendations(self) -> None:
        """Fetch HALIM recommendations and store in shared state for orchestrator."""
        from .halim_recommendations import (
            fetch_recommendations,
            validate_recommendation,
        )

        try:
            recs = fetch_recommendations(self.halim._base_url)
            if not recs:
                return
            valid = [r for r in recs if validate_recommendation(r) is None]
            if valid:
                self.state.update(halim_recommendations=valid)
                log.info("HALIM: %d valid recommendations fetched", len(valid))
        except Exception as e:
            log.debug("HALIM recommendations fetch failed: %s", e)

    def _run_thinker(self) -> None:
        """Run thinker deliberation and write to shared state."""
        alpha = self._get_latest_alpha()
        if not alpha:
            return
        prices = self.state.get_latest_prices()
        regime = self.state.get("regime_label", "unknown")
        threshold = self.state.get("threshold", 0.58)
        bars = BarSeries(prices, prices, prices, prices)
        r = self.thinker.think(
            alpha, 0.0, 1, Signal(regime=regime, bars=bars, threshold=threshold)
        )
        self.state.update(
            thinker_modifier=r.modifier,
            thinker_confidence_mod=r.confidence_mod,
            thinker_risk_scalar=r.risk_scalar,
        )

    def _persist_state(self) -> None:
        """Atomic write to state.json (disk I/O)."""
        try:
            state_dir = Path(__file__).resolve().parents[3] / "runtime"
            state_dir.mkdir(parents=True, exist_ok=True)
            tmp = state_dir / "state.json.tmp"
            tmp.write_text(json.dumps(self._persist_data(), default=str))
            tmp.replace(state_dir / "state.json")
        except Exception as e:
            log.warning("State persist failed: %s", e)

    def _persist_data(self) -> dict[str, Any]:
        """Build state data for persistence."""
        ep = getattr(getattr(self.thinker, "episodic", None), "size", 0)
        return {
            "weights": self.memory.get_weights(),
            "threshold": self.memory.threshold,
            "pred_error": self.memory.pred_error,
            "episodic_size": ep,
            "brain_state": self.state.snapshot(),
            "timestamp": time.time(),
        }

    def _get_latest_alpha(self) -> dict[str, float] | None:
        """Read latest alpha from System 1 via shared state."""
        return self.state.get_latest_alpha()

    def on_trade_close(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int = 1,
        fill: FillInfo | None = None,
    ) -> None:
        """Route trade close to thinker + buffer + HALIM postmortem."""
        from ..reflection.buffer import BUY, SELL

        alpha = self._get_latest_alpha() or {}
        self.thinker.episodic.add(alpha, won, pnl_pct)
        self.thinker.emotion.update(won, pnl_pct)
        self.state.set_refractory(2.0)
        side = BUY if direction > 0 else SELL
        qty = fill.qty if fill is not None else 0.0
        price = fill.avg_price if fill is not None else 0.0
        fees = fill.fees if fill is not None else 0.0
        self.buffer.on_fill(
            Fill(
                ticker=ticker,
                side=side,
                qty=qty,
                price=price,
                time=time.time(),
                commission=fees,
            )
        )
        # HALIM postmortem: ask the LLM to analyze this trade
        self._halim_postmortem(ticker, won, pnl_pct, direction, alpha)

    def _halim_postmortem(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float,
        direction: int,
        alpha: dict[str, Any],
    ) -> None:
        """Ask HALIM to analyze a closed trade (async, non-blocking)."""
        try:
            trade_data = {
                "ticker": ticker,
                "won": won,
                "pnl_pct": pnl_pct,
                "direction": direction,
                "alpha": alpha,
                "regime": self.state.get("regime_label", "unknown"),
            }
            result = self.halim.analyze_trade(trade_data)
            if result.get("insight"):
                log.info("HALIM POSTMORTEM %s: %s", ticker, result["insight"])
                self.state.update(halim_last_insight=result)
        except Exception as e:
            log.debug("HALIM postmortem failed: %s", e)

    def _on_trade_closed(self, trade: Trade) -> None:
        """Callback from TradeBuffer when a round-trip closes."""
        self.supervisor.on_trade_close(trade)
        log.info("Buffer trade closed: %s pnl=%.2f", trade.ticker, trade.pnl)

    def run_sleep_replay(self) -> Optional[SleepResult]:
        """Run sleep consolidation cycle for offline learning."""
        if self._sleep_engine is None:
            return None
        return self._sleep_engine.run_cycle()

    @property
    def sleep_engine(self) -> Optional[SleepReplayEngine]:
        """Access sleep replay engine."""
        return self._sleep_engine


__all__ = ["ConsolidationEngine"]
