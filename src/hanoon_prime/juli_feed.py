"""hanoon_prime.juli_feed — cross-asset refs + local regime fallback.

Extracted from juli.py (R3b): subscribes the cross-asset reference
tickers (SPY/QQQ/IWM/VXX), publishes their prices into shared state for
the brain's lead-lag engine, and publishes a LOCAL regime label when
HALIM's label is stuck at "unknown" (service down) so the strategy
organs always see a real regime. Never overrides HALIM's label.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .brain.indicators import compute_all_alpha
from .brain.regime import RegimeDetector
from .brain.shared_state import BrainState
from .cerebellum import compute_alpha
from .types import BarSeries

log = logging.getLogger(__name__)

# Cross-asset reference tickers: their snapshots feed the lead-lag engine.
REF_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "VXX")
_REGIME_FALLBACK_SECS: float = 30.0
# (brain key, snapshot key) — snapshot arrays feed alpha computation.
_KEYS = "close high low volume buy_volume bid_sizes ask_sizes"
# Snapshot keys are set by ib_cycle._snapshot as f"{k}_arr" for the
# streamer arrays() keys (close/high/low/volume/buy_volume/bid_sizes/
# ask_sizes). Mismatches here silently starve indicators of data.
_SRC = (
    "close_arr high_arr low_arr volume_arr buy_volume_arr"
    " bid_sizes_arr ask_sizes_arr"
)
ATTRS = tuple(zip(_KEYS.split(), _SRC.split()))


def compute_alpha_from_snap(snap: dict[str, Any]) -> dict[str, float]:
    """Compute all indicators from a snapshot (core + higher-order)."""
    kw: dict[str, Any] = {}
    for k, u in ATTRS:
        v = snap.get(u)
        kw[k] = [] if v is None else (v if isinstance(v, list) else list(v))
    if len(kw["close"]) < 20:
        return {}
    alpha = compute_all_alpha(BarSeries(**kw))
    return alpha if alpha else (compute_alpha(**kw) or {})


def entry_bars(
    snap: dict[str, Any], prices: list[float], regime_label: str
) -> dict[str, Any]:
    """Bar context INTO the brain (horizon classification happens there).

    NOTE: snapshot arrays are numpy arrays — ``arr or fallback`` raises
    ValueError (ambiguous truth). Use explicit length checks instead.
    """
    close = snap.get("close_arr")
    if close is None or len(close) == 0:
        close = prices
    return {
        "close": close,
        "high": snap.get("high_arr"),
        "low": snap.get("low_arr"),
        "regime": regime_label,
    }


def check_tick_latency(t0: int, ticker: str) -> None:
    """Warn only on real stalls (>=25ms); sub-ms JIT noise is expected."""
    latency_us = (time.perf_counter_ns() - t0) / 1000.0
    if latency_us > 25_000.0:
        log.warning("Tick latency spike: %.0f us for %s", latency_us, ticker)


class JuliFeed:
    """Cross-asset reference feed + local regime fallback publisher."""

    def __init__(self, state: BrainState) -> None:
        self._state = state
        self._detector = RegimeDetector()
        self._last_fallback = 0.0
        self._subscribed: set[str] = set()

    def ensure_refs(self, streamer: Any) -> None:
        """Subscribe the reference tickers once (idempotent)."""
        if not streamer or self._subscribed:
            return
        for ref in REF_TICKERS:
            try:
                streamer.subscribe(ref)
                self._subscribed.add(ref)
            except Exception as exc:
                log.debug("Ref subscribe failed for %s: %s", ref, exc)

    def publish_ref_prices(self, get_snapshot: Any) -> None:
        """Feed cross-asset reference snapshots into shared state."""
        refs: dict[str, float] = {}
        for ref in self._subscribed:
            s = get_snapshot(ref)
            if s and s.get("last", 0) > 0:
                refs[ref] = float(s["last"])
        if refs:
            self._state.update(ref_prices=refs)

    def fallback_regime(self) -> None:
        """Publish a local regime label when HALIM's label is stale."""
        now = time.time()
        if now - self._last_fallback < _REGIME_FALLBACK_SECS:
            return
        if self._state.get("regime_label", "unknown") != "unknown":
            return
        prices = self._state.get_latest_prices() or []
        if len(prices) < 20:
            return  # no stamp: a dataless attempt must not starve the next one
        self._last_fallback = now
        rs = self._detector.detect(prices)
        if rs.regime != "unknown":
            self._state.update(
                regime_label=rs.regime,
                regime_multiplier=rs.multiplier,
                regime_source="local_fallback",
            )
            log.info("REGIME FALLBACK: %s (mult=%.2f)", rs.regime, rs.multiplier)
