"""brain.exits — exit intelligence for open positions.

JULI's exit decision layer: profit-lock tiers, consolidation exit,
alpha delta tracking, and 8 exit pillars for nuanced timing.
Works alongside the mechanical ATR trailing in ib_executor.py.
Trigger math lives in ``exit_checks.py``; this module owns per-position
state and the learned thresholds.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

from .config import (
    EXIT_ADAPT_GIVEBACK_MAX,
    EXIT_ADAPT_MIN_TRADES,
    EXIT_ADAPT_STALE_MAX,
    EXIT_ADAPT_STEP_SCALE,
    GIVEBACK_KEEP_RATIO,
    STALE_EXIT_MINUTES,
)
from .exit_checks import ExitSignal, check_consolidation
from .exit_checks import check_giveback as _giveback
from .exit_checks import check_profit_lock as _profit_lock
from .exit_checks import check_stale as _stale
from .horizons import params_for
from .threshold_limits import DEFAULTS, PILLAR_KEYS

if TYPE_CHECKING:
    from .realized_ev import RealizedStats


# ── EXIT PILLARS (enhanced exit signal processing) ──────────────────
# Based on rebuild's thinker.py: 8 pillars for nuanced exit timing.
# Each pillar returns exit likelihood in [0, 1] — combined for final decision.


def compute_setup_degradation(
    alpha: dict[str, float] | None = None,
) -> float:
    """Pillar 1: Setup degradation (0.0-1.0).

    Re-evaluate alpha strength: how many indicators are still confirming?
    More weak than strong signals → setup degrading.
    """
    if not alpha:
        return 0.0
    strong = sum(1 for k, v in alpha.items() if isinstance(v, (int, float)) and v > 0.6)
    weak = sum(1 for k, v in alpha.items() if isinstance(v, (int, float)) and v < 0.4)
    total = max(strong + weak, 1)
    # More weak than strong → exit signal
    degradation = max(0.0, (weak - strong) / total)
    return min(1.0, degradation)


def compute_momentum_fading(
    momentum: float = 0.0,
    momentum_accel: float = 0.0,
    pnl_pct: float = 0.0,
) -> float:
    """Pillar 2: Momentum fading (0.0-1.0).

    Sharp deceleration → strong exit signal.
    Amplified if in profit but momentum dying.
    """
    exit_score = 0.0
    if momentum_accel < -1.0:
        exit_score = 0.8  # sharp deceleration
    elif momentum_accel < -0.3:
        exit_score = 0.5  # moderate deceleration
    elif momentum_accel < 0.0:
        exit_score = 0.2  # slight deceleration
    # If in profit but momentum fading, amplify
    if exit_score > 0 and pnl_pct > 0.02:
        exit_score = min(exit_score + 0.2, 1.0)
    return exit_score


def compute_flow_reversal(
    direction: int = 1,
    institutional_flow: float = 0.0,
) -> float:
    """Pillar 3: Flow reversal (0.0-1.0).

    Institutional money leaving the position direction.
    Longs: negative flow bearish; Shorts: positive flow bearish.
    """
    exit_score = 0.0
    if direction > 0:
        if institutional_flow < -0.3:
            exit_score = 0.7
        elif institutional_flow < -0.1:
            exit_score = 0.3
    else:
        if institutional_flow > 0.3:
            exit_score = 0.7
        elif institutional_flow > 0.1:
            exit_score = 0.3
    return exit_score


def compute_giveback_risk(
    health_score: float = 0.5,
    pnl_pct: float = 0.0,
) -> float:
    """Pillar 4: Giveback risk (0.0-1.0).

    How vulnerable is profit to reversal?
    Higher health score + profit → higher giveback risk if setup degrading.
    """
    if pnl_pct > 0.05:
        # Up 5%+ → giveback risk
        return min(health_score * 1.5, 0.8)
    elif pnl_pct > 0.02:
        # Up 2-5% → moderate risk
        return min(health_score * 0.8, 0.5)
    return 0.0


def compute_time_pressure(
    hold_minutes: float = 0.0,
    flat_timeout_minutes: float = 20.0,
    pnl_pct: float = 0.0,
) -> float:
    """Pillar 5: Time pressure (0.0-1.0).

    Position held too long without progress.
    Adaptive based on actual hold time vs flat timeout.
    """
    if hold_minutes <= 0:
        return 0.0
    ratio = hold_minutes / max(flat_timeout_minutes, 1.0)
    if ratio >= 1.0:
        # Past timeout → strong exit
        return min(0.8 + (ratio - 1.0) * 0.4, 1.0)
    elif ratio >= 0.7:
        # Approaching timeout
        return 0.3 + (ratio - 0.7) * 1.5
    elif pnl_pct < 0.005 and pnl_pct > -0.005 and ratio >= 0.5:
        # Flat trade at 50% timeout
        return 0.2 * (ratio - 0.5) * 2.0
    return 0.0


def compute_stale_risk(
    hold_minutes: float = 0.0,
    stale_minutes_force: float = 120.0,
    stale_minutes_tight: float = 60.0,
    pnl_pct: float = 0.0,
) -> float:
    """Pillar 6: Stale risk (0.0-1.0).

    Losers held too long → stronger exit signal.
    Adaptive based on stale thresholds.
    """
    if hold_minutes <= 0 or pnl_pct >= 0:
        return 0.0
    stale_force_ratio = hold_minutes / max(stale_minutes_force, 1.0)
    stale_tight_ratio = hold_minutes / max(stale_minutes_tight, 1.0)
    if stale_force_ratio >= 1.0:
        return min(0.9 + (stale_force_ratio - 1.0) * 0.3, 1.0)
    elif stale_tight_ratio >= 1.0:
        return min(0.6 + stale_force_ratio * 0.3, 1.0)
    elif stale_tight_ratio >= 0.7:
        return 0.2 + stale_tight_ratio * 1.5
    # Amplify if losing
    if pnl_pct < -0.02:
        return min(0.5, 0.3 * 1.5)
    return 0.0


def compute_episodic_recall(
    ticker: str,
    episodic_memory: Any | None = None,
) -> float:
    """Pillar 7: Episodic recall (0.0-1.0).

    What happened last time with similar patterns?
    Loss rate * 0.5 → exit likelihood.
    """
    if episodic_memory is None:
        return 0.0
    try:
        hist = episodic_memory.query(ticker, k=5)
        if hist and len(hist) >= 2:
            wins = sum(1 for h in hist if isinstance(h, dict) and h.get("won"))
            loss_rate = 1.0 - (wins / len(hist))
            return min(loss_rate * 0.5, 1.0)
    except Exception as e:
        _log.warning("episodic recall error: %s", e)
    return 0.0


def compute_sentiment_exit(
    sentiment_pull: float = 0.0,
) -> float:
    """Pillar 8: News sentiment exit (0.0-1.0).

    Negative news → stronger exit signal.
    Positive news never forces exit.
    """
    if sentiment_pull < 0:
        return min(abs(sentiment_pull) * 2.0, 1.0)
    return 0.0


def _combine_pillars(pillars: list[float], weights: list[float] | None = None) -> float:
    """Combine pillar scores into exit likelihood.

    Default weights: equal importance with emphasis on stale/giveback.
    """
    if not pillars:
        return 0.0
    if weights is None:
        weights = [1.0] * len(pillars)
    if len(weights) != len(pillars):
        # Fall back to max approach
        return max(pillars)
    weighted_sum = sum(p * w for p, w in zip(pillars, weights))
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.0
    return min(1.0, weighted_sum / weight_sum)


class ExitPolicy:
    """JULI's exit intelligence — decides when to close positions.

    v2.1: thresholds are LEARNED. ``adapt_from_realized`` shifts the
    giveback keep-ratio and the stale window from realized exit outcomes
    (many giveback exits that would have hit target → widen keep-ratio;
    many stale exits → shorten the stale window). Bounded by
    EXIT_ADAPT_* so the policy adapts without reinventing itself.
    """

    def __init__(self) -> None:
        """Initialize empty per-position state and base thresholds."""
        self._entry_price: dict[str, float] = {}
        self._peak_price: dict[str, float] = {}
        self._peak_pnl: dict[str, float] = {}
        self._entry_ts: dict[str, float] = {}
        self._flat_pulses: dict[str, int] = {}
        self._prev_price: dict[str, float] = {}
        self._entry_alpha: dict[str, dict[str, float]] = {}
        self._keep_ratio: float = GIVEBACK_KEEP_RATIO
        self._stale_minutes: float = STALE_EXIT_MINUTES
        self._adapt_count: int = 0

    def adapt_from_realized(self, realized: "RealizedStats") -> None:
        """Retune giveback/stale thresholds from realized exit outcomes.

        Signal extraction (bounded, no veto semantics — exits only):
        • When realized R:R is weak relative to the 3:1 target while
          giveback exits fire, winners are being cut — WIDEN keep-ratio
          so trailing protects more of the peak.
        • When realized R:R is strong, tighten slightly toward base.
        • Stale exits shrink when R:R underperforms (capital recycling).
        """
        rr, rel, n = realized.realized_rr()
        if n < EXIT_ADAPT_MIN_TRADES or rel <= 0.0:
            return
        self._adapt_count += 1
        shift = (3.0 - min(rr, 3.0)) * EXIT_ADAPT_STEP_SCALE
        self._keep_ratio = max(
            GIVEBACK_KEEP_RATIO - EXIT_ADAPT_GIVEBACK_MAX,
            min(
                GIVEBACK_KEEP_RATIO + EXIT_ADAPT_GIVEBACK_MAX, self._keep_ratio + shift
            ),
        )
        stale_shift = (3.0 - min(rr, 3.0)) * 10.0
        self._stale_minutes = max(
            STALE_EXIT_MINUTES - EXIT_ADAPT_STALE_MAX,
            min(
                STALE_EXIT_MINUTES + EXIT_ADAPT_STALE_MAX,
                self._stale_minutes - stale_shift,
            ),
        )

    def exit_likelihood(
        self, ticker: str, current_price: float, direction: int = 1
    ) -> float:
        """Combined adaptive exit signal from the available pillars.

        Wires the 8 exit pillars into a single likelihood in [0, 1] using
        the structural exit_w_* weights — the TIER2 input of the exit
        ladder. Momentum deceleration is approximated from the last pulse;
        flow / episodic / sentiment have no live sources yet and stay 0.
        Returns 0.0 for unknown tickers (mechanical-only path).
        """
        if ticker not in self._entry_ts:
            return 0.0
        entry = self._entry_price.get(ticker) or current_price
        if entry <= 0:
            entry = current_price
        prev = self._prev_price.get(ticker) or current_price
        if prev <= 0:
            prev = current_price
        pnl_pct = (current_price / entry - 1.0) * direction
        hold_minutes = max(
            0.0, (time.time() - self._entry_ts.get(ticker, time.time())) / 60.0
        )
        health = min(1.0, max(0.0, 0.5 + pnl_pct * 2.0))
        momentum = (current_price - prev) / prev
        pillars = [
            compute_setup_degradation(self._entry_alpha.get(ticker)),
            compute_momentum_fading(
                momentum=momentum, momentum_accel=momentum, pnl_pct=pnl_pct
            ),
            compute_flow_reversal(direction, institutional_flow=0.0),
            compute_giveback_risk(health, pnl_pct),
            compute_time_pressure(hold_minutes, self._stale_minutes, pnl_pct),
            compute_stale_risk(
                hold_minutes, self._stale_minutes, self._stale_minutes * 0.5, pnl_pct
            ),
            compute_episodic_recall(ticker),
            compute_sentiment_exit(0.0),
        ]
        weights = [DEFAULTS[key] for key in PILLAR_KEYS]
        return _combine_pillars(pillars, weights)

    def register(
        self,
        ticker: str,
        entry_price: float,
        entry_alpha: dict[str, float] | None = None,
        horizon: str = "scalp",
    ) -> None:
        """Register a new position for exit monitoring (per-horizon windows)."""
        self._stale_minutes = params_for(horizon).stale_minutes
        self._entry_price[ticker] = entry_price
        self._peak_price[ticker] = entry_price
        self._peak_pnl[ticker] = 0.0
        self._entry_ts[ticker] = time.time()
        self._flat_pulses[ticker] = 0
        self._prev_price[ticker] = entry_price
        if entry_alpha:
            self._entry_alpha[ticker] = dict(entry_alpha)

    def evaluate(
        self,
        ticker: str,
        current_price: float,
        ib_unrealized_pnl: float = 0.0,
        direction: int = 1,
        exit_pillars: dict[str, float] | None = None,
    ) -> ExitSignal:
        """Evaluate exit conditions for one position.

        Enhanced with 8 exit pillars for nuanced timing.
        If exit_pillars dict is provided, combines pillar scores with
        mechanical exits (profit-lock, giveback, stale, consolidation).
        """
        if ticker not in self._entry_ts:
            return ExitSignal()
        self._update_peaks(ticker, current_price, ib_unrealized_pnl)

        # Standard mechanical exits
        for check in (
            self._check_profit_lock,
            self._check_giveback,
            self._check_stale,
            self._check_consolidation,
        ):
            sig = check(ticker, ib_unrealized_pnl, direction, current_price)
            if sig.should_exit:
                return sig

        # Enhanced pillar-based exit check (advisory, configurable threshold)
        if exit_pillars is not None:
            pillar_signal = self._check_pillars(
                ticker, current_price, direction, exit_pillars
            )
            if pillar_signal.should_exit:
                return pillar_signal

        return ExitSignal()

    def _check_pillars(
        self,
        ticker: str,
        current_price: float,
        direction: int,
        pillars: dict[str, float],
        exit_threshold: float = 0.6,
    ) -> ExitSignal:
        """Check pillar scores for exit decision.

        Args:
            ticker: position ticker
            current_price: current market price
            direction: 1 for long, -1 for short
            pillars: dict of pillar_name -> score (0.0-1.0)
            exit_threshold: combined score threshold for exit (default 0.6)

        Returns:
            ExitSignal if pillar score exceeds threshold, else empty signal.
        """
        if not pillars:
            return ExitSignal()
        # Combine pillar scores with weights emphasizing risk pillars
        pillar_values = list(pillars.values())
        weights = []
        for name in pillars.keys():
            if name in ("stale_risk", "giveback_risk", "momentum_fading"):
                weights.append(1.2)  # Emphasize risk pillars
            elif name in ("setup_degradation", "flow_reversal"):
                weights.append(1.0)
            else:
                weights.append(0.8)  # Time pressure, episodic, sentiment = lower weight
        combined = _combine_pillars(pillar_values, weights)
        if combined >= exit_threshold:
            return ExitSignal(
                should_exit=True,
                exit_type="pillars",
                reason=f"Combined pillar score {combined:.2f} >= {exit_threshold}",
                exit_score=combined,
            )
        return ExitSignal()

    def deregister(self, ticker: str) -> None:
        """Remove a position from all exit-monitoring state."""
        for d in (
            self._entry_price,
            self._peak_price,
            self._peak_pnl,
            self._entry_ts,
            self._flat_pulses,
            self._prev_price,
            self._entry_alpha,
        ):
            d.pop(ticker, None)

    def is_registered(self, ticker: str) -> bool:
        """Whether a ticker is being exit-monitored."""
        return ticker in self._entry_ts

    def telemetry(self) -> dict[str, float | int]:
        """Learned-exit telemetry (advisor view of the current policy)."""
        return {
            "keep_ratio": round(self._keep_ratio, 3),
            "stale_minutes": round(self._stale_minutes, 1),
            "adapt_count": self._adapt_count,
        }

    def _update_peaks(self, ticker: str, price: float, pnl: float) -> None:
        """Track the running price and P&L peaks for one position."""
        if price > self._peak_price.get(ticker, 0):
            self._peak_price[ticker] = price
        if pnl > self._peak_pnl.get(ticker, 0):
            self._peak_pnl[ticker] = pnl

    def _check_profit_lock(
        self,
        ticker: str,
        pnl: float,
        _direction: int,
        _price: float,
    ) -> ExitSignal:
        """If peak gain reached a tier, lock in minimum profit."""
        return _profit_lock(
            self._entry_price.get(ticker, 0),
            self._peak_pnl.get(ticker, 0),
            pnl,
        )

    def _check_giveback(
        self,
        ticker: str,
        pnl: float,
        _direction: int = 1,
        _price: float = 0.0,
    ) -> ExitSignal:
        """Exit when P&L drops more than the learned keep-ratio from its peak."""
        return _giveback(pnl, self._peak_pnl.get(ticker, 0.0), self._keep_ratio)

    def _check_stale(
        self,
        ticker: str,
        _pnl: float = 0.0,
        _direction: int = 1,
        _price: float = 0.0,
    ) -> ExitSignal:
        """Exit if held longer than the learned stale window."""
        return _stale(self._entry_ts.get(ticker, time.time()), self._stale_minutes)

    def reset_flat_pulses(self) -> None:
        """Clear per-ticker consolidation pulse counts (session boundary)."""

        self._flat_pulses.clear()

    def _check_consolidation(
        self,
        ticker: str,
        _pnl: float = 0.0,
        _direction: int = 1,
        price: float = 0.0,
    ) -> ExitSignal:
        """Exit if price moves less than 0.1% for N consecutive pulses."""
        prev = self._prev_price.get(ticker, price)
        sig, pulses = check_consolidation(price, prev, self._flat_pulses.get(ticker, 0))
        self._flat_pulses[ticker] = pulses
        self._prev_price[ticker] = price
        return sig
