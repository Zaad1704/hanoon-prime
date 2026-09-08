"""monitor.portfolio_risk — portfolio-level risk (rebuild risk/portfolio.py port).

Continuous drawdown risk scalar, exposure + concentration caps, stress
mode, size adjustment, and portfolio profit protection (peak unrealized
P&L giveback -> exit weakest winners first). All inputs come from IB.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..immune import MAX_CONCURRENT_POSITIONS, MAX_POSITION_NOTIONAL

log = logging.getLogger(__name__)

RISK_MIN: float = 0.30
RISK_MAX: float = 1.0
CONCENTRATION_CAP: float = 0.25  # fraction of equity per ticker (rebuild)
STRESS_DRAWDOWN: float = 0.20
STRESS_SIZE_CAP: float = 0.30  # max budget fraction per entry in stress
GIVEBACK_PCT: float = 0.25  # exit weakest when unrealized P&L fades 25%
GIVEBACK_MIN_PEAK_USD: float = 20.0
GIVEBACK_BATCH: int = 3
GIVEBACK_COOLDOWN_SECS: float = 60.0
_BUDGET: float = MAX_POSITION_NOTIONAL * MAX_CONCURRENT_POSITIONS


@dataclass
class PortfolioRiskState:
    """Snapshot of portfolio risk (telemetry-facing)."""

    equity: float = 0.0
    equity_synced: bool = False
    risk_scalar: float = 1.0
    drawdown: float = 0.0
    stress_mode: bool = False
    exposure: float = 0.0
    position_count: int = 0
    max_positions: int = MAX_CONCURRENT_POSITIONS
    peak_unrealized: float = 0.0
    unrealized: float = 0.0


@dataclass
class _HoldPos:
    """One holding: IB unrealized P&L + market value."""

    pnl: float = 0.0
    value: float = 0.0
    pct: float = 0.0


@dataclass
class GivebackDecision:
    """Portfolio profit protection verdict for one pulse."""

    tickers: list[str] = field(default_factory=list)
    peak: float = 0.0
    unrealized: float = 0.0
    fade: float = 0.0
    fired: bool = False


class PortfolioRiskManager:
    """Portfolio risk: scalar, gates, sizing, profit protection."""

    def __init__(self) -> None:
        """Start unsynced; equity only ever becomes real IB NetLiq."""
        self._equity = 0.0
        self._equity_synced = False
        self._peak_equity = 0.0
        self._drawdown = 0.0
        self._risk_scalar = 1.0
        self._stress = False
        self._holdings: dict[str, _HoldPos] = {}
        self._peak_unrealized = 0.0
        self._last_giveback = 0.0

    def update_equity(self, equity: float) -> None:
        """Update equity from IB NetLiq; recompute scalar + stress."""
        if equity <= 0:
            return  # never fabricate equity from a failed read
        self._equity = float(equity)
        self._equity_synced = True
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
        if self._peak_equity > 0:
            self._drawdown = max(
                0.0, (self._peak_equity - self._equity) / self._peak_equity
            )
        self._risk_scalar = min(RISK_MAX, max(RISK_MIN, 1.0 - self._drawdown * 3.0))
        self._stress = self._drawdown > STRESS_DRAWDOWN

    def update_positions(self, portfolio: dict[str, dict[str, float]]) -> None:
        """Update holdings from read_portfolio(ib) (IB source of truth)."""
        self._holdings = {
            sym: _HoldPos(
                pnl=float(d.get("pnl", 0.0) or 0.0),
                value=float(d.get("value", 0.0) or 0.0),
                pct=float(d.get("pct", 0.0) or 0.0),
            )
            for sym, d in (portfolio or {}).items()
        }

    def total_exposure(self) -> float:
        """Open notional as a fraction of the position budget."""
        if _BUDGET <= 0:
            return 0.0
        return sum(abs(h.value) for h in self._holdings.values()) / _BUDGET

    def pre_trade_risk_gate(self, ticker: str, notional: float) -> tuple[bool, str]:
        """Should this entry proceed? Returns (allow, reason).

        Checks: equity synced, stress size cap, exposure cap, ticker
        concentration (of equity), position count, risk scalar floor.
        """
        if not self._equity_synced or self._equity <= 0:
            return False, "equity_unsynced"
        if self._stress and notional > STRESS_SIZE_CAP * _BUDGET:
            return False, f"stress_size>{STRESS_SIZE_CAP:.0%}"
        if self.total_exposure() >= 1.0:
            return False, "exposure_cap"
        held = self._holdings.get(ticker)
        held_v = abs(held.value) if held else 0.0
        conc = (held_v + notional) / self._equity
        if conc > CONCENTRATION_CAP:
            return False, f"concentration={conc:.2f}"
        if len(self._holdings) >= MAX_CONCURRENT_POSITIONS:
            return False, f"max_positions={len(self._holdings)}"
        if self._stress:
            return False, f"stress_mode dd={self._drawdown:.2f}"
        if self._risk_scalar <= RISK_MIN * 0.5:
            return False, f"risk_scalar={self._risk_scalar:.2f}"
        return True, ""

    def adjust_size(self, shares: int, price: float) -> int:
        """Apply risk scalar + concentration dampening to a size."""
        if shares <= 0 or price <= 0:
            return shares
        scaled = shares * price * self._risk_scalar
        damp = max(0.5, 1.0 - (self.total_exposure() / max(_BUDGET, 1.0)) * 0.5)
        return int(scaled * damp / price)

    def check_portfolio_giveback(self) -> GivebackDecision:
        """Exit weakest winners when unrealized P&L fades from peak.

        Weakest = lowest unrealized P&L fraction; losers are never
        exited (they are not profits to bank). Cooldown prevents
        cascading exits every cycle.
        """
        d = GivebackDecision()
        total = sum(h.pnl for h in self._holdings.values())
        d.unrealized = total
        d.peak = self._peak_unrealized = max(self._peak_unrealized, total)
        if total <= 0 or self._peak_unrealized < GIVEBACK_MIN_PEAK_USD:
            return d
        d.fade = (self._peak_unrealized - total) / self._peak_unrealized
        if d.fade < GIVEBACK_PCT:
            return d
        now = time.monotonic()
        if now - self._last_giveback < GIVEBACK_COOLDOWN_SECS:
            return d
        self._last_giveback = now
        winners = [
            (sym, h) for sym, h in self._holdings.items() if h.pnl > 0 and h.value > 0
        ]
        d.tickers = [sym for sym, _ in sorted(winners, key=lambda kv: kv[1].pct)][
            :GIVEBACK_BATCH
        ]
        d.fired = bool(d.tickers)
        if d.fired:
            log.warning(
                "PORTFOLIO GIVEBACK: fade=%.0f%% peak=$%.0f now=$%.0f -> %s",
                d.fade * 100,
                self._peak_unrealized,
                total,
                d.tickers,
            )
        return d

    def get_risk_state(self) -> dict[str, float | bool | int | str]:
        """Telemetry snapshot of portfolio risk."""
        s = PortfolioRiskState(
            equity=self._equity if self._equity_synced else 0.0,
            equity_synced=self._equity_synced,
            risk_scalar=self._risk_scalar,
            drawdown=self._drawdown,
            stress_mode=self._stress,
            exposure=self.total_exposure(),
            position_count=len(self._holdings),
            max_positions=MAX_CONCURRENT_POSITIONS,
            peak_unrealized=self._peak_unrealized,
            unrealized=sum(h.pnl for h in self._holdings.values()),
        )
        return dict(s.__dict__)


__all__ = ["PortfolioRiskManager", "PortfolioRiskState", "GivebackDecision"]
