"""hanoon_prime.micro_live — Phase 5 guarded micro-live deployment.

Purpose
-------
Act as the hard gate between Phase-4 paper success and real-money order
routing. Strictly: NEVER enter live until the pre-locked protocol says so.

Rules (mirror task_plan.md Phase 5):
  * 5.1 smallest footprint — this module does not CHOOSE size; the live
    robot's existing caps (MAX_POSITION_NOTIONAL etc.) bound notional. It
    only refuses to *authorize* entries.
  * 5.2 shadow 1:1 — every proposed live entry is compared against the
    static-weights baseline verdict computed on the SAME snapshot. A live
    verdict that differs from baseline is a DIVERGENCE (investigate, and
    the guard refuses that entry).
  * 5.3 completion gate — micro-live may only arm when the Phase-4 paper
    report verdict is PASS; otherwise entries are refused outright.

This module performs NO I/O to a broker and NO monkey-patching of risk
constants. Eligibility is read from committed artifacts + immune constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cortex import Cortex, Thought
from .immune import DAILY_LOSS_LIMIT, KILL_DAILY_LOSS_LIMIT, MAX_POSITION_NOTIONAL
from .juli_feed import compute_alpha_from_snap

_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
DEFAULT_PHASE4_REPORT = _REPORTS_DIR / "phase4_paper.json"

HOLD_ACTION = "HOLD"
BUY_ACTION = "BUY"
SELL_ACTION = "SELL"
_ACTION_BY_DIRECTION = {1: BUY_ACTION, -1: SELL_ACTION}


@dataclass
class BaselineSignal:
    """Static-weights baseline signal for one snapshot (the 'paper' signal)."""

    ticker: str
    action: str  # BUY / SELL / HOLD (mirrors cortex verdict naming)
    score: float
    reason: str = ""


@dataclass
class ShadowComparison:
    """One live-vs-baseline comparison for a proposed entry."""

    ticker: str
    live_action: str
    baseline_action: str
    live_score: float
    baseline_score: float
    agree: bool

    @property
    def is_divergence(self) -> bool:
        """A live BUY/SELL that the static baseline did not ALSO signal."""
        if self.live_action == HOLD_ACTION:
            return False
        return not self.agree


@dataclass
class MicroLiveGuardState:
    """Snapshot of guard state after one decision."""

    ticker: str
    eligible: bool
    phase4_pass: bool
    kill_latched: bool
    daily_loss_hit: bool
    divergence: bool
    reason: str
    shadow: ShadowComparison | None = None

    @property
    def authorized(self) -> bool:
        """Entry authorized to be routed (never under FAIL/divergence)."""
        return (
            self.eligible
            and not self.kill_latched
            and not self.daily_loss_hit
            and not self.divergence
        )


class MicroLiveGuard:
    """Refuse-entry-until-proven guard for micro-live deployment.

    Tracks a latched kill switch mirror of the live safety producer so the
    shadow path can render the SAME halt decisions the robot would take,
    without touching a broker. The latch is volume-limited: once set it
    clears on an explicit re-arm call (operationally: market closed + human
    confirmation), matching ``SafetyProducer.latched`` semantics.
    """

    def __init__(
        self,
        phase4_report: Path | None = None,
        daily_loss_limit: float = DAILY_LOSS_LIMIT,
        kill_daily_loss_limit: float = KILL_DAILY_LOSS_LIMIT,
        max_notional: float = MAX_POSITION_NOTIONAL,
    ) -> None:
        self._daily_loss_limit = daily_loss_limit
        self._kill_limit = kill_daily_loss_limit
        self._max_notional = max_notional
        self._kill_latched = False
        self._kill_reason = ""
        self._daily_loss_hit = False
        self.phase4_pass = self._load_phase4_pass(
            phase4_report or DEFAULT_PHASE4_REPORT
        )
        self.divergences: list[dict[str, Any]] = []

    # ── Phase-4 gate ─────────────────────────────────────────────────────
    @staticmethod
    def _load_phase4_pass(report: Path) -> bool:
        """Read the committed Phase-4 report; FAIL unless verdict is PASS."""
        try:
            blob = json.loads(report.read_text())
        except (OSError, ValueError):
            return False
        return bool(blob.get("verdict") == "PASS")

    # ── Baseline (paper) signal on the same snapshot ─────────────────────
    def baseline_signal(self, ticker: str, snap: dict[str, Any]) -> BaselineSignal:
        """Static-weights baseline verdict for a live snapshot.

        Uses the exact same alpha→cortex path the paper/backtest runs
        (compute_alpha_from_snap → fresh static-weights Cortex), so this is
        the 1:1 "paper would have said" counterpart to a live verdict.
        """
        alpha = compute_alpha_from_snap(snap)
        cortex = Cortex()  # static INDICATOR_WEIGHTS, fresh (no learning state)
        thought: Thought = cortex.evaluate(alpha)
        action = _ACTION_BY_DIRECTION.get(thought.direction, HOLD_ACTION)
        return BaselineSignal(
            ticker=ticker,
            action=action,
            score=thought.score,
            reason=",".join(thought.reasons),
        )

    # ── Safety rails (latched mirror of the live kill switch) ────────────
    def update_daily_pnl(self, pnl_usd: float, ticker: str = "") -> None:
        """Feed the guard the day's USD P&L (from the live account feed).

        Latches permanently on <= -KILL limit; marks halt-eligible on
        <= -DAILY_LOSS_LIMIT (same thresholds as immune/brain safety).
        """
        if pnl_usd <= -self._kill_limit:
            self._kill_latched = True
            self._kill_reason = f"{ticker} pnl {pnl_usd:.2f} <= -{self._kill_limit:.0f}"
        if pnl_usd <= -self._daily_loss_limit:
            self._daily_loss_hit = True

    def rearm(self) -> None:
        """Explicit re-arm (operationally: session close + human confirm)."""
        self._kill_latched = False
        self._kill_reason = ""
        self._daily_loss_hit = False

    # ── Decision ─────────────────────────────────────────────────────────
    def guard(
        self,
        ticker: str,
        snap: dict[str, Any],
        live_action: str,
        live_score: float,
    ) -> MicroLiveGuardState:
        """Evaluate one live entry proposal against the full guard stack."""
        if not compute_alpha_from_snap(snap):
            return self._state(ticker, True, "insufficient_snapshot", None)

        base = self.baseline_signal(ticker, snap)
        shadow = ShadowComparison(
            ticker=ticker,
            live_action=live_action,
            baseline_action=base.action,
            live_score=live_score,
            baseline_score=base.score,
            agree=(live_action == base.action),
        )
        if shadow.is_divergence:
            self.divergences.append(
                {
                    "ticker": ticker,
                    "live": live_action,
                    "baseline": base.action,
                    "live_score": live_score,
                    "baseline_score": base.score,
                }
            )
        return self._state(ticker, shadow.is_divergence, "", shadow)

    def _state(
        self,
        ticker: str,
        divergence: bool,
        reason_override: str,
        shadow: ShadowComparison | None,
    ) -> MicroLiveGuardState:
        """Build the guard state, collecting every blocking reason."""
        if reason_override:
            return MicroLiveGuardState(
                ticker=ticker,
                eligible=False,
                phase4_pass=self.phase4_pass,
                kill_latched=self._kill_latched,
                daily_loss_hit=self._daily_loss_hit,
                divergence=divergence,
                reason=reason_override,
                shadow=shadow,
            )
        reasons = self._blocking_reasons(divergence)
        return MicroLiveGuardState(
            ticker=ticker,
            eligible=self.phase4_pass,
            phase4_pass=self.phase4_pass,
            kill_latched=self._kill_latched,
            daily_loss_hit=self._daily_loss_hit,
            divergence=divergence,
            reason=",".join(reasons) if reasons else "authorized",
            shadow=shadow,
        )

    def _blocking_reasons(self, divergence: bool) -> list[str]:
        """Collect every guard check that blocks the proposed entry."""
        reasons: list[str] = []
        if not self.phase4_pass:
            reasons.append("phase4_paper_not_pass (Phase-5 gate 5.3)")
        if self._kill_latched:
            reasons.append("kill_switch_latched")
            if self._kill_reason:
                reasons.append(self._kill_reason)
        if self._daily_loss_hit:
            reasons.append("daily_loss_hit")
        if divergence:
            reasons.append("shadow_divergence")
        return reasons

    # ── Internals ────────────────────────────────────────────────────────
    @property
    def eligible_to_arm(self) -> bool:
        """The 5.3 completion gate: Phase-4 paper PASS is the floor."""
        return self.phase4_pass


def load_phase4_pass(report: Path | None = None) -> bool:
    """Module-level Phase-4 gate accessor (for scripts/CLI)."""
    return MicroLiveGuard._load_phase4_pass(report or DEFAULT_PHASE4_REPORT)
